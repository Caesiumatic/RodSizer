from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query, BackgroundTasks
from starlette.background import BackgroundTask
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from functools import lru_cache
import importlib
import shutil
import os
import uuid
import re
from pathlib import Path
from typing import List
from pydantic import BaseModel
import json

class ExportRequest(BaseModel):
    image_id: str
    selected_ids: List[int]
    
app = FastAPI(title="RodSizer")


# Configure CORS — the app is a localhost tool, so restrict to loopback origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:8501",
        "http://localhost:8501",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = (BASE_DIR / "uploads").resolve()
RESULTS_DIR = (BASE_DIR / "results").resolve()
FRONTEND_DIR = (BASE_DIR / "frontend").resolve()
RESULTS_SCHEMA_VERSION = 6

# Upload limits
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500 MB per file
MAX_UPLOAD_FILES = 200                # per request

# Image ids are the stem of uploaded files: "<uuid>" or "<uuid>_<sanitized_name>".
_IMAGE_ID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?:_.+)?$"
)

UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
FRONTEND_DIR.mkdir(exist_ok=True)


@lru_cache(maxsize=1)
def _processing_module():
    # Delay loading heavy image-analysis dependencies until an endpoint needs them.
    return importlib.import_module("processing")


def _processing_function(name: str):
    return getattr(_processing_module(), name)


def _cached_results_are_current(payload: dict, expected_binary_mask_tune: int = 0) -> bool:
    if payload.get("results_schema_version") != RESULTS_SCHEMA_VERSION:
        return False
    if int(payload.get("binary_mask_tune", 0)) != int(expected_binary_mask_tune):
        return False

    calibration_info = payload.get("calibration_info") or {}
    method = calibration_info.get("method")

    if method == "default":
        return False
    if method == "uncalibrated" and "is_placeholder" not in calibration_info:
        return False

    return True


def _sanitize_folder_name(folder_name: str) -> str:
    if folder_name is None:
        return ""

    # Keep common punctuation users expect in folder names while blocking
    # characters that are invalid or problematic across macOS and Windows.
    invalid_chars = '<>:"/\\|?*'
    cleaned = "".join(
        c for c in folder_name
        if ord(c) >= 32 and c not in invalid_chars
    )

    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    cleaned = cleaned.rstrip(". ")

    # Reject traversal segments and reserved dot-names.
    if cleaned in ("", ".", ".."):
        return ""
    return cleaned


def _safe_join(base: Path, *parts: str) -> Path:
    """Join under `base` and ensure the result stays inside `base` after resolving.
    Raises HTTPException(400) on traversal attempts or empty components."""
    base_resolved = base.resolve()
    candidate = base_resolved
    for raw in parts:
        if raw is None or raw == "":
            raise HTTPException(status_code=400, detail="Invalid path component")
        # Disallow separators and traversal tokens inside a single component.
        if "/" in raw or "\\" in raw or raw in (".", ".."):
            raise HTTPException(status_code=400, detail="Invalid path component")
        candidate = candidate / raw

    resolved = candidate.resolve()
    try:
        resolved.relative_to(base_resolved)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path escapes allowed directory")
    return resolved


def _validate_image_id(image_id: str) -> str:
    if not image_id or not _IMAGE_ID_RE.match(image_id):
        raise HTTPException(status_code=400, detail="Invalid image id")
    for c in image_id:
        if ord(c) < 32 or c in '/\\*?[]':
            raise HTTPException(status_code=400, detail="Invalid image id")
    return image_id


def _resolve_folder(folder_name: str) -> Path:
    """Validate and resolve a user-supplied folder name to a path inside UPLOAD_DIR."""
    safe = _sanitize_folder_name(folder_name)
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid folder name")
    return _safe_join(UPLOAD_DIR, safe)


def _find_input_and_calibration_source(image_id: str):
    _validate_image_id(image_id)
    files = [p for p in UPLOAD_DIR.rglob("*") if p.is_file() and p.stem == image_id]
    if not files:
        raise HTTPException(status_code=404, detail="Image not found")

    input_path = files[0]
    search_dir = input_path.parent
    calibration_source_path = None

    current_filename = input_path.name
    if len(current_filename) > 37 and current_filename[36] == '_':
        original_name_with_ext = current_filename[37:]
        original_stem = Path(original_name_with_ext).stem
    else:
        original_stem = input_path.stem

    if original_stem:
        for f in search_dir.glob("*"):
            if f.suffix.lower() in ['.dm3', '.dm4', '.emd']:
                if len(f.name) > 37 and f.name[36] == '_':
                    cal_stem = Path(f.name[37:]).stem
                else:
                    cal_stem = f.stem

                if cal_stem == original_stem:
                    calibration_source_path = f
                    break

    return input_path, calibration_source_path

# Serve Frontend
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

# The frontend HTML is edited in place and updated often. Tell the browser to
# always revalidate so a code update is never hidden behind a stale cached page.
_NO_CACHE_HEADERS = {"Cache-Control": "no-cache, no-store, must-revalidate"}


def _frontend_page(name: str) -> FileResponse:
    return FileResponse(FRONTEND_DIR / name, headers=_NO_CACHE_HEADERS)


# --- Results housekeeping ---

def _safe_unlink(path: Path):
    try:
        path.unlink()
    except OSError:
        pass


def _valid_upload_ids() -> set:
    """Stems of every uploaded file (results are named '<stem>_<suffix>')."""
    ids = set()
    for p in UPLOAD_DIR.rglob("*"):
        if p.is_file() and ".analysis_cache" not in p.parts:
            ids.add(p.stem)
    return ids


def _delete_results_for_ids(image_ids) -> int:
    removed = 0
    for res_file in RESULTS_DIR.iterdir():
        if not res_file.is_file():
            continue
        if any(res_file.name.startswith(image_id + "_") for image_id in image_ids):
            _safe_unlink(res_file)
            removed += 1
    return removed


def cleanup_results_dir() -> dict:
    """Remove result files whose source upload no longer exists, plus stale
    export temp files (>1 day old). Never touches uploads/."""
    import time
    valid_ids = _valid_upload_ids()
    removed_orphans = 0
    removed_exports = 0
    now = time.time()
    for res_file in RESULTS_DIR.iterdir():
        if not res_file.is_file() or res_file.name.startswith("."):
            continue
        if res_file.name.startswith("export_"):
            # Export temps now self-delete after download; sweep old leftovers.
            if now - res_file.stat().st_mtime > 86400:
                _safe_unlink(res_file)
                removed_exports += 1
            continue
        if not any(res_file.name.startswith(image_id + "_") for image_id in valid_ids):
            _safe_unlink(res_file)
            removed_orphans += 1
    return {"orphans": removed_orphans, "stale_exports": removed_exports}


@app.on_event("startup")
async def _startup_cleanup():
    try:
        summary = cleanup_results_dir()
        if summary["orphans"] or summary["stale_exports"]:
            print(f"Results cleanup: removed {summary['orphans']} orphaned files, "
                  f"{summary['stale_exports']} stale exports")
    except Exception as e:
        print(f"Results cleanup skipped: {e}")

@app.get("/")
async def read_index():
    return _frontend_page("index.html")

@app.get("/analysis")
async def read_analysis():
    return _frontend_page("analysis.html")

@app.get("/folder_analysis")
async def read_folder_analysis():
    return _frontend_page("folder_analysis.html")

@app.get("/compare")
async def read_compare():
    return _frontend_page("compare.html")

# --- Folder Management ---

@app.post("/folders")
async def create_folder(folder_name: str = Form(...)):
    try:
        safe_name = _sanitize_folder_name(folder_name)
        if not safe_name:
             raise HTTPException(status_code=400, detail="Invalid folder name")

        folder_path = _safe_join(UPLOAD_DIR, safe_name)
        if folder_path.exists():
             raise HTTPException(status_code=400, detail="Folder already exists")

        folder_path.mkdir(parents=True, exist_ok=True)
        return {"status": "success", "folder": safe_name}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/folders")
async def list_folders():
    folders = []
    # Ensure root exists
    if not UPLOAD_DIR.exists():
        return []
    
    for path in UPLOAD_DIR.iterdir():
        if path.is_dir():
            folders.append(path.name)
    folders.sort()
    return folders


@app.put("/folders/{folder_name}")
async def rename_folder(folder_name: str, new_name: str = Form(...)):
    try:
        source_path = _resolve_folder(folder_name)
        if not source_path.exists() or not source_path.is_dir():
            raise HTTPException(status_code=404, detail="Folder not found")

        safe_name = _sanitize_folder_name(new_name)
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid folder name")

        if safe_name == folder_name:
            return {"status": "success", "folder": safe_name, "renamed": False}

        target_path = _safe_join(UPLOAD_DIR, safe_name)
        if target_path.exists():
            raise HTTPException(status_code=400, detail="A folder with that name already exists")

        source_path.rename(target_path)
        return {"status": "success", "folder": safe_name, "renamed": True}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/folders/{folder_name}")
async def delete_folder(folder_name: str):
    try:
        folder_path = _resolve_folder(folder_name)
        if not folder_path.exists() or not folder_path.is_dir():
            raise HTTPException(status_code=404, detail="Folder not found")
        
        # Collect the folder's image ids first so their results can be removed too
        folder_image_ids = {p.stem for p in folder_path.rglob("*")
                            if p.is_file() and ".analysis_cache" not in p.parts}

        # Delete folder and contents
        shutil.rmtree(folder_path)

        if folder_image_ids:
            _delete_results_for_ids(folder_image_ids)

        return {"status": "success", "message": "Folder deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Folder Aggregation ---

class FolderSelectionRequest(BaseModel):
    image_id: str
    selected_ids: List[int]

@app.post("/folders/{folder_name}/save_selection")
async def save_folder_selection(folder_name: str, req: FolderSelectionRequest):
    try:
        _validate_image_id(req.image_id)
        folder_path = _resolve_folder(folder_name)
        if not folder_path.exists():
            raise HTTPException(status_code=404, detail="Folder not found")

        # Analysis cache dir inside the folder (hidden)
        cache_dir = folder_path / ".analysis_cache"
        cache_dir.mkdir(exist_ok=True)

        # Load original full results
        json_path = RESULTS_DIR / f"{req.image_id}_results.json"
        if not json_path.exists():
             raise HTTPException(status_code=404, detail="Original analysis not found")
             
        with open(json_path) as f:
            data = json.load(f)
            
        # Filter
        full_results = data.get("data", [])
        filtered = [r for r in full_results if r["id"] in req.selected_ids]
        
        if not filtered:
             raise HTTPException(status_code=400, detail="No particles selected to save")
             
        # Save payload
        save_payload = {
            "image_id": req.image_id,
            "filename": data.get("filename", req.image_id),
            "data": filtered,
            "timestamp": data.get("timestamp", ""), # if available
            "pixel_size_nm": data.get("pixel_size_nm", 0)
        }
        
        output_path = cache_dir / f"{req.image_id}.json"
        with open(output_path, "w") as f:
            json.dump(save_payload, f, indent=2)
            
        return {"status": "success", "count": len(filtered)}
        
    except Exception as e:
        print(f"Save Selection Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/folders/{folder_name}/selection/{image_id}")
async def remove_folder_selection(folder_name: str, image_id: str):
    """Remove one image's saved selection from the folder analysis."""
    try:
        _validate_image_id(image_id)
        folder_path = _resolve_folder(folder_name)
        cache_file = folder_path / ".analysis_cache" / f"{image_id}.json"
        if not cache_file.exists():
            raise HTTPException(status_code=404, detail="No saved selection for this image")
        cache_file.unlink()
        return {"status": "success", "message": "Selection removed from folder analysis"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/folders/{folder_name}/aggregate")
async def aggregate_folder(folder_name: str):
    try:
        folder_path = _resolve_folder(folder_name)
        cache_dir = folder_path / ".analysis_cache"
        
        if not cache_dir.exists():
            return {"data": [], "stats": {}, "file_count": 0}
            
        combined_data = []
        files = list(cache_dir.glob("*.json"))
        
        for p in files:
            with open(p) as f:
                payload = json.load(f)
                fname = payload.get("filename", "unknown")
                # Append source info to each particle
                for p_data in payload.get("data", []):
                    p_data["source_image"] = fname
                    combined_data.append(p_data)
                    
        # Calculate Aggregated Stats
        stats = {}
        if combined_data:
            # We can use pandas for quick stats if available or manual
            # Let's use pandas since we used it in processing
            import pandas as pd
            df = pd.DataFrame(combined_data)
            
            # Helper
            def get_stat(col):
                if col not in df: return 0
                return round(float(df[col].mean()), 1), round(float(df[col].std()), 1)
            
            l_m, l_s = get_stat("length_nm")
            w_m, w_s = get_stat("width_nm")
            ar_m, ar_s = get_stat("aspect_ratio")
            
            stats = {
                "count": len(combined_data),
                "mean_length": f"{l_m} ± {l_s}",
                "mean_width": f"{w_m} ± {w_s}",
                "mean_ar": f"{ar_m} ± {ar_s}"
            }

        extended = _processing_function("compute_summary_stats")(combined_data) if combined_data else []

        return {
            "data": combined_data,
            "stats": stats,
            "extended_stats": extended,
            "file_count": len(files)
        }

    except Exception as e:
        print(f"Aggregate Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/folders/{folder_name}/export_aggregate")
async def export_aggregate_folder(folder_name: str):
    try:
        folder_path = _resolve_folder(folder_name)
        safe_folder_name = folder_path.name
        cache_dir = folder_path / ".analysis_cache"

        if not cache_dir.exists():
            raise HTTPException(status_code=404, detail="No data to export")

        combined_data = []
        files = list(cache_dir.glob("*.json"))

        for p in files:
            with open(p) as f:
                payload = json.load(f)
                fname = payload.get("filename", "unknown")
                # Append source info to each particle
                for p_data in payload.get("data", []):
                    p_data["source_image"] = fname
                    combined_data.append(p_data)

        if not combined_data:
            raise HTTPException(status_code=400, detail="No data found")

        # Generate Temp Excel (filename is derived from already-sanitized folder name)
        temp_name = f"export_folder_{safe_folder_name}_{uuid.uuid4().hex[:8]}.xlsx"
        temp_path = RESULTS_DIR / temp_name
        
        _processing_function("save_results_to_excel")(combined_data, temp_path)
        
        return FileResponse(
            path=temp_path,
            filename=f"{safe_folder_name}_analysis_report.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            background=BackgroundTask(_safe_unlink, temp_path)
        )

    except Exception as e:
        print(f"Export Aggregate Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------

@app.post("/upload")
async def upload_images(background_tasks: BackgroundTasks, folder: str = Form(None), files: List[UploadFile] = File(...)):
    uploaded_files = []
    saved_paths = []
    try:
        if len(files) > MAX_UPLOAD_FILES:
            raise HTTPException(status_code=400, detail=f"Too many files (max {MAX_UPLOAD_FILES} per request)")

        # Resolve target directory once (validates the folder name).
        if folder:
            save_dir = _resolve_folder(folder)
            if not save_dir.exists():
                raise HTTPException(status_code=400, detail="Folder does not exist")
        else:
            save_dir = UPLOAD_DIR

        for file in files:
            file_id = str(uuid.uuid4())
            # Strip any directory components and reject empty/traversal names.
            raw_name = Path(file.filename or "").name
            if not raw_name or raw_name in (".", ".."):
                raise HTTPException(status_code=400, detail="Invalid filename")
            safe_filename = _sanitize_folder_name(raw_name)
            if not safe_filename:
                raise HTTPException(status_code=400, detail="Invalid filename")
            save_name = f"{file_id}_{safe_filename}"

            file_path = _safe_join(save_dir, save_name)

            # Stream with an explicit size ceiling to avoid unbounded disk use.
            bytes_written = 0
            with open(file_path, "wb") as buffer:
                while True:
                    chunk = file.file.read(1024 * 1024)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if bytes_written > MAX_UPLOAD_BYTES:
                        buffer.close()
                        try:
                            file_path.unlink()
                        except OSError:
                            pass
                        raise HTTPException(
                            status_code=413,
                            detail=f"File exceeds maximum size of {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                        )
                    buffer.write(chunk)
            
            # 1. Generate Immediate Preview (Sync)
            # This ensures the user sees something right away
            _processing_function("generate_preview")(file_path, RESULTS_DIR)

            uploaded_files.append({
                "id": file_path.stem,
                "filename": safe_filename,
                "status": "processing"
            })
            saved_paths.append(file_path)

        # 2. Queue Heavy Processing (Background)
        # IMPORTANT: calibration lookup happens only after EVERY file in the
        # batch is saved. Doing it inside the save loop caused a race: an image
        # saved before its .dm3 partner never got calibrated (upload order is
        # effectively random, so ~half of paired uploads lost their scale).
        def _original_stem(name: str) -> str:
            if len(name) > 37 and name[36] == '_':
                return Path(name[37:]).stem
            return Path(name).stem

        for file_path in saved_paths:
            calibration_source_path = None
            target_stem = _original_stem(file_path.name)
            for f in file_path.parent.glob("*"):
                if f.suffix.lower() in ['.dm3', '.dm4', '.emd'] and f != file_path:
                    if _original_stem(f.name) == target_stem:
                        calibration_source_path = f
                        break

            background_tasks.add_task(
                _processing_function("process_image"),
                file_path,
                RESULTS_DIR,
                None,
                calibration_source_path,
            )

        return uploaded_files
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/images")
async def list_images(folder: str = Query(None)):
    images = []

    if folder:
        try:
            target_dir = _resolve_folder(folder)
        except HTTPException:
            return []
        if not target_dir.exists():
            return []
    else:
        target_dir = UPLOAD_DIR
            
    for path in target_dir.glob("*"):
        if path.is_file() and path.suffix.lower() in ['.tif', '.tiff', '.jpg', '.jpeg', '.png', '.dm3', '.dm4', '.emd']:
            # Try to extract original name if it follows UUID_Name pattern
            display_name = path.name
            if len(path.name) > 37 and path.name[36] == '_':
                display_name = path.name[37:]
            
            # Check status
            image_id = path.stem
            # If overlay exists, it's done
            overlay_path = RESULTS_DIR / f"{image_id}_overlay.jpg"
            status = "complete" if overlay_path.exists() else "processing"

            # A saved selection in the folder's analysis cache means this image
            # has been validated and added to the folder analysis.
            in_analysis = (path.parent / ".analysis_cache" / f"{image_id}.json").exists()

            images.append({
                "id": image_id,
                "filename": path.name,
                "display_name": display_name,
                "status": status,
                "in_analysis": in_analysis
            })
    # Sort by newest first (optional, but nice)
    images.sort(key=lambda x: x['display_name'])
    return images

@app.delete("/images/{image_id}")
async def delete_image(image_id: str):
    try:
        _validate_image_id(image_id)
        files = [p for p in UPLOAD_DIR.rglob("*") if p.is_file() and p.stem == image_id]
        if not files:
            raise HTTPException(status_code=404, detail="Image not found")

        for f in files:
            resolved = f.resolve()
            try:
                resolved.relative_to(UPLOAD_DIR)
            except ValueError:
                continue
            os.remove(resolved)

        for res_file in RESULTS_DIR.iterdir():
            if not res_file.is_file() or not res_file.name.startswith(image_id):
                continue
            resolved = res_file.resolve()
            try:
                resolved.relative_to(RESULTS_DIR)
            except ValueError:
                continue
            os.remove(resolved)

        return {"status": "success", "message": "Image deleted"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process/{image_id}")
async def process_image_endpoint(
    image_id: str,
    manual_pixel_size: float = None,
    requested_bar_length_nm: float = None,
    force_reprocess: bool = False,
    binary_mask_tune: int = 0
):
    try:
        input_path, calibration_source_path = _find_input_and_calibration_source(image_id)

        if not manual_pixel_size and not force_reprocess:
            # Check for existing results to avoid re-processing
            results_path = RESULTS_DIR / f"{image_id}_results.json"
            if results_path.exists():
                import json
                try:
                    with open(results_path, 'r') as f:
                        cached = json.load(f)
                    if _cached_results_are_current(cached, expected_binary_mask_tune=binary_mask_tune):
                        return cached
                except Exception:
                    pass # If corrupt, re-process

        result = _processing_function("process_image")(
            input_path,
            RESULTS_DIR,
            manual_pixel_size,
            calibration_source_path,
            requested_bar_length_nm,
            binary_mask_tune
        )

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/{image_id}/binary_preview")
async def process_binary_preview_endpoint(
    image_id: str,
    manual_pixel_size: float = None,
    binary_mask_tune: int = 0
):
    try:
        input_path, calibration_source_path = _find_input_and_calibration_source(image_id)
        return _processing_function("generate_binary_mask_preview")(
            input_path,
            RESULTS_DIR,
            manual_pixel_size,
            calibration_source_path,
            binary_mask_tune
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/results/{filename}")
async def get_result_file(filename: str):
    # Reject path separators, traversal tokens, control chars and glob metacharacters.
    if (
        not filename
        or filename in (".", "..")
        or "/" in filename
        or "\\" in filename
        or "\x00" in filename
        or any(ord(c) < 32 for c in filename)
        or any(ch in filename for ch in "*?[]")
    ):
        raise HTTPException(status_code=400, detail="Invalid filename")

    # 1. Prefer results dir.
    results_candidate = (RESULTS_DIR / filename).resolve()
    try:
        results_candidate.relative_to(RESULTS_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if results_candidate.is_file():
        return FileResponse(results_candidate)

    # 2. Fall back to uploads (root, then nested — used for original images).
    uploads_candidate = (UPLOAD_DIR / filename).resolve()
    try:
        uploads_candidate.relative_to(UPLOAD_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid filename")
    if uploads_candidate.is_file():
        return FileResponse(uploads_candidate)

    # Recursive lookup by exact basename only — filename already vetted above.
    for found in UPLOAD_DIR.rglob(filename):
        resolved = found.resolve()
        try:
            resolved.relative_to(UPLOAD_DIR)
        except ValueError:
            continue
        if resolved.is_file() and resolved.name == filename:
            return FileResponse(resolved)

    raise HTTPException(status_code=404, detail="File not found")

# --- Export ---

@app.post("/export")
async def export_data(req: ExportRequest):
    try:
        _validate_image_id(req.image_id)
        # Load cache
        json_path = RESULTS_DIR / f"{req.image_id}_results.json"
        if not json_path.exists():
            raise HTTPException(status_code=404, detail="Results not found")
            
        with open(json_path) as f:
            data = json.load(f)
            
        full_results = data.get("data", [])
        
        # Filter Logic
        # If selected_ids is empty, interpret as "None Selected" -> Empty file?
        # Or strict adherence: Only export what is in list.
        # Frontend ensures at least one is selected ideally, or we export empty.
        
        filtered_results = [r for r in full_results if r["id"] in req.selected_ids]
        
        if not filtered_results:
             raise HTTPException(status_code=400, detail="No particles selected")

        # Generate Temp Excel
        temp_name = f"export_{req.image_id}_{uuid.uuid4().hex[:8]}.xlsx"
        temp_path = RESULTS_DIR / temp_name
        
        _processing_function("save_results_to_excel")(filtered_results, temp_path)

        # Sanitize the suggested download filename — it originates from the
        # uploaded filename and flows into a response header.
        raw_name = data.get("filename") or req.image_id
        safe_download_stem = _sanitize_folder_name(Path(raw_name).stem) or req.image_id

        return FileResponse(
            path=temp_path,
            filename=f"{safe_download_stem}_detected.xlsx",
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            background=BackgroundTask(_safe_unlink, temp_path)
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"Export Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Bind to loopback only — the app is a local tool and the launchers expect 127.0.0.1.
    uvicorn.run(app, host="127.0.0.1", port=8000)
