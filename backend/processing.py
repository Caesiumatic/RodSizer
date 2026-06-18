from pathlib import Path
import cv2
import numpy as np
from skimage import measure, morphology
from utils import get_pixel_size, read_emd_image, read_emd_pixel_size
import pandas as pd
import math
from scipy import ndimage as ndi
import ncempy.io as nio
from autodetect_utils import image_kmeans, ruecs, dilmarkers, split_clump

def save_results_to_excel(results, output_path):
    """
    Saves results to an Excel file with 'Statistics' and 'Data' sheets.
    """
    # Prepare DataFrame for export
    df_export = pd.DataFrame(results)
    if df_export.empty:
        return

    # Remove contour column if present for clean export
    cols_to_drop = [c for c in ["contour", "contour_full"] if c in df_export.columns]
    df_export = df_export.drop(columns=cols_to_drop)

    # Calculate Stats for Excel
    s_mean = df_export.mean(numeric_only=True).round(1)
    s_std = df_export.std(numeric_only=True).round(1)
    
    stats_rows = []
    stats_rows.append({"Metric": "Count", "Value": len(df_export)})
    stats_rows.append({"Metric": "Mean Length (nm)", "Value": f"{s_mean.get('length_nm', 0)} ± {s_std.get('length_nm', 0)}"})
    stats_rows.append({"Metric": "Mean Width (nm)", "Value": f"{s_mean.get('width_nm', 0)} ± {s_std.get('width_nm', 0)}"})
    stats_rows.append({"Metric": "Mean AR", "Value": f"{s_mean.get('aspect_ratio', 0)} ± {s_std.get('aspect_ratio', 0)}"})
    stats_df = pd.DataFrame(stats_rows)

    # Save to Excel
    try:
        with pd.ExcelWriter(output_path) as writer:
            stats_df.to_excel(writer, sheet_name="Statistics", index=False)
            df_export.to_excel(writer, sheet_name="Data", index=False)
    except Exception as e:
        print(f"Excel export failed: {e}")


def calculate_volume(length_nm, width_nm):
    """
    Calculate volume of a hemispherically capped cylinder (nanorod).
    V = pi * r^2 * (L - 2r) + 4/3 * pi * r^3
    where r = width / 2
    """
    r = width_nm / 2.0
    # If rod is very short (L < W), treat as sphere or prolate spheroid? 
    # Standard formula assumes L >= W. If L < W, it's not a rod.
    # We'll clamp L-2r to 0 if L < 2r (though physically L should be > W for a rod)
    cyl_height = max(0, length_nm - width_nm)
    
    v_cyl = np.pi * (r**2) * cyl_height
    v_caps = (4.0/3.0) * np.pi * (r**3)
    
    return v_cyl + v_caps


def _fast_split_large_component(crop: np.ndarray, min_size_px: int):
    """
    Fast fallback for very large fused regions where recursive rUECS can be
    prohibitively slow. It separates narrow bridges, dilates markers back, and
    keeps masks constrained to the original component.
    """
    crop_bool = crop.astype(bool)
    if not np.any(crop_bool):
        return []

    area = int(crop_bool.sum())
    radius = 2 if area < 250_000 else 3
    seed = morphology.erosion(crop_bool, morphology.disk(radius))
    seed = morphology.opening(seed, morphology.disk(1))

    labeled, _ = ndi.label(seed)
    min_marker_area = max(3, int(min_size_px / 4))
    split_masks = []

    for region in measure.regionprops(labeled):
        if region.area < min_marker_area:
            continue

        marker = labeled == region.label
        grown = marker
        for _ in range(radius + 1):
            grown = morphology.dilation(grown, morphology.disk(1))

        grown &= crop_bool
        if grown.sum() >= min_marker_area:
            split_masks.append(grown)

    return split_masks or [crop_bool]


def _make_label_overlay(img: np.ndarray, labels: np.ndarray):
    base = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    max_label = int(labels.max()) if labels.size else 0
    if max_label <= 0:
        return base

    idx = np.arange(max_label + 1, dtype=np.uint16)
    color_lut = np.zeros((max_label + 1, 3), dtype=np.uint8)
    color_lut[1:, 0] = ((37 * idx[1:]) % 200 + 40).astype(np.uint8)
    color_lut[1:, 1] = ((91 * idx[1:]) % 200 + 40).astype(np.uint8)
    color_lut[1:, 2] = ((151 * idx[1:]) % 200 + 40).astype(np.uint8)

    mask = labels > 0
    color_img = color_lut[labels]
    overlay = base.copy()
    overlay[mask] = (
        (base[mask].astype(np.uint16) * 6 + color_img[mask].astype(np.uint16) * 4) // 10
    ).astype(np.uint8)
    return overlay


def generate_preview(image_path: Path, output_dir: Path):
    """
    Generate a quick JPEG preview of the image for immediate display.
    Files are saved as {image_id}_preview.jpg
    """
    try:
        image_id = image_path.stem
        ext = image_path.suffix.lower()
        img = None
        
        if ext in ['.dm3', '.dm4']:
            try:
                dm = nio.read(str(image_path))
                raw_data = dm['data']
                if raw_data.ndim == 3:
                    raw_data = raw_data[0]
                norm_data = cv2.normalize(raw_data, None, 0, 255, cv2.NORM_MINMAX)
                img = norm_data.astype(np.uint8)
            except:
                pass

        if img is None and ext == '.emd':
            try:
                img = read_emd_image(image_path)
            except Exception:
                pass

        if img is None:
            # Try reading with OpenCV (works for TIFF, PNG, JPG)
            # Use IMREAD_UNCHANGED to get original depth then normalize
            img_raw = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if img_raw is not None:
                # Normalize to 8-bit for display
                if img_raw.dtype != np.uint8:
                     img = cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX)
                     img = img.astype(np.uint8)
                else:
                     img = img_raw
        
        if img is not None:
            preview_path = output_dir / f"{image_id}_preview.jpg"
            cv2.imwrite(str(preview_path), img)
            return True
            
    except Exception as e:
        print(f"Error generating preview for {image_path}: {e}")
    
    return False


def _save_binary_image(output_dir: Path, image_id: str, binary: np.ndarray, suffix: str):
    filename = f"{image_id}_{suffix}.png"
    path = output_dir / filename
    binary_uint8 = (binary.astype(np.uint8) * 255)
    cv2.imwrite(str(path), binary_uint8)
    return filename


def generate_binary_mask_preview(
    image_path: Path,
    output_dir: Path,
    manual_pixel_size: float = None,
    calibration_source_path: Path = None,
    binary_mask_tune: int = 0
):
    image_id = image_path.stem
    ext = image_path.suffix.lower()
    img = None
    pixel_size_nm = None
    calibration_info = {}

    def read_dm3_pixel_size(dm3_path):
        try:
            dm = nio.read(str(dm3_path))
            if 'pixelSize' in dm:
                return float(dm['pixelSize'][0])
        except Exception as e:
            print(f"Error reading Gatan metadata: {e}")
        return None

    if calibration_source_path and calibration_source_path.exists():
        cal_ext = calibration_source_path.suffix.lower()
        if cal_ext == '.emd':
            pixel_size_nm = read_emd_pixel_size(calibration_source_path)
        else:
            pixel_size_nm = read_dm3_pixel_size(calibration_source_path)
        if pixel_size_nm:
            calibration_info = {
                "method": "linked_metadata",
                "pixel_size_nm": pixel_size_nm,
                "source_file": calibration_source_path.name,
                "description": f"Calibration: {calibration_source_path.name}"
            }

    if ext in ['.dm3', '.dm4']:
        try:
            dm = nio.read(str(image_path))
            raw_data = dm['data']
            if raw_data.ndim == 3:
                raw_data = raw_data[0]
            norm_data = cv2.normalize(raw_data, None, 0, 255, cv2.NORM_MINMAX)
            img = norm_data.astype(np.uint8)
            if pixel_size_nm is None and 'pixelSize' in dm:
                pixel_size_nm = float(dm['pixelSize'][0])
                calibration_info = {"method": "metadata_dm", "pixel_size_nm": pixel_size_nm}
        except Exception as e:
            print(f"Error reading Gatan file: {e}")
    elif ext == '.emd':
        img = read_emd_image(image_path)
        if img is not None and pixel_size_nm is None:
            emd_ps = read_emd_pixel_size(image_path)
            if emd_ps:
                pixel_size_nm = emd_ps
                calibration_info = {"method": "metadata_emd", "pixel_size_nm": pixel_size_nm}

    if img is None:
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise ValueError("Could not read image")

    if manual_pixel_size is not None and manual_pixel_size > 0:
        pixel_size_nm = manual_pixel_size
        calibration_info = {"method": "manual", "scale_bar_length_nm": "Manual"}
    elif pixel_size_nm is None:
        pixel_size_nm, calibration_info = get_pixel_size(image_path)

    if pixel_size_nm is None:
        pixel_size_nm = 1.0
        calibration_info = calibration_info or {}
        calibration_info["method"] = "uncalibrated"
        calibration_info["is_placeholder"] = True
        calibration_info.setdefault(
            "warning",
            "No calibration found. Measurements are using a placeholder scale until you calibrate manually."
        )
    else:
        calibration_info = calibration_info or {}
        calibration_info.setdefault("is_placeholder", False)

    binary_mask_tune = int(np.clip(binary_mask_tune, -6, 6))
    binary = image_kmeans(img, separation_strength=binary_mask_tune)

    if calibration_info.get("scale_bar_coords"):
        x1, y1, x2, y2 = calibration_info["scale_bar_coords"]
        h_img, w_img = binary.shape
        mask_y1 = max(0, y1 - 120)
        mask_y2 = min(h_img, y2 + 40)
        mask_x1 = max(0, x1 - 40)
        mask_x2 = min(w_img, x2 + 40)
        binary[mask_y1:mask_y2, mask_x1:mask_x2] = False

    preview_filename = _save_binary_image(output_dir, image_id, binary, "binary_preview")

    return {
        "binary_preview_url": f"/results/{preview_filename}",
        "binary_mask_tune": binary_mask_tune,
        "pixel_size_nm": pixel_size_nm,
        "calibration_info": calibration_info
    }


def process_image(
    image_path: Path,
    output_dir: Path,
    manual_pixel_size: float = None,
    calibration_source_path: Path = None,
    requested_bar_length_nm: float = None,
    binary_mask_tune: int = 0
):
    # Default to 200nm if not specified, as per user request to "have blue line as 200 nm"
    if requested_bar_length_nm is None:
        requested_bar_length_nm = 200.0

    # 1. Load Image
    image_id = image_path.stem
    ext = image_path.suffix.lower()
    img = None
    pixel_size_nm = None
    calibration_info = {}

    # Helper to read DM3 metadata
    def read_dm3_pixel_size(dm3_path):
        try:
            dm = nio.read(str(dm3_path))
            if 'pixelSize' in dm:
                return float(dm['pixelSize'][0])
        except Exception as e:
            print(f"Error reading Gatan metadata: {e}")
        return None

    # Check if we have an external calibration source (linked .dm3/.dm4/.emd file)
    if calibration_source_path and calibration_source_path.exists():
        cal_ext = calibration_source_path.suffix.lower()
        if cal_ext == '.emd':
            pixel_size_nm = read_emd_pixel_size(calibration_source_path)
        else:
            pixel_size_nm = read_dm3_pixel_size(calibration_source_path)
        if pixel_size_nm:
            calibration_info = {
                "method": "linked_metadata",
                "pixel_size_nm": pixel_size_nm,
                "source_file": calibration_source_path.name,
                "description": f"Calibration: {calibration_source_path.name}"
            }

    if ext in ['.dm3', '.dm4']:
        try:
            dm = nio.read(str(image_path))
            raw_data = dm['data']
            if raw_data.ndim == 3:
                raw_data = raw_data[0]
            norm_data = cv2.normalize(raw_data, None, 0, 255, cv2.NORM_MINMAX)
            img = norm_data.astype(np.uint8)
            if pixel_size_nm is None and 'pixelSize' in dm:
                pixel_size_nm = float(dm['pixelSize'][0])
                calibration_info = {"method": "metadata_dm", "pixel_size_nm": pixel_size_nm}
        except Exception as e:
            print(f"Error reading Gatan file: {e}")

    elif ext == '.emd':
        img = read_emd_image(image_path)
        if img is not None and pixel_size_nm is None:
            emd_ps = read_emd_pixel_size(image_path)
            if emd_ps:
                pixel_size_nm = emd_ps
                calibration_info = {"method": "metadata_emd", "pixel_size_nm": pixel_size_nm}

    if img is None:
        # Standard image load
        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        raise ValueError("Could not read image")

    # 2. Get Pixel Size (Calibration) - Override or Fallback
    if manual_pixel_size is not None and manual_pixel_size > 0:
        pixel_size_nm = manual_pixel_size
        calibration_info = {"method": "manual", "scale_bar_length_nm": "Manual"}
    elif pixel_size_nm is None:
        # Try getting from utils (embedded metadata where available)
        pixel_size_nm, calibration_info = get_pixel_size(image_path)
    
    # Ensure pixel_size_nm is valid
    if pixel_size_nm is None:
        pixel_size_nm = 1.0  # Placeholder to keep pixel-domain processing alive
        calibration_info = calibration_info or {}
        calibration_info["method"] = "uncalibrated"
        calibration_info["is_placeholder"] = True
        calibration_info.setdefault(
            "warning",
            "No calibration found. Measurements are using a placeholder scale until you calibrate manually."
        )
    else:
        calibration_info = calibration_info or {}
        calibration_info.setdefault("is_placeholder", False)
    

    
    # 3. Preprocessing & Segmentation (AutoDetect-mNP)
    # User requested "Option 4": AutoDetect-mNP (K-means + rUECS)
    
    # Step 1: K-means Segmentation
    # This replaces Adaptive Thresholding
    binary_mask_tune = int(np.clip(binary_mask_tune, -6, 6))
    binary = image_kmeans(img, separation_strength=binary_mask_tune)
    
    # MASKING SCALE BAR (Fix for "detecting rods near scale")
    # image_kmeans already masks common camera footer strips; keep explicit
    # scale-bar coordinates honored when metadata/detection supplies them.
    if calibration_info.get("scale_bar_coords"):
        x1, y1, x2, y2 = calibration_info["scale_bar_coords"]
        # Mask out a slightly larger box around the line
        # The text is usually above the line.
        # Let's mask a box from y1-50 to y2+20 (approx)
        h_img, w_img = binary.shape
        
        # Safety bounds
        mask_y1 = max(0, y1 - 120) # Assume text is above (increased to 120px for large text)
        mask_y2 = min(h_img, y2 + 40) # Increased bottom margin too
        mask_x1 = max(0, x1 - 40) # Wider margin
        mask_x2 = min(w_img, x2 + 40)
        
        # Set to False (Background)
        binary[mask_y1:mask_y2, mask_x1:mask_x2] = False
    
    # Step 2: Separate Simple vs Complex objects
    labels, _ = ndi.label(binary)
    regions = measure.regionprops(labels)

    min_area_nm2 = 30
    min_size_px = max(500, int(min_area_nm2 / (pixel_size_nm * pixel_size_nm))) if pixel_size_nm else 500
    min_marker_area = max(1, min_size_px // 4)
    split_labels = np.zeros_like(labels, dtype=np.int32)
    next_label = 1

    for region in regions:
        if region.area < min_size_px:
            continue

        minr, minc, maxr, maxc = region.bbox
        label_crop = split_labels[minr:maxr, minc:maxc]

        # Solidity > 0.9 is treated as a single rod ("simple"); lower values are
        # treated as clumps and separated with a distance-transform watershed
        # (autodetect_utils.split_clump). Watershed keeps a single elongated rod
        # whole while breaking touching rods apart, and is far faster than the
        # recursive rUECS erosion it replaces.
        if region.solidity > 0.9:
            label_crop[region.image] = next_label
            next_label += 1
        else:
            split_masks = split_clump(
                region.image,
                min_marker_area,
                separation_strength=binary_mask_tune,
            )
            for d_mask in split_masks:
                label_crop[d_mask] = next_label
                next_label += 1

    labels = split_labels

    # ... (post-processing comments) ...
    
    # 5. Measurement & Filtering
    candidates = []
    output_image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Create a mask for NMS
    occupied_mask = np.zeros(binary.shape, dtype=np.uint8)

    # Iterate labels via their bounding boxes instead of scanning the whole image
    # for every label. find_objects returns one slice per label (index = label-1),
    # so all per-object work happens on a small crop — O(total object area) rather
    # than O(n_labels x image_size).
    h_img, w_img = labels.shape
    object_slices = ndi.find_objects(labels)

    for label_idx, sl in enumerate(object_slices, start=1):
        if sl is None:
            continue

        row_slice, col_slice = sl
        minr, minc = row_slice.start, col_slice.start
        maxr, maxc = row_slice.stop, col_slice.stop

        obj_mask = (labels[row_slice, col_slice] == label_idx).astype(np.uint8)

        # Skip objects touching the image border (cannot be measured reliably).
        if ((minr == 0 and obj_mask[0, :].any()) or
                (maxr == h_img and obj_mask[-1, :].any()) or
                (minc == 0 and obj_mask[:, 0].any()) or
                (maxc == w_img and obj_mask[:, -1].any())):
            continue

        # Find contours (in crop-local coordinates).
        contours, _ = cv2.findContours(obj_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        cnt_local = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(cnt_local)

        if area_px < min_size_px:
            continue

        # Fit Rotated Rectangle (User preference for "edges")
        rect = cv2.minAreaRect(cnt_local)
        (center, (w_rect, h_rect), angle_rect) = rect

        # Normalize width/height (Length is always the longer dimension)
        if w_rect < h_rect:
            width_px = w_rect
            length_px = h_rect
            # Angle logic for minAreaRect:
            # OpenCV 4.5+: angle is in [0, 90).
            # We just want the orientation of the major axis.
            angle = angle_rect
        else:
            width_px = h_rect
            length_px = w_rect
            angle = angle_rect + 90

        if width_px == 0: continue

        # Calculate Shape Descriptors
        # 1. Area
        # area_px

        # 2. Aspect Ratio (from Rectangle)
        length_nm = length_px * pixel_size_nm
        width_nm = width_px * pixel_size_nm
        aspect_ratio = length_nm / width_nm

        # 3. Solidity = Area / Convex Area
        hull = cv2.convexHull(cnt_local)
        convex_area = cv2.contourArea(hull)
        solidity = area_px / convex_area if convex_area > 0 else 0

        # 4. Convexity = Convex Perimeter / Perimeter
        perimeter = cv2.arcLength(cnt_local, True)
        hull_perimeter = cv2.arcLength(hull, True)
        convexity = hull_perimeter / perimeter if perimeter > 0 else 0

        # 5. Circularity
        circularity = (4 * np.pi * area_px) / (perimeter ** 2) if perimeter > 0 else 0

        # 6. Eccentricity (Keep using Ellipse fit for this standard definition)
        if len(cnt_local) >= 5:
            (_, (w_ell, h_ell), _) = cv2.fitEllipse(cnt_local)
            major_axis = max(w_ell, h_ell)
            minor_axis = min(w_ell, h_ell)
            eccentricity = np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0
        else:
            eccentricity = 0

        volume_nm3 = calculate_volume(length_nm, width_nm)

        # Translate contour and centre from crop-local to full-image coordinates
        # (contour points are stored as (x, y) == (col, row)).
        offset = np.array([[[minc, minr]]], dtype=cnt_local.dtype)
        cnt = cnt_local + offset
        center = (center[0] + minc, center[1] + minr)

        candidates.append({
            "center": center,
            "size": (width_px, length_px), # Store as (W, L) for consistency, though minAreaRect is (w,h)
            "angle": angle,
            "length_nm": length_nm,
            "width_nm": width_nm,
            "aspect_ratio": aspect_ratio,
            "volume_nm3": volume_nm3,
            "area_px": area_px,
            "solidity": solidity,
            "convexity": convexity,
            "eccentricity": eccentricity,
            "circularity": circularity,
            "orientation_deg": angle,
            "bbox": (minr, minc),       # crop origin for localized NMS
            "footprint": obj_mask,      # crop-local pixel mask for NMS
            "contour": cnt              # full-image contour for coloring
        })


    # Sort candidates by Area (descending)
    candidates.sort(key=lambda x: x["area_px"], reverse=True)
    
    results = []
    
    # Non-Maximum Suppression (NMS) — dedupe overlapping detections.
    # Each candidate's footprint is compared against the occupied mask only within
    # its own bounding box, so this is O(total object area) rather than allocating
    # and scanning a full-image mask per candidate.
    for cand in candidates:
        minr, minc = cand["bbox"]
        footprint = cand["footprint"]
        fh, fw = footprint.shape

        cand_area = int(footprint.sum())
        if cand_area == 0:
            continue

        occ_window = occupied_mask[minr:minr + fh, minc:minc + fw]
        overlap = int(np.count_nonzero(footprint & occ_window))
        overlap_ratio = overlap / cand_area

        if overlap_ratio > 0.15:
            continue

        # Accept it — mark its footprint as occupied (in-place on the view).
        occ_window |= footprint

        # Add to results
        results.append({
            "id": len(results) + 1,
            "length_nm": float(round(cand["length_nm"], 1)),
            "width_nm": float(round(cand["width_nm"], 1)),
            "aspect_ratio": float(round(cand["aspect_ratio"], 1)),
            "volume_nm3": float(round(cand["volume_nm3"], 1)),
            "orientation_deg": float(round(cand["orientation_deg"], 1)),
            "centroid_x": int(cand["center"][0]),
            "centroid_y": int(cand["center"][1]),
            "area_px": int(cand["area_px"]),
            "solidity": float(round(cand["solidity"], 3)),
            "convexity": float(round(cand["convexity"], 3)),
            "circularity": float(round(cand["circularity"], 3)),
            "eccentricity": float(round(cand["eccentricity"], 3)),
            "contour": cand["contour"] # Keep for coloring
        })
        
        # Boxes and ID numbers are NOT baked into the image. The frontend draws
        # them as an interactive canvas overlay so they reflect the live
        # selection state, can be toggled off when they obscure small particles,
        # and stay crisp at any zoom. The saved image keeps only the scale bar.

    # Clean up source file name in calibration info for frontend
    def get_clean_filename(path: Path):
        name = path.name
        if len(name) > 37 and name[36] == '_':
            return name[37:]
        return name

    clean_name = get_clean_filename(image_path)
    # Removed drawing filename on image as requested
    
    # Draw Scale Bar Verification
    scale_drawn = False
    has_real_scale = bool(pixel_size_nm) and not calibration_info.get("is_placeholder")
    effective_requested_bar_nm = requested_bar_length_nm if has_real_scale else None
    
    # 0. If user requested a specific length, force synthetic bar (skip detection viz)
    if effective_requested_bar_nm:
        # Will fall through to synthetic block
        pass
    # 1. Try to draw over detected line (only if no manual override)
    elif calibration_info.get("scale_bar_coords") and has_real_scale:
        x1, y1, x2, y2 = calibration_info["scale_bar_coords"]
        
        # Calculate length if missing
        if calibration_info.get("scale_bar_length_nm") is None and pixel_size_nm:
            width_px = x2 - x1
            raw_length_nm = width_px * pixel_size_nm
            calibration_info["scale_bar_length_nm"] = int(round(raw_length_nm / 10.0)) * 10
            
        y_offset = 40
        cv2.line(output_image, (x1, y1 + y_offset), (x2, y2 + y_offset), (255, 0, 0), 5)
        
        label = f"Scale: {calibration_info['scale_bar_length_nm']} nm"
        cv2.putText(output_image, label, (x1, y1 + y_offset - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        scale_drawn = True
        
    # 2. If no detected line OR manual override, draw synthetic bar
    if (not scale_drawn and has_real_scale) or effective_requested_bar_nm:
        h, w = output_image.shape[:2]
        
        if effective_requested_bar_nm:
            bar_length_nm = effective_requested_bar_nm
        else:
            # Choose a nice round number for the bar
            target_width_px = w * 0.2 # Target 20% of image width
            target_nm = target_width_px * pixel_size_nm
            
            # Snap to 10, 20, 50, 100, 200, 500, 1000...
            magnitude = 10 ** math.floor(math.log10(target_nm))
            residual = target_nm / magnitude
            if residual > 5:
                bar_length_nm = 5 * magnitude
            elif residual > 2:
                bar_length_nm = 2 * magnitude
            else:
                bar_length_nm = 1 * magnitude
            
        bar_width_px = int(bar_length_nm / pixel_size_nm)
        
        # Position: Bottom Left (User requested specific area, using safe bottom-left)
        x1 = 128
        if x1 + bar_width_px > w: x1 = 20 # Safety check
        x2 = x1 + bar_width_px
        y = h - 100 # Safe bottom margin
        
        cv2.line(output_image, (x1, y), (x2, y), (255, 0, 0), 10) # Thicker line
        cv2.putText(output_image, f"{int(bar_length_nm)} nm", (x1, y - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
        scale_drawn = True

    if not scale_drawn:
        warning_text = "Scale not calibrated" if calibration_info.get("is_placeholder") else "Scale not detected"
        cv2.putText(output_image, warning_text, (50, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Clean up source file name in calibration info for frontend
    if "source_file" in calibration_info:
        src_name = calibration_info["source_file"]
        if len(src_name) > 37 and src_name[36] == '_':
             calibration_info["source_file"] = src_name[37:]
             # Update description to be clean
             if "description" in calibration_info:
                 calibration_info["description"] = f"Calibration: {calibration_info['source_file']}"

    # 6. Save results
    result_image_filename = f"{image_id}_processed.jpg"
    result_image_path = output_dir / result_image_filename
    cv2.imwrite(str(result_image_path), output_image)
    
    # Save Binary Mask for debugging
    binary_filename = _save_binary_image(output_dir, image_id, binary, "binary")
    
    overlay_filename = f"{image_id}_overlay.jpg"
    overlay_path = output_dir / overlay_filename
    cv2.imwrite(str(overlay_path), _make_label_overlay(img, labels))
    
    csv_filename = f"{image_id}_results.csv"
    xlsx_filename = f"{image_id}_results.xlsx"
    csv_path = output_dir / csv_filename
    xlsx_path = output_dir / xlsx_filename

    # Save to CSV (Clean)
    df_export = pd.DataFrame(results)
    if "contour" in df_export.columns:
        df_export = df_export.drop(columns=["contour"])
    if "contour_full" in df_export.columns:
        df_export = df_export.drop(columns=["contour_full"])

    df_export.to_csv(csv_path, index=False)

    # Save to Excel (Reusable function)
    save_results_to_excel(results, xlsx_path)
    
    # Helper to sanitize values for JSON
    def sanitize(val):
        if isinstance(val, (float, np.floating)):
            if np.isnan(val) or np.isinf(val):
                return 0.0
        return val

    # Sanitize results
    sanitized_results = []
    for res in results:
        sanitized_res = {k: sanitize(v) for k, v in res.items()}
        sanitized_results.append(sanitized_res)

    # Calculate statistics
    stats = {}
    if results:
        df_res = pd.DataFrame(results)
        stats = {
            "count": len(results),
            "mean_length": sanitize(round(df_res["length_nm"].mean(), 1)),
            "std_length": sanitize(round(df_res["length_nm"].std(), 1)),
            "mean_width": sanitize(round(df_res["width_nm"].mean(), 1)),
            "std_width": sanitize(round(df_res["width_nm"].std(), 1)),
            "mean_volume": sanitize(round(df_res["volume_nm3"].mean(), 1)),
            "std_volume": sanitize(round(df_res["volume_nm3"].std(), 1)),
        }

    # Sanitize results for JSON serialization (remove numpy arrays like 'contour')
    sanitized_results = []
    for r in results:
        r_copy = r.copy()
        if "contour" in r_copy:
            del r_copy["contour"]
        # Apply general sanitization to other values
        sanitized_res_item = {k: sanitize(v) for k, v in r_copy.items()}
        sanitized_results.append(sanitized_res_item)

    output_data = {
        "results_schema_version": 6,
        "binary_mask_tune": binary_mask_tune,
        "filename": clean_name,
        "data": sanitized_results,
        "image_url": f"/results/{result_image_filename}",
        "binary_url": f"/results/{binary_filename}",
        "overlay_url": f"/results/{overlay_filename}",
        "csv_url": f"/results/{csv_filename}",
        "excel_url": f"/results/{xlsx_filename}",
        "statistics": stats,
        "pixel_size_nm": pixel_size_nm,
        "calibration_info": calibration_info,
        "filename": clean_name
    }
    
    # Save to JSON for caching
    json_path = output_dir / f"{image_id}_results.json"
    import json
    with open(json_path, 'w') as f:
        json.dump(output_data, f, indent=4)

    return output_data
