# RodSizer

RodSizer is a local, scientific desktop tool for measuring gold nanorods (and
similar nanoparticles) in electron-microscopy images. You drop in TEM / STEM /
SEM images and it detects every particle, separates touching ones, measures each,
and produces size distributions and exportable reports — all running on your own
machine, with no data leaving your computer.

## What it does
- **Detects & segments** particles with K-means thresholding, then **separates
  touching / overlapping rods** using a marker-controlled watershed on the
  Euclidean distance transform — so dense clumps are split into individual rods
  instead of being counted as one large blob.
- **Measures** each particle from a rotated bounding box: length, width, aspect
  ratio, area, volume (hemispherically-capped cylinder), orientation, plus shape
  descriptors (solidity, circularity, eccentricity, convexity).
- **Calibrates automatically:** reads pixel size from embedded metadata (Gatan
  DM3/DM4, Velox/EMD, OME- / ImageJ / FEI TIFF). For camera exports that only
  carry a burned-in scale bar or a "Pixel size / Fov" footer, it reads the value
  with on-image OCR. Manual draw-a-line calibration is always available.
- **Lets you curate results** on an interactive analysis page: click a particle
  to keep/exclude it, sort the table by area (or L / W / AR / ID), bulk
  select/deselect everything above or below a row, auto-remove statistical
  outliers (Tukey 1.5×IQR), and toggle the on-image ID labels.
- **Exports** per-image and per-folder statistics, histograms, and CSV / Excel /
  PDF reports.

## Typical workflow
1. Create a folder and upload images (you can drop in many at once).
2. Open each image to review detection. Fix the scale if needed, then deselect
   clumps / misreads — click them on the image, or use the sort, bulk-select, and
   "Deselect outliers" tools in the selection panel.
3. Press **Update Particles** to generate statistics and histograms, then
   download the report.
4. Use **View Analysis** to aggregate the whole folder.

## Methodology
Segmentation follows the AutoDetect-mNP approach (*JACS Au* **2021**, *1*,
316−327), extended in RodSizer with distance-transform watershed splitting for
dense clumps and OCR scale-bar calibration for un-tagged camera exports.

## Contributors
- Shi Chen — Murphy Group, UIUC
- Arda Turk — Murphy Group, UIUC
- Built with assistance from Claude and ChatGPT.

Group website: https://murphy-group.chemistry.illinois.edu/

## Instruction for Launching
1. Click on "<>code" in Github page.
2. Click on "Download ZIP".
3. Open Finder (MacOS) or Files (Windows).
4. Click on the RodSizer.zip to unzip.
5. Double-click on RodSizer_Launcher_MacOS.command or RodSizer_Launcher_Windows.bat.
6. Wait for the environment to be set up (first time only) and the local website to be opened.

## Notes:
- DO NOT double-click on RodSizer_CLEANER_MacOS.command unless you are sure that you want to clean ALL local history of data and reports.
- First-time launching may take some time, like 5-10 mins. Most of that is
  installing the deep-learning packages (`tensorflow` + `stardist`, ~1 GB). These
  are kept for possible future ML-based detection but are **not used** by the
  current pipeline (K-means + watershed), so the wait is one-time setup only —
  later launches are fast. See `backend/requirements.txt` to remove them if you
  want a leaner install.
- Try ask a coding agent if there's an issue with environment setup.
- MacOS is more recommended.
- (Windows) If Windows Defender asks, click `More Info` -> `Run Anyway`.
- (Windows) Users should ensure `Add Python to PATH` is checked during installation.
- Automatic scale-bar OCR uses Tesseract, which the launcher installs for you
  (Homebrew on macOS, winget on Windows). If it cannot be installed, RodSizer
  still runs — just calibrate manually.

## Cleaner Script (Mac)
- `RodSizer_CLEANER_MacOS.command` is a cleanup tool for clearing local history data.
- It permanently removes:
  - uploaded files in `uploads/`
  - generated results in `results/`
  - folder analysis cache in `.analysis_cache`
  - `backend/server.log`
  - Python cache files (`__pycache__` and `*.pyc`)
- It does not remove your Python virtual environment or installed packages in `backend/.venv`.
- Use it only when you want to reset the app's local working history and cached outputs.

## Parameter Guide
- [`PARAMETER_GUIDE.md`](PARAMETER_GUIDE.md) is a reference document for the current detection and measurement parameters used by RodSizer.
- It is mainly intended for developers or advanced users who want to understand or adjust processing behavior.
- The actual implementation is in `backend/processing.py`.

## Requirements
- macOS or Windows
- Python 3 installed (standard on most Macs, or downloadable from `python.org`)
