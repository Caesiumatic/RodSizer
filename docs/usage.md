# Running RodSizer

[Project overview](../README.md)

## Desktop launchers

Download the repository and run `RodSizer_Launcher_MacOS.command` or
`RodSizer_Launcher_Windows.bat`. They create `backend/.venv`, install the
requirements, and open the local application on port 8000. The macOS launcher
uses Python 3.11 through Homebrew. On Windows, install Python with **Add Python
to PATH** selected.

Tesseract supplies scale-bar OCR. Install it with `brew install tesseract` on
macOS, or use the Windows launcher's installation instructions. Metadata and
manual calibration work independently of OCR.

## Python environment

From the repository root:

```bash
python3.11 -m venv backend/.venv
source backend/.venv/bin/activate
python -m pip install -r backend/requirements.txt
python -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000
```

On Windows, activate with `backend\.venv\Scripts\activate`.
Open <http://127.0.0.1:8000>.

## Docker

Build and run the application from the repository root:

```bash
docker build -t rodsizer .
docker run --rm -p 127.0.0.1:8000:8000 \
  -v rodsizer-uploads:/app/uploads \
  -v rodsizer-results:/app/results rodsizer
```

The named volumes retain images and results between runs. This configuration
serves one local workspace. `RodSizer_web` is a separate application with a
temporary-session model.

## Data and measurement settings

Uploaded files are in `uploads/`; reports are in `results/`. Detection settings
are documented in [the parameter guide](../PARAMETER_GUIDE.md).

`RodSizer_CLEANER_MacOS.command` permanently clears uploads, results, analysis
cache, and server logs. It preserves the Python environment.

## Code map

| Path | Purpose |
| --- | --- |
| [`backend/processing.py`](../backend/processing.py) | Segmentation and particle measurements |
| [`backend/autodetect_utils.py`](../backend/autodetect_utils.py) | Thresholding and clump separation |
| [`backend/utils.py`](../backend/utils.py) | Image readers and metadata calibration |
| [`backend/main.py`](../backend/main.py) | Image, selection, and report API |
| [`frontend/`](../frontend/) | Image review and folder analysis |
