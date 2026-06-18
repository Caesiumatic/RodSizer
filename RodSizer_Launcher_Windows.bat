@echo off
cd /d "%~dp0"
setlocal enabledelayedexpansion

set "PROJECT_DIR=%CD%"
set "BACKEND_DIR=%PROJECT_DIR%\backend"
set "VENV_DIR=%PROJECT_DIR%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "APP_URL=http://127.0.0.1:8501"

echo ==================================================
echo       RodSizer - Launcher
echo ==================================================
echo.

set "PY_EXE="

REM --- 1. Try 'python' command ---
python --version >nul 2>&1
if %errorlevel% equ 0 set "PY_EXE=python"
if defined PY_EXE goto FOUND_PYTHON

REM --- 2. Try 'py' launcher ---
py --version >nul 2>&1
if !errorlevel! equ 0 set "PY_EXE=py"
if defined PY_EXE goto FOUND_PYTHON

REM --- 3. Try Auto-Detect Common Paths ---
if exist "C:\Python39\python.exe" set "PY_EXE=C:\Python39\python.exe" & goto FOUND_PYTHON
if exist "C:\Python310\python.exe" set "PY_EXE=C:\Python310\python.exe" & goto FOUND_PYTHON
if exist "C:\Python311\python.exe" set "PY_EXE=C:\Python311\python.exe" & goto FOUND_PYTHON
if exist "C:\Program Files\Python39\python.exe" set "PY_EXE=C:\Program Files\Python39\python.exe" & goto FOUND_PYTHON
if exist "C:\Program Files\Python310\python.exe" set "PY_EXE=C:\Program Files\Python310\python.exe" & goto FOUND_PYTHON
if exist "C:\Program Files\Python311\python.exe" set "PY_EXE=C:\Program Files\Python311\python.exe" & goto FOUND_PYTHON

REM --- 4. User Input Fallback ---
:ASK_USER
echo.
echo [ERROR] Could not find Python automatically.
echo.
echo Please locate your 'python.exe' file.
echo TIP: Look in "C:\Program Files\Python3x" or "AppData\Local\Programs\Python"
echo.
echo IMPORTANT: Drag the FILE, not a shortcut.
echo.
set /p "PY_EXE=Drag and Drop python.exe here and press Enter: "
set "PY_EXE=!PY_EXE:"=!"

REM Validate Shortcut
echo "!PY_EXE!" | findstr /i ".lnk" >nul
if %errorlevel% equ 0 (
    echo.
    echo [ERROR] You provided a Shortcut (.lnk).
    echo Please use the real python.exe.
    goto ASK_USER
)

:FOUND_PYTHON
echo.
echo [INFO] Using Python: "!PY_EXE!"

REM Verify selected Python
"!PY_EXE!" --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo [CRITICAL ERROR] Selected python path is invalid.
    pause
    exit /b 1
)

if not exist "%BACKEND_DIR%\requirements.txt" (
    echo.
    echo [CRITICAL ERROR] Missing backend\requirements.txt
    pause
    exit /b 1
)

if not exist "%BACKEND_DIR%\main.py" (
    echo.
    echo [CRITICAL ERROR] Missing backend\main.py
    pause
    exit /b 1
)

REM --- 4b. Tesseract OCR (optional) ---
REM Used to auto-read the pixel size / scale bar burned into camera-export
REM images. RodSizer still runs without it (manual calibration remains).
where tesseract >nul 2>&1
if %errorlevel% neq 0 (
    if not exist "C:\Program Files\Tesseract-OCR\tesseract.exe" (
        where winget >nul 2>&1
        if !errorlevel! equ 0 (
            echo [INFO] Installing Tesseract OCR for automatic scale-bar reading...
            winget install --id UB-Mannheim.TesseractOCR -e --silent --accept-source-agreements --accept-package-agreements >nul 2>&1
        ) else (
            echo [INFO] Tesseract OCR not found and winget unavailable - skipping.
            echo        Auto scale-bar reading will be off; manual calibration still works.
        )
    )
)

REM --- 5. Setup Virtual Environment ---
if exist "%VENV_PYTHON%" goto CHECK_ENV
echo.
echo [First Run] Setting up Python environment...
"!PY_EXE!" -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo.
    echo [CRITICAL ERROR] Failed to create virtual environment.
    pause
    exit /b 1
)

REM Clear screen to focus attention
cls
echo ==================================================================
echo       FIRST RUN SETUP - INSTALLING DEPENDENCIES
echo ==================================================================
echo.
echo    Downloading: OpenCV, NumPy, FastAPI, Uvicorn, and others...
echo.
echo    PLEASE WAIT: This typically takes 1-3 minutes.
echo    You will see download progress below.
echo.
echo    (Do not close this window)
echo.
echo ==================================================================
echo.

REM Give user 2 seconds to read before text flies by
timeout /t 2 >nul

"%VENV_PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
    echo.
    echo [CRITICAL ERROR] Failed to upgrade pip.
    pause
    exit /b 1
)

"%VENV_PYTHON%" -m pip install -r "%BACKEND_DIR%\requirements.txt"
if errorlevel 1 (
    echo.
    echo [CRITICAL ERROR] Dependency installation failed.
    echo Please run RodSizer_DEBUG_Windows.bat for a guided repair.
    pause
    exit /b 1
)

:CHECK_ENV
echo.
echo [INFO] Checking Python environment...
"%VENV_PYTHON%" -c "import cv2, numpy, fastapi, uvicorn, pandas, openpyxl, multipart, tifffile, h5py, skimage, scipy; import ncempy.io" >nul 2>&1
if errorlevel 1 (
    echo.
    echo [CRITICAL ERROR] The RodSizer environment is incomplete or damaged.
    echo Please run RodSizer_DEBUG_Windows.bat to repair it.
    pause
    exit /b 1
)

echo Activating environment...

REM --- 6. Open Browser ---
echo.
echo Starting Server on PORT 8501...
echo The browser will open in 10 seconds...
echo.
echo IF PAGE FAILS TO LOAD:
echo 1. Keep this window open.
echo 2. Open Chrome/Edge and go to: %APP_URL%
echo.

start "" cmd /c "timeout /t 10 >nul & start %APP_URL%"

REM --- 7. Run Server ---
echo Close this window to stop the app.
cd /d "%BACKEND_DIR%"
"%VENV_PYTHON%" -m uvicorn main:app --reload --host 127.0.0.1 --port 8501

pause
