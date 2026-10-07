@echo off
cd /d "%~dp0"
setlocal enabledelayedexpansion

set "PROJECT_DIR=%CD%"
set "BACKEND_DIR=%PROJECT_DIR%\backend"
set "VENV_DIR=%PROJECT_DIR%\.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"
set "APP_URL=http://127.0.0.1:8501"
set "TEMP_CHECK=%PROJECT_DIR%\_check_rodsizer_imports.py"
set "TEMP_REQ=%PROJECT_DIR%\_requirements_windows_debug.txt"
set "FRESH_VENV=0"
set "REPAIR_ATTEMPTED=0"

echo ==================================================
echo       RodSizer - DEBUG & REPAIR
echo ==================================================
echo.

set "PY_EXE="

REM --- 1. Find Python ---
python --version >nul 2>&1
if %errorlevel% equ 0 set "PY_EXE=python"
if defined PY_EXE goto FOUND_PYTHON

py --version >nul 2>&1
if !errorlevel! equ 0 set "PY_EXE=py"
if defined PY_EXE goto FOUND_PYTHON

REM Auto-Detect
if exist "C:\Python39\python.exe" set "PY_EXE=C:\Python39\python.exe" & goto FOUND_PYTHON
if exist "C:\Python310\python.exe" set "PY_EXE=C:\Python310\python.exe" & goto FOUND_PYTHON
if exist "C:\Python311\python.exe" set "PY_EXE=C:\Python311\python.exe" & goto FOUND_PYTHON
if exist "C:\Python312\python.exe" set "PY_EXE=C:\Python312\python.exe" & goto FOUND_PYTHON
if exist "C:\Program Files\Python39\python.exe" set "PY_EXE=C:\Program Files\Python39\python.exe" & goto FOUND_PYTHON
if exist "C:\Program Files\Python310\python.exe" set "PY_EXE=C:\Program Files\Python310\python.exe" & goto FOUND_PYTHON
if exist "C:\Program Files\Python311\python.exe" set "PY_EXE=C:\Program Files\Python311\python.exe" & goto FOUND_PYTHON
if exist "C:\Program Files\Python312\python.exe" set "PY_EXE=C:\Program Files\Python312\python.exe" & goto FOUND_PYTHON

:ASK_USER
echo.
echo [ERROR] Could not find Python automatically.
echo.
echo Please locate your 'python.exe' file.
echo IMPORTANT: Drag the FILE, not a shortcut.
set /p "PY_EXE=Drag and Drop python.exe here: "
set "PY_EXE=!PY_EXE:"=!"

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

REM --- 2. Setup Env ---
if not exist "%VENV_PYTHON%" (
    echo [INFO] Creating venv...
    "!PY_EXE!" -m venv "%VENV_DIR%"
    if errorlevel 1 (
        echo.
        echo [CRITICAL ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    set "FRESH_VENV=1"
)

if "%FRESH_VENV%"=="1" (
    echo [INFO] Installing baseline dependencies...
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
        echo [WARN] Baseline install reported an error.
        echo [WARN] Continuing into repair checks...
    )
)

:CHECK_LIBS
echo.
echo [TEST] Checking libraries...

REM Create a temporary python script to avoid batch syntax errors
echo try: > "%TEMP_CHECK%"
echo     import cv2 >> "%TEMP_CHECK%"
echo     print('  - OpenCV: OK') >> "%TEMP_CHECK%"
echo     import numpy >> "%TEMP_CHECK%"
echo     print('  - NumPy: OK') >> "%TEMP_CHECK%"
echo     import fastapi >> "%TEMP_CHECK%"
echo     print('  - FastAPI: OK') >> "%TEMP_CHECK%"
echo     import uvicorn >> "%TEMP_CHECK%"
echo     print('  - Uvicorn: OK') >> "%TEMP_CHECK%"
echo     import pandas >> "%TEMP_CHECK%"
echo     print('  - Pandas: OK') >> "%TEMP_CHECK%"
echo     import ncempy.io >> "%TEMP_CHECK%"
echo     print('  - ncempy: OK') >> "%TEMP_CHECK%"
echo     import openpyxl >> "%TEMP_CHECK%"
echo     print('  - openpyxl: OK') >> "%TEMP_CHECK%"
echo     import multipart >> "%TEMP_CHECK%"
echo     print('  - python-multipart: OK') >> "%TEMP_CHECK%"
echo     import tifffile >> "%TEMP_CHECK%"
echo     print('  - tifffile: OK') >> "%TEMP_CHECK%"
echo     import h5py >> "%TEMP_CHECK%"
echo     print('  - h5py: OK') >> "%TEMP_CHECK%"
echo     from skimage import measure >> "%TEMP_CHECK%"
echo     print('  - scikit-image: OK') >> "%TEMP_CHECK%"
echo     from scipy import ndimage >> "%TEMP_CHECK%"
echo     print('  - SciPy: OK') >> "%TEMP_CHECK%"
echo except Exception as e: >> "%TEMP_CHECK%"
echo     print('\n  [ERROR] ' + str(e)) >> "%TEMP_CHECK%"
echo     raise SystemExit(1) >> "%TEMP_CHECK%"

REM Run the check
"%VENV_PYTHON%" "%TEMP_CHECK%"
set BUILD_STATUS=%errorlevel%

REM Clean up
if exist "%TEMP_CHECK%" del "%TEMP_CHECK%"

if %BUILD_STATUS% neq 0 (
    if "%REPAIR_ATTEMPTED%"=="1" (
        echo.
        echo [CRITICAL ERROR] Repair was attempted, but the environment is still failing checks.
        echo Review the error shown above for the package that still cannot import.
        pause
        exit /b 1
    )

    echo.
    echo [CRITICAL ERROR] Libraries failed to load.
    echo.
    echo ==================================================
    echo                  ATTEMPTING REPAIR
    echo ==================================================
    echo We will switch to Standard OpenCV to fix DLL errors.
    echo.
    pause
    
    echo [REPAIR] upgrading pip...
    "%VENV_PYTHON%" -m pip install --upgrade pip
    if errorlevel 1 (
        echo.
        echo [CRITICAL ERROR] Failed to upgrade pip.
        pause
        exit /b 1
    )
    
    echo [REPAIR] Swapping OpenCV versions...
    "%VENV_PYTHON%" -m pip uninstall -y opencv-python-headless opencv-python
    if exist "%TEMP_REQ%" del "%TEMP_REQ%"
    findstr /V /I "opencv-python-headless" "%BACKEND_DIR%\requirements.txt" > "%TEMP_REQ%"
    if errorlevel 1 (
        echo.
        echo [CRITICAL ERROR] Failed to build a temporary requirements list.
        pause
        exit /b 1
    )
    
    echo [REPAIR] Installing non-OpenCV dependencies...
    "%VENV_PYTHON%" -m pip install -r "%TEMP_REQ%"
    if errorlevel 1 (
        if exist "%TEMP_REQ%" del "%TEMP_REQ%"
        echo.
        echo [CRITICAL ERROR] Failed while reinstalling dependencies.
        pause
        exit /b 1
    )

    echo [REPAIR] Installing standard OpenCV...
    "%VENV_PYTHON%" -m pip install opencv-python
    if errorlevel 1 (
        if exist "%TEMP_REQ%" del "%TEMP_REQ%"
        echo.
        echo [CRITICAL ERROR] Failed to install standard OpenCV.
        pause
        exit /b 1
    )

    if exist "%TEMP_REQ%" del "%TEMP_REQ%"
    
    set "REPAIR_ATTEMPTED=1"

    echo.
    echo [REPAIR COMPLETE]
    echo Checking again...
    echo.
    goto CHECK_LIBS
) else (
    echo [SUCCESS] Libraries loaded correctly.
)

REM --- 3. Run Server ---
echo.
echo [INFO] Starting Server...
echo [INSTRUCTION] Open browser to: %APP_URL%
echo.

cd /d "%BACKEND_DIR%"
"%VENV_PYTHON%" -m uvicorn main:app --reload --host 127.0.0.1 --port 8501

pause
