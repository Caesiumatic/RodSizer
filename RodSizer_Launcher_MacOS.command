#!/bin/bash
# =============================================================================
#  RodSizer — Mac Launcher
#  Double-click this file in Finder to start the app.
# =============================================================================

# ── Navigate to the project root (same folder as this script) ────────────────
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"
BACKEND_DIR="$PROJECT_DIR/backend"
VENV_DIR="$BACKEND_DIR/.venv"
CACHE_DIR="$BACKEND_DIR/.cache"
MPL_CACHE_DIR="$CACHE_DIR/matplotlib"

# ── Terminal colours ──────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${GREEN}[✓]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✗]${NC} $*"; }
step()  { echo -e "\n${CYAN}──────────────────────────────────────────${NC}"; echo -e "${CYAN}  $*${NC}"; echo -e "${CYAN}──────────────────────────────────────────${NC}"; }

step "RodSizer — Starting up"

# ── Detect CPU architecture ───────────────────────────────────────────────────
ARCH=$(uname -m)
if [ "$ARCH" = "arm64" ]; then
    IS_APPLE_SILICON=true
    info "Apple Silicon (M-series) Mac detected"
else
    IS_APPLE_SILICON=false
    info "Intel Mac detected"
fi

# =============================================================================
# STEP 1 — Homebrew
# =============================================================================
step "Step 1/5: Checking Homebrew"

# Ensure brew is on PATH (Apple Silicon installs to /opt/homebrew)
if [ "$IS_APPLE_SILICON" = true ] && [ -f /opt/homebrew/bin/brew ]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
elif [ -f /usr/local/bin/brew ]; then
    eval "$(/usr/local/bin/brew shellenv)"
fi

if ! command -v brew &>/dev/null; then
    warn "Homebrew not found. Installing now (this requires an internet connection)..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

    # Re-source brew after install
    if [ "$IS_APPLE_SILICON" = true ] && [ -f /opt/homebrew/bin/brew ]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [ -f /usr/local/bin/brew ]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
fi

if ! command -v brew &>/dev/null; then
    error "Homebrew installation failed."
    error "Please install it manually from https://brew.sh, then re-run this launcher."
    echo "Press Enter to close..."; read -r; exit 1
fi
info "Homebrew: $(brew --version | head -1)"

# Tesseract OCR — used to auto-read the pixel size / scale bar burned into
# camera-export images (JPG/PNG/TIF without embedded calibration). Optional:
# RodSizer still runs without it, falling back to manual calibration.
if ! command -v tesseract &>/dev/null; then
    warn "Tesseract OCR not found. Installing for automatic scale-bar reading..."
    brew install tesseract || warn "Tesseract install failed — auto-calibration will be disabled; manual calibration still works."
fi
if command -v tesseract &>/dev/null; then
    info "Tesseract: $(tesseract --version 2>&1 | head -1)"
fi

# =============================================================================
# STEP 2 — Python 3.10-3.12
#  (the pinned dependency set in backend/requirements.txt targets these
#   versions; 3.13+ lacks wheels for the pinned numpy)
# =============================================================================
step "Step 2/4: Checking Python"

PYTHON_CMD=""

# Helper: find a suitable Python (3.10-3.12), preferring newer versions
find_python() {
    local candidate
    for candidate in python3.12 python3.11 python3.10 python3; do
        if command -v "$candidate" &>/dev/null; then
            if "$candidate" -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 12) else 1)' &>/dev/null; then
                command -v "$candidate"; return
            fi
        fi
    done

    # Homebrew-managed fallback (works regardless of PATH)
    local brew_prefix ver
    for ver in 3.12 3.11 3.10; do
        brew_prefix="$(brew --prefix python@$ver 2>/dev/null)"
        if [ -n "$brew_prefix" ] && [ -x "$brew_prefix/bin/python$ver" ]; then
            echo "$brew_prefix/bin/python$ver"; return
        fi
    done
}

PYTHON_CMD="$(find_python)"

# If still not found, install via Homebrew then retry
if [ -z "$PYTHON_CMD" ]; then
    warn "Python 3.10-3.12 not found. Installing 3.12 via Homebrew..."
    brew install python@3.12
    PYTHON_CMD="$(find_python)"
fi

if [ -z "$PYTHON_CMD" ] || [ ! -x "$PYTHON_CMD" ]; then
    error "Could not find or install Python."
    error "Please install it manually:  brew install python@3.12"
    echo "Press Enter to close..."; read -r; exit 1
fi

PYTHON_VER=$("$PYTHON_CMD" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
info "Using Python $PYTHON_VER at: $PYTHON_CMD"

# =============================================================================
# STEP 3 — Virtual environment & Python dependencies
# =============================================================================
step "Step 3/4: Setting up Python environment"

# Create venv if it does not exist
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"
info "Virtual environment activated"
VENV_PYTHON="$VENV_DIR/bin/python"

if [ ! -x "$VENV_PYTHON" ]; then
    error "Virtual environment Python is missing at: $VENV_PYTHON"
    error "Try deleting backend/.venv and relaunching RodSizer."
    echo "Press Enter to close..."; read -r; exit 1
fi

# Ensure Python/matplotlib/fontconfig can cache into writable project-local paths.
mkdir -p "$MPL_CACHE_DIR"
export XDG_CACHE_HOME="$CACHE_DIR"
export MPLCONFIGDIR="$MPL_CACHE_DIR"

needs_python_packages() {
    "$VENV_PYTHON" - <<'PY' &>/dev/null
import importlib

for name in ("fastapi", "uvicorn", "numpy", "cv2", "skimage", "tifffile", "h5py"):
    importlib.import_module(name)
PY
}

# Check whether a first-time install is needed
if ! needs_python_packages; then
    info "Installing Python packages — this may take 2–3 minutes on first run."
    info "Please do NOT close this window."

    # Upgrade pip / setuptools silently
    "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel --quiet

    if ! "$VENV_PYTHON" -m pip install -r "$BACKEND_DIR/requirements.txt" --quiet; then
        error "Failed to install some packages. Check the output above."
        echo "Press Enter to close..."; read -r; exit 1
    fi

    info "All packages installed successfully."
else
    info "Python packages already installed — skipping."
fi

# =============================================================================
# STEP 4 — Launch server & open browser
# =============================================================================
step "Step 4/4: Starting server"

# Kill any leftover process on port 8000
if lsof -ti:8000 &>/dev/null; then
    warn "Port 8000 is already in use — stopping the existing process..."
    lsof -ti:8000 | xargs kill -9 2>/dev/null
    sleep 1
fi

# Start uvicorn (log goes to backend/server.log)
cd "$BACKEND_DIR"
"$VENV_PYTHON" -m uvicorn main:app --host 127.0.0.1 --port 8000 > server.log 2>&1 &
SERVER_PID=$!

# Wait until the server responds (allow extra time on first run while caches build)
echo -n "  Waiting for server to be ready"
MAX_WAIT=90
for ((i=1; i<=MAX_WAIT; i++)); do
    if curl -s http://127.0.0.1:8000 &>/dev/null; then
        break
    fi
    echo -n "."
    sleep 1
    if [ $i -eq $MAX_WAIT ]; then
        echo ""
        error "Server did not start within ${MAX_WAIT} seconds."
        error "Check $BACKEND_DIR/server.log for details."
        kill "$SERVER_PID" 2>/dev/null
        echo "Press Enter to close..."; read -r; exit 1
    fi
done
echo ""

info "Server is running at http://127.0.0.1:8000"
open http://127.0.0.1:8000

echo ""
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo -e "${GREEN}  RodSizer is ready!${NC}"
echo -e "${GREEN}  URL: http://127.0.0.1:8000${NC}"
echo -e "${GREEN}════════════════════════════════════════════${NC}"
echo ""
echo "  Press Ctrl+C or close this window to stop the server."
echo ""

# Keep the terminal open (closing it stops the server)
wait "$SERVER_PID"
