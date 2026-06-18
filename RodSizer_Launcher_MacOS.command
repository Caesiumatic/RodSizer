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

# =============================================================================
# STEP 2 — Python 3.11
#
#  TensorFlow 2.x officially supports Python 3.8–3.11.
#  Python 3.12+ does NOT have stable TensorFlow wheels yet.
#  We therefore require Python 3.11 specifically.
# =============================================================================
step "Step 2/4: Checking Python 3.11"

PYTHON_CMD=""

# Helper: find python3.11 by checking all known locations
find_python311() {
    # 1. Homebrew-managed path (most reliable — works regardless of PATH)
    local brew_prefix
    brew_prefix="$(brew --prefix python@3.11 2>/dev/null)"
    if [ -n "$brew_prefix" ] && [ -x "$brew_prefix/bin/python3.11" ]; then
        echo "$brew_prefix/bin/python3.11"; return
    fi

    # 2. Direct binary in PATH
    if command -v python3.11 &>/dev/null; then
        command -v python3.11; return
    fi

    # 3. Common Homebrew Cellar fallback (Apple Silicon)
    local cellar_py
    cellar_py=$(ls /opt/homebrew/Cellar/python@3.11/*/bin/python3.11 2>/dev/null | sort -V | tail -1)
    if [ -x "$cellar_py" ]; then
        echo "$cellar_py"; return
    fi
    # Intel
    cellar_py=$(ls /usr/local/Cellar/python@3.11/*/bin/python3.11 2>/dev/null | sort -V | tail -1)
    if [ -x "$cellar_py" ]; then
        echo "$cellar_py"; return
    fi

    # 4. pyenv
    if command -v pyenv &>/dev/null; then
        local pyenv_ver
        pyenv_ver=$(pyenv versions --bare 2>/dev/null | grep "^3\.11\." | sort -V | tail -1)
        if [ -n "$pyenv_ver" ]; then
            local pyenv_py="$(pyenv root)/versions/$pyenv_ver/bin/python3.11"
            [ -x "$pyenv_py" ] && echo "$pyenv_py"; return
        fi
    fi
}

PYTHON_CMD="$(find_python311)"

# If still not found, install via Homebrew then retry
if [ -z "$PYTHON_CMD" ]; then
    warn "Python 3.11 not found. Installing via Homebrew (needed for TensorFlow)..."
    brew install python@3.11
    PYTHON_CMD="$(find_python311)"
fi

if [ -z "$PYTHON_CMD" ] || [ ! -x "$PYTHON_CMD" ]; then
    error "Could not find or install Python 3.11."
    error "Please install it manually:  brew install python@3.11"
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

for name in ("fastapi", "uvicorn", "tensorflow", "numpy", "cv2", "stardist"):
    importlib.import_module(name)
PY
}

install_tensorflow_with_fallback() {
    local tf_spec="tensorflow==2.21.0"

    info "Installing TensorFlow..."
    if "$VENV_PYTHON" -m pip install "$tf_spec"; then
        return 0
    fi

    warn "pip could not resolve tensorflow from the package index."
    warn "Retrying TensorFlow install directly from PyPI..."
    PIP_CONFIG_FILE=/dev/null "$VENV_PYTHON" -m pip install --index-url https://pypi.org/simple "$tf_spec" && return 0

    return 1
}

# Check whether a first-time install is needed
if ! needs_python_packages; then
    info "Installing Python packages — this may take 5–10 minutes on first run."
    info "Please do NOT close this window."

    # Upgrade pip / setuptools silently
    "$VENV_PYTHON" -m pip install --upgrade pip setuptools wheel --quiet

    # ── TensorFlow first (sets numpy version constraint everything else must follow) ──
    #
    #  tensorflow-macos was deprecated at 2.13 and forces numpy<2, which conflicts
    #  with opencv-python-headless >=4.9 and ncempy >=1.15 (both require numpy>=2).
    #
    #  tensorflow >= 2.16 ships a universal wheel that runs natively on Apple
    #  Silicon (M-series) without a separate macos fork, and is compatible with
    #  numpy 2.x.  tensorflow-metal is still the GPU-acceleration plugin for M-chips.
    if ! install_tensorflow_with_fallback; then
        error "TensorFlow installation failed."
        error "This is usually a package-index or network issue, not a RodSizer code issue."
        if [ "$IS_APPLE_SILICON" = true ]; then
            error "If needed, try this manually inside backend/.venv:"
            error "  python -m pip install --index-url https://pypi.org/simple tensorflow==2.21.0"
        else
            error "Try rerunning later, or install TensorFlow manually inside backend/.venv."
        fi
        echo "Press Enter to close..."; read -r; exit 1
    fi

    # Note: tensorflow-metal is NOT installed because v1.2.0 is incompatible
    # with tensorflow >=2.16 (dlopen fails on _pywrap_tensorflow_internal.so).
    # Modern TensorFlow already runs natively on Apple Silicon without it.

    if ! "$VENV_PYTHON" -c "import tensorflow" &>/dev/null 2>&1; then
        error "TensorFlow import failed after installation."
        error "Try deleting backend/.venv and relaunching, or check network connectivity."
        echo "Press Enter to close..."; read -r; exit 1
    fi

    # ── Remaining packages (numpy version is now fixed by TF above) ─────────────
    TMP_REQ=$(mktemp /tmp/rodsizer_req_XXXX.txt)
    grep -v "^tensorflow" "$BACKEND_DIR/requirements.txt" > "$TMP_REQ"

    if ! "$VENV_PYTHON" -m pip install -r "$TMP_REQ" --quiet; then
        error "Failed to install some packages. Check the output above."
        rm -f "$TMP_REQ"
        echo "Press Enter to close..."; read -r; exit 1
    fi
    rm -f "$TMP_REQ"

    # ── Verify tensorflow is importable ───────────────────────────────────────
    if ! "$VENV_PYTHON" -c "import tensorflow" &>/dev/null 2>&1; then
        error "TensorFlow installation failed."
        error "Try running this manually inside the venv:"
        if [ "$IS_APPLE_SILICON" = true ]; then
            error "  python -m pip install --index-url https://pypi.org/simple tensorflow==2.21.0"
        else
            error "  python -m pip install tensorflow==2.21.0"
        fi
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
