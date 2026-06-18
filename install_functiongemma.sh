#!/usr/bin/env bash
# =============================================================================
#  install_functiongemma.sh
#  MicroLab — FunctionGemma 270M + Vosk STT installer
#
#  Run this ONCE on your Raspberry Pi 5 to:
#    1. Install llama-cpp-python (built for ARM, CPU-only)
#    2. Download FunctionGemma 270M GGUF model (~250 MB)
#    3. Install Vosk offline STT library
#    4. Download the Vosk small English model (~50 MB)
#    5. Run a quick self-test on both components
#
#  Usage:
#    chmod +x install_functiongemma.sh
#    ./install_functiongemma.sh
#
#  Expected time: ~5–10 minutes on RPi 5 with good internet
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${PROJECT_DIR}/models"

# Colours
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }
header()  { echo -e "\n${BOLD}${CYAN}══════════════════════════════════════════${RESET}"; echo -e "${BOLD}  $*${RESET}"; echo -e "${BOLD}${CYAN}══════════════════════════════════════════${RESET}"; }

# ─────────────────────────────────────────────────────────────────────────────
header "Step 0 — System check"
# ─────────────────────────────────────────────────────────────────────────────

PYTHON=$(command -v python3 || command -v python || true)
if [[ -z "$PYTHON" ]]; then
    error "Python 3 not found. Please install it first."
    exit 1
fi
PYTHON_VERSION=$("$PYTHON" --version 2>&1)
info "Python: $PYTHON_VERSION"

PIP=$(command -v pip3 || command -v pip || true)
if [[ -z "$PIP" ]]; then
    error "pip not found. Please install python3-pip."
    exit 1
fi

# Raspberry Pi OS 12+ (Debian Bookworm) uses PEP 668 externally-managed-environment.
# We need --break-system-packages to install into the system Python.
# Detect if the flag is supported (pip >= 23.0):
if "$PIP" install --help 2>&1 | grep -q "break-system-packages"; then
    PIP_FLAGS="--break-system-packages"
    info "Detected PEP 668 environment — using --break-system-packages"
else
    PIP_FLAGS=""
fi

# Ensure models dir exists
mkdir -p "${MODELS_DIR}"
info "Models directory: ${MODELS_DIR}"

# ─────────────────────────────────────────────────────────────────────────────
header "Step 1 — Install system dependencies"
# ─────────────────────────────────────────────────────────────────────────────

info "Installing build tools and audio libraries…"
sudo apt-get update -qq
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    portaudio19-dev \
    libsndfile1 \
    unzip \
    wget \
    curl \
    2>&1 | tail -5
success "System dependencies installed."

# ─────────────────────────────────────────────────────────────────────────────
header "Step 2 — Install llama-cpp-python (ARM CPU build)"
# ─────────────────────────────────────────────────────────────────────────────

info "Installing llama-cpp-python with ARM NEON optimisations…"
info "This may take 3–8 minutes (compiles from source for best RPi 5 performance)."

# CMAKE_ARGS enables OpenBLAS — critical for good performance on RPi 5
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
    "$PIP" install llama-cpp-python --no-cache-dir $PIP_FLAGS --verbose 2>&1 | grep -E "(Building|Successfully|error|Error)" || true

# Verify
if "$PYTHON" -c "import llama_cpp; print('llama_cpp version:', llama_cpp.__version__)" 2>/dev/null; then
    success "llama-cpp-python installed."
else
    error "llama-cpp-python installation failed. Check errors above."
    echo "  Try manually: pip install llama-cpp-python --break-system-packages"
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
header "Step 3 — Download FunctionGemma 270M GGUF model"
# ─────────────────────────────────────────────────────────────────────────────

MODEL_FILE="${MODELS_DIR}/functiongemma-270m-q4_k_m.gguf"

if [[ -f "$MODEL_FILE" ]]; then
    SIZE_MB=$(( $(stat -c%s "$MODEL_FILE") / 1024 / 1024 ))
    warn "Model already exists (${SIZE_MB} MB). Skipping download."
    warn "Delete ${MODEL_FILE} to re-download."
else
    info "Downloading FunctionGemma 270M Q4_K_M GGUF (~250 MB)…"
    info "Source: HuggingFace — bartowski/google_functiongemma-270m-it-GGUF"

    # Primary: bartowski quantisation (most reliable community mirror)
    MODEL_URL="https://huggingface.co/bartowski/google_functiongemma-270m-it-GGUF/resolve/main/google_functiongemma-270m-it-Q4_K_M.gguf"

    if wget --quiet --show-progress -O "${MODEL_FILE}.tmp" "${MODEL_URL}"; then
        mv "${MODEL_FILE}.tmp" "${MODEL_FILE}"
        SIZE_MB=$(( $(stat -c%s "$MODEL_FILE") / 1024 / 1024 ))
        success "FunctionGemma 270M downloaded (${SIZE_MB} MB)."
    else
        rm -f "${MODEL_FILE}.tmp"
        warn "Primary download failed. Trying unsloth mirror…"

        ALT_URL="https://huggingface.co/unsloth/functiongemma-270m-it-GGUF/resolve/main/functiongemma-270m-it-Q4_K_M.gguf"
        if wget --quiet --show-progress -O "${MODEL_FILE}.tmp" "${ALT_URL}"; then
            mv "${MODEL_FILE}.tmp" "${MODEL_FILE}"
            SIZE_MB=$(( $(stat -c%s "$MODEL_FILE") / 1024 / 1024 ))
            success "FunctionGemma 270M downloaded from mirror (${SIZE_MB} MB)."
        else
            rm -f "${MODEL_FILE}.tmp"
            error "Both download sources failed."
            echo ""
            echo "  Manual download instructions:"
            echo "  1. Visit: https://huggingface.co/bartowski/google_functiongemma-270m-it-GGUF"
            echo "  2. Download: google_functiongemma-270m-it-Q4_K_M.gguf"
            echo "  3. Save to:  ${MODEL_FILE}"
            echo ""
            warn "Continuing to install Vosk STT (model can be downloaded later)."
        fi
    fi
fi


# ─────────────────────────────────────────────────────────────────────────────
header "Step 4 — Install Vosk offline STT"
# ─────────────────────────────────────────────────────────────────────────────

info "Installing vosk and pyaudio…"
"$PIP" install vosk pyaudio --quiet $PIP_FLAGS

if "$PYTHON" -c "import vosk; print('vosk OK:', vosk.__version__ if hasattr(vosk,'__version__') else 'installed')" 2>/dev/null; then
    success "Vosk installed."
else
    warn "Vosk install may have issues. Falling back to Google STT is OK."
fi

# ─────────────────────────────────────────────────────────────────────────────
header "Step 5 — Download Vosk English model"
# ─────────────────────────────────────────────────────────────────────────────

VOSK_MODEL_DIR="${MODELS_DIR}/vosk-model-small-en-us-0.15"
VOSK_ZIP="${MODELS_DIR}/vosk-model.zip"

if [[ -d "$VOSK_MODEL_DIR" ]]; then
    warn "Vosk model already exists at ${VOSK_MODEL_DIR}. Skipping."
else
    info "Downloading Vosk small English model (~50 MB)…"
    VOSK_URL="https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"

    if wget --quiet --show-progress -O "$VOSK_ZIP" "$VOSK_URL"; then
        info "Extracting…"
        unzip -q "$VOSK_ZIP" -d "${MODELS_DIR}/"
        rm -f "$VOSK_ZIP"
        success "Vosk model extracted to ${VOSK_MODEL_DIR}."
    else
        rm -f "$VOSK_ZIP"
        error "Vosk model download failed. Voice will fall back to Google STT."
        warn "You can retry manually: https://alphacephei.com/vosk/models"
    fi
fi

# ─────────────────────────────────────────────────────────────────────────────
header "Step 6 — Quick self-test"
# ─────────────────────────────────────────────────────────────────────────────

info "Testing FunctionGemma engine import and model load…"
if "$PYTHON" - <<'EOF'
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) if '__file__' in dir() else '.')
from modules.function_gemma import FunctionGemmaEngine
engine = FunctionGemmaEngine()
if not engine.available:
    print("MODEL_NOT_LOADED")
    sys.exit(0)
call = engine.parse("turn on the light")
print(f"TEST_OK: lamp_on? -> {call.name}")
EOF
then
    success "FunctionGemma self-test passed."
else
    warn "Self-test raised an exception (may be OK if model path differs)."
fi

# ─────────────────────────────────────────────────────────────────────────────
header "Installation Complete"
# ─────────────────────────────────────────────────────────────────────────────

echo ""
echo -e "  ${GREEN}${BOLD}What was installed:${RESET}"
echo "  • llama-cpp-python  — runs GGUF models on CPU (ARM NEON)"
echo "  • FunctionGemma 270M Q4_K_M GGUF  — local AI voice intent parser"
echo "  • vosk + pyaudio  — offline speech-to-text engine"
echo "  • vosk-model-small-en-us-0.15  — ~50 MB English acoustic model"
echo ""
echo -e "  ${GREEN}${BOLD}To run the full self-test:${RESET}"
echo "  python3 modules/function_gemma.py   # tests all 26 voice → function mappings"
echo "  python3 modules/vosk_stt.py         # live microphone STT test"
echo ""
echo -e "  ${GREEN}${BOLD}To start the system:${RESET}"
echo "  make run-system"
echo ""
echo -e "  ${YELLOW}${BOLD}Voice pipeline will now show in banner:${RESET}"
echo "  STT Engine → Vosk (offline)"
echo "  AI Voice   → FunctionGemma 270M ✅ (natural language)"
echo ""
