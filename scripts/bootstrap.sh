#!/usr/bin/env bash
# SpectraVoice bootstrap — step 2 of the 3-step first run.
#
#   1. git clone https://github.com/nihanthnaidu007/SpectraVoice---Screen-Aware-Mac-Voice-Assistant.git
#      cd SpectraVoice---Screen-Aware-Mac-Voice-Assistant
#   2. ./scripts/bootstrap.sh          <-- this script
#   3. python main.py                  (or: spectravoice)
#
# What it does: verifies Python, creates .venv, installs system audio deps,
# installs the package (pip install -e .), seeds .env, and runs `--doctor`.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"

echo "🩺 SpectraVoice bootstrap"
echo "========================="

# 1. Python version check (>= 3.10 required)
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "❌ $PYTHON_BIN not found. Install Python 3.10+ and re-run."
    exit 1
fi
PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if ! "$PYTHON_BIN" - <<'EOF'
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF
then
    echo "❌ Python 3.10+ required, found $PY_VERSION. Install a newer Python and re-run with PYTHON_BIN=/path/to/python."
    exit 1
fi
echo "✅ Python $PY_VERSION"

# 2. Virtual environment
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment (.venv)..."
    "$PYTHON_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip >/dev/null
echo "✅ Virtual environment ready"

# 3. System audio dependencies (PortAudio, FFmpeg)
# Containers and CI images often cannot install system packages; set
# SPECTRAVOICE_SKIP_SYSTEM_DEPS=1 to defer to the --doctor run at the end.
if [ "${SPECTRAVOICE_SKIP_SYSTEM_DEPS:-0}" = "1" ]; then
    echo "⏭️  SPECTRAVOICE_SKIP_SYSTEM_DEPS=1 — skipping system packages."
    echo "    The --doctor run at the end reports anything still missing."
else
    echo "🔊 Checking system audio dependencies..."
    if command -v brew >/dev/null 2>&1; then
        for formula in portaudio ffmpeg; do
            if brew list --formula "$formula" >/dev/null 2>&1; then
                echo "✅ $formula already installed"
            else
                echo "📦 Installing $formula via Homebrew..."
                brew install "$formula"
            fi
        done
    elif command -v apt-get >/dev/null 2>&1; then
        if dpkg -s portaudio19-dev >/dev/null 2>&1; then
            echo "✅ portaudio19-dev already installed"
        elif sudo -n true 2>/dev/null; then
            echo "📦 Installing portaudio19-dev via apt..."
            sudo apt-get update -qq && sudo apt-get install -y -qq portaudio19-dev python3-dev ffmpeg
        else
            echo "⚠️  Could not install system packages (no passwordless sudo). Run:"
            echo "    sudo apt-get install -y portaudio19-dev python3-dev ffmpeg"
            echo "   then re-run this script (or set SPECTRAVOICE_SKIP_SYSTEM_DEPS=1)."
            exit 1
        fi
    else
        echo "⚠️  Neither brew nor apt-get found. Install PortAudio and FFmpeg manually, then re-run."
        exit 1
    fi
fi

# 4. Python dependencies (single source of truth: requirements.txt via pyproject)
echo "🐍 Installing Python dependencies (this can take a while — torch is large)..."
pip install -e .

# 5. Seed .env from the example if it does not exist
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✍️  Created .env from .env.example — add your OPENAI_API_KEY (and optionally TAVILY_API_KEY)."
else
    echo "✅ .env already exists"
fi

# 6. Run the doctor (best-effort — missing macOS permissions are reported, not fatal here)
echo ""
echo "🩺 Running doctor checks..."
python main.py --doctor || true
echo "If any permission shows ❌, fix it (the doctor printed the System Settings path)"
echo "and re-run: python main.py --doctor"
echo ""
echo "🎉 Bootstrap complete. Next step:  python main.py"
