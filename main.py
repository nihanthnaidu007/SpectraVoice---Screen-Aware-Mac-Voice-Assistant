#!/usr/bin/env python3
"""
SpectraVoice – Screen-Aware Mac Voice Assistant
===============================================
A powerful screen-aware voice assistant for macOS that can see your screen,
control your computer, and help with tasks.

This file is a thin launcher: the CLI lives in `assistant_app.cli` (so the
installed `spectravoice` console script shares the same entry point).

Usage:
    python main.py                          # Shows GUI pop-up for LLM selection
    python main.py --local                  # Use local Ollama LLM
    python main.py --local --model llava    # Use specific Ollama model
    python main.py --cloud                  # Explicitly use Cloud (OpenAI)
    python main.py --interactive            # Terminal-based provider selection
    python main.py --gui                    # GUI mode with status window
    python main.py --minimal                # Minimal mode (fastest)
    python main.py --debug                  # Enable debug logging
    python main.py --doctor                 # Check permissions, keys, config
    python main.py --voice nova             # Use Nova voice for TTS
    python main.py --whisper-model small    # Use small Whisper model

LLM Provider Selection:
    By default, a GUI pop-up appears for selecting Cloud or Local LLM.
    Set LLM_PROVIDER=cloud or LLM_PROVIDER=local to skip the GUI.
"""

import pathlib
import sys

# Ensure src/ is importable when running `python main.py` from anywhere
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))

from assistant_app.cli import get_llm_config, main

# Backward-compatible re-export: W1-era tests and scripts call
# main.get_llm_config(args, cfg); the implementation lives in assistant_app.cli.
__all__ = ["get_llm_config", "main"]

if __name__ == "__main__":
    sys.exit(main())
