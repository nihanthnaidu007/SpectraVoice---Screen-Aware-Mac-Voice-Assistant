"""
I/O Package
===========
Input/Output modules for audio and vision processing.

Import-light by design: audio (soundfile, Whisper) and vision (OpenCV)
modules are re-exported lazily via PEP 562 `__getattr__`, so importing
platform-independent parts of the package — as the unit tests do on
Linux CI — does not require those heavy dependencies or macOS frameworks.
"""

import importlib

_LAZY_EXPORTS = {
    "ScreenCapture": "assistant_app.io.vision.screen_capture",
    "SmartVoiceDetector": "assistant_app.io.audio.voice_detector",
    "TextToSpeech": "assistant_app.io.audio.tts",
}

__all__ = ["ScreenCapture", "SmartVoiceDetector", "TextToSpeech"]


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache for subsequent lookups
    return value
