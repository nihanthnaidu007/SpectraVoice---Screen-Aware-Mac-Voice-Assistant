"""
Audio I/O Package
=================
Audio input/output modules for voice detection and text-to-speech.

Import-light by design: PyAudio/soundfile/Whisper-dependent modules are
re-exported lazily via PEP 562 `__getattr__`, so importing this package on
Linux CI does not require audio dependencies or macOS frameworks.
"""

import importlib

_LAZY_EXPORTS = {
    "SmartVoiceDetector": "assistant_app.io.audio.voice_detector",
    "TextToSpeech": "assistant_app.io.audio.tts",
}

__all__ = ["SmartVoiceDetector", "TextToSpeech"]


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache for subsequent lookups
    return value
