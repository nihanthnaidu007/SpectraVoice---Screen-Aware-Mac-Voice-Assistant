"""
SpectraVoice – Modular Services Package
=======================================
Clean, modular architecture for the SpectraVoice screen-aware assistant.

Platform-independent modules (core, tools, llm, utils) are safe to import
anywhere. Heavy platform modules (audio, vision, macOS integration) are
re-exported lazily via PEP 562 `__getattr__`, so this package stays importable
— and unit-testable on Linux CI — without macOS-only dependencies installed.
"""

import importlib

from .utils.logging_config import get_logger, setup_logging

__version__ = "1.2.0"

_LAZY_EXPORTS = {
    "ScreenCapture": "assistant_app.io.vision.screen_capture",
    "SmartVoiceDetector": "assistant_app.io.audio.voice_detector",
    "ConversationMemory": "assistant_app.core.conversation",
    "Assistant": "assistant_app.core.assistant",
    "TextToSpeech": "assistant_app.io.audio.tts",
    "WebSearch": "assistant_app.tools.web_search",
    "SearchResult": "assistant_app.tools.web_search",
    "SearchCategory": "assistant_app.tools.web_search",
    "SpectraVoiceAssistant": "assistant_app.services.spectravoice_assistant",
    "PerformanceMetrics": "assistant_app.services.spectravoice_assistant",
}

__all__ = [
    "Assistant",
    "ConversationMemory",
    "PerformanceMetrics",
    "ScreenCapture",
    "SearchCategory",
    "SearchResult",
    "SmartVoiceDetector",
    "SpectraVoiceAssistant",
    "TextToSpeech",
    "WebSearch",
    "get_logger",
    "setup_logging",
]


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache for subsequent lookups
    return value
