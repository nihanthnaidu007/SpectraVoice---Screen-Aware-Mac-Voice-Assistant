"""
Services Package
================
High-level assistant services.

Import-light by design: the orchestrator service pulls in audio, vision, and
LLM modules, so it is re-exported lazily via PEP 562 `__getattr__` — importing
this package on Linux CI does not require those heavy dependencies.
"""

import importlib

_LAZY_EXPORTS = {
    "PerformanceMetrics": "assistant_app.services.spectravoice_assistant",
    "SpectraVoiceAssistant": "assistant_app.services.spectravoice_assistant",
}

__all__ = ["PerformanceMetrics", "SpectraVoiceAssistant"]


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache for subsequent lookups
    return value
