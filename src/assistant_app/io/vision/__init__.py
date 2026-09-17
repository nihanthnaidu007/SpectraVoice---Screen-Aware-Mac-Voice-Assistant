"""
Vision I/O Package
==================
Vision input/output modules for screen capture.

Import-light by design: the OpenCV/Pillow-dependent capture module is
re-exported lazily via PEP 562 `__getattr__`, so importing this package on
Linux CI does not require vision dependencies or macOS frameworks.
"""

import importlib

_LAZY_EXPORTS = {"ScreenCapture": "assistant_app.io.vision.screen_capture"}

__all__ = ["ScreenCapture"]


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache for subsequent lookups
    return value
