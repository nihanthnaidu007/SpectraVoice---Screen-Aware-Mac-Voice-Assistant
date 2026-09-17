"""Utility helpers for SpectraVoice.

Import-light by design: heavy or third-party-dependent modules (config pulls
in PyYAML) are re-exported lazily via PEP 562 `__getattr__`, so importing
`assistant_app.utils.logging_config` — as the unit tests do on Linux CI —
does not drag in YAML or other optional dependencies.
"""

import importlib

from .logging_config import get_logger, setup_logging

_LAZY_EXPORTS = {
    "AssistantConfig": "assistant_app.utils.config",
    "get_config": "assistant_app.utils.config",
    "get_config_manager": "assistant_app.utils.config",
    "init_config": "assistant_app.utils.config",
    "ErrorCategory": "assistant_app.utils.error_handler",
    "ErrorInfo": "assistant_app.utils.error_handler",
    "ErrorSeverity": "assistant_app.utils.error_handler",
    "get_error_handler": "assistant_app.utils.error_handler",
    "retry_with_backoff": "assistant_app.utils.error_handler",
    "safe_execute": "assistant_app.utils.error_handler",
}

__all__ = [
    "AssistantConfig",
    "ErrorCategory",
    "ErrorInfo",
    "ErrorSeverity",
    "get_config",
    "get_config_manager",
    "get_error_handler",
    "get_logger",
    "init_config",
    "retry_with_backoff",
    "safe_execute",
    "setup_logging",
]


def __getattr__(name: str):
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_path), name)
    globals()[name] = value  # cache for subsequent lookups
    return value
