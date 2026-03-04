"""
Utils Package
=============
Utility modules for SpectraVoice.
"""

from .logging_config import setup_logging, get_logger
from .config import get_config, get_config_manager, init_config, AssistantConfig
from .error_handler import (
    get_error_handler,
    retry_with_backoff,
    ErrorCategory,
    ErrorSeverity,
    ErrorInfo,
    safe_execute
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    # Configuration
    "get_config",
    "get_config_manager",
    "init_config",
    "AssistantConfig",
    # Error Handling
    "get_error_handler",
    "retry_with_backoff",
    "ErrorCategory",
    "ErrorSeverity",
    "ErrorInfo",
    "safe_execute",
]
