"""Robust Error Handler with Auto-Retry and Exponential Backoff."""

import functools
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import ParamSpec, TypeVar

from assistant_app.utils.logging_config import get_logger

logger = get_logger(__name__)

P = ParamSpec('P')
T = TypeVar('T')


class ErrorCategory(Enum):
    """Categories of errors for appropriate handling."""
    API = "api"                  # OpenAI, Tavily API errors
    NETWORK = "network"          # Connection, timeout errors
    AUDIO = "audio"              # Microphone, speaker errors
    VISION = "vision"            # Screen capture errors
    SYSTEM = "system"            # File, permission errors
    VALIDATION = "validation"    # Input validation errors
    UNKNOWN = "unknown"          # Unclassified errors


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"           # Log and continue
    MEDIUM = "medium"     # Retry, then fallback
    HIGH = "high"         # Retry with user notification
    CRITICAL = "critical" # Stop operation, require intervention


@dataclass
class ErrorInfo:
    """Structured error information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    user_message: str
    original_error: Exception | None = None
    recoverable: bool = True
    context: dict = field(default_factory=dict)
    
    def __str__(self) -> str:
        return f"[{self.category.value.upper()}] {self.message}"


# Error classification patterns
ERROR_PATTERNS: dict[str, tuple[ErrorCategory, ErrorSeverity, str]] = {
    # API Errors
    "rate_limit": (ErrorCategory.API, ErrorSeverity.MEDIUM, "API rate limit reached. Please wait a moment."),
    "insufficient_quota": (ErrorCategory.API, ErrorSeverity.HIGH, "API quota exceeded. Check your billing."),
    "invalid_api_key": (ErrorCategory.API, ErrorSeverity.CRITICAL, "Invalid API key. Please check your configuration."),
    "model_not_found": (ErrorCategory.API, ErrorSeverity.HIGH, "AI model unavailable. Using fallback."),
    "context_length": (ErrorCategory.API, ErrorSeverity.MEDIUM, "Message too long. Summarizing..."),
    "bad_request": (ErrorCategory.API, ErrorSeverity.MEDIUM, "Invalid request. Retrying..."),
    "server_error": (ErrorCategory.API, ErrorSeverity.MEDIUM, "Server error. Retrying..."),
    
    # Network Errors
    "connection": (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, "Connection issue. Checking network..."),
    "timeout": (ErrorCategory.NETWORK, ErrorSeverity.MEDIUM, "Request timed out. Retrying..."),
    "ssl": (ErrorCategory.NETWORK, ErrorSeverity.HIGH, "Secure connection failed."),
    
    # Audio Errors
    "microphone": (ErrorCategory.AUDIO, ErrorSeverity.HIGH, "Microphone not available. Check audio settings."),
    "no_speech": (ErrorCategory.AUDIO, ErrorSeverity.LOW, "No speech detected. Please try again."),
    "audio_playback": (ErrorCategory.AUDIO, ErrorSeverity.MEDIUM, "Audio playback issue."),
    
    # Vision Errors  
    "screen_capture": (ErrorCategory.VISION, ErrorSeverity.MEDIUM, "Screen capture failed. Retrying..."),
    "image_encoding": (ErrorCategory.VISION, ErrorSeverity.MEDIUM, "Image encoding error."),
    
    # System Errors
    "file_not_found": (ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM, "File not found."),
    "permission": (ErrorCategory.SYSTEM, ErrorSeverity.HIGH, "Permission denied. Check system settings."),
    "disk_space": (ErrorCategory.SYSTEM, ErrorSeverity.HIGH, "Low disk space."),
}


def classify_error(error: Exception) -> ErrorInfo:
    """
    Classify an exception into a structured ErrorInfo.
    
    Args:
        error: The exception to classify
    
    Returns:
        ErrorInfo with category, severity, and user-friendly message
    """
    error_str = str(error).lower()
    error_type = type(error).__name__.lower()
    
    # Check patterns
    for pattern, (category, severity, user_msg) in ERROR_PATTERNS.items():
        if pattern in error_str or pattern in error_type:
            return ErrorInfo(
                category=category,
                severity=severity,
                message=str(error),
                user_message=user_msg,
                original_error=error,
                recoverable=severity != ErrorSeverity.CRITICAL
            )
    
    # OpenAI specific errors
    if "openai" in error_type or "apiconnectionerror" in error_type:
        return ErrorInfo(
            category=ErrorCategory.API,
            severity=ErrorSeverity.MEDIUM,
            message=str(error),
            user_message="API connection issue. Retrying...",
            original_error=error,
            recoverable=True
        )
    
    if "ratelimit" in error_type:
        return ErrorInfo(
            category=ErrorCategory.API,
            severity=ErrorSeverity.MEDIUM,
            message=str(error),
            user_message="Too many requests. Waiting...",
            original_error=error,
            recoverable=True
        )
    
    # Speech recognition errors
    if "unknownvalueerror" in error_type:
        return ErrorInfo(
            category=ErrorCategory.AUDIO,
            severity=ErrorSeverity.LOW,
            message="Could not understand audio",
            user_message="I didn't catch that. Please try again.",
            original_error=error,
            recoverable=True
        )
    
    if "requestserror" in error_type:
        return ErrorInfo(
            category=ErrorCategory.NETWORK,
            severity=ErrorSeverity.MEDIUM,
            message=str(error),
            user_message="Network request failed. Retrying...",
            original_error=error,
            recoverable=True
        )
    
    # Default: unknown error
    return ErrorInfo(
        category=ErrorCategory.UNKNOWN,
        severity=ErrorSeverity.MEDIUM,
        message=str(error),
        user_message="An unexpected error occurred. Please try again.",
        original_error=error,
        recoverable=True
    )


def retry_with_backoff(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    exponential_base: float = 2.0,
    retryable_categories: set[ErrorCategory] | None = None,
    on_retry: Callable[[int, ErrorInfo], None] | None = None
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator for automatic retry with exponential backoff.
    
    Args:
        max_attempts: Maximum number of attempts (default: 3)
        base_delay: Initial delay in seconds (default: 1.0)
        max_delay: Maximum delay cap in seconds (default: 30.0)
        exponential_base: Base for exponential backoff (default: 2.0)
        retryable_categories: Set of error categories to retry (default: API, NETWORK)
        on_retry: Callback function called on each retry
    
    Returns:
        Decorated function with retry logic
    
    Usage:
        @retry_with_backoff(max_attempts=3)
        def call_api():
            ...
    """
    if retryable_categories is None:
        retryable_categories = {
            ErrorCategory.API, 
            ErrorCategory.NETWORK, 
            ErrorCategory.VISION
        }
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_error: ErrorInfo | None = None
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                    
                except Exception as e:
                    error_info = classify_error(e)
                    last_error = error_info
                    
                    # Log the error
                    logger.warning(
                        f"⚠️ Attempt {attempt}/{max_attempts} failed: {error_info}"
                    )
                    
                    # Check if we should retry
                    should_retry = (
                        attempt < max_attempts and
                        error_info.recoverable and
                        error_info.category in retryable_categories
                    )
                    
                    if not should_retry:
                        logger.error(f"❌ Not retrying: {error_info}")
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        base_delay * (exponential_base ** (attempt - 1)),
                        max_delay
                    )
                    
                    # Call retry callback if provided
                    if on_retry:
                        on_retry(attempt, error_info)
                    
                    logger.info(f"⏳ Retrying in {delay:.1f}s...")
                    time.sleep(delay)
            
            # All attempts failed
            if last_error:
                raise last_error.original_error or Exception(last_error.message)
            raise Exception("All retry attempts failed")
        
        return wrapper
    return decorator


class ErrorHandler:
    """
    Centralized error handling for SpectraVoice.
    
    Features:
    - Error classification and logging
    - User-friendly error messages
    - Error statistics tracking
    - Graceful degradation strategies
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.error_counts: dict[ErrorCategory, int] = {cat: 0 for cat in ErrorCategory}
        self.last_errors: list[ErrorInfo] = []
        self.max_error_history = 50
    
    def handle(
        self, 
        error: Exception, 
        context: str = "",
        notify_user: bool = True
    ) -> ErrorInfo:
        """
        Handle an error with classification and logging.
        
        Args:
            error: The exception to handle
            context: Additional context about where the error occurred
            notify_user: Whether to return user-friendly message
        
        Returns:
            ErrorInfo with classified error details
        """
        error_info = classify_error(error)
        error_info.context["location"] = context
        
        # Update statistics
        self.error_counts[error_info.category] += 1
        self.last_errors.append(error_info)
        if len(self.last_errors) > self.max_error_history:
            self.last_errors.pop(0)
        
        # Log based on severity
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.error(f"🚨 CRITICAL [{context}]: {error_info}")
            self.logger.error(traceback.format_exc())
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(f"❌ ERROR [{context}]: {error_info}")
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"⚠️ WARNING [{context}]: {error_info}")
        else:
            self.logger.debug(f"ℹ️ INFO [{context}]: {error_info}")
        
        return error_info
    
    def get_user_message(self, error_info: ErrorInfo) -> str:
        """Get a user-friendly message for the error."""
        return error_info.user_message
    
    def get_stats(self) -> dict:
        """Get error statistics."""
        return {
            "total_errors": sum(self.error_counts.values()),
            "by_category": {cat.value: count for cat, count in self.error_counts.items()},
            "recent_count": len(self.last_errors)
        }
    
    def reset_stats(self) -> None:
        """Reset error statistics."""
        self.error_counts = {cat: 0 for cat in ErrorCategory}
        self.last_errors.clear()
    
    def is_healthy(self, threshold: int = 10) -> bool:
        """
        Check if error rate is acceptable.
        
        Args:
            threshold: Max errors before considered unhealthy
        
        Returns:
            True if error count is below threshold
        """
        total = sum(self.error_counts.values())
        return total < threshold


# Global error handler instance
_error_handler: ErrorHandler | None = None


def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler()
    return _error_handler


def safe_execute(
    func: Callable[P, T],
    *args: P.args,
    default: T | None = None,
    context: str = "",
    **kwargs: P.kwargs
) -> T | None:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        default: Default value on error
        context: Context for logging
        **kwargs: Keyword arguments
    
    Returns:
        Function result or default value on error
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        handler = get_error_handler()
        handler.handle(e, context=context or func.__name__)
        return default
