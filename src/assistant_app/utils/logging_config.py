"""
Logging Configuration
=====================
Centralized logging setup for SpectraVoice: console output plus rotating file
logging (matching the `logging:` section of config.yaml: logs/assistant.log,
10 MB per file, 5 backups), and crash capture that writes tracebacks of
uncaught exceptions — including ones escaping background threads — to the log
file, so a silent death is always diagnosable after the fact.
"""

import logging
import os
import sys
import threading
import traceback
from logging.handlers import RotatingFileHandler
from types import TracebackType

DEFAULT_LOG_FILE = "logs/assistant.log"
DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MB, per config.yaml `logging.max_size_mb`
DEFAULT_BACKUP_COUNT = 5
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s - %(message)s"

# Third-party loggers that are too chatty at INFO level
_NOISY_LOGGERS = ("urllib3", "openai", "httpx", "httpcore", "PIL")


def setup_logging(
    debug: bool = False,
    log_file: str = DEFAULT_LOG_FILE,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
) -> logging.Handler | None:
    """
    Configure console + rotating file logging.

    Args:
        debug: If True, enables DEBUG level logging with verbose output.
        log_file: Path to the rotating log file (created on first write).
        max_bytes: Maximum size of each log file before rotation.
        backup_count: Number of rotated backup files to keep.

    Returns:
        The RotatingFileHandler, or None if file logging could not be set up
        (e.g. read-only filesystem) — console logging still works in that case.
    """
    level = logging.DEBUG if debug else logging.INFO

    root = logging.getLogger()
    # Replicate basicConfig(force=True): replace any previously configured handlers
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(level)

    formatter = logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    console_handler = logging.StreamHandler(stream=sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = None
    try:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            delay=True,  # create the file lazily on first record
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
    except OSError as e:
        print(f"⚠️ File logging disabled ({e})", file=sys.stderr)

    # Suppress noisy third-party loggers
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return file_handler


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance with the given name."""
    return logging.getLogger(name)


def format_exception_trace(
    exc_type: type[BaseException],
    exc_value: BaseException,
    exc_tb: TracebackType | None,
) -> str:
    """Format an exception triple into a full traceback string."""
    return "".join(traceback.format_exception(exc_type, exc_value, exc_tb))


def install_crash_handlers() -> None:
    """
    Write uncaught exceptions to the log file instead of dying silently.

    Hooks both the main thread (sys.excepthook) and every background thread
    (threading.excepthook) — before this, an uncaught exception in the audio
    or capture threads killed the assistant with no trace on disk.
    """
    crash_logger = get_logger("spectravoice.crash")

    def _capture(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
        thread_name: str | None = None,
    ) -> None:
        where = f" in thread '{thread_name}'" if thread_name else ""
        entry = f"💥 Uncaught exception{where}\n{format_exception_trace(exc_type, exc_value, exc_tb)}"
        crash_logger.critical(entry)
        sys.stderr.write(entry)

    def _sys_hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: TracebackType | None,
    ) -> None:
        # A Ctrl+C reaching the top of main is a normal exit, not a crash.
        if exc_type is not None and issubclass(exc_type, KeyboardInterrupt):
            return
        _capture(exc_type, exc_value, exc_tb)

    def _thread_hook(args: "threading._ExceptHookArgs") -> None:
        thread_name = args.thread.name if args.thread is not None else None
        _capture(args.exc_type, args.exc_value, args.exc_traceback, thread_name)

    sys.excepthook = _sys_hook
    threading.excepthook = _thread_hook
