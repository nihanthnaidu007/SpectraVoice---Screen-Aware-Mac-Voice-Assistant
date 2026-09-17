"""Tests for rotating file logging and crash capture (Wave 0 reliability).

Platform-independent: verifies that log records land in a rotating file and
that uncaught exceptions — main thread and background threads — write a full
traceback to the log file. No Mac, audio, or display required.
"""

import logging
import os
import sys
import threading
import types

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import pytest

from assistant_app.utils.logging_config import (
    format_exception_trace,
    install_crash_handlers,
    setup_logging,
)


@pytest.fixture()
def fresh_logging():
    """Isolate root-logger handlers and exception hooks per test."""
    saved_hooks = (sys.excepthook, threading.excepthook)
    yield
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    sys.excepthook, threading.excepthook = saved_hooks


def test_setup_logging_writes_records_to_file(tmp_path, fresh_logging):
    log_file = tmp_path / "assistant.log"
    handler = setup_logging(log_file=str(log_file))
    assert handler is not None, "file handler should be configured"

    logging.getLogger("test.case").info("hello-file-sentinel")
    handler.flush()

    assert "hello-file-sentinel" in log_file.read_text()


def test_log_file_rotates(tmp_path, fresh_logging):
    log_file = tmp_path / "assistant.log"
    handler = setup_logging(log_file=str(log_file), max_bytes=1000, backup_count=3)
    logger = logging.getLogger("test.rotation")
    for i in range(100):
        logger.info("rotation-payload-%03d %s", i, "x" * 60)
    handler.flush()

    rotated = list(tmp_path.glob("assistant.log.*"))
    assert rotated, "expected rotated backup files once maxBytes is exceeded"
    assert handler.backupCount == 3


def test_format_exception_trace(tmp_path, fresh_logging):
    try:
        raise ValueError("format-sentinel")
    except ValueError:
        trace = format_exception_trace(*sys.exc_info())
    assert "format-sentinel" in trace
    assert "ValueError" in trace
    assert "Traceback (most recent call last)" in trace


def test_main_thread_crash_writes_traceback(tmp_path, fresh_logging):
    log_file = tmp_path / "crash.log"
    setup_logging(log_file=str(log_file))
    install_crash_handlers()

    try:
        raise ValueError("boom-sentinel")
    except ValueError:
        exc_info = sys.exc_info()
    sys.excepthook(*exc_info)

    text = log_file.read_text()
    assert "boom-sentinel" in text
    assert "ValueError" in text
    assert "Traceback (most recent call last)" in text


def test_background_thread_crash_writes_traceback(tmp_path, fresh_logging):
    log_file = tmp_path / "thread-crash.log"
    setup_logging(log_file=str(log_file))
    install_crash_handlers()

    try:
        raise RuntimeError("thread-boom-sentinel")
    except RuntimeError:
        exc_type, exc_value, exc_tb = sys.exc_info()
    # Exercise the hook exactly as Python calls it for a dying thread.
    threading.excepthook(
        types.SimpleNamespace(
            exc_type=exc_type,
            exc_value=exc_value,
            exc_traceback=exc_tb,
            thread=threading.current_thread(),
        )
    )

    text = log_file.read_text()
    assert "thread-boom-sentinel" in text
    assert "Uncaught exception" in text


def test_keyboard_interrupt_is_not_logged_as_crash(tmp_path, fresh_logging):
    log_file = tmp_path / "ki.log"
    setup_logging(log_file=str(log_file))
    install_crash_handlers()

    try:
        raise KeyboardInterrupt
    except KeyboardInterrupt:
        exc_info = sys.exc_info()
    sys.excepthook(*exc_info)

    # The file handler is created lazily on first write — a clean shutdown may
    # leave no file at all. What must never happen is a crash record.
    written = log_file.read_text() if log_file.exists() else ""
    assert "Uncaught exception" not in written
