"""Shared artifact retention (W3 D4).

One cleanup mechanism, two consumers (spec R2): meeting directories and
dictation's persisted clips. "Age" is the file's mtime, measured in hours,
against the configured TTL. A TTL of 0 keeps artifacts forever (the default —
deleting user data unprompted would betray the local-first posture), and
per-entry failures are logged and skipped so one unreadable file can never
abort the sweep. Nothing here touches anything outside the directory given.
"""

import logging
import os
import shutil
import time
from collections.abc import Callable

from assistant_app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Files newer than this many seconds are considered "just written" even if the
# system clock jumps — 0 TTL means keep forever, not "delete immediately".
_NEVER = 0.0


def prune_expired(
    directory: str,
    retention_hours: float,
    now_fn: Callable[[], float] = time.time,
    log: logging.Logger | None = None,
) -> int:
    """Delete entries in ``directory`` older than ``retention_hours`` hours.

    Returns the number of entries removed. ``retention_hours <= 0`` keeps
    everything (0 = forever). A missing directory is not an error — there is
    nothing to prune. Entries that cannot be stat'ed or removed are logged and
    skipped so one bad file never blocks the rest of the sweep. Recursive:
    meeting directories contain the transcript, summary, and clips together
    and age as a unit (the directory mtime governs the whole tree).
    """
    if retention_hours <= _NEVER:
        return 0
    if not os.path.isdir(directory):
        return 0
    log = log or logger
    cutoff = now_fn() - retention_hours * 3600.0
    removed = 0
    try:
        entries = os.listdir(directory)
    except OSError as exc:
        log.warning(f"⚠️ Retention sweep could not list {directory}: {exc}")
        return 0
    for name in entries:
        path = os.path.join(directory, name)
        try:
            if os.path.getmtime(path) > cutoff:
                continue
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)
            removed += 1
        except OSError as exc:
            log.warning(f"⚠️ Retention sweep could not remove {path}: {exc}")
    if removed:
        log.info(f"🧹 Retention: removed {removed} expired artifact(s) from {directory}")
    return removed
