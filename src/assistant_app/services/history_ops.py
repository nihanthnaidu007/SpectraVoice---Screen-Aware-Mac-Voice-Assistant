"""History mutation ops (W4 D2/D3): delete and export — the ONLY writers W4 adds.

Separated from the read-only search module on purpose: everything here changes
what is on disk, so confirmation is STRUCTURAL — every destructive call takes
``confirm: bool`` and refuses without it (never a single accidental click, on
any surface). Callers own the UX of the confirmation (CLI ``--confirm`` flag,
dashboard checkbox + dialog); this layer enforces that one exists.

Delete semantics (locked decisions): deletes are directory-granularity rmtree
of one meeting directory — the transcript, summary, and clips age as a unit.
There is no derived index to desync (the store IS the only state), so
"store-first" is trivially satisfied. Per-utterance redaction would break the
append-only + fsync invariant and is explicitly out of scope this wave.

Export semantics: a COPY of one meeting directory to a user-chosen
destination — the only legitimate off-device path (locked decision). It never
overwrites an existing destination; a second export must be deliberate.

Dependency-light (stdlib + logging): importable and testable on Linux CI.
"""

from __future__ import annotations

import os
import shutil

from assistant_app.services.history_search import HistoryLookupError, get_meeting_dir
from assistant_app.services.meeting_store import meeting_root
from assistant_app.utils.logging_config import get_logger

logger = get_logger(__name__)

__all__ = [
    "HistoryLookupError",
    "HistoryRefusalError",
    "delete_all_meetings",
    "delete_meeting",
    "export_meeting",
]


class HistoryRefusalError(RuntimeError):
    """A destructive history action was requested without confirmation."""


def delete_meeting(cfg, meeting_id: str, *, confirm: bool) -> bool:
    """Remove one stored meeting directory (transcript, summary, clips).

    Returns True when the directory was removed, False when it was already
    gone (idempotent — a double delete is not an error). Raises
    :class:`HistoryRefusalError` when ``confirm`` is falsy and
    :class:`HistoryLookupError` when the id cannot be a stored meeting.
    """
    if not confirm:
        raise HistoryRefusalError(
            f"refusing to delete meeting {meeting_id!r} without explicit confirmation "
            "(deletion is permanent)"
        )
    meeting_dir = get_meeting_dir(cfg, meeting_id)
    if not os.path.isdir(meeting_dir):
        return False
    shutil.rmtree(meeting_dir)
    logger.info(f"🗑️ Deleted meeting {meeting_id} ({meeting_dir})")
    return True


def delete_all_meetings(cfg, *, confirm: bool) -> int:
    """Remove EVERY stored meeting directory (the corpus nuke).

    Returns the number of meeting directories removed; the corpus root itself
    is kept (it is config-owned storage, not user data). Per-entry failures
    are logged and skipped (the shared retention pattern) so one locked
    directory can never leave the user unsure whether Delete All "did
    nothing". Raises :class:`HistoryRefusalError` when ``confirm`` is falsy.
    """
    if not confirm:
        raise HistoryRefusalError(
            "refusing to delete ALL meetings without explicit confirmation (deletion is permanent)"
        )
    root = meeting_root(cfg)
    if not os.path.isdir(root):
        return 0
    removed = 0
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        try:
            shutil.rmtree(path)
            removed += 1
        except OSError as exc:
            logger.warning(f"⚠️ Could not delete {path}: {exc}")
    logger.info(f"🗑️ Deleted all meetings: {removed} directory(ies) removed from {root}")
    return removed


def export_meeting(cfg, meeting_id: str, dest_dir: str) -> str:
    """Copy one meeting's directory into ``dest_dir`` (user-chosen destination).

    Returns the destination directory path. The corpus copy is untouched —
    export never deletes, never rewrites, never uploads anything itself; it
    hands bytes to the destination the user chose. Raises
    :class:`HistoryLookupError` for an unusable meeting id and
    ``FileExistsError`` when the destination already exists (a second export
    must be deliberate, not a silent overwrite).
    """
    if not (dest_dir or "").strip():
        raise ValueError("export destination required — pass DIR or set history.export_dir")
    meeting_dir = get_meeting_dir(cfg, meeting_id)
    if not os.path.isdir(meeting_dir):
        raise HistoryLookupError(f"no such stored meeting: {meeting_id!r}")
    dest = os.path.join(dest_dir, meeting_id)
    if os.path.exists(dest):
        raise FileExistsError(f"export destination already exists: {dest}")
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copytree(meeting_dir, dest)
    logger.info(f"📤 Exported meeting {meeting_id} → {dest} (user-initiated off-device copy)")
    return dest
