"""Corpus-wide history search over the meeting store (W4 D1) — strictly read-only.

Scan-on-query (locked decision): there is NO derived index (no SQLite/FTS).
The JSONL transcripts on disk ARE the index, so a result can never diverge
from what is actually stored — the index/disk-divergence and rebuild-after-
crash hazards (pre-flight H3/H5) cannot exist by construction. Local
single-user corpora are small; the 100-meeting smoke test in
``tests/test_history_search.py`` pins a number to this choice and it reverses
only on demonstrated corpus size.

Contract:
- **Read-only**: no writes, no directories created, and the transcript
  writer's lock is never touched — reads use the tolerant
  ``meeting_store.read_transcript`` helper (extended here, not duplicated),
  so a torn last line from an in-flight meeting is skipped, not fatal (H4).
- **Live meetings are injected, not discovered**: callers pass the
  meeting_id(s) currently recording (the orchestrator owns that knowledge);
  results carry ``is_recording`` so an in-flight meeting is marked, never
  mis-ranked.
- **Search scope = meetings only** (locked decision): dictation transcripts
  are deliberately ephemeral (W1) and screen frames are never persisted, so
  there is nothing else to search. This module must never grow a second
  store — that would create unconsented data (H2/R6).
- **Local-only**: pure local file reads; importing a network/LLM/pipeline
  module here is a structural test failure (AST isolation, W4 D4).
- **Dependency-light**: stdlib + logging only — importable and testable on
  Linux CI.

Every result is a snapshot dict::

    {
        "meeting_id": str,          # directory name (the stable store ID)
        "started_at": float,        # epoch seconds (meta line; mtime fallback)
        "started_at_iso": str,      # local time, for humans
        "duration_seconds": float,  # transcript span (now-start for live)
        "utterance_count": int,
        "gap_count": int,
        "summary_present": bool,
        "matched_snippets": list[str],   # [] for plain scans
        "is_recording": bool,
    }
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable

from assistant_app.services.meeting_store import (
    TYPE_GAP,
    TYPE_META,
    TYPE_UTTERANCE,
    meeting_root,
    read_transcript,
    utterances,
)
from assistant_app.utils.logging_config import get_logger

logger = get_logger(__name__)

TRANSCRIPT_FILENAME = "transcript.jsonl"
SUMMARY_MD_FILENAME = "summary.md"
SUMMARY_JSON_FILENAME = "summary.json"

# Cap per-meeting snippets so one meeting that repeats the query cannot drown
# the rest of the corpus in the dashboard/CLI render.
SNIPPET_LIMIT = 5

SECONDS_PER_DAY = 86400.0


class HistoryLookupError(LookupError):
    """A meeting id did not resolve to a stored meeting directory."""


def get_meeting_dir(cfg, meeting_id: str) -> str:
    """Resolve + validate a meeting_id to its directory under the corpus root.

    Security seam for everything that turns a user-supplied meeting_id (CLI
    argument, UI row) into a filesystem path, read or write: the id must name
    a DIRECT child of the meetings root — no separators, no traversal, and a
    realpath containment check closes symlink escapes. Raises
    :class:`HistoryLookupError` otherwise (callers render it as "no such
    meeting", never a traceback).
    """
    if (
        not meeting_id
        or os.path.basename(meeting_id) != meeting_id
        or meeting_id in {".", ".."}
        or meeting_id.startswith(".")
    ):
        raise HistoryLookupError(f"not a stored meeting id: {meeting_id!r}")
    root = meeting_root(cfg)
    path = os.path.join(root, meeting_id)
    root_real = os.path.realpath(root)
    path_real = os.path.realpath(path)
    if os.path.dirname(path_real) != root_real or path_real == root_real:
        raise HistoryLookupError(f"not a stored meeting id: {meeting_id!r}")
    return path_real


def parse_history_date(value: str) -> float:
    """Parse a ``YYYY-MM-DD`` CLI date into a local-time epoch bound.

    Raises ``ValueError`` for anything else (the CLI turns that into a clean
    exit code, not a traceback).
    """
    stripped = (value or "").strip()
    try:
        return time.mktime(time.strptime(stripped, "%Y-%m-%d"))  # local midnight
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid date {value!r} — expected YYYY-MM-DD") from exc


def date_range_bounds(after: str | None, before: str | None) -> tuple[float | None, float | None]:
    """CLI date flags → (lower, upper) epoch bounds.

    Both ends are inclusive calendar days in local time: ``after=2026-09-01``
    starts at Sep 1 00:00:00, ``before=2026-09-03`` ends at Sep 3 23:59:59.
    """
    lower = parse_history_date(after) if after else None
    upper = parse_history_date(before) + SECONDS_PER_DAY if before else None
    return lower, upper


def scan_history(
    cfg,
    *,
    after: str | None = None,
    before: str | None = None,
    live_meeting_ids: frozenset[str] | set[str] = frozenset(),
    now_fn: Callable[[], float] = time.time,
) -> list[dict]:
    """Snapshot every stored meeting (newest first), optionally date-filtered.

    Pure read: the corpus is only ever opened, never written. A missing
    corpus root is an empty history, not an error. ``matched_snippets`` is
    always ``[]`` here — this is the "list everything" view.
    """
    lower, upper = date_range_bounds(after, before)
    meetings = [
        snapshot
        for snapshot in _scan_corpus(cfg, live_meeting_ids=live_meeting_ids, now_fn=now_fn)
        if _in_range(snapshot, lower, upper)
    ]
    return meetings


def search_history(
    cfg,
    query: str,
    *,
    after: str | None = None,
    before: str | None = None,
    live_meeting_ids: frozenset[str] | set[str] = frozenset(),
    now_fn: Callable[[], float] = time.time,
) -> list[dict]:
    """Search utterance text across the corpus (case-insensitive substring).

    Returns only meetings with at least one matching utterance, newest first;
    ``matched_snippets`` holds the matching utterance texts (capped at
    ``SNIPPET_LIMIT`` per meeting). An empty/whitespace query is a caller bug
    — ``ValueError``, not a silent full-corpus listing.
    """
    needle = (query or "").strip().lower()
    if not needle:
        raise ValueError("history search requires a non-empty query")
    matches: list[dict] = []
    for snapshot in scan_history(cfg, after=after, before=before, live_meeting_ids=live_meeting_ids, now_fn=now_fn):
        snippets = [text for text in snapshot["_texts"] if needle in text.lower()][:SNIPPET_LIMIT]
        if not snippets:
            continue
        public = {key: value for key, value in snapshot.items() if key != "_texts"}
        matches.append({**public, "matched_snippets": snippets})
    return matches


def read_summary_text(cfg, meeting_id: str) -> str:
    """Read one stored meeting's summary.md ("" when absent) — dashboard view.

    A read, but still routed through :func:`get_meeting_dir` so a
    user-supplied id can never resolve outside the corpus (same containment
    rule as delete/export).
    """
    meeting_dir = get_meeting_dir(cfg, meeting_id)
    summary_path = os.path.join(meeting_dir, SUMMARY_MD_FILENAME)
    try:
        with open(summary_path, encoding="utf-8") as handle:
            return handle.read()
    except FileNotFoundError:
        return ""
    except OSError as exc:
        logger.warning(f"⚠️ Could not read summary for {meeting_id}: {exc}")
        return ""


# === internals (pure, scan-on-query) ===


def _scan_corpus(
    cfg,
    *,
    live_meeting_ids: frozenset[str] | set[str],
    now_fn: Callable[[], float],
) -> list[dict]:
    """Walk the corpus root once and snapshot every meeting directory.

    Newest first (a history view reads back-to-front). Directories without a
    parseable transcript still appear — a privacy dashboard must show what is
    on disk, even when degenerate — with counts of 0 and an mtime fallback
    for ``started_at``.
    """
    root = meeting_root(cfg)
    if not os.path.isdir(root):
        return []
    now = now_fn()
    snapshots: list[dict] = []
    try:
        entries = sorted(os.listdir(root), reverse=True)
    except OSError as exc:
        logger.warning(f"⚠️ History scan could not list {root}: {exc}")
        return []
    for name in entries:
        meeting_dir = os.path.join(root, name)
        if not os.path.isdir(meeting_dir):
            continue  # stray files are retention's business, not history's
        records = read_transcript(os.path.join(meeting_dir, TRANSCRIPT_FILENAME))
        snapshots.append(
            _snapshot_one(
                meeting_id=name,
                meeting_dir=meeting_dir,
                records=records,
                is_live=name in live_meeting_ids,
                now=now,
            )
        )
    snapshots.sort(key=lambda s: (s["started_at"], s["meeting_id"]), reverse=True)
    return snapshots


def _snapshot_one(
    meeting_id: str, meeting_dir: str, records: list[dict], is_live: bool, now: float
) -> dict:
    """One meeting directory → its snapshot dict (tolerant of partial files)."""
    meta = next((r for r in records if r.get("type") == TYPE_META), None)
    if meta is not None and isinstance(meta.get("started_at"), (int, float)):
        started_at = float(meta["started_at"])
    else:
        # open_meeting always writes the meta line first; a missing one means
        # a foreign or crashed directory — surface it with the dir mtime.
        started_at = os.path.getmtime(meeting_dir)
    utts = utterances(records)
    gaps = [r for r in records if r.get("type") == TYPE_GAP]
    texts = [str(u.get("text", "")) for u in utts if u.get("type") == TYPE_UTTERANCE]
    last_event = started_at
    for u in utts:
        if isinstance(u.get("spoken_at"), (int, float)):
            last_event = max(last_event, float(u["spoken_at"]))
    for g in gaps:
        if isinstance(g.get("ended_at"), (int, float)):
            last_event = max(last_event, float(g["ended_at"]))
    # A live meeting's duration is "so far" — the snapshot is a moment in time.
    duration = (now - started_at) if is_live else max(0.0, last_event - started_at)
    return {
        "meeting_id": meeting_id,
        "started_at": round(started_at, 3),
        "started_at_iso": _iso(started_at),
        "duration_seconds": round(duration, 3),
        "utterance_count": len(utts),
        "gap_count": len(gaps),
        "summary_present": os.path.exists(os.path.join(meeting_dir, SUMMARY_MD_FILENAME))
        or os.path.exists(os.path.join(meeting_dir, SUMMARY_JSON_FILENAME)),
        "matched_snippets": [],
        "is_recording": is_live,
        # Not part of the public shape (the keys above are the contract) —
        # carried for search_history's snippet extraction, which strips it
        # before returning results.
        "_texts": texts,
    }


def _in_range(snapshot: dict, lower: float | None, upper: float | None) -> bool:
    started = snapshot["started_at"]
    if lower is not None and started < lower:
        return False
    return not (upper is not None and started >= upper)


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))
