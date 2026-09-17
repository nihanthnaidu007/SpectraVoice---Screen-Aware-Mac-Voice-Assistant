"""Meeting transcript store (W3 D1/D4): crash-safe JSONL, gap markers, summaries.

Layout — one meeting is one directory, stable and timestamped so Wave 4 can
index it::

    meetings/                       (meeting.audio_dir, or logs/meetings)
      20260917-101500/              (local start time, seconds resolution)
        transcript.jsonl            line 1 = meta, then one record per line
        summary.md                  human-readable summary (optional)
        summary.json                machine-readable summary + metadata
        clips/*.wav                 utterance clips (meeting.persist_audio)

The transcript is JSONL with three record shapes, all validated on write:

- ``meta``      — exactly line 1: schema version, language, timestamps.
- ``utterance`` — one per transcribed clip, with ``spoken_at`` (epoch seconds)
                  and ``spoken_at_iso`` for human skimming.
- ``gap``       — written instead of an utterance whenever audio existed but
                  was NOT transcribed: queue overflow, transcription failure
                  after retry, or a paused stretch. Silent loss is impossible
                  by construction: the failure modes become visible records.

Crash safety: every record is written and flushed to disk before ``append_*``
returns (spec R6) — a supervisor restart loses at most the in-flight clip,
never an utterance that was already returned by Whisper, and the meta line is
on disk before the first clip can possibly arrive.

Dependency-light like the rest of the services layer: stdlib + logging only.
"""

import json
import os
import threading
import time
from dataclasses import dataclass

from assistant_app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Bumped only for breaking transcript schema changes (Wave 4 reads this).
SCHEMA_VERSION = 1

# Record types (the "type" field of every JSONL line).
TYPE_META = "meta"
TYPE_UTTERANCE = "utterance"
TYPE_GAP = "gap"
TYPE_SUMMARY = "summary"

# Structured gap reasons — the transcript must be able to answer "why is
# there a hole here" without reading the logs.
GAP_OVERFLOW = "queue_overflow"
GAP_TRANSCRIBE_FAILED = "transcription_failed"
GAP_PAUSED = "paused"
GAP_NOT_TRANSCRIBED = "not_transcribed"


@dataclass(frozen=True)
class MeetingPaths:
    """Where one meeting's artifacts live (paths are absolute or cwd-relative
    exactly as configured; the store never interprets them)."""

    meeting_dir: str
    transcript_path: str
    summary_md_path: str
    summary_json_path: str
    clips_dir: str

    @property
    def meeting_id(self) -> str:
        """Directory name — the stable ID Wave 4 indexes by."""
        return os.path.basename(os.path.normpath(self.meeting_dir))


def meeting_dir_name(started_at: float, now_fn=time.localtime) -> str:
    """Timestamped directory name: 20260917-101500 (local time, second resolution).

    Injected ``now_fn`` keeps tests deterministic.
    """
    return time.strftime("%Y%m%d-%H%M%S", now_fn(started_at))


def meeting_root(cfg) -> str:
    """Resolve the meetings root from MeetingConfig (audio_dir or the default)."""
    return cfg.audio_dir or os.path.join("logs", "meetings")


def open_meeting(cfg, started_at: float) -> MeetingPaths:
    """Create the meeting directory and its transcript's meta line.

    Writes the meta record (flushed) before returning, so even a crash
    immediately after start leaves a parseable transcript that says the
    meeting began.
    """
    root = meeting_root(cfg)
    meeting_dir = os.path.join(root, meeting_dir_name(started_at))
    # Second-resolution names can collide (stop -> start within one second, or
    # a supervisor restart mid-meeting); a colliding directory would get a
    # second meta line and corrupt the one-meta-record invariant. Suffix instead.
    suffix = 0
    while os.path.exists(os.path.join(meeting_dir, "transcript.jsonl")):
        suffix += 1
        meeting_dir = os.path.join(root, f"{meeting_dir_name(started_at)}-{suffix}")
    os.makedirs(meeting_dir, exist_ok=True)
    paths = MeetingPaths(
        meeting_dir=meeting_dir,
        transcript_path=os.path.join(meeting_dir, "transcript.jsonl"),
        summary_md_path=os.path.join(meeting_dir, "summary.md"),
        summary_json_path=os.path.join(meeting_dir, "summary.json"),
        clips_dir=os.path.join(meeting_dir, "clips"),
    )
    meta = {
        "type": TYPE_META,
        "schema_version": SCHEMA_VERSION,
        "language": cfg.language,
        "started_at": round(started_at, 3),
        "started_at_iso": _iso(started_at),
    }
    _append_line(paths.transcript_path, meta)
    return paths


def write_gap(paths: MeetingPaths, gap_id: int, started_at: float, ended_at: float, reason: str) -> None:
    """Append one gap marker to an existing meeting transcript (own handle —
    used by the controller for queue overflows from the audio thread while the
    worker owns the steady-state writes)."""
    record = {
        "type": TYPE_GAP,
        "id": gap_id,
        "started_at": round(started_at, 3),
        "ended_at": round(ended_at, 3),
        "started_at_iso": _iso(started_at),
        "ended_at_iso": _iso(ended_at),
        "reason": reason,
    }
    _append_line(paths.transcript_path, record)


class MeetingTranscriptWriter:
    """Append-only JSONL writer with per-record flush (the crash-safety seam).

    One writer owns the steady-state transcript appends; queue-overflow gap
    markers can arrive from the audio thread while the worker is mid-write, so
    writes are serialized by a lock. The meta line belongs to ``open_meeting``
    — this writer only appends utterances and gaps.
    """

    def __init__(self, paths: MeetingPaths, language: str):
        self.paths = paths
        self.language = language
        self._lock = threading.Lock()

    def append_utterance(self, seq: int, text: str, spoken_at: float) -> None:
        """One transcribed utterance. Flushes before returning (spec R6)."""
        record = {
            "type": TYPE_UTTERANCE,
            "id": seq,
            "text": text,
            "spoken_at": round(spoken_at, 3),
            "spoken_at_iso": _iso(spoken_at),
            "written_at": round(time.time(), 3),
        }
        with self._lock:
            _append_line(self.paths.transcript_path, record)

    def append_gap(self, seq: int, started_at: float, ended_at: float, reason: str) -> None:
        """One gap marker (audio existed; it was not transcribed — why attached)."""
        record = {
            "type": TYPE_GAP,
            "id": seq,
            "started_at": round(started_at, 3),
            "ended_at": round(ended_at, 3),
            "started_at_iso": _iso(started_at),
            "ended_at_iso": _iso(ended_at),
            "reason": reason,
        }
        with self._lock:
            _append_line(self.paths.transcript_path, record)


def write_summary_files(paths: MeetingPaths, summary_text: str, metadata: dict, written_at: float | None = None) -> None:
    """Write summary.md + summary.json for a finished meeting (D3: files, never TTS)."""
    written_at = time.time() if written_at is None else written_at
    payload = {
        "type": TYPE_SUMMARY,
        "schema_version": SCHEMA_VERSION,
        "meeting_id": paths.meeting_id,
        "written_at": round(written_at, 3),
        "written_at_iso": _iso(written_at),
        **metadata,
    }
    with open(paths.summary_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    with open(paths.summary_md_path, "w", encoding="utf-8") as handle:
        handle.write(summary_text)
        handle.flush()
        os.fsync(handle.fileno())


def read_transcript(transcript_path: str) -> list[dict]:
    """Parse a meeting transcript JSONL (skips malformed lines with a log)."""
    records: list[dict] = []
    if not os.path.exists(transcript_path):
        return records
    with open(transcript_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.warning(f"⚠️ Skipping malformed transcript line: {exc}")
    return records


def utterances(records: list[dict]) -> list[dict]:
    """Filter to utterance records (order preserved)."""
    return [r for r in records if r.get("type") == TYPE_UTTERANCE]


def _append_line(path: str, record: dict) -> None:
    line = json.dumps(record, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _iso(epoch: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(epoch)) + f".{int(epoch % 1 * 1000):03d}"


def prune_meetings(root: str, retention_hours: float) -> int:
    """Apply the shared retention mechanism to meeting directories (D4)."""
    from assistant_app.utils.retention import prune_expired

    return prune_expired(root, retention_hours, log=logger)
