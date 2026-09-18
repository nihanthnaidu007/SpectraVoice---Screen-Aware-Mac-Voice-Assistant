"""Formatted transcript exports (W2 S5): SRT, VTT, Markdown, JSON — zero cloud.

W3 transcripts are append-only JSONL (``meeting_store``): a meta line, then
utterance records (``{type, id, text, spoken_at, spoken_at_iso}``) and
explicit gap markers (``{type: gap, started_at, ended_at, reason}``). The
serializers here are PURE: records in, export text out — the only file IO is
:func:`write_transcript_export` writing the finished string to a destination
the user chose. Nothing here touches a network (S5 is "fully in-posture":
bytes travel from the local corpus to a local directory, nothing else).

Gap policy: SRT/VTT are caption formats — they carry utterances only.
Markdown and JSON carry the gap markers too, because W3's contract is that a
transcript can answer "why is there a hole here"; dropping them from the
human/machine-readable formats would re-introduce silent loss at the very
step that is supposed to move data off-device.

Timestamps are relative to the meeting start (the meta record's
``started_at``; when a transcript has no meta line, the earliest record
timestamp becomes zero). Deterministic and dependency-free (stdlib only).
"""

from __future__ import annotations

import json
import os

from assistant_app.utils.logging_config import get_logger

logger = get_logger(__name__)

SUPPORTED_FORMATS = ("srt", "vtt", "markdown", "json")
FORMAT_EXTENSIONS = {"srt": ".srt", "vtt": ".vtt", "markdown": ".md", "json": ".json"}

# Caption cue rules: never a zero-length cue, never longer than 5s.
_MIN_CUE_SECONDS = 1.0
_MAX_CUE_SECONDS = 5.0


def _record_start(record: dict) -> float | None:
    """The epoch a record begins at (utterances speak; gaps span)."""
    if record.get("type") == "gap":
        started = record.get("started_at")
    else:
        started = record.get("spoken_at")
    if isinstance(started, (int, float)):
        return float(started)
    return None


def _meta_start(records: list[dict]) -> float:
    """Meeting start epoch: the meta line's started_at, else the earliest
    record timestamp (0.0 only when the transcript is empty)."""
    for record in records:
        if record.get("type") == "meta" and isinstance(record.get("started_at"), (int, float)):
            return float(record["started_at"])
    starts = [t for t in (_record_start(r) for r in records) if t is not None]
    return min(starts) if starts else 0.0


def _clock(seconds: float, decimal_sep: str) -> str:
    """SRT/VTT clock: HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT)."""
    millis = max(0, round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{decimal_sep}{millis:03d}"


def _short_clock(seconds: float) -> str:
    """Markdown clock: M:SS (or H:MM:SS past an hour)."""
    total = max(0, round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _cues(records: list[dict], meeting_start: float) -> list[tuple[float, float, str]]:
    """Caption cues from utterance records: (start, end, text), chronological.

    A cue ends where the next one begins (minus one millisecond); the last
    cue holds for a reading window. Gap markers are deliberately NOT cues.
    """
    utterances = sorted(
        (r for r in records if r.get("type") == "utterance" and _record_start(r) is not None),
        key=_record_start,
    )
    cues: list[tuple[float, float, str]] = []
    for index, record in enumerate(utterances):
        text = str(record.get("text", "")).strip()
        if not text:
            continue
        start = max(0.0, (_record_start(record) or 0.0) - meeting_start)
        next_start = _record_start(utterances[index + 1]) if index + 1 < len(utterances) else None
        if next_start is not None:
            raw_end = (next_start - meeting_start) - 0.001
        else:
            raw_end = start + len(text) / 15.0
        # Every cue: at least a 1s reading window, never longer than 5s —
        # no caption spans a gap or hangs on screen for the whole meeting.
        end = min(max(raw_end, start + _MIN_CUE_SECONDS), start + _MAX_CUE_SECONDS)
        cues.append((start, end, text))
    return cues


def to_srt(records: list[dict]) -> str:
    """SubRip captions: numbered cues, comma milliseconds, utterances only."""
    start = _meta_start(records)
    blocks = []
    for index, (cue_start, cue_end, text) in enumerate(_cues(records, start), start=1):
        blocks.append(
            f"{index}\n{_clock(cue_start, ',')} --> {_clock(cue_end, ',')}\n{text}\n"
        )
    return "\n".join(blocks)


def to_vtt(records: list[dict]) -> str:
    """WebVTT captions: WEBVTT header, dot milliseconds, utterances only."""
    start = _meta_start(records)
    lines = ["WEBVTT", ""]
    for cue_start, cue_end, text in _cues(records, start):
        lines.append(f"{_clock(cue_start, '.')} --> {_clock(cue_end, '.')}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def to_markdown(records: list[dict]) -> str:
    """Human-readable transcript: utterances with clock stamps, gaps kept
    visible as their own section (W3: a hole explains itself)."""
    start = _meta_start(records)
    meta = next((r for r in records if r.get("type") == "meta"), {})
    lines = ["# Meeting transcript", ""]
    if meta.get("started_at_iso"):
        lines.append(f"Started: {meta['started_at_iso']}")
        lines.append("")
    utterances = [r for r in records if r.get("type") == "utterance" and str(r.get("text", "")).strip()]
    if utterances:
        lines.append("## Utterances")
        lines.append("")
        for record in utterances:
            stamp = _short_clock(max(0.0, (_record_start(record) or 0.0) - start))
            lines.append(f"- **[{stamp}]** {str(record.get('text', '')).strip()}")
        lines.append("")
    else:
        lines.append("_No transcribed utterances._")
        lines.append("")
    gaps = [r for r in records if r.get("type") == "gap"]
    if gaps:
        lines.append("## Gaps (audio kept but not transcribed)")
        lines.append("")
        for record in gaps:
            gap_start = max(0.0, (record.get("started_at") or start) - start) if isinstance(
                record.get("started_at"), (int, float)
            ) else 0.0
            gap_end = (
                max(0.0, (record.get("ended_at") or record.get("started_at") or start) - start)
                if isinstance(record.get("ended_at"), (int, float))
                else gap_start
            )
            reason = record.get("reason", "unknown")
            lines.append(f"- [{_short_clock(gap_start)}–{_short_clock(gap_end)}] {reason}")
        lines.append("")
    return "\n".join(lines)


def to_json(records: list[dict]) -> str:
    """Verbatim transcript as pretty JSON — every record, nothing dropped."""
    return json.dumps(records, indent=2, ensure_ascii=False) + "\n"


_SERIALIZERS = {"srt": to_srt, "vtt": to_vtt, "markdown": to_markdown, "json": to_json}


def write_transcript_export(cfg, meeting_id: str, dest_dir: str, fmt: str) -> str:
    """Serialize one stored meeting's transcript to ``dest_dir`` and return
    the written path.

    Local IO only. Refuses to overwrite (a second export must be deliberate
    — the same rule history_ops.export_meeting applies to directory copies).
    """
    from assistant_app.services.history_ops import get_meeting_dir  # local: keeps import order simple
    from assistant_app.services.meeting_store import read_transcript

    normalized = (fmt or "").strip().lower()
    if normalized not in _SERIALIZERS:
        raise ValueError(f"unsupported export format: {fmt!r} (supported: {', '.join(SUPPORTED_FORMATS)})")
    meeting_dir = get_meeting_dir(cfg, meeting_id)
    transcript_path = os.path.join(meeting_dir, "transcript.jsonl")
    if not os.path.exists(transcript_path):
        raise FileNotFoundError(f"no stored transcript for meeting {meeting_id!r}")

    content = _SERIALIZERS[normalized](read_transcript(transcript_path))

    os.makedirs(dest_dir, exist_ok=True)
    out_path = os.path.join(dest_dir, f"{meeting_id}-transcript{FORMAT_EXTENSIONS[normalized]}")
    if os.path.exists(out_path):
        raise FileExistsError(f"export destination already exists: {out_path}")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(content)
    logger.info(f"📤 Exported transcript {meeting_id} as {normalized} → {out_path} (local file, no upload)")
    return out_path


