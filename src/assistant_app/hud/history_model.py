"""History dashboard view-model (W4 D2) — pure logic, no AppKit.

The darwin window (``history_window.py``) renders these snapshots. Threading
contract (H1/R1): the SNAPSHOT IS BUILT OFF-THREAD — :func:`build_history_snapshot`
scans the corpus (file IO) and reads the consent gates; the main thread only
receives finished :class:`HistorySnapshot` values and renders them. Nothing in
this module imports AppKit or speaks: dashboard content is never TTS'd and
never re-enters the assistant pipeline (H2 — locked non-negotiable).

Consent state is REFLECTED here, never flipped: the dashboard reads the gates
for display and has no mutation path — control flows only through the same
consent-gated actions every other surface uses (the W2 HUD rule).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from assistant_app.services import history_search


@dataclass(frozen=True)
class ConsentState:
    """What the consent gates say RIGHT NOW (atomic reflection, no mutation)."""

    meeting_kill_switch: bool  # meeting.enabled — the config kill-switch
    meeting_armed: bool  # explicit per-meeting start (RecordingConsent.armed)
    screen_consent: bool  # PrivacyConsent.screen_upload_allowed
    cloud_qa_consent: bool = False  # history.cloud_qa_consent (W1 cloud Q&A)


@dataclass(frozen=True)
class RetentionState:
    """Retention posture, made visible (the W4 answer to keeping 0 = forever)."""

    retention_hours: float  # meeting.retention_hours — 0 keeps everything
    last_sweep_removed: int | None = None  # artifacts removed by the latest startup sweep
    last_sweep_at: float | None = None  # epoch seconds; None = no sweep yet


TOOL_AUDIT_LIMIT = 12  # rendered tail; the full in-memory log stays readable via the CLI


@dataclass(frozen=True)
class ToolAuditEntry:
    """One tool execution as recorded in the executor's in-memory log."""

    timestamp: float
    tool: str
    arguments: dict
    dry_run: bool = False
    result: str = ""  # dry_run | success | failure | error ("" = malformed entry)
    message: str | None = None  # tool-reported outcome text (success/failure)
    error: str | None = None  # exception text (error path)


@dataclass(frozen=True)
class ToolAuditState:
    """Renderable view over the tool execution log (newest first)."""

    entries: tuple[ToolAuditEntry, ...] = field(default_factory=tuple)
    total: int = 0

    @property
    def visible(self) -> tuple[ToolAuditEntry, ...]:
        return self.entries[:TOOL_AUDIT_LIMIT]


@dataclass(frozen=True)
class HistorySnapshot:
    """Everything the dashboard renders, assembled OFF the main thread."""

    meetings: tuple[dict, ...] = field(default_factory=tuple)
    retention: RetentionState = field(default_factory=RetentionState)
    consent: ConsentState = field(default_factory=ConsentState)
    tool_audit: ToolAuditState = field(default_factory=ToolAuditState)


def consent_state_from(recording_consent, privacy_consent, *, cloud_qa_consent: bool = False) -> ConsentState:
    """Read the live gates (duck-typed: the real gates or test fakes)."""
    return ConsentState(
        meeting_kill_switch=bool(recording_consent.feature_enabled),
        meeting_armed=bool(recording_consent.armed),
        screen_consent=bool(privacy_consent.screen_upload_allowed),
        cloud_qa_consent=bool(cloud_qa_consent),
    )


def retention_state_from(meeting_cfg, last_sweep: dict | None) -> RetentionState:
    """Retention display state from the meeting config + last sweep record.

    ``last_sweep`` is the orchestrator's in-memory record of the most recent
    ``run_startup_cleanup`` outcome ({removed, at}) — there is deliberately no
    persisted sweep state to desync (pre-flight H3/H5).
    """
    sweep_removed = None
    sweep_at = None
    if last_sweep is not None:
        sweep_removed = last_sweep.get("removed")
        sweep_at = last_sweep.get("at")
    return RetentionState(
        retention_hours=float(getattr(meeting_cfg, "retention_hours", 0.0)),
        last_sweep_removed=sweep_removed,
        last_sweep_at=sweep_at,
    )


def tool_audit_state_from(
    raw_entries,
    *,
    now_fn: Callable[[], float] = time.time,
    limit: int = TOOL_AUDIT_LIMIT,
) -> ToolAuditState:
    """Normalize the executor's in-memory dicts into renderable state.

    Newer entries first; ``limit`` caps the STORED tail (the render slice is
    ``ToolAuditState.visible``). Never raises on malformed entries — the
    dashboard reflects what exists (pre-flight H-series discipline).
    """
    entries: list[ToolAuditEntry] = []
    for raw in raw_entries or ():
        if not isinstance(raw, dict):
            continue
        try:
            timestamp = float(raw.get("timestamp", 0.0))
        except (TypeError, ValueError):
            timestamp = 0.0
        arguments = raw.get("arguments")
        entries.append(
            ToolAuditEntry(
                timestamp=timestamp,
                tool=str(raw.get("tool", "unknown")),
                arguments=dict(arguments) if isinstance(arguments, dict) else {},
                dry_run=bool(raw.get("dry_run")),
                result=str(raw.get("result", "")),
                message=raw.get("message") if isinstance(raw.get("message"), str) else None,
                error=raw.get("error") if isinstance(raw.get("error"), str) else None,
            )
        )
    entries.sort(key=lambda e: e.timestamp, reverse=True)
    capped = entries[:limit]
    return ToolAuditState(entries=tuple(capped), total=len(entries))


def _tool_arg_summary(arguments: dict) -> str:
    """Compact JSON summary of tool arguments (never crashes rendering)."""
    try:
        text = json.dumps(arguments, sort_keys=True, default=str)
    except (TypeError, ValueError):
        text = str(arguments)
    return text if len(text) <= 120 else text[:117] + "…"


def _age_label(epoch: float, now: float) -> str:
    """Relative age of an in-session event: s/m/h buckets, date past that."""
    delta = max(0.0, now - epoch)
    if delta < 60:
        return f"{max(1, round(delta))}s ago"
    if delta < 3600:
        return f"{round(delta / 60)}m ago"
    if delta < 86400:
        return f"{round(delta / 3600)}h ago"
    return _iso_brief(epoch)


def format_tool_audit_lines(audit: ToolAuditState, *, now_fn: Callable[[], float] = time.time) -> list[str]:
    """Render the audit tail: one line per entry, newest first.

    The log is in-memory and session-scoped (pre-flight H4) — the copy says
    so, because a viewer that implies persistent logging would lie.
    """
    if audit.total == 0:
        return [
            (
                "Tool activity — none yet. Every tool action the assistant runs is "
                "recorded here (in-memory, this session only)."
            )
        ]
    now = now_fn()
    lines = [
        f"Tool activity — last {len(audit.visible)} of {audit.total} this session, newest first:"
    ]
    for entry in audit.visible:
        dry_tag = " [DRY-RUN]" if entry.dry_run else ""
        detail = entry.message or entry.error or _tool_arg_summary(entry.arguments)
        lines.append(f"• {_age_label(entry.timestamp, now)} — {entry.tool}{dry_tag} — {entry.result} — {detail}")
    hidden = audit.total - len(audit.visible)
    if hidden > 0:
        lines.append(f"… {hidden} earlier action(s) not shown — the CLI history path lists everything")
    return lines


def build_history_snapshot(
    meeting_cfg,
    recording_consent,
    privacy_consent,
    *,
    live_meeting_ids: frozenset[str] | set[str] = frozenset(),
    last_sweep: dict | None = None,
    cloud_qa_consent: bool = False,
    tool_log: list[dict] | None = None,
    now_fn: Callable[[], float] = time.time,
) -> HistorySnapshot:
    """Scan the corpus + read the gates into one renderable snapshot.

    Caller responsibility (enforced by convention, verified by review): this
    runs on a WORKER thread — the HUD window dispatches it off the main
    thread and renders the finished value (clock-thread precedent).
    """
    meetings = history_search.scan_history(meeting_cfg, live_meeting_ids=live_meeting_ids, now_fn=now_fn)
    return HistorySnapshot(
        meetings=tuple(meetings),
        retention=retention_state_from(meeting_cfg, last_sweep),
        consent=consent_state_from(recording_consent, privacy_consent, cloud_qa_consent=cloud_qa_consent),
        tool_audit=tool_audit_state_from(tool_log or [], now_fn=now_fn),
    )


# === pure formatting (shared by the window rows and the tests) ===


def format_duration(seconds: float) -> str:
    """Human duration: 90 -> "1m 30s"; 0 -> "0s"."""
    seconds = max(0, round(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def format_meeting_line(snapshot: dict) -> str:
    """One meeting row: id, local date, duration, counts, summary, flags."""
    flags = " [RECORDING]" if snapshot.get("is_recording") else ""
    summary = "summary yes" if snapshot.get("summary_present") else "summary no"
    return (
        f"{snapshot['meeting_id']}  {snapshot['started_at_iso']}  "
        f"{format_duration(snapshot['duration_seconds'])}  "
        f"{snapshot['utterance_count']} utterances, {snapshot['gap_count']} gaps  "
        f"{summary}{flags}"
    )


def format_retention_line(retention: RetentionState) -> str:
    """Retention setting + last-sweep visibility (H9 made visible, not silent)."""
    setting = "keep forever (0h)" if retention.retention_hours <= 0 else f"{retention.retention_hours:g}h"
    if retention.last_sweep_at is None:
        sweep = "startup sweep has not run yet"
    else:
        sweep = (
            f"last sweep removed {retention.last_sweep_removed} artifact(s) at {_iso_brief(retention.last_sweep_at)}"
        )
    return f"Retention: {setting} — {sweep}"


def format_consent_lines(consent: ConsentState) -> list[str]:
    """Three-axis consent reflection: meeting (kill-switch + arm), screen, QA."""
    if not consent.meeting_kill_switch:
        meeting = "Meeting recording: kill-switch ON (recording disabled)"
    elif consent.meeting_armed:
        meeting = "Meeting recording: enabled and ARMED (recording now)"
    else:
        meeting = "Meeting recording: enabled, not armed (no meeting started)"
    screen = "Screen upload: allowed" if consent.screen_consent else "Screen upload: OFF (local-only)"
    qa = "History Q&A: cloud allowed (explicit consent)" if consent.cloud_qa_consent else "History Q&A: local-only"
    return [meeting, screen, qa]


def format_history_snapshot(snapshot: HistorySnapshot) -> str:
    """Full text rendering of a snapshot (window fallback, tests, logs)."""
    lines = format_consent_lines(snapshot.consent)
    lines.append(format_retention_line(snapshot.retention))
    if not snapshot.meetings:
        lines.append("No stored meetings.")
    else:
        lines.append(f"{len(snapshot.meetings)} stored meeting(s):")
        lines.extend(f"  {format_meeting_line(m)}" for m in snapshot.meetings)
    return "\n".join(lines)


def _iso_brief(epoch: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(epoch))
