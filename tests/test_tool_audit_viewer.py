"""W2 S4 tests — tool-audit viewer in the history/privacy dashboard.

The executor already records every tool action in an in-memory log
(timestamp, tool, arguments, dry-run, result, message/error); the dashboard
never rendered it. S4 renders it as a pure-model section: normalize the raw
dicts into a frozen ToolAuditState, format lines, and let the window append
them in applySnapshot_. The log is in-memory and session-scoped — the copy
must say so (a viewer that implies persistence would lie, pre-flight H4).

Linux-safe throughout: the audit model is pure (no AppKit); the window and
orchestrator seams are pinned structurally (menubar-test standard).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.hud.history_model import (
    TOOL_AUDIT_LIMIT,
    HistorySnapshot,
    ToolAuditState,
    build_history_snapshot,
    format_history_snapshot,
    format_tool_audit_lines,
    tool_audit_state_from,
)

SRC = Path(project_root) / "src" / "assistant_app"

NOW = 1_000_000.0


def _entry(**overrides):
    base = {
        "timestamp": NOW - 5,
        "tool": "open_url",
        "arguments": {"url": "https://example.com"},
        "dry_run": False,
        "result": "success",
        "message": "Opened https://example.com",
    }
    base.update(overrides)
    return base


class TestNormalization:
    def test_empty_and_none_inputs_yield_empty_state(self):
        state = tool_audit_state_from([])
        assert state.entries == ()
        assert state.total == 0
        assert tool_audit_state_from(None).total == 0

    def test_full_entry_is_normalized(self):
        state = tool_audit_state_from([_entry()])
        assert state.total == 1
        entry = state.entries[0]
        assert entry.tool == "open_url"
        assert entry.result == "success"
        assert entry.message == "Opened https://example.com"
        assert entry.dry_run is False
        assert entry.arguments == {"url": "https://example.com"}

    def test_arguments_are_copied_not_aliased(self):
        raw = _entry()
        state = tool_audit_state_from([raw])
        raw["arguments"]["url"] = "mutated"
        assert state.entries[0].arguments == {"url": "https://example.com"}

    def test_malformed_entries_never_raise(self):
        # Non-dict junk is skipped outright; dict entries normalize defensively.
        state = tool_audit_state_from(
            [
                "not-a-dict",
                None,
                {"timestamp": "garbage", "tool": "x"},  # bad timestamp -> 0.0
                {"tool": "no_args", "arguments": None},
                {"tool": "bad_msg", "message": 42, "error": 7},
            ]
        )
        assert state.total == 3  # the two non-dict rows are dropped
        assert state.entries[0].timestamp == 0.0
        assert state.entries[1].arguments == {}
        assert state.entries[2].message is None and state.entries[2].error is None

    def test_newest_first_regardless_of_log_order(self):
        state = tool_audit_state_from(
            [
                _entry(timestamp=NOW - 100, tool="older"),
                _entry(timestamp=NOW - 1, tool="newest"),
                _entry(timestamp=NOW - 50, tool="middle"),
            ]
        )
        assert [e.tool for e in state.entries] == ["newest", "middle", "older"]

    def test_limit_caps_stored_tail_but_total_counts_everything(self):
        raw = [_entry(timestamp=NOW - i, tool=f"t{i}") for i in range(TOOL_AUDIT_LIMIT + 5)]
        state = tool_audit_state_from(raw)
        assert state.total == TOOL_AUDIT_LIMIT + 5
        assert len(state.entries) == TOOL_AUDIT_LIMIT
        assert state.visible[0].tool == "t0"  # newest kept


class TestFormatting:
    def test_empty_log_explains_in_memory_session_scope(self):
        lines = format_tool_audit_lines(ToolAuditState())
        assert len(lines) == 1
        assert "none yet" in lines[0]
        assert "in-memory" in lines[0] and "this session" in lines[0]

    def test_success_line_shows_age_tool_and_message(self):
        lines = format_tool_audit_lines(tool_audit_state_from([_entry()]), now_fn=lambda: NOW)
        assert len(lines) == 2
        assert lines[1] == "• 5s ago — open_url — success — Opened https://example.com"

    def test_dry_run_is_labeled(self):
        lines = format_tool_audit_lines(
            tool_audit_state_from([_entry(dry_run=True, result="dry_run", message=None)]),
            now_fn=lambda: NOW,
        )
        assert "[DRY-RUN]" in lines[1]

    def test_error_path_shows_error_text(self):
        lines = format_tool_audit_lines(
            tool_audit_state_from([_entry(result="error", message=None, error="connection refused")]),
            now_fn=lambda: NOW,
        )
        assert "connection refused" in lines[1]

    def test_missing_message_falls_back_to_argument_summary(self):
        lines = format_tool_audit_lines(
            tool_audit_state_from([_entry(message=None)]), now_fn=lambda: NOW
        )
        assert '{"url": "https://example.com"}' in lines[1]

    def test_circular_arguments_fall_back_to_str(self):
        # default=str survives most oddities; a circular ref is the case that
        # actually raises — rendering must still not crash.
        blob: dict = {"x": {1, 2}}  # default=str renders the set
        lines = format_tool_audit_lines(
            tool_audit_state_from([_entry(message=None, arguments=blob)]),
            now_fn=lambda: NOW,
        )
        assert "{1, 2}" in lines[1]

        circular: dict = {}
        circular["self"] = circular
        lines = format_tool_audit_lines(
            tool_audit_state_from([_entry(message=None, arguments=circular)]),
            now_fn=lambda: NOW,
        )
        assert lines[1].startswith("• 5s ago — open_url — success —")

    def test_long_argument_summaries_are_truncated(self):
        lines = format_tool_audit_lines(
            tool_audit_state_from([_entry(message=None, arguments={"blob": "x" * 300})]),
            now_fn=lambda: NOW,
        )
        assert len(lines[1]) < 200 and lines[1].endswith("…")

    def test_age_buckets(self):
        def line(age: float) -> str:
            return format_tool_audit_lines(
                tool_audit_state_from([_entry(timestamp=NOW - age)]), now_fn=lambda: NOW
            )[1]

        assert line(2.4).startswith("• 2s ago")
        assert line(90).startswith("• 2m ago")
        assert line(7200).startswith("• 2h ago")

    def test_hidden_tail_is_disclosed(self):
        raw = [_entry(timestamp=NOW - i, tool=f"t{i}") for i in range(TOOL_AUDIT_LIMIT + 3)]
        lines = format_tool_audit_lines(tool_audit_state_from(raw), now_fn=lambda: NOW)
        assert lines[-1].startswith("… 3 earlier action(s) not shown")


class TestSnapshotIntegration:
    def _cfg(self, tmp_path):
        return type("Cfg", (), {"audio_dir": str(tmp_path / "corpus"), "retention_hours": 0.0})()

    def test_default_snapshot_carries_an_empty_audit(self):
        # RetentionState requires retention_hours, so construct with explicit
        # gates; tool_audit is omitted and must default to empty.
        from assistant_app.hud.history_model import ConsentState, RetentionState

        snapshot = HistorySnapshot(
            retention=RetentionState(retention_hours=0.0),
            consent=ConsentState(meeting_kill_switch=False, meeting_armed=False, screen_consent=False),
        )
        assert snapshot.tool_audit.total == 0
        assert snapshot.tool_audit.entries == ()

    def test_tool_log_flows_into_the_snapshot(self, tmp_path):
        class _Rec:
            feature_enabled = True
            armed = False

        class _Priv:
            screen_upload_allowed = False

        snapshot = build_history_snapshot(
            self._cfg(tmp_path),
            _Rec(),
            _Priv(),
            tool_log=[_entry()],
        )
        assert snapshot.tool_audit.total == 1
        assert snapshot.tool_audit.entries[0].tool == "open_url"

    def test_format_tool_audit_lines_joins_the_full_render(self, tmp_path):
        """The dashboard text contains the audit section when a log exists."""
        class _Rec:
            feature_enabled = True
            armed = False

        class _Priv:
            screen_upload_allowed = False

        snapshot = build_history_snapshot(
            self._cfg(tmp_path), _Rec(), _Priv(), tool_log=[_entry()], now_fn=lambda: NOW
        )
        header = format_history_snapshot(HistorySnapshot(meetings=(), retention=snapshot.retention, consent=snapshot.consent))
        text = "\n\n".join([header, *format_tool_audit_lines(snapshot.tool_audit, now_fn=lambda: NOW)])
        assert "Tool activity — last 1 of 1 this session" in text
        assert "Opened https://example.com" in text


class TestWiringStructure:
    """spectravoice_assistant.py and history_window.py are AppKit modules —
    the seams are pinned structurally (menubar-test standard)."""

    def test_window_appends_audit_lines_in_the_snapshot_render(self):
        source = (SRC / "hud" / "history_window.py").read_text(encoding="utf-8")
        assert "format_tool_audit_lines(snapshot.tool_audit)" in source
        # the placeholder path is untouched — audit only renders with a snapshot
        assert source.count("format_tool_audit_lines") == 2  # import + call

    def test_orchestrator_feeds_the_executor_log_into_the_snapshot(self):
        source = (SRC / "services" / "spectravoice_assistant.py").read_text(encoding="utf-8")
        assert "tool_log=self.get_tool_execution_log()" in source

    def test_model_stays_appkit_free(self):
        source = (SRC / "hud" / "history_model.py").read_text(encoding="utf-8")
        assert "import AppKit" not in source
        assert "from AppKit" not in source
        assert "import objc" not in source
