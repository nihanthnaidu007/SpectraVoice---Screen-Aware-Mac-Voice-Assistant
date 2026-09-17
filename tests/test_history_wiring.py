"""W4 wiring tests — the load-bearing structural surfaces of the dashboard.

- View-model: snapshot assembly, consent/retention reflection, formatting.
- Darwin degradation: the history window imports LOUDLY off-darwin (the
  settings_window/menubar precedent) — a silent import would hide a broken
  dashboard on a real Mac.
- AST isolation: every W4 module stays free of network/LLM/screen-pipeline
  machinery; only history_window may carry AppKit (the allowed darwin
  surface) and even it must not import the orchestrator, HUD, or LLM.
- Menu structure: History… sits below the meeting controls in the menubar
  source and routes through actions.open_history (AppKit itself is
  unimportable on Linux CI, so the order is checked structurally).
"""

from __future__ import annotations

import ast
import os
import sys
import time
from pathlib import Path

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.hud.history_model import (
    ConsentState,
    HistorySnapshot,
    RetentionState,
    build_history_snapshot,
    format_consent_lines,
    format_duration,
    format_history_snapshot,
    format_meeting_line,
    format_retention_line,
    retention_state_from,
)
from assistant_app.services import meeting_store as ms

PROJECT_ROOT = Path(project_root)
SRC = PROJECT_ROOT / "src" / "assistant_app"


class _FakeRecordingConsent:
    def __init__(self, feature_enabled, armed):
        self.feature_enabled = feature_enabled
        self.armed = armed


class _FakePrivacyConsent:
    def __init__(self, allowed):
        self.screen_upload_allowed = allowed


class TestViewModel:
    def test_snapshot_reflects_all_three_gates(self, tmp_path):
        cfg = type("Cfg", (), {"audio_dir": str(tmp_path / "corpus"), "retention_hours": 0.0})()
        snapshot = build_history_snapshot(
            cfg,
            _FakeRecordingConsent(feature_enabled=True, armed=True),
            _FakePrivacyConsent(allowed=False),
        )
        assert snapshot.consent.meeting_kill_switch is True
        assert snapshot.consent.meeting_armed is True
        assert snapshot.consent.screen_consent is False
        assert snapshot.meetings == ()

    def test_live_meeting_marking_flows_through_the_model(self, tmp_path):
        cfg = type(
            "Cfg", (), {"audio_dir": str(tmp_path / "corpus"), "retention_hours": 0.0, "language": "english"}
        )()
        paths = ms.open_meeting(cfg, time.time())
        snapshot = build_history_snapshot(
            cfg,
            _FakeRecordingConsent(False, False),
            _FakePrivacyConsent(False),
            live_meeting_ids={paths.meeting_id},
        )
        assert len(snapshot.meetings) == 1
        assert snapshot.meetings[0]["is_recording"] is True

    def test_retention_state_carries_the_sweep_record(self):
        cfg = type("Cfg", (), {"retention_hours": 48.0})()
        state = retention_state_from(cfg, {"removed": 3, "at": 100.0})
        assert state == RetentionState(retention_hours=48.0, last_sweep_removed=3, last_sweep_at=100.0)
        assert retention_state_from(type("Cfg", (), {"retention_hours": 0.0})(), None) == RetentionState(
            retention_hours=0.0
        )

    def test_format_duration(self):
        assert format_duration(0) == "0s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(3661) == "1h 01m"
        assert format_duration(-5) == "0s"

    def test_format_meeting_line_marks_live_recordings(self):
        row = {
            "meeting_id": "20260917-120000-abc",
            "started_at_iso": "2026-09-17 12:00:00",
            "duration_seconds": 600.0,
            "utterance_count": 12,
            "gap_count": 2,
            "summary_present": True,
            "is_recording": True,
            "matched_snippets": [],
        }
        line = format_meeting_line(row)
        assert "20260917-120000-abc" in line
        assert "RECORDING" in line
        row["is_recording"] = False
        assert "RECORDING" not in format_meeting_line(row)

    def test_format_snapshot_reflects_consent_and_retention(self):
        snapshot = HistorySnapshot(
            meetings=(),
            retention=RetentionState(retention_hours=0.0, last_sweep_removed=2, last_sweep_at=1_000.0),
            consent=ConsentState(meeting_kill_switch=False, meeting_armed=False, screen_consent=True),
        )
        text = format_history_snapshot(snapshot)
        assert "No stored meetings" in text
        assert "forever" in text.lower()  # 0 = forever, made visible
        assert "kill-switch" in text.lower()  # the OFF state is stated, not hidden

    def test_consent_lines_state_all_three_gates(self):
        lines = format_consent_lines(ConsentState(True, False, False))
        assert len(lines) == 2  # recording line carries both kill-switch AND armed state
        assert "not armed" in lines[0].lower()
        assert "off" in lines[1].lower()

    def test_retention_line_mentions_last_sweep(self):
        line = format_retention_line(RetentionState(0.0, last_sweep_removed=4, last_sweep_at=1_000.0))
        assert "4" in line


class TestDarwinDegradation:
    def test_history_window_import_fails_loudly_off_darwin(self):
        # On Linux CI this proves the AppKit surface never imports silently;
        # on macOS the import succeeds and the test self-skips.
        if sys.platform == "darwin":
            import assistant_app.hud.history_window

            pytest.skip("AppKit available — degradation path is darwin-irrelevant")
        with pytest.raises(ImportError, match="darwin-only"):
            import assistant_app.hud.history_window  # noqa: F401


def module_imports(rel_path: str) -> list[str]:
    tree = ast.parse((SRC / rel_path).read_text(encoding="utf-8"), filename=str(rel_path))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


# Screen-aware pipeline, audio I/O, AppKit HUD machinery, hotkeys, LLM/network.
FORBIDDEN_PREFIXES = (
    "assistant_app.core.assistant",
    "assistant_app.io.",
    "assistant_app.hud.menubar",
    "assistant_app.hud.settings_window",
    "assistant_app.hud.history_window",
    "assistant_app.services.spectravoice_assistant",
    "assistant_app.services.transcription",
    "assistant_app.llm",
    "openai",
    "anthropic",
    "httpx",
    "requests",
    "socket",
    "urllib.request",
    "speech_recognition",
    "pynput",
)
# The allowed darwin surface: AppKit/Foundation UI plumbing only.
DARWIN_ALLOWED = ("AppKit", "Foundation", "Quartz")

W4_PURE_MODULES = (
    "services/history_search.py",
    "services/history_ops.py",
    "hud/history_model.py",
)


class TestHistoryIsolation:
    @pytest.mark.parametrize("rel_path", W4_PURE_MODULES)
    def test_pure_w4_modules_import_no_pipeline_or_appkit(self, rel_path):
        for module in module_imports(rel_path):
            for prefix in FORBIDDEN_PREFIXES:
                assert not module.startswith(prefix), (
                    f"{rel_path} imports {module} — W4 pure modules must stay free of "
                    "the screen-aware pipeline, LLM/network, and HUD machinery"
                )
            assert not module.startswith(("AppKit", "Quartz")), (
                f"{rel_path} imports {module} — only history_window may touch AppKit"
            )

    def test_history_window_is_appkit_only_on_the_darwin_surface(self):
        for module in module_imports("hud/history_window.py"):
            if module.startswith(DARWIN_ALLOWED):
                continue
            for prefix in FORBIDDEN_PREFIXES:
                assert not module.startswith(prefix), (
                    f"history_window imports {module} — the darwin window must not pull "
                    "the orchestrator, LLM/network, or other HUD machinery"
                )


class TestMenubarStructure:
    """AppKit is unimportable on Linux CI, so menu order is checked
    structurally: History… must sit below the meeting controls and above
    Settings, and its selector must dispatch actions.open_history."""

    def _source(self) -> str:
        return (SRC / "hud" / "menubar.py").read_text(encoding="utf-8")

    def test_history_item_sits_between_meeting_controls_and_settings(self):
        source = self._source()
        meeting_pos = source.index("meeting_item_label(None)")
        history_pos = source.index('"History')
        settings_pos = source.index('"Settings')
        assert meeting_pos < history_pos < settings_pos

    def test_history_selector_dispatches_open_history(self):
        tree = ast.parse(self._source())
        handlers = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "onHistory_"
        ]
        assert len(handlers) == 1
        body_source = ast.unparse(handlers[0])
        assert "open_history" in body_source
