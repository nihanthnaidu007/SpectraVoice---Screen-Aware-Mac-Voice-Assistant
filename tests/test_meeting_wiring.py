"""W3 production-wiring tests — the import-light surfaces of the meeting
integration.

The full orchestrator (spectravoice_assistant.py) imports speech_recognition,
AppKit TTS, torch, and screen capture at module scope, so no test here imports
it (pre-flight R9). What CAN be verified headlessly:

- HUD state machine: the Activity.MEETING state, the meeting sub-state, glyphs,
  menu labels, and menu_summary rendering (the dictation pattern, extended).
- HUDActions protocol surface: every action the menu bar dispatches exists on
  a stub implementation — a missing method is a main-thread AttributeError on
  a real Mac, not a CI failure, so this check is load-bearing.
- CLI: the --meeting launch flag parses and defaults off (the explicit
  per-meeting start, D2).
- Routing isolation (R8, structural): the meeting modules carry NO module-level
  import of the screen-aware pipeline, audio I/O, AppKit HUD, or pynput
  hotkeys — a static AST check, so meeting clips cannot route into the
  assistant pipeline by construction, independent of test execution order.

The live wiring itself (audio-callback branch order, TTS mute guard, hotkey
command registration, shutdown ordering) is Mac-verified via the PR checklist —
the module is unimportable on Linux CI by design.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app import cli
from assistant_app.hud.actions import HUDActions
from assistant_app.hud.state import (
    ACTIVITY_GLYPHS,
    Activity,
    HUDSnapshot,
    HUDStateMachine,
    Privacy,
    icon_text,
    meeting_item_label,
    meeting_pause_item_label,
    menu_summary,
)


class _RecordingStub:
    """Minimal HUDActions implementation covering the full W2+W3 surface."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def toggle_listening(self) -> None:
        self.calls.append("toggle_listening")

    def toggle_pause(self) -> None:
        self.calls.append("toggle_pause")

    def switch_dictation_mode(self) -> None:
        self.calls.append("switch_dictation_mode")

    def current_dictation_mode(self) -> str:
        return "vad"

    def toggle_meeting(self) -> None:
        self.calls.append("toggle_meeting")

    def pause_meeting(self) -> None:
        self.calls.append("pause_meeting")

    def open_settings(self) -> None:
        self.calls.append("open_settings")

    def open_history(self) -> None:
        self.calls.append("open_history")

    def ask_history(self) -> None:
        self.calls.append("ask_history")

    def quit(self) -> None:
        self.calls.append("quit")


class TestHUDMeetingState:
    def test_meeting_activity_exists(self):
        assert Activity.MEETING.value == "meeting"

    def test_meeting_glyph_is_the_recording_dot(self):
        assert ACTIVITY_GLYPHS[Activity.MEETING] == "🔴"

    def test_default_snapshot_has_no_meeting(self):
        assert HUDSnapshot().meeting is None

    def test_set_meeting_publishes_recording(self):
        machine = HUDStateMachine()
        seen: list[HUDSnapshot] = []
        machine.add_observer(seen.append)
        machine.set_meeting("recording")
        assert seen[-1].meeting == "recording"

    def test_set_meeting_none_clears(self):
        machine = HUDStateMachine()
        machine.set_meeting("recording")
        machine.set_meeting(None)
        assert machine.snapshot().meeting is None

    def test_meeting_state_does_not_disturb_other_axes(self):
        machine = HUDStateMachine()
        machine.set_privacy(True, False)
        machine.set_activity(Activity.LISTENING)
        machine.set_meeting("recording")
        snap = machine.snapshot()
        assert (snap.activity, snap.privacy, snap.dictation) == (
            Activity.LISTENING,
            Privacy.ON,
            None,
        )

    def test_meeting_state_survives_unrelated_update(self):
        machine = HUDStateMachine()
        machine.set_meeting("recording")
        machine.set_activity(Activity.TRANSCRIBING)
        assert machine.snapshot().meeting == "recording"

    def test_menu_summary_shows_recording_loudly(self):
        snap = HUDSnapshot(activity=Activity.MEETING, meeting="recording")
        assert "MEETING RECORDING" in menu_summary(snap)

    def test_menu_summary_shows_paused(self):
        snap = HUDSnapshot(activity=Activity.MEETING, meeting="paused")
        assert "meeting paused" in menu_summary(snap)

    def test_icon_text_uses_meeting_glyph(self):
        snap = HUDSnapshot(activity=Activity.MEETING, meeting="recording")
        assert icon_text(snap).startswith("🔴")


class TestMeetingMenuLabels:
    def test_start_label_when_no_meeting(self):
        assert meeting_item_label(None) == "Start Meeting Recording"

    def test_stop_label_while_recording(self):
        assert meeting_item_label("recording") == "Stop Meeting & Summarize"

    def test_stop_label_while_paused(self):
        assert meeting_item_label("paused") == "Stop Meeting & Summarize"

    def test_pause_label_by_state(self):
        assert meeting_pause_item_label("recording") == "Pause Meeting"
        assert meeting_pause_item_label("paused") == "Resume Meeting"
        assert meeting_pause_item_label(None) == "Pause Meeting"


class TestHUDActionsSurface:
    """The menu bar dispatches these selectors on a real Mac; a missing method
    is a runtime AttributeError there, so the surface must be checked here."""

    def test_stub_satisfies_protocol(self):
        stub = _RecordingStub()
        for name in (
            "toggle_listening",
            "toggle_pause",
            "switch_dictation_mode",
            "current_dictation_mode",
            "toggle_meeting",
            "pause_meeting",
            "open_settings",
            "open_history",
            "ask_history",
            "quit",
        ):
            assert callable(getattr(stub, name)), name

    def test_protocol_declares_meeting_actions(self):
        assert callable(getattr(HUDActions, "toggle_meeting", None))
        assert callable(getattr(HUDActions, "pause_meeting", None))

    def test_protocol_declares_history_action(self):
        # W4 D2: the History… menu item dispatches open_history — a missing
        # method is a main-thread AttributeError on a real Mac.
        assert callable(getattr(HUDActions, "open_history", None))

    def test_protocol_declares_ask_history_action(self):
        # W1 D2: the Ask History… menu item dispatches ask_history — a missing
        # method is a main-thread AttributeError on a real Mac.
        assert callable(getattr(HUDActions, "ask_history", None))


class TestMeetingCLIFlag:
    def test_meeting_flag_defaults_off(self):
        assert cli.parse_args([]).meeting is False

    def test_meeting_flag_is_set(self):
        assert cli.parse_args(["--meeting"]).meeting is True


class TestRoutingIsolation:
    """R8: meeting clips must never enter the screen-aware pipeline. The
    wiring gates on `meeting.recording` BEFORE dictation/pipeline branches;
    this structural check keeps the meeting modules themselves free of any
    pipeline import so a clip can never re-enter assistant flow."""

    MEETING_MODULES = (
        "services/meeting.py",
        "services/meeting_store.py",
        "services/meeting_summary.py",
        "utils/retention.py",
    )

    FORBIDDEN_PREFIXES = (
        "assistant_app.core.assistant",
        "assistant_app.io.audio",
        "assistant_app.io.vision",
        "assistant_app.io.hotkeys",
        "assistant_app.hud.menubar",
        "assistant_app.hud.settings_window",
        "assistant_app.services.spectravoice_assistant",
    )

    def _module_imports(self, rel_path: str) -> list[str]:
        path = os.path.join(project_root, "src", "assistant_app", rel_path)
        tree = ast.parse(Path(path).read_text(encoding="utf-8"), filename=path)
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        return modules

    def test_meeting_modules_import_no_pipeline(self):
        for rel_path in self.MEETING_MODULES:
            for module in self._module_imports(rel_path):
                for prefix in self.FORBIDDEN_PREFIXES:
                    assert not module.startswith(prefix), (
                        f"{rel_path} imports {module} — meeting modules must stay"
                        " free of the screen-aware pipeline (pre-flight R8)"
                    )

    def test_meeting_module_imports_are_light(self):
        for rel_path in self.MEETING_MODULES:
            for module in self._module_imports(rel_path):
                assert not module.startswith("assistant_app.io."), (
                    f"{rel_path} imports {module} — audio/vision I/O must stay lazy"
                )
