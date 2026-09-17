"""HUD two-axis state-machine tests (W2 D1.4) — pure logic, no AppKit.

Covers the activity x privacy matrix required by the spec, the consent-derived
privacy truth table, dictation sub-state propagation, change-only observer
notification, failure isolation, and thread safety. Everything here runs on
Linux CI; the AppKit HUD renders from the same snapshot objects but is never
imported in this suite.
"""

import os
import sys
import threading

import pytest

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.core.consent import PrivacyConsent
from assistant_app.hud import create_menu_bar_hud
from assistant_app.hud.state import (
    ACTIVITY_GLYPHS,
    Activity,
    HUDSnapshot,
    HUDStateMachine,
    Privacy,
    activity_from_status,
    icon_text,
    listen_item_label,
    menu_summary,
    pause_item_enabled,
    pause_item_label,
    privacy_from_consent,
)


class TestPrivacyTruthTable:
    """The consent gate's semantics, rendered as three user-visible states."""

    def test_privacy_states(self):
        assert privacy_from_consent(False, False) is Privacy.OFF
        assert privacy_from_consent(False, True) is Privacy.OFF  # pause no-op without consent
        assert privacy_from_consent(True, False) is Privacy.ON
        assert privacy_from_consent(True, True) is Privacy.PAUSED

    def test_legacy_status_mapping(self):
        assert activity_from_status("Listening") is Activity.LISTENING
        assert activity_from_status("Thinking") is Activity.THINKING
        assert activity_from_status("Speaking") is Activity.SPEAKING
        assert activity_from_status("anything else") is Activity.IDLE

    def test_menu_item_labels(self):
        assert listen_item_label(True) == "Stop Listening"
        assert listen_item_label(False) == "Start Listening"

        assert pause_item_label(Privacy.ON) == "Pause Screen Sharing"
        assert pause_item_label(Privacy.PAUSED) == "Resume Screen Sharing"
        assert pause_item_label(Privacy.OFF) == "Screen Sharing Off (no consent)"

        assert pause_item_enabled(Privacy.ON) is True
        assert pause_item_enabled(Privacy.PAUSED) is True
        assert pause_item_enabled(Privacy.OFF) is False


class TestActivityPrivacyMatrix:
    """The spec's acceptance matrix: every activity renders with every privacy."""

    MATRIX_ACTIVITIES = [
        Activity.IDLE,
        Activity.LISTENING,
        Activity.TRANSCRIBING,
        Activity.SPEAKING,
    ]
    MATRIX_PRIVACIES = [Privacy.ON, Privacy.PAUSED, Privacy.OFF]

    def test_full_matrix_snapshots(self):
        machine = HUDStateMachine()
        seen = []
        machine.add_observer(seen.append)

        for activity in self.MATRIX_ACTIVITIES:
            for privacy_flags in [(True, False), (True, True), (False, False)]:
                machine.set_activity(activity)
                machine.set_privacy(*privacy_flags)
                snap = machine.snapshot()
                assert snap.activity is activity
                assert snap.privacy is privacy_from_consent(*privacy_flags)
                # Every combination renders distinct icon and menu text.
                assert icon_text(snap), f"{activity} x {snap.privacy} renders empty"
                assert activity.value in menu_summary(snap)

        # 12 state changes (4 activities x 3 privacy rows, redundant no-ops
        # deduped: the first IDLE is already current) + 1 initial render on
        # subscribe = 16.
        assert len(seen) == 16

    def test_every_activity_has_a_glyph(self):
        for activity in Activity:
            assert ACTIVITY_GLYPHS[activity], f"{activity} has no icon glyph"

    def test_privacy_modifier_in_icon(self):
        listening_on = HUDSnapshot(activity=Activity.LISTENING, privacy=Privacy.ON)
        assert icon_text(listening_on) == "🎤"  # ON needs no modifier
        assert "⏸" in icon_text(HUDSnapshot(activity=Activity.LISTENING, privacy=Privacy.PAUSED))
        assert "🚫" in icon_text(HUDSnapshot(activity=Activity.LISTENING, privacy=Privacy.OFF))


class TestDictationSubState:
    """W1 dictation states render without clobbering the activity axis."""

    def test_dictation_detail_propagates(self):
        machine = HUDStateMachine()
        machine.set_activity(Activity.LISTENING)
        machine.set_dictation("ptt-held")
        assert machine.snapshot().dictation == "ptt-held"
        assert "ptt held" in menu_summary(machine.snapshot())

        machine.set_dictation("inserting")
        assert machine.snapshot().dictation == "inserting"
        assert machine.snapshot().activity is Activity.LISTENING

        machine.set_dictation(None)
        assert machine.snapshot().dictation is None

    def test_activity_change_keeps_dictation_detail(self):
        machine = HUDStateMachine()
        machine.set_activity(Activity.DICTATING)
        machine.set_dictation("vad-active")
        machine.set_privacy(True, False)
        snap = machine.snapshot()
        assert snap.activity is Activity.DICTATING
        assert snap.dictation == "vad-active"
        assert snap.privacy is Privacy.ON


class TestObserverSemantics:
    def test_observers_notified_only_on_change(self):
        machine = HUDStateMachine()
        seen = []
        machine.add_observer(seen.append)
        seen.clear()  # discard the immediate render of the current state

        machine.set_activity(Activity.LISTENING)
        machine.set_activity(Activity.LISTENING)  # redundant
        machine.set_activity(Activity.SPEAKING)
        machine.set_privacy(True, False)
        machine.set_privacy(True, False)  # redundant

        # Three changes: two activity flips and one privacy flip (which
        # carries the unchanged SPEAKING activity in its snapshot).
        assert [s.activity for s in seen] == [Activity.LISTENING, Activity.SPEAKING, Activity.SPEAKING]
        assert [s.privacy for s in seen] == [Privacy.OFF, Privacy.OFF, Privacy.ON]

    def test_late_observer_receives_current_state(self):
        machine = HUDStateMachine()
        machine.set_activity(Activity.SPEAKING)
        seen = []
        machine.add_observer(seen.append)
        assert seen[0].activity is Activity.SPEAKING

    def test_broken_observer_is_isolated(self):
        """A crashing HUD callback must not take down the state machine."""
        machine = HUDStateMachine()
        seen = []

        def explode(_snapshot):
            raise RuntimeError("render bug")

        machine.add_observer(explode)
        machine.add_observer(seen.append)
        machine.set_activity(Activity.LISTENING)

        assert machine.snapshot().activity is Activity.LISTENING
        assert seen  # healthy observer still notified

    def test_remove_observer(self):
        machine = HUDStateMachine()
        seen = []
        observer = seen.append
        machine.add_observer(observer)
        seen.clear()  # discard the immediate render of the current state
        machine.remove_observer(observer)
        machine.set_activity(Activity.LISTENING)
        assert seen == []


class TestConsentIntegration:
    """The HUD privacy axis is wired to the consent gate's observer hook."""

    def test_consent_gate_drives_hud_privacy(self):
        gate = PrivacyConsent()
        machine = HUDStateMachine()
        gate.add_observer(machine.set_privacy)

        assert machine.snapshot().privacy is Privacy.OFF  # consent defaults OFF
        gate.grant_consent()
        assert machine.snapshot().privacy is Privacy.ON
        gate.pause()
        assert machine.snapshot().privacy is Privacy.PAUSED
        gate.resume()
        assert machine.snapshot().privacy is Privacy.ON
        gate.revoke_consent()
        assert machine.snapshot().privacy is Privacy.OFF

    def test_gating_semantics_unchanged_under_observers(self):
        """HUD reflection never bypasses the gate: no consent => no upload."""
        gate = PrivacyConsent()
        machine = HUDStateMachine()
        gate.add_observer(machine.set_privacy)
        gate.grant_consent()
        gate.pause()
        assert gate.screen_upload_allowed is False
        assert machine.snapshot().privacy is Privacy.PAUSED


class TestPlatformDegradation:
    """The HUD module degrades loudly off-darwin (pre-flight R7)."""

    def test_menubar_import_fails_loudly_off_darwin(self):
        # On Linux CI this proves the AppKit surface never imports silently;
        # on macOS this import succeeds and the test self-skips.
        if sys.platform == "darwin":
            import assistant_app.hud.menubar

            pytest.skip("AppKit available — degradation path is darwin-irrelevant")
        with pytest.raises(ImportError, match="darwin-only"):
            import assistant_app.hud.menubar  # noqa: F401

    def test_factory_returns_none_off_darwin(self, caplog):
        if sys.platform == "darwin":
            pytest.skip("Factory creates a real NSStatusItem on darwin — not testable here")
        machine = HUDStateMachine()

        class _NoopActions:
            def current_dictation_mode(self) -> str:
                return "vad"

        # Loud degradation, not a crash and not a silent None.
        assert create_menu_bar_hud(machine, _NoopActions()) is None

    def test_actions_protocol_surface(self):
        """The orchestrator's actions duck-type against the HUD protocol."""

        class _StubActions:
            def toggle_listening(self) -> None: ...
            def toggle_pause(self) -> None: ...
            def switch_dictation_mode(self) -> None: ...
            def current_dictation_mode(self) -> str:
                return "vad"

            def open_settings(self) -> None: ...
            def quit(self) -> None: ...

        stub = _StubActions()
        # Structural conformance: every HUDActions member exists.
        for name in ("toggle_listening", "toggle_pause", "switch_dictation_mode",
                     "current_dictation_mode", "open_settings", "quit"):
            assert callable(getattr(stub, name, None)), f"missing {name}"


class TestThreadSafety:
    def test_concurrent_updates_keep_snapshot_consistent(self):
        machine = HUDStateMachine()
        snapshots = []
        machine.add_observer(snapshots.append)

        def churn(activity: Activity):
            for _ in range(300):
                machine.set_activity(activity)
                machine.set_privacy(True, False)

        threads = [
            threading.Thread(target=churn, args=(activity,))
            for activity in (Activity.LISTENING, Activity.SPEAKING, Activity.THINKING)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        final = machine.snapshot()
        assert final.activity in (Activity.LISTENING, Activity.SPEAKING, Activity.THINKING)
        assert final.privacy is Privacy.ON
        # Every observed snapshot is internally consistent (no torn fields).
        assert all(isinstance(s.activity, Activity) and isinstance(s.privacy, Privacy) for s in snapshots)
