"""RecordingConsent tests (W3 D2.4): default OFF, explicit start, kill-switch.

Sibling semantics to PrivacyConsent (test_privacy_consent.py): thread-safe
gate, change-only observer notifications, atomic snapshot.
"""

import os
import sys
import threading

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.core.consent import RecordingConsent


class TestDefaultOff:
    def test_defaults_refuse_recording(self):
        gate = RecordingConsent()
        assert gate.feature_enabled is False
        assert gate.armed is False
        assert gate.recording_allowed is False

    def test_default_construction_matches_privacy_consent_posture(self):
        """Both gates refuse by default — the shared posture for anything that
        captures content that is not the user's own dictation."""
        from assistant_app.core.consent import PrivacyConsent

        assert PrivacyConsent.from_env(env={}).screen_upload_allowed is False
        assert RecordingConsent().recording_allowed is False


class TestKillSwitch:
    def test_arm_refuses_while_kill_switch_engaged(self):
        gate = RecordingConsent(feature_enabled=False)
        assert gate.arm() is False
        assert gate.armed is False
        assert gate.recording_allowed is False

    def test_kill_switch_removes_capture_entirely(self):
        """The kill-switch is the unavailable surface: no per-meeting start can
        bypass it (that refusal is what makes it a switch, not a default)."""
        gate = RecordingConsent(feature_enabled=False)
        for _attempt in range(3):  # repeated starts all refuse
            assert gate.arm() is False
        assert gate.recording_allowed is False


class TestExplicitPerMeetingStart:
    def test_arm_allows_recording_when_feature_enabled(self):
        gate = RecordingConsent(feature_enabled=True)
        assert gate.arm() is True
        assert gate.recording_allowed is True

    def test_armed_requires_feature_enabled(self):
        """Kill-switch ON + arming = allowed; arming alone never suffices."""
        gate = RecordingConsent()
        gate.arm()
        assert gate.recording_allowed is False

    def test_disarm_revokes(self):
        gate = RecordingConsent(feature_enabled=True)
        gate.arm()
        gate.disarm()
        assert gate.recording_allowed is False
        assert gate.armed is False

    def test_kill_switch_off_reverts_allowed_state(self):
        gate = RecordingConsent(feature_enabled=True)
        gate.arm()
        assert gate.recording_allowed is True
        gate.set_feature_enabled(False)
        assert gate.recording_allowed is False


class TestSnapshot:
    def test_snapshot_is_atomic_and_correct(self):
        gate = RecordingConsent(feature_enabled=True)
        gate.arm()
        assert gate.snapshot() == (True, True, True)
        gate.disarm()
        assert gate.snapshot() == (True, False, False)

    def test_snapshot_default(self):
        assert RecordingConsent().snapshot() == (False, False, False)


class TestObservers:
    def test_observer_notified_on_arm_and_disarm(self):
        gate = RecordingConsent(feature_enabled=True)
        events: list[tuple[bool, bool]] = []
        gate.add_observer(lambda enabled, armed: events.append((enabled, armed)))
        gate.arm()
        gate.disarm()
        assert events == [(True, True), (True, False)]

    def test_redundant_arm_notifies_nothing(self):
        gate = RecordingConsent(feature_enabled=True)
        events: list[tuple[bool, bool]] = []
        gate.add_observer(lambda enabled, armed: events.append((enabled, armed)))
        gate.arm()
        gate.arm()  # no state change — no notification
        assert events == [(True, True)]

    def test_arm_refusal_notifies_nothing(self):
        gate = RecordingConsent(feature_enabled=False)
        events: list[tuple[bool, bool]] = []
        gate.add_observer(lambda enabled, armed: events.append((enabled, armed)))
        gate.arm()
        assert events == []

    def test_broken_observer_is_isolated(self):
        gate = RecordingConsent(feature_enabled=True)

        def broken(enabled, armed):
            raise RuntimeError("HUD callback bug")

        gate.add_observer(broken)
        try:
            assert gate.arm() is True  # the gate still functions
            assert gate.recording_allowed is True
        finally:
            gate.remove_observer(broken)

    def test_concurrent_arm_disarm_never_loses_updates(self):
        gate = RecordingConsent(feature_enabled=True)
        gate.arm()

        def toggle():
            for _ in range(200):
                gate.arm()
                gate.disarm()

        threads = [threading.Thread(target=toggle) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Ends consistent whatever the interleave: disarmed or armed, never torn.
        enabled, armed, allowed = gate.snapshot()
        assert allowed == (enabled and armed)
