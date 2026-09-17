"""Privacy consent gate for screen-content upload.

Screen captures contain whatever the user is looking at — passwords, messages,
email. SpectraVoice must never transmit that content to a cloud API unless the
user has explicitly opted in. This gate holds two independent switches:

- **consent** — sticky opt-in from configuration (``SPECTRAVOICE_SCREEN_CONSENT``
  environment variable). Default: **OFF**.
- **pause** — a runtime kill-switch that blocks upload even when consent was
  granted (intended to back a hotkey/voice "pause screen sharing" toggle).

Screen content may only leave the machine when consent is granted AND the
gate is not paused.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable

from assistant_app.utils.logging_config import get_logger

CONSENT_ENV_VAR = "SPECTRAVOICE_SCREEN_CONSENT"
_TRUTHY = frozenset({"1", "true", "yes", "on"})

# Observers receive (consented, paused) after each effective-state change.
ConsentObserver = Callable[[bool, bool], None]


class PrivacyConsent:
    """Thread-safe gate controlling whether screen content may be uploaded."""

    def __init__(self, consented: bool = False):
        self._lock = threading.Lock()
        self._consented = consented
        self._paused = False
        self._observers: list[ConsentObserver] = []

    @classmethod
    def from_env(cls, env: dict | None = None) -> PrivacyConsent:
        """Build the gate from environment configuration; consent defaults to OFF."""
        source = os.environ if env is None else env
        return cls(consented=source.get(CONSENT_ENV_VAR, "").strip().lower() in _TRUTHY)

    @property
    def screen_upload_allowed(self) -> bool:
        """True only when consent is granted and the gate is not paused."""
        with self._lock:
            return self._consented and not self._paused

    @property
    def consented(self) -> bool:
        with self._lock:
            return self._consented

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    def snapshot(self) -> tuple[bool, bool, bool]:
        """Atomic (consented, paused, screen_upload_allowed) read.

        Each property is individually atomic, but composing them across
        separate acquisitions races against concurrent mutations. The HUD and
        thread-safety tests need one consistent view under a single lock.
        """
        with self._lock:
            return self._consented, self._paused, self._consented and not self._paused

    # === observer hook (W2 HUD) ===
    # Observers fire after a mutation that changed the effective state
    # (consented, paused) — never on redundant grants/pauses. Callbacks run on
    # the mutating thread and must be quick and non-blocking; they run OUTSIDE
    # the gate lock, so they may read the gate's properties safely.

    def add_observer(self, observer: ConsentObserver) -> None:
        with self._lock:
            self._observers.append(observer)

    def remove_observer(self, observer: ConsentObserver) -> None:
        with self._lock:
            try:
                self._observers.remove(observer)
            except ValueError:
                pass  # already removed — idempotent

    def _mutate(self, consented: bool | None, paused: bool | None) -> None:
        """Atomically apply a mutation and notify observers on real change.

        ``None`` means "leave as-is"; the new state is computed INSIDE the lock
        so concurrent mutations never overwrite each other (no lost updates).
        Observers run outside the lock (they may read the gate's properties) —
        a broken observer must not corrupt the gate or kill the mutating thread
        (HUD callbacks, hotkeys, tests).
        """
        with self._lock:
            new_c = self._consented if consented is None else consented
            new_p = self._paused if paused is None else paused
            changed = (self._consented, self._paused) != (new_c, new_p)
            self._consented, self._paused = new_c, new_p
            observers = tuple(self._observers) if changed else ()
        for observer in observers:
            try:
                observer(new_c, new_p)
            except Exception:
                get_logger(__name__).exception("PrivacyConsent observer raised")

    def grant_consent(self) -> None:
        self._mutate(True, None)

    def revoke_consent(self) -> None:
        self._mutate(False, None)

    def pause(self) -> None:
        """Runtime toggle: block upload even while consent is granted."""
        self._mutate(None, True)

    def resume(self) -> None:
        """Clear the runtime pause (does not grant consent)."""
        self._mutate(None, False)


# Observers receive (feature_enabled, armed) after each effective-state change.
RecordingConsentObserver = Callable[[bool, bool], None]


class RecordingConsent:
    """Thread-safe gate for meeting audio recording (W3, sibling of PrivacyConsent).

    Two inputs compose, and BOTH default to OFF:

    - **feature_enabled** — the config kill-switch (``meeting.enabled``). When
      it is false, recording is impossible: the ``--meeting`` flag, the HUD
      item, and the hotkey all refuse. This is the switch an organization or a
      cautious user keeps off to make third-party capture unreachable.
    - **armed** — the explicit per-meeting start. Not sticky, not config-only:
      every recording requires a fresh, visible start action, and the armed
      state is exactly what the visible recording UI reflects.

    Recording may start only when feature_enabled AND armed. Observer
    discipline mirrors PrivacyConsent: callbacks fire on real state change,
    run outside the lock, and a broken observer can never corrupt the gate.
    """

    def __init__(self, feature_enabled: bool = False):
        self._lock = threading.Lock()
        self._feature_enabled = feature_enabled
        self._armed = False
        self._observers: list[RecordingConsentObserver] = []

    @property
    def recording_allowed(self) -> bool:
        """True only when the kill-switch is on AND this meeting was explicitly started."""
        with self._lock:
            return self._feature_enabled and self._armed

    @property
    def feature_enabled(self) -> bool:
        with self._lock:
            return self._feature_enabled

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    def snapshot(self) -> tuple[bool, bool, bool]:
        """Atomic (feature_enabled, armed, recording_allowed) read."""
        with self._lock:
            return self._feature_enabled, self._armed, self._feature_enabled and self._armed

    def arm(self) -> bool:
        """Explicit per-meeting start. Returns False (and changes nothing) when
        the config kill-switch is engaged — the refusal IS the kill-switch."""
        with self._lock:
            if not self._feature_enabled:
                return False
            changed = not self._armed
            self._armed = True
            observers = tuple(self._observers) if changed else ()
        for observer in observers:
            try:
                observer(self._feature_enabled, self._armed)
            except Exception:
                get_logger(__name__).exception("RecordingConsent observer raised")
        return True

    def disarm(self) -> None:
        """End the per-meeting arming (recording stops being allowed)."""
        self._mutate(armed=False)

    def set_feature_enabled(self, enabled: bool) -> None:
        """Update the kill-switch (config-derived; constructed ONCE from
        meeting.enabled at startup — this setter exists for tests and a future
        hot-apply wiring, not for bypassing config)."""
        self._mutate(feature_enabled=enabled)

    # === observer hook (same contract as PrivacyConsent) ===

    def add_observer(self, observer: RecordingConsentObserver) -> None:
        with self._lock:
            self._observers.append(observer)

    def remove_observer(self, observer: RecordingConsentObserver) -> None:
        with self._lock:
            try:
                self._observers.remove(observer)
            except ValueError:
                pass  # already removed — idempotent

    def _mutate(self, feature_enabled: bool | None = None, armed: bool | None = None) -> bool:
        """Apply a mutation and notify observers on real change.

        Returns whether the gate allows recording after the change. ``None``
        means "leave as-is"; the new state is computed INSIDE the lock so
        concurrent mutations never overwrite each other. Observers run outside
        the lock and are failure-isolated (HUD callbacks, hotkeys, tests).
        """
        with self._lock:
            new_e = self._feature_enabled if feature_enabled is None else feature_enabled
            new_a = self._armed if armed is None else armed
            changed = (self._feature_enabled, self._armed) != (new_e, new_a)
            self._feature_enabled, self._armed = new_e, new_a
            observers = tuple(self._observers) if changed else ()
        for observer in observers:
            try:
                observer(new_e, new_a)
            except Exception:
                get_logger(__name__).exception("RecordingConsent observer raised")
        return new_e and new_a
