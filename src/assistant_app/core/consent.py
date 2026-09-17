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
