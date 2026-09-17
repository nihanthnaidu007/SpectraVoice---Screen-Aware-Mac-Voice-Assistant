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

CONSENT_ENV_VAR = "SPECTRAVOICE_SCREEN_CONSENT"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


class PrivacyConsent:
    """Thread-safe gate controlling whether screen content may be uploaded."""

    def __init__(self, consented: bool = False):
        self._lock = threading.Lock()
        self._consented = consented
        self._paused = False

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

    def grant_consent(self) -> None:
        with self._lock:
            self._consented = True

    def revoke_consent(self) -> None:
        with self._lock:
            self._consented = False

    def pause(self) -> None:
        """Runtime toggle: block upload even while consent is granted."""
        with self._lock:
            self._paused = True

    def resume(self) -> None:
        """Clear the runtime pause (does not grant consent)."""
        with self._lock:
            self._paused = False
