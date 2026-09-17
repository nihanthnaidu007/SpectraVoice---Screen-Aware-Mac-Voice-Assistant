"""Two-axis HUD state machine — pure logic, no AppKit (W2 D1).

Consolidates the assistant's scattered live-state surfaces (the legacy
indicator status strings, TTS state callbacks, transcription-worker activity,
and W1 dictation) into one thread-safe snapshot:

- **activity axis**: idle / listening / transcribing / thinking / speaking /
  dictating (the spec's quartet plus the two states the existing signals
  already report — `update_status("Thinking")` — and D1.3's dictation states).
- **privacy axis**: ON / PAUSED / OFF, derived from the PrivacyConsent gate.

State flows IN via callback updates only (no polling loop); observers are
notified when the effective snapshot changes. Rendering helpers turn a
snapshot into menu-bar text so the icon logic is unit-testable without AppKit.

Observer discipline mirrors PrivacyConsent: callbacks run outside the state
lock on the mutating thread, must be quick, and are failure-isolated — a
broken HUD callback can never take down the assistant's audio path.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from assistant_app.utils.logging_config import get_logger

StateObserver = Callable[["HUDSnapshot"], None]


class Activity(str, Enum):
    """What the assistant is doing right now (the pre-flight §4.2 mapping)."""

    IDLE = "idle"  # running but not listening (stopped via menu)
    LISTENING = "listening"  # mic live, no utterance in flight
    TRANSCRIBING = "transcribing"  # Whisper working on a captured utterance
    THINKING = "thinking"  # LLM round-trip
    SPEAKING = "speaking"  # TTS synthesis or playback
    DICTATING = "dictating"  # W1 dictation armed/inserting


class Privacy(str, Enum):
    """The three user-visible consent states (consent.py's truth table)."""

    ON = "ON"  # consented and not paused
    PAUSED = "PAUSED"  # consented but paused
    OFF = "OFF"  # not consented (pause is a no-op here)


def privacy_from_consent(consented: bool, paused: bool) -> Privacy:
    """Map the consent gate's two flags to the HUD's three privacy states."""
    if not consented:
        return Privacy.OFF
    return Privacy.PAUSED if paused else Privacy.ON


def activity_from_status(status: str) -> Activity:
    """Map the legacy indicator status strings to activity states."""
    return {
        "Listening": Activity.LISTENING,
        "Thinking": Activity.THINKING,
        "Speaking": Activity.SPEAKING,
    }.get(status, Activity.IDLE)


@dataclass(frozen=True)
class HUDSnapshot:
    """One consistent view of the two-axis state (rendered by the HUD)."""

    activity: Activity = Activity.IDLE
    privacy: Privacy = Privacy.OFF
    dictation: str | None = None  # None | "ptt-held" | "vad-active" | "inserting"

    @property
    def is_listening(self) -> bool:
        return self.activity == Activity.LISTENING


# Menu-bar icon per activity (kept as text glyphs — no image assets, renders
# in the system menu bar without AppKit-specific image plumbing).
ACTIVITY_GLYPHS: dict[Activity, str] = {
    Activity.IDLE: "○",
    Activity.LISTENING: "🎤",
    Activity.TRANSCRIBING: "⏳",
    Activity.THINKING: "💭",
    Activity.SPEAKING: "🔊",
    Activity.DICTATING: "✍️",
}

PRIVACY_GLYPHS: dict[Privacy, str] = {
    Privacy.ON: "",
    Privacy.PAUSED: "⏸",
    Privacy.OFF: "🚫",
}


def icon_text(snapshot: HUDSnapshot) -> str:
    """Compact menu-bar icon string for a snapshot."""
    return f"{ACTIVITY_GLYPHS[snapshot.activity]}{PRIVACY_GLYPHS[snapshot.privacy]}".strip()


def menu_summary(snapshot: HUDSnapshot) -> str:
    """Human-readable one-line state for the HUD menu header."""
    privacy_note = {
        Privacy.ON: "screen sharing on",
        Privacy.PAUSED: "screen sharing paused",
        Privacy.OFF: "screen sharing off (no consent)",
    }[snapshot.privacy]
    detail = f" ({snapshot.dictation.replace('-', ' ')})" if snapshot.dictation else ""
    return f"{snapshot.activity.value}{detail} · {privacy_note}"


class _Unset:
    """Sentinel for "leave this axis unchanged" — dictation's None is a real
    value (clears the sub-state), so it cannot double as the no-change marker."""


_UNSET = _Unset()


class HUDStateMachine:
    """Thread-safe two-axis state store with change observers."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot = HUDSnapshot()
        self._observers: list[StateObserver] = []

    # === reads ===

    def snapshot(self) -> HUDSnapshot:
        with self._lock:
            return self._snapshot

    # === writes (event-driven; called from assistant/hotkey/TTS threads) ===

    def set_activity(self, activity: Activity) -> None:
        self._publish(activity=activity)

    def set_privacy(self, consented: bool, paused: bool) -> None:
        """Update the privacy axis (wire directly to a PrivacyConsent observer)."""
        self._publish(privacy=privacy_from_consent(consented, paused))

    def set_dictation(self, state: str | None) -> None:
        """Update the dictation sub-state (None when dictation is inactive)."""
        self._publish(dictation=state)

    # === observers ===

    def add_observer(self, observer: StateObserver) -> None:
        with self._lock:
            self._observers.append(observer)
        # Never miss the current state: a late subscriber still renders.
        try:
            observer(self._current())
        except Exception:
            get_logger(__name__).exception("HUD observer raised on initial snapshot")

    def remove_observer(self, observer: StateObserver) -> None:
        with self._lock:
            try:
                self._observers.remove(observer)
            except ValueError:
                pass  # already removed — idempotent

    # === internals ===

    def _current(self) -> HUDSnapshot:
        with self._lock:
            return self._snapshot

    def _publish(
        self,
        activity: Activity | None = None,
        privacy: Privacy | None = None,
        dictation: str | None | _Unset = _UNSET,
    ) -> None:
        """Merge the given field updates, notify only when the snapshot changed.

        ``None`` means "clear this axis" (legitimate for the dictation
        sub-state); the ``_UNSET`` sentinel means "leave as-is".
        """
        with self._lock:
            current = self._snapshot
            updated = HUDSnapshot(
                activity=current.activity if activity is None else activity,
                privacy=current.privacy if privacy is None else privacy,
                dictation=current.dictation if isinstance(dictation, _Unset) else dictation,
            )
            changed = updated != current
            if changed:
                self._snapshot = updated
            observers = tuple(self._observers) if changed else ()

        for observer in observers:
            try:
                observer(updated)
            except Exception:
                get_logger(__name__).exception("HUD state observer raised")
