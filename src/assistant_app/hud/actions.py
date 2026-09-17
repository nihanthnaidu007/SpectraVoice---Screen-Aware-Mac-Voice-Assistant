"""HUD action protocol — the menu actions the assistant implements (W2 D1).

Pure logic, importable on Linux CI. The AppKit HUD (``menubar.py``) calls
through these; the orchestrator wires them to the real behaviors. The HUD
only *controls* via these callbacks — consent semantics stay in the gate.
"""

from __future__ import annotations

from typing import Protocol


class HUDActions(Protocol):
    """Callbacks the MenuBarHUD invokes (on the AppKit main thread)."""

    def toggle_listening(self) -> None:
        """Start or stop the assistant's listen loop (mirrors the CLI flag)."""
        ...

    def toggle_pause(self) -> None:
        """Pause/resume screen sharing via the consent gate (never bypass it)."""
        ...

    def switch_dictation_mode(self) -> None:
        """Cycle the W1 dictation mode (push-to-talk <-> VAD) via config."""
        ...

    def toggle_meeting(self) -> None:
        """Explicit per-meeting start/stop (W3 D2) — refused without consent."""
        ...

    def pause_meeting(self) -> None:
        """Pause/resume the running meeting (skipped speech becomes a gap)."""
        ...

    def current_dictation_mode(self) -> str:
        """The active dictation mode, for the menu label."""
        ...

    def open_settings(self) -> None:
        """Open the native settings window (W2 D2)."""
        ...

    def quit(self) -> None:
        """Full shutdown: stop workers, persist state, terminate the app."""
        ...
