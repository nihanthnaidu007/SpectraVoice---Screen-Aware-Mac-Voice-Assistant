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

    def toggle_screen_consent(self) -> None:
        """Grant/revoke screen consent via the gate (W2 S1: consent you can
        see) — the in-UI path to the same consent every consumer reads."""
        ...

    def current_screen_consent(self) -> bool:
        """Whether screen consent is currently granted, for the menu label."""
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

    def dictation_enabled(self) -> bool:
        """Whether dictation is currently enabled, for the menu label (W2 S3)."""
        ...

    def toggle_dictation_enabled(self) -> None:
        """Enable/disable dictation at runtime — no relaunch (W2 S3).

        The change flows through the config system (persisted + applied
        live); pynput stays the sole hotkey path.
        """
        ...

    def open_settings(self) -> None:
        """Open the native settings window (W2 D2)."""
        ...

    def open_history(self) -> None:
        """Open the history / privacy dashboard window (W4 D2)."""
        ...

    def ask_history(self) -> None:
        """Open the history window focused on the Ask input (W1 D2)."""
        ...

    def quit(self) -> None:
        """Full shutdown: stop workers, persist state, terminate the app."""
        ...
