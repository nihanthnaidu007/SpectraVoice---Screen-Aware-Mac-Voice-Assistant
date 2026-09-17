"""Native menu-bar HUD (W2 D1) — raw PyObjC NSStatusItem, darwin-only.

Locked spec decision (art_uqHzj1pS / pre-flight R1-R2): the HUD uses a raw
NSStatusItem (no third-party framework) and *owns the main thread* by running
the shared NSApplication. Replaces the keep-alive loop in
``SpectraVoiceAssistant.run()`` on macOS.

Threading contract:
- The HUD is created on the main thread; ``run()`` blocks there running the
  NSApplication run loop (menu events, icon updates).
- State arrives via :class:`~assistant_app.hud.state.HUDStateMachine`
  observers, which fire on arbitrary assistant threads. Updates are marshaled
  to the main thread with ``performSelectorOnMainThread_`` — AppKit calls
  never happen on audio/TTS/hotkey threads (pre-flight R2).
- Menu actions dispatch to the orchestrator's ``HUDActions`` implementation;
  consent semantics stay in the gate — the HUD reflects and controls, never
  bypasses (it toggles pause through the gate like every other consumer).

Importing this module on a non-darwin platform raises ImportError with a
clear message (tests assert this explicitly — never a silent skip).
"""

from __future__ import annotations

try:
    from AppKit import (
        NSApplication,
        NSApplicationActivationPolicyAccessory,
        NSMenu,
        NSMenuItem,
        NSStatusBar,
        NSVariableStatusItemLength,
    )
    from Foundation import NSObject
except ImportError as exc:  # pragma: no cover - exercised on non-darwin only
    raise ImportError(
        "assistant_app.hud.menubar is darwin-only: AppKit/PyObjC is unavailable "
        "on this platform. The pure-logic HUD state machine "
        "(assistant_app.hud.state) has no such dependency."
    ) from exc

from assistant_app.hud.state import (
    HUDSnapshot,
    HUDStateMachine,
    Privacy,
    icon_text,
    listen_item_label,
    menu_summary,
    pause_item_enabled,
    pause_item_label,
)
from assistant_app.utils.logging_config import get_logger

logger = get_logger(__name__)

QUIT_KEY = "q"  # ⌘Q, standard quit shortcut


class MenuBarHUD(NSObject):
    """NSStatusItem HUD rendering the two-axis state machine live."""

    def initWithStateMachine_actions_(self, state_machine: HUDStateMachine, actions) -> MenuBarHUD | None:
        # PyObjC two-phase init: super().init() may return a different
        # instance (or None), so configure and return that object directly.
        hud = super().init()
        if hud is None:
            return None
        hud._state_machine = state_machine
        hud._actions = actions
        hud._listening_active = False  # orchestrator's listen loop (mirrors CLI flag)
        hud._build_status_item()
        hud._build_menu()
        # Late-attaching observer: the state machine immediately renders the
        # current state, so the icon is correct before any event arrives.
        state_machine.add_observer(hud._observe)
        return hud

    # === observer (any thread) → main thread ===

    def _observe(self, snapshot: HUDSnapshot) -> None:
        """State-machine observer: marshal the snapshot to the main thread."""
        self.performSelectorOnMainThread_withObject_waitUntilDone_(
            "applySnapshot:", snapshot, False
        )

    # === main-thread rendering ===

    def applySnapshot_(self, snapshot: HUDSnapshot) -> None:
        """Apply a snapshot to the icon and menu (main thread only)."""
        if self._button is not None:
            self._button.setTitle_(icon_text(snapshot))
        if self._header_item is not None:
            self._header_item.setTitle_(menu_summary(snapshot))
        if self._listen_item is not None:
            self._listening_active = snapshot.is_listening
            self._listen_item.setTitle_(listen_item_label(self._listening_active))
        if self._pause_item is not None:
            self._pause_item.setTitle_(pause_item_label(snapshot.privacy))
            self._pause_item.setEnabled_(pause_item_enabled(snapshot.privacy))

    # === NSStatusItem / menu construction (main thread) ===

    def _build_status_item(self) -> None:
        self._status_item = NSStatusBar.systemStatusBar().statusItemWithLength_(
            NSVariableStatusItemLength
        )
        self._button = self._status_item.button()
        if self._button is None:
            # Pre-10.10 fallback: title lives on the item itself.
            self._status_item.setTitle_(icon_text(HUDSnapshot()))
            self._button = None

    def _build_menu(self) -> None:
        menu = NSMenu.alloc().init()
        menu.setAutoenablesItems_(False)  # enabled flags are snapshot-driven

        self._header_item = self._add_item(menu, menu_summary(HUDSnapshot()), None)
        self._header_item.setEnabled_(False)
        menu.addItem_(NSMenuItem.separatorItem())

        self._listen_item = self._add_item(menu, listen_item_label(False), "onListenToggle:")
        self._pause_item = self._add_item(menu, pause_item_label(Privacy.OFF), "onPauseToggle:")
        self._dictation_item = self._add_item(
            menu, f"Dictation Mode: {self._actions.current_dictation_mode()}", "onDictationMode:"
        )
        menu.addItem_(NSMenuItem.separatorItem())

        self._add_item(menu, "Settings…", "onSettings_")
        quit_item = self._add_item(menu, "Quit SpectraVoice", "onQuit_")
        quit_item.setKeyEquivalent_(QUIT_KEY)

        self._menu = menu
        self._status_item.setMenu_(menu)

    def _add_item(self, menu, title: str, action: str | None) -> NSMenuItem:
        item = NSMenuItem.alloc().initWithTitle_action_keyEquivalent_(title, action, "")
        if action is not None:
            item.setTarget_(self)
        menu.addItem_(item)
        return item

    # === menu actions (main thread, via the actions protocol) ===

    def onListenToggle_(self, sender) -> None:
        self._actions.toggle_listening()

    def onPauseToggle_(self, sender) -> None:
        self._actions.toggle_pause()

    def onDictationMode_(self, sender) -> None:
        self._actions.switch_dictation_mode()
        # Refresh the label through the normal state path (mode is config, not
        # HUD state) — cheap direct update keeps the label honest immediately.
        if self._dictation_item is not None:
            self._dictation_item.setTitle_(f"Dictation Mode: {self._actions.current_dictation_mode()}")

    def onSettings_(self, sender) -> None:
        self._actions.open_settings()

    def onQuit_(self, sender) -> None:
        self._actions.quit()

    # === main-thread ownership ===

    def run(self) -> None:
        """Own the main thread: run the NSApplication loop (replaces the W1
        keep-alive loop). Blocks until :meth:`request_terminate`."""
        app = NSApplication.sharedApplication()
        app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)  # no Dock icon
        logger.info("Menu-bar HUD taking over the main thread")
        app.run()

    def request_terminate(self) -> None:
        """Stop the run loop; the process exits after ``run()`` returns."""
        NSApplication.sharedApplication().terminate_(None)
