"""Global-hotkey listener — the single global-key input path (W2 D3).

Locked W2 decision (spec art_uqHzj1pS, D3): retain W1's pynput keyboard hook
as the ONE global-hotkey input path rather than migrating to a Quartz
CGEventTap. Rationale: pynput already ships (no new dependency — the wave
forbids input libraries beyond the darwin-pinned PyObjC), it degrades
gracefully when Accessibility permission is absent (``start()`` reports
False and the doctor surfaces the degradation), and a CGEventTap migration
would add a second input path without a behavior gain — both hooks need the
same permission for global capture.

W2 scope (D3): every configured hotkey fires through this listener —
dictation push-to-talk + mode toggle (W1) and the app commands mute_toggle /
pause_resume / quit (W2) — with chord-aware matching: a combo fires only when
ALL of its keys are simultaneously down (fixes W1's false trigger when a
single modifier belonging to a multi-key combo was pressed). PTT hold/cancel
semantics are preserved: ``on_ptt_press`` when the chord completes,
``on_ptt_release`` when any chord key releases.

Key events are dispatched on the pynput callback thread and must never block:
they only touch lock-guarded state and invoke registered callbacks.

Dependency-light at import time: pynput is imported inside the listener
class, never at module import, keeping unit tests dependency-free.
"""

import threading

from assistant_app.utils.logging_config import get_logger


def parse_hotkey(spec: str) -> frozenset[str] | None:
    """Parse a hotkey string like "cmd+shift+d" into a normalized key set.

    Accepts the existing config.yaml hotkey conventions (lowercase, ``+``
    separated). Returns None for empty/unparseable specs.
    """
    if not spec or not isinstance(spec, str):
        return None
    aliases = {
        "ctrl": "ctrl",
        "control": "ctrl",
        "cmd": "cmd",
        "command": "cmd",
        "alt": "alt",
        "option": "alt",
        "opt": "alt",
        "shift": "shift",
    }
    keys: list[str] = []
    for token in spec.casefold().split("+"):
        token = token.strip()
        if not token:
            return None
        keys.append(aliases.get(token, token))
    return frozenset(keys)


class DictationHotkeyListener:
    """Background global-key listener driving dictation + app commands.

    Constructor takes the dictation controller (W1: ``on_ptt_press`` /
    ``on_ptt_release`` / ``on_mode_toggle``) plus optional W2 app commands,
    ``{name: (key_set, callback)}``, dispatched once per chord completion.

    Threading contract: pynput callbacks run on their own thread; they only
    touch lock-guarded state and never block.
    """

    def __init__(
        self,
        controller,
        ptt_keys: frozenset[str],
        mode_keys: frozenset[str] | None = None,
        commands: dict[str, tuple[frozenset[str], "callable"]] | None = None,
    ):
        self.controller = controller
        self.ptt_keys = ptt_keys
        self.mode_keys = mode_keys or frozenset()
        self.commands = dict(commands or {})
        self.logger = get_logger(__name__)
        self._listener = None
        self._lock = threading.Lock()
        self._pressed: set[str] = set()
        self._ptt_active = False
        self._mode_active = False
        self._command_active: set[str] = set()

    # === dispatch (takes a key NAME — testable without pynput) ===

    def _dispatch_press(self, name: str) -> None:
        with self._lock:
            self._pressed.add(name)
            # Push-to-talk: hold semantics — fires when the full chord is down.
            if self.ptt_keys and self.ptt_keys <= self._pressed:
                if not self._ptt_active:
                    self._ptt_active = True
                    self.controller.on_ptt_press()
            elif self.mode_keys and self.mode_keys <= self._pressed:
                if not self._mode_active:
                    self._mode_active = True
                    self.controller.on_mode_toggle()
            for cmd_name, (keys, callback) in self.commands.items():
                if cmd_name not in self._command_active and keys <= self._pressed:
                    self._command_active.add(cmd_name)
                    callback()

    def _dispatch_release(self, name: str) -> None:
        with self._lock:
            self._pressed.discard(name)
            if self._ptt_active and not self.ptt_keys <= self._pressed:
                self._ptt_active = False
                self.controller.on_ptt_release()  # PTT-cancel semantics (W1)
            if self._mode_active and not self.mode_keys <= self._pressed:
                self._mode_active = False
            for cmd_name, (keys, _callback) in self.commands.items():
                if cmd_name in self._command_active and not keys <= self._pressed:
                    self._command_active.discard(cmd_name)

    # === pynput callbacks (only started when pynput is available) ===

    def _on_press(self, key) -> None:
        name = self._key_name(key)
        if name is not None:
            self._dispatch_press(name)

    def _on_release(self, key) -> None:
        name = self._key_name(key)
        if name is not None:
            self._dispatch_release(name)

    @staticmethod
    def _key_name(key) -> str | None:
        """Normalize a pynput key to the config vocabulary (e.g. "cmd", "d")."""
        try:
            import pynput  # optional dependency, imported lazily

            if isinstance(key, pynput.keyboard.Key):
                return getattr(key, "name", None)
            if isinstance(key, pynput.keyboard.KeyCode):
                return key.char.casefold() if key.char else (key.vk and chr(key.vk).casefold()) or None
        except Exception:  # listener callbacks must never raise
            return None
        return None

    # === lifecycle ===

    def start(self) -> bool:
        """Start the listener. Returns False (and logs why) when pynput is
        missing or the platform rejects the global hook — the app continues
        without global hotkeys rather than failing startup."""
        try:
            from pynput import keyboard

            self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self._listener.daemon = True
            self._listener.start()
            self.logger.info(
                f"⌨️ Global hotkeys active: PTT={sorted(self.ptt_keys) or 'n/a'}, "
                f"mode toggle={sorted(self.mode_keys) or 'n/a'}, "
                f"commands={sorted(self.commands) or 'n/a'}"
            )
            return True
        except Exception as e:  # missing dep or denied permission degrade to UI-only
            self.logger.warning(
                f"⚠️ Global hotkeys unavailable ({e}); use the HUD menu and "
                "--dictation-mode instead (install pynput + grant Accessibility "
                "permission for global hotkeys)"
            )
            return False

    def stop(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:  # shutdown must never raise
                pass
            self._listener = None
