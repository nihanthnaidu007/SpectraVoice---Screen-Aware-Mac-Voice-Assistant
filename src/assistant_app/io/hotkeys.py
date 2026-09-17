"""Push-to-talk / mode-toggle hotkey listener.

W1 uses ``pynput.keyboard.Listener`` (a global keyboard hook — this is what a
system-wide dictation hotkey requires). The listener is entirely optional at
runtime: on macOS pynput needs Accessibility permission for global capture, and
headless environments have no keyboard at all, so ``DictationHotkeys.start``
reports ``available=False`` instead of crashing and the assistant logs the
remediation (``pip install pynput`` + granting permission). Dictation itself
keeps working without the hotkeys via the CLI ``--dictation-mode`` flag and,
for VAD mode, without any keypress at all.

Key events are dispatched on the pynput callback thread and must never block,
so they only touch the controller's lock-protected state and return.

Dependency-light at import time: pynput is imported inside the listener class,
never at module import, keeping unit tests dependency-free.
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
    """Background global-key listener driving the DictationController.

    Requires a config section entry per action:
    - ``hotkeys.dictation_mode`` — tap to toggle push_to_talk <-> vad
    - ``hotkeys.push_to_talk`` — hold to dictate (existing config key)

    Threading contract: pynput callbacks run on their own thread; they only
    call the controller's lock-guarded callbacks and never block.
    """

    def __init__(self, controller, ptt_keys: frozenset[str], mode_keys: frozenset[str] | None = None):
        self.controller = controller
        self.ptt_keys = ptt_keys
        self.mode_keys = mode_keys or frozenset()
        self.logger = get_logger(__name__)
        self._listener = None
        self._ptt_down = False
        self._mode_down = False
        self._lock = threading.Lock()

    # === pynput callbacks (only started when pynput is available) ===

    def _on_press(self, key) -> None:
        name = self._key_name(key)
        if name is None:
            return
        with self._lock:
            if name in self.ptt_keys and not self._ptt_down:
                self._ptt_down = True
                self.controller.on_ptt_press()
            elif self.mode_keys and name in self.mode_keys and not self._mode_down:
                self._mode_down = True
                self.controller.on_mode_toggle()

    def _on_release(self, key) -> None:
        name = self._key_name(key)
        if name is None:
            return
        with self._lock:
            if name in self.ptt_keys and self._ptt_down:
                self._ptt_down = False
                self.controller.on_ptt_release()
            elif self.mode_keys and name in self.mode_keys:
                self._mode_down = False

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
        missing or the platform rejects the global hook — dictation continues
        without hotkeys rather than failing startup."""
        try:
            from pynput import keyboard

            self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
            self._listener.daemon = True
            self._listener.start()
            self.logger.info(f"⌨️ Dictation hotkeys active: PTT={sorted(self.ptt_keys)}, mode toggle={sorted(self.mode_keys) or 'n/a'}")
            return True
        except Exception as e:  # missing dep or denied permission degrade to flag-only
            self.logger.warning(
                f"⚠️ Dictation hotkeys unavailable ({e}); use --dictation-mode and "
                "the push_to_talk config instead (install pynput + grant Accessibility "
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
