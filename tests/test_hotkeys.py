"""W2 D3 hotkey tests: chord-aware dispatch, PTT semantics, command taps.

The pynput callback-thread contract is tested through the name-level dispatch
methods — no pynput dependency, no threads. Degradation when pynput is
missing is covered in test_dictation.py (start() returns False, never raises).
"""

from assistant_app.io.hotkeys import DictationHotkeyListener, parse_hotkey


class Recorder:
    """Records controller callbacks: PTT press/release + mode toggles."""

    def __init__(self):
        self.events: list[str] = []

    def on_ptt_press(self):
        self.events.append("ptt_press")

    def on_ptt_release(self):
        self.events.append("ptt_release")

    def on_mode_toggle(self):
        self.events.append("mode_toggle")


def make_listener(controller, commands=None):
    return DictationHotkeyListener(
        controller,
        frozenset({"space"}),
        frozenset({"cmd", "shift", "d"}),
        commands,
    )


# --- parse_hotkey (W1 behavior preserved) ------------------------------------


def test_parse_hotkey_normalizes_and_rejects():
    assert parse_hotkey("cmd+shift+d") == frozenset({"cmd", "shift", "d"})
    assert parse_hotkey("Command+Shift+D") == frozenset({"cmd", "shift", "d"})
    assert parse_hotkey("option+space") == frozenset({"alt", "space"})
    assert parse_hotkey("") is None
    assert parse_hotkey(None) is None


# --- PTT hold/cancel semantics (W1, preserved) -------------------------------


def test_ptt_fires_on_press_and_cancels_on_release():
    controller = Recorder()
    listener = make_listener(controller)
    listener._dispatch_press("space")
    listener._dispatch_release("space")
    assert controller.events == ["ptt_press", "ptt_release"]


def test_ptt_does_not_refire_while_held():
    controller = Recorder()
    listener = make_listener(controller)
    listener._dispatch_press("space")
    listener._dispatch_press("space")  # pynput repeats presses while held
    assert controller.events == ["ptt_press"]


# --- chord-aware matching (fixes W1 single-key false trigger) ----------------


def test_combo_fires_only_when_full_chord_is_down():
    controller = Recorder()
    listener = make_listener(controller)
    listener._dispatch_press("cmd")
    listener._dispatch_press("shift")
    assert controller.events == []  # modifiers alone never fire
    listener._dispatch_press("d")  # chord completes
    assert controller.events == ["mode_toggle"]


def test_single_key_of_combo_does_not_fire():
    controller = Recorder()
    listener = make_listener(controller)
    listener._dispatch_press("d")
    listener._dispatch_release("d")
    assert controller.events == []


def test_combo_fires_on_each_full_completion():
    controller = Recorder()
    listener = make_listener(controller)
    for key in ("cmd", "shift", "d"):
        listener._dispatch_press(key)
    listener._dispatch_release("d")
    for key in ("cmd", "shift", "d"):
        listener._dispatch_press(key)
    assert controller.events.count("mode_toggle") == 2


# --- command taps (W2: mute / pause_resume / quit) ---------------------------


def test_command_tap_fires_once_per_completion():
    fired = []
    controller = Recorder()
    listener = make_listener(
        controller,
        commands={"quit": (frozenset({"cmd", "shift", "q"}), lambda: fired.append("quit"))},
    )
    for key in ("cmd", "shift", "q"):
        listener._dispatch_press(key)
    assert fired == ["quit"]
    for key in ("cmd", "shift", "q"):  # still held — no refire
        listener._dispatch_press(key)
    assert fired == ["quit"]
    listener._dispatch_release("q")
    for key in ("cmd", "shift", "q"):
        listener._dispatch_press(key)
    assert fired == ["quit", "quit"]


def test_commands_dispatch_independently_of_dictation():
    fired = []
    controller = Recorder()
    listener = DictationHotkeyListener(
        controller,
        frozenset({"space"}),
        None,
        {
            "mute": (frozenset({"m"}), lambda: fired.append("mute")),
            "pause_resume": (frozenset({"cmd", "p"}), lambda: fired.append("pause")),
        },
    )
    listener._dispatch_press("m")
    assert fired == ["mute"]
    listener._dispatch_press("cmd")
    listener._dispatch_press("p")
    assert fired == ["mute", "pause"]
    assert controller.events == []  # dictation untouched


# --- doctor visibility (degradation is doctor-reported) ----------------------


def test_doctor_hotkey_probe_warns_without_pynput(tmp_path, monkeypatch):
    from assistant_app import doctor
    from assistant_app.utils.config import ConfigManager

    monkeypatch.chdir(tmp_path)  # no config file → built-in defaults
    monkeypatch.setattr(doctor, "_pynput_available", lambda: False)
    outcome = doctor._check_hotkeys(ConfigManager()).probe()
    assert outcome.state == doctor.CheckState.WARN
    assert "HUD menu" in outcome.detail


def test_doctor_hotkey_probe_passes_with_pynput(tmp_path, monkeypatch):
    from assistant_app import doctor
    from assistant_app.utils.config import ConfigManager

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(doctor, "_pynput_available", lambda: True)
    outcome = doctor._check_hotkeys(ConfigManager()).probe()
    assert outcome.state == doctor.CheckState.PASS


def test_default_checks_include_hotkeys(tmp_path, monkeypatch):
    from assistant_app import doctor
    from assistant_app.utils.config import ConfigManager

    monkeypatch.chdir(tmp_path)
    names = [c.name for c in doctor.build_default_checks(ConfigManager())]
    assert "Global hotkeys" in names
