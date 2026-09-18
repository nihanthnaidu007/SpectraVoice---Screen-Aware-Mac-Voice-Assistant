"""W2 S3 tests — runtime dictation toggle.

Dictation on/off from the HUD menu or settings at runtime — no relaunch.
Mutation flows through the config system ONLY (ConfigManager
.set_dictation_enabled persists via save()/to_dict() — the W2 round-trip
gate — and notifies; the orchestrator's on_reload hook applies the change
live). The pynput listener is never rebuilt: pynput stays the SOLE hotkey
path, and the same listener also drives the non-dictation commands.

Covered here (no AppKit):
- Config round trip: set_dictation_enabled -> in-memory config -> to_dict
  -> save() -> a FRESH ConfigManager.load() reads the new value (the gate
  the brief names), and observers fire exactly on change.
- Controller gate: a disabled session never arms (PTT press no-ops), mode
  toggles are refused, an active session is canceled on disable, and
  re-enabling re-arms per the activation mode.
- Pure label helper: the menu offer follows the applied state.
- Wiring structure: the orchestrator module is unimportable on Linux CI
  (speech_recognition at module scope), so the hot-apply hook, protocol
  methods, and the single-mutation-path rule are pinned structurally —
  including that NO hotkey-dispatch change sneaks a second hotkey path in.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.hud.state import dictation_toggle_item_label
from assistant_app.services.dictation import DictationController
from assistant_app.utils.config import ConfigManager

PROJECT_ROOT = Path(project_root)
SRC = PROJECT_ROOT / "src" / "assistant_app"


def _make_controller(activation: str = "push_to_talk") -> DictationController:
    from assistant_app.utils.config import DictationConfig

    return DictationController(DictationConfig(activation=activation), on_status=lambda _s: None)


def _make_manager(tmp_path) -> ConfigManager:
    """ConfigManager bound to an EXISTING temp file — a missing path falls
    through _find_config_file to the repo's config.yaml (which has dictation
    enabled: true), silently changing what the test reads. An empty file
    means dataclass defaults."""
    path = tmp_path / "config.yaml"
    path.write_text("", encoding="utf-8")
    return ConfigManager(config_path=str(path))


class TestConfigRoundTrip:
    """The W2 round-trip gate: to_dict -> save -> fresh load preserves the
    toggled value; observers fire only on change."""

    def test_set_enabled_flips_memory_persists_and_notifies(self, tmp_path):
        manager = _make_manager(tmp_path)
        seen: list[bool] = []
        manager.on_reload(lambda cfg: seen.append(cfg.dictation.enabled))

        manager.set_dictation_enabled(True)

        assert manager.config.dictation.enabled is True
        assert seen == [True]

    def test_round_trip_through_save_and_fresh_load(self, tmp_path):
        manager = _make_manager(tmp_path)
        manager.set_dictation_enabled(True)

        # save() already ran inside the setter; a fresh manager bound to the
        # same file must read the toggled value back.
        fresh = ConfigManager(config_path=str(tmp_path / "config.yaml"))
        assert fresh.config.dictation.enabled is True
        assert fresh.to_dict()["dictation"]["enabled"] is True

    def test_disable_round_trips_back_to_false(self, tmp_path):
        manager = _make_manager(tmp_path)
        manager.set_dictation_enabled(True)
        manager.set_dictation_enabled(False)

        fresh = ConfigManager(config_path=str(tmp_path / "config.yaml"))
        assert fresh.config.dictation.enabled is False

    def test_no_change_fires_no_callbacks(self, tmp_path):
        manager = _make_manager(tmp_path)  # dictation disabled by default
        seen: list[bool] = []
        manager.on_reload(lambda cfg: seen.append(cfg.dictation.enabled))

        manager.set_dictation_enabled(False)

        assert seen == []

    def test_other_sections_survive_the_mutation(self, tmp_path):
        manager = _make_manager(tmp_path)
        mode_before = manager.config.mode
        manager.set_dictation_enabled(True)

        assert manager.to_dict()["dictation"]["activation"] == manager.config.dictation.activation
        assert manager.config.mode == mode_before  # untouched section stays


class TestControllerGate:
    """A disabled session never arms and never inserts; the pynput listener
    keeps its controller reference — no relaunch, no second hotkey path."""

    def test_default_controller_is_enabled(self):
        controller = _make_controller()
        assert controller.enabled is True

    def test_disabled_ptt_press_does_not_arm(self):
        controller = _make_controller()
        controller.set_enabled(False)

        controller.on_ptt_press()

        assert controller.armed is False

    def test_disabled_mode_toggle_does_not_flip_activation(self):
        controller = _make_controller(activation="push_to_talk")
        controller.set_enabled(False)

        controller.on_mode_toggle()

        assert controller.activation == "push_to_talk"

    def test_disable_cancels_armed_session(self):
        controller = _make_controller(activation="vad")
        controller.start()
        assert controller.armed is True  # VAD starts armed

        controller.set_enabled(False)

        assert controller.enabled is False
        assert controller.armed is False

    def test_reenable_rearms_vad(self):
        controller = _make_controller(activation="vad")
        controller.set_enabled(False)
        controller.set_enabled(True)

        assert controller.enabled is True
        assert controller.armed is True  # VAD re-arms on enable

    def test_reenable_keeps_ptt_unarmed(self):
        controller = _make_controller(activation="push_to_talk")
        controller.set_enabled(False)
        controller.set_enabled(True)
        assert controller.armed is False  # PTT waits for the hotkey

        controller.on_ptt_press()
        assert controller.armed is True  # hotkey path works again

    def test_set_enabled_is_idempotent(self):
        controller = _make_controller()
        controller.set_enabled(False)
        controller.set_enabled(False)
        assert controller.enabled is False
        controller.set_enabled(True)
        controller.set_enabled(True)
        assert controller.enabled is True


class TestMenuLabel:
    def test_off_states_offer_enable(self):
        assert dictation_toggle_item_label(None) == "Enable Dictation"
        assert dictation_toggle_item_label("off") == "Enable Dictation"

    def test_live_states_offer_disable(self):
        for state in ("ptt-held", "vad-active", "inserting"):
            assert dictation_toggle_item_label(state) == "Disable Dictation"


class TestWiringStructure:
    """spectravoice_assistant.py is unimportable on Linux CI — the hot-apply
    wiring is pinned structurally (menubar-test standard); behavior is
    Mac-verified via the PR checklist."""

    def _source(self) -> str:
        return (SRC / "services" / "spectravoice_assistant.py").read_text(encoding="utf-8")

    def test_orchestrator_registers_the_hot_apply_hook(self):
        source = self._source()
        assert "self.config_manager.on_reload(self._apply_runtime_dictation)" in source
        assert "self._dictation_runtime_state = self.dictation_enabled" in source

    def test_apply_hook_is_idempotent_and_builds_on_enable(self):
        tree = ast.parse(self._source())
        hooks = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_apply_runtime_dictation"
        ]
        assert len(hooks) == 1
        body = ast.unparse(hooks[0])
        assert "DictationController(" in body  # constructs when launched without
        assert "self.dictation.start()" in body
        assert "set_enabled" in body
        assert "== self._dictation_runtime_state" in body  # no-op on unchanged

    def test_toggle_goes_through_the_config_system_only(self):
        tree = ast.parse(self._source())
        setters = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "set_dictation_enabled"
        ]
        assert len(setters) == 1
        body = ast.unparse(setters[0])
        # The orchestrator method delegates to the ConfigManager — it never
        # writes config.yaml itself and never touches pynput.
        assert "self.config_manager.set_dictation_enabled(" in body
        assert "yaml" not in body
        assert "pynput" not in body

    def test_pynput_stays_the_sole_hotkey_path(self):
        # The listener class must be untouched by S3: no dictation-toggle
        # dispatch in io/hotkeys.py, no second key listener anywhere.
        hotkeys = (SRC / "io" / "hotkeys.py").read_text(encoding="utf-8")
        assert "set_dictation_enabled" not in hotkeys
        assert "dictation_enabled" not in hotkeys
        assert "toggle_dictation_enabled" not in hotkeys

    def test_hud_actions_expose_the_toggle(self):
        source = self._source()
        assert "def toggle_dictation_enabled" in source  # protocol impl + orchestrator
        tree = ast.parse(source)
        protocol_methods = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        assert "dictation_enabled" in protocol_methods

    def test_settings_window_dispatches_to_the_provider(self):
        source = (SRC / "hud" / "settings_window.py").read_text(encoding="utf-8")
        assert "set_dictation_enabled" in source
        assert "dictationAction_" in source
