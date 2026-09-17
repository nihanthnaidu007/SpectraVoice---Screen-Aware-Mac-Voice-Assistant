"""Tests for ConfigManager as the single source of runtime settings (Wave 1).

Platform-independent: these tests validate YAML loading, VA_* env overrides,
unknown-key tolerance, validation warnings, mode-profile resolution, and the
legacy mode-profile contract using tmp files — no Mac, audio device, or
display required.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import pytest

from assistant_app.utils import config as config_module
from assistant_app.utils.config import (
    DEFAULT_MODE_PROFILES,
    ConfigManager,
    init_config,
)


@pytest.fixture(autouse=True)
def isolated_config_env(tmp_path, monkeypatch):
    """Run every test in an empty cwd with no VA_* overrides and no global config."""
    monkeypatch.chdir(tmp_path)
    for key in list(os.environ):
        if key.startswith("VA_"):
            monkeypatch.delenv(key)
    # Reset the module-level global so tests cannot leak state into each other
    config_module._config_manager = None
    yield
    config_module._config_manager = None


def write_config(tmp_path, text, name="config.yaml"):
    path = tmp_path / name
    path.write_text(text)
    return path


class TestDefaultsWithoutFile:
    def test_defaults_when_no_config_file_found(self):
        cfg = ConfigManager().config
        assert cfg.mode == "terminal"
        assert cfg.debug is False
        assert cfg.voice.tts_voice == "shimmer"
        assert cfg.voice.whisper_model == "base"
        assert cfg.barge_in.enabled is True
        assert cfg.microphone.input_device is None
        assert cfg.tts.output_device is None
        assert cfg.llm.provider == "auto"

    def test_missing_explicit_path_falls_back_to_defaults(self, tmp_path):
        cfg = ConfigManager(config_path=str(tmp_path / "nowhere.yaml")).config
        assert cfg.voice.whisper_model == "base"


class TestYAMLLoading:
    def test_loads_yaml_values(self, tmp_path):
        write_config(tmp_path, """
voice:
  whisper_model: small
  tts_voice: nova
screen:
  quality: 55
  scale_factor: 0.5
microphone:
  input_device: 2
  energy_threshold: 250
tts:
  output_device: 3
  hd_quality: false
""")
        cfg = ConfigManager().config
        assert cfg.voice.whisper_model == "small"
        assert cfg.voice.tts_voice == "nova"
        assert cfg.screen.quality == 55
        assert cfg.screen.scale_factor == 0.5
        assert cfg.microphone.input_device == 2
        assert cfg.microphone.energy_threshold == 250
        assert cfg.tts.output_device == 3
        assert cfg.tts.hd_quality is False

    def test_unknown_keys_warn_and_are_ignored(self, tmp_path):
        write_config(tmp_path, """
voice:
  tts_voice: nova
  not_a_real_key: 42
totally_unknown_section:
  foo: bar
""")
        cfg = ConfigManager().config
        assert cfg.voice.tts_voice == "nova"
        assert not hasattr(cfg.voice, "not_a_real_key")

    def test_non_mapping_section_falls_back_to_defaults(self, tmp_path):
        write_config(tmp_path, "voice: just a string")
        cfg = ConfigManager().config
        assert cfg.voice.tts_voice == "shimmer"  # default preserved

    def test_mode_profile_from_yaml(self, tmp_path):
        write_config(tmp_path, """
modes:
  minimal:
    screen_quality: 90
    whisper_model: small
    max_tokens: 1500
""")
        cfg = ConfigManager().config
        profile = cfg.mode_profile("minimal")
        assert profile.screen_quality == 90
        assert profile.whisper_model == "small"
        assert profile.max_tokens == 1500

    def test_mode_profile_partial_override_fills_from_defaults(self, tmp_path):
        write_config(tmp_path, """
modes:
  gui:
    max_tokens: 2000
""")
        cfg = ConfigManager().config
        profile = cfg.mode_profile("gui")
        assert profile.max_tokens == 2000
        assert profile.screen_quality == DEFAULT_MODE_PROFILES["gui"].screen_quality

    def test_invalid_mode_profile_ignored(self, tmp_path):
        write_config(tmp_path, """
modes:
  broken:
    max_tokens: not a number
""")
        cfg = ConfigManager().config
        # Falls back to the built-in terminal profile (unknown mode)
        assert cfg.mode_profile("broken") == DEFAULT_MODE_PROFILES["terminal"]


class TestEnvOverrides:
    def test_env_overrides_beat_yaml(self, tmp_path, monkeypatch):
        write_config(tmp_path, "voice:\n  whisper_model: small\n")
        monkeypatch.setenv("VA_VOICE_WHISPER_MODEL", "tiny")
        cfg = ConfigManager().config
        assert cfg.voice.whisper_model == "tiny"

    def test_typed_env_overrides(self, tmp_path, monkeypatch):
        monkeypatch.setenv("VA_SCREEN_QUALITY", "95")
        monkeypatch.setenv("VA_SCREEN_SCALE", "0.3")
        monkeypatch.setenv("VA_MICROPHONE_INPUT_DEVICE", "5")
        monkeypatch.setenv("VA_BARGE_IN_ENABLED", "false")
        monkeypatch.setenv("VA_MODE", "minimal")
        cfg = ConfigManager().config
        assert cfg.screen.quality == 95
        assert cfg.screen.scale_factor == 0.3
        assert cfg.microphone.input_device == 5
        assert cfg.barge_in.enabled is False
        assert cfg.mode == "minimal"


class TestValidation:
    def test_validate_flags_bad_values(self, tmp_path):
        write_config(tmp_path, """
screen:
  quality: 500
  scale_factor: 3.0
voice:
  whisper_model: gigantic
barge_in:
  min_confidence: 3.0
""")
        issues = ConfigManager().validate()
        joined = "\n".join(issues)
        assert "quality" in joined
        assert "Scale factor" in joined
        assert "Whisper model" in joined
        assert "min_confidence" in joined

    def test_validate_flags_invalid_llm_provider(self, tmp_path):
        write_config(tmp_path, "llm:\n  provider: quantum\n")
        joined = "\n".join(ConfigManager().validate())
        assert "llm.provider" in joined

    def test_validate_flags_invalid_speech_rate(self, tmp_path):
        write_config(tmp_path, "voice:\n  speech_rate: 9.0\n")
        joined = "\n".join(ConfigManager().validate())
        assert "Speech rate" in joined

    def test_valid_config_has_no_issues(self, tmp_path):
        write_config(tmp_path, "screen:\n  quality: 70\n")
        assert ConfigManager().validate() == []


class TestModeProfileContract:
    """The built-in profiles must match the pre-W1 hardcoded per-mode settings."""

    def test_terminal_profile(self):
        p = DEFAULT_MODE_PROFILES["terminal"]
        assert (p.screen_quality, p.screen_scale) == (80, 0.8)
        assert p.whisper_model == "base"
        assert p.max_tokens == 800

    def test_minimal_profile(self):
        p = DEFAULT_MODE_PROFILES["minimal"]
        assert (p.screen_quality, p.screen_scale) == (60, 0.6)
        assert p.whisper_model == "tiny"
        assert p.max_tokens == 300

    def test_unknown_mode_falls_back_to_terminal(self):
        cfg = ConfigManager().config
        assert cfg.mode_profile("nonexistent") == DEFAULT_MODE_PROFILES["terminal"]


class TestGlobalInit:
    def test_init_config_sets_global(self, tmp_path):
        write_config(tmp_path, "voice:\n  tts_voice: echo\n")
        manager = init_config()
        assert manager.config.voice.tts_voice == "echo"
        assert config_module.get_config().voice.tts_voice == "echo"

    def test_init_config_with_explicit_path(self, tmp_path):
        path = write_config(tmp_path, "voice:\n  tts_voice: aria\n", name="custom.yaml")
        manager = init_config(str(path))
        assert manager.config.voice.tts_voice == "aria"


class TestSaveRoundtrip:
    def test_save_and_reload_preserves_values(self, tmp_path):
        write_config(tmp_path, """
voice:
  whisper_model: small
microphone:
  input_device: 4
""")
        out = tmp_path / "saved.json"
        ConfigManager().save(str(out))
        reloaded = ConfigManager(config_path=str(out)).config
        assert reloaded.voice.whisper_model == "small"
        assert reloaded.microphone.input_device == 4


class TestSupervisorSection:
    """supervisor.* — the Wave 2 restart policy, read through the config system."""

    def test_code_defaults_are_conservative(self):
        cfg = ConfigManager().config
        assert cfg.supervisor.enabled is False
        assert cfg.supervisor.max_restarts == 5
        assert cfg.supervisor.window_seconds == 60.0
        assert cfg.supervisor.backoff_seconds == 1.0
        assert cfg.supervisor.backoff_max_seconds == 30.0

    def test_shipped_config_enables_supervision(self):
        # The repo's config.yaml opts in even though the code default is off.
        shipped = os.path.join(project_root, "config.yaml")
        cfg = ConfigManager(config_path=shipped).config
        assert cfg.supervisor.enabled is True
        assert cfg.supervisor.max_restarts == 5

    def test_yaml_values_are_loaded(self, tmp_path):
        write_config(tmp_path, """
supervisor:
  enabled: true
  max_restarts: 3
  window_seconds: 30
  backoff_seconds: 2
  backoff_max_seconds: 10
""")
        cfg = ConfigManager().config
        assert cfg.supervisor.enabled is True
        assert cfg.supervisor.max_restarts == 3
        assert cfg.supervisor.window_seconds == 30.0
        assert cfg.supervisor.backoff_seconds == 2.0
        assert cfg.supervisor.backoff_max_seconds == 10.0

    def test_env_overrides_beat_yaml(self, tmp_path, monkeypatch):
        write_config(tmp_path, "supervisor:\n  enabled: true\n  max_restarts: 5\n")
        monkeypatch.setenv("VA_SUPERVISOR_ENABLED", "false")
        monkeypatch.setenv("VA_SUPERVISOR_MAX_RESTARTS", "2")
        cfg = ConfigManager().config
        assert cfg.supervisor.enabled is False
        assert cfg.supervisor.max_restarts == 2

    def test_serialization_roundtrips(self, tmp_path):
        write_config(tmp_path, "supervisor:\n  max_restarts: 7\n")
        manager = ConfigManager()
        assert manager.to_dict()["supervisor"]["max_restarts"] == 7

    def test_negative_restarts_are_rejected(self, tmp_path):
        write_config(tmp_path, "supervisor:\n  max_restarts: -1\n")
        issues = ConfigManager().validate()
        assert any("max_restarts" in issue for issue in issues)
