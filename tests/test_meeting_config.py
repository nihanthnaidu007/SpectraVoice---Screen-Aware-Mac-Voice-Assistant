"""MeetingConfig tests (W3 D2.4): defaults, validation, env overrides, YAML load.

The config round-trip gate (test_config_roundtrip.py) covers save()/load()
dynamically — a meeting key that to_dict() drops fails there automatically.
This file covers the posture the spec pins: everything default-OFF, the cloud
summarizer unreachable without its separate consent flag, and the env/CLI
surfaces resolving.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import pytest

from assistant_app.utils.config import (
    AssistantConfig,
    ConfigManager,
    MeetingConfig,
)


def write_config(tmp_path, text: str) -> ConfigManager:
    (tmp_path / "config.yaml").write_text(text)
    return ConfigManager(config_path=tmp_path / "config.yaml")


class TestMeetingDefaults:
    def test_meeting_section_exists_on_assistant_config(self):
        cfg = AssistantConfig()
        assert isinstance(cfg.meeting, MeetingConfig)

    def test_everything_defaults_off(self):
        """No stealth anything: the kill-switch and the cloud-consent flag are
        both OFF on a fresh config."""
        cfg = AssistantConfig()
        assert cfg.meeting.enabled is False
        assert cfg.meeting.cloud_consent is False
        assert cfg.meeting.persist_audio is False

    def test_local_summarizer_is_the_default(self):
        cfg = AssistantConfig()
        assert cfg.meeting.summarizer == "local"

    def test_language_defaults_to_english(self):
        cfg = AssistantConfig()
        assert cfg.meeting.language == "english"

    def test_queue_depth_is_generous(self):
        """Fidelity queue, not the dictation depth of 2 that evicts oldest."""
        cfg = AssistantConfig()
        assert cfg.meeting.queue_depth >= 128

    def test_dictation_gains_retention_knob_defaulting_to_forever(self):
        cfg = AssistantConfig()
        assert cfg.dictation.retention_hours == 0.0

    def test_meeting_toggle_hotkey_default(self):
        cfg = AssistantConfig()
        assert cfg.hotkeys.meeting_toggle == "cmd+shift+e"


class TestMeetingValidation:
    def test_defaults_validate_clean(self):
        assert ConfigManager().validate() == []

    def test_cloud_summarizer_without_consent_is_invalid(self):
        manager = ConfigManager()
        manager.config.meeting.summarizer = "cloud"
        manager.config.meeting.cloud_consent = False
        issues = manager.validate()
        assert any("cloud_consent" in i for i in issues)

    def test_cloud_summarizer_with_consent_is_valid(self):
        manager = ConfigManager()
        manager.config.meeting.summarizer = "cloud"
        manager.config.meeting.cloud_consent = True
        assert manager.validate() == []

    def test_invalid_summarizer_rejected(self):
        manager = ConfigManager()
        manager.config.meeting.summarizer = "carrier-pigeon"
        assert any("summarizer" in i for i in manager.validate())

    def test_queue_depth_must_be_positive(self):
        manager = ConfigManager()
        manager.config.meeting.queue_depth = 0
        assert any("queue_depth" in i for i in manager.validate())

    def test_negative_retention_rejected(self):
        manager = ConfigManager()
        manager.config.meeting.retention_hours = -1
        assert any("retention_hours" in i for i in manager.validate())

    def test_negative_dictation_retention_rejected(self):
        manager = ConfigManager()
        manager.config.dictation.retention_hours = -0.5
        assert any("retention_hours" in i for i in manager.validate())

    def test_empty_language_rejected(self):
        manager = ConfigManager()
        manager.config.meeting.language = "  "
        assert any("language" in i for i in manager.validate())

    def test_tiny_chunk_size_rejected(self):
        manager = ConfigManager()
        manager.config.meeting.chunk_chars = 10
        assert any("chunk_chars" in i for i in manager.validate())


class TestMeetingEnvOverrides:
    def test_va_meeting_enabled_flips_kill_switch(self, monkeypatch):
        monkeypatch.setenv("VA_MEETING_ENABLED", "true")
        cfg = ConfigManager().config
        assert cfg.meeting.enabled is True

    def test_va_meeting_language(self, monkeypatch):
        monkeypatch.setenv("VA_MEETING_LANGUAGE", "german")
        cfg = ConfigManager().config
        assert cfg.meeting.language == "german"

    def test_va_meeting_summarizer(self, monkeypatch):
        monkeypatch.setenv("VA_MEETING_SUMMARIZER", "cloud")
        cfg = ConfigManager().config
        assert cfg.meeting.summarizer == "cloud"

    def test_va_meeting_retention_hours(self, monkeypatch):
        monkeypatch.setenv("VA_MEETING_RETENTION_HOURS", "48")
        cfg = ConfigManager().config
        assert cfg.meeting.retention_hours == 48.0

    def test_va_dictation_retention_hours(self, monkeypatch):
        monkeypatch.setenv("VA_DICTATION_RETENTION_HOURS", "24")
        cfg = ConfigManager().config
        assert cfg.dictation.retention_hours == 24.0


class TestMeetingYamlSection:
    def test_meeting_section_loads_from_yaml(self, tmp_path):
        manager = write_config(
            tmp_path,
            "meeting:\n"
            "  enabled: true\n"
            "  language: spanish\n"
            "  queue_depth: 512\n"
            "  summarizer: cloud\n"
            "  cloud_consent: true\n"
            "  retention_hours: 72\n",
        )
        meeting = manager.config.meeting
        assert meeting.enabled is True
        assert meeting.language == "spanish"
        assert meeting.queue_depth == 512
        assert meeting.summarizer == "cloud"
        assert meeting.cloud_consent is True
        assert meeting.retention_hours == 72.0
        # Untouched keys keep their defaults.
        assert meeting.persist_audio is False

    def test_unknown_meeting_keys_warn_and_are_dropped(self, tmp_path, caplog):
        manager = write_config(tmp_path, "meeting:\n  enabled: true\n  stealth_mode: true\n")
        assert manager.config.meeting.enabled is True
        assert not hasattr(manager.config.meeting, "stealth_mode")

    def test_repo_config_ships_meeting_disabled(self):
        """The shipped config.yaml keeps the kill-switch engaged."""
        manager = ConfigManager()  # finds the repo config.yaml from the repo root
        if manager.config_path is None:
            pytest.skip("no config.yaml found from this cwd")
        assert manager.config.meeting.enabled is False

    def test_settings_ui_discovers_meeting_keys(self):
        """The W2 settings UI auto-discovers MeetingConfig fields (spec D2.2)."""
        from assistant_app.hud.settings_model import discover_setting_keys

        keys = {k.path for k in discover_setting_keys(AssistantConfig())}
        assert "meeting.enabled" in keys
        assert "meeting.language" in keys
        assert "meeting.cloud_consent" in keys
