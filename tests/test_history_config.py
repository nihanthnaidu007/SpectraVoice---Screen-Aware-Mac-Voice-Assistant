"""HistoryConfig tests (W4 D2): defaults, validation, env overrides, YAML.

Mirrors the MeetingConfig posture tests: the section is minimal by design
(H7 — only genuine user knobs), so the coverage proves the one knob exists,
its default is the privacy-preserving one ("" = ask every time), validation
guards it, the env surface resolves, and the section sits in the settings
ordering right after meeting. The dynamic round-trip gate
(test_config_roundtrip.py) additionally fails if to_dict() ever drops a
history key.
"""

from __future__ import annotations

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.hud.settings_model import SECTION_ORDER
from assistant_app.utils.config import (
    AssistantConfig,
    ConfigManager,
    HistoryConfig,
)


def write_config(tmp_path, text: str) -> ConfigManager:
    (tmp_path / "config.yaml").write_text(text)
    return ConfigManager(config_path=tmp_path / "config.yaml")


class TestHistoryDefaults:
    def test_history_section_exists_on_assistant_config(self):
        cfg = AssistantConfig()
        assert isinstance(cfg.history, HistoryConfig)

    def test_export_dir_defaults_to_ask_every_time(self):
        """No default off-device destination: export must be a deliberate act."""
        assert AssistantConfig().history.export_dir == ""


class TestHistoryValidation:
    def test_defaults_validate_clean(self):
        assert ConfigManager().validate() == []

    def test_whitespace_padded_export_dir_rejected(self):
        manager = ConfigManager()
        manager.config.history.export_dir = "  /tmp/exports  "
        assert any("export_dir" in i for i in manager.validate())

    def test_clean_export_dir_validates(self):
        manager = ConfigManager()
        manager.config.history.export_dir = "/tmp/exports"
        assert manager.validate() == []


class TestHistoryEnvOverrides:
    def test_va_history_export_dir(self, monkeypatch):
        monkeypatch.setenv("VA_HISTORY_EXPORT_DIR", "/tmp/from-env")
        assert ConfigManager().config.history.export_dir == "/tmp/from-env"


class TestHistoryPersistence:
    def test_yaml_section_loads(self, tmp_path):
        manager = write_config(tmp_path, "history:\n  export_dir: /tmp/from-yaml\n")
        assert manager.config.history.export_dir == "/tmp/from-yaml"

    def test_to_dict_carries_the_history_section(self):
        manager = ConfigManager()
        manager.config.history.export_dir = "/tmp/exports"
        dumped = manager.to_dict()
        assert dumped["history"]["export_dir"] == "/tmp/exports"


class TestSettingsOrdering:
    def test_history_sits_right_after_meeting(self):
        """Meeting controls first, history immediately after — the dashboard
        is meeting-adjacent, not buried under infrastructure sections."""
        assert SECTION_ORDER.index("history") == SECTION_ORDER.index("meeting") + 1
