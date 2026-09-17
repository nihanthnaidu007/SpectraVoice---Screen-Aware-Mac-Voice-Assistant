"""Config round-trip gate (W2 D2): save() must never drop a section or key.

Regression guard for the `to_dict()` droppage bug: W1's config grew sections
that `to_dict()` silently omitted, so `save()` would have written a config.yaml
that lost `llm:`, `modes:`, and individual keys (`hotkeys.dictation_mode`,
`logging.format`) on the first save. This suite enumerates the dataclass
sections dynamically — a future section or key added to `AssistantConfig`
fails here unless `to_dict()` covers it.

Platform-independent: tmp YAML files only, no Mac/audio/network.
"""

import dataclasses
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import yaml

from assistant_app.utils.config import AssistantConfig, ConfigManager


def section_types() -> dict[str, type]:
    """Map section name -> dataclass type for every dataclass-typed field."""
    out = {}
    for f in dataclasses.fields(AssistantConfig):
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default
        if dataclasses.is_dataclass(default):
            out[f.name] = type(default)
    return out


class TestToDictCoversEveryKey:
    def test_to_dict_has_every_dataclass_section(self):
        to_dict_keys = set(ConfigManager().to_dict())
        sections = section_types()
        assert sections, "AssistantConfig lost its dataclass sections?"
        missing = set(sections) - to_dict_keys
        assert not missing, f"to_dict() drops sections: {sorted(missing)}"

    def test_to_dict_has_every_dataclass_field(self):
        exported = ConfigManager().to_dict()
        for name, dc_type in section_types().items():
            field_names = {f.name for f in dataclasses.fields(dc_type)}
            exported_keys = set(exported.get(name, {}))
            missing = field_names - exported_keys
            assert not missing, f"to_dict() drops {name} keys: {sorted(missing)}"


class TestSaveLoadRoundTrip:
    def _mutated_config(self) -> ConfigManager:
        """A config with a distinctive non-default value in every single field."""
        manager = ConfigManager()
        cfg = manager.config

        cfg.mode = "gui"
        cfg.debug = True
        cfg.modes = {"terminal": cfg.mode_profile("terminal").__class__(80, 0.7, "small", 700)}

        cfg.voice.tts_voice = "nova"
        cfg.voice.whisper_model = "small"
        cfg.voice.language = "de"
        cfg.voice.speech_rate = 1.5

        cfg.screen.quality = 65
        cfg.screen.scale_factor = 0.55
        cfg.screen.refresh_interval = 0.35
        cfg.screen.cache_duration = 2.5

        cfg.llm.provider = "local"
        cfg.llm.cloud_model = "gpt-5.1"
        cfg.llm.ollama_url = "http://localhost:11500"
        cfg.llm.ollama_model = "llava"
        cfg.llm.max_tokens = 1234
        cfg.llm.temperature = 0.15
        cfg.llm.timeout = 99
        cfg.llm.enable_vision = False
        cfg.llm.enable_tools = False

        cfg.api.openai_model = "gpt-5.2"
        cfg.api.max_tokens = 4321
        cfg.api.temperature = 1.25
        cfg.api.timeout = 21
        cfg.api.max_retries = 7

        cfg.hotkeys.mute_toggle = "cmd+shift+z"
        cfg.hotkeys.push_to_talk = "fn"
        cfg.hotkeys.pause_resume = "cmd+shift+k"
        cfg.hotkeys.quit = "cmd+shift+x"
        cfg.hotkeys.dictation_mode = "cmd+shift+j"

        cfg.logging.level = "DEBUG"
        cfg.logging.file = "logs/roundtrip.log"
        cfg.logging.max_size_mb = 3
        cfg.logging.backup_count = 2
        cfg.logging.format = "%(name)s %(message)s"

        cfg.safety.dry_run = True
        cfg.safety.block_dangerous_commands = False
        cfg.safety.require_confirmation = True

        cfg.barge_in.enabled = False
        cfg.barge_in.min_confidence = 0.9
        cfg.barge_in.min_words = 4
        cfg.barge_in.min_chars = 9

        cfg.microphone.input_device = 3
        cfg.microphone.energy_threshold = 2222
        cfg.microphone.dynamic_energy_threshold = True
        cfg.microphone.pause_threshold = 1.3
        cfg.microphone.ambient_noise_seconds = 0.4
        cfg.microphone.inline_transcription = True

        cfg.tts.output_device = 5
        cfg.tts.hd_quality = False

        cfg.supervisor.enabled = True
        cfg.supervisor.max_restarts = 9
        cfg.supervisor.window_seconds = 44.0
        cfg.supervisor.backoff_seconds = 2.5
        cfg.supervisor.backoff_max_seconds = 60.0

        cfg.dictation.enabled = True
        cfg.dictation.activation = "vad"
        cfg.dictation.style = "minimal"
        cfg.dictation.app_styles = {"slack": "standard"}
        cfg.dictation.filler_removal = False
        cfg.dictation.filler_words = ["hmm", "well"]
        cfg.dictation.snippets = {"comma": ","}
        cfg.dictation.vad_silence_timeout = 0.9
        cfg.dictation.insert_enter = True
        cfg.dictation.persist_audio = True
        cfg.dictation.audio_dir = "logs/custom_audio"

        return manager

    def test_every_field_survives_save_and_load(self, tmp_path):
        source = self._mutated_config()
        path = tmp_path / "config.yaml"

        source.save(path)

        reloaded = ConfigManager(config_path=path)
        assert reloaded.to_dict() == source.to_dict()

    def test_round_tripped_file_is_valid_yaml_with_all_sections(self, tmp_path):
        source = self._mutated_config()
        path = tmp_path / "config.yaml"
        source.save(path)

        raw = yaml.safe_load(path.read_text())
        for section in section_types():
            assert section in raw, f"saved YAML is missing the '{section}:' section"

    def test_save_then_validate_reports_no_issues(self, tmp_path):
        """The saved file must not just be complete but also valid."""
        source = self._mutated_config()
        path = tmp_path / "config.yaml"
        source.save(path)

        reloaded = ConfigManager(config_path=path)
        assert reloaded.validate() == []

    def test_fresh_defaults_round_trip(self, tmp_path):
        """A default config must also survive (no information invented or lost)."""
        source = ConfigManager()
        path = tmp_path / "config.yaml"
        source.save(path)

        reloaded = ConfigManager(config_path=path)
        assert reloaded.to_dict() == source.to_dict()
