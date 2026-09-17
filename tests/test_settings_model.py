"""W2 D2 settings-model tests: discovery, taxonomy, validation-gated application.

The gate for the settings surface: valid changes persist through the config
system, validation-rejected input is never written (memory or disk), and the
hot-apply/restart taxonomy is explicit and complete.
"""

import pytest

from assistant_app.hud.settings_model import (
    HOT_APPLY_KEYS,
    SettingKey,
    apply_settings,
    coerce,
    discover_setting_keys,
    get_config_value,
)
from assistant_app.utils.config import ConfigManager

MINIMAL_YAML = """\
voice:
  tts_voice: nova
  whisper_model: base
  speech_rate: 1.0
  language: english
screen:
  quality: 80
  scale_factor: 0.5
mode: terminal
"""


@pytest.fixture()
def manager(tmp_path, monkeypatch):
    """ConfigManager over a temp file, with env overrides cleared."""
    for var in (
        "VA_VOICE_TTS_VOICE",
        "VA_SCREEN_QUALITY",
        "VA_LOG_LEVEL",
        "VA_BARGE_IN_MIN_CONFIDENCE",
        "VA_MODE",
    ):
        monkeypatch.delenv(var, raising=False)
    path = tmp_path / "config.yaml"
    path.write_text(MINIMAL_YAML)
    return ConfigManager(config_path=path)


def key(section: str, field: str, **overrides) -> SettingKey:
    """SettingKey for tests -- discovery supplies the real flags in live use."""
    defaults = {"hot_apply": False, "env_var": None, "value_type": "str"}
    defaults.update(overrides)
    return SettingKey(section=section, field=field, **defaults)


# --- Discovery and taxonomy -------------------------------------------------


def test_discovery_covers_every_dataclass_section(manager):
    keys = discover_setting_keys(manager.config)
    sections = {k.section for k in keys}
    for section in (
        "voice", "screen", "llm", "api", "hotkeys", "dictation",
        "logging", "safety", "barge_in", "microphone", "tts", "supervisor",
    ):
        assert section in sections, f"settings UI missing section {section}"


def test_discovery_includes_top_level_scalars(manager):
    keys = discover_setting_keys(manager.config)
    top = {k.field for k in keys if k.section == ""}
    assert top == {"mode", "debug"}


def test_discovery_marks_env_override_keys(manager):
    keys = discover_setting_keys(manager.config)
    by_path = {k.path: k for k in keys}
    assert by_path["voice.tts_voice"].env_var == "VA_VOICE_TTS_VOICE"
    assert by_path["mode"].env_var == "VA_MODE"
    assert by_path["screen.quality"].env_var == "VA_SCREEN_QUALITY"


def test_hot_apply_taxonomy_is_complete_and_honest(manager):
    """Every taxonomy entry must exist in discovery, and discovery must agree
    with the taxonomy -- an unknown hot key or a misflagged key is a bug."""
    keys = discover_setting_keys(manager.config)
    discovered = {(k.section, k.field) for k in keys}
    assert HOT_APPLY_KEYS <= discovered, "taxonomy references an unknown key"
    for k in keys:
        assert k.hot_apply == ((k.section, k.field) in HOT_APPLY_KEYS)


# --- Coercion ---------------------------------------------------------------


def test_coerce_bool_variants():
    assert coerce("true", "bool") is True
    assert coerce("On", "bool") is True
    assert coerce("0", "bool") is False
    assert coerce("no", "bool") is False
    with pytest.raises(ValueError):
        coerce("maybe", "bool")


def test_coerce_numeric_and_str():
    assert coerce(" 42 ", "int") == 42
    assert coerce("1.5", "float") == 1.5
    assert coerce("nova", "str") == "nova"
    with pytest.raises(ValueError):
        coerce("abc", "int")
    with pytest.raises(ValueError, match="structured"):
        coerce("x", "dict")


# --- Application: accept, reject, persist -----------------------------------


def test_valid_change_is_applied_and_persisted(manager):
    k = key("voice", "speech_rate", value_type="float")
    outcomes = apply_settings(manager, {k: "2.0"})
    assert outcomes[0].accepted and outcomes[0].new_value == 2.0
    assert get_config_value(manager.config, k) == 2.0

    # Persisted: a fresh manager over the same file sees the new value.
    fresh = ConfigManager(config_path=manager.config_path)
    assert get_config_value(fresh.config, k) == 2.0


def test_validation_rejected_input_is_never_written(manager):
    """The acceptance gate: invalid values change nothing, on disk or in memory."""
    k = key("screen", "quality", value_type="int")
    before = get_config_value(manager.config, k)

    outcomes = apply_settings(manager, {k: "500"})  # validate(): 1-100
    assert not outcomes[0].accepted
    assert "1-100" in outcomes[0].reason
    assert get_config_value(manager.config, k) == before  # reverted in memory

    fresh = ConfigManager(config_path=manager.config_path)
    assert get_config_value(fresh.config, k) == before  # nothing persisted


def test_coercion_failure_rejected_with_reason(manager):
    k = key("api", "max_tokens", value_type="int")
    outcomes = apply_settings(manager, {k: "not-a-number"})
    assert not outcomes[0].accepted
    assert "invalid int" in outcomes[0].reason


def test_mixed_batch_persists_only_accepted_changes(manager):
    """One invalid change must not block or corrupt valid siblings."""
    good = key("voice", "speech_rate", value_type="float")
    bad = key("screen", "quality", value_type="int")
    outcomes = apply_settings(manager, {good: "1.5", bad: "999"})
    by_key = {o.key.path: o for o in outcomes}
    assert by_key["voice.speech_rate"].accepted
    assert not by_key["screen.quality"].accepted
    assert get_config_value(manager.config, good) == 1.5
    assert get_config_value(manager.config, bad) == 80

    fresh = ConfigManager(config_path=manager.config_path)
    assert get_config_value(fresh.config, good) == 1.5
    assert get_config_value(fresh.config, bad) == 80


def test_no_accepted_changes_means_no_save(manager):
    """An all-rejected batch must not touch the config file at all."""
    k = key("screen", "quality", value_type="int")
    before = manager.config_path.read_text()
    apply_settings(manager, {k: "999"})
    assert manager.config_path.read_text() == before


def test_accepted_change_notifies_hot_apply_callbacks(manager):
    """Hot-apply dispatch: the same registry reload() notifies fires on apply."""
    seen: list[object] = []
    manager.on_reload(seen.append)
    k = key("barge_in", "min_confidence", value_type="float", hot_apply=True)
    apply_settings(manager, {k: "0.9"})
    assert len(seen) == 1 and seen[0] is manager.config


def test_rejected_change_does_not_notify(manager):
    seen: list[object] = []
    manager.on_reload(seen.append)
    k = key("screen", "quality", value_type="int")
    apply_settings(manager, {k: "999"})
    assert seen == []


def test_reload_is_functional_and_dispatches(manager):
    """The reload path (never called before W2) re-reads the file and notifies."""
    seen: list[object] = []
    manager.on_reload(seen.append)
    manager.config_path.write_text(MINIMAL_YAML.replace("quality: 80", "quality: 95"))
    manager.reload()
    assert manager.config.screen.quality == 95
    assert len(seen) == 1

    # Isolation: a failing observer must not break the reload or its siblings.
    def boom(_cfg):
        raise RuntimeError("observer bug")

    manager.on_reload(boom)
    assert manager.reload().screen.quality == 95  # still reloaded
    assert len(seen) == 2  # first observer still ran


def test_coerce_optional_int_accepts_none(manager):
    k = key("microphone", "input_device", value_type="int?")
    assert coerce("none", "int?") is None
    assert coerce("0", "int?") == 0
    outcomes = apply_settings(manager, {k: "none"})
    assert outcomes[0].accepted and outcomes[0].new_value is None
