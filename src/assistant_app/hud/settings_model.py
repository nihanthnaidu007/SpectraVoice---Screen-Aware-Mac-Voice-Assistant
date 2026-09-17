"""Settings surface model for the native settings UI (W2 D2).

Pure logic, no AppKit: key discovery over every config section, the explicit
hot-apply vs restart-required taxonomy, coercion, and validation-gated
application. The AppKit window (settings_window.py) renders this model.

Every mutation goes through the config system: values are coerced, applied
tentatively, and validated with ConfigManager.validate(). A change that
introduces validation issues is reverted and reported — rejected input is
never written to the running config nor persisted to disk.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from assistant_app.utils.config import ConfigManager

if TYPE_CHECKING:
    from assistant_app.utils.config import AssistantConfig

# Sections rendered by the settings UI, in display order. `modes` profiles are
# discovered dynamically from the live config; top-level scalars (mode, debug)
# use section "".
SECTION_ORDER = (
    "voice",
    "screen",
    "llm",
    "api",
    "hotkeys",
    "dictation",
    "logging",
    "safety",
    "barge_in",
    "microphone",
    "tts",
    "supervisor",
)

# Explicit hot-apply taxonomy: {(section, field)} keys that take effect in the
# running process via the ConfigManager change-notification callbacks. Every
# other key is restart-required — components (TTS engine, screen capture,
# LLM provider, hotkey listener, mic setup) hold constructed references and
# re-reading them requires a restart. An unknown section here is a bug, not a
# silent default: the taxonomy must stay a superset of the callbacks actually
# registered by the assistant.
HOT_APPLY_KEYS: frozenset[tuple[str, str]] = frozenset(
    {
        ("logging", "level"),
        ("barge_in", "enabled"),
        ("barge_in", "min_confidence"),
        ("barge_in", "min_words"),
        ("barge_in", "min_chars"),
    }
)


@dataclass(frozen=True)
class SettingKey:
    """One editable setting, addressed as section.field (section "" = top level)."""

    section: str
    field: str
    hot_apply: bool
    env_var: str | None
    value_type: str  # "bool" | "int" | "float" | "str" | "dict"

    @property
    def path(self) -> str:
        """Dot-notation path for display and change records."""
        return f"{self.section}.{self.field}" if self.section else self.field


@dataclass(frozen=True)
class SettingOutcome:
    """Result of applying one requested change."""

    key: SettingKey
    accepted: bool
    old_value: Any
    new_value: Any | None
    reason: str = ""


def _type_label(annotation: Any) -> str:
    """Normalize a dataclass annotation to a coercible type label."""
    label = annotation if isinstance(annotation, str) else getattr(annotation, "__name__", str(annotation))
    label = label.replace("typing.", "")
    if label.startswith("Optional[") and label.endswith("]"):
        label = label[len("Optional[") : -1]
    return label


def _section_fields(section: Any) -> list[dataclasses.Field]:
    return dataclasses.fields(section)


def discover_setting_keys(config: AssistantConfig) -> list[SettingKey]:
    """Enumerate every editable setting, including env-only overrides.

    Env-only keys are config fields that also (or only) have an environment
    override — the UI shows the variable name so users know precedence.
    """
    env_table = ConfigManager.env_mapping_table()
    keys: list[SettingKey] = []

    def add(section: str, f_name: str, annotation: Any) -> None:
        keys.append(
            SettingKey(
                section=section,
                field=f_name,
                hot_apply=(section, f_name) in HOT_APPLY_KEYS,
                env_var=env_table.get((section, f_name)),
                value_type=_type_label(annotation),
            )
        )

    for section_name in SECTION_ORDER:
        for f in _section_fields(getattr(config, section_name)):
            add(section_name, f.name, f.type)

    # Mode profiles are dataclasses too, discovered from the live config.
    for profile_name, profile in config.modes.items():
        for f in _section_fields(profile):
            add(f"modes.{profile_name}", f.name, f.type)

    # Top-level scalars (mode, debug) — section "".
    add("", "mode", "str")
    add("", "debug", "bool")
    return keys


def get_config_value(config: AssistantConfig, key: SettingKey) -> Any:
    if key.section == "":
        return getattr(config, key.field)
    if key.section.startswith("modes."):
        profile = config.modes[key.section.split(".", 1)[1]]
        return getattr(profile, key.field)
    return getattr(getattr(config, key.section), key.field)


def set_config_value(config: AssistantConfig, key: SettingKey, value: Any) -> None:
    if key.section == "":
        setattr(config, key.field, value)
    elif key.section.startswith("modes."):
        profile = config.modes[key.section.split(".", 1)[1]]
        setattr(profile, key.field, value)
    else:
        setattr(getattr(config, key.section), key.field, value)


def coerce(raw: str, value_type: str) -> Any:
    """Coerce a UI string to the field's type. Raises ValueError/TypeError."""
    if value_type == "bool":
        low = raw.strip().lower()
        if low in {"1", "true", "yes", "on"}:
            return True
        if low in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"not a boolean: {raw!r}")
    if value_type == "int":
        return int(raw.strip())
    if value_type == "float":
        return float(raw.strip())
    if value_type == "dict":
        raise ValueError("structured value — edit config.yaml directly")
    return raw


def apply_settings(
    config_manager: ConfigManager,
    changes: dict[SettingKey, str],
) -> list[SettingOutcome]:
    """Apply UI edits through the config system with validation gating.

    Each change is coerced, applied tentatively, and re-validated against the
    baseline issues; a change introducing new validation issues is reverted
    and reported. Only accepted changes are persisted (save) and dispatched
    to the hot-apply callbacks (the same registry reload() notifies).
    """
    config = config_manager.config
    baseline = set(config_manager.validate())
    outcomes: list[SettingOutcome] = []
    accepted: list[tuple[SettingKey, Any]] = []

    for key, raw in changes.items():
        old = get_config_value(config, key)
        try:
            value = coerce(raw, key.value_type)
        except (TypeError, ValueError) as exc:
            outcomes.append(SettingOutcome(key, False, old, None, f"invalid {key.value_type}: {exc}"))
            continue
        set_config_value(config, key, value)
        new_issues = set(config_manager.validate()) - baseline
        if new_issues:
            set_config_value(config, key, old)  # revert — rejected, never written
            outcomes.append(SettingOutcome(key, False, old, value, "; ".join(sorted(new_issues))))
        else:
            outcomes.append(SettingOutcome(key, True, old, value))
            accepted.append((key, value))

    if accepted:
        config_manager.save()  # persist only accepted changes
        config_manager.notify_config_changed(config)  # hot-apply dispatch (same registry as reload())

    return outcomes
