"""Native settings window (W2 D2) — AppKit, darwin-only.

Renders the pure settings model (:mod:`assistant_app.hud.settings_model`):
one row per discovered setting (path, current value, hot/restart tag, env
override name), Apply validates through the config system — validation-
rejected input is reverted and reported, never written — and Done closes.

Threading contract: ``open_settings_window()`` is invoked from the HUD menu,
i.e. on the main thread inside the live NSApplication run loop. The window
just goes key; there is no nested run loop. A module-level controller
reference keeps the open window alive.

Importing this module on a non-darwin platform raises ImportError with a
clear message (tests assert this explicitly — never a silent skip).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from assistant_app.hud.settings_model import (
    SettingKey,
    apply_settings,
    discover_setting_keys,
    get_config_value,
)

if TYPE_CHECKING:
    from assistant_app.utils.config import ConfigManager

logger = logging.getLogger(__name__)

try:
    from AppKit import (
        NSAlert,
        NSBackingStoreBuffered,
        NSButton,
        NSButtonTypeSwitch,
        NSEdgeInsetsMake,
        NSFont,
        NSScrollView,
        NSStackView,
        NSTextField,
        NSUserInterfaceLayoutOrientationHorizontal,
        NSUserInterfaceLayoutOrientationVertical,
        NSViewWidthSizable,
        NSWindow,
        NSWindowStyleMaskClosable,
        NSWindowStyleMaskMiniaturizable,
        NSWindowStyleMaskResizable,
        NSWindowStyleMaskTitled,
    )
    from Foundation import NSMakeRect, NSObject
except ImportError as exc:  # pragma: no cover - exercised on non-darwin only
    raise ImportError(
        "assistant_app.hud.settings_window is darwin-only: AppKit/PyObjC is "
        "unavailable on this platform. The pure settings model "
        "(assistant_app.hud.settings_model) has no such dependency."
    ) from exc

_WINDOW_W, _WINDOW_H = 720, 580


def _display(value: Any) -> str:
    """Value → UI text; optional fields show the literal 'none' for None."""
    return "none" if value is None else str(value)


class _SettingsController(NSObject):
    """Window controller: renders the discovered keys, applies via the model."""

    def initWithManager_consentProvider_(self, config_manager: ConfigManager, consent_provider=None):
        # PyObjC two-phase init: super().init() may return a different
        # instance (or None), so configure and return that object directly.
        controller = super().init()
        if controller is None:
            return None
        controller._manager = config_manager
        # W2 S1: the live provider (the orchestrator) backing the consent
        # toggle row — None (headless/tests) renders config rows only.
        controller._consent_provider = consent_provider
        controller._consent_checkbox = None
        controller._field_map = {}
        controller.window = None
        return controller

    # --- UI construction ---------------------------------------------------

    def build_window(self) -> None:
        keys = discover_setting_keys(self._manager.config)

        content = NSStackView.alloc().init()
        content.setOrientation_(NSUserInterfaceLayoutOrientationVertical)
        content.setSpacing_(4)
        content.setEdgeInsets_(NSEdgeInsetsMake(16, 16, 16, 16))

        # W2 S1: the consent toggle leads the window — the one setting that
        # governs what leaves the machine outranks every preference below.
        if self._consent_provider is not None:
            content.addArrangedSubview_(self._consent_header())
            content.addArrangedSubview_(self._consent_row())

        current_section: str | None = None
        for key in keys:
            if key.section != current_section:
                current_section = key.section
                content.addArrangedSubview_(self._section_header(current_section))
            content.addArrangedSubview_(self._make_row(key))

        content.addArrangedSubview_(self._button_row())

        scroll = NSScrollView.alloc().initWithFrame_(NSMakeRect(0, 0, _WINDOW_W, _WINDOW_H))
        scroll.setDocumentView_(content)
        scroll.setHasVerticalScroller_(True)
        content.setAutoresizingMask_(NSViewWidthSizable)

        window = NSWindow.alloc().initWithContentRect_styleMask_title_backing_defer_(
            NSMakeRect(0, 0, _WINDOW_W, _WINDOW_H),
            NSWindowStyleMaskTitled
            | NSWindowStyleMaskClosable
            | NSWindowStyleMaskMiniaturizable
            | NSWindowStyleMaskResizable,
            "SpectraVoice Settings",
            NSBackingStoreBuffered,
            False,
        )
        window.setContentView_(scroll)
        window.setReleasedWhenClosed_(False)
        window.center()
        self.window = window

    def _section_header(self, section: str) -> Any:
        title = section if section else "general"
        label = NSTextField.labelWithString_(f"— {title} —")
        label.setFont_(NSFont.boldSystemFontOfSize_(12))
        return label

    def _consent_header(self) -> Any:
        label = NSTextField.labelWithString_("— consent —")
        label.setFont_(NSFont.boldSystemFontOfSize_(12))
        return label

    def _consent_row(self) -> Any:
        row = NSStackView.alloc().init()
        row.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        row.setSpacing_(8)

        label = NSTextField.labelWithString_("screen upload consent (runtime grant)")
        label.setToolTip_(
            "When checked, every query sends a screenshot of your screen to "
            "your LLM provider (local or cloud). Off or paused always keeps "
            "upload blocked. The grant lives in the running app — the "
            "SPECTRAVOICE_SCREEN_CONSENT env default applies again on next launch."
        )
        label.widthAnchor().constraintEqualToConstant_(240.0).setActive_(True)
        row.addArrangedSubview_(label)

        granted = bool(self._consent_provider.screen_consent_enabled())
        checkbox = NSButton.buttonWithTitle_target_action_("Granted", self, "consentAction:")
        checkbox.setButtonType_(NSButtonTypeSwitch)
        checkbox.setState_(1 if granted else 0)
        self._consent_checkbox = checkbox
        row.addArrangedSubview_(checkbox)

        status = NSTextField.labelWithString_("runtime · resets to env default on relaunch")
        status.setFont_(NSFont.systemFontOfSize_(10))
        status.widthAnchor().constraintEqualToConstant_(200.0).setActive_(True)
        row.addArrangedSubview_(status)
        return row

    def refresh_consent(self) -> None:
        """Re-read the gate (a HUD-side grant while this window was open)."""
        if self._consent_checkbox is not None and self._consent_provider is not None:
            granted = bool(self._consent_provider.screen_consent_enabled())
            self._consent_checkbox.setState_(1 if granted else 0)

    def _make_row(self, key: SettingKey) -> Any:
        row = NSStackView.alloc().init()
        row.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        row.setSpacing_(8)

        label = NSTextField.labelWithString_(key.path)
        label.setToolTip_(f"{key.path}  ({key.value_type})")
        label.widthAnchor().constraintEqualToConstant_(240.0).setActive_(True)
        row.addArrangedSubview_(label)

        field = NSTextField.textFieldWithString_(_display(get_config_value(self._manager.config, key)))
        field.widthAnchor().constraintEqualToConstant_(220.0).setActive_(True)
        self._field_map[key] = field
        row.addArrangedSubview_(field)

        # Status column: hot-apply / restart-required, plus the env override.
        tags = ["hot" if key.hot_apply else "restart"]
        if key.env_var:
            tags.append(key.env_var)
        status = NSTextField.labelWithString_(" · ".join(tags))
        status.setFont_(NSFont.systemFontOfSize_(10))
        status.widthAnchor().constraintEqualToConstant_(200.0).setActive_(True)
        row.addArrangedSubview_(status)
        return row

    def _button_row(self) -> Any:
        row = NSStackView.alloc().init()
        row.setOrientation_(NSUserInterfaceLayoutOrientationHorizontal)
        row.setSpacing_(12)
        apply_btn = NSButton.buttonWithTitle_target_action_("Apply", self, "applyAction:")
        done_btn = NSButton.buttonWithTitle_target_action_("Done", self, "doneAction:")
        row.addArrangedSubview_(apply_btn)
        row.addArrangedSubview_(done_btn)
        return row

    # --- Actions (main thread, inside the HUD run loop) ----------------------

    def applyAction_(self, sender) -> None:
        changes: dict[SettingKey, str] = {}
        for key, field in self._field_map.items():
            new = str(field.stringValue())
            if new != _display(get_config_value(self._manager.config, key)):
                changes[key] = new
        if not changes:
            self._alert("Settings", "No changes to apply.")
            return

        outcomes = apply_settings(self._manager, changes)
        accepted = [o for o in outcomes if o.accepted]
        rejected = [o for o in outcomes if not o.accepted]

        lines = [f"Applied {len(accepted)} change(s):"]
        lines += [
            f"  {o.key.path} — {'live (hot)' if o.key.hot_apply else 'saved (restart required)'}"
            for o in accepted
        ]
        lines += [f"  REJECTED {o.key.path}: {o.reason}" for o in rejected]
        logger.info(f"⚙️ Settings applied: {len(accepted)} accepted, {len(rejected)} rejected")
        self._alert("Settings", "\n".join(lines))

        # Refresh fields to the normalized post-validation values.
        for key, field in self._field_map.items():
            field.setStringValue_(_display(get_config_value(self._manager.config, key)))

    def doneAction_(self, sender) -> None:
        if self.window is not None:
            self.window.orderOut_(sender)

    def consentAction_(self, sender) -> None:
        """W2 S1: the checkbox toggles the live gate via the provider — the
        same set_screen_consent path the HUD menu uses (never a config write;
        consent is gate state, not a config key)."""
        if self._consent_provider is None:
            return
        enabled = sender.state() != 0  # NSControlStateValueOn
        self._consent_provider.set_screen_consent(bool(enabled))
        logger.info(f"👁️ Settings: screen consent {'granted' if enabled else 'revoked'}")

    def windowWillClose_(self, notification) -> None:  # NSWindow delegate
        _release_controller(self)

    def _alert(self, title: str, message: str) -> None:
        alert = NSAlert.alloc().init()
        alert.setMessageText_(title)
        alert.setInformativeText_(message)
        alert.runModal()


# Module-level reference keeps the open window/controller alive.
_CONTROLLER: _SettingsController | None = None


def _release_controller(controller: _SettingsController) -> None:
    global _CONTROLLER
    if _CONTROLLER is controller:
        _CONTROLLER = None


def open_settings_window(config_manager: ConfigManager, consent_provider=None) -> None:
    """Open (or focus) the native settings window for this config.

    Must be called on the main thread — the HUD menu is the entry point, so
    the NSApplication run loop is already live. ``consent_provider`` (W2 S1)
    is the live orchestrator backing the consent toggle row; omitting it
    renders the config keys only.
    """
    global _CONTROLLER
    if _CONTROLLER is not None and _CONTROLLER.window is not None:
        _CONTROLLER.refresh_consent()  # the gate may have moved since open
        _CONTROLLER.window.makeKeyAndOrderFront_(None)
        return
    controller = _SettingsController.alloc().initWithManager_consentProvider_(config_manager, consent_provider)
    controller.build_window()
    controller.window.makeKeyAndOrderFront_(None)
    _CONTROLLER = controller
