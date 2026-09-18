"""W2 S1 tests — consent you can see.

The in-UI grant/revoke surface (HUD menu item + settings toggle) must reach
the SAME PrivacyConsent gate every consumer reads, with semantics unchanged:
off/paused still blocks vision upload (the phase-3 W1 vision_calls==0
standard). Pure logic only — the AppKit menu item itself is verified by the
structural menubar checks plus the manual macOS checklist.

Covered here:
- Pure labels: consent_item_label mirrors the gate's three privacy states.
- Protocol surface: the menu's new selectors exist on HUDActions (a missing
  method is a main-thread AttributeError on a real Mac, not a CI failure).
- Gate dispatch: set_screen_consent (the S1 entry point's target) flips the
  gate, fires the phase-2 W2 observer hook, and a grant while PAUSED still
  leaves vision upload blocked — vision_calls == 0.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.core.assistant import Assistant
from assistant_app.core.consent import PrivacyConsent
from assistant_app.hud.actions import HUDActions
from assistant_app.hud.state import HUDStateMachine, Privacy, consent_item_label, privacy_from_consent
from assistant_app.llm.base import LLMProvider
from assistant_app.llm.types import LLMConfig, LLMResponse

PROJECT_ROOT = Path(project_root)
SRC = PROJECT_ROOT / "src" / "assistant_app"


class TestConsentItemLabel:
    def test_off_offers_the_grant(self):
        assert consent_item_label(Privacy.OFF) == "Grant Screen Consent…"

    def test_granted_states_offer_revocation(self):
        assert consent_item_label(Privacy.ON) == "Revoke Screen Consent"
        assert consent_item_label(Privacy.PAUSED) == "Revoke Screen Consent"

    def test_label_follows_the_gate_mapping(self):
        for consented, paused in [(False, False), (True, False), (True, True)]:
            assert consent_item_label(privacy_from_consent(consented, paused)) in (
                "Grant Screen Consent…",
                "Revoke Screen Consent",
            )


class TestHUDStateMachineRendersConsentStates:
    """The observer hook (phase-2 W2) must carry a runtime grant to the HUD
    privacy axis — the menu label and icon read from it."""

    def test_runtime_grant_reaches_the_hud_privacy_axis(self):
        gate = PrivacyConsent(consented=False)
        machine = HUDStateMachine()
        gate.add_observer(machine.set_privacy)
        gate.grant_consent()
        assert machine.snapshot().privacy is Privacy.ON
        gate.revoke_consent()
        assert machine.snapshot().privacy is Privacy.OFF


class TestConsentActionSurface:
    def test_protocol_declares_consent_actions(self):
        assert callable(getattr(HUDActions, "toggle_screen_consent", None))
        assert callable(getattr(HUDActions, "current_screen_consent", None))


class TestMenubarConsentStructure:
    """AppKit is unimportable on Linux CI — structural checks on the source:
    the consent item sits between the pause toggle and the meeting toggle,
    and its selector dispatches actions.toggle_screen_consent."""

    def _source(self) -> str:
        return (SRC / "hud" / "menubar.py").read_text(encoding="utf-8")

    def test_consent_item_sits_between_pause_and_meeting(self):
        source = self._source()
        pause_pos = source.index('"onPauseToggle:"')
        consent_pos = source.index('"onConsentToggle:"')
        meeting_pos = source.index('"onMeetingToggle:"')
        assert pause_pos < consent_pos < meeting_pos

    def test_consent_selector_dispatches_toggle_screen_consent(self):
        tree = ast.parse(self._source())
        handlers = [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "onConsentToggle_"
        ]
        assert len(handlers) == 1
        assert "toggle_screen_consent" in ast.unparse(handlers[0])

    def test_consent_item_label_is_snapshot_driven(self):
        source = self._source()
        assert "consent_item_label" in source
        assert "consent_item_label(snapshot.privacy)" in source

    def test_consent_item_explains_what_is_shared(self):
        # S1's contract: the control says what will be shared. The tooltip is
        # the menu item's explanation surface.
        source = self._source()
        assert "setToolTip_" in source
        assert "screenshot" in source.lower()


class TestConsentDispatchKeepsGateSemantics:
    """The S1 entry point targets Assistant.set_screen_consent — the same
    gate, so off/paused keeps vision_calls == 0 (phase-3 W1 standard)."""

    def test_set_screen_consent_fires_the_observer_hook(self):
        gate = PrivacyConsent(consented=False)
        seen: list[tuple[bool, bool]] = []
        gate.add_observer(lambda consented, paused: seen.append((consented, paused)))
        consent = _grant_through_set_screen_consent(gate, True)
        assert seen[-1] == (True, False)
        assert consent.screen_upload_allowed is True

    def test_grant_while_paused_still_blocks_vision(self):
        gate = PrivacyConsent(consented=False)
        gate.pause()  # the HUD pause state, pre-consent
        consent = _grant_through_set_screen_consent(gate, True)
        assert consent.consented is True
        assert consent.screen_upload_allowed is False, "paused gate must block upload despite the runtime grant"

    def test_revocation_blocks_vision_again(self):
        gate = PrivacyConsent(consented=True)
        consent = _grant_through_set_screen_consent(gate, False)
        assert consent.screen_upload_allowed is False


class _RecordingProvider(LLMProvider):
    """Records which generation path was used (the W1 vision_calls standard)."""

    def __init__(self):
        super().__init__(config=LLMConfig())
        self.vision_calls = 0
        self.text_calls = 0

    @property
    def name(self) -> str:
        return "consent-ui-fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    @property
    def supports_vision(self) -> bool:
        return True

    @property
    def supports_tools(self) -> bool:
        return False

    def is_available(self) -> tuple[bool, str]:
        return (True, "fake provider ready")

    def generate(self, messages, **kwargs) -> LLMResponse:
        self.text_calls += 1
        return LLMResponse(content="text-only answer", provider=self.name, model=self.model_name)

    def generate_with_vision(self, prompt, image_data, **kwargs) -> LLMResponse:
        self.vision_calls += 1
        return LLMResponse(content="vision answer", provider=self.name, model=self.model_name)


def _grant_through_set_screen_consent(gate: PrivacyConsent, enabled: bool) -> PrivacyConsent:
    """The exact call chain the S1 UI entry points use (HUD menu and settings
    row both land on Assistant.set_screen_consent, which mutates the gate)."""
    provider = _RecordingProvider()
    assistant = Assistant(provider=provider, privacy_consent=gate, enable_web_search=False, enable_tools=False)
    assistant.set_screen_consent(enabled)
    return gate


def test_off_or_paused_keeps_vision_calls_at_zero():
    provider = _RecordingProvider()
    gate = PrivacyConsent(consented=False)
    assistant = Assistant(provider=provider, privacy_consent=gate, enable_web_search=False, enable_tools=False)
    # Grant through the S1 path, then pause (the HUD pause toggle) — the
    # combined state must keep the screenshot upload blocked.
    assistant.set_screen_consent(True)
    gate.pause()
    assistant.process("what's on my screen?", b"fake-image-bytes")
    assert provider.vision_calls == 0, "off/paused must keep vision upload blocked"
    assert provider.text_calls == 1
