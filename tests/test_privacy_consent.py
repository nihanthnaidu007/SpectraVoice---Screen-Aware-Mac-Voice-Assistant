"""Tests for the screen-sharing privacy consent gate (Wave 0 privacy).

Platform-independent: the gate is exercised through Assistant with a recording
fake provider — no Mac, audio, or network required. Proves that screen capture
content is NOT sent to the LLM provider unless consent was granted, and that
the runtime pause toggle blocks upload even when consent is on.
"""

import os
import sys
import threading

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.core.assistant import Assistant
from assistant_app.core.consent import CONSENT_ENV_VAR, PrivacyConsent
from assistant_app.llm.base import LLMProvider
from assistant_app.llm.types import LLMConfig, LLMResponse


class RecordingProvider(LLMProvider):
    """Fake provider that records which generation path was used."""

    def __init__(self):
        super().__init__(config=LLMConfig())
        self.vision_calls = 0
        self.text_calls = 0
        self.last_image = None

    @property
    def name(self) -> str:
        return "recording-fake"

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
        self.last_image = image_data
        return LLMResponse(content="vision answer", provider=self.name, model=self.model_name)


def make_assistant(consented: bool = False):
    provider = RecordingProvider()
    consent = PrivacyConsent(consented=consented)
    assistant = Assistant(
        provider=provider,
        privacy_consent=consent,
        enable_web_search=False,
        enable_tools=False,
    )
    return assistant, provider, consent


def test_consent_defaults_to_off():
    assert PrivacyConsent().screen_upload_allowed is False


def test_no_consent_blocks_screen_upload():
    assistant, provider, _ = make_assistant(consented=False)
    response = assistant.process("what's on my screen?", b"fake-image-bytes")
    assert provider.vision_calls == 0, "screenshot must NOT be sent without consent"
    assert provider.text_calls == 1, "query must still be answered, without vision"
    assert response == "text-only answer"


def test_granted_consent_allows_vision():
    assistant, provider, consent = make_assistant(consented=True)
    response = assistant.process("what's on my screen?", b"fake-image-bytes")
    assert consent.screen_upload_allowed
    assert provider.vision_calls == 1
    assert provider.last_image == b"fake-image-bytes"
    assert response == "vision answer"


def test_set_screen_consent_toggles():
    assistant, _provider, _ = make_assistant(consented=False)
    assistant.set_screen_consent(True)
    assert assistant.privacy_consent.screen_upload_allowed
    assistant.set_screen_consent(False)
    assert not assistant.privacy_consent.screen_upload_allowed


def test_pause_toggle_blocks_even_with_consent():
    assistant, provider, _ = make_assistant(consented=True)
    assistant.pause_screen_sharing()
    assistant.process("read my screen", b"img")
    assert provider.vision_calls == 0, "paused gate must block upload despite consent"
    assert provider.text_calls == 1
    assistant.resume_screen_sharing()
    assistant.process("read my screen", b"img")
    assert provider.vision_calls == 1


def test_resume_without_consent_stays_blocked():
    _assistant, _, consent = make_assistant(consented=False)
    consent.pause()
    consent.resume()
    assert not consent.screen_upload_allowed, "resume must not grant consent"


def test_consent_from_env():
    cases = {
        "1": True,
        "true": True,
        "Yes": True,
        "on": True,
        "0": False,
        "": False,
        "false": False,
        "random": False,
    }
    for value, expected in cases.items():
        consent = PrivacyConsent.from_env({CONSENT_ENV_VAR: value})
        assert consent.consented is expected, f"{CONSENT_ENV_VAR}={value!r}"
    # Absent variable → OFF
    assert PrivacyConsent.from_env({}).consented is False
    assert PrivacyConsent.from_env({CONSENT_ENV_VAR: "1"}).screen_upload_allowed


def test_consent_gate_is_thread_safe():
    consent = PrivacyConsent(consented=True)
    stop = threading.Event()

    def flip():
        while not stop.is_set():
            consent.pause()
            consent.resume()

    threads = [threading.Thread(target=flip, daemon=True) for _ in range(8)]
    for t in threads:
        t.start()
    try:
        for _ in range(2000):
            allowed = consent.screen_upload_allowed
            # Invariant: upload is allowed iff consented and not paused.
            assert allowed == (consent.consented and not consent.paused)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2)
