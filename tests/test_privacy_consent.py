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


class TestConsentObservers:
    """W2 observer hook: state changes reach subscribers without polling."""

    def test_observer_receives_effective_state_on_change(self):
        gate = PrivacyConsent(consented=False)
        seen = []
        gate.add_observer(lambda consented, paused: seen.append((consented, paused)))

        gate.grant_consent()
        gate.pause()
        gate.resume()
        gate.revoke_consent()

        assert seen == [(True, False), (True, True), (True, False), (False, False)]

    def test_no_notification_without_state_change(self):
        gate = PrivacyConsent(consented=True)
        calls = []
        gate.add_observer(lambda c, p: calls.append((c, p)))

        gate.grant_consent()  # already granted
        gate.pause()
        gate.pause()  # already paused
        gate.resume()
        gate.resume()  # already resumed

        assert calls == [(True, True), (True, False)]

    def test_remove_observer_stops_notifications(self):
        gate = PrivacyConsent()
        calls = []
        observer = lambda c, p: calls.append((c, p))
        gate.add_observer(observer)
        gate.grant_consent()
        gate.remove_observer(observer)
        gate.revoke_consent()

        assert calls == [(True, False)]

    def test_broken_observer_does_not_break_the_gate(self):
        """The privacy gate must keep gating even if a HUD callback crashes."""
        gate = PrivacyConsent()

        def explode(consented, paused):
            raise RuntimeError("observer bug")

        gate.add_observer(explode)
        gate.grant_consent()
        gate.pause()

        assert gate.consented is True
        assert gate.paused is True
        assert gate.screen_upload_allowed is False

    def test_observer_reads_fresh_state(self):
        gate = PrivacyConsent()
        observed = []
        gate.add_observer(lambda c, p: observed.append(gate.screen_upload_allowed))
        gate.grant_consent()
        gate.pause()

        assert observed == [True, False]

    def test_observer_runs_outside_the_gate_lock(self):
        """An observer may call gate properties — no deadlock (lock is released
        before callbacks run)."""
        gate = PrivacyConsent()
        results = []
        gate.add_observer(lambda c, p: results.append(gate.paused))
        gate.pause()

        assert results == [True]

    def test_threaded_mutations_notify_exactly_on_change(self):
        gate = PrivacyConsent(consented=True)
        count = []
        gate.add_observer(lambda c, p: count.append(1))

        def churn():
            for _ in range(200):
                gate.pause()
                gate.resume()

        threads = [threading.Thread(target=churn) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # 1600 mutations (4 threads x 200 x pause+resume), each a real change:
        # every one of them must notify.
        assert len(count) == 1600


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
            # Atomic snapshot: asserting across three separate property reads
            # would be a torn read (state can flip between acquisitions) — a
            # gate bug the snapshot is required to detect.
            consented, paused, allowed = consent.snapshot()
            assert allowed == (consented and not paused)
    finally:
        stop.set()
        for t in threads:
            t.join(timeout=2)
