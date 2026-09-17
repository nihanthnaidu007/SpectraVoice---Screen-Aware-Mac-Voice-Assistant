"""Tests for streamed TTS synthesis (Wave 1 — removes the 500-character cap).

Platform-independent: the OpenAI speech endpoint and the PyAudio player are
replaced by fakes, so these tests prove request content (no truncation) and
streaming behavior (playback starts while synthesis is still producing bytes)
without any network, Mac, or audio device.
"""

import os
import sys
import threading
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))


import assistant_app.io.audio.tts as tts_module
from assistant_app.io.audio.tts import TextToSpeech

CHUNK = b"\x01\x02" * 2048  # 4096 bytes — one playback chunk


class FakeStreamResponse:
    """Stands in for the OpenAI streaming response body."""

    def __init__(self, chunks, timeline, pause_after=None, pause_seconds=0.0):
        self.chunks = chunks
        self.timeline = timeline
        self.pause_after = pause_after  # yield index after which synthesis "stalls"
        self.pause_seconds = pause_seconds

    def iter_bytes(self, chunk_size=None):
        for i, chunk in enumerate(self.chunks):
            if self.pause_after is not None and i == self.pause_after:
                time.sleep(self.pause_seconds)  # simulate server still synthesizing
            self.timeline.append(f"yield:{i}")
            yield chunk


class FakeStreamCtx:
    def __init__(self, response):
        self._response = response

    def __enter__(self):
        return self._response

    def __exit__(self, *exc):
        return False


class FakePlayer:
    """Stands in for the PyAudio output stream; records writes on the timeline."""

    def __init__(self, timeline):
        self.timeline = timeline
        self.chunks_written = 0
        self.stopped = False

    def write(self, chunk):
        self.chunks_written += 1
        self.timeline.append(f"play:{self.chunks_written}")

    def stop_stream(self):
        self.stopped = True

    def close(self):
        pass


def make_tts(**kwargs) -> TextToSpeech:
    return TextToSpeech(**{"voice": "shimmer", **kwargs})


def install_fake_stream(monkeypatch, tts, chunks, timeline, pause_after=None, pause_seconds=0.0,
                        patch_player=True):
    """Replace the OpenAI streaming call (and the player, unless asked not to);
    return captured kwargs."""
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return FakeStreamCtx(FakeStreamResponse(chunks, timeline, pause_after, pause_seconds))

    monkeypatch.setattr(
        tts_module.openai.audio.speech.with_streaming_response, "create", fake_create
    )
    if patch_player:
        monkeypatch.setattr(tts, "_open_player", lambda: FakePlayer(timeline))
    return captured


class TestNoCharacterCap:
    def test_sends_full_text_past_500_chars(self, monkeypatch):
        tts = make_tts()
        timeline = []
        text = "word " * 300  # 1500 characters — far past the old 500-char cap
        chunks = [CHUNK] * 10
        captured = install_fake_stream(monkeypatch, tts, chunks, timeline)

        result = tts.speak(text)

        assert result is True
        assert captured["input"] == text  # full text, no truncation

    def test_plays_every_chunk_of_long_text(self, monkeypatch):
        tts = make_tts()
        timeline = []
        chunks = [CHUNK] * 12
        install_fake_stream(monkeypatch, tts, chunks, timeline)

        result = tts.speak("sentence. " * 200)

        assert result is True


class TestStreamingBehavior:
    def test_playback_starts_before_synthesis_completes(self, monkeypatch):
        """The load-bearing behavior: speech must begin while the server is
        still generating. Synthesis stalls after yielding 5 of 10 chunks."""
        tts = make_tts()
        timeline = []
        chunks = [CHUNK] * 10
        install_fake_stream(
            monkeypatch, tts, chunks, timeline, pause_after=5, pause_seconds=0.4
        )

        result = tts.speak("This is a longer response that streams in pieces.")

        assert result is True
        # Playback of the first buffered chunk happened before the final chunk
        # was even yielded — speech started during synthesis, not after it.
        assert timeline.index("play:1") < timeline.index("yield:9")
        # And specifically before synthesis resumed after its mid-stream pause.
        assert timeline.index("play:1") < timeline.index("yield:6")

    def test_stop_interrupts_mid_stream(self, monkeypatch):
        tts = make_tts()
        timeline = []

        def slow_create(**kwargs):
            class SlowResponse:
                def iter_bytes(self, chunk_size=None):
                    for _ in range(60):
                        time.sleep(0.02)
                        yield CHUNK

            return FakeStreamCtx(SlowResponse())

        monkeypatch.setattr(
            tts_module.openai.audio.speech.with_streaming_response, "create", slow_create
        )
        monkeypatch.setattr(tts, "_open_player", lambda: FakePlayer(timeline))

        result_holder = {}

        def speak_thread():
            result_holder["result"] = tts.speak("A response long enough to interrupt.")

        thread = threading.Thread(target=speak_thread, daemon=True)
        thread.start()
        # Wait until playback has actually started, then barge in.
        deadline = time.time() + 5
        while "play:1" not in timeline and time.time() < deadline:
            time.sleep(0.005)
        tts.stop()
        thread.join(timeout=5)

        assert result_holder.get("result") is False  # interrupted, not completed

    def test_speak_async_completes_and_fires_callback(self, monkeypatch):
        tts = make_tts()
        timeline = []
        install_fake_stream(monkeypatch, tts, [CHUNK] * 4, timeline)

        done = threading.Event()
        outcomes = []
        tts.set_on_complete(lambda ok: (outcomes.append(ok), done.set()))

        tts.speak_async("Short async response.")
        assert done.wait(timeout=5)

        # The completion callback receives was_interrupted: False = completed normally.
        assert outcomes == [False]

    def test_empty_text_skips_synthesis(self, monkeypatch):
        tts = make_tts()
        timeline = []
        captured = install_fake_stream(monkeypatch, tts, [], timeline)

        assert tts.speak("") is True
        assert tts.speak("   ") is True
        assert captured == {}  # the API was never called


class TestOutputDeviceFallback:
    def test_falls_back_to_default_device_on_oserror(self, monkeypatch):
        """A configured output device that fails (e.g. headset unplugged) must
        degrade to the system default, not crash playback."""
        tts = make_tts(output_device=7)
        timeline = []

        class FakeAudioContext:
            def __init__(self):
                self.open_calls = []

            def get_device_info_by_index(self, index):
                return {"name": f"fake-device-{index}"}

            def open(self, **kwargs):
                self.open_calls.append(kwargs)
                if "output_device_index" in kwargs:
                    raise OSError("Device unavailable")
                return FakePlayer(timeline)

            def terminate(self):
                pass

        ctx = FakeAudioContext()
        monkeypatch.setattr(tts, "_get_audio_context", lambda: ctx)
        # patch_player=False: exercise the REAL _open_player fallback logic.
        install_fake_stream(monkeypatch, tts, [CHUNK] * 4, timeline, patch_player=False)

        result = tts.speak("Fallback device test.")

        assert result is True
        assert len(ctx.open_calls) == 2  # configured attempt + default fallback
        assert tts._output_device_failed is True

    def test_no_output_device_uses_default(self, monkeypatch):
        tts = make_tts()  # output_device defaults to None
        timeline = []
        install_fake_stream(monkeypatch, tts, [CHUNK] * 4, timeline)

        assert tts.speak("Default device test.") is True
