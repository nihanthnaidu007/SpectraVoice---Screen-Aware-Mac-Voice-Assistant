"""Meeting summarizer tests (W3 D3.3): map-reduce over the LLMProvider seam.

Provider is stubbed with the same ``generate`` surface LLMProvider exposes —
no Ollama, no OpenAI SDK, no network.
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

import pytest

from assistant_app.services.meeting_summary import (
    MeetingSummarizer,
    build_summarizer,
    chunk_utterances,
)


class FakeProvider:
    """LLMProvider stand-in: records prompts, returns scripted responses."""

    name = "fake-local"
    model_name = "fake-model"

    def __init__(self, responses=None):
        self.responses = responses or ["chunk summary", "final summary"]
        self.prompts: list[str] = []

    def generate(self, messages, tools=None, tool_choice=None, max_tokens=None, temperature=None):
        self.prompts.append(messages[-1].content)
        from assistant_app.llm.types import LLMResponse

        return LLMResponse(content=self.responses[len(self.prompts) - 1] if len(self.prompts) <= len(self.responses) else "extra")


def utterance(text: str, iso: str = "2026-09-17T10:00:00.000") -> dict:
    return {"type": "utterance", "text": text, "spoken_at": 1000.0, "spoken_at_iso": iso}


class TestChunking:
    def test_groups_utterances_by_char_budget(self):
        utterances = [utterance("word " * 10, iso=f"2026-09-17T10:0{i}:00.000") for i in range(10)]
        chunks = chunk_utterances(utterances, 200)
        assert len(chunks) >= 2
        joined = "\n".join(chunks)
        assert all(u["text"].strip() in joined for u in utterances)

    def test_never_splits_an_utterance(self):
        long_text = "x" * 500
        chunks = chunk_utterances([utterance(long_text)], 200)
        assert len(chunks) == 1
        assert long_text in chunks[0]

    def test_timestamps_rendered_inline(self):
        chunks = chunk_utterances([utterance("spoken words", iso="2026-09-17T10:00:00.000")], 6000)
        assert chunks[0].startswith("[2026-09-17T10:00:00.000]")

    def test_rejects_tiny_chunk_size(self):
        with pytest.raises(ValueError):
            chunk_utterances([], 10)

    def test_empty_input_gives_no_chunks(self):
        assert chunk_utterances([], 6000) == []


class TestMapReduce:
    def test_single_chunk_skips_reduce(self):
        provider = FakeProvider(responses=["final"])
        summarizer = MeetingSummarizer(provider, chunk_chars=6000)
        text, meta = summarizer.summarize([utterance("hello world")])
        assert text == "final"
        assert meta["chunks"] == 1
        assert len(provider.prompts) == 1
        assert "hello world" in provider.prompts[0]

    def test_multi_chunk_runs_map_then_reduce(self):
        provider = FakeProvider(responses=["m1", "m2", "m3", "m4", "final"])
        summarizer = MeetingSummarizer(provider, chunk_chars=200)
        # ~127 rendered chars per utterance → exactly one utterance per chunk.
        utterances = [utterance("word " * 20, iso=f"2026-09-17T10:0{i}:00.000") for i in range(4)]
        text, meta = summarizer.summarize(utterances)
        assert text == "final"
        assert meta["chunks"] == 4
        assert len(provider.prompts) == 5  # 4 maps + 1 reduce
        # The reduce prompt carries the per-chunk summaries.
        for chunk_summary in ("m1", "m2", "m3", "m4"):
            assert chunk_summary in provider.prompts[-1]

    def test_empty_meeting_never_calls_the_provider(self):
        provider = FakeProvider()
        summarizer = MeetingSummarizer(provider)
        text, meta = summarizer.summarize([])
        assert "No speech" in text
        assert provider.prompts == []
        assert meta["chunks"] == 0

    def test_prompts_are_transcript_only(self):
        """The prompt contains the transcript chunk and nothing from any
        conversation memory (D3: prompts must not ride ConversationMemory)."""
        provider = FakeProvider(responses=["s"])
        summarizer = MeetingSummarizer(provider, language="english")
        summarizer.summarize([utterance("the quarterly numbers look good")])
        prompt = provider.prompts[0]
        assert "quarterly numbers" in prompt
        assert "meeting transcript" in prompt.lower()


class TestBuildSummarizer:
    def test_defaults_to_local_factory(self):
        cfg_provided = []

        def factory(config=None):
            cfg_provided.append(config)
            return FakeProvider()

        from assistant_app.utils.config import MeetingConfig

        summarizer = build_summarizer(MeetingConfig(), llm_provider_factory=factory)
        assert summarizer.chunk_chars == MeetingConfig().chunk_chars

    def test_cloud_without_consent_refused(self):
        from assistant_app.utils.config import MeetingConfig

        cfg = MeetingConfig(summarizer="cloud", cloud_consent=False)
        with pytest.raises(ValueError, match="cloud_consent"):
            build_summarizer(cfg, llm_provider_factory=FakeProvider)

    def test_cloud_with_consent_builds(self):
        calls = []

        def factory(config=None):
            calls.append(config)
            return FakeProvider()

        from assistant_app.utils.config import MeetingConfig

        build_summarizer(MeetingConfig(summarizer="cloud", cloud_consent=True), llm_provider_factory=factory)
        assert len(calls) == 1
