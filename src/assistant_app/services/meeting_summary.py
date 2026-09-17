"""Meeting summary generation (W3 D3): chunked map-reduce over the transcript.

Sits on the existing LLMProvider seam (factory.py) — the same Strategy
interface the assistant uses for screen analysis, so local Ollama and cloud
OpenAI are swappable without this module caring. The DEFAULT is local Ollama:
the production wiring builds a local provider unless the config explicitly
selects cloud AND the separate ``meeting.cloud_consent`` flag is set
(enforced in config validation and again at provider construction below —
defense in depth for third-party speech leaving the device).

Prompts here are built ONLY from the transcript text passed in. They never
ride ConversationMemory or any assistant conversation state: meeting content
must not leak into (or be contextualized by) the user's private assistant
history, and the summarizer must be reproducible from the transcript alone.

Output is (markdown, metadata) written to files by the meeting store. Nothing
in this module speaks — summaries are never sent to TTS (locked decision).
"""

from assistant_app.llm.types import LLMMessage
from assistant_app.utils.logging_config import get_logger

logger = get_logger(__name__)

# Prompts are transcript-only (no ConversationMemory) and ask for markdown the
# meeting store can write verbatim into summary.md.
_MAP_PROMPT = (
    "You are summarizing one chunk of a meeting transcript ({language}). "
    "Extract the key points, decisions made, and open questions from this chunk. "
    "Be factual — only what the transcript says, no invention. Keep it under 250 words.\n\n"
    "TRANSCRIPT CHUNK:\n{chunk}"
)

_REDUCE_PROMPT = (
    "You are writing the final summary of a meeting ({language}) from per-chunk "
    "summaries. Produce markdown with exactly these sections:\n\n"
    "## Overview\n(one short paragraph)\n\n"
    "## Key Points\n(bulleted)\n\n"
    "## Decisions\n(bulleted; write 'None recorded' if none)\n\n"
    "## Action Items\n(bulleted 'task — speaker/context' when attributable; "
    "write 'None recorded' if none)\n\n"
    "Be factual: only what the chunk summaries support.\n\n"
    "CHUNK SUMMARIES:\n{chunks}"
)


def chunk_utterances(utterances: list[dict], chunk_chars: int) -> list[str]:
    """Group utterance records into transcript chunks of ~``chunk_chars`` chars.

    Utterance boundaries are respected (never split mid-utterance); a single
    utterance longer than the target becomes its own chunk. Timestamps are
    rendered inline so every chunk summary can cite when things were said.
    """
    if chunk_chars < 200:
        raise ValueError("chunk_chars must be >= 200")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for record in utterances:
        line = f"[{record.get('spoken_at_iso', '?')}] {record.get('text', '')}".strip()
        if current and current_len + len(line) + 1 > chunk_chars:
            chunks.append("\n".join(current))
            current, current_len = [], 0
        current.append(line)
        current_len += len(line) + 1
    if current:
        chunks.append("\n".join(current))
    return chunks


class MeetingSummarizer:
    """Map-reduce summarization over meeting transcripts via an LLMProvider.

    ``provider`` is the already-constructed LLMProvider (local Ollama by
    default; cloud only when separately consented). Kept provider-agnostic so
    tests inject a stub with the same ``generate`` signature.
    """

    def __init__(self, provider, chunk_chars: int = 6000, language: str = "english"):
        self.provider = provider
        self.chunk_chars = chunk_chars
        self.language = language

    def summarize(self, utterances: list[dict]) -> tuple[str, dict]:
        """Summarize utterance records → (markdown, metadata).

        Raises whatever the provider raises — the MeetingController turns a
        failure into a visible failure note next to the intact transcript.
        """
        metadata = {
            "summarizer": self.provider.name,
            "model": self.provider.model_name,
            "language": self.language,
            "utterance_count": len(utterances),
        }
        if not utterances:
            return (
                "# Meeting Summary\n\nNo speech was captured in this meeting.",
                {**metadata, "chunks": 0},
            )
        chunks = chunk_utterances(utterances, self.chunk_chars)
        metadata["chunks"] = len(chunks)
        if len(chunks) == 1:
            final = self._complete(_MAP_PROMPT.format(language=self.language, chunk=chunks[0]))
            return final, metadata
        chunk_summaries = [
            self._complete(_MAP_PROMPT.format(language=self.language, chunk=chunk)) for chunk in chunks
        ]
        joined = "\n\n---\n\n".join(f"### Chunk {i + 1}\n{summary}" for i, summary in enumerate(chunk_summaries))
        final = self._complete(_REDUCE_PROMPT.format(language=self.language, chunks=joined))
        return final, metadata

    def _complete(self, prompt: str) -> str:
        """One provider round-trip (text-only, no tools, no vision)."""
        response = self.provider.generate([LLMMessage(role="user", content=prompt)], temperature=0.3)
        return (response.content or "").strip()


def build_summarizer(meeting_cfg, llm_provider_factory=None) -> MeetingSummarizer:
    """Construct the meeting summarizer from MeetingConfig (production wiring).

    Defaults to the LOCAL Ollama provider. Cloud (OpenAI) is built only when
    ``summarizer == "cloud"`` AND ``cloud_consent`` is true — the second,
    explicit consent check at construction time (config validation is the
    first). ``llm_provider_factory`` is injectable for tests; production lazy-
    imports ``assistant_app.llm.factory.get_provider`` so this module never
    imports the OpenAI/Ollama SDKs at module scope.
    """
    if meeting_cfg.summarizer == "cloud":
        if not meeting_cfg.cloud_consent:
            raise ValueError(
                "Cloud meeting summaries require meeting.cloud_consent: true — refusing to build "
                "a cloud summarizer for consented-third-party speech without it"
            )
        from assistant_app.llm.factory import get_provider
        from assistant_app.llm.types import LLMConfig

        factory = llm_provider_factory or (lambda: get_provider(LLMConfig.for_cloud()))
        logger.warning(
            "☁️ Meeting summaries will use the CLOUD provider — third-party speech "
            "(meeting transcripts) leaves the device under the explicit cloud_consent flag"
        )
    else:
        from assistant_app.llm.factory import get_provider
        from assistant_app.llm.types import LLMConfig

        factory = llm_provider_factory or (lambda: get_provider(LLMConfig.for_local()))
    return MeetingSummarizer(
        factory(),
        chunk_chars=meeting_cfg.chunk_chars,
        language=meeting_cfg.language,
    )
