"""User-visible refusal messages for gate-blocked actions (W2 S2).

The Wave-2 pre-flight finding: a refusal that only lives in the log is
invisible to the person it affects — the meeting kill-switch refusal was a
log line, and an Ollama-down history Ask surfaced the raw provider exception
in the dashboard. These pure mappers turn each refusal into one actionable,
user-visible sentence. No AppKit, no network — fully Linux-testable; the
HUD, history window, and CLI render the strings.
"""

from __future__ import annotations

from assistant_app.llm.types import LLMError, LLMErrorType

# The meeting controller only exists when meeting.enabled was true at launch;
# this message covers the kill-switch refusal for that path.
MEETING_CONSENT_REFUSAL_MESSAGE = (
    "Meeting recording was refused — the meeting.enabled kill-switch is off "
    "right now, so nothing was recorded. Turn it on in Settings (meeting "
    "section) or config.yaml, then start the meeting again."
)

# No meeting controller at all: the feature was disabled before launch.
MEETING_DISABLED_MESSAGE = (
    "Meeting recording is disabled by the meeting.enabled kill-switch. Turn it "
    "on in Settings (meeting section) or config.yaml, then start the meeting again."
)


def ask_failure_message(
    exc: Exception,
    *,
    ollama_url: str | None = None,
    ollama_model: str | None = None,
) -> str:
    """Map a history-Ask provider failure to one actionable sentence (pure).

    The dashboard renders the result directly; the raw exception stays in the
    log (surfaced there by the caller, never swallowed).
    """
    if isinstance(exc, LLMError):
        if exc.error_type in (LLMErrorType.PROVIDER_UNAVAILABLE, LLMErrorType.CONNECTION_ERROR):
            at = f" at {ollama_url}" if ollama_url else ""
            return (
                f"Couldn't reach your local Ollama{at} — start Ollama (or check "
                "llm.ollama_url / llm.ollama_model in Settings) and ask again."
            )
        if exc.error_type is LLMErrorType.TIMEOUT:
            return (
                "Your local Ollama took too long to answer — try asking again, "
                "or pick a smaller llm.ollama_model in Settings."
            )
        if exc.error_type is LLMErrorType.MODEL_NOT_FOUND:
            name = f"'{ollama_model}'" if ollama_model else "the configured llm.ollama_model"
            return (
                f"Your local Ollama doesn't have {name} — pull it (for example: "
                "ollama pull llama3.2) and ask again."
            )
        if exc.error_type in (LLMErrorType.RATE_LIMITED, LLMErrorType.QUOTA_EXCEEDED):
            return "The LLM provider is rate-limited right now — wait a moment and ask again."
        if exc.user_message:
            return f"Ask failed: {exc.user_message}"
    return "Ask failed — the provider reported an error (details are in the log)."
