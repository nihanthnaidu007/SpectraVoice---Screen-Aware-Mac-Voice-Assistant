"""W2 S2 tests — visible refusals.

A refusal that only lives in the log is invisible to the person it affects:
the meeting kill-switch refusal was a log line, and an Ollama-down history
Ask surfaced the raw provider exception in the dashboard. These tests pin
the pure refusal mappers (services/refusals.py) and the structural wiring
that renders them (the orchestrator module is unimportable on Linux CI —
speech_recognition at module scope — so its wiring is checked structurally,
the same standard as the menubar tests).

Covered here:
- Refusal mapping: every LLMErrorType a local-first user can hit maps to
  one actionable sentence naming the fix (start Ollama, pull the model...).
- Rendering contract: format_answer renders an "error" answer's message
  as-is (the dashboard's visible refusal).
- Wiring structure: the orchestrator's toggle_meeting/history_ask reference
  the mappers; the AppKit alert is verified by the manual macOS checklist.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.insert(0, os.path.join(project_root, "src"))

from assistant_app.llm.types import LLMError, LLMErrorType
from assistant_app.services.refusals import (
    MEETING_CONSENT_REFUSAL_MESSAGE,
    MEETING_DISABLED_MESSAGE,
    ask_failure_message,
)

PROJECT_ROOT = Path(project_root)
SRC = PROJECT_ROOT / "src" / "assistant_app"


def _llm_error(error_type: LLMErrorType, message: str = "boom", user_message: str = "") -> LLMError:
    return LLMError(error_type=error_type, message=message, user_message=user_message)


class TestMeetingRefusalMessages:
    def test_disabled_message_names_the_kill_switch_and_the_fix(self):
        assert "meeting.enabled" in MEETING_DISABLED_MESSAGE
        assert "config" in MEETING_DISABLED_MESSAGE.lower() or "settings" in MEETING_DISABLED_MESSAGE.lower()

    def test_consent_refusal_names_the_gate_and_states_nothing_recorded(self):
        assert "refused" in MEETING_CONSENT_REFUSAL_MESSAGE.lower()
        assert "meeting.enabled" in MEETING_CONSENT_REFUSAL_MESSAGE
        assert "nothing was recorded" in MEETING_CONSENT_REFUSAL_MESSAGE.lower()


class TestAskFailureMapping:
    def test_ollama_down_names_the_url_and_the_fix(self):
        msg = ask_failure_message(_llm_error(LLMErrorType.PROVIDER_UNAVAILABLE), ollama_url="http://localhost:11434")
        assert "http://localhost:11434" in msg
        assert "Ollama" in msg
        assert "ask again" in msg

    def test_connection_error_maps_the_same_way(self):
        msg = ask_failure_message(_llm_error(LLMErrorType.CONNECTION_ERROR))
        assert "Ollama" in msg
        assert "ask again" in msg

    def test_timeout_suggests_retry_or_smaller_model(self):
        msg = ask_failure_message(_llm_error(LLMErrorType.TIMEOUT))
        assert "again" in msg
        assert "llm.ollama_model" in msg

    def test_model_not_found_names_the_pull_command(self):
        msg = ask_failure_message(_llm_error(LLMErrorType.MODEL_NOT_FOUND), ollama_model="llama3.2")
        assert "llama3.2" in msg
        assert "ollama pull" in msg

    def test_model_not_found_without_a_name_falls_back_to_the_setting(self):
        msg = ask_failure_message(_llm_error(LLMErrorType.MODEL_NOT_FOUND))
        assert "llm.ollama_model" in msg

    def test_rate_limit_is_actionable(self):
        for error_type in (LLMErrorType.RATE_LIMITED, LLMErrorType.QUOTA_EXCEEDED):
            msg = ask_failure_message(_llm_error(error_type))
            assert "rate-limited" in msg
            assert "again" in msg

    def test_unknown_llm_error_carries_the_user_message(self):
        msg = ask_failure_message(_llm_error(LLMErrorType.UNKNOWN, user_message="provider exploded"))
        assert "provider exploded" in msg

    def test_non_llm_error_gets_the_generic_visible_refusal(self):
        msg = ask_failure_message(RuntimeError("connection refused"))
        assert msg.startswith("Ask failed")
        assert "log" in msg  # points at where the details live

    def test_llm_error_without_user_message_still_visible(self):
        msg = ask_failure_message(_llm_error(LLMErrorType.SERVER_ERROR))
        assert msg.startswith("Ask failed")


class TestErrorAnswerRendering:
    """The dashboard renders the mapped message through format_answer."""

    def test_error_answer_message_is_rendered_verbatim(self):
        from assistant_app.services.history_qa import HistoryAnswer, format_answer

        answer = HistoryAnswer(
            status="error", answer=ask_failure_message(_llm_error(LLMErrorType.PROVIDER_UNAVAILABLE))
        )
        rendered = format_answer(answer)
        assert "Ollama" in rendered
        assert "ask again" in rendered
        # No raw exception text leaks into the user-visible refusal.
        assert "boom" not in rendered
        assert "Traceback" not in rendered


class TestRefusalsModuleIsolation:
    """The mappers are pure — no AppKit, no network client imports."""

    def test_module_imports_are_pure(self):
        path = SRC / "services" / "refusals.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        modules: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                modules.append(node.module)
        for module in modules:
            assert not module.startswith(("AppKit", "Foundation", "httpx", "requests", "urllib.request", "socket")), (
                f"refusals.py imports {module} — refusal mapping must stay pure"
            )


class TestOrchestratorWiringStructure:
    """spectravoice_assistant.py is unimportable on Linux CI — the S2 wiring
    is pinned structurally (menubar-test standard); behavior is Mac-verified
    via the PR checklist."""

    def _source(self) -> str:
        return (SRC / "services" / "spectravoice_assistant.py").read_text(encoding="utf-8")

    def test_toggle_meeting_returns_both_refusal_messages(self):
        source = self._source()
        assert "MEETING_DISABLED_MESSAGE" in source
        assert "MEETING_CONSENT_REFUSAL_MESSAGE" in source
        tree = ast.parse(source)
        handlers = [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "toggle_meeting"
        ]
        # One on the orchestrator, one on AssistantHUDActions.
        assert len(handlers) == 2
        body = "\n".join(ast.unparse(node) for node in handlers)
        assert "MEETING_DISABLED_MESSAGE" in body
        assert "refusal" in body

    def test_history_ask_maps_failures_to_the_mapper(self):
        tree = ast.parse(self._source())
        handlers = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "history_ask"]
        assert len(handlers) == 1
        body = ast.unparse(handlers[0])
        assert "ask_failure_message" in body
        assert "status=" in body and "'error'" in body  # ast.unparse normalizes quotes
        assert "self.logger.exception" in body  # surfaced, never swallowed

    def test_hud_action_shows_the_refusal_alert(self):
        tree = ast.parse(self._source())
        handlers = [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_show_refusal_alert"
        ]
        assert len(handlers) == 1
        assert "NSAlert" in ast.unparse(handlers[0])
