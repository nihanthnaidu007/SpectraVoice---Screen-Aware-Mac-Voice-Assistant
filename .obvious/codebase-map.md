# Codebase Map — SpectraVoice

Folder-level overview. `assistant_app` is imported from `src/` (`main.py` puts `src/` on `sys.path`; tests do the same).

| Path | Purpose |
|---|---|
| `main.py` | CLI entry point: argument parsing, LLM provider selection/validation (fail-fast), launches the assistant |
| `config.yaml` | Default settings (mode, voice, screen capture, llm, safety, logging); `VA_`-prefixed env overrides |
| `requirements.txt` | Python deps — includes macOS-only pyobjc packages and heavy STT deps (see `.obvious/obvious.md` boundaries) |
| `src/assistant_app/` | Application package |
| `src/assistant_app/core/` | `assistant.py` — Assistant core wiring LLM ↔ tools ↔ conversation memory; `conversation.py` — short-term memory |
| `src/assistant_app/llm/` | Provider-agnostic LLM layer: `types.py` (LLMConfig/Message/Response/Error), `base.py` (LLMProvider ABC), `factory.py` (create/select + env override), `openai_provider.py` (cloud), `ollama_provider.py` (local), `gui_selector.py` (tkinter startup picker with terminal fallback) |
| `src/assistant_app/io/` | Device I/O: `audio/tts.py` (OpenAI TTS + PyAudio playback), `audio/voice_detector.py` (wake word + barge-in), `vision/screen_capture.py` (screen → JPEG), `indicator.py` (on-screen status dot, macOS AppKit) |
| `src/assistant_app/services/` | `spectravoice_assistant.py` — main orchestrator: mic loop, screen analysis, LLM calls, TTS |
| `src/assistant_app/tools/` | Function calling: `tool_executor.py` (13 tools), `safety.py` (risk levels + blocked commands), `mouse_controller.py`, `smart_click.py`, `web_search.py` (Tavily) |
| `src/assistant_app/utils/` | `config.py` (YAML + env), `logging_config.py`, `error_handler.py` |
| `tests/` | `test_llm_providers.py` (18 tests, mocked endpoints), `test_function_calling.py` (9 tests) — self-running or pytest |
| `scripts/` | `setup_ollama.sh` — Ollama status / model-pull helper |
| `docs/` | `BARGE_IN_TESTING.md` — manual barge-in (voice interruption) test guide |

Primary flow: `main.py` → `services/spectravoice_assistant.py` (mic → STT → screen capture) → `core/assistant.py` → `llm/` provider → `tools/tool_executor.py` → TTS.
