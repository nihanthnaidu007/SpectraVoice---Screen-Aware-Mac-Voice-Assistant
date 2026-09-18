# SpectraVoice — Screen-Aware Mac Voice Assistant

Repository: `nihanthnaidu007/SpectraVoice---Screen-Aware-Mac-Voice-Assistant` · Default branch: `main`

Hands-free, screen-aware voice assistant for **macOS**: Whisper speech-to-text, OpenAI TTS, real-time screen analysis, and computer control via tools (open apps, type, click, scroll, run terminal commands). Two LLM backends behind one provider interface: **Cloud (OpenAI GPT‑5 family)** and **Local (Ollama)**.

## Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Package manager | pip + venv (`requirements.txt`) |
| LLM (cloud) | OpenAI GPT‑5 family via `openai` SDK — requires `OPENAI_API_KEY` |
| LLM (local) | Ollama at `http://localhost:11434`, default model `llama3.2` — no API key needed |
| Speech | SpeechRecognition + PyAudio (mic input), OpenAI TTS (output), openai-whisper (local STT) |
| Vision / control | OpenCV, Pillow, PyAutoGUI, pyobjc Quartz/Cocoa (macOS-only) |
| Config | `config.yaml` with `VA_`-prefixed env overrides; `.env` via python-dotenv |
| Tests | Self-running suites in `tests/` (also pytest-compatible) |
| Infra services | None — no database, no Redis, no Docker Compose. Optional local service: `ollama serve` |

## Commands

Setup (Linux sandbox — see environment boundaries below):

```bash
sudo apt-get install -y portaudio19-dev libgl1 libglib2.0-0 xvfb zstd
python3 -m venv .venv && .venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt   # FAILS on Linux: pyobjc packages are macOS-only
.venv/bin/pip install python-dotenv PyYAML pydantic openai tiktoken httpx requests \
    tavily-python SpeechRecognition soundfile numpy Pillow opencv-python PyAutoGUI PyAudio pytest
```

Run (canonical dev command):

```bash
.venv/bin/python main.py --help                    # CLI surface, exit 0
.venv/bin/python main.py --local --model llama3.2  # local LLM mode (needs ollama serve)
.venv/bin/python main.py --cloud                   # cloud mode (needs OPENAI_API_KEY in .env)
# other flags: --gui --minimal --debug --voice <name> --whisper-model <size>
#              --interactive --model <name> --ollama-url <url>
```

Local LLM backend:

```bash
ollama serve &            # keep running (default http://localhost:11434)
ollama pull llama3.2      # ~2 GB download
./scripts/setup_ollama.sh --status
```

Tests:

```bash
.venv/bin/python -m pytest tests/ -q              # 27 passed
.venv/bin/python tests/test_llm_providers.py      # 18 passed (README-documented runner)
.venv/bin/python tests/test_function_calling.py   # 9 passed (README-documented runner)
```

Lint / typecheck: none configured in the repo (no ruff/flake8/mypy config). TODO(confirm) whether maintainers want one added.

Environment variables: `OPENAI_API_KEY` (required for cloud mode; not provisioned in this workspace — use local mode), `TAVILY_API_KEY` (optional, web search), `LLM_PROVIDER` = `cloud`|`local` (skips the startup selector), `ASSISTANT_INDICATOR_ENABLED`, `ASSISTANT_INDICATOR_POSITION`, `VA_`-prefixed overrides of `config.yaml`.

## Codebase map

See `codebase-map.md`. Entry point: `main.py` → `assistant_app.services.SpectraVoiceAssistant` (voice loop) → `assistant_app.core.Assistant` (LLM ↔ tools ↔ memory), providers from `assistant_app.llm`.

## Environment boundaries (headless Linux)

- macOS-only deps (`pyobjc-framework-Quartz`, `pyobjc-framework-Cocoa`, `rubicon-objc`) and heavy STT deps (`torch`, `openai-whisper` from git) are intentionally not installed on Linux; `pip install -r requirements.txt` fails on pyobjc by design.
- PyAutoGUI needs an X display: `touch ~/.Xauthority`, run `Xvfb :99 -ac &`, `export DISPLAY=:99`.
- opencv needs system libs on headless Debian: `libgl1`, `libglib2.0-0`.
- The full voice loop requires a microphone and macOS permissions (mic, screen recording, accessibility). In the sandbox the app boots through provider validation and full Assistant init, then stops at `OSError: No Default Input Device Available` — the documented hardware boundary, not a setup defect.

## Local Verification Summary

Result: **dev_stack_healthy: true** (2026-09-17, UTC)

- `.venv` on Python 3.13.14 with the Linux-compatible dependency subset installed (PyAudio built against portaudio19-dev).
- `python main.py --help` → exit 0.
- Tests: `pytest tests/` 27 passed / 0 failed; direct runners 18 + 9 passed / 0 failed.
- Ollama service live at `http://localhost:11434` (HTTP 200), model `llama3.2` (3.2B, Q4_K_M) pulled.
- Live primary flow: `create_provider(LLMConfig.for_local(model="llama3.2"))` → `generate()` returned the expected completion — end-to-end local-LLM round trip through the repo's own provider layer.
- App boot `python main.py --local --model llama3.2`: provider validated, Assistant initialized, stops at the microphone boundary (above).
- Cloud mode without a key fails fast with the documented setup instructions.

Quick health check:

```bash
curl -s http://localhost:11434/api/tags   # → {"models":[...]} when ollama is up
.venv/bin/python -m pytest tests/ -q      # → 27 passed
```

## Sandbox snapshot

- Snapshot (E2B template): `mcbfxpz4v8rvdyx5x4am:default`, captured 2026-09-17T16:56:17Z from live session `ivevoeyu9ecmdtc1ivdix`.
- Baked-in state: repo checkout at `main` (54c7dd3), `.venv` with installed deps, Ollama server + `llama3.2` model, Xvfb on `:99`, apt packages (portaudio19-dev, libgl1, libglib2.0-0, xvfb, zstd).
