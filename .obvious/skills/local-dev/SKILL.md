---
name: local-dev
description: Stand up SpectraVoice's local dev environment on headless Linux — venv + dependency subset, Xvfb for PyAutoGUI, Ollama for the local LLM backend — and verify with the test suites plus a live LLM round trip.
---

# Local Dev — SpectraVoice (headless Linux sandbox)

Durable record of the 2026-09-17 onboarding run. Every step below was executed and verified; the resulting machine state is baked into snapshot `mcbfxpz4v8rvdyx5x4am:default`.

## What "healthy" means here

- Python 3.13 venv (`.venv/`) with the Linux-compatible dependency subset installed.
- `python main.py --help` exits 0.
- `pytest tests/` → 27 passed; direct runners → 18 + 9 passed.
- Ollama serving `llama3.2` at `http://localhost:11434`.
- `generate()` round trip through `assistant_app.llm.create_provider` returns text.
- The full voice loop is NOT expected to run on Linux: it needs a microphone and macOS permissions. The app boots through provider validation and full Assistant init and then stops at `OSError: No Default Input Device Available` — the documented hardware boundary, not a setup failure.

## Steps (fresh sandbox)

1. System packages:
   `sudo apt-get install -y portaudio19-dev libgl1 libglib2.0-0 xvfb zstd`
2. Virtualenv:
   `python3 -m venv .venv && .venv/bin/pip install --upgrade pip`
3. Dependencies — `pip install -r requirements.txt` fails on Linux (pyobjc is macOS-only); install the subset instead:
   `.venv/bin/pip install python-dotenv PyYAML pydantic openai tiktoken httpx requests tavily-python SpeechRecognition soundfile numpy Pillow opencv-python PyAutoGUI PyAudio pytest`
4. X display for PyAutoGUI:
   `touch ~/.Xauthority && Xvfb :99 -ac -screen 0 1280x800x24 &` then `export DISPLAY=:99`
5. Ollama (local LLM backend, no API key):
   `curl -fsSL https://ollama.com/install.sh | sh` (needs `zstd`; no systemd — start the server yourself), then in a tmux session: `ollama serve`, and `ollama pull llama3.2` (~2 GB).
6. Verify (see Verification transcript).

## Pitfalls (all hit during onboarding)

- `pip install -r requirements.txt` fails at `pyobjc-framework-Quartz` — it runs `sw_vers` (macOS-only) during build. Expected on Linux; use the subset install.
- `import cv2` → `ImportError: libGL.so.1` → install `libgl1` (and `libglib2.0-0`).
- `import pyautogui` → `KeyError: 'DISPLAY'` / `~/.Xauthority` errors → start Xvfb with `-ac`, create an empty `~/.Xauthority`, export `DISPLAY=:99`.
- Ollama installer aborts without `zstd`; it also warns "systemd is not running" — run `ollama serve` in tmux yourself.
- `OPENAI_API_KEY` is not provisioned in this workspace, so cloud mode cannot be exercised live (it fail-fasts with setup instructions — that is its documented behavior). Use local (Ollama) mode for live LLM verification. TODO(confirm) whether a real key should be requested via the credentials flow.
- No lint/typecheck is configured in the repo; pytest is the only automated check.
- Committed `__pycache__/` `.pyc` files exist in git; leave them alone and never stage regenerated ones.

## Verification transcript (2026-09-17)

```
$ .venv/bin/python main.py --help            → exit 0 (full usage printed)
$ .venv/bin/python tests/test_llm_providers.py      → RESULTS: 18 passed, 0 failed
$ .venv/bin/python tests/test_function_calling.py   → RESULTS: 9 passed, 0 failed
$ .venv/bin/python -m pytest tests/ -q      → 27 passed, 1 warning in 1.13s
$ curl http://localhost:11434/api/tags      → HTTP 200, llama3.2 (3.2B, Q4_K_M) listed
$ python /tmp/live_llm_check.py             → PROVIDER_OK name='Ollama' model='llama3.2' tools=True
                                             → GENERATE_OK RESPONSE: SpectraVoice local LLM round-trip OK
$ python main.py --local --model llama3.2   → 🏠 Using Local LLM (Ollama, model=llama3.2)
                                               🤖 Assistant using Ollama (llama3.2)
                                               OSError: No Default Input Device Available  ← mic boundary
$ python main.py --cloud                    → ❌ OPENAI_API_KEY environment variable not set (fail-fast, documented)
```

Live round-trip script (through the repo's own provider layer):

```python
import sys; sys.path.insert(0, "src")
from assistant_app.llm import LLMConfig, create_provider, LLMMessage
provider = create_provider(LLMConfig.for_local(model="llama3.2", base_url="http://localhost:11434"))
resp = provider.generate([LLMMessage(role="user", content="Reply with exactly: SpectraVoice local LLM round-trip OK")])
print(resp.content)
```
