# SpectraVoice Setup — 3-Step First Run

This is the packaged setup path. It targets a **≤3-step first run** and replaces
the old six-section manual install (kept below as a fallback).

## The 3 steps

**Step 1 — Get the code** (clone + enter the directory):

```bash
git clone https://github.com/nihanthnaidu007/SpectraVoice---Screen-Aware-Mac-Voice-Assistant.git
cd SpectraVoice---Screen-Aware-Mac-Voice-Assistant
```

**Step 2 — One-command bootstrap:**

```bash
./scripts/bootstrap.sh
```

The script verifies Python ≥ 3.10, creates `.venv`, installs PortAudio +
FFmpeg (brew on macOS, apt on Ubuntu/Debian), installs the package with
`pip install -e .` (dependencies come from `requirements.txt` via
`pyproject.toml` — one source of truth), seeds `.env` from `.env.example` if
you don't have one, and finishes by running `python main.py --doctor`.

On containers and CI images without root, export
`SPECTRAVOICE_SKIP_SYSTEM_DEPS=1` to skip the system-package step — the
doctor run at the end then reports anything still missing.

**Step 3 — Run it:**

```bash
python main.py          # or: spectravoice (console script installed by step 2)
```

That's it. On first launch a GUI pop-up asks Cloud (OpenAI) vs Local (Ollama);
set `LLM_PROVIDER=cloud` or `LLM_PROVIDER=local` in `.env` to skip it.

## What `--doctor` checks

Run any time with `python main.py --doctor`. It is fully headless (no GUI, no
assistant startup) and exits 0 only when nothing blocking is found:

| Check | macOS probe | Fix (printed on failure) |
|---|---|---|
| Microphone | `AVCaptureDevice` authorization status; falls back to opening the default input | System Settings → Privacy & Security → Microphone |
| Screen recording | `CGPreflightScreenCaptureAccess` (Quartz); falls back to a test `ImageGrab.grab()` frame | System Settings → Privacy & Security → Screen Recording |
| Accessibility | `AXIsProcessTrusted()` | System Settings → Privacy & Security → Accessibility |
| OpenAI API key | Presence of `OPENAI_API_KEY` (required only when `llm.provider: cloud`) | Copy `.env.example` → `.env`, set the key |
| Ollama server | HTTP ping of `llm.ollama_url` | `brew install ollama && ollama serve` |
| Tavily API key | Presence of `TAVILY_API_KEY` (optional — web search) | Set `TAVILY_API_KEY` in `.env` |
| Config health | `ConfigManager.validate()` issues | Follow the printed issue |

On non-macOS systems (Linux CI, containers) the three permission probes report
`skipped` instead of failing, so the doctor is safe to run anywhere.

Each failing check prints its fix: a human-readable System Settings path and an
`x-apple.systempreferences:…` deep link that opens the exact pane.

## Auto-restart on crash

`config.yaml` ships with `supervisor.enabled: true`: `python main.py` runs the
assistant as a supervised child process. On an unhandled crash the supervisor
restarts it (exponential backoff, at most `supervisor.max_restarts` restarts
within `supervisor.window_seconds`, then it gives up with a clear error).

Every restart — and everything the child prints, including the crash
tracebacks from the Wave 0 crash capture — is appended to
`logs/assistant.log` (see the `logging:` section of `config.yaml`), so a crash
is always diagnosable after the fact.

Run once without supervision: `python main.py --no-supervisor`, or set
`supervisor.enabled: false` / `VA_SUPERVISOR_ENABLED=false`.

## Manual setup (fallback, what the bootstrap automates)

1. `python3 -m venv .venv && source .venv/bin/activate`
2. System deps: `brew install portaudio ffmpeg` (macOS) or
   `sudo apt-get install -y portaudio19-dev python3-dev ffmpeg` (Ubuntu/Debian)
3. `pip install --upgrade pip && pip install -r requirements.txt`
4. `cp .env.example .env` and add `OPENAI_API_KEY` (and optionally `TAVILY_API_KEY`)
5. Grant permissions (see the doctor table above)
6. `python main.py`

## Verified end-to-end

The 3-step path is exercised from a clean clone as part of the W2 work; see the
PR description for the run log and any environment-specific gaps (e.g. system
packages that need a sudo password are printed, not silently skipped).

## Uninstall

`pip uninstall spectravoice` inside the venv; delete the clone. `.venv`,
`logs/`, and `.env` live inside the repo directory and are removed with it.
