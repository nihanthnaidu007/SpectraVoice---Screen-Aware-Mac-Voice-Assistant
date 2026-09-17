# SpectraVoice W2 Grounded Spec — Reliability + First-Run UX

Task: todo_MMnPWZGH · Branch `feat/w2-reliability-ux` (from origin/main @ 4351be0) · PR → main, squash.

## Codebase areas touched (verified this session)

| Area | File(s) | State on main (W0+W1 merged) | W2 change |
|---|---|---|---|
| Entry point | `main.py` (315 lines) | `parse_args()` + `main()`; config loaded via `init_config(args.config)`; logging via `setup_logging` + `install_crash_handlers()` | Move flow to `src/assistant_app/cli.py` (keeps `python main.py` working, enables console script); add `--doctor` and `--no-supervisor` flags |
| Config system (W1) | `src/assistant_app/utils/config.py` (568 lines), `config.yaml` | Dataclass sections + `VA_*` env overrides + `validate()` | Add `SupervisorConfig` section (`enabled`, `max_restarts`, `window_seconds`, `backoff_seconds`, `backoff_max_seconds`), env overrides `VA_SUPERVISOR_*`, validation, `to_dict` |
| Logging + crash capture (W0) | `src/assistant_app/utils/logging_config.py` | Rotating file logging (`logs/assistant.log`, 10 MB × 5) + `sys.excepthook`/`threading.excepthook` crash capture writing tracebacks to that file | Supervisor builds on this: same file, append-only, restart events logged; child stdout/stderr piped into the same log; nothing truncated |
| Provider availability | `src/assistant_app/llm/{openai_provider,ollama_provider}.py` `is_available()` | Exist; used by `validate_provider` | Doctor reuses the *patterns* (key presence, Ollama HTTP ping) but standalone probes — doctor must run before any provider exists |
| Screen capture | `src/assistant_app/io/vision/screen_capture.py` | `ImageGrab.grab()` loop; silent after 3 errors without permission | Doctor's screen-recording probe uses `CGPreflightScreenCaptureAccess` (Quartz, already in requirements) with an active `ImageGrab.grab()` fallback |
| Tests / CI | `tests/` (8 files), `.github/workflows/ci.yml` (ruff 0.16.8 + pytest on ubuntu-latest; installs only pytest/openai/httpx/python-dotenv/pyyaml) | Platform-independent suite | New `tests/test_doctor.py`, `tests/test_supervisor.py`, `tests/test_cli_flags.py`; config-section tests appended to `test_config_loading.py` |
| Packaging | none (no `pyproject.toml`); README's 6-step manual install; `.env.example` exists (W0) | Not installable | `pyproject.toml` (setuptools, src layout, dynamic deps from requirements.txt, console script `spectravoice`), `scripts/bootstrap.sh`, `docs/SETUP.md`, README quickstart |

## Design

### 1. `--doctor` (`src/assistant_app/doctor.py`)
- `CheckState` enum: `PASS / FAIL / WARN / SKIP`; `CheckResult(name, state, detail, fix_it)`; probes return `ProbeOutcome(state, detail)`.
- Default probes are functions imported lazily inside — platform-gated on `sys.platform == "darwin"`:
  - **Microphone**: `AVCaptureDevice.authorizationStatusForEntity_(AVMediaTypeAudio)` via pyobjc-framework-AVFoundation (added to requirements); fallback: open default input via PyAudio (active probe).
  - **Screen recording**: `Quartz.CGPreflightScreenCaptureAccess()`; fallback: `PIL.ImageGrab.grab()` test frame.
  - **Accessibility**: `AXIsProcessTrusted()` (pyobjc-framework-ApplicationServices, added); fallback: `pyautogui` cursor event probe.
  - **OpenAI key**: required (FAIL) only when `cfg.llm.provider == "cloud"`; WARN when `auto`; PASS/SKIP when local.
  - **Ollama**: `httpx` GET `{cfg.llm.ollama_url}/api/tags`, 2 s timeout → WARN when unreachable (optional provider).
  - **Config**: `config_manager.validate()` issues → WARN per issue.
- On non-macOS (Linux CI) the three permission probes return `SKIP` with detail — suite stays green with zero mocking for the default path; macOS outcomes are exercised via injected fake probes.
- Fix-it lines: pane path + deep link, e.g. `x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture`.
- `run_doctor(config_manager, checks) -> list[CheckResult]`; prints report; `doctor_exit_code(results)` = 1 iff any FAIL.

### 2. Auto-restart supervisor (`src/assistant_app/supervisor.py`)
- `RestartPolicy(max_restarts, window_seconds, backoff_seconds, backoff_max_seconds)` from `cfg.supervisor`.
- `Supervisor(command: list[str], policy, log_file)`: spawn child → wait → restart on crash. No restart on: exit 0 (clean), SIGINT/SIGTERM death of child (user interrupt). Crash loop guard: > `max_restarts` within `window_seconds` → give up, log critical pointer to log file, exit 1. Exponential backoff capped at `backoff_max_seconds`.
- Log capture: supervisor runs `setup_logging(log_file=cfg.logging.file)` + `install_crash_handlers()` (W0), opens the log in append mode and pipes child stdout+stderr there — child tracebacks (W0 hooks) plus supervisor restart events (`child exited code=X, restarting i/max`) coexist in the rotating log.
- Signal forwarding: SIGINT/SIGTERM → child; supervisor exits with child's code.
- main.py flow: `--doctor` short-circuits; else if `cfg.supervisor.enabled` and not `--no-supervisor`: child argv = `[sys.executable, main_script, *original_args, "--no-supervisor"]` (recursion guard); supervise never supervises a child that itself has supervision on.
- Tests use a real child script that crashes N times then exits 0 — asserts restart count, give-up behavior, and log capture (child stderr lands in log file).

### 3. Packaged setup path (≤3 steps)
1. `git clone … && cd SpectraVoice…`
2. `./scripts/bootstrap.sh` — checks Python ≥3.10, creates `.venv`, installs system deps (brew on macOS; apt hint/attempt on Linux), `pip install -e .` (deps single-truth: pyproject dynamic ← requirements.txt), copies `.env.example` → `.env` if missing, runs `--doctor` (non-fatal), prints next step.
3. `python main.py` (or `spectravoice`).
- `docs/SETUP.md`: the 3-step path, manual fallback, what doctor checks, permission fix-it panes, known gaps.
- README: quickstart at top of Installation pointing at the 3 steps; manual steps retained below.

## Verification plan
- `pytest tests/ -q` green locally (Linux sandbox) incl. new doctor/supervisor/cli tests; crash-capture supervisor test simulates crashes and asserts restart + log capture.
- `ruff check .` clean (pinned 0.16.8).
- E2E: fresh clone in sandbox → run bootstrap → `--doctor` headless (probes SKIP on Linux, exit 0) → console script works. Heavy deps (torch/whisper/PyAudio) attempted; gaps reported honestly if system headers unavailable.
- CI green; visual evidence N/A (CLI/reliability work, no rendered UI).
