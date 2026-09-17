# Delivery Record — SpectraVoice Checkpoint

Session-verified facts recorded when this folder was committed (2026-09-17).
Every claim below was observed directly during the checkpoint run; the program
documents in this folder carry the fuller narrative and are quoted nowhere
beyond their own files.

## Merge state (verified via git)

- `origin/main` tip: `1f6247b` — "feat(w2): add --doctor diagnostics, crash
  auto-restart, and 3-step setup (#4)".
- Wave 2 chain confirmed in history:
  - `310eb6a` — "fix(security): sandbox run_command, add privacy consent gate,
    logging, and repo hygiene (#2)"
  - `4351be0` — "feat: wire config system, stream TTS, and move Whisper off
    the hot path (#3)"
  - `1f6247b` — "feat(w2): add --doctor diagnostics, crash auto-restart, and
    3-step setup (#4)"

## Verification run (this checkpoint)

- `ruff check .` with the CI-pinned `ruff==0.16.8`: clean.
- `pytest tests/ -q` on the pristine `1f6247b` tree: **162 passed** —
  matching the independent audit's 162/162 clean-checkout result.
- Docs-only diff: no source, test, or CI files touched by this checkpoint PR.

## Artifact fetch log (obvious CLI `artifacts get`, 2026-09-17)

| File | Artifact | Result |
|---|---|---|
| dossier.md | `art_jV2n9Tta` | fetched verbatim (26,925 chars, version 3) |
| master-plan.md | `art_gwXo54Vz` | fetched verbatim (7,403 chars, version 1) |
| survey.md | `art_BVfU87TR` | fetched verbatim (14,203 chars, version 1) |
| w2-spec.md | `art_10m8rkCi` | fetched verbatim (6,470 chars, version 0) |
| independent-audit.md | `art_tsz9H96x` | fetched verbatim (18,144 chars, version 0) |
| w2-pr-record.md | `art_8ejgup9W` | rendered from pull_request card metadata (version 2) |

All six fetches succeeded, so no failure-fallback claims were needed.

## Tag

Per the delivery runbook, the annotated tag `production-upgrade/final`
(message: "Production upgrade checkpoint: Wave 2 delivered
(310eb6a → 4351be0 → 1f6247b); SpectraVoice reliability + first-run UX
shipped") is pushed at the merge commit of the checkpoint PR that introduced
this folder.
