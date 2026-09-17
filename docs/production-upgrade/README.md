# Production Upgrade — Checkpoint Paper Trail

This folder is the committed record of the **four-repo production upgrade
program** (Nixus-Sql, AXIOM_Adaptive_RAG, Research_Forge, and SpectraVoice) —
a coordinated effort that took each repository from an unhardened research
prototype to a secure, tested, CI-governed, documented deliverable for a
single-operator production environment, delivered across Waves 0–2.

The program was planned, executed, audited, and versioned in the Obvious
project workspace ([Production Upgrade Planning For Repos][project]). This
folder checks the final paper trail into each delivered repository so the
record ships with the code it describes.

[project]: https://app.obvious.ai/p/production-upgrade-planning-for-repos-E3IkGpyk

## Contents

| File | What it is | Source artifact |
|---|---|---|
| [dossier.md](dossier.md) | Final delivery dossier — all four repos, Waves 0–2, verified merge SHAs, limitations, restoration checkpoints | `art_jV2n9Tta` |
| [master-plan.md](master-plan.md) | Cross-repo master plan — roadmap, acceptance criteria, CI-gated PR model, scope | `art_gwXo54Vz` |
| [survey.md](survey.md) | SpectraVoice production-readiness survey — the baseline findings that seeded the work | `art_BVfU87TR` |
| [w2-spec.md](w2-spec.md) | SpectraVoice Wave 2 grounded spec — reliability, `--doctor`, bounded recovery, three-step setup | `art_10m8rkCi` |
| [independent-audit.md](independent-audit.md) | Independent read-only audit of the delivered Research_Forge + SpectraVoice state | `art_tsz9H96x` |
| [w2-pr-record.md](w2-pr-record.md) | Record of the merged SpectraVoice Wave 2 PR (#4, merge commit `1f6247b`) | `art_8ejgup9W` |
| [delivery-record.md](delivery-record.md) | Session-verified checkpoint facts for this repo (merge state, lint/test evidence, artifact fetch log) | this checkpoint PR |

The five program documents (`dossier.md`, `master-plan.md`, `survey.md`,
`w2-spec.md`, `independent-audit.md`) were fetched **verbatim** from the
project artifacts with the Obvious CLI; `w2-pr-record.md` is rendered from the
artifact card's metadata. The README and delivery record were authored for
this checkpoint and contain only session-verified facts.

## Delivery state (this repo)

SpectraVoice Wave 2 merged at `1f6247b` on `origin/main`
(chain `310eb6a` → `4351be0` → `1f6247b`) and was independently audited.
The annotated tag `production-upgrade/final` marks the merge commit of the
checkpoint PR that introduced this folder.
