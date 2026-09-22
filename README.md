# SDE learning with adaptive random Fourier features

This repository contains the SDE coefficient-learning implementations, experiment runners, archived scalar evidence, and an **isolated review draft** of the method-and-applications paper. The current review is [results/conservative_manuscript_revision_v1/](results/conservative_manuscript_revision_v1/); it is not the authoritative Overleaf manuscript. [AUTHOR_DECISIONS.md](results/conservative_manuscript_revision_v1/AUTHOR_DECISIONS.md) records unresolved issues, including Experiment 4.

## Quick start: rebuild the review from archived outputs

Run these commands at the repository root. Python 3.11, the locked Python packages in `requirements-dev.txt`, and TeX Live with `latexmk`/pdfLaTeX are required. A CPU is sufficient for this archive-only build.

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
mkdir -p /tmp/sde-review-output
.venv/bin/python scripts/rebuild_review_snapshot.py --output /tmp/sde-review-output/review
```

The resulting PDF is `/tmp/sde-review-output/review/manuscript.pdf`; `REBUILD_CHECKS.json` records input-hash, numerical-table and figure checks. The output path must be new, so the command cannot replace an accepted result. On the project server the same rebuild was tested with the existing `arff-sde` Python environment. A fresh pinned package installation and one complete Ex8 ARFF fit were subsequently tested; see the bounded verification below.

The compact archive inputs and prevalidated seed-0 coefficient maps needed for the review rebuild are in Git. Full datasets and per-seed model checkpoints are not. For saved-checkpoint reevaluation or training, place authenticated files under the expected `data/` and `results/` layouts, or pass their roots to `scripts/verify_review_external.py`. Their hashes and the missing distribution route are documented in [REPRODUCING.md](REPRODUCING.md). No download link is asserted.

The review compares corrected modern Experiment 8 on an independent test split with historical-compatible applications whose displayed results were selected on validation. The exploratory lag/ridge supplement is a separate, three-seed intervention. Rebuilding its archived metrics does not repeat training.

## Reproduction status: four distinct routes

| Route | Verified scope and limits |
|---|---|
| Archived manuscript rebuild | Compact Git aggregates and saved coefficient maps rebuild tables, figures and PDF without model evaluation or fitting. |
| Supplied historical checkpoint evaluation | Previously selected Ex1/Ex8 seed-0 models passed archived-metric reconstruction checks. Requires the external bundle; this does not verify fresh training or every checkpoint. |
| Fresh training, original execution settings | Corrected Ex8 ARFF K128/N80000/seed 0 completed all seven fits in an isolated environment. Two fresh processes differed; one recovered the archive and one failed. Strict fresh-process repeatability is **not established**. |
| Optional autotune-disabled execution | Three minimal processes and two complete seven-fit processes passed bounded repeatability checks. Scientific outputs of the two full runs were bit-identical, but **neither matched the historical archive** at rtol=1e-5, atol=1e-6. Not a production default or general determinism guarantee. |

Both training checks used packaged corrected data, not newly generated data. Data generation, other configurations, other methods and other hardware remain unverified by these checks. Historical discrepancies are not fully explained; paper numbers and failed comparisons are preserved.

[REPRODUCING.md](REPRODUCING.md) contains the evidence map, tested command, exact runtime and protocol. [REPRODUCTION_STATUS.md](REPRODUCTION_STATUS.md) summarizes closure and release tasks. The [ordered route guide](docs/independent_reproduction_v1/ROUTES.md) covers data preparation and individual runners; its inspect-only commands do not train unless explicitly passed `--execute`.

The 6.50 GiB external bundle is prepared locally, **not publicly hosted**. No independently verified off-server backup is recorded. Obtain it from the custodians and follow the [checksum and transfer instructions](docs/independent_reproduction_v1/BACKUP.md). A fresh clone alone supports the archived rebuild, not saved-model evaluation or fitting. No further numerical runs are part of this documentation closure.
