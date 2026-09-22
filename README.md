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

The resulting PDF is `/tmp/sde-review-output/review/manuscript.pdf`; `REBUILD_CHECKS.json` records input-hash, numerical-table and figure checks. The output path must be new, so the command cannot replace an accepted result. On the project server the same rebuild was tested with the existing `arff-sde` Python environment. A fresh package installation is not claimed as tested here.

The compact archive inputs and prevalidated seed-0 coefficient maps needed for the review rebuild are in Git. Full datasets and per-seed model checkpoints are not. For saved-checkpoint reevaluation or training, place authenticated files under the expected `data/` and `results/` layouts, or pass their roots to `scripts/verify_review_external.py`. Their hashes and the missing distribution route are documented in [REPRODUCING.md](REPRODUCING.md). No download link is asserted.

The review compares corrected modern Experiment 8 on an independent test split with historical-compatible applications whose displayed results were selected on validation. The exploratory lag/ridge supplement is a separate, three-seed intervention. Rebuilding its archived metrics does not repeat training.

## Independent numerical reproduction

Three routes have different requirements:

1. **Archived-result rebuild:** the quick start above uses compact Git inputs only; already checked.
2. **Exact checkpoint evaluation:** restore the external v1 bundle, verify every input, then use `scripts/evaluate_reproduction_checkpoint.py` (defaults and flags in the route guide). Checkpoints are necessary for evaluating the published models, not for fresh fitting.
3. **Data generation and fresh training:** `scripts/reproduction_generate.py` and `scripts/reproduction_route.py` inspect the accepted recipes/configurations by default. Explicit `--execute` with a new output directory is required to run. No training campaign was executed to prepare this package.

See [the ordered route guide](docs/independent_reproduction_v1/ROUTES.md), [file-level hashes/availability](docs/independent_reproduction_v1/artifact_manifest.json), and [verification record](docs/independent_reproduction_v1/VERIFICATION.json). Example after restoring the bundle:

```sh
.venv/bin/python scripts/package_independent_reproduction.py verify
.venv/bin/python scripts/reproduction_route.py baseline --method arff --seed 0
```

The latter inspects; it does not train. The bundle is prepared locally for durable hosting, **not publicly downloadable yet**. No off-server backup is verified. Original historical search budgets remain incomplete. These limitations do not prevent archive-only rebuilding, but full independent numerical reproduction is not claimed.
