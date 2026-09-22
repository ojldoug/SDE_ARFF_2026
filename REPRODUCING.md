# Reproducing the conservative manuscript review

This page maps the **review draft** in `results/conservative_manuscript_revision_v1/` to its archived evidence. It preserves the original method → applications → results → sensitivity structure. The review is not the authoritative Overleaf source, and [AUTHOR_DECISIONS.md](results/conservative_manuscript_revision_v1/AUTHOR_DECISIONS.md) remains open. A rebuild of archived metrics is distinct from training the estimators again.

## Environment, inputs and commands

The archive-only build needs Python 3.11, NumPy, Matplotlib and `latexmk`/pdfLaTeX. `requirements-dev.txt` is the project package lock, including JAX/CUDA for optional numerical work. The snapshot was built using the existing `arff-sde` environment (Python 3.11.15, NumPy 2.4.6, Matplotlib 3.11.1, JAX 0.10.2) and TeX Live. The archive-only path runs on CPU. Checkpoint inference requires the authenticated source and a compatible GPU/JAX backend; the prior Ex1/Ex8 reconstruction check used GPU with `rtol=1e-5, atol=1e-6` before drawing grids. Training requires a CUDA GPU and substantially more time.

From a fresh checkout, install packages, then choose an unused output path:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
REVIEW_OUT=/tmp/sde-review-$(date +%Y%m%d%H%M%S)
.venv/bin/python scripts/rebuild_review_snapshot.py --output "$REVIEW_OUT"
```

This command verifies `archive_inputs/MANIFEST.json`, computes the Ex1–3/5–8 and N/h/K result tables from archived per-seed values, checks their displayed precision against the review source, regenerates the ten included figures from archived metrics or previously validated coefficient maps, and compiles `manuscript.pdf`. It writes only to `$REVIEW_OUT` and refuses an existing path. `REBUILD_CHECKS.json` reports each check. To separate graphics/table generation from TeX compilation, add `--skip-tex`, then run:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -cd "$REVIEW_OUT/sources/RaulCom_ARFFSDELearning.tex"
```

The compact archived input CSVs, their origin paths and SHA-256 hashes are in `results/conservative_manuscript_revision_v1/archive_inputs/`. The seed-0 grid arrays and the 27 prior reconstruction metric checks are in `evaluation/`; `RECONSTRUCTION_CHECKS_PUBLIC.json` lists source, dataset, checkpoint and grid-array hashes with repository-relative paths. `scripts/rebuild_review_snapshot.py` uses those saved grid arrays and does **not** rerun inference. `plot_coefficients.py` checks their hashes and prior validation record before plotting. The historical local `aggregate_source_record/build_assets.py` is a build-history script with a machine-specific path; use the portable rebuild command above.

For fresh saved-checkpoint inference or training, obtain the exact datasets and model archives from the project custodians. There is no public download URL or durable archive identifier yet. A fresh clone therefore has enough compact evidence to rebuild the current review, but not enough large files to repeat fitting or reevaluate checkpoints. Place datasets at `data/` and checkpoints under the repository-relative paths in `RECONSTRUCTION_CHECKS_PUBLIC.json`, or verify separate storage roots with:

```bash
python scripts/verify_review_external.py --data-root /path/to/data --results-root /path/to/results
sha256sum -c data_checksums.sha256
sha256sum -c docs/review_data_checksums.sha256
```

The last two checksum commands apply only when the complete dataset set is present. The corrected Ex6 dataset is `data/ex6_labels_v2.npz` (SHA-256 `47c4a91d37f494621c4174e5c8a87d951a9b78b16b1da68dc4b146f72dbc85c3`); corrected Ex8 is `data/ex8_float64_v2.npz` (`6fb009e3d6fa5f241cb1a15f9a9815cbd16ea19c3a9e4ca546ae99bdae091a72`). Original `data/ex8.npz` is the retained float32 historical version. The corrected Ex8 source generator integrated with float64 and the same underlying normal draws; the model arithmetic remained float32. The full raw model archives, generated N/h/K dataset views and lag/ridge arm artifacts remain outside ordinary Git. The compact CSVs in this snapshot preserve the reported values and source hashes, not those large arrays.

## Evidence key and interpretation

All paths below are repository-relative. `A/` means `results/conservative_manuscript_revision_v1/archive_inputs/`; `R/` means the review directory itself. The command `rebuild` means the `scripts/rebuild_review_snapshot.py --output "$REVIEW_OUT"` command above; it invokes `R/plot_coefficients.py`, `R/lag_ridge_assets.py`, `R/aggregate_source_record/replot_final_editorial.py` and `R/aggregate_source_record/replot_mechanism.py`. Static application/protocol tables are preserved source text checked against the historical specification and source manifest; they are not newly inferred from model outputs.

**Historical-compatible applications Ex1–3/5–7:** `results/final_reproduction/production_manifest.v5.json` and `scripts/run_final_historical_{adam,arff}.py` define the executed settings. Their displayed 30-seed results use the selected outer **validation** population. Ex6 uses corrected labels. Historical Owen execution details and discrepancies are not all recovered or explained. **Modern corrected Ex8:** five approximately parameter-matched estimators, K=128 or MLP width 27, 30 seeds, 80/10/10, independent test metrics; ARFF uses five-fold out-of-fold covariance targets and internal validation, with raw covariance RMSE and fixed-floor NLL kept separate. **Exploratory diagnostics:** shared-data seed/repeat variability and prior test exposure limit inference; neither lag/ridge nor component-swap results replace production settings.

Status terms: **verified from archived outputs** means recomputed or checked from the compact values and source records, without fresh fitting; **rerun verified** means earlier saved-checkpoint inference passed its archived-metric check and its grid arrays are now packaged; **historical** means descriptive specification or validation-selected evidence; **unresolved** means a scientific convention still needs an author decision; **unavailable** means the necessary raw archive is not distributed with Git. The source PDFs and model snapshots are identified, not claimed to be fresh-training replications.

## Every numerical table in the active review

| Manuscript label / purpose | Experiment and exact evidence | Rebuild / needs | Status |
|---|---|---|---|
| `tab:experiment_summary`, application counts/structure | Ex1–8 descriptions; `src/experiments/definitions.py`, `results/final_reproduction/production_manifest.v5.json`; Ex4 line explicitly withheld | Preserved in `R/sources/RaulCom_ARFFSDELearning.tex`; TeX only, CPU | Historical; Ex4 unresolved |
| `tab:ex1`, polynomial recovery | Ex1, 30 seeds, outer validation; `A/historical_per_seed.csv`, `A/historical_summary.json`, corrected historical manifest | `rebuild`; CPU | Historical protocol, verified from archived outputs |
| `tab:ex2`, linear/triangular | Ex2, same 30-seed validation protocol; same archived files | `rebuild`; CPU | Historical protocol, verified from archived outputs |
| `tab:ex3`, ten-dimensional cubic | Ex3, executed shallow MLP and ARFF, 30 seeds, outer validation; same archived files | `rebuild`; CPU | Historical protocol, verified from archived outputs; discrepancy unresolved |
| `tab:ex5`, fixed-lag SIR/SSA | Ex5, population 1024; same historical CSV and manifest | `rebuild`; CPU | Historical protocol, verified from archived outputs |
| `tab:ex6`, wave-derived | Ex6 `ex6_labels_v2.npz`; same historical CSV and manifest | `rebuild`; CPU | Historical protocol with corrected labels, verified from archived outputs |
| `tab:ex7`, broad spectrum | Ex7; same historical CSV and manifest | `rebuild`; CPU | Historical protocol, verified from archived outputs; NLL history unresolved |
| Raw ARFF SPD table (after `tab:ex7`) | Ex1–7 validation from `A/historical_per_seed.csv`; Ex8 test from `A/baseline_per_seed.csv`; raw eigenvalues, NLL floors stated in caption | `rebuild` validates retained `R/sources/tables/spd.tex`; CPU | Verified from archived outputs |
| `tab:ex8`, corrected five-method comparison | `data/ex8_float64_v2.npz`, K=128/MLP width 27, 30 seeds; `A/baseline_per_seed.csv`, `A/baseline_summary.json` | `rebuild`; CPU | Verified independent-test aggregates |
| Appendix data tables `tab:polynomial_training_data`, `tab:wide_spectrum_data`, `tab:near_singular_data`, `tab:SIR_training_data`, `tab:SPDE_training_data` | Original application specifications, accepted corrections in `results/final_reproduction/production_manifest.v5.json`, `src/experiments/definitions.py`; Ex5 generation `src/experiments/sir_ssa.py`, Ex6 wave `src/experiments/wave_data.py` | Preserved `R/sources/appendices_part2.tex`; TeX only | Historical descriptive protocol, with Ex5/6 corrections |
| `tab:Langevin_training_data` | Experiment 4 historical specification in `R/sources/appendices_part2.tex`; sign/conditioning issue in `R/AUTHOR_DECISIONS.md` | Preserved source; TeX only | Unresolved; no numerical comparison |
| `tab:hyperparams`, `tab:adam_hyperparams` | Historical Ex1–7 and modern Ex8 settings; tracked production manifest, accepted runners, `R/sources/appendices_part3.tex` | Preserved source; TeX only | Historical/modern definitions explicitly separated |
| Complete N/h/K tables (`N_full.tex`, `h_full.tex`, `capacity_full.tex`) | Corrected Ex8, K=1024 sample-size and nine-lag studies, and fixed N=640000/h=1e-4 capacity sweep; `A/N_per_seed.csv`, `A/h_per_seed.csv`, `A/capacity_per_seed.csv` | `rebuild`; CPU | Verified from archived outputs; ten fitting seeds on shared data |
| Raw-SPD sensitivity table (`sensitivity_spd.tex`) | Ex8 N/h/K raw ARFF covariance; `A/sensitivity_spd.csv` | Retained table checked against archived CSV; CPU | Verified from archived outputs |
| Lag/ridge Tables 22–23 (`lag_ridge_arms.tex`, `lag_ridge_paired.tex`) | K=1024, N=640000, h={.0001,.0004,.002}, drift ridge={.001,.008,.064}, seeds 0–2; `A/lag_ridge_per_run.csv`, `A/lag_ridge_paired.csv` | `rebuild` invokes `R/lag_ridge_assets.py`; CPU | Verified exploratory test-set means and every within-lag pair; earlier test exposure stated |
| `tab:drift_diagnostic`, fixed-basis/adaptive ridge and resampling | Shared-noise three-repeat surrogate, not Ex8 trajectory data; `A/fixed_basis_ridge_paired.csv`, `A/adaptive_ridge_paired.csv`, `A/resampling_paired.csv` | Retained editorial table; verify cited archived values, TeX only | Verified diagnostic; mechanism not uniquely identified |
| `tab:swaps` and floor-sensitivity table | Post-hoc corrected Ex8 CPU component swaps, 30 seeds; `A/oracle_cpu_per_seed.csv`, `A/oracle_floor_per_seed.csv`, `A/oracle_projection_per_seed.csv`, `R/aggregate_source_record/TABLE_S14_RECONCILIATION.json` | Retained editorial tables, CPU scalar source checks; no predictions | Verified diagnostic; float32/float64 CPU and production GPU remain distinct |

Tables embedded in the application/reproducibility appendices are descriptive protocol tables; numerical result tables above are generated or checked from the listed archived values. The review has no active Experiment 4 result table.

## Every figure in the active review

| Manuscript figure / purpose | Input and protocol | Rebuild / needs | Status |
|---|---|---|---|
| `fig:ex1coeff`, true versus learned slices | Ex1 seed 0 selected models; 201 points on each prescribed coordinate slice, raw covariance; `R/evaluation/ex1_seed0_coefficients.npz`, reconstruction record | `rebuild` → `R/plot_coefficients.py`; CPU for plotting, GPU only for optional saved-checkpoint inference | Rerun verified earlier; illustrative seed 0, validation-selected |
| `fig:ex8distribution`, five-method RMSE distributions | Modern corrected Ex8, 30 seeds, raw covariance; `A/baseline_per_seed.csv` | `rebuild` → `replot_final_editorial.py`; CPU | Verified from archived outputs |
| `fig:ex8drift`, true/learned drift field | Corrected Ex8 seed 0, fixed 100×100 cell-centre grid on [-2,2]², all five models; `R/evaluation/ex8_seed0_coefficients.npz` | `rebuild` → `plot_coefficients.py`; CPU to plot | Rerun verified earlier; illustrative seed 0 |
| `fig:ex8cov`, true/learned raw covariance | Same seed/grid and saved arrays; raw Σ entries, no projection | Same command; CPU to plot | Rerun verified earlier; illustrative seed 0 |
| `fig:hmodern`, lag sensitivity | Corrected Ex8, N=640000, K=1024, nine lags, ten seeds; `A/h_per_seed.csv` | `rebuild` → `replot_final_editorial.py`; CPU | Verified descriptive test metrics; not historical modified Ex7 |
| `fig:kmodern`, capacity sensitivity | Corrected Ex8, N=640000, h=1e-4, K={64,128,256,512,1024}, matched MLP widths={18,27,39,57,81}; `A/capacity_per_seed.csv`, `A/parameter_counts.csv` | Same command; CPU | Verified descriptive test metrics; Phase 1 did not establish capacity dominance |
| Projection diagnostic (first supplement figure) | Corrected Ex8 seed-0 raw and projected ARFF covariance, same fixed grid; `R/evaluation/ex8_seed0_coefficients.npz`, floor 0.001 | `rebuild` → `plot_coefficients.py`; CPU | Rerun verified earlier; grid diagnostics do not replace test SPD rate |
| N-sensitivity figure (sensitivity appendix) | Corrected Ex8, K=1024, h=1e-4, N={80000,160000,320000,640000}, ten seeds; `A/N_per_seed.csv` | `rebuild` → `replot_final_editorial.py`; CPU | Verified descriptive test metrics |
| `fig:lag_ridge_nll`, paired NLL intervention | Same h/λ/seed design as Tables 22–23; each seed minus its λ=.001 score at the same lag; `A/lag_ridge_paired.csv` | `rebuild` → `lag_ridge_assets.py`; CPU | Verified exploratory diagnostic; no penalty selected |
| `fig:drift_diagnostic`, training/validation/frequency trajectories | Shared-noise surrogate M/MR histories and selected iterations, repeats 0–2; `A/mechanism_*.csv`, `A/mechanism_selections.csv` | `rebuild` → `replot_mechanism.py`; CPU | Verified from recorded histories; no intermediate true-error predictions |

The original unauthenticated trajectory histogram, isolated timing and historical modified-Ex7 scaling figures are absent from the active review. The missing evidence is described in `R/CHANGE_LOG.md`; none has been replaced by a different figure while retaining its old claim.

## Optional numerical reruns and limitations

The review build above is fast (about 20–30 seconds in the tested environment) and uses no GPU. With authenticated datasets and an otherwise idle GPU, one historical-compatible seed can be run without overwriting the archives:

```bash
RUN_OUT=/tmp/sde-ex1-seed0-check
mkdir -p "$RUN_OUT"
python scripts/run_final_historical_adam.py ex1 fourier --seed 0 --artifact-path "$RUN_OUT/adam.npz"
python scripts/run_final_historical_arff.py ex1 --seed 0 --artifact-path "$RUN_OUT/arff.npz"
```

These commands are **examples of the accepted runners**, not commands executed for this snapshot. The production manifest and `data/ex1.npz` must match their stored hashes; runner code checks them. Ex1/2/3/5/6/7 runners and `scripts/run_final_adam_campaign.py`/`run_final_arff_campaign.py` define their historical-compatible 30-seed campaigns. The corrected Ex8 baseline uses `scripts/run_controlled_ex8_float64_v2.py` with job JSON and the frozen local `results/controlled_study_2026/float64_v2/manifest.json`; the N/h/K suites use `scripts/run_regime_ex8_v2.py` and `scripts/run_regime_ex8_final_h.py`. The lag/ridge arm runner and protocol are preserved locally at `results/ex8_h_lambda_drift_v1/study.py` and `PROTOCOL.md`. The external v1 package now inventories and supplies those dataset views, manifests, checkpoints and diagnostic sources. Restore and verify it before invoking numerical routes. See [ROUTES.md](docs/independent_reproduction_v1/ROUTES.md) for the portable single-job commands; they leave the accepted numerical runners unchanged. Public hosting and fresh-training verification remain pending.

Archived lag/ridge full-estimator fits each took about 1,140–1,160 seconds including warm-up on the study hardware, with seven regressions; 18 new arms ran over roughly two hours on four exclusive GPUs. A full 30-seed five-method corrected Ex8 baseline or 800-record N/h/K study is a multi-hour to multi-day GPU campaign, depending on concurrency and data generation. Concurrent four-GPU algorithm times are marked non-isolated and do not establish publication-quality runtime advantages. No full training was launched for this snapshot.

The remaining scientific decisions are Experiment 4 sign/conditioning, unaccounted historical numerical differences, lack of authenticated corrected trajectory/isolated-runtime evidence, and a durable public data/model archive. Archive-based numerical agreement and a successful TeX build do **not** prove that fresh training reproduces the reported distributions.

## Bounded review-source correction (22 September 2026)

The local mathematical, bibliographic and protocol-documentation corrections are recorded in `results/conservative_manuscript_revision_v1/CORRECTION_RECORD_20260922.md`; archive-only checks are in `CORRECTION_BUILD_CHECKS_20260922.json`. No numerical result, checkpoint, training setting or coefficient grid changed. Appendix B.5 now states preprocessing, dense solver, historical stopping and seed conventions. The large-artifact release and historical search-history limitations above remain.

## Independent-reproduction package v1

The [ordered route guide](docs/independent_reproduction_v1/ROUTES.md) separates archived rebuilding, exact-model evaluation and fresh generation/training for every retained experiment/diagnostic. The [artifact manifest](docs/independent_reproduction_v1/artifact_manifest.json) gives each payload file's exact hash, size, availability, purpose and dependency references. [BUNDLE.json](docs/independent_reproduction_v1/BUNDLE.json) identifies the prepared tar; [VERIFICATION.json](docs/independent_reproduction_v1/VERIFICATION.json) distinguishes executed metadata/hash checks from unexecuted numerical routes.

Private manuscript/feedback preservation dependencies in old diagnostic audit guards are not computational inputs and are excluded from distribution. The portable adapter explicitly omits only those private-document checks; numerical source/data hashes remain mandatory. Restored historical metadata may contain old local path strings; adapters resolve the recorded repository prefix without editing source or manifest bytes. No private document is needed for training/evaluation.

All prior scientific qualifications and unrecovered historical search/selection records remain in force. No fresh training, new model predictions or stochastic data generation were performed for this package. A matching input hash and a successful metadata inspection are not evidence of numerical training reproduction.
