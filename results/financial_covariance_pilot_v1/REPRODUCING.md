# Reproducing this bounded pilot

Exploratory research, separate from accepted manuscript evidence. No public raw-data mirror is provided. Official monthly archives can be downloaded without credentials; see RIGHTS_AND_SOURCES.md. Archive revisions must be treated as different inputs, not silently accepted as the original experiment. The recorded source revision is in FROZEN.json; its additional SHA-256 entries authenticate pilot scripts that were not yet committed when fitting began.

## Dependencies and exact inputs

ENVIRONMENT.txt is the executed environment's package list. RUNTIME.json records Python/JAX/dtypes; execution.json records GPU UUID/driver, launch command, cache paths and allocation cost. Numerical training uses float32 and the existing repository ARFF/Adam functions. Data preparation and common evaluation use NumPy float64. The only tested optional compiler flag is `--xla_gpu_autotune_level=0`, set by the supervisor before JAX initialization with fresh study caches. This is not a production default or a determinism guarantee.

DATA_MANIFEST.json contains all147 official URL/checksum pairs (three assets, December2021 plus2022–25). Official CHECKSUM files are verified at acquisition and archives are checked again when loaded. Nine incomplete/missing asset-hours are recorded; incomplete bars are excluded, never stretched into a full hourly observation. SPLIT_MANIFEST.json contains fitting-only normalization and target eligibility. Structural test timestamps were accessible before fitting; substantive test prices were first opened after PRETEST_GATE.json. `development.npz` and `forecast_timestamps_development.csv` are generated intermediates, not additional input data. Source snapshots and archives remain local; source file hashes plus the committed code define the calculation. No private correspondence is needed.

Selected checkpoint pickles, full forecasts/score arrays and market archives are intentionally outside ordinary Git. They are locally available under this directory on KW61146; no independent backup or public hosting is asserted. Pickles should only be loaded from this trusted study and verified against their adjacent SHA-256 envelopes. They are needed to evaluate these exact checkpoints, but not for fresh training. Compact CSVs/JSON and figures can be read without model files. The original failed acquisition log and source are retained locally; DATA_PREFLIGHT_NOTE.md and initial_timestamp_failures.json document it publicly.

## Executed command order

Run from repository root with Python3.11 and the recorded packages. `$PILOT_PY` denotes the selected environment's absolute Python executable. The executed one was `../bounded_reproduction_verification_v1/venv/bin/python` relative to this repository; this is an environment location, not an untracked scientific source dependency.

These commands are for a **fresh disposable checkout**, never for overwriting this completed directory. In such a checkout, first move the tracked publication records aside:

```bash
mv results/financial_covariance_pilot_v1 results/financial_covariance_pilot_v1_reference
mkdir results/financial_covariance_pilot_v1
cp results/financial_covariance_pilot_v1_reference/PROTOCOL.md results/financial_covariance_pilot_v1/
export PILOT_PY=/absolute/path/to/your/recorded-environment/bin/python
"$PILOT_PY" -B -m scripts.financial_pilot.download
"$PILOT_PY" -B -m scripts.financial_pilot.data
OPENBLAS_NUM_THREADS=1 "$PILOT_PY" -B -m scripts.financial_pilot.checks
"$PILOT_PY" -B -m scripts.financial_pilot.freeze
```

Compare downloaded archive SHA-256 values with the reference DATA_MANIFEST.json before fitting. Compare generated arrays/preprocessing/split membership with the original data manifest; NPZ compressed-file bytes can depend on library/archive metadata. Any mismatch must be explained before claiming identical inputs. The current execution used precisely the FROZEN.json development archive hash.

The following command starts the finite queue on an otherwise idle A6000 and is **new fitting**, not an archived-result rebuild. It is documented for reproducibility; it is not an authorization for more runs. Existing production GPU-lock directory must exist (`mkdir -p results/production/ex8_campaign_control` in the fresh checkout); when other workers are present use their shared lock location, not a separate private lock domain. The original launch used the existing repository's cooperative lock files.

```bash
tmux new-session -d -s financial_covariance_pilot_v1 -c "$PWD" \
  "$PILOT_PY -B -m scripts.financial_pilot.supervise >> results/financial_covariance_pilot_v1/supervisor.log 2>&1"
tmux attach -t financial_covariance_pilot_v1
tail -f results/financial_covariance_pilot_v1/console.log
```

One worker,18 ARFF/neural fits plus bounded DCC fitting; eight allocated GPU-hours and twelve supervisor wall-hours are hard ceilings. Allocation accounting conservatively includes CPU preprocessing/DCC while holding the GPU. Selected stages are saved with hashes and frozen identity. A interrupted nonterminal execution can reuse valid stages under the remaining cumulative budget. Terminal markers refuse automatic restart; one scientific-invariant orchestration recovery was authorized, not arbitrary retries. The budget and recovery record must be inspected before a manual resume.

`pipeline` performs checkpoint selection on2024, validates reconstructed development predictions and independent likelihood/recursion checks, records PRETEST_GATE.json, then opens2025 for chronological forecasting and evaluation. It never refits on validation/test data. To rebuild only tables/figures from **already saved selected forecasts**, with no fitting or new predictions, use in an isolated output copy containing those two NPZ files and selection/candidate/model-envelope records:

```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 "$PILOT_PY" -B -m scripts.financial_pilot.report
```

After restoring all saved inputs, audit completion without new model predictions using `JAX_PLATFORMS=cpu "$PILOT_PY" -B -m scripts.financial_pilot.validate_outputs`, then append the completed-outcome synthesis with `"$PILOT_PY" -B -m scripts.financial_pilot.close_report`. The latter reads aggregates and execution records only.

This reporting command writes into the fixed study directory of its checkout; use a disposable copy so the accepted outputs are preserved. It does not change forecasts or choose configurations. Reconstructing saved models requires trusted `models/*.pkl`, exact numerical source and environment; each newly fitted checkpoint was reconstructed on validation before test exposure at rtol1e-5,atol1e-6. The report alone cannot verify fresh-training repeatability.

## Colloidal evaluation correction

`scripts/correct_colloidal_evaluation_float64.py` uses the original local `results/experimental_trajectory_pilot_v1/{test_predictions.npz,test_evaluation_inputs.npz,summary.json,VALIDATION.json,PROTOCOL.md}`. Their exact hashes are in the correction's summary.json. No new inference, data or fitting is performed. Raw files are not included in Git and must be supplied by the authors; no fabricated download URL is provided. In a disposable checkout where the new output directory does not exist:

```bash
MPLBACKEND=Agg OPENBLAS_NUM_THREADS=1 "$PILOT_PY" -B scripts/correct_colloidal_evaluation_float64.py
```

The output is a new sibling directory; existing originals and their failed1e-10 comparison are preserved. The corrected run retains that tolerance and passes it using float64 operations on the saved raw predictions.
