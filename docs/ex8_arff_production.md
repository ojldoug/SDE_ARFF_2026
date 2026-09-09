# Experiment 8 validation-selected ARFF production runner

Run one seed explicitly (this command performs the full fit):

```sh
python scripts/run_ex8_arff_validation_selected_crossfit.py --seed 1 \
  --artifact-path results/production/arff_validation_selected_ex8/seed_1_artifacts.npz
```

Without `--artifact-path`, the runner uses that directory and the requested seed
in the filename. Existing files are never overwritten. The generic production
batch scheduler still invokes the older ARFF runner; it has not been redirected.

## Frozen statistical protocol

The source of the production sequence is
`scripts/diagnose_ex8_validation_selected_crossfit.py`. The regression fitter
`src/arff/validation_selected.py` is unchanged.

- Canonical `data/ex8.npz`, including its stored split indices and row ordering.
- K=128, M_min=M_max=300, lambda_reg=1e-3, gamma=1, delta=0.2.
- Resampling disabled, Metropolis enabled; five folds with fold seed 2026.
- Each regression reserves its own 10% internal holdout. Its seed is
  `101000 + seed*10000 + fold_number` for cross-fit folds, `200000 + seed` for
  final drift, and `300000 + seed` for covariance.
- Five out-of-fold drift fits, then final drift, then covariance regression on
  out-of-fold residual covariance targets. One production PRNG key is threaded
  through those seven calls in that order, exactly as in the diagnostic.
- All 300 adaptations run. The minimum raw internal validation MSE selects the
  checkpoint, with the earliest minimum winning ties. Iterations are one-based;
  initialization is not a candidate. Moving-average length and patience remain
  5; the existing stopping rule cannot terminate early with these bounds.
- There is no post-selection refit, including on internal holdouts. No
  per-iteration training predictions or true-function diagnostics are added.

The canonical loader reads and validates the entire dataset, including finite
values and the split partition. Only training observations enter learning or
warm-up. Canonical validation and test observations enter metric evaluation
after both final models have been selected. No outer split metrics select a
checkpoint, hyperparameter, or run.

Final metrics use the existing ARFF evaluation functions: raw covariance RMSE
and raw SPD diagnostics, and Gaussian NLL with eigenvalue projection at 1e-3.
Negative raw eigenvalues and nonzero violation rates are recorded, not grounds
for rejecting an artifact. Integrity checks reject malformed/nonfinite model
parameters or final metrics, not scientific accuracy. The fitter's `+inf`
entries for unsuccessful internal validation steps are preserved if a finite
checkpoint was selected.

## Timing

- `compilation_time`: construction of the shared adaptation kernel plus a
  discarded execution of the complete training path with one adaptation per
  regression, on the actual training shapes. Includes initial ridge fits,
  internal validation, and out-of-fold target operations. Uses a separate key
  seeded 987654321. This is first-call/warm-up wall time, not pure compiler time.
- `algorithm_time`: synchronized full learning after warm-up, including fold
  construction, regression setup/initialization, all seven fits, internal
  validation and checkpoint selection, and cross-fitted target construction.
  No final metric evaluation, hashing, Git inspection, serialization or logging
  occurs inside this clock. Warm-up does not advance the production key.
- `crossfit_algorithm_time`: fold construction through completed OOF targets.
- `<stage>_algorithm_time`: regression call and synchronization. Fold regression
  clocks exclude their OOF prediction/target construction; cross-fit and total
  clocks include that work. Stage times need not sum to total algorithm time.
- `<stage>_start_offset` / `end_offset`: stage-call boundaries relative to the
  algorithm clock.
- `<stage>_cumulative_time`: the frozen fitter's original local clock, starting
  after internal splitting and initial amplitude fitting. It includes internal
  validation. `best_time` indexes this clock. **Do not treat stage start plus
  local history time as an exact global timestamp**: setup precedes its origin.
- `end_to_end_time`: compilation_time + algorithm_time, matching newer runner
  conventions. It excludes loading, final metrics and artifact work.
- `provenance_time`, `data_loading_time`, `final_evaluation_time`,
  `artifact_assembly_time`, and `wall_time_before_serialization` are separate
  wall-clock scopes. Loading includes training device transfer. Serialization,
  integrity checking, round-trip prediction checks and full runner wall time are
  printed after saving; their completed durations cannot be embedded in the
  same immutable archive without rewriting it. Runner wall time starts in
  `main`, excluding Python import/startup time.

The tiny CPU smoke test checks adaptation-kernel cache reuse after warm-up.
Full-size GPU timing and numerical reproduction require an explicitly approved
full run; no full fit is part of the smoke suite.

## NPZ schema

Identity: `method=arff_validation_selected_crossfit`, `artifact_version=2`,
`experiment=ex8`. Version 1 was the validation-only diagnostic. Arrays and JSON
strings load with `allow_pickle=False`.

- Final `drift_omega`, `drift_amp`, `covariance_omega`, `covariance_amp`, plus
  `diff_type` and dimensions reconstruct a `TwoStageARFFModel`.
- For each `fold_0` through `fold_4`, `final_drift`, and `covariance`: selected
  `omega`/`amp`, `validation_mse`, `moving_average`, `cumulative_time`,
  `best_iteration`, `best_validation_mse`, `best_time`, `stopped_iteration`,
  stage timing, internal validation seed and exact fitting/holdout positions.
- `fold_id` and `crossfit_covariance_targets` follow canonical training order.
  Internal `*_train_positions` index `train_idx`, not the original dataset
  directly. The deterministic split helper is reused outside algorithm timing.
- Original `train_idx`, `validation_idx`, `test_idx`; split sizes; seed and all
  actual fitting hyperparameters; `config_json` with production overrides;
  original data dtypes, effective training dtype, and final PRNG key.
- For train/validation/test: `nll`, `drift_rmse`, `covariance_rmse`,
  `raw_spd_violation_rate`, `min_raw_eigenvalue`, `min_projected_eigenvalue`.
- Git commit/dirty/status, SHA-256 of dataset and relevant sources (including
  untracked fitter/diagnostic), Python/NumPy/JAX/package versions, devices,
  backend, precision settings, selected runtime environment variables,
  invocation and UTC provenance timestamp. No precision settings are changed.

The archive is validated before and after serialization. Selected-model
predictions are compared after reconstruction on a small training subset.
Publication uses an atomic, same-directory hard link that fails if the target
already exists, including if another writer creates it during serialization.

## Checks without a full fit

```sh
JAX_PLATFORMS=cpu python -B scripts/smoke_test_ex8_arff_production.py
python -B scripts/run_ex8_arff_validation_selected_crossfit.py --help
git diff --check
```

The suite uses fabricated artifacts and small synthetic problems only. Its
numerical parity comparison uses 31 observations, K=4 and three adaptations per
regression; it does not read the canonical dataset or run a full seed.
