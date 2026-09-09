# Historical ARFF compatibility and corrections

The separate `scripts/final_historical_arff_compat.py` preserves the primary manuscript Algorithm 1 order: resample frequencies, random-walk proposal, optional Metropolis decision, ridge fit. The modern `src/arff/regression.py` adaptation routine uses another order and therefore is not substituted wholesale. Its tested ridge solve, prediction and correctly grouped amplitude-norm functions are reused. No accepted Experiment 8 file changes.

Preserved from `GPU/lib/lib_ARFF.py`: zero frequencies at initialization; key-based internal 10% split with floor sample count and retained input order; PRNG split progression; five-point float32 rolling sum; strict earliest-minimum ties; zero-based `i > M_min`; patience condition `minimum_index + 5 < i`; stopping before candidate retention. The terminal validation observation is archived even though the historical `[:i]` archive omitted it. Index 0 means the first adaptation. No post-selection refit.

Explicit correction records:

- Historical `amp.reshape(-1,K)` mixes frequencies and output channels when q>1 -> Raul §5.2 requires an explicit multi-output amplitude norm -> reuse `frequency_amplitude_norm`, pooling cosine/sine coefficients for each frequency across outputs. A deterministic q=3 fixture demonstrates the old grouping differs from the intended norm.
- Historical default-precision ridge system and approximate CG solve -> accepted full-float32 normal-equation reproducibility correction -> reuse existing HIGHEST-precision matrix products and ridge solve without changing lambda or features.
- Historical same-sample drift residuals -> accepted out-of-fold covariance-target blocker -> reuse the accepted five-fold orchestration, fold seed2026, five nuisance fits followed by final drift and covariance fits.
- Historical absolute-value diagonal covariance and potentially indefinite full matrices -> raw covariance diagnostics and SPD requirement -> preserve raw outputs/RMSE; apply existing eigenvalue-floor projection only for NLL. Use existing experiment-specific floors, no epsilon search.
- Zero amplitude probability/ratio edge cases -> feedback Algorithm1 edge-case request -> reuse existing dtype-tiny lower-bound convention. Both resampling and Metropolis are never simultaneously enabled in the recovered settings for these six experiments.
- Historical truncated histories/time-to-best -> reproducible artifact and full algorithm timing -> archive every computed validation MSE, moving average, full-stage time, selected model, OOF targets, fold memberships and actual internal split positions; time all seven regressions. No per-iteration training predictions are added.

The adapter is tested against the unchanged historical loop while substituting the same corrected numerical primitives into that loop. Selected arrays, final key and historical history prefix match exactly for resampling and Metropolis fixtures. A zero-error fixture verifies stopping at index6 and earliest selection at index0. A complete tiny seven-fit CPU run verifies OOF exclusion, 90/10 outer split, no test set, artifact serialization and validation.

Compilation uses a discarded full seven-fit path with one adaptation per regression and independent seed987654321. The production seed is initialized afresh. Final metrics, hashes and serialization are outside algorithm time. Internal split-index archiving is done once per fit, not inside adaptation. Parallel campaign timings remain non-isolated.
