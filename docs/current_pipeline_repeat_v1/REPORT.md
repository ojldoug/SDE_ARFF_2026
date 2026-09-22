# Complete current-pipeline repeatability check

**Result: the two current complete runs differ. Strict repeatability was not established for this configuration.** Exactly one fresh-process full repeat was executed; no subsequent fit or numerical-setting sweep was run. Original historical-comparison failure and tolerances remain unchanged.

The repeat used the same pinned checkout (4c5dc6b), same isolated Python executable/venv, same packaged data, GPU 0, cooperative lock and numerical execution settings. The full pip freeze was byte-identical, and all 29,009 input checks passed before execution. Only output/log paths and process execution were new. The unchanged numerical route completed five nuisance drift fits, final drift, covariance, internal checkpoint selection, three-split evaluation and serialization. Native artifact validation passed for both current artifacts.

Detached session: ex8_current_repeat_v1 (finished normally). Outputs remain outside Git at `../current_pipeline_repeat_v1/`; accepted results and manuscripts were not changed. No data generation occurred. The external repeat supervisor only separates resource paths (existing checkout/venv) from fresh output paths; its numerical launch environment matches the preceding supervisor.

## Exact identities and differing models

Data/configuration/source hashes, seed, canonical splits, fold IDs, all seven internal split IDs and validation seeds, and final PRNG key match bit-for-bit. Recorded numerical environment, package versions, precision and Python executable also match. Timing, process-specific provenance and archive bytes are not expected to be identical.

| Stage | Previous new → repeat iteration | Frequency equality | Maximum amplitude difference | First unequal retained loss iteration |
|---|---|---|---:|---:|
| fold_0 | 1 → 1 | bitwise | 0.000122219324 | 2 |
| fold_1 | 1 → 1 | bitwise | 6.85751438e-05 | 2 |
| fold_2 | 4 → 4 | bitwise | 0.000159084797 | 2 |
| fold_3 | 4 → 4 | bitwise | 0.000145405531 | 1 |
| fold_4 | 1 → 1 | bitwise | 7.11728353e-05 | 3 |
| final_drift | 4 → 4 | bitwise | 0.000817507505 | 2 |
| covariance | 294 → 299 | different | 0.378804907 | 1 |

**Earliest retained divergence:** fold 0’s selected iteration-1 amplitudes differ by maximum 1.222193e-4, despite identical selected frequencies and identical rounded iteration-1 validation loss. Its next recorded loss differs at iteration 2. No initialization, Gram/RHS, or per-iteration acceptance buffers were saved in these complete runs, so this does not locate the first arithmetic operation or Metropolis divergence.

OOF covariance targets differ by maximum 1.840591e-4. Final-drift amplitude maximum difference is 8.175075e-4; covariance frequency maximum difference is 5.051947. All detailed bitwise/numerical checks, including all saved histories, appear in COMPARISON.json. Timing-history differences alone are not numerical replay failures.

## Reported metrics

| Split / metric | Previous new run | Repeat | Repeat − previous |
|---|---:|---:|---:|
| train drift_rmse | 0.909711719 | 0.9097929 | +8.11815262e-05 |
| train covariance_rmse | 0.113840468 | 0.113910511 | +7.00429082e-05 |
| train nll | -8.93320465 | -8.93233299 | +0.000871658325 |
| train raw_spd_violation_rate | 0.461887479 | 0.468099982 | +0.00621250272 |
| train min_raw_eigenvalue | -0.148236245 | -0.154878274 | -0.00664202869 |
| validation drift_rmse | 0.911778808 | 0.911858618 | +7.98106194e-05 |
| validation covariance_rmse | 0.112086013 | 0.111880228 | -0.000205785036 |
| validation nll | -8.96161366 | -8.96770954 | -0.00609588623 |
| validation raw_spd_violation_rate | 0.455899984 | 0.468099982 | +0.0121999979 |
| validation min_raw_eigenvalue | -0.147912621 | -0.134864286 | +0.0130483359 |
| test drift_rmse | 0.912845135 | 0.912942946 | +9.78112221e-05 |
| test covariance_rmse | 0.113032795 | 0.113298044 | +0.00026524812 |
| test nll | -8.92202759 | -8.94420719 | -0.0221796036 |
| test raw_spd_violation_rate | 0.459799975 | 0.46359998 | +0.00380000472 |
| test min_raw_eigenvalue | -0.147845313 | -0.135000736 | +0.0128445774 |

Coefficient RMSE remains raw/pre-projection; NLL retains the fixed production SPD floor. These numbers are diagnostics, not replacement manuscript results.

## Saved-model predictions

Both saved final models were reconstructed with the pinned native loader and evaluated on the same canonical train/validation/test coordinates, after fitting was complete. This is evaluation-only and used no new observations or selection. Drift and raw covariance predictions fail the unchanged rtol=1e-5, atol=1e-6 comparison on all three splits. Array hashes and bitwise results are recorded.

| Split / output | Maximum absolute difference |
|---|---:|
| train_drift | 0.00207662582 |
| train_raw_covariance | 0.21098407 |
| validation_drift | 0.00207614899 |
| validation_raw_covariance | 0.203816921 |
| test_drift | 0.00207614899 |
| test_raw_covariance | 0.21042794 |

## Execution cost

- algorithm_time: 7.958238 → 8.205638 seconds.
- compilation_time: 58.384097 → 59.217011 seconds.
- end_to_end_time: 66.342335 → 67.422649 seconds.

Whole repeat child wall time: 86.613 seconds; sampled device memory peak: 834 MiB. Native end_to_end_time remains warm-up plus algorithm only, excluding final metrics/serialization. No timing conclusions are drawn from this pair.

## Interpretation and historical comparison

These two fresh-process full executions differ despite identical recorded inputs/settings and environment. The bounded evidence establishes lack of strict repeatability across these two executions; it does not identify a nondeterministic kernel, compiler/autotuning cause, or historical environment change. The earlier fixed-basis repeat checks do not establish repeatability of the full adaptive procedure.

As a separate saved-array observation, the repeat’s selected frequencies/amplitudes, all seven validation-MSE histories and OOF targets match the historical artifact exactly. This does **not** erase or supersede the first new-versus-historical failed comparison, and the repeat is not selected as a replacement result. No new acceptance criterion was introduced. The historical failed report remains byte-identical.

The requested affirmative statement “Complete current execution is repeatable for this tested configuration; strict agreement with the historical checkpoint remains unresolved” is **not warranted**. Native execution/schema checks succeeded; complete-run numerical repeatability failed. No further computation is launched or proposed here.

## Records

COMPARISON.json compares every native field (distinguishing exact byte equality and tolerance checks), stage indices and all three split predictions. OUTCOME.json identifies artifacts, identities, execution scope and preservation checks. Scripts: `scripts/repeat_current_pipeline.py` and `scripts/compare_current_full_runs.py`. The latter only loads/evaluates existing models. Source and artifact hashes are in OUTCOME.json. The prior failed comparison retains its original rtol=1e-5, atol=1e-6 and original files.
