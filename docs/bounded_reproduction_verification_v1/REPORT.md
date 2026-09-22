# Bounded single-run reproduction verification

**Outcome: complete execution and successful saved-model reconstruction; agreement with the archived seed-0 fit failed the predeclared tolerance.** No second training run was launched. No result, tolerance, method or manuscript was changed.

## Installation, isolation and inputs

A full local Git clone with independent Git objects was detached at `4c5dc6b54af216ebba49477874c955d259d4140f`. Tracked source remained unchanged. A new Python 3.11.15 venv (without system site packages) was installed from that commit’s exact `requirements-dev.txt`; pip check passed. The existing Python executable provided the base interpreter/standard library; all third-party runtime modules came from the new venv. The research environment was not changed. Installation used available cached wheels and the configured package index. No dependency pins were relaxed.

The 6.50 GiB tar matched BUNDLE.json. All 29,009 payload files passed checks after independent extraction, after copy-only-missing overlay into the checkout, and after execution. Bundle-supplied untracked computational source is intentionally present in the restored checkout; no tracked-code edits were made. The original tree payload preservation check also passed. This run **used the packaged corrected dataset**, including its split IDs; it did not execute data generation.

The child blocked Python opens under the original working-tree prefix and recorded module/source paths; no original-tree module was loaded. Scientific inputs and numerical imports came from the restored checkout. The only intended original-tree interaction was the supervisor’s existing cooperative GPU lock, not any model/data/source read. The guard is a Python file-open audit, not an OS sandbox for arbitrary native-library I/O; archived data/source hashes and module paths provide complementary checks. Historical absolute strings in provenance were rebased by the pinned adapter in memory.

## Execution and cost

Detached tmux `ex8_arff_seed0_reproduction_v1` survived the launch connection and was checked from an independent server call. GPU 0 was idle before dispatch and reserved using the existing `results/production/ex8_campaign_control/gpu_0.lock`. Exactly one child ran the five nuisance drift fits, final drift and covariance fit, all 300 adaptations, selection, three-split evaluation and serialization. The session exited normally after completion.

Algorithm: **7.958238 s**; warm-up/compilation: **58.384097 s**; native end-to-end: **66.342335 s** (warm-up + algorithm only). Whole child wall time: **85.577 s**; native runner wall time 82.842 s, including final evaluation/provenance/serialization. The 1-second GPU samples peaked at **834 MiB** device memory; this is a sampled observed peak, not an allocator-guaranteed maximum. Installation/extraction are outside these timings. This is not a repeated isolated-runtime benchmark.

JAX/jaxlib/CUDA plugin/PJRT 0.10.2 and NumPy 2.4.6 match the archived artifact. Both record A6000, float32 learning, x64 disabled and default matmul precision. New runtime explicitly set OMP_NUM_THREADS=2 and XLA_PYTHON_CLIENT_PREALLOCATE=false; the archived artifact did not record those settings. These operational differences are disclosed, not proven causes. Full package versions are in environment.txt.

## Predeclared comparisons

Native schema/round-trip validation passed; all seven 300-entry histories and earliest-minimum checkpoint checks passed. Seed, dataset/source/configuration hashes, split IDs, fold IDs, every internal fit/holdout index and final PRNG key matched exactly. Final amplitude-fit count is 72,000; each nuisance fit uses 57,600 observations (nominal training pool 80,000); nuisance regression pools contain 64,000 with 6,400 holdout rows. Final regression holdouts contain 8,000. No post-selection refit.

All comparisons retain rtol=1e-5, atol=1e-6. Reloaded test drift RMSE differs from its own saved metric by 1.19209e-7; covariance RMSE, NLL and raw SPD metrics agree exactly. This passes the reconstruction check. All three splits’ reported coefficient metrics/NLL differ from the archived run beyond the requested tolerance.

| Split / metric | Archived | New | New − archived |
|---|---:|---:|---:|
| train drift_rmse | 0.90979296 | 0.909711719 | -8.12411308e-05 |
| train covariance_rmse | 0.113910511 | 0.113840468 | -7.00429082e-05 |
| train nll | -8.93233299 | -8.93320465 | -0.000871658325 |
| train raw_spd_violation_rate | 0.468099982 | 0.461887479 | -0.00621250272 |
| train min_raw_eigenvalue | -0.154878274 | -0.148236245 | +0.00664202869 |
| validation drift_rmse | 0.911858618 | 0.911778808 | -7.98106194e-05 |
| validation covariance_rmse | 0.111880228 | 0.112086013 | +0.000205785036 |
| validation nll | -8.96770954 | -8.96161366 | +0.00609588623 |
| validation raw_spd_violation_rate | 0.468099982 | 0.455899984 | -0.0121999979 |
| validation min_raw_eigenvalue | -0.134864286 | -0.147912621 | -0.0130483359 |
| test drift_rmse | 0.912942946 | 0.912845135 | -9.78112221e-05 |
| test covariance_rmse | 0.113298044 | 0.113032795 | -0.00026524812 |
| test nll | -8.94420719 | -8.92202759 | +0.0221796036 |
| test raw_spd_violation_rate | 0.46359998 | 0.459799975 | -0.00380000472 |
| test min_raw_eigenvalue | -0.135000736 | -0.147845313 | -0.0128445774 |

| Regression | Archived → new iteration | Archived → new selected noisy validation MSE |
|---|---|---|
| fold_0 | 1 → 1 | 5191.619141 → 5191.619141 |
| fold_1 | 1 → 1 | 4951.133789 → 4951.133789 |
| fold_2 | 4 → 4 | 5069.422363 → 5069.422852 |
| fold_3 | 4 → 4 | 5074.163574 → 5074.166016 |
| fold_4 | 1 → 1 | 4959.225586 → 4959.225586 |
| final_drift | 4 → 4 | 5003.597168 → 5003.597168 |
| covariance | 299 → 294 | 0.6081925035 → 0.6077946424 |

## Bounded discrepancy diagnosis

Selected nuisance and final drift frequency arrays match **bit-for-bit**, but amplitudes do not: final drift amplitude maximum difference 8.17508e-4, RMS 5.45125e-5. OOF covariance-target maximum difference is 1.84059e-4, RMS 8.68966e-6. Thus the discrepancy already exists upstream of covariance fitting; it is not merely a saved-model loading or final metric convention problem. Covariance selected frequencies differ (maximum 5.05195), and its selected iteration changed 299→294.

Validation-history entries outside tolerance: fold 0/1/2 = 0; fold 3 = 114; fold 4 = 64; final drift = 0; covariance = 285. Small floating-point solve differences followed by adaptive-path sensitivity are consistent with this pattern, but these saved outputs do not identify the first arithmetic/backend cause or first Metropolis decision. No replay/solver investigation or further fit was launched. Matching major package versions and GPU model is insufficient to establish identical compiler/library execution. This remains an unresolved strict numerical-replay discrepancy, not evidence that all published distributions fail to reproduce.

The published seed-0 and manuscript results remain unchanged. Negative raw covariance eigenvalues are reported rather than filtered; production NLL keeps its original 1e-3 floor.

## Commands and records

From the restored pinned checkout, the supervised child executed:
```sh
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda JAX_ENABLE_X64=false \
  XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
  ../venv/bin/python -B scripts/reproduction_route.py baseline --method arff --seed 0 \
  --execute --output ../run
```
The external supervisor installed the read guard before invoking the same entry point with runpy. It inherits the cooperative lock into the child. See scripts/verify_single_reproduction.py. For comparison use scripts/compare_single_reproduction.py with --reference, --new and a fresh --output. Reconstruction used the pinned scripts/evaluate_reproduction_checkpoint.py --new-run --protocol modern --method arff --dataset data/ex8_float64_v2.npz --artifact ../run/artifact.npz --output ../reevaluation.json under the same GPU lock.

Small machine-readable records: COMPARISON.json, REEVALUATION.json, OUTCOME.json, environment.txt, gpu_memory.csv. Full installation/preflight/console/supervisor logs, isolation trace, extracted payload, venv and new checkpoint remain outside Git at the project sibling directory `bounded_reproduction_verification_v1/`. The new artifact SHA-256 is recorded in OUTCOME.json. No independently verified off-server copy exists; follow ../independent_reproduction_v1/BACKUP.md and record a destination receipt.

## Verification boundary

Verified: pinned installation, extraction/hash identity, portable full seven-fit execution, checkpoint/schema validation, saved-model self-reconstruction and preserved inputs. **Not verified:** data generation, exact agreement with the published fit, 30-seed distributional reproduction, other estimators/experiments, cross-hardware agreement, or independent backup. No further training is authorized by this report.
