# Prospective verification: autotuning disabled

**Outcome:** the single candidate policy stabilized the bounded minimal check and both complete current-path executions in this test. **Neither complete execution agrees with the historical archive at the unchanged rtol=1e-5, atol=1e-6.** This is prospective repeatability evidence, not proof of a historical cause. No production default, estimator, manuscript result or accepted artifact changed.

## Policy, support and caches

The only candidate compiler flag was `XLA_FLAGS=--xla_gpu_autotune_level=0`, set in the child environment before importing JAX. The installed JAX/jaxlib 0.10.2 accepted it, initialized GPU 0, and completed a trivial compiled addition (result 2). All subsequent native compilations also succeeded. No alternative flag, dependency upgrade or numerical precision/solver intervention was attempted.

[Official OpenXLA determinism guidance](https://openxla.org/xla/determinism), consulted 22 September 2026, distinguishes compilation-time autotuning variability from execution-time nondeterminism. Disabling autotuning removes compile-time benchmarking but does not provide a general execution-determinism guarantee. The flag can trade runtime performance for stable compilation choices. The present evidence concerns only the executed tests.

All processes used the existing isolated venv, Python 3.11.15, JAX/jaxlib/plugin/PJRT 0.10.2, NumPy 2.4.6 and pinned 4c5dc6b checkout. GPU index 0 was exclusively held with the existing cooperative lock and checked before each dispatch: RTX A6000, UUID `GPU-d2cfc73c-186a-6651-e9f1-c6d93a00f0b2`, driver 595.84. Same packaged corrected data and scientific settings as the preceding verification; no generation occurred.

Runtime settings were retained: CUDA_VISIBLE_DEVICES=0, JAX_PLATFORMS=cuda, JAX_ENABLE_X64=false, XLA_PYTHON_CLIENT_PREALLOCATE=false, OMP_NUM_THREADS=2, OPENBLAS_NUM_THREADS=1 and Python user-site/bytecode isolation. No higher matmul precision or new determinism flag was added. The native Gram/RHS retain their existing explicit HIGHEST products; other products retain native defaults.

**Cache policy:** each child, including each full fit, received separate initially empty JAX_COMPILATION_CACHE_DIR and CUDA_CACHE_PATH directories. JAX cache-enable/default options were otherwise unchanged. In-memory caches were fresh at process start and naturally reused within that process. Existing project/user caches were not deleted or overwritten. This intentional cache isolation prevents admitting an old XLA executable or cached driver code; it is verification hygiene alongside the single compiler-policy change. Launch records retain exact cache paths and flags. These checks therefore describe the combined documented launch procedure, not an uncontrolled default-cache launch.

## Minimal gate: three processes, two executions each

The same authenticated 57,600-row float32 fold-0 fixture was used, with zero-frequency initialization, original key, K128 and lambda=.001. Each process computed the initial amplitudes once, then executed the same native compiled first-step calculation **twice on identical initial model/key/inputs**. Each native result, fitting prediction, internal-validation prediction and validation loss was explicitly synchronized before conversion/comparison. This is actual device execution, not comparison of one saved output twice.

All three processes passed **bitwise** equality:

- Within-process: native next key, frequencies, coefficients, fitting/validation predictions and validation loss.
- Between-process: all saved numerical arrays, including initialization and separately reconstructed feature/Gram/RHS matrices.
- Source/fixture and mapped numerical-library hashes matched; CPU/runtime identities and settings are retained in minimal_0/1/2.json.
- Lowered program text, optimized program text and runtime fingerprints matched exactly. There were **no program-dump differences to normalize away**, semantic or nonsemantic. Process IDs and private cache paths differ in provenance, not program dumps.

The common lowered hash is `15cf4d8dcfefdc0c7037865f8d15a5c8575e4dd289bb2fce2cb3367b13849157`; optimized hash is `d613541db9106241c23560eb1e010dd909ff0f7a7d3c0c63bb1201bb7322f32d`. BACKEND_EXCERPTS.txt records selected backend configurations. Full dumps remain with each external process record.

Native coefficients/frequencies/keys are outputs of the executed step. Feature/Gram/RHS matrices are **separate reconstructions**, not captured internal native buffers. Fitting and validation predictions/losses are explicit post-step evaluations using the native model. No intermediate output was added to the native step signature. This limitation prevents claiming that identical reconstructed matrices prove every hidden internal operation was identical.

Only after this exact three-process gate passed did the controller dispatch the two complete fits. All three minimal processes were required and run; no failed result was discarded.

## Complete-path verification

Exactly two fresh processes executed corrected Ex8 ARFF K128, nominal N80000, seed 0: all five nuisance fits, final drift, covariance fit, 300 adaptations per regression, internal selection, OOF targets, three-split evaluation and serialization. Frozen lambda=.001, delta=.2, gamma=1, Metropolis on, resampling off, five folds, internal holdout .1, no post-selection refit and SPD evaluation floor .001 were retained.

Native schema validation passed for both artifacts. Dataset/source/configuration hashes, splits, final PRNG key, every saved frequency/amplitude array, OOF targets, validation/MSE/moving-average histories, selected iterations and all reported metrics match **bit-for-bit**. Saved-model drift/raw covariance predictions on train, validation and test also match bit-for-bit. The original tolerances pass with zero numerical differences for those scientific outputs.

The nonidentical fields are exclusively timing values/histories and run-specific environment-cache paths, command/output paths and timestamps. Serialized NPZ hashes consequently differ; byte-identical archive files are not claimed. FULL_COMPARISON.json includes every field, including expected timing differences, so a timing mismatch is not silently removed from the record.

| Regression | Full 0 | Full 1 | Historical archive |
|---|---:|---:|---:|
| fold 0 | 1 | 1 | 1 |
| fold 1 | 1 | 1 | 1 |
| fold 2 | 3 | 3 | 4 |
| fold 3 | 4 | 4 | 4 |
| fold 4 | 1 | 1 | 1 |
| final drift | 4 | 4 | 4 |
| covariance | 299 | 299 | 299 |

## Historical agreement is separate

| Test metric | Either new-policy run | Historical archive | New − historical |
|---|---:|---:|---:|
| Drift RMSE | 0.912831843 | 0.912942946 | −0.000111103 |
| Raw covariance RMSE | 0.113099642 | 0.113298044 | −0.000198402 |
| Gaussian NLL | −8.879138947 | −8.944207191 | +0.065068245 |

Neither run passes the original historical comparison. HISTORICAL_0.json and HISTORICAL_1.json retain all split metrics, selected indices, model and history differences; neither replaces the original failed comparison. The numerical change is not used to prefer this policy scientifically. No test result selected a flag, hyperparameter or checkpoint.

Observed algorithm times were 16.878344 and 17.131590 seconds; warm-up/compilation 17.613223 and 17.500019 seconds. Whole-process walls were 49.919 and 49.751 seconds. These two observations are not a performance campaign or a speed claim. All native timing definitions remain unchanged.

## Optional verified reproduction command

The following is **opt-in** for the tested corrected Ex8 ARFF K128/N80000/seed0 configuration, exact isolated software and this A6000/driver. It is not installed as a production default. Start from a restored, verified checkout at 4c5dc6b and the same isolated venv; choose a new absolute output root and an idle GPU with the stated UUID.

```sh
REPO="$PWD"
RESOURCES="$(cd ../bounded_reproduction_verification_v1 && pwd)"
PY="$RESOURCES/venv/bin/python"
RUN=/new/absolute/ex8-policy-verification
mkdir "$RUN"
mkdir "$RUN/jax-cache" "$RUN/cuda-cache"
cd "$RESOURCES/checkout"
flock -n "$REPO/results/production/ex8_campaign_control/gpu_0.lock" \
 env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE \
 XLA_FLAGS=--xla_gpu_autotune_level=0 \
 JAX_COMPILATION_CACHE_DIR="$RUN/jax-cache" CUDA_CACHE_PATH="$RUN/cuda-cache" \
 CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda JAX_ENABLE_X64=false \
 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
 "$PY" -B scripts/reproduction_route.py baseline --method arff --seed 0 \
 --execute --output "$RUN/model"
```

The tested detached launcher additionally checks occupancy/UUID, freezes exclusive paths and gates full dispatch. See scripts/verify_autotune0_policy.py. The command documents a verified scope, not authorization to launch more jobs now.

## Conclusions and verification boundary

1. **Minimal reproducer stabilized:** yes, three fresh processes × two executions, bit-identical numerical outputs and program identities.
2. **Complete current path repeated:** yes, two fresh full executions, bit-identical scientific outputs and predictions for this tested configuration under the documented policy/cache isolation.
3. **Historical archive agreement:** no, for either execution at the original tolerance.
4. **Unverified:** generated-data reproduction, other seeds/configurations, other estimators, other software versions/GPUs/hardware, general determinism and distributional reproduction. Full-run optimized kernels were not individually dumped; complete-run equality is established from outputs, not an asserted all-kernel identity trace.

The results do not establish the cause of historical variation. Prior failures and manuscript results remain untouched. No default change, flag sweep, retry or further full run was performed or queued. The supervisor completed and exited.

## Records

The external workspace `../autotune0_policy_v1/execution/` contains support/launch logs, fresh caches, three minimal captures and program dumps, two native artifacts, historical comparisons, and COMPLETE.json. Compact gate, comparison, environment/configuration records and backend excerpts are in this directory. SOURCE_INDEX.json records source and preserved comparison hashes. Large artifacts/caches remain outside Git. SCOPE.md records the official documentation source and bounds.
