# Fold 0, iteration 1: native-context comparison

**The archived-versus-new checkpoint difference is reproduced, but a native-versus-standalone difference on the identical fold-0 problem is not.** The native one-step output and standalone solve match each other and the new full-run checkpoint bit-for-bit. Both differ from the historical archived checkpoint. The original failed comparison and rtol=1e-5, atol=1e-6 remain unchanged. Numerical work stops after this single comparison.

## Exactly what ran

On idle A6000 GPU 0, under the existing cooperative lock, the isolated verification venv imported the authenticated numerical source from pinned commit `4c5dc6b54af216ebba49477874c955d259d4140f`. Python 3.11.15, JAX 0.10.2, NumPy 2.4.6; float32 learning, x64 disabled, default matmul precision with the native explicit HIGHEST Gram/RHS products. No precision, solver or scientific setting changed.

The corrected packaged dataset hash is `6fb009e3d6fa5f241cb1a15f9a9815cbd16ea19c3a9e4ca546ae99bdae091a72`. Canonical training IDs were checked against the dataset. The five-fold split (seed 2026) and internal holdout (seed 101000) were reconstructed and checked exactly against the archived fit and holdout positions: 64,000 fold-0 regression-pool observations, 57,600 amplitude-fitting rows, 6,400 internal-validation rows. Targets followed the native operation order: gather the fold pool, divide float32 increments by float32 h, then take the internal fitting positions.

Initialization used native zero frequencies, native initial amplitude solve and PRNGKey(0). These are source-defined reconstructed initialization values; historical initialization buffers and intermediate keys were not archived, so they cannot be checked directly against saved arrays. No new seed or initialization was chosen.

Exactly four native amplitude-solve invocations occurred:

1. Zero-frequency initialization, as in the native fitter.
2. Proposal amplitude solve inside one unchanged compiled native step.
3. The required mixed-population refit inside that same step.
4. One standalone jitted native amplitude solve on the **same resulting fold-0 frequencies and fitting data**.

No second iteration, other fold, whole-path seven-regression warm-up, checkpoint selection, test inference or full training run occurred. The single native step includes the Metropolis decision needed to reach the saved first-step model; the adaptation loop was not completed. Numerical diagnostic work after initialization took 29.87 seconds, including compilation and matrix diagnostics, excluding final archive compression. This is not a benchmark timing measurement.

The reconstructed first proposal, native returned frequencies, archived fold-0 frequencies and new full-run frequencies all match exactly. Hence this comparison does not confound different selected bases. Captured initial/next keys, indices, inputs and array hashes are retained.

## Results

Predictions below are on the 57,600 fitting rows, not test data. Differences use the original tolerance without adjustment.

| Comparison | Maximum coefficient difference | RMS coefficient difference | Maximum prediction difference | RMS prediction difference |
|---|---:|---:|---:|---:|
| Native vs standalone | 0 | 0 | 0 | 0 |
| Native vs new full-run checkpoint | 0 | 0 | 0 | 0 |
| Native vs archived checkpoint | 1.222193e-4 | 2.259753e-5 | 7.476807e-4 | 3.580811e-4 |

The last row fails the original tolerance for both coefficients and predictions. The native internal-validation MSE is **5191.619140625**, exactly equal to both archived and new recorded first-step scalar losses. Scalar rounding therefore hides a real model/prediction difference.

**Earliest observable historical discrepancy:** first-step amplitudes. Historical/new full-run feature matrices, Gram matrices and RHS were not saved. Their equality or earliest difference cannot be measured retrospectively. Here the native and standalone amplitudes agree; no earlier difference between those executions is established from hidden buffers.

The earlier final-drift fixed-basis check matched historical amplitudes. This fold-0 check instead matches new amplitudes. Those are different regression problems: neither observation can substitute for the other or establish a general standalone-versus-adaptation explanation.

## Captures and instrumentation limits

The native `make_compiled_adaptation_step` was invoked unchanged, with its original outputs and static scientific settings. Its lowered StableHLO is retained in `native_lowered.txt`. No callback, intermediate return value, monkey patch, altered solver or instrumented native rerun was introduced. Lowering exposes the graph, not runtime matrix values or a historically authenticated optimized executable.

The captured features, Gram, RHS and regularized Gram were **separately reconstructed** from the identical input/basis using the native feature function and explicit HIGHEST products. They are not asserted to be the native compiled kernel's internal buffers. Capturing those buffers directly would require changing the computation's outputs and potentially fusion/compilation. This check deliberately does not make that change.

The source-native static ridge shift rounds to **57.599998474121094** in float32. The standalone dynamically passed lambda expression rounds to **57.60000228881836**, a difference of **3.814697265625e-6**. These scalars were captured independently; the lowered native graph shows the static constant. Native and standalone coefficients nevertheless match exactly. Thus this observed scalar rounding difference does not explain the archived amplitude discrepancy in this check, and no alternate-lambda experiment was performed. Both calls retain lambda=.001 and the existing native source formula.

The bounded context is not a replay of all surrounding full-run execution state: it omits the whole-path warm-up and other compilation/allocation activity. Matching the new checkpoint is informative, but it does not authenticate historical internal matrices or prove identical compiler/autotuning state.

## System residuals and conditioning

Diagnostics use the captured float32 system A=Phi^T Phi+57.6 I and b=Phi^T y, promoted to float64 only for residual/eigenvalue calculations. The amplitude solves remain native float32. Residuals are with respect to this reconstructed system, not an asserted original internal system.

| Amplitudes | ||A beta-b||F | Relative to ||b||F | Normwise backward error |
|---|---:|---:|---:|
| Native / standalone / new checkpoint | 0.3127448 | 9.060459e-7 | 4.882039e-9 |
| Archived checkpoint | 0.3683765 | 1.067215e-6 | 5.750472e-9 |

Backward error is ||A beta-b||F/(||A||F ||beta||F+||b||F). Neither residual proves scientific superiority or identifies a faulty solve.

The symmetrized captured regularized Gram has minimum eigenvalue **57.47669**, maximum **6,635,457.47**, and spectral condition number **115,446.06**. Condition number times float32 epsilon is about **0.01376**. This permits amplification of small arithmetic/system perturbations and explains why small backward errors do not guarantee identical coefficients. It is a sensitivity bound/scale, not a measured perturbation, predicted error or proof that conditioning caused the historical difference. No historical Gram/RHS perturbation is available to propagate through that bound.

## What remains unproven

- A packaging/scientific-configuration error is not supported: inputs, split positions, source hashes, proposal and returned frequencies agree.
- A native-versus-standalone compilation-context amplitude gap is **not reproduced** on the identical fold-0 problem.
- The archived checkpoint discrepancy **is reproduced**, with the new checkpoint recovered exactly. Its cause is not identified by the captured quantities.
- Historical CUDA/kernel identity, compiler/autotuning state, surrounding warm-up/allocation effects and unsaved internal matrix differences remain possible explanations, not established ones.
- Adaptive amplification remains consistent with the earlier differing OOF targets and later covariance checkpoint. This one-step check does not identify the later first differing Metropolis decision or attribute it causally.

Saved-checkpoint evaluation remains validated for the new model's own recorded metrics. Fresh training agreement with the archived full result remains failed. No result is substituted into the manuscript. No further run, instrumentation variant or solver/compiler sweep is launched or queued.

## Records and invocation

Small records: PROTOCOL.md, RESULTS.json, SOURCE_INDEX.json and native_lowered.txt. The full capture (about 54 MiB), including coordinates, targets, feature/Gram/RHS matrices, ridge shifts, amplitudes, fitting predictions, initialization and keys, remains outside Git at `../reproduction_fold0_step1_v1/capture.npz` relative to the repository. Full stdout is `../fold0_step1_console.log`.

The executed command used the existing isolated `venv/bin/python -B scripts/check_reproduction_fold0_step1.py` with `--checkout ../bounded_reproduction_verification_v1/checkout`, the archived baseline seed-0 `--reference`, `--new ../bounded_reproduction_verification_v1/run/artifact.npz`, and exclusive `--output ../reproduction_fold0_step1_v1`. Environment: CUDA_VISIBLE_DEVICES=0, JAX_PLATFORMS=cuda, JAX_ENABLE_X64=false, XLA_PYTHON_CLIENT_PREALLOCATE=false, OMP_NUM_THREADS=2, OPENBLAS_NUM_THREADS=1; PYTHONPATH and XLA_FLAGS unset. The existing `results/production/ex8_campaign_control/gpu_0.lock` was held throughout execution.
