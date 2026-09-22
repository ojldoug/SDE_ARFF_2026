# Bounded reproduction discrepancy investigation

**The reference comparison remains failed at rtol=1e-5, atol=1e-6.** Its report, arrays and original tolerances are preserved. No complete training run, adaptation step, new random data, checkpoint selection or manuscript change occurred here.

The new result is specific: four standalone fixed-basis native solves, two in each current environment, reproduce the archived final-drift amplitudes **bit-for-bit**. They do not reproduce the new full-run amplitudes. This rules out neither compiler-context sensitivity nor adaptive-path sensitivity, but supplies no evidence of an ordinary fixed-basis environment mismatch or within-process solve non-repeatability in these four calls. The exact historical-to-new arithmetic cause remains unresolved.

## 1. Environment comparison

Historical run: September 9, 2026; reproduction: September 22. The current research environment is not assumed to be a preserved snapshot of September 9.

| Item | Archived full run versus new full run | Evidence/qualification |
|---|---|---|
| Python | Confirmed match: 3.11.15 | Artifact fields; executable paths differ (research environment versus new venv) |
| JAX / jaxlib / CUDA plugin / PJRT | Confirmed match: 0.10.2 | Artifact package_versions_json |
| NumPy | Confirmed match: 2.4.6 | Artifact field |
| Host / OS | KW61146, Linux 5.15.0-190, glibc 2.35 match | Artifact fields |
| GPU | RTX A6000, device index 0 match | Artifacts and archived exit.json; historical UUID not established from artifact |
| Learning precision | Confirmed float32, x64 disabled; default matmul precision None | Artifact fields. Native Gram/RHS explicitly use HIGHEST; feature/prediction matmuls retain their native defaults |
| XLA_FLAGS, JAX_COMPILATION_CACHE_DIR, JAX_DEFAULT_MATMUL_PRECISION | Absent from both recorded environment dictionaries | The native provenance function records these keys when present; absence is not a record of every compiler option/autotuning decision |
| CUDA_VISIBLE_DEVICES / JAX_PLATFORMS / JAX_ENABLE_X64 | Confirmed match: 0 / cuda / false | Artifact environment dictionaries |
| OMP_NUM_THREADS | Historical unset in recorded keys; new 2 | Confirmed difference |
| XLA_PYTHON_CLIENT_PREALLOCATE | Historical unset in recorded keys; new false | Confirmed difference; not demonstrated causal |
| OPENBLAS_NUM_THREADS | New supervisor set 1; historical unavailable | Not collected by historical artifact |
| cuBLAS/cuSOLVER/CUDA runtime/cuDNN/driver build identity | Historical runtime identity unavailable; new driver 595.84 | Major JAX version or current requirements alone cannot authenticate historically loaded library bytes |
| CUBLAS_WORKSPACE_CONFIG, NVIDIA_TF32_OVERRIDE, determinism options, compiler autotuning/cache state | Historical information unavailable | Not collected by historical provenance; no explicit determinism setting was introduced in this check |
| Original campaign scheduling | Four-worker campaign versus single idle-GPU reproduction | No same-GPU concurrency intended; shared-host context differs, without demonstrated arithmetic consequence |

**Current-environment cross-check:** all numerical package versions match between the current research installation and new venv. Only pip (26.2.1 versus 24.0), setuptools (83.0.0 versus 79.0.1), and wheel (0.47.0 versus absent) differ. CUDA runtime package 13.3.29, cuBLAS 13.6.1.10, cuSOLVER 12.2.6.9 and cuDNN 9.24.0.43 match. SHA-256 of every compared mapped CUDA/JAX shared library matches, including libcublas, libcublasLt, libcudart, libcusolver, libcudnn, libcuda and libjax_common. Full inventories/hashes are in the two environment JSONs and RESULTS.json. This comparison is current-to-current, not proof of historical CUDA identity.

Both diagnostic processes used the same GPU, source checkout, runtime flags and inputs. We did not sweep flags or change solvers/precision to find agreement. Historical unset-versus-explicit operational settings were not tested as interventions.

## 2. Earliest observable divergence

The preceding verification confirmed exact dataset, scientific configuration/source hashes, training/validation/test IDs, OOF fold IDs, internal holdout IDs and final PRNG key. Identical input construction can be reconstructed from these records. Actual transient historical float32 x/y buffers were not saved, so a claim of measured historical buffer equality would exceed the evidence.

| Stage | Earliest unequal stored validation loss | First loss outside original tolerance | Archived → new selected iteration |
|---|---:|---:|---:|
| fold 0 | 2 | none | 1 → 1 |
| fold 1 | 2 | none | 1 → 1 |
| fold 2 | 2 | none | 4 → 4 |
| fold 3 | 1 | 180 | 4 → 4 |
| fold 4 | 3 | 229 | 1 → 1 |
| final drift | 2 | none | 4 → 4 |
| covariance | 1 | 8 | 299 → 294 |

The first stage's **selected iteration-1 amplitudes already differ**, maximum 1.22219e-4, RMS 2.25975e-5, while its selected frequencies and scalar iteration-1 validation loss agree exactly. Thus a rounded scalar loss can hide model differences. This is the earliest retained checkpoint evidence, not the first differing arithmetic operation. Initial amplitudes, proposal amplitudes, intermediate frequency populations, uniforms/acceptance decisions and internal matrices were not saved at every iteration. We cannot identify the first Metropolis divergence.

All selected nuisance and final-drift frequencies match exactly. Final-drift amplitudes differ by max 8.17508e-4, RMS 5.45125e-5. Saved OOF targets differ by max 1.84059e-4, RMS 8.68966e-6; thus covariance training did not receive bit-identical targets. Covariance selected frequencies and selected checkpoint differ substantially. Its first stored validation loss already differs; 285 of its 300 losses fail tolerance. These observations support propagation through the two-stage adaptive procedure, but do not prove which differing target or arithmetic operation caused a particular acceptance decision.

**Availability boundary:** historical/new full-run feature matrices, Gram matrices, RHS and fitting predictions were not archived. The new fixed-basis matrices below are recomputations, not recovered original buffers. Saved test metrics differ, while new-model reload reproduces its own metrics (prior REEVALUATION.json). No historical intermediate prediction equality is inferred from those scalar metrics.

## 3. Precisely bounded computation

The protocol was saved before execution. Exactly **four** amplitude solves ran, with no additional warm-up solve: two sequential jitted native fit_amplitudes calls per environment. They reused the archived final-drift basis (K128, selected iteration 4), the exact archived 72,000 fitting positions from the N80000 corrected dataset, native float32 conversion and increment/h targets, and lambda=.001. No test data were indexed. Both ran under the existing cooperative GPU-0 lock after idle occupancy was checked. The source regression.py hash matches the archived/native source.

Per-process numerical work through matrix/prediction/residual calculation took approximately 14.45 and 14.56 seconds, excluding final serialization/library hashing. These are diagnostic wall times, not benchmark timings. Compact arrays remain outside Git at `../reproduction_discrepancy_v1/{isolated,current_research}/arrays.npz` relative to the repository. Each is about 2 MiB. No accepted checkpoint was overwritten.

| Comparison | Result |
|---|---|
| Input x, y, frequencies, recomputed feature matrix | Identical SHA-256 across current environments |
| Recomputed Gram, RHS, regularized Gram | Bit-identical across current environments |
| Two amplitudes within each process | Bit-identical |
| Amplitudes across current environments | Bit-identical |
| Fixed-basis amplitudes versus archived full-run amplitudes | Bit-identical |
| Fixed-basis amplitudes versus new full-run amplitudes | Fail unchanged tolerance; max 8.17508e-4, RMS 5.45125e-5 |
| Fitting predictions across repeats/environments and versus archived-amplitude predictions | Bit-identical |

The native solve is unmodified. Feature/Gram/RHS recomputation is separate from its JIT and uses the same mathematical formula with HIGHEST products. It is not claimed to expose the native adaptation kernel's exact internal buffers: changing compilation outputs/context can change arithmetic. Residual norms below are computed in float64 **for diagnosis only**, against the recomputed float32 A=Phi^T Phi+n lambda I and b=Phi^T y; no solve precision was changed.

| Amplitudes | ||A beta-b||F | Relative to ||b||F | Normwise backward error |
|---|---:|---:|---:|
| Archived and all four fixed-basis solves | 1.319215 | 2.74876e-6 | 1.32629e-8 |
| New full-run saved amplitudes | 1.101863 | 2.29588e-6 | 1.10777e-8 |

Backward error is ||A beta-b||F/(||A||F ||beta||F+||b||F). Both residuals are small; the new amplitudes even have a smaller residual against this reconstructed system. This does **not** mean they are scientifically better, nor establish either as an exact solution of the unsaved historical system. Small residuals do not bound coefficient error without conditioning information. No new condition-number/SVD experiment was run.

## 4. What the evidence supports

- **Packaging/scientific configuration discrepancy:** none found. Hashes, splits, source and package checks pass; both environments reproduce the archived standalone amplitude solution. This does not certify every unrecorded historical runtime property.
- **Repeatability within one environment:** no discrepancy in two repeated fixed-basis calls in either process. This is not a repeatability test of the full adaptive trajectory or independent process restarts.
- **Differences between current environments:** none observed for this fixed-basis computation; loaded numerical libraries agree. Historical versus current lower-level runtime identity remains incompletely recorded.
- **Numerical sensitivity amplified by adaptation/selection:** consistent with saved amplitude differences, perturbed OOF targets, differing covariance histories/frequencies and checkpoint 299→294. Standalone solves recover the archive even in the new environment, focusing suspicion on full-path arithmetic/compilation context or runtime state. No causal attribution to a particular kernel, allocator, precision option or first acceptance branch is established.
- **Root cause:** unresolved. Do not relabel the failed full-run comparison as a pass because the bounded standalone check succeeds.

## 5. Verification boundary and one possible next check

**Saved-checkpoint evaluation:** the new saved model reconstructs its own test metrics within original tolerance; this verifies loading/evaluation/serialization consistency. Exact archived final-drift amplitudes are also recovered by the present standalone check. Neither statement proves that every archived model is reproducible by fresh adaptation.

**Fresh training:** one complete seven-fit run executes and validates with exact scientific input identities; it fails agreement with the archived trajectory/result at the predeclared tolerance. No new full run was attempted here. Manuscript aggregates/checkpoints remain untouched.

If a further check is authorized, the concrete remaining question is whether the native **first fold's iteration-1 compiled adaptation context** differs from the standalone amplitude solve on the same final frequencies. A single initialization/one-step capture on that fold, with native uninstrumented output compared to a separately instrumented copy, could localize the first retained mismatch without a 300-step or seven-fit rerun. Instrumentation itself must be treated as a possible compiler-context change; do not call an instrumented trace an exact native replay without checking its outputs. This check is proposed only, not executed or automatically queued. No broader experiment or tuning is required by this report.

## Reproduction/source records

- PROTOCOL.md: frozen four-solve bound and comparisons.
- RESULTS.json: stages, current environments, matrix/solve comparisons, input/output hashes.
- isolated_environment.json and current_research_environment.json: exact package versions, numerical flags, mapped-library hashes, residuals.
- NEW_AMPLITUDE_RESIDUAL.json: CPU residual calculation using saved arrays only.
- SOURCE_INDEX.json: source and preserved failed-comparison identities.
- scripts/check_reproduction_fixed_basis.py: two native solves per invocation; exclusive output directory.
- scripts/summarize_reproduction_discrepancy.py: saved-array analysis only.

The run used `--checkout ../bounded_reproduction_verification_v1/checkout --artifact ../bounded_reproduction_verification_v1/checkout/results/controlled_study_2026/float64_v2/production/baseline/K_128_N_80000/arff/seed_0/artifact.npz`, once with each Python executable, new output directories, CUDA_VISIBLE_DEVICES=0, JAX_PLATFORMS=cuda, JAX_ENABLE_X64=false, XLA_PYTHON_CLIENT_PREALLOCATE=false, OMP_NUM_THREADS=2, OPENBLAS_NUM_THREADS=1, and unset PYTHONPATH/XLA_FLAGS. Full local console records are preserved outside Git. No original comparison tolerance or artifact changed. Investigation stops here.
