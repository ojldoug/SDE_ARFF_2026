# Minimal fresh-process repeatability and execution audit

**Strict fresh-process repeatability is not established under the recorded settings.** The latest full repeat recovered the historical selected models/histories, but disagreed with the preceding current full run. Both complete runs and the original failed comparison remain unchanged. This investigation ran no complete pipeline.

The bounded controller stopped after **two of the maximum four** fresh processes: identical verified inputs produced different first-step amplitudes and predictions. The lowered program text was identical, while optimized native program text/fingerprints and a concrete projection implementation differed. This is evidence of differing executable selection under the recorded settings, not evidence that a particular library is defective, nor retrospective proof of what happened inside the two complete runs.

## 1. Identity audit of the two complete current runs

| Item | Finding |
|---|---|
| Actual physical GPU | Confirmed identical UUID `GPU-d2cfc73c-186a-6651-e9f1-c6d93a00f0b2`, index 0, RTX A6000 |
| Driver | Confirmed identical: 595.84, recorded in both supervisor status files |
| Interpreter/package versions | Same isolated executable, Python 3.11.15; byte-identical pip-freeze lists; JAX/jaxlib/plugin/PJRT 0.10.2, NumPy 2.4.6 |
| Loaded Python/module paths | Same paths except the external supervisor `__main__` filename; checkout file-open sets identical |
| Loaded native-library identities during the full runs | **Not recorded**. Same installation/path/version is not a retrospective binary mapping/hash record. Later fixed-basis library hashes cannot supply that missing historical measurement |
| Recorded environment | Exact dictionary match, including CUDA_VISIBLE_DEVICES=0, JAX_PLATFORMS=cuda, JAX_ENABLE_X64=false, preallocation=false, OMP_NUM_THREADS=2, OPENBLAS_NUM_THREADS=1, Python isolation flags |
| XLA_FLAGS / explicit JAX cache directory | Both supervisors explicitly removed XLA_FLAGS and JAX_COMPILATION_CACHE_DIR; effective compiler/cache state was not captured |
| Full inherited environment | Unavailable. In particular LD_PRELOAD/LD_LIBRARY_PATH, CUBLAS_WORKSPACE_CONFIG, NVIDIA overrides and all PATH/runtime discovery inputs were not exhaustively recorded |
| Launch mechanics | Both detached tmux supervisors, same cooperative lock and pinned-checkout cwd, same Python -B/runpy numerical route. Wrapper/output names and the repeat's resource-path argument differ; numerical launch settings do not |
| Effective JAX config, optimized executable, cache contents, CPU affinity during full runs | Unavailable retrospectively |

Source/data/configuration/split identities are covered by existing hashes. Installed native binaries, compiler/autotuning decisions, process memory/allocation state, CPU scheduling/affinity, driver caches and system GPU state are outside those scientific input/configuration checks. Historical metadata/provenance calls also inspect Git/environment state, but they do not define the regression targets. The Python file-open guard is not an OS-level audit of native-library reads. No undocumented dataset or alternate source path was found; the numerical root cause cannot be assigned from matching scientific hashes alone.

## 2. Minimal frozen problem and command

`fixture.npz` contains the already authenticated fold-0 fitting x/y, fit/holdout IDs, initial key and zero frequencies, expected selected first-step frequencies, and both reference amplitude arrays. Every extracted array was verified against the prior capture's recorded SHA-256. It has 57,600 fitting rows, two inputs/outputs, 128 frequencies, float32 data and native lambda=.001. Its archive SHA-256 is:

`fd71907f7be2889af9ad6d6ce082e6a5c6f729b9a692bdf69640288e5e366985`.

There is no new stochastic generation. The target fixture is the native float32 increment/h target, not a NumPy reimplementation of that transformation. Regression source SHA-256 is `e48d15e3701468255d1307467b864e94c672836fde17843a0f5f5827ccc759e7` at pinned commit `4c5dc6b`.

One process executes native initialization and one unchanged compiled native adaptation step (three source-level amplitude calls: initialization, proposal, mixed refit). It verifies returned frequencies against the fixture. It does not run a standalone solve, later iteration, other fold, selection or test inference. The native output signature is unchanged; no callbacks/intermediate outputs were injected.

Minimal command, after obtaining the preserved fixture and pinned restored checkout:

```sh
# Use the existing isolated interpreter and an idle UUID-verified GPU 0.
REPO="$PWD"
RESOURCES="$(cd ../bounded_reproduction_verification_v1 && pwd)"
PY="$RESOURCES/venv/bin/python"
CHECKOUT="$RESOURCES/checkout"
FIXTURE="$(cd ../minimal_fold0_repeatability_v1 && pwd)/fixture.npz"
cd "$CHECKOUT"
flock -n "$REPO/results/production/ex8_campaign_control/gpu_0.lock" \
 env -u PYTHONPATH -u PYTHONHOME -u PYTHONUSERBASE -u XLA_FLAGS -u JAX_COMPILATION_CACHE_DIR \
 CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda JAX_ENABLE_X64=false \
 XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
 PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
 "$PY" -B "$REPO/scripts/minimal_fold0_reproducer.py" \
 --checkout "$CHECKOUT" --fixture "$FIXTURE" \
 --fixture-sha256 fd71907f7be2889af9ad6d6ce082e6a5c6f729b9a692bdf69640288e5e366985 \
 --output /new/exclusive/output
```

This command documents the reproducer, not authorization for more executions. The controller ran both children from the pinned-checkout cwd and checked occupancy before each. `scripts/run_minimal_fold0_check.py` records the exact bounded invocation/settings. The example explicitly restores that cwd. `scripts/extract_minimal_fold0_fixture.py` can extract/verify this fixture from the retained preceding capture and its RESULTS.json without evaluation or fitting. The fixture/capture are local-only, outside Git; no public download availability is claimed.

Differences from the complete pipeline: disk-loaded already-converted fitting tensors instead of full dataset loading/gather/division, omitted seven-fit warm-up, no other shapes/compiled functions, different wrapper/source location, no validation/test pipeline, fewer allocations. Initial amplitudes are recomputed natively. These differences bound the reproducer's relevance; it does not reproduce all full-process state.

## 3. Two-process result and captures

Both processes used the same UUID/driver, input fixture/source hashes, recorded environment, effective JAX configuration, CPU affinity and hashes of actually mapped CUDA/JAX libraries. Both had JAX compilation caching enabled as a setting, but no configured JAX cache directory; `.cache/jax` did not exist. The default CUDA ComputeCache existed with 35 files and matching path/size/mtime inventory digest at the recorded times. This is an inventory, **not a content hash or proof of identical cache accesses**; snapshots were taken after execution. LD_PRELOAD was not recorded by this diagnostic's environment filter either, so exhaustive environment equality is not claimed.

| Quantity, process 0 versus 1 | Bitwise match? | Maximum absolute difference | RMS difference |
|---|---|---:|---:|
| Native initialization amplitudes | Yes | 0 | 0 |
| Native returned frequencies and next key | Yes | 0 | 0 |
| Native returned amplitudes | No | 0.002729774 | 0.0004884715 |
| Separately reconstructed features | No | 0.0006608739 | 0.0000629760 |
| Separately reconstructed Gram | No | 6.477783 | 0.9101259 |
| Separately reconstructed RHS | No | 11.81152 | 2.201862 |
| Ridge shift | Yes | 0 | 0 |
| Fitting predictions | No | 0.006225228 | 0.001921171 |

Numerical disagreements retain rtol=1e-5, atol=1e-6 and are not turned into passes. The static native ridge shift remains float32 57.599998474121094. Process 1 exactly recovered the archived fold-0 amplitudes; process 0 matched neither complete-run checkpoint. Full hashes, reference-amplitude differences and exact/tolerance comparisons are in COMPARISON.json.

Native coefficients/frequencies/key are direct outputs of the executed calculation. Features, Gram/RHS and fitting predictions are separate diagnostic calculations using the same inputs and returned basis. Their differences are observable but cannot be identified as the native kernel's first internal divergence. The native internal feature/Gram/RHS buffers were not exposed; instrumentation to return them would change compilation. Initialization agrees; among retained native outputs the earliest difference is the first-step amplitude array. In the separate diagnostic chain the first retained difference is the feature matrix.

## 4. Compiled-program evidence

Native lowered StableHLO is identical across processes, SHA-256 `15cf4d8dcfefdc0c7037865f8d15a5c8575e4dd289bb2fce2cb3367b13849157`.

Optimized program hashes differ:

- Process 0: `4d30764022f377bae19158dbd0c4d8a4ad27a828740d2edbea88970b192b0ec9`
- Process 1: `8d42d3d00574032230047a2585ae40067807bb37f096fa7636c6773aea5f218e`

Runtime executable fingerprints also differ. Program text was obtained by lowering/compiling the same cached signature after the native call; no second native step was executed. Fingerprints/text identify the returned compiled program, not a saved historical GPU instruction trace. Optimized texts include metadata differences, so unequal text hashes alone would not demonstrate different arithmetic. Here there is an additional concrete difference:

- Process 0 contains a `__cublas$lt$matmul` custom call for the mixed-frequency feature projection, with default operand precision and selected_algorithm 4.
- Process 1 implements that projection as a `__triton_nested_gemm_fusion` with a 64×64 output tile and two warps.

`PROGRAM_EXCERPTS.txt` preserves those exact lines. This is observed compiler/backend implementation variation under the same recorded settings. The independently reconstructed feature matrices also differ. It is consistent with context/backend arithmetic sensitivity, but does not isolate which arithmetic implementation caused which amplitude perturbation. No TF32, autotuner timing, library-bug or solver-failure attribution is established. The two complete runs did not save optimized programs, so their two outcomes cannot be retroactively assigned to these two choices.

## 5. Practical conclusion and corrective action

The minimal problem **does exhibit fresh-process variation**, so the investigation stopped after two processes. It is not just a historical-versus-current package comparison. Inputs, scientific settings and even lowered program identity do not freeze the optimized executable in this test.

No identified execution difference conclusively explains the exact two complete-run outcomes. We now have a concrete optimized-program difference in the bounded reproducer, but not a controlled causal intervention or historical executable comparison. Preserve that distinction.

The smallest justified action is a **reproducibility-record correction**: state the demonstrated limitation and retain GPU UUID, runtime/library identities, effective compiler/cache configuration, optimized-program hashes and outputs with any future authorized verification. This diagnostic implements that recording. There is no supported solver, precision, seed, checkpoint or compiler-flag change to apply automatically. We do not prescribe a numerical fix from this evidence or broaden the campaign.

Saved-checkpoint evaluation remains a distinct verified route; complete fresh training is not strictly repeatable at the original tolerance under the recorded settings. Neither manuscript results nor the old failed comparison are replaced. No full training, sweep, tuning or follow-up process is queued.

## Preservation

Both original full artifacts and preceding reports remain unchanged. All minimal fixtures, two ~54 MiB numerical captures, lowered/optimized texts, logs and terminal COMPLETE.json remain at `../minimal_fold0_repeatability_v1/`. Small source scripts, protocol, environment records, array/program hashes, comparison and excerpts are in Git. EXECUTION_AUDIT.json distinguishes recorded full-run facts from unavailable information. No entire-study independent-reproduction claim is made.
