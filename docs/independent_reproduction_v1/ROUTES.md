# Independent reproduction routes (v1)

Scope: the conservative review at `results/conservative_manuscript_revision_v1/`, based on commit `4e62557e5f9074e63843af9ef2d4bdd43549baab`. This package enables access and inspection. A subsequent [bounded full-fit verification](../bounded_reproduction_verification_v1/REPORT.md) passed execution and self-reconstruction but failed archived-reference numerical agreement; broader independent reproduction is not established. Ex4, withdrawn trajectory/speed claims and the unauthenticated modified-Ex7 scaling fit are not reproduction targets.

## Availability and exact dependency identity

`artifact_manifest.json` is the complete file-level inventory for this release: repository-relative path, exact SHA-256, byte length, tracked/local-only status, purpose and incoming hash references. `missing_referenced_files` concerns computational files; private manuscript preservation dependencies are explicitly excluded. No URL is available. Original float32 Ex8 models and older forensic antecedents appear only where existing provenance guards require their preservation; they are not corrected manuscript result points. Source/configuration identity is the archived hash, not merely a current Git revision or fitting seed. The payload includes some generated audit/provenance files because original integrity guards hash them; they are not new scientific inputs.

| Retained evidence | Archived rebuild | Exact saved-model evaluation | Fresh data/fitting prerequisites and route |
|---|---|---|---|
| Ex1 | Git: historical scalar CSV + seed-0 coefficient grid | Bundle: `data/ex1.npz`, 4×30 selected models under `results/final_reproduction/production/accuracy/ex1/`, historical source and manifests | `original --experiment ex1`; `historical --experiment ex1 --method METHOD`; fourier, mlp_shallow, mlp_deep, arff; seeds 0–29 |
| Ex2 | Same historical CSV, Ex2 rows | `data/ex2.npz`, 4×30 models at corresponding Ex2 path | Original Ex2 generator; same four historical methods/seeds |
| Ex3 | Same historical CSV, Ex3 rows | `data/ex3.npz`, shallow MLP and ARFF ×30 | Original Ex3 generator; only mlp_shallow/arff are the retained comparison; no invented Fourier/deep result |
| Ex5 | Same historical CSV, Ex5 rows | `data/ex5.npz`, 4×30 models | Fixed-lag SIR/SSA generator, population 1024; same four historical methods; splitting is by row, not independent trajectory |
| Ex6 | Same historical CSV, Ex6 rows | `data/ex6_labels_v2.npz`, 4×30 models | Original wave dataset + its metadata, `correct_ex6_labels_v2.py`, changes NPZ and validation record; historical methods on corrected labels only |
| Ex7 | Same historical CSV, Ex7 rows | `data/ex7.npz`, 4×30 models | Original Ex7 generator, four historical methods; not the withdrawn modified-Ex7 study |
| Corrected Ex8 baseline | Git baseline CSV and validated grids | `data/ex8_float64_v2.npz`, five×30 selected models in `results/controlled_study_2026/float64_v2/production/baseline/K_128_N_80000/` | Original Ex8 + matched-noise correction/validation + N80000 data view; `baseline` route, five methods, seeds 0–29 |
| Ex8 N/h/K | Git N/h/capacity CSVs | Bundle `capacity_regime_ex8_v{1,2}` and `capacity_regime_ex8_final_h_v1` datasets, plans, reuse contexts, all checkpoints | Corrected baseline → coupled lags → extended N pool → larger lags; `regime` / `final-h` jobs selected from frozen plans; ten seeds. Reused checkpoints remain linked by hashes, not independently fitted replicates |
| Lag–ridge | Git all 27 scalar rows / pairs | Bundle `ex8_h_lambda_drift_v1/jobs`, own fold targets and models, and nine reused baseline models | `lag-ridge` route, h=.0001/.0004/.002, drift λ=.001/.008/.064, seeds 0/1/2; covariance λ=.001, all seven fits; no post-hoc arm |
| Fixed-basis ridge | Git paired/risk summaries | Bundle cases, A/C models, common surrogate arrays, spectrum inputs and source snapshot | `fixed-basis` runs existing fixed-basis diagnostic, including its registered amplitude solves. Its frequencies require preceding A/C fits, not just an algorithm seed |
| Adaptive B ridge | Git paired summary | Bundle baseline B, Bpost/Bpath models and histories | `adaptive-ridge --seed S`; existing archived B input required for Bpost. Parent A/B/C procedure is `scripts/arff_drift_K_phase2_final.py`, preserved in payload |
| M/MR resampling | Git recorded histories/pairs | Bundle M/MR selected and terminal models, `stream_0..2.npz`, jobs and RNG_SPEC | `resampling --seed S --arm M` or MR; same three repeats and shared noise. Recorded streams are essential for exact pairing |
| Hybrid / oracle / floor diagnostics | Git CPU scalar CSVs | Five-method baseline models, corrected test set, hybrid integrity records, oracle source | Existing CPU evaluation scripts; they create outputs exclusively and are not training. Floor grid and float32/float64 scopes stay distinct |

**File roles:** datasets (including split IDs), shared surrogate data and registered streams are essential inputs for exact replay. Selected checkpoints are essential to exact published-model evaluation but optional when retraining; nuisance checkpoints/OOF targets can be regenerated by the complete fit and must not be substituted across arms. CSVs, histories and validation/completion records are generated outputs. Duplicate dataset views, plots, reports and historical protection records are convenience/provenance, not additional independent observations. Code snapshots are essential whenever they differ from current code. No fitted preprocessing statistics exist: physical coordinates and the documented increment/residual transformations are used.

The original Ex8 normals were regenerated from PRNG key 0 with shape `(1000,100000,2)` and float32 draws, then used in float64 integration; shape/backend are part of the recipe. The correction stores hashes and numerical replay checks, not a claim that a scalar seed is portable across arbitrary JAX versions. Extended observations use the archived block namespaces/shapes and Brownian-coupling records. Surrogate data use NumPy's registered data/noise seeds and separate algorithm/internal-validation seeds in `seed_manifest.json`; the actual shared arrays and M/MR streams are shipped.

## Install and restore

Use a full Git clone (not a shallow export) for historical `git show` generator checks. Python 3.11, `requirements-dev.txt`, CUDA/JAX and compatible NVIDIA drivers are needed for numerical routes. Archive-only plots need NumPy/Matplotlib and TeX Live/latexmk. The initial packaging tested the existing environment. The subsequent bounded verification installed these exact requirements in a new venv successfully.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
export PY="$PWD/.venv/bin/python"
```

Obtain the versioned tar from the project custodians (no public hosting has yet been authorized). Restore into a **fresh clone**, never over a working study. Check the tar SHA-256 against `BUNDLE.json` first. Inspect `tar -tf` before extracting, then:

```sh
tar -xf /path/to/independent_reproduction_v1.tar --keep-old-files
"$PY" scripts/package_independent_reproduction.py verify
```

The payload also contains immutable copies of tracked computational files; GNU tar may report existing tracked files with `--keep-old-files`. Do not overwrite them: use the verifier to establish identical contents. For a quiet overlay, extract to an empty staging directory, verify with `--root /path/to/staging`, then copy only missing paths using `cp -an /path/to/staging/. .`. A hash mismatch is a stop, not permission to replace code.

The bundle preserves historical absolute path strings inside hashed provenance. Entry points translate the known old repository prefix **in memory**. They do not rewrite manifests or numerical sources. Private manuscript PDFs/TeX and private review text are excluded; surrogate entry points omit only their noncomputational preservation checks. All computational data/source guards remain active. No private attachment or correspondence is required to fit or score these estimators.

## Route A: archived-output rebuild (already verified)

```sh
"$PY" scripts/rebuild_review_snapshot.py --output /tmp/review-new
```

No external payload, JAX device execution or predictions are required. The existing 24 input-hash / 17 numerical table-group checks and ten figures are reused. This is the route to the manuscript tables, figures and PDF from the *published archived outputs*.

## Route B: evaluate exact saved models

Inspect only (no JAX predictions):

```sh
"$PY" scripts/evaluate_reproduction_checkpoint.py --inspect \
  --artifact results/controlled_study_2026/float64_v2/production/baseline/K_128_N_80000/arff/seed_0/artifact.npz \
  --dataset data/ex8_float64_v2.npz --protocol modern --method arff
```

For explicit evaluation remove `--inspect`, add `--output /tmp/arff-seed0-evaluation.json`, and set `CUDA_VISIBLE_DEVICES` to an idle GPU. Methods: arff, joint_fourier, split_fourier, joint_mlp, split_mlp. Historical models instead use `--protocol historical --split validation`, their own dataset and method arff/fourier/mlp_shallow/mlp_deep. The source's existing raw covariance/error and NLL evaluators are reused. Model-specific SPD floors are loaded from artifacts; projected coefficients never replace raw RMSE.

For the original nine coefficient-map reconstructions:

```sh
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/reproduction_route.py coefficient-grids \
  --execute --output /tmp/reconstructed-coefficients
```

This calls the existing complete metric gate (`rtol=1e-5, atol=1e-6`) before deterministic grid evaluation. It writes only to the new output. The explicit GPU setting survives the legacy evaluator's default. It does not choose new seeds/checkpoints. Neither evaluation command was executed in this packaging task; metadata inspections were.

Hybrid/oracle evaluations are in `scripts/evaluate_ex8_hybrid_jointmlp_arff_v2.py` and `scripts/evaluate_ex8_oracle_component_swap.py`; their outputs already exist in the bundle, so direct invocation correctly refuses replacement. For a new evaluation use a fresh checkout with baseline inputs and source/protocol files, but without the corresponding evaluation-output directory. Their CPU affinity/precision and floors are intentional. These commands were inspected, not executed here.

## Route C: fresh fitting with authenticated data

All commands below default to **inspect**, which validates inputs and prints the registered configuration. Add `--execute --output NEW_DIRECTORY` only when deliberately starting a job. Output must be outside accepted `results/`. No supervisor or campaign is started by these entry points.

```sh
"$PY" scripts/reproduction_route.py historical --experiment ex1 --method arff --seed 0
"$PY" scripts/reproduction_route.py baseline --method arff --seed 0
"$PY" scripts/reproduction_route.py regime --K 128 --N 640000 --h .0001 --method joint_mlp --seed 0
"$PY" scripts/reproduction_route.py final-h --K 1024 --N 640000 --h .002 --method arff --seed 0
"$PY" scripts/reproduction_route.py lag-ridge --h .002 --drift-lambda .008 --seed 0
"$PY" scripts/reproduction_route.py fixed-basis
"$PY" scripts/reproduction_route.py adaptive-ridge --seed 0
"$PY" scripts/reproduction_route.py resampling --seed 0 --arm MR
```

The full estimators include checkpoint selection, final evaluation and native artifact serialization; ARFF includes five nuisance fits, final drift and covariance fit. The lag/ridge adapter changes both nuisance/final drift penalties while retaining each arm's own targets and covariance λ=.001. Frozen plan matching prevents synthesizing unregistered N/h/K jobs. Fitting output is a new realization; never overwrite the published artifacts or use test performance to accept/reject seeds. Run one process per GPU. The inspector does not establish that the unexecuted training path succeeds.

The fixed-basis and adaptive/resampling routes have their original bounded designs; they are not full SDE fits and do not reproduce the production comparison. Their code is restored from the package at original relative paths. Three paired surrogate repeats share one noise/data pool.

## Data generation and its verification boundary

`reproduction_generate.py` defaults to inspection. Add `--execute --output NEW_DIRECTORY` for a later deliberate generation. It redirects output only and invokes the original generator. It does not modify accepted data or relax replay tests.

```sh
"$PY" scripts/reproduction_generate.py original --experiment ex1
"$PY" scripts/reproduction_generate.py ex8-float64
"$PY" scripts/reproduction_generate.py coupled-lags
"$PY" scripts/reproduction_generate.py regime-v1
"$PY" scripts/reproduction_generate.py regime-v2
"$PY" scripts/reproduction_generate.py final-h
"$PY" scripts/reproduction_generate.py surrogate
```

- `original`: applies `generate_dataset.py EX --output-dir ...`, including configured generation seed and canonical split seed 2026. Ex1/2/3/7 use fine EM; Ex5 fixed-lag SSA; Ex6 the original wave stencil. The original Ex8 float32 dataset is a **provenance prerequisite**, never the corrected comparison's training data.
- `ex8-float64`: needs exact original Ex8 and the archived matched-noise GPU fixture. Replays normals, requires original float32 replay equality, integrates float64, validates zero-noise/refinement and preserves split IDs. Output is a new corrected dataset and provenance.
- Ex6 correction: in a fresh full Git checkout containing only generated/restored `data/ex6.npz` and `ex6.json` (no v2 outputs), create `results/final_reproduction/evidence/` and run `python scripts/correct_ex6_labels_v2.py`. It verifies historical source through Git and records every changed label; no trajectory regeneration is needed for correction.
- `coupled-lags`: needs corrected baseline, its validation record and frozen float64-v2 manifest; generates the original coupled small-lag views.
- `regime-v1`: needs those views and frozen v1 manifest; extends the iid pool with registered block keys. `regime-v2` needs v1's h=.0004 parent and archived completion/provenance; `final-h` needs v2's h=.001 parent, previous normal-block hashes and frozen final amendment. The wrapper replays each stage from authenticated archived parents. The old parent-completion records establish ancestry, not completion of a new campaign. A full chained regeneration must arrange these parents under the recorded relative paths in a fresh workspace, retaining the source manifests. Do not rerun preparation/supervisor scripts that could start training.
- `surrogate`: uses `make_common_data()` from the preserved final Phase-2 script and the registered data/noise recipes, with no fits. A/C/B bases are generated by that script's registered worker procedures, not from a fitting seed alone. M/MR streams are stored explicitly; `RNG_SPEC.json` and `study.py:preflight` record their construction. Restore those arrays for exact replay.

Fresh ZIP/metadata bytes may differ because of serialization, paths or Git provenance. Compare array hashes and split IDs first, then the archived numerical replay/refinement criteria. A byte-identical training input is obtained by restoring the bundle, not by asserting any new dataset matches from a seed label. Production wrappers intentionally require archived dataset hashes; a different generated dataset must be diagnosed rather than admitted by editing the gate.

No learned normalization statistics are missing. Historical tuning/search budgets are unavailable: this prevents recovering original tuning/selection history, **not** repeating the chosen compatible configurations. The withdrawn modified-Ex7 NLL-infinity and old trajectory/timing evidence remain unauthenticated; they are not needed to reproduce currently reported results.

## Resources and original bounded end-to-end verification proposal

The following proposal was subsequently executed once; see the [actual outcome](../bounded_reproduction_verification_v1/REPORT.md). Its tolerances were unchanged. Use one otherwise idle RTX A6000 (49 GB) and the locked source/environment. Proposed configuration: **corrected Ex8 baseline ARFF, K128, N80000, seed 0**, with λ=.001, 300 adaptations, five-fold cross-fitting and all accepted settings. Restore and hash-check the original/corrected datasets, registered N80000 view and sources; no new stochastic data are needed. Run:

```sh
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/reproduction_route.py baseline --method arff --seed 0 \
  --execute --output /tmp/ex8-arff-seed0-full-verification
```

Archived seed 0: algorithm 7.238907 s, compilation 64.057015 s, their reported sum 71.295922 s. Budget **2–5 minutes** including loading/evaluation/provenance; this is an estimate, not an isolated timing claim. Reserve **8 GB GPU memory conservatively**; this is not a measured peak. One 72000×256 float32 feature matrix is ~74 MB, but compilation, Gram/RHS, arrays and simultaneous buffers add memory. For K1024/N640000 the feature matrix alone is ~4.7 GB and archived full seven-fit lag/ridge jobs take ~19 minutes; use A6000-class capacity and no same-GPU concurrency. Larger campaigns take hours/days.

Predeclared checks: exact data/source/config hashes, seed/fold/internal split IDs and counts; finite output, 300-entry histories, correct earliest selected minimum and no refit; complete native artifact validation and source/setting identity. Reload the new model and compare its own saved metrics at rtol=1e-5/atol=1e-6 on the same backend. Compare new and archived metrics/histories descriptively under the same tolerance, recording any deviation; do not automatically declare reproduction or rerun until a preferred answer occurs. Floating-point branch differences may change adaptive trajectories. Cross-hardware bitwise equality is not required; one fitting seed cannot establish distributional equivalence across hardware or seeds. The run would verify the complete seven-fit software path, not all methods, datasets, historical choices or 30-seed distributions. A short smoke is not a substitute.

## Backup/hosting and closure

The existing `../backups/SDE_ARFF_2026_final_ex8_2026-09/` is on KW61146; its README explicitly says the Mac/Synology destination was not mounted and transfer remained pending. No independently verified off-server copy or authenticated hosting destination was found. The repository remote contains code/compact evidence, not these large artifacts. Do not equate that Git remote or the same-server backup with a dataset/checkpoint backup.

The concrete remaining decision is an authorized durable destination (institutional archive or repository accepting ~7 GB payload, license/access policy and uploader credentials supplied outside Git), followed by upload and independent download/hash verification. No upload URL is invented. Local manifest/bundle preparation is complete; public independent access and fresh numerical reproduction remain unverified.

### Surrogate saved-checkpoint evaluation

`python scripts/evaluate_surrogate_checkpoint.py --artifact results/arff_B_ridge_intervention_v1/jobs/0/Bpath.npz` checks identity and model structure only. Add `--execute --output /tmp/Bpath-evaluation.json` to score the fixed 5000 test rows with the native predictor and check archived predictions where present. It also accepts M/MR `selected.npz`/`final.npz`, parent A/B/C model NPZs and fixed-basis `scaled_amplitudes.npz`; spectral `analysis_lambda*.npz` files are intermediates, not models. Fixed-basis expected-risk/SVD quantities are rebuilt by the bounded fixed-basis diagnostic, not inferred from a single realization's test error. This new evaluation entry point was syntax/metadata checked only, not numerically exercised here.

For the proposed new ARFF verification artifact, `evaluate_reproduction_checkpoint.py --new-run` enables its native schema-validation gate before reevaluation; it still requires the exact inventoried dataset. Other fresh native runners already evaluate their frozen models before serialization. This option does not designate a new fit as a published checkpoint.

### Custodian packaging and archive verification

The immutable file list was frozen before the tar was written. To recreate the same payload from its listed files:

```sh
python scripts/package_independent_reproduction.py bundle --output /new/path/independent_reproduction_v1.tar
python scripts/package_independent_reproduction.py verify-tar --archive /new/path/independent_reproduction_v1.tar
```

Both commands verify hashes; bundle creation refuses existing output. `verify-tar` streams every unique payload, checks all internal hardlinks and refuses unexpected/missing members without extraction. The release tar's complete payload was verified this way. Do not run `inventory` to verify a release: it deliberately creates a new file list from the current tree, so changes would define a different version. Manifest and tar checksums are in BUNDLE.json. Future versions should use new names.

The original packaging verification record is preserved as historical evidence. Actual subsequent run results are in `../bounded_reproduction_verification_v1/OUTCOME.json`; self-reconstruction passed, archived-reference agreement failed. The off-server backup procedure is in [BACKUP.md](BACKUP.md).
