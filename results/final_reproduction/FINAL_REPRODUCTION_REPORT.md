# Full-paper reproduction — production in progress

The accepted decisions are implemented: Experiment6 uses `data/ex6_labels_v2.npz`; Experiments1–7 use historical90/10 train/validation with no separate test; Experiment3 Adam uses the historical Sigma=(LLᵀ)² model through unchanged legacy numerical functions. Accepted Experiment8 datasets,120 production artifacts and poster figures remain frozen. No manuscript file has been edited.

## Production and validation

The historical Adam campaign contains480 runs:30 seeds for Fourier, shallow tanh and deep tanh on experiments1,2,5,6,7, plus30 shallow-tanh runs for experiment3. Session: `final_historical_adam_accuracy`. Outputs: `production/accuracy/<experiment>/<method>/seed_<seed>_artifacts.npz`, one preserved log and supervisor reservation per run. The launch verified all four GPUs had0% utilization,1MiB allocated and no compute processes. Four exclusive workers check occupancy before each dispatch. The numerical source/data/manifest hashes are frozen. Completed outputs are validated and never overwritten; failure stops the affected worker.

The corrected historical ARFF campaign contains180 runs (experiments1,2,3,5,6,7,30 seeds each), queued in persistent tmux for the same four GPUs after Adam finishes. Session: `final_historical_arff_accuracy`. See `HISTORICAL_ARFF_COMPATIBILITY.md` for exact historical control flow and every correction. Experiment8's fitter and numerical runners are unchanged.

Static syntax and diff checks passed. Three tiny CPU Adam fixtures reproduce the unchanged historical validation histories exactly, including historical squared covariance. Mocked supervisor tests verify valid-output reuse and failed-output preservation without launching numerical processes. ARFF tests reproduce historical control-flow selected models, keys and histories under the same corrected primitives; a complete tiny seven-fit artifact passes OOF, history, split and serialization checks. Production artifact validation is performed separately after each job; these smoke checks are not claimed to establish final repeated-run results.

Immutable dispatch specifications: `production_manifest.adam_v1.json`, `production_manifest.arff_v1.json`, and each campaign control directory's `launch_manifest.json`. `production_manifest.json` is the current audit index; earlier versions and discovery-stage notes remain archived. Consult `adam_campaign_control/state.json`, `arff_campaign_control/state.json` and per-job completion records for live status; planned jobs are not completed results.

## Accepted Experiment6 correction

Full technical evidence is in `EX6_LABEL_CORRECTION_NOTE.md`, `evidence/ex6_v2_validation.json`, and `data/ex6_labels_v2.json`/`data/ex6_labels_v2_changes.npz`.

- Original `data/ex6.npz`: SHA256 `0966236d8a2a6aecf0eca85d9dc568145d46ef499eb5755e1bf20dc92c6304f7`.
- Corrected `data/ex6_labels_v2.npz`: SHA256 `47c4a91d37f494621c4174e5c8a87d951a9b78b16b1da68dc4b146f72dbc85c3`.
-498 initialization rows: lag becomes Δ²/4 instead of Δ²/2.497502 alternating-block rows: spatial label corrected to the actual forcing coordinate. Increments, trajectories and stored split arrays are unchanged. Source replay reproduces the original trajectory-derived observations bitwise. Corrected deterministic identity maximum residual is about2.01e-11, versus2.50 before correction.

`historical dataset -> confirmed labeling defect -> blocker-required label correction -> corrected canonical dataset`.

This correction is accepted and no longer blocks other experiments. No dataset was overwritten or regenerated for performance.

## Historical settings and correction boundaries

The primary manuscript Appendix tables supply each K,lambda,delta,Mmin/Mmax,resampling/Metropolis choice, Adam epochs/lr/batch and shallow/deep architecture. The matrix and manifests record these settings rather than newer blanket defaults. No tuning was performed.

The modern80/10/10 split for1–7 was a scaffold choice, not an SPD/OOF blocker requirement. The user explicitly restored90/10. The final fitting view uses the already accepted deterministic canonical permutation, reserving its final floor(.1N) rows for validation; actual indices are archived. It has no independent test score. This retains deterministic common splits as an accepted reproducibility correction, rather than claiming bitwise equivalence with historically reshuffled notebook splits.

Historical symmetric Adam returns LLᵀ as the diffusion factor, yielding Sigma=(LLᵀ)². The modern L-factor model is a different scientific choice. Experiment3 now uses the unmodified historical initializer, objective, factor reconstruction and optimizer loop. The only checkpoint bookkeeping correction retains the earliest minimum-validation epoch so that saved weights and reported minimum loss describe the same model, with no refit. Historical unused tanh amp leaves are archived but excluded from active parameter counts.

Historical ARFF uses resampling before mutation, key-based internal splitting, zero-based stopping and stopping-before-retention. The separate compatibility path preserves these; it reuses accepted corrected multi-output norms, ridge precision, five-fold covariance targets, raw diagnostics and SPD likelihood evaluation. Full histories include the previously omitted terminal observation. Canonical outer validation does not influence ARFF's training-only internal regression selection.

## Independently unresolved components

Experiment4: primary Appendix Table5 gives -x³-x+0.5v, while surviving code gives -x³-x-0.5v. Main-text symplectic conditioning uses(v0,x1); the current fine-step dataset is labeled(v0,x0). No arbitrary sign or conditioning choice has been made, and experiment4 is excluded from dispatch.

Historical split MLP: primary §6.1 and Tables2/3 establish Experiment8 joint/split shallow/deep comparisons and30 repetitions. The reported split validation NLL means are-8.263(shallow),-8.931(deep), with reported drift errors.8544 for both and diffusion/covariance-column errors.2494,.0701. These are manuscript transcriptions, not newly validated distributions. Table2 K512 versus Appendix K1024 and the absent originating split trainer prevent an exact matched-capacity reproduction. Objectives, initialization, per-stage budgets, covariance representation, cross-fitting status and seeds are not recovered. `HISTORICAL_SPLIT_MLP_AUDIT.md` gives evidence and parameter formulas. No modern split implementation is being relabeled as Owen's historical result, and no new architecture is tuned.

Section6.2: modified ex7 uses sigma=diag(.25xi²+.25), whose covariance is its square. K512 and10 repetitions are recovered for N/h. Excess NLL is learned-minus-oracle on the same evaluation set for N/h; for K it is NLL(K)-NLL∞. Exact grids, fixed data settings and the NLL∞ regression family/fitting domain/weights remain unrecovered. No grid is inferred from image pixels or invented. N/h can proceed independently once fully specified; they are not held solely for the K regression.

Search evidence includes229 original worktree files,523 historical source blobs, a follow-up261-file search including archives/rebuttals/appendices, and an expanded project filename inventory. The surviving polyfit hit is trajectory interpolation, not NLL∞ estimation. See `evidence/history_error_search.json`, `worktree_error_search.json`, `followup_historical_search.json` and `expanded_source_candidates.json`. The historical trajectory-histogram initial law/horizon/step/count/binning/seed protocol is also not fully recovered.

## Timing and remaining outputs

Four-GPU accuracy timings are explicitly non-isolated. Adam's historical loop retains its epoch shuffle/gather/printing timing semantics; the discarded batch warm-up does not prove all first-epoch compilation overhead is removed. These times must not be promoted to controlled runtime claims. ARFF warms the complete seven-fit path. End-to-end fields denote warm-up plus learning where recorded; they are not whole-process launch-to-serialization times.

After resolved accuracy runs finish, validate full seed coverage, produce per-seed distributions, mean±sample SD, medians/ranges, parameter/SPD tables and figures. Then assess cost of an otherwise-idle, single-job runtime protocol with complete-path discarded warm-up. Use all historical seeds if projected within the authorized12-hour budget, otherwise document a predetermined performance-independent subset before dispatch. No isolated timing run has started. Manuscript replacement proposals follow validated results; the editable TeX stays unchanged pending review.

## Persistent completion watcher

`final_reproduction_summary` waits for both campaigns to finish successfully, revalidates660 new artifacts and120 accepted Experiment8 artifacts, then writes a new `resolved_accuracy_v1/` bundle: per-seed CSV/NPZ, mean/sample-SD/median/range JSON, tables, RMSE/NLL distribution figures and separate objective-history plots. It refuses incomplete campaigns and stops on recorded worker failure. Its runtime cost plan is explicitly a proposal, not evidence of completed isolated timing. Source/command provenance is in `summary_control/launch.json`. No final aggregate is claimed before its `complete.json` exists.
