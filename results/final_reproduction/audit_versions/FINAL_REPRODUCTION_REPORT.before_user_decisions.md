# Full-paper reproduction — audit checkpoint, not completion

## Status

No new numerical production, isolated timing, hyperparameter search, canonical-data generation, or manuscript editing was performed. The user requested a stop/report if source/data evidence reveals a scientifically invalid previously accepted core experiment. The existing canonical Experiment 6 labels demonstrably fail the observation identity of their own generator, so this checkpoint records that hard stop before dispatch. It does not declare the other experiments invalid and does not imply the entire existing pipeline should be replaced.

The immutable `production_manifest.json` is explicitly an audit manifest, not a runnable dispatch manifest. A subsequent resolved dispatch manifest must be a new file/version, preserving this evidence. The matrix distinguishes existing implementation, existing canonical dataset, and trained-artifact reuse. Existing generated data and results have not been overwritten.

## Confirmed Experiment 6 defect

Source: `src/experiments/wave_data.py`, also present in `GPU/generate_SPDE_data.ipynb` and the reference repository `sde/spde_utils.py`.

The canonical dataset contains 1,998 time blocks of 498 interior points each (995,004 rows), all labeled with h=Δ²/2, Δ=.001, and spatial coordinates from `space[::2]`.

1. The first block is formed from the initialization update whose deterministic term is Δ² f/2 in u, hence Δ² f/4 in the learning increment. Its effective learning lag is Δ²/4, not Δ²/2. Its 498 rows are mislabeled by a factor of two.
2. For even integration j, the generator evaluates forcing at odd-grid coordinates but labels the resulting learning rows with even-grid coordinates. 999 blocks, or 497,502 rows, have a spatial label offset of Δ.
3. Actual RNG draws are independent across array entries and successive calls: initialization W0 has variance Δ²; later Wj has variance 2Δ². The learning noise coefficient is g/4. Thus effective sigma=g/2 is consistent with the corrected lags; the current truth implementation already uses g/2. Replacing it with g would introduce another error.
4. Time labels require documentation of the buffer offset; current forcing is autonomous, so this does not change its coefficient truth. Do not generalize the existing labels to non-autonomous forcing without another audit.

Deterministic proof: `scripts/check_ex6_staggered_labels.py` disables realized noise, uses f(x)=1+x, and compares the actual reformulated increments with the declared labels. The first block gives r/h=f/2; alternating later blocks give r/h=f(x+Δ). Correcting only the candidate spatial/lag labels restores the identity to floating-point roundoff. The observed noise-draw standard deviations are also asserted. Output: `evidence/ex6_label_check.json`.

Minimal candidate correction, not executed: preserve all existing trajectory-derived increments, assign h=Δ²/4 to the first 498 rows, and shift the spatial labels by +Δ in even-j blocks. Publish corrected x/h arrays in a NEW dataset artifact with original/corrected hashes and lineage; preserve the existing canonical file. This is a data-label repair, not a proposal to retune the wave model or regenerate random trajectories. Validate against the actual source revision recorded in ex6.json before publishing a corrected dataset.

Required record:

`historical staggered update + uniform h/even-grid labels -> Raul wave increment/observation-convention blocker -> explicitly correct initialization lag and alternate spatial labels, with independently verified noise law and preserved observation increments`.

## Historical settings recovered

Primary manuscript Appendix B Tables 10–11 agree with the current editable manuscript's actual sibling appendices. Exact values are transcribed into the audit matrix and manifest for Experiments 1–7. They differ from current blanket defaults and from older trained artifacts. All use 30 runs where specified by the primary tables. The manuscript's historical baseline family is ARFF, Fourier Adam, shallow tanh K, and deep tanh K/2 × K/2, except Experiment 3 which lists only ARFF and shallow tanh. The accepted Experiment 8 four-method exception supersedes its historical widths.

Do not simply replace the modern runner's configuration and claim exact historical estimator equivalence:

- Historical symmetric Adam uses (LLᵀ)^2 as covariance; modern Adam uses LLᵀ. Both ensure SPD. This is not automatically a blocker-required change.
- Historical MLP covariance initialization differs from modern initialization; shallow/deep widths also differ from the modern automatic parameter-matching baseline.
- Historical Adam returns the last model while tables summarize minimum validation loss; modern code retains the earliest minimum-validation checkpoint.
- Historical ARFF uses zero-based stopping checks before candidate retention; the frozen modern validation-selected fitter has a different boundary/candidate order when M_min<M_max. The accepted ex8 M_min=M_max case avoided this boundary issue.

These differences require explicit provenance/correction classification before reuse. No accepted numerical runner was changed during the audit.

## Canonical data reuse

All eight existing NPZ hashes match their JSON metadata. Array shapes, finite values, positive lags, disjoint exhaustive splits and deterministic split reconstruction passed. Metadata specifies the existing data seed and source commit. No dataset was regenerated.

- Experiments 1–3 and 7: saved sample counts and retained lags match the primary tables, with 1,000 fine EM steps recorded in generation metadata and the existing source implementation. No regeneration is indicated by this audit.
- Experiment 4: lag/count match, but primary Appendix Table5 has +.5v while surviving notebook/current source use -.5v. The manuscript conditions the velocity likelihood on x1 while fine-step generator stores x0. These discrepancies are recorded, not resolved by choosing whichever trains better.
- Experiment 5: the current generator implements the accepted fixed-lag correction, N=1024, count hazards (4IS/N,I,0), h=.01,T=4,250 trajectories, and retains the transition into extinction. Canonical N=99719 differs from manuscript Table1's99750 and Table8's approximate22664. Do not trim or pad data to force a stale count. Exact SSA mechanics, integer-state lattice, and source provenance remain the relevant correctness criteria.
- Experiment 6: checksum-valid does not imply scientifically valid labels; see confirmed defect above.
- Experiment 8: canonical data and all accepted artifact source snapshots revalidated; no change indicated.

Canonical 80/10/10 seed2026 splitting differs from historical 90/10 and per-run notebook reshuffling. The user's existing reproducibility infrastructure is preserved byte-for-byte; this discrepancy must be described as an explicit correction rather than claimed as historical identity.

## Missing Section 6.2 evidence

Recovered from the primary PDF and active manuscript: modified Experiment7 uses the *diffusion factor* diag(.25 x_i²+.25), not covariance of that value; covariance must be squared. N/h studies use K512; all studies specify10 runs. N/h excess is learned minus oracle NLL on the same test set. K excess is NLL(K)-NLL_infinity.

Not recovered: exact N,h,K grids; fixed N/h settings; study-specific optimizer settings/seeds; independent-test generation; NLL_infinity regression family, fitting range, weights, fitting per method/run versus pooled, and uncertainty handling.

Search evidence: 229 working-tree source files including ignored/hidden files, 377 reference-repository historical source blobs, and146 current-repository historical source blobs. No matching grid/asymptotic-fit implementation or error-named numeric archive was located. Historical images exist but do not establish exact grids or the fitted asymptote. Both source repositories and the actual current editable manuscript tree were searched. This is a missing-evidence finding, not permission to invent a grid or fit.

Figure6 trajectory initial law, number of trajectories, horizon, EM step, histogram bins/seeds, and exact SPD treatment were also not recovered. Existing general simulation utilities can be reused once a protocol is evidenced.

## Existing artifacts and outputs

`evidence/existing_production_inventory.json` inventories748 existing production artifacts with hashes and available configuration fields. It separates completed status from publication validity. Current historical-suite runs do not establish exact agreement with the recovered Table10/11 settings and baseline architectures.

All120 accepted Experiment8 artifacts were revalidated through the existing accepted validators. `ex8_accepted/` contains byte-identical copies of the accepted capacity-matched CSV, distribution NPZ, summary JSON/text and figures. `reuse_validation.json` records240 artifact/log source hashes and copy hashes. The original width57 results and all accepted K128/width27 results remain untouched. No run was filtered using test results.

Remaining final outputs: experiments1–7 repeated-run tables/figures, additional validation/SPD bundles, Section6.2 studies, trajectory comparison, isolated timing, and manuscript_update_plan.md are NOT complete. They are not populated with stale numbers or placeholder claims of success. Isolated timing is deferred until resolved accuracy production as instructed; parallel accuracy times remain non-isolated.

## Validation performed

- Existing canonical loader, checksum verification and deterministic split reconstruction:8/8 passed structural checks.
- Deterministic wave label identity check:confirmed current labels fail; candidate label-only correction passes the fixture to roundoff. No canonical file mutation.
- Existing accepted Experiment8 collectors/validators:120/120 passed.
- Audit source hash verification:passed.
- Syntax compilation of the two new audit scripts and `git diff --check`:passed.

No manuscript or arXiv content was edited. Source attachments and PDFs remain unchanged.
