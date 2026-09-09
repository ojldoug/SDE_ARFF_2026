# Historical/current metric and protocol reconciliation

2026-09-09. Read-only investigation of source, datasets and saved models; outputs are this note and `metric_reconciliation_evidence/`. No optimizer steps, training, timing campaign, dataset changes, canonical artifact changes, or summary replacements were performed. Experiment4 remains unresolved.

## Conclusions and limits

1. A concrete surviving coefficient evaluator was recovered: `GPU/plot_true_trained.ipynb`, cells1–6. It evaluates **all loaded input rows**, compares **diffusion factor sigma**, and takes a single square root after averaging squared errors over all array entries. Current summaries use canonical validation rows and **covariance Sigma=sigma sigma^T**. These are real definition differences.
2. Applying the recovered population convention to all30 current models does **not** repair the large drift discrepancies. Sigma-to-sigma error conversion explains some apparent diffusion-error scale changes, but is nonlinear and cannot be obtained by simply square-rooting covariance RMSE.
3. Experiment3's saved MLP genuinely predicts an almost affine function on the cubic-drift domain. Independent reconstruction reproduces its poor error. Neither an incorrect artifact decoder nor an accidentally shared model file explains it. The precise training cause of the common epoch118 optimum is not identifiable from endpoint weights alone; no retraining was used to manufacture an explanation.
4. The Experiment7 shift is close to log(10), but is neither exactly constant across estimators nor the ordinary Gaussian constant. Recovered likelihood implementations use the complete Gaussian formula. An old data-lag/density-scale difference remains possible, not established.
5. **Provenance qualification:** the recovered plotting notebook loads one saved model per method, and its scalar reader prints single-model RMSEs. The surviving repeated-training notebook currently specifies10 runs and discards model parameters. No originating30-run coefficient-table driver, historical model files, raw table archives, or exact table-producing revision was recovered. Therefore “recovered historical evaluator” below means demonstrated surviving code, **not proof that this exact execution produced Owen's Table3**. This corrects the overly broad implication in earlier notes that matching the GPU loop establishes the exact historical manuscript path.

## Exact definitions recovered

Let P contain n evaluation points and d output dimensions. The notebook computes

\[
R_f(P)=\sqrt{\frac1{nd}\sum_{i\in P}\sum_{a=1}^{d}(\hat f_a(x_i)-f_a(x_i))^2},\qquad
R_\sigma(P)=\sqrt{\frac1{nd^2}\sum_{i\in P}\sum_{a,b=1}^{d}(\hat\sigma_{ab}(x_i)-\sigma_{ab}(x_i))^2}.
\]

Current covariance error is instead

\[
R_\Sigma(V)=\sqrt{\frac1{|V|d^2}\sum_{i\in V}\|\hat\Sigma(x_i)-\sigma(x_i)\sigma(x_i)^T\|_F^2}.
\]

There is no relative-error denominator, factor of h, hidden sample-count normalization, or state standardization in these coefficient formulas. Dimensions are **averaged**, not summed. For diagonal matrices the d² denominator includes zero off-diagonal entries: reporting diagonal-vector RMSE instead would multiply matrix RMSE by sqrt(d). Extracting triangular entries would likewise change the denominator/weighting. The historical notebook extracts diagonals/lower triangles only **after** computing RMSE (cell6). Square roots occur once per model; a mean across model RMSEs is not a square root of pooled across-run MSE. The current summary averages per-run RMSEs. The originating historical30-run reduction cannot be verified from the surviving single-model reader.

Population and preprocessing:

- Notebook cell1 loads `training_data/<ex>.npz` x_data in full. If trajectories have3 dimensions, it selects every1000th x state, sums1000 micro-increments, and sums the associated lags. Its retained ex2 output explicitly shows (10000,3,1000) becoming (10000,3), h shape(10000,1).
- Optional random subsetting is commented out in cell2. There is no grid or new random evaluation population in this evaluator.
- `Adam_training.ipynb` cell5 also has a different, **plot-only** random10000-point subset. Do not confuse that visualization with the scalar RMSE notebook.
- Source truth functions in `GPU/true_functions/true_functions.ipynb` agree with current ex3 (cell7), ex6 (cell6), and ex7 (cell8). Ex6 already uses g/2, explicitly marked a corrective term. No extra factor-two change in truth is justified.
- No fitted state normalization/rescaling is applied by the GPU training/evaluation path. Older TensorFlow report utilities support a `data_transform_network`; that is a different source path, not evidence that a transform was applied to these manuscript-table runs.

Historical factor reconstruction matters. `GPU/lib/lib_ARFF.py:45–84` uses abs(raw diagonal covariance)+EPS followed by square root for diagonal sigma; Cholesky for triangular sigma; and principal symmetric square root with negative eigenvalues clipped to zero for symmetric sigma. Thus historical factor error can incorporate absolute-value/PSD corrections; it is not raw covariance error. Current raw Sigma RMSE performs none of those corrections. Current NLL uses the existing SPD projection. In the five investigated experiments, archived validation raw covariances were SPD, so the reported large drift gaps cannot be attributed to SPD projection.

For the symmetric ARFF legacy factor call, current JAX rejects the old spelling `jnp.clip(..., a_min=0)`. The evaluation-only calculation used the same mathematical eigenvalue clipping/square-root formula via NumPy, without modifying the historical module or any runner. This is an API compatibility issue in forensic evaluation, not a production-training failure.

## NLL formula, normalization and checkpoints

Surviving GPU Adam (`lib_Adam_tanh.py:86–104`, `lib_Adam_FF.py:81–99`) uses

\[
L=\frac1n\sum_i\left[\frac12(r_i-h_i\hat f_i)^T(h_i\hat\Sigma_i)^{-1}(r_i-h_i\hat f_i)
+\frac12\log\det(h_i\hat\Sigma_i)+\frac d2\log(2\pi)\right].
\]

There is **no division by d**. Quadratic/logdet contributions sum dimensions, then the loss averages samples. Cholesky logdet is2 sum(log(diagonal(scale))), with scale containing sqrt(h). The current Adam compatibility runner calls this same function, without rewriting the loss. The diagonal model emits softplus(raw)+1e-13 as covariance. For ex3 the reconstructed sigma=LL^T is squared in likelihood: Sigma=(LL^T)^2, not LL^T. Current coefficient evaluation uses the same square.

Legacy ARFF `lib_ARFF.py:87–112` uses the same Gaussian constants/h/sample averaging but an unsafe full-matrix branch: inverse uses S+1e-10 I, logdet uses slogdet(S) and discards its sign. Its diagonal covariance uses abs(raw)+EPS. Corrected ARFF instead projects Sigma before multiplying by h, solves the SPD system and requires positive determinant. These differences can change NLL; they are not omitted Gaussian constants. Older TensorFlow `sde/sde_learning_network.py` uses sample-mean negative standard Gaussian log_prob, also with the Gaussian constants.

`GPU/saved_results/loss_time_data/loss_stats.ipynb` cell2 computes mean(min(validation_loss, axis=epochs)) for Adam. Cell3 averages ARFF endpoint validation losses. The RMSE reader `GPU/saved_results/true_trained_RMSE/read_RMSE_data.ipynb` merely reads/prints two saved scalars. It does not associate an RMSE with the NLL-minimizing epoch.

The single-fit Adam notebook cell4 returns final-epoch parameters, and cell6's commented save would save those. The repeated loop cell7 discards parameters. Current artifacts retain the earliest minimum-validation epoch. Final historical checkpoint weights cannot be recovered by reevaluating a selected-only current archive: full epoch weights were not saved. This is a concrete limitation, not permission to refit.

## Experiment3: source comparison and saved-model diagnosis

The following comparison is exact against the surviving GPU path; the exact manuscript-producing path remains unlocated.

| Stage/source | Surviving GPU path | Current compatibility path | Difference |
|---|---|---|---|
| Truth/domain | true_functions notebook cell7: f=-16x³+8x-1.5; constant10×10 symmetric sigma; [-1,1]^10 | definitions.py:91–118, same functions/domain | No detected truth-formula mismatch |
| Data load | Adam notebook cell1, optional1000-step aggregation | run_final_historical_adam.py:14–20 loads canonical aggregated triples | Original raw files/hash absent, so historical dataset identity unproved |
| Hyperparameters | Notebook cell3 currently300epochs,1e-4,batch256,width1024, ex_name ex1 | Primary ex3 table values300,1e-4,batch1024,width1024 | Notebook is mutable working state, not exact table-run configuration |
| Initialization | Cell7: PRNGKey(seed), split3; init_drift_params/init_diffusion_params | Runner:41–43, same split3 and functions | Single-fit cell4 instead split2; no claim they are identical |
| Drift/factor forward | lib_Adam_tanh.py:20–64 | Direct import of those functions | Same tanh/linear maps; L softplus diagonal; sigma=LL^T |
| Loss | lib_Adam_tanh.py:86–104 | Same function; observer returns unchanged loss | Same squared covariance and complete Gaussian NLL |
| Optimizer | set_opt/train_step:81–116, Adam(.9,.999,eps1e-7) | Direct call of same implementation | No numerical update rewrite |
| Outer split | Cell7 permutes all rows using retained run key before90/10 | Fixed canonical permutation,90/10 | Accepted deterministic-common-split correction, not original per-run split |
| Epoch shuffle | training_loop:138, PRNGKey(epoch) | Same training_loop | Same training-row order across runs given fixed split |
| Warm-up | Notebook no separate full warm-up path | Discarded batch updates before production loop | Original parameter arrays/optimizer states used afresh; no production update consumed |
| Checkpoint | Last parameters returned; min NLL summarized separately | Runner:56–73 retains earliest strict validation minimum | Accepted checkpoint/metric consistency correction; no post-selection refit |
| Coefficient metric | Plot notebook cells4–5: all x, sigma RMSE | Runner:22–31: validation x, Sigma RMSE | Population and coefficient target differ |

All30 saved MLP parameter hashes are distinct. All30 stored best_epoch fields equal the independently computed argmin of their300-element validation histories:118 (zero-based). Independent NumPy forward reconstruction matches saved drift RMSE within1.44e-6. Thus neither a hard-coded index nor duplicate parameter files explains the result.

For seed0, hidden preactivations lie in[-.459105,.507815], with no |activation input|>5 saturation. The fitted drift RMS is1.6083 versus true drift RMS2.9869; its per-coordinate standard deviations are roughly.70–.74 versus true2.56–2.60. The network differs from its affine Taylor approximation at x=0 by RMS only.01122 (range across seeds .01091–.01127). Weights changed from initialization; this is not an untrained model or a dead saturated network.

For a uniform scalar X in[-1,1], the exact best affine L² projection of f(X)=-16X³+8X-1.5 is -1.6X-1.5. Its irreducible affine error is

\[
\sqrt{256(E[X^6]-E[X^4]^2/E[X^2])}=\sqrt{1024/175}=2.41897\ldots
\]

On the actual validation rows that analytic affine function has RMSE2.41882, close to the saved neural RMSE2.42778. This strongly supports **near-affine underfitting of the learned model** as the immediate reason for poor drift recovery. Common data/order and nearly identical fitted functions are consistent with a common validation minimum. They do not prove which optimizer/initialization/covariance-scale mechanism prevents nonlinear learning, or uniquely explain why epoch118 rather than another epoch wins; epoch weights/gradients were not archived. No covariance parametrization change or tuning was attempted.

Important historical-path discrepancy: older TensorFlow `reference/sde-identification/sde/sde_learning_network.py:196–205` puts abs(raw) on L's diagonal, whereas the GPU code uses softplus(raw)+EPS. With small uniform initial raw values, their initial diagonal scales are about.005 versus.693. Both may use Sigma=(LL^T)^2, but **the squared-covariance identity alone does not make training identical**. TF diagonal models (230–233) also emit sigma through softplus, whereas GPU diagonal models emit covariance through softplus. These are estimator/initialization changes, not metric changes. However the available TF nd-cubic notebook describes a different experiment ([-2,2],different cubic drift,diagonal state-dependent noise,ELU/Adamax); it cannot be substituted as the exact2026 ex3 procedure. The current compatibility implementation matches the GPU source, not every historical TF implementation. Exact table provenance is a remaining scientific blocker.

## Evaluation-only recalculations

For each current run, the all-row drift error can be reconstructed without any predictions or fitting:

\[
R_f(T\cup V)=\sqrt{\frac{n_T R_f(T)^2+n_V R_f(V)^2}{n_T+n_V}}.
\]

This is exact up to rounding of saved scalar RMSEs because T,V partition the same canonical rows. It changes only evaluation population; selected models and truth definitions remain fixed. Below are means over all30 current models, in order ARFF/Fourier/shallow/deep where present.

| Experiment | Historical drift means | Current all-row drift means | Reconciled by population alone? |
|---|---|---|---|
|1|.0487/.0498/.0519/.0484|.351331/.518350/.721338/.578988|No|
|2|.0072/.0089/.0068/.0065|.035319/.271322/.023451/.027144|No|
|3|.0413/—/.0376/—|1.528782/—/2.428242/—|No|
|6|.0098/.0101/.0104/.0097|.282984/3.267724/1.316578/3.003273|No|
|7|.0287/.0305/.0297/.0283|.106238/.158956/.195761/.100675|No|

The recovered sigma evaluator was also applied to **every current point for seed0** of each method. These are individual-run diagnostics, not30-run historical-mean reproductions:

| Experiment | ARFF sigma RMSE | Fourier sigma RMSE | Shallow sigma RMSE | Deep sigma RMSE |
|---|---:|---:|---:|---:|
|1|.0263597|.0322759|.0150846|.0278334|
|2|.00224685|.00244396|.00111941|.00060954|
|3|.0190446|—|.0337941|—|
|6|.00027056|.00137151|.00044818|.00035422|
|7|.00054849|.00070954|.00038991|.00043815|

Ex6 ARFF sigma error .00027056 is close in scale to historical Diff .0003, unlike current covariance error ~.000019. Ex3 ARFF sigma error .01904 is similarly much closer to historical .0215 than its covariance error. Neither observation proves an exact historical30-run match.

For **Experiment2 Fourier all30 seeds**, direct manual reconstruction Phi=[cos(x omega),sin(x omega)], f=Phi amp, on all10000 points gives mean drift RMSE.27132199, agreeing with the partition reconstruction. There is no missing sqrt(K),sqrt(2/K), complex-to-real scaling, intercept, or frequency transpose in the surviving forward definition (`lib_Adam_FF.py:20–31`). All30 historical-form sigma errors average **.00246041 ± .00015557**, versus historical Diff .0022. Thus factor semantics account for much of the apparent diffusion scale difference; the drift .0089→.2713 gap remains and is not caused by artifact reconstruction or validation-only evaluation.

## Experiment7: analytic likelihood-offset tests

Historical-to-current mean NLL shifts (current minus historical): ARFF -2.31192; Fourier -2.33232; shallow -2.32395; deep -2.31281. They are not exactly constant, even allowing the manuscript's three-decimal rounding.

At d=2,h=.001,sigma=.1I, the terms are:

- Gaussian constant d/2 log(2pi)=**1.837877**, not2.31.
- Half logdet(h Sigma)=log(.001×.01)=**-11.512925**.
- On the saved ex7 seed0 validation population the oracle mean half-quadratic is**.988217**, yielding oracle NLL**-8.686831**, consistent with the reproduced model NLL scale.
- Omitting h from the logdet alone shifts NLL by**+6.907755**, not+2.31.
- Changing from increment density r to r/sqrt(h) changes NLL by **-d/2 log(h)=+6.907755**; r/h gives **-d log(h)=+13.815511**, not2.31.
- Dividing the entire likelihood by d halves its value (~-4.343); it is not the observed additive shift.
- A factor10 change solely in the determinant contributes d/2 log(10)=**2.302585**. Adding it to current means leaves residual differences from historical means of about-.00934,-.02973,-.02136,-.01022. This is not an exact repair.
- A consistent change in h on the same observations also changes drift residuals and the quadratic term. It cannot be justified by adding log(10) alone. Different historical datasets generated at different h could have a roughly log(10) entropy shift, but the manuscript and current truth/data specifyh=.001 and no original runtime h-array has been recovered.

Both recovered GPU and TF Gaussian routines include the constants and normal sample averaging. No evidenced omitted-term code change explains the gap. No stored NLL was shifted. Required evidence: original ex7 r/h arrays, exact loss source revision and raw per-run validation histories.

## Experiment6: isolate label effects from metric effects

`GPU/true_functions/true_functions.ipynb` cell6 already uses f(s)=5sin(4pi s), sigma(s)=.025[1+exp(-150(s-.5)^2)] = g/2. Current definitions agree. The accepted label correction does not change this truth formula.

For the same saved ARFF seed0, evaluated over all995004 rows:

- Original coordinates: drift RMSE **.24544343**.
- Corrected coordinates: drift RMSE **.24564701**.
- RMS change in the truth caused solely by the coordinate-label change: **.03135257**.

Thus reverting the evaluation coordinates on this fixed model does not approach the historical .0098. The498 changed h labels do not enter the coefficient-error formula at all, although they affect training targets and NLL. The497502 coordinate corrections affect both inputs and truth; the direct evaluation difference above is small for this model. The counterfactual effect of **training** on the original erroneous labels cannot be inferred from corrected-data endpoints. It would require old fitted weights or a separate training comparison, which was not run or authorized here. Factor-vs-covariance error explains part of the Diff scale discrepancy independently of the label repair.

## Per-experiment decision matrix

| Experiment | Historical definition recovered | Current definition | Exact mathematical difference | Evidence/source | Historical number recoverable by evaluation only? | Rerun required? |
|---|---|---|---|---|---|---|
|1|Whole loaded x; absolute R_f and R_sigma; saved single model, epoch provenance missing|Validation R_f and raw R_Sigma; selected checkpoints|P=all→V; sigma→sigma sigma^T; possible last→best model|Plot notebook1–6; Adam notebook4–8; forward functions; all-row recalculation|No drift reconciliation; seed0 sigma also does not match historical means|No rerun for recovered metrics. Exact historical model/protocol provenance needed before deciding training rerun|
|2|Same evaluator, retained output confirms10000 aggregated3D points|1000 validation rows of10000; raw covariance|Same P/target/checkpoint differences; features are identical real cos/sin form|Plot notebookcell1 output; FF20–31;30-model independent forward evaluation|No drift reconciliation (.271322 vs.0089); sigma .002460±.000156 is closer to Diff .0022, not proven exact match|No evaluation rerun needed; training-path discrepancy remains unresolved|
|3|Same whole-data sigma evaluator; source square-root reconstruction; table checkpoints absent|10000 validation rows; squared-factor MLP and raw covariance error|P/target/checkpoint differences; no dimension-normalization change found; old TF differs in factor diagonal transform|Ex3 line-by-line table above;30 distinct reconstructed models; analytic affine bound|No. Genuine near-affine endpoint behavior remains; historical .0376 cannot be recovered by moving to all rows|No automatic retraining. Exact table-producing implementation and checkpoint evidence required|
|6|Whole staggered-label dataset; sigma=g/2; no h in coefficient RMSE|Validation on corrected labels_v2; same truth; covariance error|P/target differences plus accepted coordinate-label correction; h only affects fitting/NLL|Truth notebookcell6; source wave generator; fixed-model original/corrected evaluation|Diff scale partly reconciled; drift not reconciled; no historical30-model checkpoint set|No label rollback or new training. Cannot infer original-label training counterfactual from current models|
|7|Whole data sigma evaluator; full Gaussian NLL source; table data runtimeh missing|Validation raw Sigma and full NLL,h=.001|P/target differences; tested constants do not yield exact observed shift|Truth notebookcell8; likelihood source; oracle decomposition and analytic tests|No drift or exactNLL reconciliation; do not addlog10 to stored results|No automatic rerun. Recover original h/data/loss provenance first|

## Historical split MLP, Section6.2, and source search

The requested parallel source investigation examined plotting/scalar-reader/loss-stat notebooks, true-function source, GPU and olderTF training paths, and available repository history; GPU Adam source has only its initial imported revision eaaecb4 in available history. Earlier searches covered523 historical source blobs. No saved NN_param/scalar-RMSE archives, exact Table3 repeated-coefficient driver, historical split-MLP trainer, exact N/h/K grid implementation or NLL_infinity regression was recovered. Older unrelated nd-cubic experiments must not be substituted based on filenames. Existing search evidence remains preserved. Experiment4 remains unresolved. These missing scientific definitions do not authorize new settings, tuning or replacement studies.

## Reproducibility of this reconciliation

Evidence JSONs contain all30 all-row drift recalculations, all30 independent ex3 reconstructions/weight hashes, all30 ex2 Fourier historical-form evaluations, seed0 full-population sigma errors for every affected method, ex7 likelihood decomposition and ex6 fixed-model label comparison. Evaluation used CPU only and existing functions; no optimizer/training loop was executed. Model parameters and canonical summary files were read, never rewritten. The isolated-timing campaign remains paused.
