# Experiment 6: audited label-only correction, version 2

**Original:** `data/ex6.npz`, SHA-256 `0966236d8a2a6aecf0eca85d9dc568145d46ef499eb5755e1bf20dc92c6304f7`.

**Corrected:** `data/ex6_labels_v2.npz`, SHA-256 `47c4a91d37f494621c4174e5c8a87d951a9b78b16b1da68dc4b146f72dbc85c3`.

The generator in metadata commit `2288344cb4f36ceb35216621941597982eef36c1` is byte-identical to the current `src/experiments/wave_data.py` (source SHA-256 `f65412a9fd35444aa747a5e38a7ce4b5fba742c783a96913b51a57f5950e917f`). Replaying that source with its original seed reproduces the canonical `x_data` and `r_data` bit-for-bit before correction. This was verification replay, not replacement of the generated observations.

## Identity and defect

Let Δ=.001, b=0,…,1997 index flattened blocks, j=b+1 the integration-buffer index, and ℓ=1,…,498 the interior packed spatial index. Each block has498 rows. The retained increment is

\[
r_{j,\ell}=\tfrac12(u_{j+1,\ell}+u_{j-1,\ell})
-\tfrac12(u_{j,\ell-1}+u_{j,\ell+1}).
\]

Cancellation of the stored generator's stencil gives

\[
r_{j,\ell}=h_j f(s_{j,\ell})+\tfrac14g(s_{j,\ell})W_{j,\ell},\qquad
h_1=\Delta^2/4,\quad h_{j\ge2}=\Delta^2/2.
\]

The independent normal draws have Var(W₁)=Δ² and Var(Wⱼ)=2Δ² for j≥2. Consequently Var(r|s)=hⱼ[g(s)/2]²; the effective diffusion factor remains g/2. Here f(s)=5sin(4πs), g(s)=.05[1+exp(-150(s-.5)²)].

**Lag:** the initialization generator has a Δ²f/2 term before the additional factor1/2 in the learning transformation. Thus the first498 rows require2.5e-7, while the original labels incorrectly assign5e-7 everywhere.

**Spatial coordinate:** the generator evaluates f/g at the odd grid for even j and the even grid otherwise, whereas the splitter always writes the even grid. Even j=2,4,…,1998 gives999 blocks ×498=497,502 mislabeled rows.

**Deterministic check:** set every realized W to zero, retaining the actual grid, forcing, initial displacement and initial velocity. The tested identity is r/hⱼ=f(sⱼ). A separate stochastic replay checks r/hⱼ−f(sⱼ)−g(sⱼ)Wⱼ/(4hⱼ)=0 using every captured original normal draw. These are algebraic construction checks, not estimates of training accuracy.

## Exact published changes

Using zero-based NPZ row indices:

- `step_sizes[0:498,0]`:5e-7 →2.5e-7.
- `x_data[498*b:498*(b+1),1]` for b=1,3,…,1997: replace `space[::2][1:-1]` by `space[1::2][1:-1]`. This is a +Δ spatial correction, evaluated from the original grid to preserve its floating-point convention.
- No other array elements change. `r_data`, `train_idx`, `validation_idx`, `test_idx`, and `x_data[:,0]` are bitwise preserved.

The trajectory is not stored as u in the canonical NPZ; its deterministic replay hash is `6393cab5e9d22006338c1be1b8391c0bfc400db6cf2aeef80c43d8b490e800d8`. No trajectory values or learning increments were changed. This does not alter the historical integrator or spatial stencil.

`data/ex6_labels_v2_changes.npz` records the row, column, old value and new value of **every changed element**. SHA-256:`5263675cc48022108078354c213ea5bff235366546f219d7d42194fc07b27917`.

`data/ex6_labels_v2.json` records parent path/hash, original generation revision/configuration, correction-script hash, change-record hash, and validation statistics. Original data and original metadata remain unchanged.

## Before/after validation

Errors below are in drift units across all995,004 rows:

| Construction check | Before max absolute | After max absolute | Before RMSE | After RMSE |
|---|---:|---:|---:|---:|
| Actual-grid, actual-forcing, zero-noise r/h−f |2.49980261|2.00999217e-11|0.05053045|2.01090527e-12|
| Original stochastic replay, realized noise subtracted |2.49980261|2.75512946e-11|0.10992493|3.04863115e-12|

The initialization variance ratio Var[(g/4)W]/[h(g/2)²] changes from0.5 to1; it is1 for every corrected row. Serialization round-trip equality and original-file checksum preservation pass. The corrected file preserves all995,004 observations and the existing split memberships. `evidence/ex6_v2_validation.json` holds full-precision statistics.

## Uniqueness and scope

For the existing **autonomous coefficient-learning construction**, both corrections are uniquely determined by cancellation of the actual generator and its recorded noise law; no new hyperparameter, stochastic realization, lag design, or sample-selection choice is needed. This resolves Raul's requirement to specify and correctly apply the wave increment variance/observation convention.

The first input column remains the historical integration-buffer coordinate jΔ. It is not reinterpreted as physical observation time: forcing is autonomous and the correction changes only the two demonstrated labels. This note does not certify a different/non-autonomous SPDE scheme or silently modify its spatial discretization.

`historical dataset -> confirmed labeling defect -> blocker-required label correction -> corrected canonical dataset`.

Production eligibility remains separate from this data correction: the global split and historical estimator-definition decisions must not be inferred from dataset correctness.
