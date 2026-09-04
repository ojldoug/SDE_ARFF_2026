# Final experiment specification

This document defines the intended reproducible experiment protocol for the
revised ARFF SDE manuscript.

The legacy notebooks are retained as historical material, but they are not
treated as an authoritative experiment specification. Where legacy code,
manuscript descriptions, and methodological corrections conflict, the final
protocol prioritizes:

1. mathematical correctness and Raul's blocker corrections;
2. fair comparison between training methods;
3. reproducibility;
4. consistency with the manuscript experiments where scientifically justified.

## Global protocol

### Learned quantities

The drift model approximates

\[
f(x).
\]

The second-stage model approximates the covariance

\[
\Sigma(x) = \sigma(x)\sigma(x)^\top.
\]

The revised implementation therefore reports covariance errors against
\(\Sigma(x)\), rather than interpreting the stage-2 output as a diffusion
factor.

### Two-stage covariance learning

Covariance targets are formed using out-of-fold drift predictions:

\[
e_n
=
r_n
-
h_n \widehat f^{(-\mathrm{fold}(n))}(x_n),
\]

\[
C_n
=
\frac{e_n e_n^\top}{h_n}.
\]

Five-fold cross-fitting is used for these targets.

After constructing all covariance targets, the final drift model is fitted
using all training observations, and the covariance model is fitted using all
training inputs and their out-of-fold covariance targets.

### Symmetry and positive definiteness

The raw learned covariance is reconstructed as a symmetric matrix.

For diagnostics and covariance RMSE, use the raw learned covariance

\[
\widehat\Sigma_{\mathrm{raw}}(x).
\]

Record the fraction of evaluation points at which the raw covariance is not
positive definite.

For likelihood evaluation and SDE simulation, project the raw covariance by
eigenvalue clipping:

\[
\widehat\Sigma_\varepsilon
=
Q\,
\operatorname{diag}
\bigl(
\max(\lambda_i,\varepsilon)
\bigr)
Q^\top,
\]

with

\[
\varepsilon = 10^{-6}.
\]

Thus SPD enforcement does not alter the covariance-RMSE diagnostic.

### Method comparison

ARFF and Fourier-feature Adam use the same experiment-level Fourier-feature
count \(K\).

The feature budget is a property of the experiment, not of the optimizer.

The same generated dataset and the same train/validation split are used by all
methods being compared.

### Data split

Use a fixed 90/10 train-validation split for the final manuscript comparison.

The split is generated once from a documented seed and reused across methods.

No test set is introduced unless the manuscript protocol is explicitly changed
before the final numerical campaign.

### Repetitions

Use 30 independent training runs for reported timing/loss statistics.

The dataset and split remain fixed.

The run index controls optimizer / random-feature randomness, not regeneration
of the underlying benchmark dataset.

Record the backend, seed, and repository commit associated with each run.

GPU execution is not assumed to be bitwise reproducible across separate
processes.

## ARFF configuration

Unless an experiment-specific setting is documented:

- iterations: 300;
- regularization: \(\lambda = 10^{-3}\);
- \(\gamma = 1\);
- \(\delta = 0.2\);
- resampling: false;
- Metropolis test: true;
- covariance-target folds: 5.

## Adam configuration

The rebuilt Adam baseline uses:

- epochs: 300;
- learning rate: \(10^{-4}\);
- batch size: 256.

Fourier-feature Adam uses the same experiment-level \(K\) as ARFF.

Any shallow/deep tanh baselines must have their architecture documented
explicitly before final runs.

## Evaluation

For known synthetic benchmarks report at least:

1. validation Gaussian negative log-likelihood;
2. drift RMSE;
3. raw covariance RMSE;
4. raw SPD-violation rate;
5. minimum raw covariance eigenvalue.

For likelihood evaluation, use the SPD-projected covariance.

Covariance RMSE is evaluated before projection.

Historical "diffusion RMSE" values are not assumed to be directly comparable
with covariance RMSE because the revised second stage learns covariance.

# Experiment-specific specification

## Experiment 1

Benchmark: two-dimensional cubic SDE.

Status:

- sample count: 10,000;
- retained observation lag: \(h = 10^{-2}\);
- EM substeps per retained observation: 1000;
- fine EM step: \(10^{-5}\);
- Fourier-feature count: use the final manuscript value;
- data seed: fixed and documented.

The \(h=10^{-2}\) reconstruction is supported both by the manuscript
1000-substep description and by the reconstructed validation-NLL scale.

Status: CONFIRMED, except final manuscript \(K\) must remain synchronized with
the final experiment table.

## Experiment 2

Benchmark: three-dimensional SDE with lower-triangular diffusion factor.

Known:

- 1000 EM substeps per retained observation;
- Fourier-feature count: use the final manuscript value;
- same \(K\) for ARFF and Fourier-feature Adam.

Retained observation lag \(h\): UNRESOLVED.

Fine step is defined as \(h/1000\).

## Experiment 3

Benchmark: ten-dimensional SDE with symmetric diffusion/covariance structure.

Known:

- 1000 EM substeps per retained observation;
- Fourier-feature count: use the final manuscript value;
- same \(K\) for ARFF and Fourier-feature Adam.

Retained observation lag \(h\): UNRESOLVED.

Fine step is defined as \(h/1000\).

This experiment is a principal stress test for raw-covariance SPD violations.

## Experiment 4

Benchmark: underdamped Langevin system.

The learned stochastic component is the velocity equation. The position is
coupled through the deterministic update corresponding to

\[
dx_t = v_t\,dt.
\]

Known:

- 1000 EM substeps per retained observation;
- Fourier-feature count: use the final manuscript value;
- same \(K\) for ARFF and Fourier-feature Adam.

Retained observation lag \(h\): UNRESOLVED.

Fine step is defined as \(h/1000\).

## Experiment 5

Benchmark: SIR stochastic simulation algorithm.

Parameters:

- population \(N=1024\);
- \(k_1=1\);
- \(k_2=1\);
- \(k_3=0\);
- 250 trajectories;
- final time \(T=4\).

Use the independently implemented fixed-observation SSA generator.

Do not reconstruct training pairs by skipping event-triggered Gillespie
records.

Observation times are prescribed explicitly, giving exact known pairwise
lags.

The final number / spacing of retained observations must match the final
manuscript description.

## Experiment 6

Benchmark: stochastic wave equation.

Physical space/time discretization:

\[
\Delta = 10^{-3}.
\]

The learning formulation uses the effective step

\[
h_{\mathrm{network}}
=
\frac{1}{2}\Delta^2
=
5\times10^{-7}.
\]

Use the documented staggered-grid wave-equation construction.

Status: CONFIRMED.

## Experiment 7

Benchmark: two-dimensional fat-tail example.

Known:

- 1000 EM substeps per retained observation;
- Fourier-feature count: use the final manuscript value;
- same \(K\) for ARFF and Fourier-feature Adam.

Retained observation lag \(h\): UNRESOLVED.

Fine step is defined as \(h/1000\).

## Experiment 8

Benchmark: two-dimensional near-singular rotated covariance example.

This experiment was added after the earlier submitted experiment table and is
intended to stress-test covariance learning and SPD enforcement.

Known:

- raw covariance has a very small eigenvalue by construction;
- covariance type: symmetric;
- state domain: \([-2,2]^2\).

The final manuscript Fourier-feature count must be used consistently by ARFF
and Fourier-feature Adam.

Data-generation parameters must be explicitly fixed before the final run.

Status: PARTIALLY RESOLVED.

# Remaining blockers before final numerical campaign

The following must be resolved before canonical datasets are regenerated:

1. retained observation lag \(h\) for Experiments 2, 3, 4, and 7;
2. final manuscript \(K\) table for Experiments 1--8;
3. final fixed observation schedule for Experiment 5;
4. final data-generation specification for Experiment 8;
5. exact shallow/deep tanh architecture if those baselines remain in the paper.

Once these items are resolved:

1. update `src/experiments/config.py` once;
2. generate each canonical dataset once;
3. save split indices with the dataset;
4. record dataset metadata/checksums;
5. run smoke tests;
6. run ARFF and Adam on identical datasets/splits;
7. run the full 30-run campaign;
8. regenerate every numerical table and figure from saved results;
9. update the manuscript to describe exactly this implementation.
