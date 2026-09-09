# Historical split-MLP investigation (revised paper, not poster replacement)

Authoritative primary manuscript §6 introduces split Adam only for Experiment8; Table2/3 list joint and split shallow/deep tanh variants only in that row. §6.1 Figure3 reports30-run mean curves. The problem is the rotated near-singular diffusion-factor experiment, not modified ex7 used in §6.2.

| Historical method | Validation NLL (mean minima) | Drift RMSE | Table3 diffusion/covariance RMSE |
|---|---:|---:|---:|
| Joint shallow tanh | -7.063 |1.2264|.6761|
| Joint deep tanh |-10.10|1.3786|.3252|
| Split shallow tanh |-8.263|.8544|.2494|
| Split deep tanh |-8.931|.8544|.0701|

These are manuscript transcriptions, not newly validated runs. Raw versus projected covariance and sigma versus Sigma must be verified against the missing originating split implementation before treating that last column as a fully audited covariance metric. No sample SD/raw repetitions are recoverable from these aggregate table entries alone.

Table2 says cumulative K512, whereas Appendix Table11 specifies log2K10=1024, batch512, lr1e-3,2000epochs for ex8. Its caption defines deep widthsK/2 each. Thus there are conflicting capacities: shallow512 versus1024; deep256×256 versus512×512. No width is chosen on performance grounds.

For two independent biased tanh networks with2 inputs, drift2 outputs and full factor3 outputs: shallow widthw has drift5w+2, factor6w+3, total11w+5; two equal hidden layers have driftw²+6w+2, factorw²+7w+3, total2w²+13w+5. The old tanh source also stores unused amp leaves (2+3 parameters); these are not active estimator parameters. These formulas do not prove which candidate width generated the published results.

Surviving GPU/lib/lib_Adam_tanh.py supplies joint Gaussian NLL, historical symmetric sigma=LLᵀ, Sigma=(LLᵀ)², Glorot drift weights/zero biases, covariance weights AND biases uniform[-.01,.01], and Adam(b1=.9,b2=.999,eps1e-7). It contains no split trainer. It is supporting joint-code evidence, not proof of the missing historical split initializer. The modern src/adam/split_mlp.py is a later implementation and must not be relabeled as Owen's historical source.

Unresolved: exact split drift/covariance objectives, direct covariance versus factor output, initialization, learning rates and epoch budgets per stage, cross-fitting, model-selection semantics, exact repetition seeds, numerical timings and association of each saved aggregate with its source revision. The manuscript says sequential regression targets but does not resolve these implementation details. No older split-tanh source was found in the available project/history searches.

Required correction record when source is recovered:

`Owen historical split-MLP residual construction -> out-of-fold nuisance prediction required by accepted Raul correction -> same evidenced architecture/objectives/initialization/budgets with cross-fitted drift residuals`.

Joint and split must use identical evidenced architecture within each pair. No new tuned model, no larger/smaller paired capacity, and no replacement of accepted poster artifacts is authorized by this investigation. A distribution figure/table is deferred until the paired historical procedure is recoverable and corrected runs exist.
