# Evidence classification before affected Experiments 1–7 runs

## 90/10 versus 80/10/10

**Finding:** determinism, fixed splits and provenance are accepted reproducibility corrections; the numerical change from90/10 to80/10/10 is **not required by Raul's blocker**. The primary PDF §5 and active current TeX line1005 say90/10. `FINAL_EXPERIMENT_SPEC.md` lines107–113 also explicitly say90/10 and require an explicit manuscript-protocol change before adding a test set. `SplitConfig` already contained80/10/10 in initial scaffold commit1182e5c. That source commit proves implementation history, not experimental acceptance. The available conversation explicitly approves canonical80/10/10 for Experiment8; no equally explicit approval overriding the historical split for Experiments1–7 was found.

**Implication:** all methods for Experiments1–7 are affected. Their existing canonical observations are reusable where independently valid, but the stored80/10/10 memberships cannot be described as a historical90/10 reproduction. Restoring90/10 needs a versioned index/protocol artifact without overwriting observations. Retaining80/10/10 requires an explicit, non-blocker protocol exception for1–7. A90/10 protocol has no independent test split; the requested test-NLL reporting also needs a defined external-test protocol or an explicit validation-only scope. Do not silently merge splits, repurpose validation as test, or generate an invented test dataset.

## Historical versus modern symmetric Adam

**Finding:** the legacy `GPU/lib/lib_Adam_FF.py` and `lib_Adam_tanh.py` build lower-triangular L with positive diagonal, return sigma=LLᵀ for `diff_type='symmetric'`, then use Sigma=sigma sigmaᵀ=(LLᵀ)² in NLL. The modern Fourier baseline introduced at commitc3fa90a uses sigma=L and Sigma=LLᵀ. The modern MLP added at7de69dc uses the same convention. Both constructions are valid SPD covariance parametrizations. Raul requires SPD likelihood/simulation and consistent sigma/Sigma semantics; he does **not** require removing the extra matrix product from an already valid historical factor parametrization. The modern form is therefore a newer model/implementation choice, not an established blocker-required correction for1–7. Explicit acceptance of the modern Experiment8 campaign remains in force.

**Implication:** among Experiments1–7, only Experiment3 has `diff_type='symmetric'`. Experiment2 is triangular and1,4,5,6,7 are diagonal, so this particular squared-versus-unsquared issue does not apply to them. The historical Experiment3 table includes ARFF and shallow tanh; the shallow tanh baseline is affected. Any additional Fourier Adam ex3 comparison would also be affected. Restoring the historical model preserves Sigma=(LLᵀ)² and its historical initialization; choosing the modern model requires an explicit exception and must not be reported as a labeling-only correction. Same number of parameters does not imply identical optimization geometry or feature-restricted model family.

Other historical/modern initialization and checkpoint differences remain recorded in the audit matrix; changing learning rates alone does not resolve them.

## Dispatch status

Experiment6's data-label stop is cleared by the independently verifiedv2 dataset. It does not block unrelated work. However, the unresolved split policy independently affects **every remaining Experiment1–7 production run**. No affected training was launched while classifying these choices. Accepted Experiment8 accuracy work is already complete and remains unchanged; its isolated timing is a separate phase, scheduled after resolved accuracy production per the approved plan. Section6.2 N/h grids and fixed settings remain unrecovered, not merely the K-study asymptote. Therefore those studies are not eligible for a faithful numerical dispatch yet.

Necessary explicit decisions: whether1–7 retain the existing80/10/10 as an approved exception or use historical90/10 (and how independent test reporting is handled); whether ex3's symmetric Adam must retain the historical squared covariance or receive an explicit modern-parametrization exception. This is not a request to approve the audit matrix again.
