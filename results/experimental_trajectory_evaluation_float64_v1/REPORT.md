# Common float64 reevaluation of the colloidal pilot

All four saved raw covariance and drift arrays were promoted to float64 before symmetrization, eigenanalysis, flooring, solves, log determinants and reductions. This cannot restore precision lost during prediction. Same observations, h=.2s, physical units, floor1e-4um²/s and equal retained-increment weighting; no new model predictions or fits. Original outputs and failed1e-10 check remain unchanged. Corrected eigen/Cholesky checks pass the original rtol=atol=1e-10.

| Method | Mode | Corrected NLL | New minus old | 95% coverage |
|---|---|---:|---:|---:|
| affine | end_to_end | -1.077787593 | -4.44e-16 | 95.375% |
| affine | shared_affine_drift | -1.077787593 | -4.44e-16 | 95.375% |
| kernel | end_to_end | -1.074057669 | 0 | 95.290% |
| kernel | shared_affine_drift | -1.075741488 | 0 | 95.370% |
| arff | end_to_end | -1.076604043 | 0 | 95.320% |
| arff | shared_affine_drift | -1.077028228 | 0 | 95.340% |
| neural | end_to_end | -1.076933019 | 1.05e-08 | 95.565% |
| neural | shared_affine_drift | -1.076929623 | 1.05e-08 | 95.540% |

Conclusion unchanged: very similar scores; no demonstrated ARFF advantage over constant covariance. All raw SPD violation/floor rates remain zero. Coverage changes are recorded explicitly in summary.json. The four-column versus six-column discrepancy remains unresolved: local README/header inspections establish a plausible mapping, not author-confirmed semantics. Local screening_notes.md and source paper/readme supply no unique experimental-cell IDs or per-movie optical-phase calibration. Do not claim these issues resolved. No author contact occurred. Archived supplementary candidate only; no manuscript inclusion.
