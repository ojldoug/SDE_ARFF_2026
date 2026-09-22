# Bounded review-source corrections, 22 September 2026

This is a review draft, not the authoritative manuscript. Original structure, authorship, applications, numerical tables, grids and findings are retained.

- Generalized driving noise to dimension m and stated the positive-definite scored-coordinate domain. A regularized Gaussian score is not the true degenerate transition law.
- Corrected ordinary ridge versus empirical weighted GLS wording; added separate target second/fourth-moment conditions and qualified factor-class equivalence.
- Explained the frozen Langevin conditional degeneracy; its sign/conditioning decision remains unresolved. Defined SIR susceptible/recovered coordinates locally.
- Cited existing Adam and cross-fitting entries, corrected the kernel-SDE reference to Darcy et al., added bounded classical regression context from Comte et al., and made both project preprint identifiers render explicitly. Cross-fitting supplies no automatic DML theorem.
- Documented physical-unit preprocessing, dense LU normal-equation solves, float32/HIGHEST products, recomputed features, historical guarded resampling/stopping differences, exact fitting seed lists, missing tuning history and code/archive availability.
- Added compact notation and observation/learned-object overview columns. Explained zero initialization and qualified possible precision sensitivity. Figure 6 error bars are sample SD, verified against replot_final_editorial.py. Table 22 already had the requested shared-data wording and is unchanged.

Sources: src/arff/regression.py; src/arff/validation_selected.py; scripts/final_historical_arff_compat.py; scripts/run_final_historical_arff.py; accepted manifests and reconstruction records indexed in REPRODUCING.md. Bibliographic metadata verified against arXiv:1412.6980, 1608.00060, 2209.12086 and 0708.4165; the last concerns stationary one-dimensional regular observations, not a theorem for our multivariate adaptive estimator.

The private feedback crosswalk remains outside the public snapshot. No confidential correspondence is included. Remaining decisions are unchanged in AUTHOR_DECISIONS.md: Ex4, unresolved historical attribution, withheld trajectory/speed claims, and durable large-artifact release. A successful archived-output rebuild does not verify fresh training.

Verification: 24 compact input hashes and 17 numerical table/group checks passed; ten figures rebuilt only in a new temporary directory. The revised 30-page PDF compiled with no unresolved references/citations or overfull boxes. Changed mathematical, overview, figure-caption, bibliography and reproducibility pages were visually inspected. Accepted numerical table files and figure files remain byte-for-byte unchanged.
