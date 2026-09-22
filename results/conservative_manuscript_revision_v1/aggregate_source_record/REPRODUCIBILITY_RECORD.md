# Final editorial pass: reproducibility record

Only the isolated review package was changed. No model predictions, fits, simulations, data generation or canonical artifact updates occurred. Table aggregates were recomputed only from archived scalar CSVs and checked against their archived JSON means/sample SDs. Earlier preparation scripts are an editorial history; final TeX sources are the build inputs.

## Mathematical corrections
Equation (1) had an actual tab followed by `frac12`; restored `\tfrac12`. Proposition 1 now specifies positive finite mean lag, integrable increment squared divided by lag, and absolute log-lag integrability. The scaled-increment identity has its separate integrability qualification. Equation (2) conditions on the fitted nuisance sigma-field and explicitly states conditional independence. OOF exclusion alone does not establish that independence for dependent trajectory/grid data.

## Table S14 and main Table 2
`evidence/oracle_per_seed.csv` and `oracle_summary.json` agree. S14 float32/float64 CPU blocks are genuinely different but previously rounded to identical displayed values. Added paired differences and explicit CPU labels. Main Table 2 remains stored production-GPU metrics, not CPU re-evaluation. CPU ARFF minus production-GPU mean NLL is 0.0005174954732259115 (sample SD 0.0018618519296143175). Within CPU, float32 minus float64 mean ARFF NLL is 4.1124861844821224e-7. The archived hybrid report documents backend prediction differences; this editorial pass does not further apportion their cause. No inconsistent source aggregate was found. Full verification values: TABLE_S14_RECONCILIATION.json.

The source evaluation script `scripts/evaluate_ex8_oracle_component_swap.py` was read, never executed. Native CPU likelihood uses float32 projected covariance, solves/logdet, increments and reductions. The float64 path uses spectral arithmetic and archived float64 increments with unchanged CPU component predictions. These are different evaluation paths, not independently trained models.

## Citation
Verified project source: `results/manuscript_reconciliation_v1/feedback.txt`, reference [15], lines 1245–1247. Owen Douglas, Aku Kammonen, Anamika Pandey and Raúl Tempone, *An Adaptive Random Fourier Features Approach Applied to Learning Stochastic Differential Equations*, arXiv:2507.15442 (2025). The supplied Manchester poster reference [5] independently confirms identifier/year/authors. No specific arXiv revision number is asserted.

## Dataset and protocol provenance
Ex8 float64 v2 SHA-256: 6fb009e3d6fa5f241cb1a15f9a9815cbd16ea19c3a9e4ca546ae99bdae091a72.
Ex6 labels v2 SHA-256: 47c4a91d37f494621c4174e5c8a87d951a9b78b16b1da68dc4b146f72dbc85c3.
The portable review snapshot records exact source hashes in `evaluation/RECONSTRUCTION_CHECKS_PUBLIC.json`, dataset hashes in `docs/review_data_checksums.sha256`, and compact metric origins in `archive_inputs/MANIFEST.json`. Historical ARFF settings are in the tracked `results/final_reproduction/production_manifest.v5.json` and `HISTORICAL_ARFF_COMPATIBILITY.md`. Earlier corrected lower-N/K studies and separately calibrated modes remain in the server-side `results/controlled_study_2026/float64_v2/final_curves/summary.json`, with their separate grids/search budgets; they are not pooled here. Old float32 curves and nine K32 outputs remain superseded diagnostics. Ex4 and unauthenticated historical trajectory/runtime/scaling plots remain excluded. No source exclusion was silently converted into a scientific result.

## Editorial and display conventions
Reader-facing text describes scientific protocols rather than internal approval status. Sequential regression motivation now explains the fixed-frequency multi-output ridge solves, drift fitting without covariance inversion, direct covariance targeting, and their limitations. Negative results and all data points remain. Metric table means/SDs use four significant digits, with explicit additional precision for evaluation-path comparisons and threshold checks. Figure fonts increased; labels/legends inspected at page size. Tables S14–S15 derive only from archived scalar values. The public release location remains an author decision.

## Additional sources read for this pass
- `results/manuscript_reconciliation_v1/feedback.txt` SHA-256 `6d371e4a274ac0ab9c8a37b321c5a64d95eba9ae561d976df975568cac7ce07e`
- `scripts/evaluate_ex8_oracle_component_swap.py` SHA-256 `a58661f38ca2e57e9691c370768d5df5eaa9db87cd5e41ca43d21779ab0a1cc6`
- `results/controlled_study_2026/float64_v2/oracle_component_swap/report.md` SHA-256 `0696480991c44c2cf00ed40eff4a2b34108c929522ee2d84ba3522577e0a7f4b`
