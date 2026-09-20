# Final authenticated Experiment 8 publication outputs

The final bundle is `results/final_ex8_publication_bundle/`, with `FINAL_EX8_RESULTS_README.md` and `SHA256_MANIFEST.json`. Results/data remain intentionally Git-ignored; this commit does not override those rules. The server backup preserves these outputs and final model/data archives independently of Git.

## Audit

`results/capacity_regime_ex8_v2/ARFF_DRIFT_K_CONSISTENCY_AUDIT.md` and JSON classify the flat/noisy ARFF drift curve as **verified scientific result**. All 50 artifacts share the correct data/indices/target convention. GPU seed-0 reconstruction at every K matches saved drift RMSE within 2.98e-8. K1024 anchors are byte-identical; K128 runs are newly fitted at N640000. Selected iteration 1 occurs frequently but follows the accepted internal-validation rule. No genuine correctness defect was found.

## Figures

Under the bundle's `figures/`, all of these have PDF, PNG, CSV and JSON source data:

- `ex8_baseline_rmse`: corrected float64, N80000, 30 seeds, five methods.
- `ex8_capacity_rmse` and `ex8_capacity_nll`: fixed N640000, h1e-4, ten seeds, x-axis retained parameter count.
- `ex8_h_rmse` and `ex8_h_nll`: final nine lags, K1024, N640000, ten seeds.
- `ex8_N_rmse`: frozen N study, K1024, h1e-4, ten seeds.
- `arff_raw_spd.csv/json`: complete raw SPD diagnostics across all four studies.

Covariance RMSE is raw/pre-projection. ARFF NLL uses its accepted SPD floor. Repetition sets are not pooled. Phase 1 established neither capacity dominance nor universal N/h equivalence. No empirical O(1/K) rate is claimed. The h=.002 endpoint was governed by the preregistered bias tolerance, not method performance. Timings were concurrent/non-isolated.

## Reproduction and preservation

`build_final_ex8_publication_bundle.py` verifies frozen source hashes and seed coverage and plots only stored metrics; it refuses an existing output directory. `audit_final_ex8_drift_capacity.py` and `confirm_final_ex8_drift_audit_gpu.py` document saved-model reconstruction without fitting. `backup_final_ex8_publication.py` preserves the bundle, data, final native archives, authenticated logs and Git bundle with SHA-256 checks and internal hard-link deduplication. Its exclusive destination is `../backups/SDE_ARFF_2026_final_ex8_2026-09/`.

Experiment 8 is numerically closed unless future claims need genuinely new evidence. Corrected float64 outputs supersede original float32 scientific conclusions; historical outputs remain preserved. No training, data regeneration, estimator changes, poster edits or manuscript edits occurred during this preservation task.
