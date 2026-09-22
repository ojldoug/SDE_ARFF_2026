# Conservative manuscript review

This isolated revision starts from the authoritative red/blue manuscript and its active appendices. It retains the method → applications → results → sensitivity progression and original authorship. The alternative review was used for checked corrections, numerical tables and existing publication assets; it was not used as the manuscript template.

## Review files

- `manuscript.pdf`: complete review draft, including active appendices and explanatory supplement.
- `sources/RaulCom_ARFFSDELearning.tex`: main editable source; `appendices_part2.tex`, `appendices_part3.tex`, `supplementary_diagnostics.tex`, `references.bib` and `tables/` complete the source.
- `CHANGE_LOG.md`: section-by-section required corrections, retained content, withheld figures and separately identified optional editorial choices.
- `AUTHOR_DECISIONS.md`: unresolved decisions only.
- `archive_inputs/`: compact archived scalar metrics and provenance, with SHA-256 manifest.
- `evaluation/RECONSTRUCTION_CHECKS_PUBLIC.json`: portable record of the earlier saved-checkpoint checks, with relative paths and hashes.

The local historical `original/` and build-history manifests are retained on the research server but are not part of the Git snapshot; the current review sources are the active build inputs. See the repository-root `REPRODUCING.md` for the complete evidence map.

Rebuild tables, figures and manuscript in a fresh output directory from the repository root:

```bash
python scripts/rebuild_review_snapshot.py --output /tmp/sde-review-example
```

Keep `sources/` and its sibling `figures/` together when sharing. No external manuscript images are needed by the revised source. The PDF includes the supplement; a separate supplement document has not been substituted for the original active appendices.

## New coefficient figures

Vector PDF and PNG versions are in `figures/`:

- `ex1_seed0_recovery`: four methods, both prescribed 201-point coordinate slices on [-1,1].
- `ex8_seed0_drift`: truth and all five methods on the prescribed 100×100 cell-centre grid on [-2,2]², including component errors on shared scales.
- `ex8_seed0_raw_covariance`: truth and all five raw covariance predictions, with shared scales by component.
- `ex8_seed0_projection`: separately labelled ARFF raw/projected covariance diagnostic at epsilon=0.001.

Seed 0 is fixed and illustrative, not representative, best or selected by appearance. New grid values do not replace archived evaluation metrics. Coordinates, truth and model predictions are saved in `evaluation/ex1_seed0_coefficients.npz` and `evaluation/ex8_seed0_coefficients.npz`. The original evaluation-population predictions are also retained.

## Reconstruction and provenance

`evaluate_coefficients.py` was run once on the GPU backend, without fitting or generating stochastic observations. All nine model reconstructions were checked against archived drift RMSE, raw covariance RMSE and NLL before any visualization grid was evaluated. All 27 checks passed the predeclared rtol=1e-5, atol=1e-6 bounds; no tolerance was relaxed. Ex1 uses its original validation indices; corrected Ex8 uses its canonical test indices. `evaluation/RECONSTRUCTION_CHECKS_PUBLIC.json` records those checks and relative source/checkpoint/dataset hashes. The exact loader sources are present in the tracked `src/` and `GPU/lib/` tree, with their hashes checked against the record. Checkpoint arrays were not converted or refitted to obtain a pass.

`plot_coefficients.py` draws only the saved coordinate/prediction arrays. The portable rebuild script regenerates N/h/K/baseline and mechanism figures from the compact archive inputs and checks the main numerical tables at their reported precision. The earlier build-history script in `aggregate_source_record/` is not the portable rebuild command. Existing 30-seed results are unchanged.

The projection grid has an ARFF minimum projected eigenvalue slightly below 0.001 at float32 rounding precision; this is recorded in `evaluation/GRID_DIAGNOSTICS.json`, not concealed or reprojected. Grid SPD rates are visualization-domain diagnostics and are not test-set rates.

## Exploratory lag/ridge supplement

The supplement contains the completed three-lag, three-penalty, three-seed ARFF intervention, with all nine arm means and all 18 individual paired changes. The main sensitivity discussion has one pointer to it. `lag_ridge_assets.py` reads only compact copies in `archive_inputs/`, checks each paired change against the underlying run rows, and generates two TeX tables plus `figures/lag_ridge_paired_nll.pdf` and `.png`. The figure plots within-lag paired NLL changes against drift penalty 0.001; it does not compare raw NLL levels across lags. The server-side `results/ex8_h_lambda_drift_v1/` study retains the full artifacts and logs. No model predictions or fitting were used for this addition.

## Boundaries

This review build performed no training, hyperparameter tuning, stochastic data generation, trajectory simulation or timing campaign. The authoritative manuscript and alternative draft remain unchanged. Ex4's description is retained with a visible author-review note and no numerical comparison. Missing trajectory and isolated-runtime evidence is identified rather than replaced by unrelated coefficient or sensitivity figures. Historical discrepancy causes remain partly unresolved. These limits and final presentation decisions are listed in `AUTHOR_DECISIONS.md`.
