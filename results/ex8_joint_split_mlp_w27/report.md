# Experiment8: modern Joint versus Split MLP

All30 new Split MLP seeds completed without failures or reruns. All150 artifacts across five methods validated. The120 previously accepted runs and their metric values are preserved. Both MLPs have893 drift and921 covariance-factor parameters (1814 total), with27×27 tanh hidden layers.

| Method | Test drift RMSE mean ± sample SD | Test covariance RMSE mean ± sample SD | Test NLL mean ± sample SD |
|---|---:|---:|---:|
| Joint Fourier Adam | 0.894109 ± 0.073995 | 0.715407 ± 0.150445 | -8.346652 ± 0.064269 |
| Split Fourier Adam | 0.662862 ± 0.043253 | 0.617333 ± 0.092378 | -8.337515 ± 0.060840 |
| ARFF | 0.944982 ± 0.289836 | 0.113834 ± 0.002534 | -8.873750 ± 0.097421 |
| Joint MLP Adam | 0.306213 ± 0.058301 | 0.265343 ± 0.081204 | -10.018567 ± 0.187634 |
| Split MLP Adam | 0.720134 ± 0.135234 | 0.217470 ± 0.057651 | -10.139605 ± 0.184133 |

| MLP | Metric | Median | Range |
|---|---|---:|---|
| Joint MLP Adam | drift_rmse | 0.291582 | [0.229265, 0.447318] |
| Joint MLP Adam | covariance_rmse | 0.238213 | [0.182700, 0.480786] |
| Joint MLP Adam | nll | -10.034187 | [-10.433053, -9.680981] |
| Split MLP Adam | drift_rmse | 0.723604 | [0.428625, 0.985762] |
| Split MLP Adam | covariance_rmse | 0.202488 | [0.141645, 0.359362] |
| Split MLP Adam | nll | -10.175495 | [-10.463038, -9.720182] |

| Metric | Split−Joint mean | Percentage change | Seeds improved | Leave-one-seed-out mean difference range |
|---|---:|---:|---:|---|
| drift_rmse | 0.413921 | +135.174% | 0/30 | [0.40394252396203667, 0.42391449845934087] |
| covariance_rmse | -0.047873 | -18.042% | 20/30 | [-0.05475743692434627, -0.03983707775726366] |
| nll | -0.121038 | -1.208% | 20/30 | [-0.1386249897769062, -0.10064399878779993] |

Percentage=100×(Split mean−Joint mean)/abs(Joint mean). NLL has an arbitrary additive convention, so its absolute change is the primary comparison; the1.208% figure only describes this shared stored convention. Lower is better for all three metrics.

Split does not uniformly improve performance: drift worsens in all30 paired seeds, while covariance and NLL improve in20/30 each. Median covariance/NLL also improve, and excluding any single seed preserves all mean-change directions. Benefits therefore are not explained by one outlier, but are not uniform across seeds.

This is defensible as a modern comparison of complete joint versus cross-fitted two-stage procedures at matched final capacity. It is not equal-compute training and does not reproduce a uniquely recovered Owen historical split procedure. Split uses300 epochs for each of fiveOOF drift fits, final drift fit and covariance fit; Joint uses300 joint epochs. Stage-specific checkpoint objectives and the extra PRNG progression are intrinsic to the approved split procedure. No settings were tuned.

Historical source remains incomplete: Table2 K512 conflicts with Appendix K1024 (deep width K/2); Appendix reports2000 epochs,lr.001,batch512 but per-stage assignment is unresolved. Originating split objectives, initializer, covariance representation, checkpoint chain, OOF status and seeds are unverified. Manuscript shallow split aggregates are NLL−8.263/drift.8544/Diff..2494; deep split−8.931/.8544/.0701; these are not substituted into the modern comparison. See ../production/ex8_split_mlp_w27_campaign_control/historical_search.md and preflight.md.

Validation: CPU synthetic integration and mocked supervisor tests passed; native schema/configuration checks, architecture/1814 counts, dataset/source hashes, deterministic fold IDs, checkpoint histories, all30 seeds and GPU backend checked. Figure visually inspected with no cropped labels/annotation; PDF is vector. Four-GPU campaign timing is non-isolated. Accepted numerical modules, existing artifacts, prior figures, poster and manuscript remain unchanged. No isolated timing or unrelated production jobs were launched.

Artifacts/logs: ../production/adam_split_mlp_ex8_w27/seed_{0..29}_artifacts.npz and seed_{0..29}.txt. Persistent session: split_mlp_ex8_w27_campaign (completed).

New figure: ex8_capacity_matched_five_methods_rmse_two_panel.pdf and .png. Exact numerical distributions, per-seed values and provenance: summary.json, test_distributions.npz, per_seed_test_metrics.csv.
