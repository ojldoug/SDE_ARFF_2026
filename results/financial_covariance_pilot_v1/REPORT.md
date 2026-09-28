# Fixed-basket financial conditional-second-moment pilot

Exploratory BTCUSDT/ETHUSDT/BNBUSDT hourly spot returns; common zero mean. Fits2022–23, validation2024 reused for checkpoint/configuration selection, sealed test2025. D means fitting-only return scales; all NLLs below are in the same physical decimal-log-return coordinates. This is not the seven-fit SDE estimator or an investment-performance study.

## All selected-seed outcomes

| Method/seed | Validation NLL | Test NLL | Raw SPD violations | Projection frequency | 95% joint coverage |
|---|---:|---:|---:|---:|---:|
| constant | -11.827966 | -11.805733 | 0.000% | 0.000% | 91.016% |
| ewma | -12.102773 | -12.389138 | 0.000% | 0.000% | 90.742% |
| dcc | -12.221172 | -12.419359 | 0.000% | 0.000% | 92.774% |
| arff_s0 | 76.957916 | 136.789199 | 21.541% | 21.587% | 70.674% |
| arff_s0_diagonal | 40.975732 | 91.791269 | 4.486% | 4.486% | 88.767% |
| arff_s1 | 242.957174 | 224.607199 | 26.952% | 26.975% | 65.514% |
| arff_s1_diagonal | 188.668006 | 147.321057 | 8.174% | 8.185% | 83.858% |
| arff_s2 | 193.883970 | 194.508944 | 30.833% | 30.856% | 63.744% |
| arff_s2_diagonal | 130.722289 | 136.245893 | 12.957% | 12.957% | 80.080% |
| neural_s0 | -12.175584 | -12.422168 | 0.000% | 0.000% | 93.687% |
| neural_s1 | -12.204665 | -12.379841 | 0.000% | 0.000% | 93.425% |
| neural_s2 | -12.180957 | -12.412547 | 0.000% | 0.000% | 93.436% |

## Selection and stability
ARFF grid index 2, neural grid index 1, EWMA decay 0.97. Configurations selected by2024 only; all three selected fitting seeds retained. DCC valid: True. All18 candidate validation/checkpoint results are in candidate_validation.csv; DCC optimizer start/convergence details are in its saved model and DCC_REPORT.json.

ARFF minus constant: seed NLL differences [148.59493162034542, 236.41293199139324, 206.3146774583924]; months with lower NLL [0, 0, 0]/12. No independent-hour confidence interval or seed-as-market replication claim.
ARFF minus ewma: seed NLL differences [149.1783369365896, 236.99633730763742, 206.89808277463658]; months with lower NLL [0, 0, 0]/12. No independent-hour confidence interval or seed-as-market replication claim.
ARFF minus dcc: seed NLL differences [149.20855751312882, 237.02655788417664, 206.9283033511758]; months with lower NLL [0, 0, 0]/12. No independent-hour confidence interval or seed-as-market replication claim.

Off-diagonal contribution (full minus same-raw-diagonal ARFF score) by seed: [44.99792953116081, 77.28614222655219, 58.263051310071035]. Negative values favor retaining off-diagonals. This is a fixed descriptive ablation, not another selected model.

## Calibration and interpretation
Gaussian nominal coverage is diagnostic; heavy tails and regime changes are not assumed absent. Squared returns are noisy realizations, not covariance truth. Reported raw eigenvalues/projection magnitudes are in fitting-scaled coordinates; score/calibration use the fixed1e-4 floor before mapping to physical units. No runs are removed for non-SPD raw covariance. Portfolio scores for fixed equal weights and BTC-ETH spread are in portfolio_scores.csv, with monthly counts. No weights were optimized and no profits computed.

Months are dependent descriptive periods. Fitting seeds vary numerical initialization/adaptation on a single market history. The asset basket, exchange and USDT numeraire are fixed retrospective choices and do not support broad equity/FX or investment claims.

## Reproduction and audit
PROTOCOL.md, FROZEN.json, DATA_MANIFEST.json, SPLIT_MANIFEST.json, PREFIT_CHECKS.json and PRETEST_GATE.json identify data/config/source and causal/numerical gates. Rightsholder archive files are not distributed with Git; use the downloader and checksums. Models, full predictions and score arrays remain in the local study directory. Per-run/per-month/paired/portfolio CSVs retain all valid outcomes. Native ARFF and covariance-MLP functions were reused without production edits. See execution.json and terminal records for resource use and any incomplete comparator. No manuscript or accepted result was changed.

## Final findings and bounded completion

1. **Colloidal correction:** all eight float64 eigen/Cholesky comparisons pass the original rtol=atol=1e-10. The neural mean NLL changed by about1.05e-8; other changes are at double-precision rounding scale. No conclusion or coverage ranking changed. The old failed checks remain preserved. The four-versus-six-column schema and experimental-cell/optical-phase limitations remain unresolved; retain it as an archived supplementary candidate, without automatic manuscript inclusion.

2. **Financial execution and validation:** nine ARFF and nine neural fits completed, with a valid DCC comparator. All147 archive hashes,19 serialized stage envelopes,18 earliest-minimum histories, pre-test checkpoint reconstruction, configuration selection, causal timestamps, common Gaussian algebra and output consistency passed. Fitting17349 targets; validation8784; test8760. DCC had 1 unsuccessful start(s) among12 bounded marginal/correlation optimizer starts; every objective had a valid converged candidate, chosen by fitting objective. No fit was rerun, replaced or discarded for poor performance. The initial incomplete-hour acquisition failure is preserved; incomplete bars were excluded before fitting, not interpolated.

3. **Predictive value:** selected ARFF lambda=.064 test NLL mean 185.301781, sample SD 44.627113, range[136.789199,224.607199]. It is worse than all three domain baselines in each of12 test months for each of3 fitting seeds. There is no positive predictive result to promote. Selected neural LR=.001 mean -12.404852, sample SD 0.022188, range[-12.422168,-12.379841]; DCC -12.419359, EWMA(.97) -12.389138, constant -11.805733. Neural-versus-DCC results vary by seed; this pilot does not establish neural superiority. Seed SD describes fitting variation on one shared market history.

4. **Off-diagonal/SPD findings:** retaining ARFF off-diagonals worsens NLL versus its own raw-diagonal ablation by [44.99792953116081, 77.28614222655219, 58.263051310071035] nats per observation for seeds0,1,2. Test raw SPD violations are [0.21541095890410958, 0.26952054794520547, 0.30833333333333335]; the fixed floor therefore changes a substantial portion of forecasts. Diagonal-only forecasts also have raw negative entries and poor scores. Large covariance-weighted residuals after flooring show poor calibration under the declared rule; no alternate-floor analysis was run, so a quantitative causal attribution to the particular floor is not established. This does not justify choosing a new floor or ridge from test outcomes.

5. **Inclusion recommendation:** retain the financial pilot as an archival negative application diagnostic, with optional supplementary reporting if the paper discusses scope limits. It is not evidence of a successful real-data ARFF application and is not recommended as a new main-text application. No automatic manuscript update or further search is warranted by this bounded task. These findings concern this fixed basket, features, finite candidate budget and scoring rule; they are not a universal statement about ARFF or financial models.

### Calibration and raw covariance detail

Raw eigenvalues and projection magnitudes below are in fitting-scaled coordinates. Joint95% coverage is already reported above; the Gaussian nominal mean squared Mahalanobis value is3. Full symmetric-whitened means/second moments and50/90/95% coverage are retained for both splits in summary.json.

| Forecast | Mean Mahalanobis squared | Min raw eigenvalue | Mean projection Frobenius | Max projection Frobenius |
|---|---:|---:|---:|---:|
| constant | 3.49083 | 0.116881 | 0 | 0 |
| ewma | 3.53671 | 0.0161018 | 0 | 0 |
| dcc | 3.07985 | 0.0222958 | 0 | 0 |
| arff_s0 | 303.219 | -3.60023 | 0.0213228 | 3.60033 |
| arff_s0_diagonal | 209.509 | -1.23231 | 0.00754567 | 1.76999 |
| arff_s1 | 479.377 | -1.72207 | 0.0331375 | 1.72217 |
| arff_s1_diagonal | 321.389 | -0.880912 | 0.0165817 | 0.940229 |
| arff_s2 | 419.224 | -3.23973 | 0.0339769 | 3.23983 |
| arff_s2_diagonal | 299.362 | -1.27768 | 0.0150905 | 1.43514 |
| neural_s0 | 2.74693 | 0.0210026 | 0 | 0 |
| neural_s1 | 2.7923 | 0.0210309 | 0 | 0 |
| neural_s2 | 2.79812 | 0.0229729 | 0 | 0 |

### Fixed portfolio outcomes

Scalar Gaussian scores use the same frozen physical forecasts; lower is better. Forecast variance, observed squared-return mean, standardized second moment and95% coverage are in portfolio_scores.csv, including all months.

| Forecast | Equal-weight NLL | BTC−ETH spread NLL |
|---|---:|---:|
| constant | -3.761506 | -3.789727 |
| ewma | -3.841730 | -3.970961 |
| dcc | -3.873886 | -3.969462 |
| arff_s0 | -3.225325 | -3.932852 |
| arff_s0_diagonal | 10.064608 | 1.908373 |
| arff_s1 | -3.223137 | -3.679884 |
| arff_s1_diagonal | -0.811235 | 1.988029 |
| arff_s2 | -3.418253 | -3.768452 |
| arff_s2_diagonal | 14.757684 | -0.054670 |
| neural_s0 | -3.864172 | -3.993585 |
| neural_s1 | -3.851275 | -3.976801 |
| neural_s2 | -3.857099 | -3.994668 |

The joint-score ablation does not imply that off-diagonal forecasts are uniformly useless: full ARFF forecasts score better than their diagonal-only versions for both fixed portfolios in all three seeds. Neither full ARFF portfolio beats the corresponding EWMA/DCC score. These are fixed descriptive projections of the same forecasts, not optimized portfolios or substitutes for the primary joint score.

### Selection and execution record

All nine ARFF runs selected native post-adaptation iteration1 (all300 iterations were still executed). Neural native epoch indices are zero-based; all300 epochs were executed. Selected-configuration neural indices by seed are ['43', '80', '28']. Checkpoint_summary.csv retains native validation losses and elapsed cost for every candidate. ARFF has3072 retained parameters (1536 frequencies+1536 amplitudes); neural has1275. This is not a capacity-matched or equally optimized comparison.

One A6000, GPU UUID GPU-d2cfc73c-186a-6651-e9f1-c6d93a00f0b2; allocated 330.640s = 0.091844 GPU-hours, supervisor wall 330.854s, sampled peak device memory 442MiB (5-second samples, not an exact instantaneous maximum). Warm-up9.141s is included. Allocation conservatively includes CPU DCC and reporting time while the GPU was held. No recovery was used. The detached controller exited normally and launched no follow-up.

The frozen pre-test sources were unchanged. The added output-audit and closing-report scripts only read completed archives/aggregates. Figures were visually inspected for readable labels and clipping. Verification is automated and is not an independent human scientific review. No production, data-generation or manuscript sources were changed.
