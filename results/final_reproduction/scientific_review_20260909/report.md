# Scientific review of completed accuracy campaigns

No training or isolated timing was launched.660 new artifacts completed and were validated;120 accepted Experiment8 artifacts remain frozen. This review distinguishes archive integrity from successful historical numerical reproduction.

All values below are30-seed mean±sample SD (ddof=1). Experiments1–7 use canonical90/10 validation-only; there is no independent test set. Experiment8 retains its accepted80/10/10 test results. Covariance RMSE is for raw Sigma, before projection. NLL is comparable across current methods within the same experiment; it is a Gaussian quasi-likelihood, not proof of drift/covariance recovery.

Historical comparisons use Latest_manuscrip_draft.pdf Tables2/3. The historical Diff. column is ambiguously labeled diffusion/covariance and its originating coefficient-evaluation code/sampling/normalization is not recovered. Consequently old/new coefficient entries flag discrepancies, but are not yet certified identical estimands. Historical minimum-validation NLL and corrected selected-model evaluation must also be distinguished. Historical tables give means, not recoverable sample SD.

The main1–7 campaign used historical joint Fourier, shallow tanh(K), deep tanh(K/2,K/2) and corrected cross-fitted ARFF; experiment3 used only shallow tanh and ARFF. K values: ex1=256,ex2=128,ex3=1024,ex5=256,ex6=512,ex7=512. These historical architectures are not parameter-matched. Experiment3 preserves Sigma=(LLᵀ)².

## ex1

| Estimator | Parameters | Drift RMSE | Raw covariance RMSE | Validation NLL |
|---|---:|---:|---:|---:|
| ARFF | 3,072 | 0.35194878 ± 0.023181 | 0.024149077 ± 0.0034717 | -3.2953461 ± 0.0015724 |
| Joint Fourier Adam | 3,072 | 0.51815045 ± 0.023103 | 0.03160798 ± 0.00078715 | -3.2962359 ± 0.00066447 |
| Joint MLP shallow | 2,564 | 0.71913356 ± 0.026997 | 0.014583575 ± 0.00017365 | -3.2830268 ± 0.0010374 |
| Joint MLP deep | 34,308 | 0.5792844 ± 0.020425 | 0.027452061 ± 0.0007586 | -3.2988482 ± 0.00095525 |

Historical → reproduced means:

| Estimator | Drift | Historical Diff. → raw covariance | NLL |
|---|---:|---:|---:|
| ARFF | 0.0487 → 0.3519488 | 0.0091 → 0.02414908 | -3.252 → -3.295346 |
| Joint Fourier Adam | 0.0498 → 0.5181504 | 0.0093 → 0.03160798 | -3.251 → -3.296236 |
| Joint MLP shallow | 0.0519 → 0.7191336 | 0.0109 → 0.01458358 | -3.235 → -3.283027 |
| Joint MLP deep | 0.0484 → 0.5792844 | 0.0089 → 0.02745206 | -3.253 → -3.298848 |

Partial agreement in NLL ranking: deep MLP and ARFF are close, shallow MLP is worse. Absolute coefficient errors disagree substantially: ARFF drift is7.2× the historical value; Fourier10.4×, shallow13.9×, deep12.0×. Covariance errors also increase. The historical claim of uniformly similar coefficient accuracy is not reproduced.

ARFF validation SPD: 0/30 seeds affected; mean/median/max rate 0%/0%/0%; minimum raw eigenvalue across seeds 0.073098965.

## ex2

| Estimator | Parameters | Drift RMSE | Raw covariance RMSE | Validation NLL |
|---|---:|---:|---:|---:|
| ARFF | 3,072 | 0.035574252 ± 0.0095537 | 0.00048769766 ± 8.0265e-05 | -11.602973 ± 0.01329 |
| Joint Fourier Adam | 3,072 | 0.27313932 ± 0.010137 | 0.00079676949 ± 6.2663e-05 | -11.583691 ± 0.0021168 |
| Joint MLP shallow | 2,185 | 0.023654203 ± 0.00019282 | 0.00027273508 ± 6.8627e-06 | -11.609893 ± 0.00065731 |
| Joint MLP deep | 9,417 | 0.02760299 ± 0.0012758 | 0.00017955099 ± 9.2957e-06 | -11.604817 ± 0.0011673 |

Historical → reproduced means:

| Estimator | Drift | Historical Diff. → raw covariance | NLL |
|---|---:|---:|---:|
| ARFF | 0.0072 → 0.03557425 | 0.0019 → 0.0004876977 | -11.613 → -11.60297 |
| Joint Fourier Adam | 0.0089 → 0.2731393 | 0.0022 → 0.0007967695 | -11.593 → -11.58369 |
| Joint MLP shallow | 0.0068 → 0.0236542 | 0.0018 → 0.0002727351 | -11.625 → -11.60989 |
| Joint MLP deep | 0.0065 → 0.02760299 | 0.0017 → 0.000179551 | -11.629 → -11.60482 |

Only partial agreement. Adam MLPs retain strong coefficient accuracy, but Fourier drift is30.7× the historical reported value and much worse than the other reproduced estimators. ARFF drift is4.9× historical. Shallow MLP now has the lowest mean NLL, while historical deep MLP was best. Current covariance errors are smaller than the reported historical Diff. column; verify metric semantics before calling this improvement.

ARFF validation SPD: 0/30 seeds affected; mean/median/max rate 0%/0%/0%; minimum raw eigenvalue across seeds 2.8563827e-05.

## ex3

| Estimator | Parameters | Drift RMSE | Raw covariance RMSE | Validation NLL |
|---|---:|---:|---:|---:|
| ARFF | 153,600 | 1.5291694 ± 0.27047 | 0.0071728582 ± 0.00097018 | -22.805261 ± 0.95994 |
| Joint MLP shallow | 89,153 | 2.4277797 ± 6.5421e-06 | 0.014177489 ± 6.7665e-06 | -20.735642 ± 3.7963e-05 |

Historical → reproduced means:

| Estimator | Drift | Historical Diff. → raw covariance | NLL |
|---|---:|---:|---:|
| ARFF | 0.0413 → 1.529169 | 0.0215 → 0.007172858 | -25.484 → -22.80526 |
| Joint MLP shallow | 0.0376 → 2.42778 | 0.0188 → 0.01417749 | -25.75 → -20.73564 |

Not reproduced quantitatively or qualitatively in ranking: historical shallow MLP beat ARFF in NLL/drift, whereas reproduced ARFF is better. ARFF drift is37.0× historical; MLP64.6×. All30 MLP runs select zero-based epoch118 and have almost identical poor drift errors. This warrants source/initialization/optimizer/checkpoint and metric-protocol investigation, not tuning. ARFF drift median1.57545, range1.15615–2.05432; NLL median-22.8896, range-24.3020–-20.6204. No numerical execution failed.

ARFF validation SPD: 0/30 seeds affected; mean/median/max rate 0%/0%/0%; minimum raw eigenvalue across seeds 0.010213208.

## ex5

| Estimator | Parameters | Drift RMSE | Raw covariance RMSE | Validation NLL |
|---|---:|---:|---:|---:|
| ARFF | 3,072 | 0.0026308717 ± 0.0003076 | 4.9927594e-06 ± 9.0296e-07 | -11.975527 ± 0.032563 |
| Joint Fourier Adam | 3,072 | 0.067464635 ± 0.00014619 | 0.0010172799 ± 5.6594e-05 | -11.702746 ± 0.0010345 |
| Joint MLP shallow | 2,564 | 0.067665332 ± 5.8727e-05 | 0.0011062912 ± 9.235e-06 | -11.699443 ± 0.00018838 |
| Joint MLP deep | 34,308 | 0.044873987 ± 0.0033443 | 0.00028673622 ± 1.313e-05 | -11.794944 ± 0.0068171 |

Historical → reproduced means:

| Estimator | Drift | Historical Diff. → raw covariance | NLL |
|---|---:|---:|---:|
| ARFF | 0.0143 → 0.002630872 | 0.0008 → 4.992759e-06 | -9.458 → -11.97553 |
| Joint Fourier Adam | 0.0121 → 0.06746463 | 0.0007 → 0.00101728 | -9.529 → -11.70275 |
| Joint MLP shallow | 0.0125 → 0.06766533 | 0.0007 → 0.001106291 | -9.519 → -11.69944 |
| Joint MLP deep | 0.0127 → 0.04487399 | 0.0008 → 0.0002867362 | -9.501 → -11.79494 |

Historical near-parity between methods does not reproduce under the accepted fixed-lag SSA/cross-fitting/SPD protocol. ARFF now has the lowest drift/covariance RMSE and NLL; its drift is about5.4× lower than historical, whereas Adam drift is3.5–5.6× higher. NLL changes substantially for every method. Dataset/observation corrections make old/new NLL comparisons diagnostic rather than evidence of pure optimizer improvement. Raw SPD violations require explicit disclosure.

ARFF validation SPD: 29/30 seeds affected; mean/median/max rate 5.20108%/4.31251%/12.3257%; minimum raw eigenvalue across seeds -6.6074081e-06.

## ex6

| Estimator | Parameters | Drift RMSE | Raw covariance RMSE | Validation NLL |
|---|---:|---:|---:|---:|
| ARFF | 4,096 | 0.28226183 ± 0.036915 | 1.9037374e-05 ± 3.2525e-06 | -9.4148547 ± 5.8573e-05 |
| Joint Fourier Adam | 4,096 | 3.2669828 ± 0.0025938 | 9.5605081e-05 ± 1.962e-05 | -9.4104813 ± 0.00040828 |
| Joint MLP shallow | 4,098 | 1.3145615 ± 1.1175 | 0.00012238919 ± 0.00013418 | -9.4089428 ± 0.011717 |
| Joint MLP deep | 133,634 | 3.0024801 ± 0.82305 | 2.4943001e-05 ± 2.8573e-06 | -9.4119848 ± 0.00098475 |

Historical → reproduced means:

| Estimator | Drift | Historical Diff. → raw covariance | NLL |
|---|---:|---:|---:|
| ARFF | 0.0098 → 0.2822618 | 0.0003 → 1.903737e-05 | -9.413 → -9.414855 |
| Joint Fourier Adam | 0.0101 → 3.266983 | 0.0003 → 9.560508e-05 | -9.403 → -9.410481 |
| Joint MLP shallow | 0.0104 → 1.314561 | 0.0003 → 0.0001223892 | -9.395 → -9.408943 |
| Joint MLP deep | 0.0097 → 3.00248 | 0.0003 → 2.4943e-05 | -9.412 → -9.411985 |

Qualitative agreement in mean NLL: ARFF is lowest, with deep MLP close. Coefficient recovery does not reproduce: ARFF drift28.8× historical; Adam variants roughly126–324×. All120 produced artifacts use data/ex6_labels_v2.npz, SHA25647c4a91d37f494621c4174e5c8a87d951a9b78b16b1da68dc4b146f72dbc85c3. This matches the corrected file. Shallow MLP drift median.493996, range.345056–3.275878; covariance median3.38597e-5, range3.02443e-5–5.10574e-4. Deep MLP drift median3.27205, range.455541–3.27249: its poor mean is not a single outlier. NLL proximity does not establish drift recovery.

ARFF validation SPD: 0/30 seeds affected; mean/median/max rate 0%/0%/0%; minimum raw eigenvalue across seeds 0.00052193733.

## ex7

| Estimator | Parameters | Drift RMSE | Raw covariance RMSE | Validation NLL |
|---|---:|---:|---:|---:|
| ARFF | 6,144 | 0.10568924 ± 0.0079133 | 0.00011555281 ± 1.1753e-05 | -8.6859209 ± 0.00028458 |
| Joint Fourier Adam | 6,144 | 0.15927461 ± 0.0024901 | 0.00014106203 ± 2.1498e-06 | -8.6863171 ± 0.00017716 |
| Joint MLP shallow | 5,124 | 0.19619783 ± 1.7881e-05 | 7.8442342e-05 ± 1.2532e-07 | -8.6849489 ± 2.3558e-06 |
| Joint MLP deep | 134,148 | 0.10038902 ± 0.014237 | 0.00015760137 ± 5.6518e-05 | -8.6868093 ± 0.00011536 |

Historical → reproduced means:

| Estimator | Drift | Historical Diff. → raw covariance | NLL |
|---|---:|---:|---:|
| ARFF | 0.0287 → 0.1056892 | 0.0011 → 0.0001155528 | -6.374 → -8.685921 |
| Joint Fourier Adam | 0.0305 → 0.1592746 | 0.0013 → 0.000141062 | -6.354 → -8.686317 |
| Joint MLP shallow | 0.0297 → 0.1961978 | 0.0012 → 7.844234e-05 | -6.361 → -8.684949 |
| Joint MLP deep | 0.0283 → 0.100389 | 0.0011 → 0.0001576014 | -6.374 → -8.686809 |

Partial qualitative agreement: deep MLP and ARFF remain the best drift estimators and close in NLL, but shallow MLP is poorest in drift. Drift errors are3.5–6.6× historical. Every NLL is about2.31 lower than the manuscript, despite matching the stated h=.001. Investigate historical likelihood scale/normalization/data provenance; proximity of the shift to ln(10) is a clue, not proof of a lag error and not permission to change h.

ARFF validation SPD: 0/30 seeds affected; mean/median/max rate 0%/0%/0%; minimum raw eigenvalue across seeds 0.0092772134.

## Experiment5 raw SPD detail

Validation:29/30 seeds affected; mean5.201083%, median4.312506%, maximum12.325745%. Minimum raw eigenvalue-6.607408e-6; median seed-minimum-2.537128e-6; seed-minimum range[-6.607408e-6,1.366586e-6]. Training:29/30 seeds affected; mean5.197182%, median4.319873%, maximum12.109462%; worst raw eigenvalue-7.390510e-6. Test statistics do not exist: these runs intentionally have no test split.

The archived coefficient RMSE is raw/pre-projection. The production runner calls existing gaussian_nll, which symmetrizes/eigenvalue-floors Sigma at epsilon=1e-6 before multiplying by h, inversion and log determinant. No raw-indefinite covariance is used for NLL. Negative raw predictions are preserved, not filtered; tiny absolute negative eigenvalues still invalidate raw covariance and must not be called SPD. These checks concern NLL only; no new trajectory simulation was run.

## Experiment8 — frozen accepted comparison

| Method | Parameters | Test drift RMSE | Test covariance RMSE | Test NLL |
|---|---:|---:|---:|---:|
| Joint Fourier Adam | 1,792 | 0.89410867 ± 0.073995 | 0.71540701 ± 0.15045 | -8.3466522 ± 0.064269 |
| Split Fourier Adam | 1,792 | 0.6628618 ± 0.043253 | 0.617333 ± 0.092378 | -8.3375147 ± 0.06084 |
| ARFF | 1,792 | 0.94498191 ± 0.28984 | 0.11383387 ± 0.002534 | -8.87375 ± 0.097421 |
| Joint MLP Adam | 1,814 | 0.30621254 ± 0.058301 | 0.26534329 ± 0.081204 | -10.018567 ± 0.18763 |

ARFF has best covariance RMSE; joint MLP has best drift RMSE and NLL. Split Fourier improves both coefficient means relative to joint Fourier but has slightly worse mean NLL. This replaces neither the accepted artifacts nor their scientific definition with historical settings.

New label-only plot: results/capacity_matched_ex8_w27/ex8_capacity_matched_w27_rmse_two_panel_publication_labels.pdf and .png. Every plotted metric vector exactly matches frozen test_distributions.npz; jitter, boxes, means, scales and title unchanged. Visual inspection confirms labels and footer are fully visible; PDF contains vector graphics, no raster-image objects.

## Overall assessment

Experiment8 is ready for the accepted accuracy comparison. Experiment5 can supply a corrected-protocol numerical table with explicit fixed-lag/cross-fitting/SPD provenance, not a claim that the old table was replicated. All other completed measurements can enter a clearly marked review draft, but coefficient-metric reconciliation and the large discrepancies should be resolved before final numerical-reproduction claims. No isolated runtime/convergence-speed claim is supported by these parallel timings. Experiment4, historical split MLP, Section6.2 and the unrecovered trajectory protocol remain unresolved; no replacement settings are proposed.

| Experiment | Reproduction status | Agrees with Owen? | Blocker remaining? | Paper-ready? |
|---|---|---|---|---|
|1|30×4 complete|NLL ranking partly; coefficient magnitudes no|Historical metric/protocol reconciliation|Review draft; not final reproduction claim|
|2|30×4 complete|Partial; Fourier drift discrepancy|Historical metric/protocol reconciliation|Review draft; not final reproduction claim|
|3|30×2 complete|No; ranking and magnitudes differ|Large error and near-identical MLP outcomes|No final sign-off|
|4|Not dispatched|Undetermined|Drift sign and conditioning|No|
|5|30×4 corrected protocol complete|Historical near-parity no|Raw-SPD disclosure; old/new protocol distinction|Yes, as corrected result with caveats|
|6|30×4 on accepted labels_v2|Mean NLL partly; coefficient errors no|Metric/protocol reconciliation and MLP variability|Review draft; not final reproduction claim|
|7|30×4 complete|Relative drift ranking partly; absolute results no|NLL offset and coefficient discrepancies|Review draft; not final reproduction claim|
|8|Frozen120-run capacity-matched comparison|Accepted exception, not historical-width replication|None for accepted accuracy panel|Yes; no timing claim|
