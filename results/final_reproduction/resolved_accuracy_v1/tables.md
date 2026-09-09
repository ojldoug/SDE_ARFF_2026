# Resolved accuracy results

Experiments1–7: canonical validation, no independent test. Experiment8: frozen accepted test results.30 seeds per method; SD uses ddof=1. Non-isolated timings are not controlled runtime results. Experiment4 and unresolved historical split-MLP/Section6.2 studies are excluded.

| Experiment | Method | Scope | Parameters | Drift RMSE mean±SD | Covariance RMSE mean±SD | NLL mean±SD |
|---|---|---|---:|---:|---:|---:|
| ex1 | fourier | validation | 3072 | 0.5181504 ± 0.02310338 | 0.03160798 ± 0.0007871516 | -3.296236 ± 0.0006644653 |
| ex1 | mlp_shallow | validation | 2564 | 0.7191336 ± 0.02699695 | 0.01458358 ± 0.0001736462 | -3.283027 ± 0.001037408 |
| ex1 | mlp_deep | validation | 34308 | 0.5792844 ± 0.02042453 | 0.02745206 ± 0.0007586022 | -3.298848 ± 0.0009552484 |
| ex2 | fourier | validation | 3072 | 0.2731393 ± 0.01013738 | 0.0007967695 ± 6.266303e-05 | -11.58369 ± 0.002116832 |
| ex2 | mlp_shallow | validation | 2185 | 0.0236542 ± 0.0001928209 | 0.0002727351 ± 6.862711e-06 | -11.60989 ± 0.0006573094 |
| ex2 | mlp_deep | validation | 9417 | 0.02760299 ± 0.001275841 | 0.000179551 ± 9.295729e-06 | -11.60482 ± 0.00116735 |
| ex3 | mlp_shallow | validation | 89153 | 2.42778 ± 6.542103e-06 | 0.01417749 ± 6.766525e-06 | -20.73564 ± 3.796347e-05 |
| ex5 | fourier | validation | 3072 | 0.06746463 ± 0.0001461912 | 0.00101728 ± 5.659368e-05 | -11.70275 ± 0.001034461 |
| ex5 | mlp_shallow | validation | 2564 | 0.06766533 ± 5.872683e-05 | 0.001106291 ± 9.235015e-06 | -11.69944 ± 0.0001883783 |
| ex5 | mlp_deep | validation | 34308 | 0.04487399 ± 0.003344299 | 0.0002867362 ± 1.313002e-05 | -11.79494 ± 0.006817083 |
| ex7 | fourier | validation | 6144 | 0.1592746 ± 0.002490115 | 0.000141062 ± 2.149786e-06 | -8.686317 ± 0.0001771563 |
| ex7 | mlp_shallow | validation | 5124 | 0.1961978 ± 1.788127e-05 | 7.844234e-05 ± 1.253192e-07 | -8.684949 ± 2.355846e-06 |
| ex7 | mlp_deep | validation | 134148 | 0.100389 ± 0.01423686 | 0.0001576014 ± 5.651765e-05 | -8.686809 ± 0.000115363 |
| ex6 | fourier | validation | 4096 | 3.266983 ± 0.002593832 | 9.560508e-05 ± 1.961969e-05 | -9.410481 ± 0.0004082798 |
| ex6 | mlp_shallow | validation | 4098 | 1.314561 ± 1.117474 | 0.0001223892 ± 0.0001341849 | -9.408943 ± 0.01171735 |
| ex6 | mlp_deep | validation | 133634 | 3.00248 ± 0.823048 | 2.4943e-05 ± 2.857313e-06 | -9.411985 ± 0.00098475 |
| ex1 | arff_historical_corrected | validation | 3072 | 0.3519488 ± 0.02318108 | 0.02414908 ± 0.00347173 | -3.295346 ± 0.001572432 |
| ex2 | arff_historical_corrected | validation | 3072 | 0.03557425 ± 0.009553664 | 0.0004876977 ± 8.026548e-05 | -11.60297 ± 0.0132898 |
| ex3 | arff_historical_corrected | validation | 153600 | 1.529169 ± 0.270471 | 0.007172858 ± 0.0009701846 | -22.80526 ± 0.9599436 |
| ex5 | arff_historical_corrected | validation | 3072 | 0.002630872 ± 0.0003075996 | 4.992759e-06 ± 9.029617e-07 | -11.97553 ± 0.03256305 |
| ex7 | arff_historical_corrected | validation | 6144 | 0.1056892 ± 0.007913315 | 0.0001155528 ± 1.175285e-05 | -8.685921 ± 0.0002845774 |
| ex6 | arff_historical_corrected | validation | 4096 | 0.2822618 ± 0.03691508 | 1.903737e-05 ± 3.252521e-06 | -9.414855 ± 5.857331e-05 |
| ex8 | joint | test | 1792 | 0.8941087 ± 0.07399532 | 0.715407 ± 0.150445 | -8.346652 ± 0.06426917 |
| ex8 | split | test | 1792 | 0.6628618 ± 0.04325289 | 0.617333 ± 0.09237801 | -8.337515 ± 0.06083971 |
| ex8 | arff | test | 1792 | 0.9449819 ± 0.2898358 | 0.1138339 ± 0.00253401 | -8.87375 ± 0.09742135 |
| ex8 | mlp | test | 1814 | 0.3062125 ± 0.05830111 | 0.2653433 ± 0.08120428 | -10.01857 ± 0.1876337 |
