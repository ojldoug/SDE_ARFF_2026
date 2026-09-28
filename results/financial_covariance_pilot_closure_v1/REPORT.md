# Closure diagnostic: where the financial Gaussian score fails

Both pilots are closed at their existing scope. This CPU-only calculation reads archived financial raw forecasts, returns and score arrays; it performs no fitting, checkpoint inference, tuning, data acquisition or new application study. The original scores, floor, selections, reports and failed checks are unchanged.

For the common zero-mean model, each physical-return score is

$$\ell_t=L_t+Q_t+C,\qquad L_t=\tfrac12\log\det H_t,\quad Q_t=\tfrac12 r_t^\top H_t^{-1}r_t,\quad C=\tfrac32\log(2\pi)=2.756815600.$$

Here $H_t=D U_t\operatorname{diag}(\max(\lambda_{ti},10^{-4}))U_t^\top D$, with the original fitting-only return scale $D$ and raw symmetrized scaled eigenvalues $\lambda_{ti}$. Thus $L_t$ includes $\sum_i\log D_i$: all scores are in the same physical decimal-log-return coordinates. Floor activation means any raw eigenvalue is **strictly below** $10^{-4}$; it includes non-SPD forecasts and positive but sub-floor eigenvalues.

For a group $G$, the within-group mean is $\bar L_G=\sum_{t\in G}L_t/n_G$; its contribution to the **overall** mean is $(n_G/N)\bar L_G$. The same weighting applies to $Q$, $C$ and archived NLL. Empty groups have zero contribution and undefined conditional means (blank CSV cells), not zero conditional loss. Each seed is summarized separately on the same 8,760 test hours; the counts must not be treated as independent seed-by-hour replications.

## Selected ARFF: conditional severity and frequency

All three models use validation-selected $\lambda=.064$. Values below are conditional means within each indicated test group; the constant is 2.756816 in every nonempty group.

| Seed | Floor active? | Count / 8,760 | Fraction | Mean L | Mean Q | Mean archived NLL |
|---|---|---:|---:|---:|---:|---:|
| 0 | Yes | 1,891 | 21.587% | -20.837784 | 692.727581 | 674.646612 |
| 0 | No | 6,869 | 78.413% | -16.679728 | 2.642761 | -11.280152 |
| 1 | Yes | 2,363 | 26.975% | -20.621357 | 879.217522 | 861.352980 |
| 1 | No | 6,397 | 73.025% | -16.810305 | 3.451953 | -10.601537 |
| 2 | Yes | 2,703 | 30.856% | -20.828592 | 674.459689 | 656.387912 |
| 2 | No | 6,057 | 69.144% | -16.534696 | 2.168474 | -11.609406 |

The corresponding contributions to the **overall** test mean are:

| Seed | Floor active? | Weighted L | Weighted Q | Weighted C | Weighted archived NLL |
|---|---|---:|---:|---:|---:|
| 0 | Yes | -4.498202 | 149.537426 | 0.595107 | 145.634331 |
| 0 | No | -13.079116 | 2.072274 | 2.161708 | -8.845133 |
| 1 | Yes | -5.562588 | 237.167923 | 0.743648 | 232.348983 |
| 1 | No | -12.275745 | 2.520793 | 2.013168 | -7.741784 |
| 2 | Yes | -6.426905 | 208.112390 | 0.850648 | 202.536133 |
| 2 | No | -11.432723 | 1.499366 | 1.906168 | -8.027189 |

Adding the two groups recovers the original seed scores **136.789199, 224.607199 and 194.508944**. Floor-active forecasts supply **98.633%, 98.948% and 99.285%** of the total quadratic contribution. These are frequent failures, not a handful of negligible observations. Of the floor-active counts 1,891/2,363/2,703, respectively 1,887/2,361/2,701 have nonpositive raw minimum eigenvalues; only 4/2/2 are positive but below the floor.

The large positive penalty is the quadratic term. The floor-active log-determinant contributions are negative and partially offset that penalty. The non-floor group has much smaller conditional quadratic means (2.17–3.45), although the Gaussian nominal value for Q is 1.5. These method-dependent subsets are not a common comparison population: their conditional scores must not be compared directly with a baseline's full-test score.

## Controls and diagonal ablation

Constant, EWMA, DCC and all three neural forecasts have **zero floor-active hours** in both validation and test. Their test no-floor groups each contain all 8,760 hours, so conditional and overall contributions coincide:

| Forecast | Mean/weighted L | Mean/weighted Q | Mean/weighted C | Archived NLL |
|---|---:|---:|---:|---:|
| constant | -16.307964 | 1.745416 | 2.756816 | -11.805733 |
| ewma | -16.914308 | 1.768354 | 2.756816 | -12.389138 |
| dcc | -16.716098 | 1.539923 | 2.756816 | -12.419359 |
| neural_s0 | -16.552449 | 1.373466 | 2.756816 | -12.422168 |
| neural_s1 | -16.532807 | 1.396151 | 2.756816 | -12.379841 |
| neural_s2 | -16.568422 | 1.399059 | 2.756816 | -12.412547 |

The archived diagonal-only ARFF ablation also retains the same floor. Its floor-active counts and weighted contributions are:

| Seed | Floor active? | Count | Weighted L | Weighted Q | Weighted C | Weighted archived NLL |
|---|---|---:|---:|---:|---:|---:|
| 0 | Yes | 393 | -0.984169 | 103.390189 | 0.123679 | 102.529699 |
| 0 | No | 8,367 | -14.736052 | 1.364486 | 2.633137 | -10.738430 |
| 1 | Yes | 717 | -1.736113 | 159.037525 | 0.225643 | 157.527056 |
| 1 | No | 8,043 | -14.394231 | 1.657060 | 2.531172 | -10.205999 |
| 2 | Yes | 1,135 | -2.697776 | 147.937577 | 0.357190 | 145.596991 |
| 2 | No | 7,625 | -13.494154 | 1.743431 | 2.399625 | -9.351098 |

The CSV provides conditional means as well as weighted contributions for **all 12 forecasts, both splits and both floor groups**, plus overall rows (72 rows total). Reducing the activation frequency does not remove the diagonal ablation's poor joint score. The earlier fixed-portfolio results remain unchanged and need not rank the forecasts in the same order as the joint score.

## What is established—and what is not

**All nine ARFF candidates—three ridges × three fitting seeds—selected post-adaptation iteration 1.** This is recorded in both candidate_validation.json and checkpoint_summary.csv. All 300 iterations were executed, but the later 299 iterations do not enter the retained predictions. Prolonged adaptation is therefore **not an established explanation for the selected-model failure**. The first checkpoint has already undergone a random proposal followed by target-dependent Metropolis acceptance and amplitude fitting; this is not evidence about an entirely nonadaptive basis.

This fixed exploratory application establishes poor predictive calibration and frequent raw covariance invalidity for the tested unconstrained ARFF conditional-second-moment procedure, under its specified features, finite validation budget, checkpoint rule and scoring floor. The original monthly results show underperformance against constant covariance, EWMA and DCC for every selected seed in every test month. It does not establish failure of every ARFF formulation, coefficient-recovery error, or continuous-time SDE identification.

The decomposition locates the score penalty in forecasts requiring covariance repair; it does **not** identify why those raw forecasts were learned. Contributions of target variability/heavy tails, finite-sample estimation, feature representation, first-step adaptation, regularization, regime change and MSE-versus-NLL selection remain unresolved. No alternative floor was evaluated. For indefinite raw forecasts an unprojected Gaussian score is not valid, so this analysis does not measure a causal penalty of projection or demonstrate that changing the floor would repair calibration.

The colloidal float64 correction still leaves the previous scientific conclusion unchanged; its original strict-check failures and schema/cell/optical-phase limitations remain documented. Retain that pilot as an archived supplementary candidate and the financial pilot as an archival negative result, potentially useful for a concise limitations discussion. Neither warrants automatic manuscript inclusion or another application search.

## Verification and reproduction

Every decomposed per-hour score equals its archived score numerically (maximum absolute difference **0.0** across both splits and all forecasts). Group contributions sum to the original means within the unchanged financial tolerance rtol=atol=1e-9. Floor counts exactly match archived projection frequencies. Input hashes match the pre-existing artifact index and remain unchanged after analysis; the original colloidal failed-validation record and corrected summary are also hash-preserved. The original tolerances have not been relaxed.

Files: `components_by_floor.csv` (full numeric table), `summary.json` (checks, hashes and all aggregates), and `PROPOSED_LIMITATIONS.md` (author-review text only). Script: `scripts/diagnose_financial_pilot_nll_components.py`. From the repository root, with the archived inputs present and the new output directory absent:

```bash
OPENBLAS_NUM_THREADS=1 /path/to/recorded-environment/bin/python -B scripts/diagnose_financial_pilot_nll_components.py
```

The executed Python was `../bounded_reproduction_verification_v1/venv/bin/python` relative to the repository. The script uses NumPy on CPU, refuses to overwrite its output directory, and does not initialize JAX or load models. The report is generated from those aggregates by `scripts/report_financial_pilot_closure.py` (no numerical evaluation). No manuscript was edited and no commit, push or follow-up run was performed for this closure.
