"""Write the short closure report from saved decomposition aggregates only."""
from pathlib import Path
import json
p=Path(__file__).resolve().parents[1]/'results/financial_covariance_pilot_closure_v1'
s=json.loads((p/'summary.json').read_text());rows=s['rows']
def row(method,group='all',split='test'):
 return next(r for r in rows if (r['split'],r['method'],r['group'])==(split,method,group))
lines=[
'# Closure diagnostic: where the financial Gaussian score fails',
'',
'Both pilots are closed at their existing scope. This CPU-only calculation reads archived financial raw forecasts, returns and score arrays; it performs no fitting, checkpoint inference, tuning, data acquisition or new application study. The original scores, floor, selections, reports and failed checks are unchanged.',
'',
'For the common zero-mean model, each physical-return score is',
'',
r'$$\ell_t=L_t+Q_t+C,\qquad L_t=\tfrac12\log\det H_t,\quad Q_t=\tfrac12 r_t^\top H_t^{-1}r_t,\quad C=\tfrac32\log(2\pi)=2.756815600.$$',
'',
r'Here $H_t=D U_t\operatorname{diag}(\max(\lambda_{ti},10^{-4}))U_t^\top D$, with the original fitting-only return scale $D$ and raw symmetrized scaled eigenvalues $\lambda_{ti}$. Thus $L_t$ includes $\sum_i\log D_i$: all scores are in the same physical decimal-log-return coordinates. Floor activation means any raw eigenvalue is **strictly below** $10^{-4}$; it includes non-SPD forecasts and positive but sub-floor eigenvalues.',
'',
r'For a group $G$, the within-group mean is $\bar L_G=\sum_{t\in G}L_t/n_G$; its contribution to the **overall** mean is $(n_G/N)\bar L_G$. The same weighting applies to $Q$, $C$ and archived NLL. Empty groups have zero contribution and undefined conditional means (blank CSV cells), not zero conditional loss. Each seed is summarized separately on the same 8,760 test hours; the counts must not be treated as independent seed-by-hour replications.',
'',
'## Selected ARFF: conditional severity and frequency',
'',
r'All three models use validation-selected $\lambda=.064$. Values below are conditional means within each indicated test group; the constant is 2.756816 in every nonempty group.',
'',
'| Seed | Floor active? | Count / 8,760 | Fraction | Mean L | Mean Q | Mean archived NLL |',
'|---|---|---:|---:|---:|---:|---:|']
for seed in range(3):
 for group,label in [('floor_active','Yes'),('no_floor','No')]:
  r=row(f'arff_s{seed}',group)
  lines.append(f'| {seed} | {label} | {r["count"]:,} | {r["fraction"]:.3%} | {r["conditional_logdet_half"]:.6f} | {r["conditional_quadratic_half"]:.6f} | {r["conditional_archived_nll"]:.6f} |')
lines += ['', 'The corresponding contributions to the **overall** test mean are:', '',
'| Seed | Floor active? | Weighted L | Weighted Q | Weighted C | Weighted archived NLL |',
'|---|---|---:|---:|---:|---:|']
for seed in range(3):
 for group,label in [('floor_active','Yes'),('no_floor','No')]:
  r=row(f'arff_s{seed}',group)
  lines.append(f'| {seed} | {label} | {r["overall_logdet_contribution"]:.6f} | {r["overall_quadratic_contribution"]:.6f} | {r["overall_constant_contribution"]:.6f} | {r["overall_archived_nll_contribution"]:.6f} |')
lines += ['',
'Adding the two groups recovers the original seed scores **136.789199, 224.607199 and 194.508944**. Floor-active forecasts supply **98.633%, 98.948% and 99.285%** of the total quadratic contribution. These are frequent failures, not a handful of negligible observations. Of the floor-active counts 1,891/2,363/2,703, respectively 1,887/2,361/2,701 have nonpositive raw minimum eigenvalues; only 4/2/2 are positive but below the floor.',
'',
'The large positive penalty is the quadratic term. The floor-active log-determinant contributions are negative and partially offset that penalty. The non-floor group has much smaller conditional quadratic means (2.17–3.45), although the Gaussian nominal value for Q is 1.5. These method-dependent subsets are not a common comparison population: their conditional scores must not be compared directly with a baseline\'s full-test score.',
'',
'## Controls and diagonal ablation',
'',
'Constant, EWMA, DCC and all three neural forecasts have **zero floor-active hours** in both validation and test. Their test no-floor groups each contain all 8,760 hours, so conditional and overall contributions coincide:',
'',
'| Forecast | Mean/weighted L | Mean/weighted Q | Mean/weighted C | Archived NLL |',
'|---|---:|---:|---:|---:|']
for method in ['constant','ewma','dcc','neural_s0','neural_s1','neural_s2']:
 r=row(method)
 for split in ['validation','test']:assert row(method,'floor_active',split)['count']==0
 lines.append(f'| {method} | {r["overall_logdet_contribution"]:.6f} | {r["overall_quadratic_contribution"]:.6f} | {r["overall_constant_contribution"]:.6f} | {r["overall_archived_nll_contribution"]:.6f} |')
lines += ['',
'The archived diagonal-only ARFF ablation also retains the same floor. Its floor-active counts and weighted contributions are:', '',
'| Seed | Floor active? | Count | Weighted L | Weighted Q | Weighted C | Weighted archived NLL |',
'|---|---|---:|---:|---:|---:|---:|']
for seed in range(3):
 for group,label in [('floor_active','Yes'),('no_floor','No')]:
  r=row(f'arff_s{seed}_diagonal',group)
  lines.append(f'| {seed} | {label} | {r["count"]:,} | {r["overall_logdet_contribution"]:.6f} | {r["overall_quadratic_contribution"]:.6f} | {r["overall_constant_contribution"]:.6f} | {r["overall_archived_nll_contribution"]:.6f} |')
lines += ['',
'The CSV provides conditional means as well as weighted contributions for **all 12 forecasts, both splits and both floor groups**, plus overall rows (72 rows total). Reducing the activation frequency does not remove the diagonal ablation\'s poor joint score. The earlier fixed-portfolio results remain unchanged and need not rank the forecasts in the same order as the joint score.',
'',
'## What is established—and what is not',
'',
'**All nine ARFF candidates—three ridges × three fitting seeds—selected post-adaptation iteration 1.** This is recorded in both candidate_validation.json and checkpoint_summary.csv. All 300 iterations were executed, but the later 299 iterations do not enter the retained predictions. Prolonged adaptation is therefore **not an established explanation for the selected-model failure**. The first checkpoint has already undergone a random proposal followed by target-dependent Metropolis acceptance and amplitude fitting; this is not evidence about an entirely nonadaptive basis.',
'',
'This fixed exploratory application establishes poor predictive calibration and frequent raw covariance invalidity for the tested unconstrained ARFF conditional-second-moment procedure, under its specified features, finite validation budget, checkpoint rule and scoring floor. The original monthly results show underperformance against constant covariance, EWMA and DCC for every selected seed in every test month. It does not establish failure of every ARFF formulation, coefficient-recovery error, or continuous-time SDE identification.',
'',
'The decomposition locates the score penalty in forecasts requiring covariance repair; it does **not** identify why those raw forecasts were learned. Contributions of target variability/heavy tails, finite-sample estimation, feature representation, first-step adaptation, regularization, regime change and MSE-versus-NLL selection remain unresolved. No alternative floor was evaluated. For indefinite raw forecasts an unprojected Gaussian score is not valid, so this analysis does not measure a causal penalty of projection or demonstrate that changing the floor would repair calibration.',
'',
'The colloidal float64 correction still leaves the previous scientific conclusion unchanged; its original strict-check failures and schema/cell/optical-phase limitations remain documented. Retain that pilot as an archived supplementary candidate and the financial pilot as an archival negative result, potentially useful for a concise limitations discussion. Neither warrants automatic manuscript inclusion or another application search.',
'',
'## Verification and reproduction',
'',
'Every decomposed per-hour score equals its archived score numerically (maximum absolute difference **0.0** across both splits and all forecasts). Group contributions sum to the original means within the unchanged financial tolerance rtol=atol=1e-9. Floor counts exactly match archived projection frequencies. Input hashes match the pre-existing artifact index and remain unchanged after analysis; the original colloidal failed-validation record and corrected summary are also hash-preserved. The original tolerances have not been relaxed.',
'',
'Files: `components_by_floor.csv` (full numeric table), `summary.json` (checks, hashes and all aggregates), and `PROPOSED_LIMITATIONS.md` (author-review text only). Script: `scripts/diagnose_financial_pilot_nll_components.py`. From the repository root, with the archived inputs present and the new output directory absent:',
'',
'```bash',
'OPENBLAS_NUM_THREADS=1 /path/to/recorded-environment/bin/python -B scripts/diagnose_financial_pilot_nll_components.py',
'```',
'',
'The executed Python was `../bounded_reproduction_verification_v1/venv/bin/python` relative to the repository. The script uses NumPy on CPU, refuses to overwrite its output directory, and does not initialize JAX or load models. The report is generated from those aggregates by `scripts/report_financial_pilot_closure.py` (no numerical evaluation). No manuscript was edited and no commit, push or follow-up run was performed for this closure.',
]
(p/'REPORT.md').write_text('\n'.join(lines)+'\n')
paragraph='''# Proposed limitations paragraph — author review only

Suggested placement: the existing applications discussion or limitations paragraph, without changing the method-and-applications structure. This text has not been inserted into either manuscript.

> Two bounded exploratory pilots did not establish an additional real-data advantage for the method. In the colloidal application, common float64 reevaluation preserved the near-equal predictive scores; unresolved trajectory-schema, experimental-cell and optical-phase information limits interpretation. In the fixed-basket financial application, unconstrained ARFF conditional-second-moment regression produced frequent non-SPD forecasts and poorer joint Gaussian scores than constant covariance, EWMA and DCC. Forecasts requiring the prescribed eigenvalue floor accounted for approximately 99% of the quadratic score contribution, while the log-determinant term partly offset this penalty. All nine ARFF candidates selected the first post-adaptation checkpoint, so prolonged adaptation is not an established explanation for the retained models' failure. These results concern a fixed feature set, validation budget and scoring convention; they neither identify the cause of the raw covariance errors nor test whether another floor would resolve them. Shared-data fitting seeds do not provide independent application replications, and the discrete-time financial experiment does not establish instantaneous SDE coefficient recovery.
'''
(p/'PROPOSED_LIMITATIONS.md').write_text(paragraph)
(p/'COMPLETE.json').write_text(json.dumps(dict(complete=True,scope='Archived prediction/score decomposition and pilot closure only',new_training=False,new_model_predictions=False,new_search=False,manuscript_edited=False),indent=2)+'\n')
