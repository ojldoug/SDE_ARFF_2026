"""All-method and paired summaries from frozen saved forecasts; no fitting."""
import json,csv,datetime as dt
from pathlib import Path
import numpy as np
from scipy.stats import norm
from .data import P
from .evaluation import evaluate

def csvwrite(name,rows):
 with (P/name).open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
def report():
 rows=[];monthly=[];portfolios=[];detail={};scores={}
 for split in ['validation','test']:
  a=dict(np.load(P/f'selected_{split}_predictions.npz'));r=a.pop('returns');ends=a.pop('ends');D=a.pop('D');months=np.array([dt.datetime.fromtimestamp(int(t-1),dt.timezone.utc).strftime('%Y-%m') for t in ends]);detail[split]={};scores[split]={}
  for name,raw in a.items():
   m,z=evaluate(r,raw,scale=D);detail[split][name]=m;scores[split][name]=z['nll'];rows.append(dict(split=split,method=name,**{k:m[k] for k in ['count','nll','raw_spd_violation_rate','min_raw_eigenvalue','projection_frequency','mean_projection_frobenius','max_projection_frobenius','mean_mahalanobis_squared']},coverage50=m['coverage']['0.5'],coverage90=m['coverage']['0.9'],coverage95=m['coverage']['0.95']))
   for month in np.unique(months):
    mask=months==month;mm,_=evaluate(r[mask],raw[mask],scale=D);monthly.append(dict(split=split,month=month,method=name,count=int(mask.sum()),nll=mm['nll'],coverage50=mm['coverage']['0.5'],coverage90=mm['coverage']['0.9'],coverage95=mm['coverage']['0.95'],raw_spd_rate=mm['raw_spd_violation_rate'],projection_frequency=mm['projection_frequency']))
   for label,w in [('equal_weight',np.ones(3)/3),('BTC_ETH_spread',np.array([1.,-1.,0.]))]:
    variance=np.einsum('i,nij,j->n',w,z['projected_physical'],w);obs=r@w;ratio=obs**2/variance;pnll=.5*(np.log(2*np.pi*variance)+ratio)
    for month in ['ALL',*np.unique(months)]:
     mask=np.ones(len(r),bool) if month=='ALL' else months==month
     portfolios.append(dict(split=split,method=name,portfolio=label,month=month,count=int(mask.sum()),scalar_nll=float(pnll[mask].mean()),mean_forecast_variance=float(variance[mask].mean()),mean_observed_squared_return=float(np.mean(obs[mask]**2)),mean_squared_standardized_return=float(ratio[mask].mean()),coverage95=float(np.mean(ratio[mask]<=norm.ppf(.975)**2))))
  np.savez_compressed(P/f'{split}_score_arrays.npz',**scores[split],ends=ends)
 paired=[]
 for split,sc in scores.items():
  a=np.load(P/f'{split}_score_arrays.npz');months=np.array([dt.datetime.fromtimestamp(int(t-1),dt.timezone.utc).strftime('%Y-%m') for t in a['ends']])
  for name in sc:
   if name in ['constant','ewma','dcc']:continue
   for comparator in ['constant','ewma','dcc']:
    if comparator not in sc:continue
    delta=sc[name]-sc[comparator]
    for month in ['ALL',*np.unique(months)]:
     mask=np.ones(len(delta),bool) if month=='ALL' else months==month
     paired.append(dict(split=split,method=name,comparator=comparator,month=month,count=int(mask.sum()),mean_nll_difference=float(delta[mask].mean()),median_hour_difference=float(np.median(delta[mask])),lower_hours=int(np.sum(delta[mask]<0)),higher_hours=int(np.sum(delta[mask]>0))))
 csvwrite('per_run.csv',rows);csvwrite('per_month.csv',monthly);csvwrite('portfolio_scores.csv',portfolios);csvwrite('paired_differences.csv',paired)
 candidates=json.loads((P/'candidate_validation.json').read_text());csvwrite('candidate_validation.csv',candidates)
 sel=json.loads((P/'SELECTION.json').read_text());dcc_meta=json.loads((P/'models/dcc.json').read_text())
 summary=dict(metrics=detail,selection=sel,gaussian_algebra_all_pass=True,rtol=1e-9,atol=1e-9,observation='Zero-mean conditional second moment; not coefficient truth or continuous-time diffusion identification',test_preregistered_before_values=True)
 (P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 methods=['constant','ewma']+(['dcc'] if 'dcc' in detail['test'] else [])+['arff_s0','arff_s1','arff_s2','neural_s0','neural_s1','neural_s2']
 fig,axes=plt.subplots(1,2,figsize=(11,4.8));y=np.arange(len(methods));axes[0].scatter([detail['test'][k]['nll'] for k in methods],y);axes[0].set_yticks(y,methods);axes[0].invert_yaxis();axes[0].set_xlabel('Physical-return test Gaussian NLL (lower is better)');axes[0].grid(axis='x',alpha=.2)
 for k in ['arff_s0','arff_s1','arff_s2','neural_s0','neural_s1','neural_s2']:
  rr=[z for z in paired if z['split']=='test' and z['method']==k and z['comparator']=='ewma' and z['month']!='ALL'];axes[1].plot(np.arange(len(rr)),[z['mean_nll_difference'] for z in rr],'-o',label=k,markersize=3)
 axes[1].axhline(0,color='black',lw=.8);axes[1].set_xticks(range(12),range(1,13));axes[1].set_xlabel('2025 calendar month');axes[1].set_ylabel('Paired NLL minus EWMA');axes[1].legend(fontsize=8);fig.suptitle('Fixed-basket hourly second-moment pilot — exploratory');fig.tight_layout();fig.savefig(P/'comparison.pdf');fig.savefig(P/'comparison.png',dpi=180)
 lines=['# Fixed-basket financial conditional-second-moment pilot','','Exploratory BTCUSDT/ETHUSDT/BNBUSDT hourly spot returns; common zero mean. Fits2022–23, validation2024 reused for checkpoint/configuration selection, sealed test2025. D means fitting-only return scales; all NLLs below are in the same physical decimal-log-return coordinates. This is not the seven-fit SDE estimator or an investment-performance study.','','## All selected-seed outcomes','','| Method/seed | Validation NLL | Test NLL | Raw SPD violations | Projection frequency | 95% joint coverage |','|---|---:|---:|---:|---:|---:|']
 for k in detail['test']:
  m=detail['test'][k];lines.append(f"| {k} | {detail['validation'][k]['nll']:.6f} | {m['nll']:.6f} | {m['raw_spd_violation_rate']:.3%} | {m['projection_frequency']:.3%} | {m['coverage']['0.95']:.3%} |")
 lines+=['','## Selection and stability',f'ARFF grid index {sel["arff"]["grid_index"]}, neural grid index {sel["neural"]["grid_index"]}, EWMA decay {sel["ewma"]["decay"]}. Configurations selected by2024 only; all three selected fitting seeds retained. DCC valid: {sel["dcc_valid"]}. All18 candidate validation/checkpoint results are in candidate_validation.csv; DCC optimizer start/convergence details are in its saved model and DCC_REPORT.json.','']
 stability={}
 for comparator in ['constant','ewma','dcc']:
  if comparator not in detail['test']:continue
  values=[];wins=[]
  for seed in range(3):
   name=f'arff_s{seed}';delta=detail['test'][name]['nll']-detail['test'][comparator]['nll'];rr=[r for r in paired if r['split']=='test' and r['method']==name and r['comparator']==comparator and r['month']!='ALL'];win=sum(z['mean_nll_difference']<0 for z in rr);values.append(delta);wins.append(win)
  stability[comparator]=dict(seed_differences=values,months_lower=wins)
  lines.append(f'ARFF minus {comparator}: seed NLL differences {values}; months with lower NLL {wins}/12. No independent-hour confidence interval or seed-as-market replication claim.')
 off=[detail['test'][f'arff_s{i}']['nll']-detail['test'][f'arff_s{i}_diagonal']['nll'] for i in range(3)];lines+=['',f'Off-diagonal contribution (full minus same-raw-diagonal ARFF score) by seed: {off}. Negative values favor retaining off-diagonals. This is a fixed descriptive ablation, not another selected model.','', '## Calibration and interpretation','Gaussian nominal coverage is diagnostic; heavy tails and regime changes are not assumed absent. Squared returns are noisy realizations, not covariance truth. Reported raw eigenvalues/projection magnitudes are in fitting-scaled coordinates; score/calibration use the fixed1e-4 floor before mapping to physical units. No runs are removed for non-SPD raw covariance. Portfolio scores for fixed equal weights and BTC-ETH spread are in portfolio_scores.csv, with monthly counts. No weights were optimized and no profits computed.','', 'Months are dependent descriptive periods. Fitting seeds vary numerical initialization/adaptation on a single market history. The asset basket, exchange and USDT numeraire are fixed retrospective choices and do not support broad equity/FX or investment claims.','', '## Reproduction and audit','PROTOCOL.md, FROZEN.json, DATA_MANIFEST.json, SPLIT_MANIFEST.json, PREFIT_CHECKS.json and PRETEST_GATE.json identify data/config/source and causal/numerical gates. Rightsholder archive files are not distributed with Git; use the downloader and checksums. Models, full predictions and score arrays remain in the local study directory. Per-run/per-month/paired/portfolio CSVs retain all valid outcomes. Native ARFF and covariance-MLP functions were reused without production edits. See execution.json and terminal records for resource use and any incomplete comparator. No manuscript or accepted result was changed.']
 (P/'REPORT.md').write_text('\n'.join(lines)+'\n');(P/'interpretation.json').write_text(json.dumps(dict(stability=stability,off_diagonal_full_minus_diagonal=off),indent=2)+'\n')
if __name__=='__main__':report()
