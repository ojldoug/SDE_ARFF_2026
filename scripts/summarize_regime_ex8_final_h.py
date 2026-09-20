"""Final descriptive lag summary, using saved metrics only; never select a lag."""
import os
os.environ.update(JAX_PLATFORMS='cpu',MPLBACKEND='Agg',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import json,csv
import numpy as np
import matplotlib.pyplot as plt
import run_regime_ex8_final_h_campaign as s
from summarize_controlled_ex8_float64_v2 import native_metrics,LABELS
OUT=s.STUDY/'h_summary'
def describe(values):
 a=np.asarray(values,float);assert len(a)==10 and np.isfinite(a).all()
 return dict(n=10,mean=float(a.mean()),sd=float(a.std(ddof=1)),median=float(np.median(a)),min=float(a.min()),max=float(a.max()))
def main():
 assert not OUT.exists()
 m=s.r.adapter.verify_registration();control=s.STUDY/'campaigns/h'
 assert s.c.infra.read_json(control/'state.json')['status']=='complete'
 plan=s.c.verify(control);records=[];hashes={}
 for job in plan['jobs']:
  d,a,log,ctx=s.c.paths(job);done=s.c.infra.read_json(d/'complete.json')
  checked=s.c.validate(job,a,log,None if done.get('reused') else ctx)
  assert checked['artifact_sha256']==done['artifact_sha256'] and checked['log_sha256']==done['log_sha256']
  with np.load(a,allow_pickle=False) as z:arr={k:z[k] for k in z.files}
  row=dict(**job,parameters=checked['parameter_count'],algorithm_time=checked['algorithm_time'],reused=bool(done.get('reused')))
  for split,values in native_metrics(job['method'],arr,log).items():
   for metric,value in values.items():row[split+'_'+metric]=value
   row[split+'_nll_minus_log_h']=values['nll']-np.log(job['h'])
  records.append(row);hashes[str(a)]=checked['artifact_sha256'];hashes[str(log)]=checked['log_sha256']
 points=[]
 for split in ['validation','test']:
  for method in LABELS:
   for h in m['h_grid']:
    rows=sorted([v for v in records if v['method']==method and v['h']==h],key=lambda v:v['seed'])
    assert [v['seed'] for v in rows]==list(range(10))
    metrics={}
    for key in ['drift_rmse','covariance_rmse','nll','nll_minus_log_h','raw_spd_violation_rate','min_raw_eigenvalue','min_covariance_eig','numerical_nonpositive_eig_rate']:
     if all(split+'_'+key in row for row in rows):metrics[key]=describe([v[split+'_'+key] for v in rows])
    points.append(dict(split=split,method=method,h=h,metrics=metrics))
 s.r.adapter.verify_registration()
 OUT.mkdir()
 s.c.infra.write_json(OUT/'summary.json',dict(label='Final oracle-criterion-driven extension; descriptive, not an optimal-lag search',points=points,source_sha256=hashes,screening=m['amendment']['oracle_screening'],unchanged_phase2=m['amendment']['frozen_phase2'],no_selection=True,hard_maximum_h=.002,timing='non-isolated accuracy',limitations=['Conditional seed variability on one fixed dataset.','Drift-bias tolerance does not certify full covariance finite-lag bias.','No optimal lag or capacity-dominated regime inferred.']),exclusive=True)
 with (OUT/'per_seed_metrics.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for row in records for k in row)));w.writeheader();w.writerows(records)
 lines=['# Final oracle-criterion-driven Experiment 8 lag extension','',
 'Descriptive diagnostic, not a search for ARFF’s optimal lag. All five fixed methods, K1024, N640000, seeds0–9. Original settings and artifacts are unchanged. Added lags are included solely by the same discrete drift oracle and threshold0.001. No lag beyond0.002 is authorized or dispatched.','',
 'Phase2 remains N*=640000,h*=0.0001 with the previously recorded unresolved equivalence flags. Selection is neither rerun nor reinterpreted. The present report does not start another campaign.','',
 'Each value is mean ± sample SD over10 seeds on one fixed dataset. Covariance RMSE is raw/pre-projection; NLL uses each accepted method’s unchanged covariance/SPD convention. NLL−log(h) removes the known two-dimensional scale offset, not finite-lag transition changes. Positive factor covariance is distinct from ARFF’s raw indefinite predictions.','',
 '| Candidate h | Exact discrete relative drift bias | Included |','|---|---:|---|']
 for v in m['amendment']['oracle_screening']:lines.append(f"| {v['h']} | {v['relative_discrete_drift_bias']:.16g} | {v['included']} |")
 for split in ['validation','test']:
  lines+=['',f'## {split.capitalize()} results','', '| Method | h | Drift RMSE | Raw covariance RMSE | NLL | NLL−log h |','|---|---:|---:|---:|---:|---:|']
  for p in [p for p in points if p['split']==split]:
   cells=[f"{p['metrics'][k]['mean']:.8g} ± {p['metrics'][k]['sd']:.5g}" for k in ['drift_rmse','covariance_rmse','nll','nll_minus_log_h']]
   lines.append('| '+' | '.join([LABELS[p['method']],str(p['h'])]+cells)+' |')
  lines+=['','| ARFF h | Raw SPD violation rate | Minimum raw eigenvalue: mean ± SD | Worst minimum across seeds |','|---|---:|---:|---:|']
  for p in [p for p in points if p['split']==split and p['method']=='arff']:
   rate=p['metrics']['raw_spd_violation_rate'];eig=p['metrics']['min_raw_eigenvalue']
   lines.append(f"| {p['h']} | {rate['mean']:.8g} ± {rate['sd']:.5g} | {eig['mean']:.8g} ± {eig['sd']:.5g} | {eig['min']:.8g} |")
  fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
  for ax,key in zip(axes.flat,['drift_rmse','covariance_rmse','nll','nll_minus_log_h']):
   for method,label in LABELS.items():
    pp=[p for p in points if p['split']==split and p['method']==method]
    ax.errorbar([p['h'] for p in pp],[p['metrics'][key]['mean'] for p in pp],yerr=[p['metrics'][key]['sd'] for p in pp],marker='o',capsize=3,label=label)
   ax.set_xscale('log');ax.set_xlabel('Observation lag h');ax.set_ylabel(key.replace('_',' '));ax.grid(alpha=.2)
  axes[0,0].legend(fontsize=8);fig.suptitle(f'Experiment 8: final oracle-criterion-driven lag extension\n{split.capitalize()} metrics; 10 seeds, mean ± sample SD; no lag selection')
  for ext in ['pdf','png']:
   with (OUT/f'ex8_final_h_{split}_candidate.{ext}').open('xb') as f:fig.savefig(f,format=ext,dpi=250)
  plt.close(fig)
 lines+=['','All450 source artifacts and logs passed the existing artifact validator and completion-hash checks. Exact earlier-lag reuses retain their source bytes. Commands, dataset provenance, seeds, raw metrics, timings and hashes remain archived. No poster/manuscript changes. This final bounded extension is complete; no subsequent experiment is launched.']
 with (OUT/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 print('FINAL EXTENSION COMPLETE; STOP. Report:',OUT/'report.md',flush=True)
if __name__=='__main__':main()
