#!/usr/bin/env python3
"""Separate validation-only regime selection from full held-out result reporting."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['MPLBACKEND']='Agg'
import sys,json,csv
from pathlib import Path
import numpy as np
import regime_ex8 as r
import run_regime_ex8_campaign as scheduler
from summarize_controlled_ex8_float64_v2 import log_metrics,native_metrics,LABELS,METRICS
ROOT,OUT,c=r.ROOT,r.STUDY,r.campaign

def checked_jobs(phase):
 control=OUT/'campaigns'/phase;assert c.infra.read_json(control/'state.json')['status']=='complete'
 plan=c.verify(control)
 for job in plan['jobs']:
  d,a,l,ctx=c.paths(job);done=c.infra.read_json(d/'complete.json');v=c.validate(job,a,l,None if done.get('reused') else ctx)
  assert v['artifact_sha256']==done['artifact_sha256'] and v['log_sha256']==done['log_sha256']
  yield job,a,l,v

def equivalent(reference,candidate,metric):
 a,b=np.asarray(reference),np.asarray(candidate);assert a.shape==b.shape==(10,)
 delta=b-a;half=1.8331129326536335*delta.std(ddof=1)/np.sqrt(10)
 band=max(.05*abs(a.mean()),a.std(ddof=1)) if metric!='nll' else max(.05,a.std(ddof=1))
 lo,hi=float(delta.mean()-half),float(delta.mean()+half)
 return dict(passes=bool(lo>=-band and hi<=band),mean_difference=float(delta.mean()),paired_90_ci=[lo,hi],equivalence_band=float(band))

def select():
 m=r.adapter.verify_registration();records=[];hashes={}
 for phase in ['N','h']:
  for job,a,l,v in checked_jobs(phase):
   # Endpoint validation only; test metrics never enter this record/decision.
   if job['method'].startswith('joint'):values={k:log_metrics(l,'validation')[k] for k in METRICS}
   else:
    with np.load(a,allow_pickle=False) as z:values={k:float(z['validation_'+k]) for k in METRICS}
   records.append(dict(phase=phase,**job,**values));hashes[str(a)]=v['artifact_sha256']
 def values(phase,method,key,point,metric):
  rows=sorted([v for v in records if v['phase']==phase and v['method']==method and v[key]==point],key=lambda v:v['seed'])
  assert [v['seed'] for v in rows]==list(range(10))
  return np.array([v[metric]-(np.log(v['h']) if phase=='h' and metric=='nll' else 0.) for v in rows])
 n_checks={};nstar=max(m['N_grid']);resolved_N=False
 for N in m['N_grid'][:-1]:
  checks=[]
  for larger in m['N_grid']:
   if larger<=N:continue
   for method in m['methods']:
    for metric in METRICS:checks.append(dict(method=method,metric=metric,larger_N=larger,**equivalent(values('N',method,'N',larger,metric),values('N',method,'N',N,metric),metric)))
  n_checks[str(N)]=checks
  if all(v['passes'] for v in checks) and not resolved_N:nstar=N;resolved_N=True
 h_checks={};hstar=.0001;resolved_h=False
 for smaller,h in zip(m['h_grid'][:-1],m['h_grid'][1:]):
  checks=[]
  for method in m['methods']:
   for metric in METRICS:checks.append(dict(method=method,metric=metric,**equivalent(values('h',method,'h',smaller,metric),values('h',method,'h',h,metric),metric)))
  bias=abs(np.expm1(round(h/1e-7)*np.log1p(-1e-7))/h+1.)
  passed=bool(all(v['passes'] for v in checks) and bias<=.001)
  h_checks[str(h)]=dict(previous_h=smaller,drift_relative_finite_lag_bias=float(bias),passes=passed,checks=checks)
  if passed:hstar=h;resolved_h=True
 result=dict(N_star=nstar,h_star=hstar,N_equivalence_demonstrated=resolved_N,h_equivalence_demonstrated=resolved_h,test_used=False,selection_metric='Endpoint validation RMSE and NLL; for h use NLL-log(h)',N_checks=n_checks,h_checks=h_checks,validation_source_sha256=hashes,limitations=['Seed uncertainty conditions on one fixed dataset, not repeated independent datasets.','N screening at reference h; h screening at max N. Joint interaction not certified.','Exact drift oracle only; covariance finite-lag bias has no certified bound.','If equivalence fails, Phase2 is descriptive; no claim of capacity-dominated or O(1/K) behavior.'])
 c.infra.write_json(OUT/'selection.json',result,exclusive=True)
 c.infra.write_json(OUT/'validation_selection_inputs.json',records,exclusive=True)
 # Materialize a missing selected nested prefix from an already generated h view.
 folder=OUT/f'dataset_roots/N_{nstar}/h_{hstar:.8g}'
 if not folder.exists():
  import create_regime_ex8_data as builder
  from src.experiments.dataset import load_dataset
  d=load_dataset(OUT/f'dataset_roots/N_640000/h_{hstar:.8g}/data/ex8.npz')
  base=load_dataset(ROOT/'results/controlled_study_2026/float64_v2'/json.loads((ROOT/'results/controlled_study_2026/float64_v2/lag_datasets/complete.json').read_text())['dataset_roots'][format(hstar,'.8g')]/'data/ex8.npz')
  builder.make_view(nstar,hstar,d.x,d.r,base)
 print('Validation-only frozen regime:',nstar,hstar,'equivalence',resolved_N,resolved_h,flush=True)

def describe(x):
 x=np.asarray(x);assert len(x)==10 and np.isfinite(x).all()
 return dict(n=10,mean=float(x.mean()),sd=float(x.std(ddof=1)),median=float(np.median(x)),min=float(x.min()),max=float(x.max()))
def report(phase):
 out=OUT/(phase+'_summary');assert not out.exists();rows=[];hashes={}
 for job,a,l,v in checked_jobs(phase):
  with np.load(a,allow_pickle=False) as z:arr={k:z[k] for k in z.files}
  row=dict(**job,parameters=v['parameter_count'],algorithm_time=v['algorithm_time'])
  for checkpoint in ['best_epoch','final_drift_best_epoch','covariance_best_epoch','final_drift_best_iteration','covariance_best_iteration']:
   if checkpoint in arr:row[checkpoint]=int(arr[checkpoint])
  for split,metrics in native_metrics(job['method'],arr,l).items():
   for k,val in metrics.items():row[split+'_'+k]=val
  rows.append(row);hashes[str(a)]=v['artifact_sha256'];hashes[str(l)]=v['log_sha256']
 points=[];key='N' if phase=='N' else 'h' if phase=='h' else 'K'
 for method in LABELS:
  for point in sorted(set(v[key] for v in rows if v['method']==method)):
   data=sorted([v for v in rows if v['method']==method and v[key]==point],key=lambda v:v['seed']);assert [v['seed'] for v in data]==list(range(10))
   stats={k:describe([v['test_'+k] for v in data]) for k in METRICS}
   for k in ['raw_spd_violation_rate','min_raw_eigenvalue']:
    if all('test_'+k in v for v in data):stats[k]=describe([v['test_'+k] for v in data])
   points.append(dict(method=method,point=point,parameters=data[0]['parameters'],metrics=stats))
 interpretation={}
 if phase=='capacity':
  for method in LABELS:
   p=sorted([v for v in points if v['method']==method],key=lambda v:v['parameters']);interpretation[method]={}
   for metric in METRICS:
    y=np.array([v['metrics'][metric]['mean'] for v in p]);d=np.diff(y)
    records=sorted([v for v in rows if v['method']==method],key=lambda v:(v['K'],v['seed']))
    low=np.array([v['test_'+metric] for v in records if v['K']==64]);high=np.array([v['test_'+metric] for v in records if v['K']==1024]);prev=np.array([v['test_'+metric] for v in records if v['K']==512])
    endpoint=equivalent(low,high,metric);last_step=equivalent(prev,high,metric)
    label=('systematic mean decrease with resolved endpoint gain' if np.all(d<0) and endpoint['paired_90_ci'][1]<0 else 'resolved high-capacity deterioration' if last_step['paired_90_ci'][0]>0 else 'endpoint plateau compatible with uncertainty' if endpoint['passes'] else 'mixed/nonmonotonic or unresolved capacity effect')
    interpretation[method][metric]=dict(pattern=label,endpoint=endpoint,last_step=last_step,warning='Conditional seed uncertainty only. Finite-N, covariance finite-lag bias and optimization error are not eliminated. No O(1/K) rate is imposed.')
 out.mkdir()
 c.infra.write_json(out/'summary.json',dict(points=points,source_sha256=hashes,interpretation=interpretation,timing='non-isolated four-GPU accuracy'),exclusive=True)
 with (out/'per_seed_metrics.csv').open('x',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for v in rows for k in v)));w.writeheader();w.writerows(rows)
 lines=[f'# Experiment8 {phase} diagnostic: ten seeds','', '| Method | Point | Parameters | Drift RMSE mean ± SD | Covariance RMSE mean ± SD | NLL mean ± SD |','|---|---|---|---|---|---|']
 for p in points:lines.append('| '+ ' | '.join([LABELS[p['method']],str(p['point']),str(p['parameters'])]+[f"{p['metrics'][k]['mean']:.7g} ± {p['metrics'][k]['sd']:.5g}" for k in METRICS])+' |')
 lines+=['','No posterior selection on test results. Native raw covariance RMSE; ARFF raw SPD statistics retained in JSON/CSV. NLL uses native SPD/factor conventions. Finite-N, finite-lag, small-h noise, and optimization effects are not automatically separated by a performance curve.']
 for p in points:
  if p['method']=='arff':
   rate=p['metrics']['raw_spd_violation_rate'];eig=p['metrics']['min_raw_eigenvalue']
   lines.append(f"ARFF point {p['point']}: test raw SPD rate {rate['mean']:.7g} ± {rate['sd']:.5g}, max {rate['max']:.7g}; minimum raw eigenvalue over seeds {eig['min']:.7g}.")
 if phase=='h':lines+=['','See ../oracle_moments.json: exact conditional drift and covariance trace, with empirical trace checks. Full covariance orientation/bias is not certified. Raw NLL shifts with log(h); selection uses NLL-log(h). Small-h drift noise rises like h^(-1/2).']
 if phase=='capacity':
  resolved=sum('resolved endpoint gain' in interpretation[m][k]['pattern'] for m in LABELS for k in ['drift_rmse','covariance_rmse'])
  rationale=json.loads((OUT/'selection.json').read_text())
  verdict=('The candidate shows resolved monotonic coefficient trends in at least6/10 method-metric combinations and adds capacity information beyond the one-capacity boxplot; consider it for review, subject to the stated finite-lag and fixed-dataset limits.' if resolved>=6 and rationale['N_equivalence_demonstrated'] and rationale['h_equivalence_demonstrated'] else 'The diagnostics do not establish that this candidate is substantially clearer or more scientifically conclusive than the accepted boxplot; retain the existing poster pending human review.')
  lines+=['',json.dumps(interpretation,indent=2),'',verdict,'','Poster candidate only; no poster or manuscript changed. No power-law slope is fitted automatically from four points. See selection.json for frozen N*,h* and unresolved flags.']
 (out/'report.md').write_text('\n'.join(lines)+'\n')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,3,figsize=(16,5));colors=['#4477AA','#66CCEE','#228833','#AA3377','#CCBB44']
 for ax,metric in zip(axes,METRICS):
  for (method,name),color in zip(LABELS.items(),colors):
   pts=sorted([p for p in points if p['method']==method],key=lambda v:v['point'])
   x=[p['parameters'] if phase=='capacity' else p['point'] for p in pts]
   ax.errorbar(x,[p['metrics'][metric]['mean'] for p in pts],yerr=[p['metrics'][metric]['sd'] for p in pts],label=name,color=color,marker='o',capsize=3)
  ax.set_xscale('log');ax.set_xlabel('Final retained parameters' if phase=='capacity' else 'Nominal training N' if phase=='N' else 'Observation lag h');ax.set_ylabel('Test '+metric.replace('_',' '));ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);fig.suptitle('Experiment 8: '+phase+' diagnostic (10 seeds; mean ± sample SD)');fig.tight_layout()
 for ext in ['pdf','png']:
  with (out/f'ex8_regime_{phase}_candidate.{ext}').open('xb') as f:fig.savefig(f,format=ext,bbox_inches='tight',dpi=300)
 plt.close(fig);print('Validated summary',out,flush=True)
if __name__=='__main__':
 if sys.argv[1]=='select':select()
 else:report(sys.argv[1])
