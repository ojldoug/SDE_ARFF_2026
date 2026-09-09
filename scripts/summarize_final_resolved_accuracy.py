#!/usr/bin/env python3
"""Validate complete resolved campaigns and publish a new immutable result bundle.

--watch waits in tmux. No training, no selection, no overwrites, no manuscript edits.
"""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import argparse,csv,json,time
import numpy as np
import run_final_adam_campaign as adam
import run_final_arff_campaign as arff
import summarize_ex8_capacity_matched as ex8
ROOT=adam.ROOT;BASE=adam.OUT;OUT=BASE/'resolved_accuracy_v1'
FIELDS=['drift_rmse','covariance_rmse','nll']

def stats(values):
 x=np.asarray(values,float)
 if len(x)!=30 or not np.all(np.isfinite(x)):raise ValueError('Require all30 finite seeds')
 return dict(n=30,mean=float(x.mean()),sample_sd=float(x.std(ddof=1)),median=float(np.median(x)),minimum=float(x.min()),maximum=float(x.max()))

def main(watch=False):
 while True:
  states=[json.loads((c.CONTROL/'state.json').read_text()) for c in (adam,arff)]
  if any(s.get('errors') for s in states):raise RuntimeError('Worker failure; no aggregate published and no numerical patch/retry')
  if all(s['status']=='complete' for s in states):break
  if not watch:raise RuntimeError('Accuracy campaigns not complete')
  print('Waiting:',[s['status'] for s in states],flush=True);time.sleep(30)
 rows=[];hashes={};curves={};runtime=[]
 for campaign in (adam,arff):
  plan=campaign.verify()
  for job in plan['jobs']:
   artifact,log,reservation=campaign.paths(job);check=campaign.validate(job,artifact,log)
   complete=campaign.infra.read_json(reservation/'complete.json')
   if any(check[k]!=complete[k] for k in ['artifact_sha256','log_sha256']):raise ValueError('Published result changed')
   with np.load(artifact,allow_pickle=False) as a:
    row=dict(experiment=job['experiment'],method=job['method'],seed=job['seed'],evaluation_split='validation',parameters=int(a['total_active_parameters']),algorithm_time=float(a['algorithm_time']),compilation_time=float(a['compilation_time']),**{k:float(a['validation_'+k]) for k in FIELDS})
    row.update(raw_spd_violation_rate=float(a['validation_raw_spd_violation_rate']),min_raw_eigenvalue=float(a['validation_min_raw_eigenvalue']))
    rows.append(row)
    key=job['experiment']+'_'+job['method'];curves.setdefault(key,[])
    if campaign is adam:curves[key].append((np.array(a['cumulative_time']),np.array(a['validation_nll_history']),'validation Gaussian NLL'))
    else:
     for stage in ['final_drift','covariance']:
      curves.setdefault(key+'_'+stage,[]).append((np.array(a[stage+'_cumulative_time']),np.array(a[stage+'_validation_mse']),stage+' internal MSE'))
    runtime.append(dict(job,runner='adam' if campaign is adam else 'arff',estimated_seconds=float(a['end_to_end_time'])))
   hashes[str(artifact.relative_to(ROOT))]=check['artifact_sha256'];hashes[str(log.relative_to(ROOT))]=check['log_sha256']
 accepted,accepted_hashes=ex8.collect();hashes.update(accepted_hashes)
 for a in accepted:
  rows.append(dict(experiment='ex8',method=a['method'],seed=a['seed'],evaluation_split='test',parameters=1814 if a['method']=='mlp' else 1792,algorithm_time=a['algorithm_time'],compilation_time='',raw_spd_violation_rate='',min_raw_eigenvalue='',**{k:a[k] for k in FIELDS}))
 groups={}
 for r in rows:groups.setdefault((r['experiment'],r['method']),[]).append(r)
 summary={};arrays={};table=['# Resolved accuracy results','', 'Experiments1–7: canonical validation, no independent test. Experiment8: frozen accepted test results.30 seeds per method; SD uses ddof=1. Non-isolated timings are not controlled runtime results. Experiment4 and unresolved historical split-MLP/Section6.2 studies are excluded.','', '| Experiment | Method | Scope | Parameters | Drift RMSE mean±SD | Covariance RMSE mean±SD | NLL mean±SD |','|---|---|---|---:|---:|---:|---:|']
 for (ex,method),rr in groups.items():
  rr.sort(key=lambda x:x['seed'])
  if [r['seed'] for r in rr]!=list(range(30)):raise ValueError('Seed coverage')
  if len({r['parameters'] for r in rr})!=1:raise ValueError('Parameter count changed')
  key=ex+'_'+method;summary[key]={k:stats([r[k] for r in rr]) for k in FIELDS}
  for k in FIELDS:arrays[key+'_'+k]=np.array([r[k] for r in rr])
  cells=[f"{summary[key][k]['mean']:.7g} ± {summary[key][k]['sample_sd']:.7g}" for k in FIELDS]
  table.append(f"| {ex} | {method} | {rr[0]['evaluation_split']} | {rr[0]['parameters']} | "+' | '.join(cells)+' |')
 OUT.mkdir(exist_ok=False)
 with (OUT/'per_seed_metrics.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 with (OUT/'summary.json').open('x') as f:json.dump(dict(statistics=summary,source_sha256=hashes,timing='non-isolated accuracy',ex8_missing_spd_fields='See unchanged accepted artifacts/earlier validation bundle; not fabricated in this merged CSV'),f,indent=2)
 with (OUT/'distributions.npz').open('xb') as f:np.savez_compressed(f,seeds=np.arange(30),**arrays)
 (OUT/'tables.md').write_text('\n'.join(table)+'\n')
 import matplotlib.pyplot as plt
 for ex in sorted({r['experiment'] for r in rows}):
  methods=[m for e,m in groups if e==ex];fig,axes=plt.subplots(1,3,figsize=(15,4.5))
  for ax,metric in zip(axes,FIELDS):
   values=[arrays[ex+'_'+m+'_'+metric] for m in methods]
   ax.boxplot(values,tick_labels=methods,showfliers=True);ax.set_ylabel(('Test ' if ex=='ex8' else 'Validation ')+metric);ax.tick_params(axis='x',labelrotation=25);ax.grid(axis='y',alpha=.25)
  fig.suptitle(ex+' —30 seeds, resolved production');fig.tight_layout()
  with (OUT/(ex+'_distributions.pdf')).open('xb') as f:fig.savefig(f,format='pdf',bbox_inches='tight')
  plt.close(fig)
 for name,cc in curves.items():
  if not cc:continue
  fig,ax=plt.subplots(figsize=(7,4))
  for t,y,label in cc:ax.plot(t,y,alpha=.25,linewidth=.8)
  ax.set(xlabel='Non-isolated recorded algorithm seconds',ylabel=cc[0][2],title=name+' —30 recorded histories');ax.grid(alpha=.2);fig.tight_layout()
  with (OUT/(name+'_history.pdf')).open('xb') as f:fig.savefig(f,format='pdf')
  plt.close(fig)
 total=sum(r['estimated_seconds'] for r in runtime)
 selected=list(range(30)) if total<=12*3600 else [0,6,12,18,24]
 with (OUT/'isolated_runtime_cost_plan.json').open('x') as f:json.dump(dict(projected_resolved_main_seconds=total,selected_seeds=selected,selection_rule='All seeds if recorded warmup+algorithm total <=12h, otherwise predetermined seeds0,6,12,18,24; no metric-based selection',status='COST_PROPOSAL_NOT_DISPATCHED',limitation='Parallel times are an estimate. Complete-path Adam warmup and accepted ex8 timing jobs must be included before final runtime dispatch.'),f,indent=2)
 with (OUT/'complete.json').open('x') as f:json.dump(dict(artifacts=len(rows),new_artifacts=660,reused_ex8=120,status='validated',files_sha256={str(p.relative_to(OUT)):adam.digest(p) for p in OUT.iterdir() if p.is_file() and p.name!='complete.json'}),f,indent=2)
 print('Validated resolved accuracy bundle:',OUT,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args();main(a.watch)
