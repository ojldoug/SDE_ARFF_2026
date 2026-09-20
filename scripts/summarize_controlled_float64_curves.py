#!/usr/bin/env python3
"""Validated complete registered curves; no selection by test performance."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,json,csv
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_campaign_float64_v2_extended as c
from summarize_controlled_ex8_float64_v2 import native_metrics,LABELS,METRICS

def describe(values):
 x=np.asarray(values,dtype=float);assert x.size in (5,30) and np.isfinite(x).all()
 return dict(n=len(x),mean=float(x.mean()),sd=float(x.std(ddof=1)),median=float(np.median(x)),min=float(x.min()),max=float(x.max()))
def main(stage):
 assert stage in ('stage1','final')
 out=c.STUDY/(stage+'_curves');assert not out.exists()
 m=c.adapter.verify_registration();rows=[];sources={};points={}
 phases=['stage1_capacity_N','stage1_h','stage1_adaptation']
 if stage=='final':
  phases += [p for p in ['stage2_capacity','stage2_N','stage2_h','stage2_adaptation'] if (c.STUDY/'campaigns'/p/'plan.json').exists()]
 for phase in phases:
  control=c.STUDY/'campaigns'/phase
  assert c.infra.read_json(control/'state.json')['status']=='complete'
  plan=c.verify(control)
  for job in plan['jobs']:
   directory,a,l,ctx=c.paths(job);done=c.infra.read_json(directory/'complete.json')
   check=c.validate(job,a,l,None if done.get('reused') else ctx)
   assert check['artifact_sha256']==done['artifact_sha256'] and check['log_sha256']==done['log_sha256']
   with np.load(a,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
   values=native_metrics(job['method'],arrays,l)
   row=dict(study=job['study'],variant=job.get('variant',''),method=job['method'],seed=job['seed'],K=job['K'],N=job['N'],h=job['h'],parameters=check['parameter_count'],algorithm_time=check['algorithm_time'])
   for split,v in values.items():
    for k,value in v.items():row[split+'_'+k]=value
   rows.append(row);sources[str(a.relative_to(c.ROOT))]=check['artifact_sha256'];sources[str(l.relative_to(c.ROOT))]=check['log_sha256']
 # The full-N point is exactly the mandatory corrected baseline, not a new fit.
 extended_N='stage2_N' in phases
 baseplan=c.verify(c.STUDY/'campaigns/baseline')
 for job in baseplan['jobs']:
  if job['seed']>= (30 if extended_N else 5):continue
  directory,a,l,ctx=c.paths(job);checked=c.validate(job,a,l,ctx)
  with np.load(a,allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
  row=dict(study='N',variant='',method=job['method'],seed=job['seed'],K=128,N=80000,h=.0001,parameters=checked['parameter_count'],algorithm_time=checked['algorithm_time'])
  for split,v in native_metrics(job['method'],arrays,l).items():
   for k,value in v.items():row[split+'_'+k]=value
  rows.append(row);sources[str(a.relative_to(c.ROOT))]=checked['artifact_sha256'];sources[str(l.relative_to(c.ROOT))]=checked['log_sha256']
 for row in rows:
  group=(row['study'],row['method'],row['variant'],row['K'],row['N'],row['h'],row['parameters'])
  points.setdefault(group,[]).append(row)
 summaries=[]
 for group,values in points.items():
  seeds=sorted(v['seed'] for v in values);assert seeds==list(range(len(values))) and len(seeds) in (5,30)
  s=dict(zip(('study','method','variant','K','N','h','parameters'),group))
  s['metrics']={metric:describe([r['test_'+metric] for r in values]) for metric in METRICS}
  s['algorithm_time']=describe([r['algorithm_time'] for r in values]);summaries.append(s)
 # Explicit grid completeness; no isolated good-looking points may be plotted.
 selection=c.infra.read_json(c.STUDY/'calibration_selection.json')
 for study,grid,key in [('capacity',[v['K'] for v in m['capacity']],'K'),('N',m['N_grid'],'N'),('h',m['h_grid'],'h')]:
  for method in m['methods']:
   selected=[p for p in summaries if p['study']==study and p['method']==method]
   assert sorted(p[key] for p in selected)==sorted(grid)
   assert len({p['metrics']['nll']['n'] for p in selected})==1
 assert sorted(p['variant'] for p in summaries if p['study']=='adaptation')==sorted(v['variant'] for v in selection['selected_modes'])
 out.mkdir(exist_ok=False)
 c.infra.write_json(out/'summary.json',dict(points=summaries,source_sha256=sources,manifest_sha256=c.adapter.digest(c.STUDY/'manifest.json'),script_sha256=c.adapter.digest(__file__),timing='Non-isolated accuracy only',covariance='Raw/pre-projection RMSE; native SPD convention for NLL; all poor valid points retained'),exclusive=True)
 keys=list(dict.fromkeys(k for r in rows for k in r))
 with (out/'per_seed_metrics.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 import matplotlib.pyplot as plt
 colors=['#4477AA','#66CCEE','#228833','#AA3377','#CCBB44']
 for study,key,xlabel in [('capacity','parameters','Final retained parameters'),('N','N','Nominal training sample count'),('h','h','Observation lag h')]:
  fig,axes=plt.subplots(1,3,figsize=(16,4.8))
  for ax,metric in zip(axes,METRICS):
   for (method,label),color in zip(LABELS.items(),colors):
    values=sorted([p for p in summaries if p['study']==study and p['method']==method],key=lambda p:p[key])
    ax.errorbar([p[key] for p in values],[p['metrics'][metric]['mean'] for p in values],yerr=[p['metrics'][metric]['sd'] for p in values],label=label,color=color,marker='o',capsize=3)
   ax.set_xscale('log');ax.set_xlabel(xlabel);ax.set_ylabel('Test '+metric.replace('_',' '));ax.grid(alpha=.2)
  axes[0].legend(fontsize=8);fig.suptitle('Experiment 8: corrected-data '+study+' study (mean ± sample SD)');fig.tight_layout()
  for ext in ('pdf','png'):
   with (out/f'ex8_float64_v2_{study}_curves.{ext}').open('xb') as f:fig.savefig(f,format=ext,bbox_inches='tight',dpi=300)
  plt.close(fig)
 adaptation=[p for p in summaries if p['study']=='adaptation'];fig,axes=plt.subplots(1,3,figsize=(13,4.8))
 labels=[('Metropolis + resampling' if 'metro1_resample1' in p['variant'] else 'Metropolis only' if 'metro1' in p['variant'] else 'Resampling only') for p in adaptation]
 for ax,metric in zip(axes,METRICS):
  ax.errorbar(range(3),[p['metrics'][metric]['mean'] for p in adaptation],yerr=[p['metrics'][metric]['sd'] for p in adaptation],fmt='o',capsize=4)
  ax.set_xticks(range(3),labels,rotation=15);ax.set_ylabel('Test '+metric.replace('_',' '));ax.grid(alpha=.2)
 fig.suptitle('ARFF adaptation: separately validation-calibrated settings');fig.tight_layout()
 for ext in ('pdf','png'):
  with (out/f'ex8_float64_v2_adaptation.{ext}').open('xb') as f:fig.savefig(f,format=ext,bbox_inches='tight',dpi=300)
 plt.close(fig)
 print('Validated and plotted',len(rows),'run references at',len(summaries),'registered points:',out)
if __name__=='__main__':main(sys.argv[1])
