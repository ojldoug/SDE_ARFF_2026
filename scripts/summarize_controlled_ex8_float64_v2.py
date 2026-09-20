#!/usr/bin/env python3
"""Read-only model/artifact analysis, paired old/new data results and new figure."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import sys,json,csv,re
import numpy as np
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parent)]
import run_controlled_campaign_float64_v2 as campaign
import summarize_ex8_capacity_matched as stats
ROOT=campaign.ROOT;STUDY=campaign.STUDY;OUT=STUDY/'baseline_summary'
LABELS=dict(joint_fourier='Joint Fourier Adam',split_fourier='Split Fourier Adam',arff='ARFF',joint_mlp='Joint MLP Adam',split_mlp='Split MLP Adam')
OLD_KEYS=dict(joint_fourier='joint',split_fourier='split',arff='arff',joint_mlp='mlp',split_mlp='split_mlp')
METRICS=('drift_rmse','covariance_rmse','nll')

def log_metrics(log,label):
 text=log.read_text()
 block=re.search(r'^'+label+r'\s*\n(.*?)(?=^(?:train|validation|test|artifact)\b|\Z)',text,re.M|re.S)
 assert block is not None,(log,label)
 result={}
 for key,title in [('nll','NLL'),('drift_rmse','drift RMSE'),('covariance_rmse','covariance RMSE'),('min_raw_eigenvalue','min covariance eig')]:
  hit=re.search(r'^\s*'+title+r'\s*:\s*([-+\d.eE]+)\s*$',block[1],re.M)
  if hit:result[key]=float(hit[1])
 assert all(k in result for k in METRICS)
 return result

def native_metrics(method,a,log):
 if method.startswith('joint'):return {label:log_metrics(log,label) for label in ('train','validation','test')}
 values={}
 for label in ('train','validation','test'):
  values[label]={k:float(a[label+'_'+k]) for k in METRICS}
  for k in ['raw_spd_violation_rate','min_raw_eigenvalue','min_covariance_eig','numerical_nonpositive_eig_rate']:
   if label+'_'+k in a:values[label][k]=float(a[label+'_'+k])
 return values

def spd_from_model(method,a,x,indices):
 """Supplement missing raw SPD rates only; never replace native endpoint metrics."""
 import jax.numpy as jnp
 if method=='arff':
  return {label:{k:float(a[label+'_'+k]) for k in ['raw_spd_violation_rate','min_raw_eigenvalue']} for label in indices}
 if method.endswith('mlp'):
  from src.adam.mlp import MLPParams,AdamMLPModel,predict_covariance
  def params(prefix):return MLPParams(tuple(jnp.asarray(a[f'{prefix}_weight_{i}']) for i in range(3)),tuple(jnp.asarray(a[f'{prefix}_bias_{i}']) for i in range(3)))
  model=AdamMLPModel(params('drift'),params('covariance'),str(a['diff_type']))
 else:
  from src.adam.fourier import FourierParams,AdamFourierModel,predict_covariance
  def params(prefix):return FourierParams(jnp.asarray(a[prefix+'_omega']),jnp.asarray(a[prefix+'_amp']))
  model=AdamFourierModel(params('drift'),params('covariance'),str(a['diff_type']))
 result={}
 for label,idx in indices.items():
  count=0;minimum=float('inf')
  for chunk in np.array_split(idx, max(1,int(np.ceil(len(idx)/4096)))):
   eigenvalues=np.linalg.eigvalsh(np.asarray(predict_covariance(model,x[chunk])))
   count+=int(np.sum(eigenvalues[:,0]<=0));minimum=min(minimum,float(eigenvalues.min()))
  result[label]=dict(raw_spd_violation_rate=count/len(idx),min_raw_eigenvalue=minimum)
 return result

def paired(old,new):
 d=new-old
 return dict(mean_difference=float(d.mean()),sd_difference=float(d.std(ddof=1)),median_difference=float(np.median(d)),range_difference=[float(d.min()),float(d.max())],mean_change_percent=float(100*d.mean()/abs(old.mean())),percent_definition='100*(corrected mean-original mean)/abs(original mean), negative means lower',seeds_lower=int(np.sum(d<0)),seeds_equal=int(np.sum(d==0)),seeds_higher=int(np.sum(d>0)),differences=d.tolist(),leave_one_out_mean_difference_range=[float(((d.sum()-d)/29).min()),float(((d.sum()-d)/29).max())])

def main():
 assert not OUT.exists(),'Never overwrite an existing summary'
 plan=campaign.verify(STUDY/'campaigns/baseline')
 assert campaign.infra.read_json(STUDY/'campaigns/baseline/state.json')['status']=='complete'
 manifest=campaign.adapter.verify_registration()
 snapshot=json.loads((ROOT/'results/controlled_study_2026/float64_v2_validation/preserved_historical_sha256.json').read_text())
 for group in snapshot.values():
  for path,h in group.items():assert campaign.adapter.digest(ROOT/path)==h,path
 with np.load(ROOT/'results/ex8_joint_split_mlp_w27/test_distributions.npz') as z:old_arrays={k:z[k] for k in z.files}
 from src.experiments.dataset import load_dataset
 data=load_dataset(ROOT/manifest['dataset']['path']);indices=dict(train=data.train_idx,validation=data.validation_idx,test=data.test_idx)
 records=[];provenance={};spd=[]
 for job in plan['jobs']:
  directory,path,log,ctx=campaign.paths(job);checked=campaign.validate(job,path,log,ctx)
  complete=campaign.infra.read_json(directory/'complete.json')
  assert checked['artifact_sha256']==complete['artifact_sha256'] and checked['log_sha256']==complete['log_sha256']
  with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
  method,seed=job['method'],job['seed'];values=native_metrics(method,a,log)
  assert checked['parameter_count']==(1814 if method.endswith('mlp') else 1792)
  for split,v in values.items():records.append(dict(method=method,seed=seed,split=split,**v))
  for p in (path,log,ctx):provenance[str(p.relative_to(ROOT))]=campaign.adapter.digest(p)
  for version,aa in [('corrected',a),('original',None)]:
   if aa is None:
    p=ROOT/'results/production'/campaign.BASE_DIRS[method]/f'seed_{seed}_artifacts.npz'
    with np.load(p,allow_pickle=False) as z:aa={k:z[k] for k in z.files}
   for split,v in spd_from_model(method,aa,data.x,indices).items():spd.append(dict(version=version,method=method,seed=seed,split=split,**v))
  print('Validated metrics and SPD diagnostics',method,seed,flush=True)
 arrays={m+'_'+k:np.array([r[k] for r in records if r['method']==m and r['split']=='test']) for m in LABELS for k in METRICS}
 for m in LABELS:assert [r['seed'] for r in records if r['method']==m and r['split']=='test']==list(range(30))
 corrected={m:{k:stats.describe(arrays[m+'_'+k]) for k in METRICS} for m in LABELS}
 original={m:{k:stats.describe(old_arrays[OLD_KEYS[m]+'_'+k]) for k in METRICS} for m in LABELS}
 comparisons={m:{k:paired(old_arrays[OLD_KEYS[m]+'_'+k],arrays[m+'_'+k]) for k in METRICS} for m in LABELS}
 joint_split={family:{k:paired(arrays['joint_'+family+'_'+k],arrays['split_'+family+'_'+k]) for k in METRICS} for family in ['fourier','mlp']}
 spd_summary={v:{m:{split:{k:stats.describe([r[k] for r in spd if r['version']==v and r['method']==m and r['split']==split]) for k in ['raw_spd_violation_rate','min_raw_eigenvalue']} for split in indices} for m in LABELS} for v in ['original','corrected']}
 OUT.mkdir(exist_ok=False)
 summary=dict(dataset=manifest['dataset'],labels=LABELS,statistics=corrected,original_statistics=original,paired_original_to_corrected=comparisons,corrected_joint_to_split=joint_split,spd=spd_summary,source_sha256=provenance,script_sha256=campaign.adapter.digest(__file__),timing='Concurrent four-GPU accuracy campaign; timings are non-isolated; no isolated timing performed',metric_source='Native endpoint metrics: joint logs, split/ARFF artifacts. Original test vectors preserved exactly. Supplemental factor-model SPD diagnostics reconstructed on CPU with native functions, float32, and np.linalg.eigvalsh; ARFF native GPU raw SPD diagnostics retained.',interpretation='Only the data integration precision changed. Training precision/settings/checkpoint policies are unchanged. Paired NLL changes combine model changes and changed observed increments; drift/covariance targets and evaluation states are identical. Negative NLL percentage uses absolute original mean and is not a relative likelihood gain.')
 with (OUT/'summary.json').open('x') as f:json.dump(summary,f,indent=2)
 for name,rows in [('per_seed_split_metrics.csv',records),('per_seed_spd_diagnostics.csv',spd)]:
  keys=list(dict.fromkeys(k for r in rows for k in r))
  with (OUT/name).open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
 with (OUT/'test_distributions.npz').open('xb') as f:np.savez_compressed(f,seeds=np.arange(30),**arrays)
 lines=['# Corrected-data Experiment 8 capacity-matched baseline','', 'All 150 corrected artifacts validated; all accepted historical artifact/report/figure hashes unchanged. No retuning.','', '| Method | Metric | Original mean ± SD | Corrected mean ± SD | Corrected median [range] | Change | Seeds lower/higher |','|---|---|---|---|---|---|---|']
 for m in LABELS:
  for k in METRICS:
   o,n,d=original[m][k],corrected[m][k],comparisons[m][k]
   lines.append(f"| {LABELS[m]} | {k} | {o['mean']:.7g} ± {o['sd']:.5g} | {n['mean']:.7g} ± {n['sd']:.5g} | {n['median']:.7g} [{n['min']:.7g}, {n['max']:.7g}] | {d['mean_change_percent']:+.3f}% | {d['seeds_lower']}/{d['seeds_higher']} |")
 lines+=['','Sample SD uses ddof=1. Negative NLL percentage divides by the absolute original mean; it is not a likelihood improvement percentage. All covariance RMSE is raw/pre-projection. ARFF NLL uses its unchanged SPD projection; Adam uses its unchanged covariance factor and numerical convention. NLL changes include both changed fitted models and observations. No ranking-based rejection or hyperparameter change was made.','', '## Raw SPD diagnostics','']
 for m in LABELS:
  for split in indices:
   s=spd_summary['corrected'][m][split]
   lines.append(f"- {LABELS[m]}, {split}: violation mean {s['raw_spd_violation_rate']['mean']:.8g}, median {s['raw_spd_violation_rate']['median']:.8g}, max {s['raw_spd_violation_rate']['max']:.8g}; minimum raw eigenvalue over all seeds {s['min_raw_eigenvalue']['min']:.8g}.")
 with (OUT/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
 figure(arrays)
 print('\n'.join(lines),flush=True)

def figure(arrays):
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(17,5.7));colors=['#4477AA','#66CCEE','#228833','#AA3377','#CCBB44']
 labels=[name+'\n'+('1,814' if m.endswith('mlp') else '1,792')+' parameters' for m,name in LABELS.items()]
 for ax,k,logscale,title in zip(axes,['covariance_rmse','drift_rmse'],[True,False],['(a) Covariance','(b) Drift']):
  data=[arrays[m+'_'+k] for m in LABELS]
  boxes=ax.boxplot(data,tick_labels=labels,showfliers=False,patch_artist=True,medianprops={'color':'black','linewidth':1.6})
  for box,color in zip(boxes['boxes'],colors):box.set_facecolor(color);box.set_alpha(.25)
  rng=np.random.default_rng(2026)
  for i,(values,color) in enumerate(zip(data,colors),1):
   ax.scatter(i+rng.normal(0,.045,len(values)),values,s=19,alpha=.65,color=color,zorder=3)
   ax.scatter(i,values.mean(),marker='D',s=48,color=color,edgecolor='black',zorder=4)
  if logscale:ax.set_yscale('log')
  ax.set_ylabel('Test '+('covariance RMSE' if logscale else 'drift RMSE'));ax.grid(axis='y',alpha=.25);ax.tick_params(axis='x',labelsize=9);ax.set_title(title)
 fig.suptitle('Experiment 8: capacity-matched comparison',fontsize=14)
 fig.text(.5,.015,'30 seeds per method; points = runs, diamonds = means, boxes = quartiles and median.',ha='center',fontsize=9)
 fig.tight_layout(rect=(0,.06,1,.94))
 for ext in ('pdf','png'):
  with (OUT/('ex8_capacity_matched_five_methods_float64_v2_rmse_two_panel.'+ext)).open('xb') as f:fig.savefig(f,format=ext,dpi=300,bbox_inches='tight')
 plt.close(fig)
if __name__=='__main__':main()
