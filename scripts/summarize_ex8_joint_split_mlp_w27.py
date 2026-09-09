"""Validate all150 accepted artifacts and create a separate five-method poster candidate."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['MPLBACKEND']='Agg'
import sys,json,csv
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import summarize_ex8_capacity_matched as old
import run_ex8_split_mlp_w27_campaign as campaign
ROOT=old.ROOT;OUT=ROOT/'results/ex8_joint_split_mlp_w27'
LABELS={'joint':'Joint Fourier Adam','split':'Split Fourier Adam','arff':'ARFF','mlp':'Joint MLP Adam','split_mlp':'Split MLP Adam'}

def main():
 plan=campaign.verify()
 if campaign.infra.read_json(campaign.CONTROL/'state.json')['status']!='complete':raise RuntimeError('Campaign is not complete')
 records,hashes=old.collect()
 with np.load(old.OUT/'test_distributions.npz',allow_pickle=False) as z:
  for m in old.LABELS:
   for k in old.METRICS:np.testing.assert_array_equal([r[k] for r in records if r['method']==m],z[m+'_'+k])
 for job in plan['jobs']:
  a,l,_=campaign.paths(job);campaign.validate(job,a,l)
  with np.load(a,allow_pickle=False) as z:
   from src.arff.two_stage import make_folds
   expected_fold=np.full(80000,-1,dtype=np.int32)
   for i,idx in enumerate(make_folds(80000,5,2026)):expected_fold[idx]=i
   np.testing.assert_array_equal(z['fold_id'],expected_fold)
   records.append(dict(method='split_mlp',label=LABELS['split_mlp'],seed=job['seed'],**{k:float(z['test_'+k]) for k in old.METRICS},algorithm_time=float(z['algorithm_time'])))
  for p in (a,l):hashes[str(p.relative_to(ROOT))]=campaign.digest(p)
 arrays={m+'_'+k:np.array([r[k] for r in records if r['method']==m]) for m in LABELS for k in old.METRICS}
 stats={m:{k:old.describe(arrays[m+'_'+k]) for k in old.METRICS} for m in LABELS}
 paired={}
 for k in old.METRICS:
  j,s=arrays['mlp_'+k],arrays['split_mlp_'+k];d=s-j
  paired[k]=dict(mean_difference=float(d.mean()),mean_change_percent=float(100*d.mean()/abs(j.mean())),percent_definition='100*(Split mean-Joint mean)/abs(Joint mean); negative means lower',seeds_lower=int(np.sum(s<j)),seeds_equal=int(np.sum(s==j)),seeds_higher=int(np.sum(s>j)),median_paired_difference=float(np.median(d)),min_paired_difference=float(d.min()),max_paired_difference=float(d.max()),paired_differences=d.tolist(),leave_one_out_mean_difference_range=[float(((d.sum()-d)/29).min()),float(((d.sum()-d)/29).max())])
 OUT.mkdir(exist_ok=False)
 with (OUT/'summary.json').open('x') as f:json.dump(dict(labels=LABELS,statistics=stats,paired_joint_to_split_mlp=paired,source_sha256=hashes,summary_script_sha256=campaign.digest(Path(__file__)),timing_note='Non-isolated four-GPU accuracy campaign; not equal total compute',interpretation='Modern joint versus five-fold cross-fitted two-stage training; same final architecture and optimizer settings. Not an Owen historical replication.'),f,indent=2)
 with (OUT/'test_distributions.npz').open('xb') as f:np.savez_compressed(f,seeds=np.arange(30),**arrays)
 with (OUT/'per_seed_test_metrics.csv').open('x',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
 lines=['All150 artifacts validated; all120 prior accepted metric vectors unchanged.','']
 for m in LABELS:
  lines.append(LABELS[m])
  for k,v in stats[m].items():lines.append(f"  {k}: {v['mean']:.9g} +/- {v['sd']:.9g}; median {v['median']:.9g}; range [{v['min']:.9g}, {v['max']:.9g}]")
 lines.append(json.dumps(paired,indent=2));(OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(17,5.7));colors=['#4477AA','#66CCEE','#228833','#AA3377','#CCBB44']
 labels=[name+'\n'+('1,814' if m in ('mlp','split_mlp') else '1,792')+' parameters' for m,name in LABELS.items()]
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
  with (OUT/('ex8_capacity_matched_five_methods_rmse_two_panel.'+ext)).open('xb') as f:fig.savefig(f,format=ext,dpi=300,bbox_inches='tight')
 plt.close(fig)
 print('\n'.join(lines));print('Saved:',OUT)
if __name__=='__main__':main()
