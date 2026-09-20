"""Package frozen Experiment 8 summaries; no model evaluation or fitting."""
import csv, hashlib, json, shutil, subprocess
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
O=R/'results/final_ex8_publication_bundle'
METHODS=['joint_fourier','split_fourier','arff','joint_mlp','split_mlp']
LABELS=['Joint Fourier Adam','Split Fourier Adam','ARFF','Joint MLP Adam','Split MLP Adam']
COLORS=['#0072B2','#56B4E9','#009E73','#D55E00','#CC79A7']
SOURCES={'baseline':'results/controlled_study_2026/float64_v2/baseline_summary','capacity':'results/capacity_regime_ex8_v2/capacity_summary','N':'results/capacity_regime_ex8_v2/N_summary','h':'results/capacity_regime_ex8_final_h_v1/h_summary'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def csvout(p,rows):
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 O.mkdir(exist_ok=False);(O/'figures').mkdir();refs={};data={};summaries={}
 for name,src in SOURCES.items():
  p=R/src;j=json.loads((p/'summary.json').read_text());summaries[name]=j
  for path,digest in j['source_sha256'].items():
   q=Path(path);q=q if q.is_absolute() else R/q
   assert sha(q)==digest, str(q)
   refs[str(q.relative_to(R))]=digest
  dest=O/'summaries'/name;dest.mkdir(parents=True)
  for q in p.iterdir():
   if q.suffix in ['.md','.json','.csv']:shutil.copy2(q,dest/q.name)
  rows=list(csv.DictReader(next(p.glob('per_seed*metrics.csv')).open()))
  if name=='baseline':
   rows=[r for r in rows if r['split']=='test']
   rows=[dict(r,**{'test_'+k:r[k] for k in ['drift_rmse','covariance_rmse','nll','raw_spd_violation_rate','min_raw_eigenvalue']},N='80000',h='0.0001',K='128',parameters=str(1814 if 'mlp' in r['method'] else 1792)) for r in rows]
  assert len(rows)=={'baseline':150,'capacity':250,'N':200,'h':450}[name]
  data[name]=rows
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':220})
 captions={'baseline':'Corrected float64 baseline: N=80,000, h=0.0001; 30 frozen seeds; approximately matched capacities, not identical procedures.', 'capacity':'Performance versus model capacity at fixed N=640,000, h=0.0001; Phase 1 did not establish a capacity-dominated regime.', 'N':'N dependence at K=1024, h=0.0001; no universal N-plateau/equivalence was established.', 'h':'Nine-lag diagnostic at K=1024, N=640,000. h=0.002 is the bounded endpoint from the pre-existing 0.001 oracle drift-bias tolerance, not a performance-selected ARFF optimum.'}
 def figure(name,metrics,suffix):
  rows=data[name];fig,axs=plt.subplots(1,len(metrics),figsize=(8.7,3.9),squeeze=False);stats=[]
  for ax,(metric,ylab) in zip(axs[0],metrics):
   for i,m in enumerate(METHODS):
    rr=[r for r in rows if r['method']==m]
    points=[128] if name=='baseline' else sorted({float(r[{'capacity':'K','N':'N','h':'h'}[name]]) for r in rr})
    xs=[];ys=[];ss=[]
    for point in points:
     sub=rr if name=='baseline' else [r for r in rr if float(r[{'capacity':'K','N':'N','h':'h'}[name]])==point]
     assert sorted(int(r['seed']) for r in sub)==list(range(30 if name=='baseline' else 10))
     vals=np.array([float(r['test_'+metric]) for r in sub]);x=i if name=='baseline' else float(sub[0]['parameters']) if name=='capacity' else point
     xs.append(x);ys.append(float(vals.mean()));ss.append(float(vals.std(ddof=1)))
     stats.append(dict(method=m,point=point,parameters=int(sub[0]['parameters']),metric=metric,n=len(vals),mean=ys[-1],sample_sd=ss[-1],median=float(np.median(vals)),minimum=float(vals.min()),maximum=float(vals.max())))
     if name=='baseline':
      ax.boxplot([vals],positions=[i],widths=.5,showfliers=False,patch_artist=True,boxprops={'facecolor':COLORS[i],'alpha':.22},medianprops={'color':'black'})
      ax.scatter(i+np.random.default_rng(123+i).uniform(-.16,.16,len(vals)),vals,s=9,color=COLORS[i],alpha=.65)
      ax.scatter(i,vals.mean(),marker='D',s=26,color=COLORS[i],edgecolor='black',linewidth=.5,zorder=4)
    if name!='baseline':ax.errorbar(xs,ys,yerr=ss,marker=['o','s','D','^','v'][i],ms=4,lw=1.2,capsize=2,color=COLORS[i],label=LABELS[i])
   ax.set_ylabel(ylab);ax.grid(alpha=.2)
   if name=='baseline':ax.set_xticks(range(5),[s.replace(' Adam','\nAdam')+'\n'+('1,814' if 'MLP' in s else '1,792')+'\nparameters' for s in LABELS],fontsize=7)
   else:
    ax.set_xscale('log');ax.set_xlabel({'capacity':'Final retained parameter count','N':'Nominal training samples N','h':'Observation lag h'}[name])
  if name!='baseline':fig.legend(*axs[0,0].get_legend_handles_labels(),loc='lower center',ncol=3,frameon=False,fontsize=8)
  fig.suptitle({'baseline':'Experiment 8: capacity-matched comparison','capacity':'Experiment 8: performance versus capacity','N':'Experiment 8: sample-count dependence','h':'Experiment 8: bounded observation-lag diagnostic'}[name],fontsize=11)
  fig.tight_layout(rect=(0,.14 if name!='baseline' else .02,1,.95))
  stem=O/'figures'/f'ex8_{name}_{suffix}'
  for ext in ['pdf','png']:fig.savefig(str(stem)+'.'+ext,bbox_inches='tight')
  plt.close(fig);csvout(Path(str(stem)+'.csv'),stats)
  dump(Path(str(stem)+'.json'),{'caption':captions[name]+' All coefficient covariance RMSE is raw/pre-projection. Uncertainty: sample SD across seeds; no repetition sets are pooled. Parallel timings are non-isolated.','statistics':stats,'per_seed_input':rows,'source_sha256':summaries[name]['source_sha256']})
 for name in SOURCES:figure(name,[('drift_rmse','Test drift RMSE'),('covariance_rmse','Test raw covariance RMSE')],'rmse')
 figure('capacity',[('nll','Test Gaussian NLL')],'nll')
 figure('h',[('nll','Test Gaussian NLL'),('nll_minus_log_h',r'Test NLL $-\log h$')],'nll')
 spd=[]
 for name,rows in data.items():
  for r in rows:
   if r['method']=='arff':spd.append(dict(study=name,seed=r['seed'],K=r['K'],N=r['N'],h=r['h'],raw_spd_violation_rate=r['test_raw_spd_violation_rate'],minimum_raw_eigenvalue=r['test_min_raw_eigenvalue']))
 csvout(O/'figures/arff_raw_spd.csv',spd);dump(O/'figures/arff_raw_spd.json',{'data':spd,'source_sha256':refs,'note':'Raw predictions; violation rate is not hidden by projection.'})
 # Preserve original reports and lightweight provenance, avoiding mutable run logs.
 roots=['results/capacity_regime_ex8_v1','results/capacity_regime_ex8_v2','results/capacity_regime_ex8_final_h_v1','results/controlled_study_2026/float64_v2_validation','results/controlled_study_2026/float64_v2/hybrid_jointmlp_arff','results/controlled_study_2026/float64_v2/oracle_component_swap','results/controlled_study_2026','results/controlled_study_2026/float64_v2']
 for root in roots:
  for p in (R/root).iterdir():
   if p.is_file() and p.suffix in ['.json','.md','.csv']:
    dest=O/'provenance'/p.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
 accounts=[]
 for n in [80000,160000,320000,640000]:
  accounts.append(dict(nominal_N=n,validation=10000,test=10000,adam_final=n,adam_oof_fit=int(.8*n),arff_final_fit=int(.9*n),arff_final_internal_validation=int(.1*n),arff_oof_fit=int(.72*n),arff_oof_internal_validation=int(.08*n),oof_heldout=int(.2*n)))
 csvout(O/'sample_accounting.csv',accounts)
 csvout(O/'parameter_counts.csv',[dict(K=k,fourier_parameters=14*k,mlp_hidden_width=w,mlp_hidden_layers=2,mlp_parameters=2*w*w+13*w+5) for k,w in [(64,18),(128,27),(256,39),(512,57),(1024,81)]])
 for root in roots[:3]:
  for p in (R/root/'data').glob('*.json'):
   dest=O/'provenance'/p.relative_to(R);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)

 dump(O/'large_file_references.json',{'artifacts_and_authenticated_logs':refs,'datasets':{p:sha(R/p) for p in ['data/ex8.npz','data/ex8_float64_v2.npz']},'note':'Large models/data remain at original repository-relative paths; backup preserves their bytes.'})
 dump(O/'source_commit.json',{'source_head_before_publication_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'branch':subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip(),'builder':str(Path(__file__).relative_to(R)),'builder_sha256':sha(Path(__file__))})
 (O/'FINAL_EX8_RESULTS_README.md').write_text('''# Final authenticated Experiment 8 results

Corrected float64 data/results supersede earlier float32 Experiment 8 numerical outputs for scientific conclusions. Historical/superseded artifacts remain preserved for provenance.

The corrected baseline uses N=80,000 and 30 seeds. Capacity, N and h studies use separately documented regimes and ten seeds. Do not pool repetition sets. All figures use frozen test metrics. Covariance coefficient RMSE is raw, before SPD projection. ARFF requires the accepted 1e-3 eigenvalue floor for Gaussian NLL. NLL measures finite-lag likelihood/calibration, not coefficient accuracy alone.

Phase 1 did not establish capacity dominance or universal N/h equivalence. The capacity sweep is descriptive at fixed N=640,000, h=1e-4. No empirical O(1/K) claim is supported. The nine-lag endpoint h=.002 is the authorized endpoint governed by the pre-existing oracle drift-bias tolerance, not an ARFF performance optimum.

The ARFF drift-K consistency audit passes: verified scientific result. Frequent early selected checkpoints are authentic; no checkpoint/loading or target mismatch was found. RMSE averages squared error over samples AND output coordinates (not a vector-norm sum). Training remains float32; corrected integrations/increments were archived in float64. Parallel timings are non-isolated.

Experiment 8 is numerically closed unless a future manuscript claim requires genuinely new evidence. Poster and manuscript were not changed.

## Contents and reproduction

figures/: vector PDFs, PNGs, exact aggregated CSV and JSON containing per-seed input, captions and authenticated source hashes. arff_raw_spd.csv/json explicitly retain every raw violation rate and minimum eigenvalue. summaries/: frozen original reports and metrics. provenance/: immutable audit, selection, study plans/manifests, corrected-data validation, hybrid and oracle-floor diagnostic reports. sample_accounting.csv distinguishes nominal from actual fitting counts. large_file_references.json identifies models and data, without duplicating them here. source_commit.json identifies the source HEAD before this publication commit. SHA256_MANIFEST.json authenticates this bundle (excluding itself).

Run scripts/build_final_ex8_publication_bundle.py only into a nonexistent destination; it verifies source hashes and seed coverage without evaluating or fitting models. results/ is intentionally Git-ignored and remains excluded from the commit. The tracked docs/final_ex8_publication.md points to this bundle; its figures and essential large artifacts are preserved in the server backup.
''')
 dump(O/'SHA256_MANIFEST.json',{str(p.relative_to(O)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(O.rglob('*')) if p.is_file()})
 print('Created',O)
if __name__=='__main__':main()
