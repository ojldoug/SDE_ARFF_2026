import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
import sys,json,hashlib
import numpy as np
sys.path[:0]=['.','GPU']
from src.experiments.definitions import get_experiment
from lib import lib_Adam_tanh as mlp,lib_Adam_FF as ff,lib_ARFF as arff
import jax.numpy as j
root=Path('results/final_reproduction');out=root/'metric_reconciliation_evidence';records=[];detail={}
for ex in ['ex1','ex2','ex3','ex6','ex7']:
 data=np.load('data/'+('ex6_labels_v2' if ex=='ex6' else ex)+'.npz');x=data['x_data'];definition=get_experiment(ex)
 for directory in sorted((root/'production/accuracy'/ex).iterdir()):
  if not directory.is_dir():continue
  vals=[]
  for p in sorted(directory.glob('seed_*_artifacts.npz')):
   with np.load(p,allow_pickle=False) as a:
    nt=len(a['train_idx']);nv=len(a['validation_idx'])
    v=float(np.sqrt((nt*float(a['train_drift_rmse'])**2+nv*float(a['validation_drift_rmse'])**2)/(nt+nv)))
    vals.append(v);records.append(dict(experiment=ex,method=directory.name,seed=int(a['seed']),whole_population_drift_rmse=v))
  print('ALLPOINTS',ex,directory.name,np.mean(vals),np.std(vals,ddof=1),flush=True)
  # Exact recovered sigma evaluation on all current points, seed0; batch for memory.
  p=directory/'seed_0_artifacts.npz'
  with np.load(p,allow_pickle=False) as a:
   isar=directory.name=='arff_historical_corrected';lib=arff if isar else ff if directory.name=='fourier' else mlp
   dp={};cp={}
   for k in ['omega','amp','W1','W2','W3','b1','b2','b3']:
    if 'drift_'+k in a:dp[k]=j.asarray(a['drift_'+k])
    if 'covariance_'+k in a:cp[k]=j.asarray(a['covariance_'+k])
   sse=0.;count=0;nonfinite=0
   for start in range(0,len(x),4096):
    xx=j.asarray(x[start:start+4096])
    if isar and definition.diff_type=='symmetric':
     cov=np.asarray(lib.Functions.diffusion_cov(cp,xx,definition.diff_type));v,u=np.linalg.eigh(cov);pred=(u*np.sqrt(np.maximum(v,0))[:,None,:])@np.swapaxes(u,-1,-2)
    else:pred=np.asarray(lib.Functions.diffusion(cp,xx,definition.diff_type))
    true=np.asarray(definition.diffusion_factor(xx));err=(pred-true).astype('float64')
    nonfinite+=int(np.sum(~np.isfinite(err)));sse+=float(np.sum(err**2));count+=err.size
   detail[ex+'_'+directory.name]=dict(seed=0,population='all canonical rows',n=len(x),sigma_rmse=None if nonfinite else float(np.sqrt(sse/count)),nonfinite_entries=nonfinite)
   print('SIGMA',ex,directory.name,detail[ex+'_'+directory.name],flush=True)
(out/'whole_population_drift.json').write_text(json.dumps(records,indent=2)+'\n')
(out/'historical_sigma_seed0.json').write_text(json.dumps(detail,indent=2)+'\n')
