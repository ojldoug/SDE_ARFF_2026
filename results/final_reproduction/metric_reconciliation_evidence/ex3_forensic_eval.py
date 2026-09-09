import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,json,hashlib
from pathlib import Path
import numpy as np
sys.path[:0]=['.','GPU']
from src.experiments.definitions import get_experiment
root=Path('results/final_reproduction/production/accuracy/ex3/mlp_shallow');data=np.load('data/ex3.npz');rows=[]
for seed in range(30):
 with np.load(root/f'seed_{seed}_artifacts.npz',allow_pickle=False) as a:
  x=data['x_data'][a['validation_idx']];p={k:a['drift_'+k] for k in ['W1','W2','b1','b2']};pre=x@p['W1']+p['b1'];f=np.tanh(pre)@p['W2']+p['b2'];truth=np.asarray(get_experiment('ex3').drift(x))
  affine=(x@p['W1']*(1-np.tanh(p['b1'])**2))@p['W2']+np.tanh(p['b1'])@p['W2']+p['b2']
  projection=-1.6*x-1.5
  rows.append(dict(seed=seed,epoch=int(a['best_epoch']),argmin=int(np.argmin(a['validation_nll_history'])),manual_drift_rmse=float(np.sqrt(np.mean((f-truth)**2))),stored_rmse=float(a['validation_drift_rmse']),affine_approximation_rms=float(np.sqrt(np.mean((f-affine)**2))),population_linear_projection_rmse=float(np.sqrt(np.mean((projection-truth)**2))),preactivation_min=float(pre.min()),preactivation_max=float(pre.max()),weights_sha256=hashlib.sha256(b''.join(v.tobytes() for v in p.values())).hexdigest()))
print('max manual delta',max(abs(r['manual_drift_rmse']-r['stored_rmse']) for r in rows));print('distinct weight hashes',len(set(r['weights_sha256'] for r in rows)));print('epoch set',set(r['epoch'] for r in rows));print('seed0',rows[0]);print('affine departure range',min(r['affine_approximation_rms'] for r in rows),max(r['affine_approximation_rms'] for r in rows))
Path('results/final_reproduction/metric_reconciliation_evidence/ex3_reconstruction.json').write_text(json.dumps(rows,indent=2)+'\n')
