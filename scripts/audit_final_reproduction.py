#!/usr/bin/env python3
"""Read-only source/data audit; exclusively creates a new audit bundle, never trains."""
import os
os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import sys,json,hashlib,subprocess
from dataclasses import asdict
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.experiments.dataset import load_dataset,make_split_indices
from src.experiments.config import get_config
OUT=ROOT/'results/final_reproduction'
REF=Path('/home/kammonaa/projects/SDE_ARFF/reference/sde-identification')
TEX=Path('/home/kammonaa/projects/SDE_NN_overleaf/arXiv_2026/RaulCom_ARFFSDELearning.tex')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(name,v):
 with (OUT/name).open('x') as f:json.dump(v,f,indent=2);f.write('\n')
# From primary manuscript Appendix B Tables 10–11, confirmed active editable appendices.
ARFF=[(256,.05,.001,30,True,False),(128,.01,.00001,20,True,False),(1024,.01,.01,50,True,False),(128,.1,.001,20,True,False),(256,.1,.002,20,True,False),(512,.3,.001,50,True,False),(512,.5,.01,50,False,True)]
ADAM=[(800,.001,512),(100,.0001,256),(300,.0001,1024),(100,.0001,256),(100,.0001,256),(600,.001,512),(300,.0001,512)]
def main():
 datasets={};studies={};inventory={}
 for i in range(1,9):
  name=f'ex{i}';p=ROOT/'data'/f'{name}.npz';m=json.loads(p.with_suffix('.json').read_text());d=load_dataset(p)
  actual=sha(p);assert actual==m['dataset_sha256'],name
  for a,b in zip((d.train_idx,d.validation_idx,d.test_idx),make_split_indices(len(d.x),get_config(name).split)):np.testing.assert_array_equal(a,b)
  datasets[name]=dict(path=str(p.relative_to(ROOT)),sha256=actual,metadata=m,
    structural_validation='PASS: finite arrays, positive lags, disjoint exhaustive indices, deterministic saved split, metadata checksum',
    h_min=float(d.h.min()),h_max=float(d.h.max()),
    scientific_reuse=('NO as-is: half-lag initialization and alternating spatial-label mismatch' if i==6 else 'PENDING Langevin sign/conditioning reconciliation' if i==4 else 'YES subject to final implementation audit; no regeneration indicated'))
  if i<8:
   k,delta,lam,mmin,res,metro=ARFF[i-1];epochs,lr,batch=ADAM[i-1]
   studies[name]=dict(historical_arff=dict(K=k,delta=delta,lambda_reg=lam,M_min=mmin,M_max=1000,resampling=res,metropolis_test=metro,gamma=1),
    historical_adam=dict(K=k,epochs=epochs,learning_rate=lr,batch_size=batch,shallow_hidden_sizes=[k],deep_hidden_sizes=[k//2,k//2]),
    repetitions=30,current_config=asdict(get_config(name)),
    production_status='NOT_DISPATCHED: audit hard-stop on canonical ex6 label invalidity; estimator/checkpoint reconciliation pending',
    evidence=['Latest_manuscrip_draft.pdf Tables 4–6,8–11','current arXiv_2026/appendices_part2.tex and appendices_part3.tex'],
    methods_shown_in_historical_table=(['arff','adam_tanh_shallow'] if i==3 else ['arff','adam_fourier','adam_tanh_shallow','adam_tanh_deep']))
  else:studies[name]=dict(production_status='FROZEN_ACCEPTED_REUSE',repetitions=30,fourier_K=128,fourier_parameters=1792,mlp_hidden_sizes=[27,27],mlp_parameters=1814,
    evidence='Explicit user acceptance; results/capacity_matched_ex8_w27/summary.json')
  for method in ('arff','adam','mlp','adam_split'):
   directory=ROOT/'results/production'/f'{method}_{name}'
   records=[]
   for artifact in sorted(directory.glob('seed_*_artifacts.npz')):
    with np.load(artifact,allow_pickle=False) as z:
     fields=['artifact_version','method','experiment','seed','fourier_frequencies','K','M_min','M_max','lambda_reg','delta','resampling','metropolis_test','epochs','epochs_per_regression','learning_rate','batch_size','hidden_width','hidden_layers','n_folds']
     records.append(dict(path=str(artifact.relative_to(ROOT)),sha256=sha(artifact),settings={k:z[k].item() for k in fields if k in z and z[k].ndim==0}))
   if records:inventory[f'{method}_{name}']=records
 for var in ('N','h','K'):
  studies[f'section6_2_{var}']=dict(status='UNRESOLVED_NO_DISPATCH',repetitions=10,
    fixed_K=512 if var!='K' else None,
    drift='Experiment 7 localized broad-spectrum drift',diffusion_factor='diag(0.25*x0**2+0.25,0.25*x1**2+0.25); covariance is its square',
    grid=None,missing=['exact grid','fixed N and h','study-specific optimizer settings and seeds','test-set generation protocol']+(['NLL_infinity regression family, fitting domain, weights, treatment across runs/methods'] if var=='K' else []),
    excess_definition='NLL(K)-NLL_infinity' if var=='K' else 'NLL_learned-NLL_oracle on same test set',
    evidence='Primary PDF Section 6.2 and Appendix A/Table 6; active TeX lines 1413–1516; available images do not identify exact grids')
 sources=[REF/'Latest_manuscrip_draft.pdf',REF/'Raul_feedback.pdf',TEX,TEX.parent/'appendices_part2.tex',TEX.parent/'appendices_part3.tex']
 sources+=list((ROOT/'src').rglob('*.py'))+[ROOT/'GPU/lib/lib_ARFF.py',ROOT/'GPU/lib/lib_Adam_FF.py',ROOT/'GPU/lib/lib_Adam_tanh.py',ROOT/'GPU/generate_SPDE_data.ipynb',ROOT/'scripts/audit_final_reproduction.py']
 manifest=dict(manifest_version=1,status='AUDIT_HARD_STOP_EX6_DATA_LABELS; NOT A DISPATCH MANIFEST',
    authorization='No tuning. Reuse blocker-compliant existing code/data. Stop on invalid previously accepted core experiment.',
    source_sha256={str(p):sha(p) for p in sources},datasets=datasets,studies=studies,
    global_split=dict(current='stored deterministic 80/10/10 seed 2026',historical='90/10; surviving notebooks reshuffle split across repetitions',decision='Preserve canonical files unchanged; document existing reproducibility correction. No silent resplit.'),
    publication_timings='Parallel accuracy timings not isolated. Isolated protocol deferred until resolved accuracy production and scientific hard-stop cleared.',
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
    git_status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True))
 write('production_manifest.json',manifest)
 write('evidence/existing_production_inventory.json',inventory)
 print('Saved immutable audit manifest and',sum(map(len,inventory.values())),'historical/current production artifact inventory records.')
if __name__=='__main__':main()
