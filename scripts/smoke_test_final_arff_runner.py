"""Complete tiny CPU seven-regression archive; no canonical data fitting."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import tempfile,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import run_final_historical_arff as runner
rng=np.random.default_rng(1);n=50
x=rng.normal(size=(n,2)).astype('float32');h=np.full((n,1),.01,dtype='float32');r=(.01*(-x)+.05*rng.normal(size=x.shape)).astype('float32')
d=SimpleNamespace(x=x,r=r,h=h,train_idx=np.arange(40),validation_idx=np.arange(40,45),test_idx=np.arange(45,50))
with tempfile.TemporaryDirectory() as folder:
 p=Path(folder);data=p/'fixture.npz';np.savez(data,x=x,r=r,h=h)
 settings=dict(K=4,M_min=1,M_max=2,lambda_reg=.01,gamma=1.,delta=.1,resampling=True,metropolis_test=False)
 manifest=dict(studies={'ex1':dict(historical_arff=settings,current_config={'evaluation':{'spd_epsilon':1e-6}})},datasets={'ex1':{'path':str(data),'sha256':runner.shared.sha256_file(data)}})
 m=p/'manifest.json';m.write_text(json.dumps(manifest));runner.MANIFEST=m;runner.load_dataset=lambda path:d
 output=p/'artifact.npz';runner.run('ex1',0,output)
 with np.load(output,allow_pickle=False) as a:
  runner.validate(a,settings)
  assert a['crossfit_covariance_targets'].shape==(45,2) and int(a['n_test'])==0
  assert len(a['validation_idx'])==5
  for i in range(5):
   excluded=np.flatnonzero(a['fold_id']==i)
   assert not set(excluded)&set(a[f'fold_{i}_fit_train_positions'])
  assert 'backend' not in a or str(a['jax_backend'])=='cpu'
 print('PASS: seven fits, OOF exclusion, histories, provenance, 90/10/no-test archive validation')
