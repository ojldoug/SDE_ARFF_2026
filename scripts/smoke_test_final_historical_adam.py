#!/usr/bin/env python3
"""Tiny synthetic fixtures only; compare compatibility output with original historical updates."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,json,tempfile,hashlib,contextlib,io
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_final_historical_adam as runner
import jax
import jax.numpy as jnp
import importlib

def main():
 for name,method,dim in [('ex1','fourier',2),('ex1','mlp_deep',2),('ex3','mlp_shallow',10)]:
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);(root/'data').mkdir();(root/'results/final_reproduction').mkdir(parents=True)
   p=root/'data'/f'{name}.npz'
   x=np.linspace(-.5,.5,40*dim,dtype=np.float32).reshape(40,dim);r=np.zeros_like(x);h=np.full((40,1),.01,dtype=np.float32)
   np.savez(p,x_data=x,r_data=r,step_sizes=h,train_idx=np.arange(32),validation_idx=np.arange(32,36),test_idx=np.arange(36,40))
   spec=dict(K=4,epochs=2,batch_size=8,learning_rate=1e-4)
   (root/'results/final_reproduction/production_manifest.adam_v1.json').write_text(json.dumps(dict(studies={name:dict(historical_adam=spec)},datasets={name:dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest())})))
   out=root/'fixture.npz'
   with patch.object(runner,'ROOT',root),contextlib.redirect_stdout(io.StringIO()):runner.run(name,method,0,out)
   lib=importlib.import_module('lib.lib_Adam_FF' if method=='fourier' else 'lib.lib_Adam_tanh')
   hidden=(2,2) if method=='mlp_deep' else (4,)
   _,kd,kc=jax.random.split(jax.random.PRNGKey(0),3)
   typ=runner.get_experiment(name).diff_type
   dp=lib.init_drift_params(kd,hidden,dim,dim);cp=lib.init_diffusion_params(kc,hidden,dim,dim,typ)
   hp=dict(epochs=2,batch_size=8,learning_rate=1e-4,layer_widths=hidden);lib.AdamTrain.set_opt(hp)
   with contextlib.redirect_stdout(io.StringIO()):_,_,_,_,losses=lib.AdamTrain.training_loop(hp,dp,cp,jnp.asarray(x),jnp.asarray(r),jnp.asarray(h),typ,.1,plot=False)
   with np.load(out,allow_pickle=False) as z:
    np.testing.assert_array_equal(z['validation_nll_history'],losses)
    assert int(z['best_epoch'])==int(np.argmin(losses)) and int(z['n_test'])==0
    assert len(z['train_idx'])==36 and len(z['validation_idx'])==4
    assert int(z['epochs'])==2
    if name=='ex3':assert z['covariance_convention'].item()=='Sigma=(L L^T)^2'
   print('PASS exact historical validation-history comparison:',name,method)
 print('Tiny fixture checks passed; no canonical dataset or production artifact changed.')
if __name__=='__main__':main()
