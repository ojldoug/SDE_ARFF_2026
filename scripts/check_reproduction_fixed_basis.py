#!/usr/bin/env python3
"""Two fixed-basis native solves only; no adaptation or selection."""
import argparse, hashlib, importlib.metadata, json, os, sys, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(exist_ok=False,parents=True);sys.path.insert(0,str(a.checkout.resolve()))
import numpy as np
import jax,jax.numpy as jnp
from src.arff.regression import fit_amplitudes,fourier_features
from src.experiments.dataset import load_dataset
assert not jax.config.jax_enable_x64
z=dict(np.load(a.artifact,allow_pickle=False)); data_path=a.checkout/'data/ex8_float64_v2.npz'
assert hashlib.sha256(data_path.read_bytes()).hexdigest()==str(z['dataset_sha256'])
data=load_dataset(data_path);idx=z['train_idx'];pos=z['final_drift_fit_train_positions']
x,r,h=(jnp.asarray(v[idx]) for v in [data.x,data.r,data.h]); y=(r/h)[pos];x=x[pos];omega=jnp.asarray(z['final_drift_omega'])
assert x.shape==(72000,2) and omega.shape==(2,128) and float(z['lambda_reg'])==.001
solve=jax.jit(fit_amplitudes);out={};start=time.monotonic()
for repeat in range(2):
 amp=solve(x,y,omega,.001);amp.block_until_ready();out[f'amp_{repeat}']=np.asarray(amp)
# Independent recomputation, not a trace of the native fused adaptation kernel.
phi=fourier_features(omega,x);gram=jnp.matmul(phi.T,phi,precision=jax.lax.Precision.HIGHEST);rhs=jnp.matmul(phi.T,y,precision=jax.lax.Precision.HIGHEST)
A=gram+x.shape[0]*.001*jnp.eye(256,dtype=phi.dtype)
out.update(gram=np.asarray(gram),rhs=np.asarray(rhs),regularized_gram=np.asarray(A),omega=np.asarray(omega))
for repeat in range(2):out[f'pred_{repeat}']=np.asarray(phi@jnp.asarray(out[f'amp_{repeat}']))
out['pred_archived']=np.asarray(phi@jnp.asarray(z['final_drift_amp']))
def digest(v):return hashlib.sha256(np.asarray(v).tobytes()).hexdigest()
residuals={}
for name,b in [('archived',z['final_drift_amp']),('repeat_0',out['amp_0']),('repeat_1',out['amp_1'])]:
 aa=np.asarray(A,dtype=np.float64);bb=np.asarray(b,dtype=np.float64);rr=np.asarray(rhs,dtype=np.float64);res=aa@bb-rr
 residuals[name]=dict(absolute_frobenius=float(np.linalg.norm(res)),relative_rhs=float(np.linalg.norm(res)/np.linalg.norm(rr)),normwise_backward_error=float(np.linalg.norm(res)/(np.linalg.norm(aa)*np.linalg.norm(bb)+np.linalg.norm(rr))))
packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
record=dict(executable=sys.executable,python=sys.version,packages=packages,jax_backend=jax.default_backend(),devices=[str(d) for d in jax.devices()],jax_x64=jax.config.jax_enable_x64,matmul_precision=str(jax.config.jax_default_matmul_precision),environment={k:v for k,v in os.environ.items() if k.startswith(('CUDA','CUBLAS','CUDNN','NVIDIA','JAX','XLA','OMP','OPENBLAS','TF_DETERMINISTIC'))},input_hashes={k:digest(v) for k,v in dict(x=x,y=y,omega=omega,features=phi).items()},source_sha256=hashlib.sha256((a.checkout/'src/arff/regression.py').read_bytes()).hexdigest(),array_hashes={k:digest(v) for k,v in out.items()},residuals=residuals,elapsed_seconds=time.monotonic()-start,solve_calls=2)
np.savez_compressed(a.output/'arrays.npz',**out)
# Record actually mapped CUDA/XLA libraries, rather than claiming wheel versions prove loading.
libs=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/' in line and any(s in line.lower() for s in ['cuda','cublas','cusolver','cudnn','jaxlib','pjrt'])})
record['loaded_libraries']=[dict(path=s,sha256=hashlib.sha256(Path(s).read_bytes()).hexdigest()) for s in libs if Path(s).is_file()]
(a.output/'record.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
