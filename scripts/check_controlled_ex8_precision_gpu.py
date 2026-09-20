"""Paired arithmetic replay of existing canonical observations; no estimator fitting."""
import os
os.environ['JAX_PLATFORMS']='cuda'
from pathlib import Path
import json,time
import jax,jax.numpy as jnp,numpy as np
jax.config.update('jax_enable_x64',True)
ROOT=Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0,str(ROOT))
from src.experiments.definitions import ex8_drift,ex8_diffusion_factor
OUT=ROOT/'results/controlled_study_2026/preflight';n=4096
with np.load(ROOT/'data/ex8.npz',allow_pickle=False) as z:x=np.array(z['x_data'][:n]);stored=np.array(z['r_data'][:n])
key_x,key_noise=jax.random.split(jax.random.PRNGKey(0))
# Preserve the original full random-array shape, then take an analysis subset.
print('Generating original-shaped float32 noise on GPU',flush=True)
noise=jax.random.normal(key_noise,shape=(1000,100000,2),dtype=jnp.float32)[:,:n,:];noise.block_until_ready()
print('Replaying paired float32 and float64 arithmetic',flush=True)
def replay(dtype):
 xx=jnp.asarray(x,dtype=dtype);zz=jnp.asarray(noise,dtype=dtype);dt=jnp.asarray(1e-7,dtype=dtype)
 def step(state,z):
  drift=ex8_drift(state);sigma=ex8_diffusion_factor(state)
  dw=jnp.sqrt(dt)*z;inc=jnp.einsum('nij,nj->ni',sigma,dw)
  return state+dt*drift+inc,None
 end,_=jax.lax.scan(step,xx,zz);return np.array(end-xx)
jax.config.update('jax_enable_x64',False)
a=replay(jnp.float32)
jax.config.update('jax_enable_x64',True)
b=replay(jnp.float64);h=1e-4
D=(a-b)/h;X=np.column_stack([np.ones(n),x]);coeff=np.linalg.lstsq(X,D,rcond=None)[0];fitted=X@coeff
result=dict(N=n,backend='GPU; actual stored GPU data compared explicitly',noise='original PRNG key/full1000×100000×2 float32 shape, same draws cast to float64 for comparison',float32_replay_vs_stored_max_increment_error=float(abs(a-stored).max()),float32_replay_vs_stored_rms_increment_error=float(np.sqrt(np.mean((a-stored)**2))),paired_float32_minus_float64_time_normalized_rmse=float(np.sqrt(np.mean(D**2))),paired_difference_mean=D.mean(0).tolist(),paired_difference_linear_coefficients_rows_intercept_x0_x1=coeff.tolist(),linear_component_rmse=float(np.sqrt(np.mean(fitted**2))),residual_rmse=float(np.sqrt(np.mean((D-fitted)**2))),caveat='GPU replay is compared directly with original saved GPU increments; all comparisons use the same underlying normal draws.')
with (OUT/'matched_noise_precision_gpu.json').open('x') as f:json.dump(result,f,indent=2)
with (OUT/'matched_noise_precision_gpu.npz').open('xb') as f:np.savez_compressed(f,x=x,stored=stored,cpu32=a,cpu64=b)
print(json.dumps(result,indent=2),flush=True)
