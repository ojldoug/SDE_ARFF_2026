#!/usr/bin/env python3
"""Append independent corrected observations; preserve original coupled data exactly."""
import os
os.environ.setdefault('JAX_PLATFORMS','cuda')
from pathlib import Path
import sys,json,hashlib
import numpy as np
import jax,jax.numpy as jnp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiments.definitions import ex8_drift,ex8_diffusion_factor
from src.experiments.dataset import load_dataset
OUT=ROOT/'results/capacity_regime_ex8_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
def advance(x,noise,delta=1e-7):
 def step(s,z):
  dt=jnp.asarray(delta,dtype=jnp.float64)
  return s+dt*ex8_drift(s)+jnp.einsum('nij,nj->ni',ex8_diffusion_factor(s),jnp.sqrt(dt)*z.astype(jnp.float64)),None
 return jax.lax.scan(step,x,noise)[0]
def main():
 m=json.loads((OUT/'manifest.json').read_text());dest=OUT/'data';dest.mkdir(exist_ok=False)
 parent=ROOT/'results/controlled_study_2026/float64_v2';lag=json.loads((parent/'lag_datasets/complete.json').read_text())
 old={h:load_dataset(parent/lag['dataset_roots'][format(h,'.8g')]/'data/ex8.npz') for h in m['h_grid']}
 for h,d in old.items():assert sha(parent/lag['dataset_roots'][format(h,'.8g')]/'data/ex8.npz')==lag['records'][format(h,'.8g')]['dataset_sha256']
 jax.config.update('jax_enable_x64',True)
 n=80000;extensions=[];draw_hashes=[]
 for b in range(7):
  key=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(0),20260913),b);kx,kn=jax.random.split(key)
  x0=jax.random.uniform(kx,(n,2),dtype=jnp.float32,minval=-2.,maxval=2.);s=x0.astype(jnp.float64);start=s;ends={};hashes=[]
  for t in range(4):
   z=jax.random.normal(jax.random.fold_in(kn,t),(1000,n,2),dtype=jnp.float32)
   hashes.append(hashlib.sha256(np.asarray(z).tobytes()).hexdigest())
   if t==0:
    offset=0
    for stop in [250,500,1000]:
     s=advance(s,z[offset:stop]);ends[stop]=np.asarray(s-start);offset=stop
   else:
    s=advance(s,z)
    if t in (1,3):ends[(t+1)*1000]=np.asarray(s-start)
  path=dest/f'extension_block_{b}.npz'
  arrays=dict(x_data=np.asarray(x0),**{f'r_{steps}':v for steps,v in ends.items()})
  assert all(np.isfinite(v).all() for v in arrays.values()) and all(v.dtype==np.float64 for k,v in arrays.items() if k.startswith('r_'))
  with path.open('xb') as f:np.savez_compressed(f,**arrays)
  extensions.append(path);draw_hashes.append(dict(block=b,path=str(path),sha256=sha(path),normal_sha256=hashes))
  print('Validated extension block',b,flush=True)
 # Deterministic zero-noise update and dtype check on same distribution.
 xx=np.asarray(x0[:1024],dtype=np.float64);state=jnp.asarray(xx)
 state=jax.lax.fori_loop(0,1000,lambda i,s:s-jnp.asarray(1e-7,dtype=jnp.float64)*s,state)
 reference=xx*np.expm1(1000*np.log1p(-1e-7));error=float(np.sqrt(np.mean(((np.asarray(state)-xx-reference)/1e-4)**2)))
 assert error<1e-8
 # Coupled refinement on4096 new points, independent of model performance.
 with np.load(extensions[0]) as z: rx=jnp.asarray(z['x_data'][:4096],dtype=jnp.float64);coarse=z['r_1000'][:4096]
 k0=jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(0),20260913),0);_,nk=jax.random.split(k0)
 rz=jax.random.normal(jax.random.fold_in(nk,0),(1000,80000,2),dtype=jnp.float32)[:,:4096,:].astype(jnp.float64)
 replay=np.asarray(advance(rx,rz)-rx);np.testing.assert_allclose(replay,coarse,rtol=0,atol=1e-14)
 refinements=[];last=coarse
 for level in [1,2]:
  bridge=jax.random.normal(jax.random.fold_in(nk,1000+level),rz.shape,dtype=jnp.float64)
  rz=jnp.stack([(rz+bridge)/jnp.sqrt(2.),(rz-bridge)/jnp.sqrt(2.)],axis=1).reshape((-1,4096,2))
  refined=np.asarray(advance(rx,rz,1e-7/(2**level))-rx)
  refinements.append(float(np.sqrt(np.mean(((refined-last)/1e-4)**2))));last=refined
 assert refinements[1]<refinements[0],refinements
 records={}
 # Canonical global row numbering: original100k then the seven training blocks.
 for h in m['h_grid']:
  d=old[h];steps=round(h/1e-7);xs=[d.x];rs=[d.r]
  for path in extensions:
   with np.load(path) as z:xs.append(z['x_data']);rs.append(z[f'r_{steps}'])
  x=np.concatenate(xs);r=np.concatenate(rs);assert x.shape==(660000,2) and r.dtype==np.float64
  for N in m['N_grid']:
   if h!=.0001 and N!=640000:continue # Other frozen Phase2 views can be materialized later without new generation.
   record=make_view(N,h,x,r,d)
   records[f'{N}_{h:.8g}']=record
 write(dest/'complete.json',dict(status='passed',records=records,extensions=draw_hashes,zero_noise_normalized_rmse=error,matched_noise_refinement_rms=refinements,source_lag_manifest_sha256=sha(parent/'lag_datasets/complete.json'),script_sha256=sha(__file__),jax=jax.__version__,devices=[str(v) for v in jax.devices()],covariance_oracle='No closed-form oracle assumed; exact drift finite-lag identity reported separately'))
 print('All nested/coupled views validated',flush=True)
def make_view(N,h,x,r,d):
 folder=OUT/'dataset_roots'/f'N_{N}'/f'h_{h:.8g}';folder.mkdir(parents=True,exist_ok=False);(folder/'data').mkdir()
 for link in ['src','scripts']:(folder/link).symlink_to(ROOT/link,target_is_directory=True)
 n=N+20000;train=np.concatenate([d.train_idx,np.arange(100000,n)]);assert len(train)==N
 target=folder/'data/ex8.npz'
 with target.open('xb') as f:np.savez_compressed(f,x_data=x[:n],r_data=r[:n],step_sizes=np.full((n,1),h,dtype=np.float32),train_idx=train,validation_idx=d.validation_idx,test_idx=d.test_idx)
 actual=load_dataset(target)
 for field in ['x','r']:
  np.testing.assert_array_equal(getattr(actual,field)[:100000],getattr(d,field))
 for split in ['validation_idx','test_idx']:np.testing.assert_array_equal(getattr(actual,split),getattr(d,split))
 record=dict(n_train=N,n_validation=10000,n_test=10000,h=h,dataset_sha256=sha(target),source_dataset='original corrected coupled rows + preregistered independent extension blocks',training_original_row_ids_sha256=hashlib.sha256(train.tobytes()).hexdigest(),validation_original_row_ids_sha256=hashlib.sha256(d.validation_idx.tobytes()).hexdigest(),test_original_row_ids_sha256=hashlib.sha256(d.test_idx.tobytes()).hexdigest(),subset_rule='original train_idx then appended training IDs; validation/test original IDs unchanged',stored_r_dtype=str(r.dtype))
 write(folder/'dataset_view.json',record);print('Validated view',N,h,flush=True);return record
if __name__=='__main__':main()
