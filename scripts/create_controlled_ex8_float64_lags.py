#!/usr/bin/env python3
"""Coupled lag views anchored exactly to approved float64 v2 baseline draws."""
import os
os.environ.setdefault('JAX_PLATFORMS','cuda')
from pathlib import Path
import sys,json,hashlib
import jax,jax.numpy as jnp,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiments.definitions import ex8_drift,ex8_diffusion_factor
from src.experiments.dataset import load_dataset
STUDY=ROOT/'results/controlled_study_2026/float64_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
def main():
 assert json.loads((STUDY/'campaigns/baseline/state.json').read_text())['status']=='complete'
 assert (STUDY/'baseline_summary/summary.json').exists(), 'Baseline validation/report has priority'
 m=json.loads((STUDY/'manifest.json').read_text());v=json.loads((ROOT/'results/controlled_study_2026/float64_v2_validation/validation.json').read_text())
 source=ROOT/m['dataset']['path'];assert sha(source)==v['corrected_sha256']
 out=STUDY/'lag_datasets';out.mkdir(exist_ok=False)
 jax.config.update('jax_enable_x64',True)
 d=load_dataset(source);xx=jnp.asarray(d.x,dtype=jnp.float64);state=xx
 _,key=jax.random.split(jax.random.PRNGKey(0));noise_hashes=[];endpoints={}
 def advance(initial,noise):
  def step(state,z):
   dt=jnp.asarray(1e-7,dtype=jnp.float64)
   increment=jnp.einsum('nij,nj->ni',ex8_diffusion_factor(state),jnp.sqrt(dt)*z.astype(jnp.float64))
   return state+dt*ex8_drift(state)+increment,None
  return jax.lax.scan(step,initial,noise)[0]
 # Existing block0 is unchanged. Subsequent independent key-addressed blocks
 # extend that very same path; shorter lags are exact prefixes, not new draws.
 for block in range(4):
  block_key=key if block==0 else jax.random.fold_in(key,block)
  noise=jax.random.normal(block_key,(1000,100000,2),dtype=jnp.float32)
  noise_hashes.append(hashlib.sha256(np.asarray(noise).tobytes()).hexdigest())
  if block==0:
   assert noise_hashes[-1]==v['noise_sha256']
   offset=0
   for stop in (250,500,1000):
    state=advance(state,noise[offset:stop]);endpoints[stop]=np.asarray(state-xx);offset=stop
   np.testing.assert_array_equal(endpoints[1000],d.r)
  else:
   state=advance(state,noise)
   if block in (1,3):endpoints[(block+1)*1000]=np.asarray(state-xx)
 roots={};records={}
 for count,h in zip(m['lag_dataset']['steps'],m['h_grid']):
  assert count==round(h/1e-7)
  name=format(h,'.8g');folder=out/('h_'+name);folder.mkdir();(folder/'data').mkdir()
  for link in ('src','scripts'):(folder/link).symlink_to(ROOT/link,target_is_directory=True)
  target=folder/'data/ex8.npz'
  if count==1000:target.symlink_to(source)
  else:
   with target.open('xb') as f:np.savez_compressed(f,x_data=d.x,r_data=endpoints[count],step_sizes=np.full((100000,1),h,dtype=np.float32),train_idx=d.train_idx,validation_idx=d.validation_idx,test_idx=d.test_idx)
  actual=load_dataset(target);assert actual.r.dtype==np.float64 and np.isfinite(actual.r).all()
  for field in ('x','train_idx','validation_idx','test_idx'):np.testing.assert_array_equal(getattr(actual,field),getattr(d,field))
  np.testing.assert_array_equal(actual.r,endpoints[count])
  with (folder/'original_row_ids.npz').open('xb') as f:np.savez_compressed(f,original_row_ids=np.arange(100000))
  record=dict(n_train=80000,n_validation=10000,n_test=10000,h=h,fine_steps=count,fine_step=1e-7,integration_dtype='float64',stored_increment_dtype='float64',source_dataset=str(source),source_sha256=sha(source),dataset_sha256=sha(target),original_row_ids_sha256=sha(folder/'original_row_ids.npz'),training_original_row_ids_sha256=hashlib.sha256(d.train_idx.tobytes()).hexdigest(),validation_original_row_ids_sha256=hashlib.sha256(d.validation_idx.tobytes()).hexdigest(),test_original_row_ids_sha256=hashlib.sha256(d.test_idx.tobytes()).hexdigest(),subset_rule='all original rows and splits, exactly coupled fine Brownian prefix',noise_block_sha256=noise_hashes[:max(1,count//1000)],noise_key_rule='block0: original split(PRNGKey(0))[1], shape1000x100000x2 float32; blocks1–3: fold_in(original_noise_key, block), same shape/dtype; cast each draw to float64 for integration')
  write(folder/'dataset_view.json',record);records[name]=record;roots[name]=str(folder.relative_to(STUDY))
  print('Validated lag',h,record['dataset_sha256'],flush=True)
 assert sha(source)==v['corrected_sha256']
 write(out/'complete.json',dict(status='passed',baseline_bit_identical=True,dataset_roots=roots,records=records,script_sha256=sha(__file__),source_definition_sha256=sha(ROOT/'src/experiments/definitions.py'),jax_version=jax.__version__,devices=[str(x) for x in jax.devices()]))
if __name__=='__main__':main()
