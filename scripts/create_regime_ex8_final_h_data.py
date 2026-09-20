"""Coupled continuation to oracle-approved lags; old data are read-only."""
import os
os.environ.setdefault('JAX_PLATFORMS','cuda')
from pathlib import Path
import sys,json,hashlib
import numpy as np
import jax,jax.numpy as jnp
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import create_regime_ex8_data as old
from src.experiments.dataset import load_dataset
from oracle_regime_ex8 import oracle
from run_ex8_split_mlp_w27 import check_assigned_gpu
OUT=ROOT/'results/capacity_regime_ex8_final_h_v1';PARENT=ROOT/'results/capacity_regime_ex8_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def main():
 check_assigned_gpu(allow_cpu=False)
 m=json.loads((OUT/'manifest.json').read_text());assert json.loads((PARENT/'pipeline/state.json').read_text())['status']=='complete'
 dest=OUT/'data';dest.mkdir(exist_ok=False)
 jax.config.update('jax_enable_x64',True)
 source=load_dataset(PARENT/'dataset_roots/N_640000/h_0.001/data/ex8.npz')
 original=load_dataset(ROOT/'data/ex8_float64_v2.npz')
 original_lags=json.loads((ROOT/'results/controlled_study_2026/float64_v2/lag_datasets/complete.json').read_text())
 prior=json.loads((PARENT/'data/complete.json').read_text())
 protected={str(PARENT/'dataset_roots/N_640000/h_0.001/data/ex8.npz'):sha(PARENT/'dataset_roots/N_640000/h_0.001/data/ex8.npz')}
 endpoints={h:[] for h in m['amendment']['included_h']};records=[]
 for block in range(8):
  start=0 if block==0 else 100000+(block-1)*80000
  count=100000 if block==0 else 80000
  x0=jnp.asarray(source.x[start:start+count],dtype=jnp.float64);state=x0
  if block==0:_,key=jax.random.split(jax.random.PRNGKey(0))
  else:
   _,key=jax.random.split(jax.random.fold_in(jax.random.fold_in(jax.random.PRNGKey(0),20260913),block-1))
  hashes=[];fixture=[]
  for t in range(round(max(endpoints)/1e-7)//1000):
   k=key if block==0 and t==0 else jax.random.fold_in(key,t)
   noise=jax.random.normal(k,(1000,count,2),dtype=jnp.float32)
   noise_sha=hashlib.sha256(np.asarray(noise).tobytes()).hexdigest();hashes.append(noise_sha)
   if t<10:
    expected=prior['brownian_blocks'][block]['normal_sha256'][t]
    assert noise_sha==expected,'Previously registered Brownian block changed'
   if block==0:fixture.append(np.asarray(noise[:,:1024,:]))
   if t==0:
    for lo,hi in [(0,250),(250,500),(500,1000)]:state=old.advance(state,noise[lo:hi])
   else:state=old.advance(state,noise)
   inc=np.asarray(state-x0)
   if t==9:np.testing.assert_array_equal(inc,source.r[start:start+count])
   for h in endpoints:
    if (t+1)*1000==round(h/1e-7):
     assert inc.dtype==np.float64 and np.isfinite(inc).all();endpoints[h].append(inc.copy())
  records.append(dict(block=block,count=count,normal_sha256=hashes,existing_10000_step_prefix_bit_identical=True))
  if block==0:
   # Independent Brownian-bridge refinements on 1024 unchanged states;
   # no model or test performance enters the data validity gate.
   xx=x0[:1024];z=jnp.asarray(np.concatenate(fixture),dtype=jnp.float64)
   last=np.asarray(old.advance(xx,z)-xx);ref=[]
   for level in [1,2]:
    bridge=jax.random.normal(jax.random.fold_in(key,2026091600+level),z.shape,dtype=jnp.float64)
    z=jnp.stack([(z+bridge)/jnp.sqrt(2.),(z-bridge)/jnp.sqrt(2.)],axis=1).reshape((-1,1024,2))
    refined=np.asarray(old.advance(xx,z,1e-7/(2**level))-xx)
    ref.append(float(np.sqrt(np.mean(((refined-last)/max(endpoints))**2))));last=refined
   assert ref[1]<ref[0],ref
   exact=np.asarray(xx)*np.expm1(round(max(endpoints)/1e-7)*np.log1p(-1e-7))
   zero=jax.lax.fori_loop(0,round(max(endpoints)/1e-7),lambda i,v:v-jnp.asarray(1e-7,dtype=jnp.float64)*v,xx)
   zero_error=float(np.sqrt(np.mean(((np.asarray(zero-xx)-exact)/max(endpoints))**2)))
   assert zero_error<1e-8,zero_error
   print('New-lag refinement and zero-noise gates passed',ref,zero_error,flush=True)
  print('Validated continued Brownian pool',block,flush=True)
 old.OUT=OUT
 view_records={};moments=[]
 from dataclasses import replace
 for h,parts in endpoints.items():
  inc=np.concatenate(parts);assert inc.shape==(660000,2)
  d=replace(original,r=inc[:100000],h=np.full((100000,1),h,dtype=np.float32))
  rec=old.make_view(640000,h,source.x,inc,d);view_records[f'640000_{h:.8g}']=rec
  # Same accepted trace gate as Phase1, now on each added lag.
  actual=load_dataset(OUT/f'dataset_roots/N_640000/h_{h:.8g}/data/ex8.npz');v=oracle(h)
  idx=actual.train_idx;res=actual.r[idx]-h*v['discrete_drift_coefficient']*actual.x[idx].astype(float)
  sq=np.sum(res**2,axis=1)/h;se=sq.std(ddof=1)/np.sqrt(len(idx))
  assert abs(sq.mean()-v['oracle_effective_covariance_trace'])<max(6*se,1e-4)
  assert v['relative_discrete_drift_bias']<=m['amendment']['drift_bias_threshold']
  v.update(training_empirical_covariance_trace=float(sq.mean()),training_trace_standard_error=float(se));moments.append(v)
 for p,digest in protected.items():assert sha(p)==digest
 write(OUT/'oracle_moments.json',dict(status='passed',parent_sha256=sha(PARENT/'oracle_moments.json'),parent_records=json.loads((PARENT/'oracle_moments.json').read_text())['records'],records=moments,limitation='Exact drift and covariance trace only; no full-matrix finite-lag covariance-bias certification'))
 write(dest/'complete.json',dict(status='passed',records=view_records,brownian_blocks=records,zero_noise_normalized_rmse=zero_error,matched_noise_refinement_rms=ref,protected_sha256=protected,script_sha256=sha(__file__),jax=jax.__version__,devices=[str(v) for v in jax.devices()]))
 print('Amended data validated; no original dataset changed',flush=True)
if __name__=='__main__':main()
