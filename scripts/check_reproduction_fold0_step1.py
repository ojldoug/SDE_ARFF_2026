#!/usr/bin/env python3
"""One unchanged native fold-0 step versus one standalone solve. No loop."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(a.checkout.resolve()))
import numpy as np
import jax,jax.numpy as jnp
from src.arff.regression import ARFFModel,fit_amplitudes,fourier_features,make_compiled_adaptation_step
from src.arff.validation_selected import _split_indices
from src.arff.two_stage import make_folds
from src.experiments.dataset import load_dataset
assert not jax.config.jax_enable_x64
old=dict(np.load(a.reference));new=dict(np.load(a.new));dp=a.checkout/'data/ex8_float64_v2.npz';assert hashlib.sha256(dp.read_bytes()).hexdigest()==str(old['dataset_sha256'])
data=load_dataset(dp);train=old['train_idx'];assert np.array_equal(train,data.train_idx)
folds=make_folds(len(train),5,2026);mask=np.ones(len(train),bool);mask[folds[0]]=False;pool=np.arange(len(train))[mask]
fit,val=_split_indices(len(pool),validation_fraction=.1,seed=101000)
assert np.array_equal(pool[fit],old['fold_0_fit_train_positions'])
assert np.array_equal(pool[val],old['fold_0_internal_validation_train_positions'])
x,r,h=(jnp.asarray(v[train]) for v in [data.x,data.r,data.h]);xx=x[pool];yy=r[pool]/h[pool];xf=xx[fit];yf=yy[fit];xv=xx[val];yv=yy[val]
key=jax.random.PRNGKey(0);omega=jnp.zeros((2,128),dtype=xf.dtype);amp=fit_amplitudes(xf,yf,omega,.001);amp.block_until_ready();initial=ARFFModel(omega,amp)
step=make_compiled_adaptation_step(delta=.2,lambda_reg=.001,gamma=1.,resampling=False,metropolis_test=True)
(a.output/'native_lowered.txt').write_text(step.lower(key,initial,xf,yf).as_text())
start=time.monotonic();nextkey,native=step(key,initial,xf,yf);native.amp.block_until_ready()
# Proposal reconstruction only: exactly the first native key split and expression.
@jax.jit
def proposed(k,w):
 k,sub=jax.random.split(k);return w+.2*jax.random.normal(sub,w.shape)
proposal=proposed(key,omega)
checks=dict(proposal_matches_archived=bool(np.array_equal(proposal,old['fold_0_omega'])),native_frequencies_match_archived=bool(np.array_equal(native.omega,old['fold_0_omega'])),archived_new_frequencies_match=bool(np.array_equal(old['fold_0_omega'],new['fold_0_omega'])))
(a.output/'frequency_gate.json').write_text(json.dumps(checks,indent=2))
assert all(checks.values()),'Frequency mismatch: stop, no alternative keys'
standalone=jax.jit(fit_amplitudes)(xf,yf,native.omega,.001);standalone.block_until_ready()
# Reconstructed diagnostic buffers, not hidden native-kernel intermediates.
phi=fourier_features(native.omega,xf);gram=jnp.matmul(phi.T,phi,precision=jax.lax.Precision.HIGHEST);rhs=jnp.matmul(phi.T,yf,precision=jax.lax.Precision.HIGHEST)
static_shift=jnp.asarray(xf.shape[0]*.001,dtype=phi.dtype)
dynamic_shift=jax.jit(lambda lam:xf.shape[0]*lam)(.001)
mat=gram+static_shift*jnp.eye(256,dtype=phi.dtype)
arrays=dict(x=xf,y=yf,fit_positions=pool[fit],validation_positions=pool[val],initial_key=key,next_key=nextkey,initial_omega=omega,initial_amp=amp,proposal=proposal,omega=native.omega,features=phi,gram=gram,rhs=rhs,ridge_shift_static=static_shift,ridge_shift_dynamic=dynamic_shift,regularized_gram=mat,amp_native=native.amp,amp_standalone=standalone,amp_archived=old['fold_0_amp'],amp_new=new['fold_0_amp'])
for label,beta in [('native',native.amp),('standalone',standalone),('archived',jnp.asarray(old['fold_0_amp'])),('new',jnp.asarray(new['fold_0_amp']))]:arrays['pred_'+label]=phi@beta
arrays={k:np.asarray(v) for k,v in arrays.items()}
def comparison(x,y):
 d=y.astype(float)-x.astype(float)
 return dict(exact=bool(np.array_equal(x,y)),within_original_tolerance=bool(np.allclose(x,y,rtol=1e-5,atol=1e-6)),max_abs=float(np.max(np.abs(d))),rms=float(np.sqrt(np.mean(d*d))))
comp={}
for label in ['standalone','archived','new']:
 for field in ['amp','pred']:comp[field+'_native_vs_'+label]=comparison(arrays[field+'_native'],arrays[field+'_'+label])
A=arrays['regularized_gram'].astype(float);B=arrays['rhs'].astype(float);residual={}
for label in ['native','standalone','archived','new']:
 beta=arrays['amp_'+label].astype(float);rr=A@beta-B
 residual[label]=dict(absolute=float(np.linalg.norm(rr)),relative_rhs=float(np.linalg.norm(rr)/np.linalg.norm(B)),backward_error=float(np.linalg.norm(rr)/(np.linalg.norm(A)*np.linalg.norm(beta)+np.linalg.norm(B))))
eig=np.linalg.eigvalsh((A+A.T)/2)
loss=float(jnp.mean((fourier_features(native.omega,xv)@native.amp-yv)**2))
def sha(v):return hashlib.sha256(np.asarray(v).tobytes()).hexdigest()
record=dict(frequency_gate=checks,comparisons=comp,residuals=residual,conditioning=dict(min_eigenvalue=float(eig.min()),max_eigenvalue=float(eig.max()),spectral_condition=float(eig.max()/eig.min()),float32_epsilon_times_condition=float(np.finfo(np.float32).eps*eig.max()/eig.min())),ridge_shift_static=float(static_shift),ridge_shift_dynamic=float(dynamic_shift),native_validation_loss=loss,archived_validation_loss=float(old['fold_0_validation_mse'][0]),new_validation_loss=float(new['fold_0_validation_mse'][0]),hashes={k:sha(v) for k,v in arrays.items()},rtol=1e-5,atol=1e-6,solve_calls=4,native_steps=1,source_sha256=hashlib.sha256((a.checkout/'src/arff/regression.py').read_bytes()).hexdigest(),dataset_sha256=str(old['dataset_sha256']),python=sys.version,jax=jax.__version__,numpy=np.__version__,devices=[str(d) for d in jax.devices()],environment={k:v for k,v in os.environ.items() if k.startswith(('JAX','XLA','CUDA','OMP','OPENBLAS'))},elapsed_seconds=time.monotonic()-start)
np.savez_compressed(a.output/'capture.npz',**arrays)
(a.output/'RESULTS.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
