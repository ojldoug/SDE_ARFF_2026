#!/usr/bin/env python3
"""One frozen-input initialization + first native step; no training loop."""
import argparse,hashlib,json,os,sys,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--fixture-sha256',required=True);a=p.parse_args()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(a.fixture)==a.fixture_sha256
assert sha(a.checkout/'src/arff/regression.py')=='e48d15e3701468255d1307467b864e94c672836fde17843a0f5f5827ccc759e7'
a.output.mkdir(exist_ok=False);sys.path.insert(0,str(a.checkout.resolve()))
import numpy as np
import jax,jax.numpy as jnp
from src.arff.regression import ARFFModel,fit_amplitudes,fourier_features,make_compiled_adaptation_step
assert not jax.config.jax_enable_x64
smi=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip();assert smi.startswith('GPU-d2cfc73c-186a-6651-e9f1-c6d93a00f0b2,')
f=dict(np.load(a.fixture));x,y,w,key=(jnp.asarray(f[k]) for k in ['x','y','initial_omega','initial_key']);assert x.shape==(57600,2) and x.dtype==y.dtype==jnp.float32
amp=fit_amplitudes(x,y,w,.001);amp.block_until_ready();model=ARFFModel(w,amp)
step=make_compiled_adaptation_step(delta=.2,lambda_reg=.001,gamma=1.,resampling=False,metropolis_test=True)
nextkey,result=step(key,model,x,y);result.amp.block_until_ready()
assert np.array_equal(result.omega,f['omega']),'Frequency identity failed'
# Introspection after execution; native output signature is unchanged.
lowered=step.lower(key,model,x,y);compiled=lowered.compile()
(a.output/'lowered.txt').write_text(lowered.as_text());(a.output/'optimized.txt').write_text(compiled.as_text())
fp=getattr(compiled.runtime_executable(),'fingerprint',None)
if callable(fp):fp=fp()
if isinstance(fp,bytes):fp=fp.hex()
phi=fourier_features(result.omega,x);gram=jnp.matmul(phi.T,phi,precision=jax.lax.Precision.HIGHEST);rhs=jnp.matmul(phi.T,y,precision=jax.lax.Precision.HIGHEST)
arrays={k:np.asarray(v) for k,v in dict(initial_amp=amp,omega=result.omega,amp=result.amp,nextkey=nextkey,features=phi,gram=gram,rhs=rhs,ridge_shift=jnp.asarray(len(x)*.001,dtype=x.dtype),predictions=phi@result.amp).items()}
np.savez_compressed(a.output/'arrays.npz',**arrays)
libs=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if '/' in line and any(s in line.lower() for s in ['cuda','cublas','cusolver','cudnn','jaxlib','pjrt'])})
cache={}
for q in [Path.home()/'.nv/ComputeCache',Path.home()/'.cache/jax']:
 items=sorted((str(v.relative_to(q)),v.stat().st_size,v.stat().st_mtime_ns) for v in q.rglob('*') if v.is_file()) if q.exists() else []
 cache[str(q)]=dict(exists=q.exists(),files=len(items),inventory_sha256=hashlib.sha256(json.dumps(items).encode()).hexdigest())
r=dict(gpu=smi,pid=os.getpid(),cwd=os.getcwd(),python=sys.executable,affinity=sorted(os.sched_getaffinity(0)),fixture_sha256=sha(a.fixture),source_sha256=sha(a.checkout/'src/arff/regression.py'),array_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()},program_hashes={k:sha(a.output/k) for k in ['lowered.txt','optimized.txt']},runtime_fingerprint=str(fp),jax_config={k:str(v) for k,v in jax.config.values.items()},environment={k:v for k,v in os.environ.items() if k.startswith(('JAX','XLA','CUDA','CUBLAS','NVIDIA','TF_','OMP','OPENBLAS','LD_LIBRARY','CONDA'))},loaded_libraries=[dict(name=Path(s).name,sha256=sha(s)) for s in libs if Path(s).is_file()],cache_inventories=cache,scientific_calls=dict(native_steps=1,amplitude_calls=3),vs_archived_amp_exact=bool(np.array_equal(arrays['amp'],f['amp_archived'])),vs_first_current_amp_exact=bool(np.array_equal(arrays['amp'],f['amp_new'])))
(a.output/'record.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
