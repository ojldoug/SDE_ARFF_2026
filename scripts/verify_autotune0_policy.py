#!/usr/bin/env python3
"""Single prospective policy, three-process gate, at most two full fits. No retries."""
import argparse,fcntl,json,os,subprocess,sys,time,runpy
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('mode',choices=['supervise','support','full']);p.add_argument('--workspace',type=Path,required=True);p.add_argument('--resources',type=Path,required=True);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--lock-dir',type=Path,required=True);a=p.parse_args();w=a.workspace.resolve();r=a.resources.resolve();scripts=Path(__file__).resolve().parent
if a.mode!='supervise':
 assert os.environ['XLA_FLAGS']=='--xla_gpu_autotune_level=0'
 if a.mode=='support':
  import jax,jax.numpy as jnp
  result=jax.jit(lambda x:x+1)(jnp.asarray(1));result.block_until_ready()
  print(json.dumps(dict(jax=jax.__version__,devices=[str(x) for x in jax.devices()],result=int(result),flag=os.environ['XLA_FLAGS'])));sys.exit(0)
 sys.path.insert(0,str(r/'checkout/scripts'));sys.argv=[str(r/'checkout/scripts/reproduction_route.py'),'baseline','--method','arff','--seed','0','--execute','--output',str(w/'run')];runpy.run_path(sys.argv[0],run_name='__main__');sys.exit(0)
import numpy as np
w.mkdir(exist_ok=False);(w/'STARTED.json').write_text(json.dumps(dict(pid=os.getpid(),started=time.time())))
fd=os.open(a.lock_dir/'gpu_0.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not os.environ.get('XLA_FLAGS'),'Unexpected existing flag policy'
state=dict(status='preflight',minimal_completed=0,full_completed=0)
def save(): (w/'status.json').write_text(json.dumps(state,indent=2))
def execute(name,command):
 assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 gpu=subprocess.check_output(['nvidia-smi','-i','0','--query-gpu=uuid,name,driver_version','--format=csv,noheader'],text=True).strip();assert gpu.startswith('GPU-d2cfc73c-186a-6651-e9f1-c6d93a00f0b2,')
 caches=w/(name+'_caches');(caches/'jax').mkdir(parents=True);(caches/'cuda').mkdir()
 env=dict(os.environ)
 for k in ['PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','XLA_FLAGS','JAX_COMPILATION_CACHE_DIR']:env.pop(k,None)
 env.update(XLA_FLAGS='--xla_gpu_autotune_level=0',JAX_COMPILATION_CACHE_DIR=str(caches/'jax'),CUDA_CACHE_PATH=str(caches/'cuda'),CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cuda',JAX_ENABLE_X64='false',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
 (w/(name+'_launch.json')).write_text(json.dumps(dict(command=command,gpu=gpu,cache_empty_before=True,environment={k:v for k,v in env.items() if k.startswith(('JAX','XLA','CUDA','OMP','OPENBLAS','LD_'))}),indent=2))
 state['active']=name;save();start=time.time()
 with (w/(name+'.log')).open('x') as log:subprocess.run(command,cwd=r/'checkout',env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(fd,),check=True)
 state[name+'_wall_seconds']=time.time()-start;save()
py=str(r/'venv/bin/python');common=['--resources',str(r),'--fixture',str(a.fixture.resolve()),'--lock-dir',str(a.lock_dir.resolve())]
try:
 execute('support',[py,'-B',str(Path(__file__).resolve()),'support','--workspace',str(w),*common]);state['support_passed']=True
 first=None
 for i in range(3):
  execute(f'minimal_{i}',[py,'-B',str(scripts/'check_autotune0_minimal.py'),'--checkout',str(r/'checkout'),'--fixture',str(a.fixture.resolve()),'--fixture-sha256','fd71907f7be2889af9ad6d6ce082e6a5c6f729b9a692bdf69640288e5e366985','--output',str(w/f'minimal_{i}')])
  z=dict(np.load(w/f'minimal_{i}/arrays.npz'))
  for key in ['key','omega','amp','predictions','validation_predictions','validation_loss']:assert z[key+'_0'].tobytes()==z[key+'_1'].tobytes(),('within process',i,key)
  if first is None:first=z
  else:
   for key in first:assert first[key].tobytes()==z[key].tobytes(),('between processes',i,key)
  state['minimal_completed']=i+1;save()
 state['minimal_gate_passed']=True;save()
 for i in range(2):
  out=w/f'full_{i}';out.mkdir()
  execute(f'full_{i}',[py,'-B',str(Path(__file__).resolve()),'full','--workspace',str(out),*common]);state['full_completed']=i+1;save()
 execute('full_comparison',[py,'-B',str(scripts/'compare_current_full_runs.py'),'--checkout',str(r/'checkout'),'--previous',str(w/'full_0/run/artifact.npz'),'--repeat',str(w/'full_1/run/artifact.npz'),'--output',str(w/'full_comparison.json')])
 state['status']='complete';save();(w/'COMPLETE.json').write_text(json.dumps(state,indent=2))
except Exception as e:
 state.update(status='failed',failure=repr(e));save();(w/'FAILURE.json').write_text(json.dumps(state,indent=2));raise
