#!/usr/bin/env python3
"""Bounded serial controller: at most four fresh processes, stop on divergence."""
import argparse,fcntl,json,os,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--resources',type=Path,required=True);p.add_argument('--lock-dir',type=Path,required=True);a=p.parse_args();w=a.workspace.resolve();resources=a.resources.resolve()
with (w/'STARTED.json').open('x') as f:json.dump(dict(max_processes=4,pid=os.getpid()),f)
fd=os.open(a.lock_dir/'gpu_0.lock',os.O_CREAT|os.O_RDWR,0o600);fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
env=dict(os.environ)
for k in ['PYTHONPATH','PYTHONHOME','PYTHONUSERBASE','XLA_FLAGS','JAX_COMPILATION_CACHE_DIR']:env.pop(k,None)
env.update(CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cuda',JAX_ENABLE_X64='false',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1')
records=[];state={}
try:
 for number in range(4):
  assert not subprocess.check_output(['nvidia-smi','-i','0','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
  cmd=[str(resources/'venv/bin/python'),'-B',str(Path(__file__).with_name('minimal_fold0_reproducer.py').resolve()),'--checkout',str(resources/'checkout'),'--fixture',str(w/'fixture.npz'),'--fixture-sha256',(w/'fixture.sha256').read_text().strip(),'--output',str(w/f'process_{number}')]
  with (w/f'process_{number}.log').open('x') as log:subprocess.run(cmd,cwd=resources/'checkout',env=env,stdout=log,stderr=subprocess.STDOUT,pass_fds=(fd,),check=True)
  r=json.loads((w/f'process_{number}/record.json').read_text());records.append(r)
  changed=number>0 and any(r['array_hashes'][k]!=records[0]['array_hashes'][k] for k in ['amp','omega','predictions'])
  state=dict(completed=len(records),maximum=4,divergence=changed,commands=cmd,status='running');(w/'status.json').write_text(json.dumps(state,indent=2))
  if changed:break
 state['status']='complete';(w/'COMPLETE.json').write_text(json.dumps(state,indent=2));(w/'status.json').write_text(json.dumps(state,indent=2))
except Exception as e:
 state.update(status='failed',error=repr(e));(w/'FAILURE.json').write_text(json.dumps(state,indent=2));raise
