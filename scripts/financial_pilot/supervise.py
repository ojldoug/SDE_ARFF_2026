"""Finite detached one-GPU supervisor; cumulative allocation and wall ceilings."""
import os,sys,time,json,fcntl,subprocess,signal
from pathlib import Path
from .data import ROOT,P,sha
from .pipeline import dump,verify

def query(args):return subprocess.check_output(['nvidia-smi',*args],text=True)
def main():
 for name in ['COMPLETE','FAILURE','PARTIAL']:
  if (P/(name+'.json')).exists():raise RuntimeError('Terminal marker exists; no automatic restart')
 lock=open(P/'supervisor.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
 for path in (P/'models').glob('*.pkl'):
  meta=json.loads(path.with_suffix('.json').read_text());assert meta['sha256']==sha(path) and meta['frozen_sha256']==sha(P/'FROZEN.json')
 previous=json.loads((P/'execution.json').read_text()) if (P/'execution.json').exists() else {};spent=previous.get('allocated_seconds',0.);start=previous.get('supervisor_start',time.time())
 if spent>=8*3600 or time.time()-start>=12*3600:raise RuntimeError('Budget exhausted; no restart')
 dump(P/'status.json',dict(status='waiting_idle_gpu',supervisor_start=start))
 while time.time()-start<12*3600:
  info=query(['--query-gpu=index,uuid,name,memory.used,utilization.gpu,driver_version','--format=csv,noheader,nounits']);chosen=None
  for line in info.splitlines():
   idx,uuid,name,mem,util,driver=[x.strip() for x in line.split(',')]
   if 'A6000' not in name or float(mem)>200 or float(util)>0:continue
   fd=open(ROOT/f'results/production/ex8_campaign_control/gpu_{idx}.lock','a')
   try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
   except BlockingIOError:fd.close();continue
   apps=query(['--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'])
   if uuid in apps:fd.close();continue
   chosen=(idx,uuid,fd);break
  if chosen:break
  time.sleep(30)
 else:
  dump(P/'PARTIAL.json',dict(reason='12-hour wall limit while waiting',supervisor_start=start));return
 idx,uuid,fd=chosen;launch=time.time();env=os.environ.copy()
 for key in ['PYTHONPATH','PYTHONHOME','PYTHONUSERBASE']:env.pop(key,None)
 cache=P/f'cache_{int(launch)}';cache.mkdir();(cache/'jax').mkdir();(cache/'cuda').mkdir()
 # Single authorized policy. Reject inherited conflicting flags, do not sweep.
 if env.get('XLA_FLAGS','') not in ['', '--xla_gpu_autotune_level=0']:raise RuntimeError('Unexpected inherited XLA_FLAGS')
 env.update(CUDA_VISIBLE_DEVICES=idx,JAX_PLATFORMS='cuda',JAX_ENABLE_X64='false',XLA_FLAGS='--xla_gpu_autotune_level=0',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',JAX_COMPILATION_CACHE_DIR=str(cache/'jax'),CUDA_CACHE_PATH=str(cache/'cuda'))
 cmd=[sys.executable,'-B','-m','scripts.financial_pilot.pipeline'];rec=dict(supervisor_start=start,start=launch,gpu_index=idx,gpu_uuid=uuid,gpu_info=info,command=cmd,settings={k:env[k] for k in ['JAX_ENABLE_X64','XLA_FLAGS','JAX_COMPILATION_CACHE_DIR','CUDA_CACHE_PATH','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','CUDA_VISIBLE_DEVICES']},allocated_seconds=spent,peak_device_memory_MiB=0,supervisor_sha256=sha(Path(__file__)),frozen_sha256=sha(P/'FROZEN.json'))
 dump(P/'execution.json',rec)
 with (P/'console.log').open('a') as log:
  child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=(fd.fileno(),));rec['pid']=child.pid
  while child.poll() is None:
   elapsed=time.time()-launch;rec['allocated_seconds']=spent+elapsed
   try:
    mem=float(query(['--id='+idx,'--query-gpu=memory.used','--format=csv,noheader,nounits']).strip());rec['peak_device_memory_MiB']=max(mem,rec['peak_device_memory_MiB'])
   except Exception:pass
   dump(P/'execution.json',rec)
   if spent+elapsed>=8*3600-10 or time.time()-start>=12*3600-10:
    os.killpg(child.pid,signal.SIGTERM)
    try:child.wait(timeout=5)
    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
    dump(P/'PARTIAL.json',dict(reason='hard GPU/wall allocation cap',allocated_seconds=spent+time.time()-launch));break
   time.sleep(5)
  code=child.wait();rec.update(returncode=code,end=time.time(),allocated_seconds=spent+time.time()-launch,supervisor_wall_seconds=time.time()-start);dump(P/'execution.json',rec)
  if code!=0 and not (P/'FAILURE.json').exists() and not (P/'PARTIAL.json').exists():dump(P/'FAILURE.json',dict(returncode=code))
 fd.close()
 print('TERMINAL',code,flush=True)
if __name__=='__main__':main()
