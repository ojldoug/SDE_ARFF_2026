#!/usr/bin/env python3
"""Durable four-exclusive-GPU supervisor for width27 split MLP jobs."""
import os
os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import sys,json,subprocess,threading,time,hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_ex8_k128_campaigns as infra
OUT=ROOT/'results/production';CONTROL=OUT/'ex8_split_mlp_w27_campaign_control';PYTHON=infra.PYTHON
LOCK=threading.Lock()
def digest(p):return infra.digest(p)
def state(**changes):
 with LOCK:
  path=CONTROL/'state.json';v=infra.read_json(path) if path.exists() else {};v.update(changes);v['updated_at']=infra.now();infra.write_json(path,v)
def paths(job):
 d=OUT/'adam_split_mlp_ex8_w27';seed=job['seed']
 return d/f'seed_{seed}_artifacts.npz',d/f'seed_{seed}.txt',d/'state'/f'seed_{seed}'
def verify():
 plan=infra.read_json(CONTROL/'launch_manifest.json')
 for p,h in plan['frozen_sha256'].items():
  if digest(ROOT/p)!=h:raise RuntimeError('Frozen source/config/data changed: '+p)
 return plan

def validate(job,artifact,log):
 import run_adam_split_mlp_experiment as runner
 from dataclasses import asdict
 from run_ex8_mlp_w27 import effective_config
 with np.load(artifact,allow_pickle=False) as z:
  a={k:z[k] for k in z.files}
 runner.validate_artifact(a)
 expected=dict(artifact_version=4,method='adam_split_mlp',experiment='ex8',seed=job['seed'],hidden_width=27,hidden_layers=2,mlp_parameter_count=1814,actual_mlp_parameter_count=1814,fourier_parameter_count=1792,epochs_per_regression=300,batch_size=256,learning_rate=.001,n_folds=5,fold_seed=2026,n_train=80000,n_validation=10000,n_test=10000,diff_type='symmetric')
 for k,v in expected.items():
  if a[k].item()!=v:raise ValueError('Configuration mismatch '+k)
 if json.loads(str(a['config_json']))!=asdict(effective_config('ex8')):raise ValueError('Effective config mismatch')
 if str(a['dataset_sha256'])!=verify()['dataset_sha256']:raise ValueError('Dataset hash')
 counts=[]
 for prefix,q in [('drift',2),('covariance',3)]:
  for i,shape in enumerate([(2,27),(27,27),(27,q)]):
   if a[f'{prefix}_weight_{i}'].shape!=shape or a[f'{prefix}_bias_{i}'].shape!=(shape[1],):raise ValueError('Architecture mismatch')
  counts.append(sum(a[f'{prefix}_{kind}_{i}'].size for i in range(3) for kind in ['weight','bias']))
 if counts!=[893,921]:raise ValueError('Parameter count')
 if str(a['jax_backend'])!='gpu' or 'backend    : gpu' not in log.read_text():raise ValueError('GPU backend')
 for k,v in a.items():
  if v.dtype.hasobject:raise ValueError('Object array '+k)
 return dict(artifact_sha256=digest(artifact),log_sha256=digest(log),algorithm_time=float(a['algorithm_time']))

def prepare():
 CONTROL.mkdir(parents=True,exist_ok=True)
 if (CONTROL/'launch_manifest.json').exists():verify();return
 jobs=[dict(experiment='ex8',method='split_mlp',seed=i) for i in range(30)]
 for job in jobs:
  if any(p.exists() for p in paths(job)):raise FileExistsError('Existing output/reservation; inspect instead of overwriting')
 frozen=['scripts/run_ex8_split_mlp_w27.py','scripts/run_ex8_split_mlp_w27_campaign.py','scripts/run_adam_split_mlp_experiment.py','scripts/run_ex8_mlp_w27.py','data/ex8.npz']
 frozen+=[str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.py')]
 from run_ex8_mlp_w27 import effective_config
 from dataclasses import asdict
 infra.write_json(CONTROL/'launch_manifest.json',dict(jobs=jobs,gpus=[0,1,2,3],session='split_mlp_ex8_w27_campaign',assignment='seed modulo4',timing_context='four-GPU concurrent accuracy; non-isolated',effective_config=asdict(effective_config('ex8')),parameters=1814,dataset_sha256=digest(ROOT/'data/ex8.npz'),frozen_sha256={p:digest(ROOT/p) for p in frozen}),exclusive=True)
 state(status='prepared',jobs_total=30);print('Prepared30 width27 split MLP jobs; no launch.')

def run():
 plan=verify();errors=[]
 with infra.lock_file(CONTROL/'supervisor.lock'):
  while infra.gpu_processes():state(status='waiting_machine_idle');time.sleep(15)
  state(status='running')
  def worker(gpu):
   with infra.lock_file(CONTROL/f'gpu_{gpu}.lock') as fd:
    uuid=infra.gpu_uuid(gpu)
    for index,job in enumerate(plan['jobs']):
     if index%4!=gpu:continue
     artifact,log,reservation=paths(job)
     try:
      if artifact.exists():
       checked=validate(job,artifact,log);saved=infra.read_json(reservation/'complete.json')
       if any(saved[k]!=checked[k] for k in ('artifact_sha256','log_sha256')):raise ValueError('Completed output changed')
       continue
      if log.exists() or reservation.exists():raise FileExistsError('Incomplete/suspicious prior output; no automatic retry')
      while any(row[0].strip()==uuid for row in infra.gpu_processes()):time.sleep(15)
      verify();reservation.parent.mkdir(parents=True,exist_ok=True);reservation.mkdir()
      staged=reservation/artifact.name
      command=[PYTHON,'-B','-u',str(ROOT/'scripts/run_ex8_split_mlp_w27.py'),'--seed',str(job['seed']),'--artifact-path',str(staged)]
      record=dict(job,gpu=gpu,uuid=uuid,command=command,started_at=infra.now(),status='reserved')
      infra.write_json(reservation/'job.json',record,exclusive=True)
      env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',CONDA_DEFAULT_ENV='arff-sde',CONDA_PREFIX=str(Path(PYTHON).parent.parent));env['PATH']=str(Path(PYTHON).parent)+os.pathsep+env.get('PATH','')
      while any(row[0].strip()==uuid for row in infra.gpu_processes()):time.sleep(15)
      with log.open('x') as f:
       p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,pass_fds=(fd,))
       record.update(pid=p.pid,status='running');infra.write_json(reservation/'job.json',record)
       print(infra.now(),'START',job,'gpu',gpu,'pid',p.pid,flush=True);code=p.wait()
      record.update(returncode=code,finished_at=infra.now());infra.write_json(reservation/'exit.json',record,exclusive=True)
      if code:raise RuntimeError('Numerical process exited '+str(code))
      checked=validate(job,staged,log);verify();os.link(staged,artifact)
      record.update(status='complete',**checked);infra.write_json(reservation/'complete.json',record,exclusive=True)
      print(infra.now(),'COMPLETE',job,'gpu',gpu,flush=True)
     except Exception as exc:
      errors.append(dict(job,gpu=gpu,error=str(exc)));state(errors=errors)
      print(infra.now(),'WORKER STOPPED',gpu,job,str(exc),flush=True);return
  with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,range(4)))
  if errors:state(status='failed_workers',errors=errors);return
  for job in plan['jobs']:validate(job,*paths(job)[:2])
  state(status='complete');print('All30 width27 split MLP artifacts validated.',flush=True)
def main():
 action=sys.argv[1]
 if action=='prepare':prepare()
 elif action=='run':run()
 elif action=='status':print(json.dumps(infra.read_json(CONTROL/'state.json'),indent=2))
 else:raise ValueError(action)
if __name__=='__main__':main()
