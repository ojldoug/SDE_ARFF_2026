#!/usr/bin/env python3
"""Durable four-exclusive-GPU supervisor for resolved historical corrected ARFF jobs."""
import os
os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import sys,json,subprocess,threading,time,hashlib
from concurrent.futures import ThreadPoolExecutor
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import run_ex8_k128_campaigns as infra
OUT=ROOT/'results/final_reproduction';CONTROL=OUT/'arff_campaign_control';PYTHON=infra.PYTHON
LOCK=threading.Lock()
def digest(p):return infra.digest(p)
def state(**changes):
 with LOCK:
  path=CONTROL/'state.json';v=infra.read_json(path) if path.exists() else {};v.update(changes);v['updated_at']=infra.now();infra.write_json(path,v)
def paths(job):
 d=OUT/'production/accuracy'/job['experiment']/job['method'];seed=job['seed']
 return d/f'seed_{seed}_artifacts.npz',d/f'seed_{seed}.txt',d/'state'/f'seed_{seed}'
def verify():
 plan=infra.read_json(CONTROL/'launch_manifest.json')
 for p,h in plan['frozen_sha256'].items():
  if digest(ROOT/p)!=h:raise RuntimeError('Frozen source/config/data changed: '+p)
 return plan

def validate(job,artifact,log):
 if not artifact.is_file() or not log.is_file():raise ValueError('Missing artifact/log')
 import run_final_historical_arff as runner
 m=infra.read_json(OUT/'production_manifest.arff_v1.json');settings=m['studies'][job['experiment']]['historical_arff']
 with np.load(artifact,allow_pickle=False) as z:
  runner.validate(z,settings)
  if str(z['experiment'])!=job['experiment'] or int(z['seed'])!=job['seed'] or str(z['method'])!='arff_historical_corrected':raise ValueError('Identity mismatch')
  if str(z['production_manifest_sha256'])!=digest(OUT/'production_manifest.arff_v1.json'):raise ValueError('Manifest mismatch')
  if str(z['dataset_sha256'])!=m['datasets'][job['experiment']]['sha256']:raise ValueError('Dataset mismatch')
  d=z['crossfit_covariance_targets'].shape[1];input_d=z['drift_omega'].shape[0];output_d=z['drift_amp'].shape[1];K=settings['K']
  if z['drift_omega'].shape!=(input_d,K) or z['drift_amp'].shape!=(2*K,output_d) or z['covariance_omega'].shape!=(input_d,K) or z['covariance_amp'].shape!=(2*K,d):raise ValueError('Model shapes')
  if int(z['total_active_parameters'])!=2*input_d*K+2*K*(output_d+d):raise ValueError('Parameter count')
  if str(z['jax_backend'])!='gpu' or 'backend: gpu' not in log.read_text():raise ValueError('GPU backend missing')
  return dict(artifact_sha256=digest(artifact),log_sha256=digest(log),algorithm_time=float(z['algorithm_time']))

def prepare():
 CONTROL.mkdir(parents=True,exist_ok=True)
 if (CONTROL/'launch_manifest.json').exists():verify();return
 m=infra.read_json(OUT/'production_manifest.arff_v1.json');jobs=[]
 for ex in ['ex1','ex2','ex3','ex5','ex7','ex6']:
  methods=['arff_historical_corrected']
  for method in methods:
   for seed in range(30):jobs.append(dict(experiment=ex,method=method,seed=seed))
 for job in jobs:
  if any(p.exists() for p in paths(job)):raise FileExistsError('Unexpected existing job paths')
 frozen=['scripts/run_final_historical_arff.py','scripts/final_historical_arff_compat.py','scripts/run_final_arff_campaign.py','scripts/run_ex8_arff_validation_selected_crossfit.py','GPU/lib/lib_ARFF.py','results/final_reproduction/production_manifest.arff_v1.json']
 frozen+=[str(p.relative_to(ROOT)) for p in (ROOT/'src').rglob('*.py')]
 frozen += [m['datasets'][ex]['path'] for ex in ['ex1','ex2','ex3','ex5','ex6','ex7']]
 infra.write_json(CONTROL/'launch_manifest.json',dict(jobs=jobs,gpus=[0,1,2,3],assignment='job index modulo4',session='final_historical_arff_accuracy',timing_context='four-GPU concurrent accuracy; non-isolated',frozen_sha256={p:digest(ROOT/p) for p in frozen}),exclusive=True)
 state(status='prepared',jobs_total=len(jobs));print('Prepared',len(jobs),'resolved historical corrected ARFF jobs; no launch.')
def run():
 plan=verify();errors=[]
 while True:
  previous=infra.read_json(OUT/'adam_campaign_control/state.json')
  if previous['status']=='complete':break
  if previous.get('errors'):state(status='waiting_review_of_adam_failure');raise RuntimeError('Earlier Adam worker failed; inspect before further dispatch')
  state(status='waiting_adam_campaign');time.sleep(15)
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
      command=[PYTHON,'-B','-u',str(ROOT/'scripts/run_final_historical_arff.py'),job['experiment'],'--seed',str(job['seed']),'--artifact-path',str(staged)]
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
  state(status='complete');print('All historical ARFF campaign jobs validated.',flush=True)
def main():
 action=sys.argv[1]
 if action=='prepare':prepare()
 elif action=='run':run()
 elif action=='status':print(json.dumps(infra.read_json(CONTROL/'state.json'),indent=2))
 else:raise ValueError(action)
if __name__=='__main__':main()
