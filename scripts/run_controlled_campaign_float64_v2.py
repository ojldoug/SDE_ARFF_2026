#!/usr/bin/env python3
"""Persistent exclusive-GPU execution, immutable jobs and explicit artifact reuse."""
import os
os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import sys,json,subprocess,threading,time,shutil
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import run_ex8_k128_campaigns as infra
import run_controlled_ex8_float64_v2 as adapter
from validate_controlled_ex8 import validate
STUDY=adapter.STUDY;PYTHON=infra.PYTHON;LOCK=threading.Lock()
BASE_DIRS=dict(joint_fourier='adam_fourier_ex8_k128',split_fourier='adam_split_fourier_ex8_k128',arff='arff_ex8_corrected',joint_mlp='adam_mlp_ex8_w27',split_mlp='adam_split_mlp_ex8_w27')
def key(job):
 variant=('/'+job['variant']) if 'variant' in job else (('/h_'+format(job['h'],'.8g')) if job['study']=='h' else '')
 return f"{job['study']}{variant}/K_{job['K']}_N_{job['N']}/{job['method']}/seed_{job['seed']}"
def paths(job):
 p=STUDY/'production'/key(job)
 return p,p/'artifact.npz',p/'console.log',p/'artifact.context.json'
def state(control,**changes):
 with LOCK:
  p=control/'state.json';v=infra.read_json(p) if p.exists() else {};v.update(changes);v['updated_at']=infra.now();infra.write_json(p,v)
def verify(control):
 plan=infra.read_json(control/'plan.json');adapter.verify_registration()
 for p,h in plan['frozen_sha256'].items():
  if adapter.digest(ROOT/p)!=h:raise RuntimeError('Frozen launcher changed: '+p)
 return plan

def prepare(phase):
 m=adapter.verify_registration();control=STUDY/'campaigns'/phase
 if control.exists():raise FileExistsError('Existing campaign; inspect rather than overwrite')
 gate=infra.read_json(STUDY/'dietrich/gate.json');assert gate['broad_controlled_training_allowed']
 jobs=[]
 is_stage2=phase.startswith('stage2_');seeds=list(range(30)) if phase=='baseline' else (list(range(5,30)) if is_stage2 else m['stage1_seeds'])
 if phase!='baseline':
  assert infra.read_json(STUDY/'campaigns/baseline/state.json')['status']=='complete'
  assert (STUDY/'baseline_summary/summary.json').is_file(), 'Validate and summarize baseline before broader dispatch'
 studies={'baseline':['baseline'],'stage1_capacity_N':['capacity','N'],'stage1_h':['h'],'stage2_capacity':['capacity'],'stage2_N':['N'],'stage2_h':['h'],'calibration':['calibration'],'stage1_adaptation':['adaptation'],'stage2_adaptation':['adaptation']}[phase]
 if 'baseline' in studies:
  for method in m['methods']:
   for seed in seeds:jobs.append(dict(study='baseline',method=method,K=128,N=80000,h=.0001,seed=seed))
 if 'capacity' in studies:
  for k in [v['K'] for v in m['capacity']]:
   for method in m['methods']:
    for seed in seeds:jobs.append(dict(study='capacity',method=method,K=k,N=80000,h=.0001,seed=seed))
 if 'N' in studies:
  for n in m['N_grid'][:-1]:
   for method in m['methods']:
    for seed in seeds:jobs.append(dict(study='N',method=method,K=128,N=n,h=.0001,seed=seed))
 if 'h' in studies:
  hviews=infra.read_json(STUDY/'lag_datasets/complete.json')
  for h in m['h_grid']:
   for method in m['methods']:
    for seed in seeds:jobs.append(dict(study='h',method=method,K=128,N=80000,h=h,seed=seed,dataset_root=hviews['dataset_roots'][format(h,'.8g')]))
 if 'calibration' in studies:
  for mode in m['calibration']['modes']:
   for delta in m['calibration']['delta']:
    for lam in m['calibration']['lambda_reg']:
     variant=f"metro{int(mode['metropolis_test'])}_resample{int(mode['resampling'])}_delta{delta}_lambda{lam}"
     for seed in m['calibration']['seeds']:jobs.append(dict(study='calibration',method='arff',K=128,N=80000,h=.0001,seed=seed,variant=variant,delta=delta,lambda_reg=lam,**mode))
 if 'adaptation' in studies:
  selection=infra.read_json(STUDY/'calibration_selection.json')
  for mode in selection['selected_modes']:
   for seed in seeds:jobs.append(dict(study='adaptation',method='arff',K=128,N=80000,h=.0001,seed=seed,**mode))
 # Reuse only validated, configuration-identical corrected-data baseline runs.
 reuse={}
 for j in jobs:
  if j['study']=='capacity' and j['K']==128 and j['N']==80000:
   original=dict(j,study='baseline');d,a,l,ctx=paths(original)
   complete=infra.read_json(d/'complete.json');checked=validate(original,a,l,ctx)
   assert checked['artifact_sha256']==complete['artifact_sha256'] and checked['log_sha256']==complete['log_sha256']
   reuse[key(j)]=dict(artifact=str(a),log=str(l),**checked)
 for j in jobs:
  view=STUDY/j.get('dataset_root',f"dataset_roots/N_{j['N']}")
  assert (view/'dataset_view.json').exists()
  if paths(j)[0].exists():raise FileExistsError(str(paths(j)[0]))
 if is_stage2:
  predicted=0.
  for j in jobs:
   first=dict(j,seed=0);done=infra.read_json(paths(first)[0]/'complete.json')
   predicted+=float(done['algorithm_time'])+120.
  spent=0.
  for oldplan in (STUDY/'campaigns').glob('stage2_*/plan.json'):
   spent+=float(infra.read_json(oldplan).get('predicted_gpu_seconds',0.))
  if spent+predicted>192*3600:raise RuntimeError('Predetermined192 GPU-hour extension allowance exceeded; defer whole curve, no performance filtering')
 else:predicted=None
 control.mkdir(parents=True)
 frozen=['scripts/run_controlled_ex8_float64_v2.py','scripts/run_controlled_campaign_float64_v2.py','scripts/run_controlled_ex8.py','scripts/run_controlled_campaign.py','scripts/validate_controlled_ex8.py','scripts/run_ex8_split_mlp_w27.py','scripts/run_ex8_k128_campaigns.py']
 infra.write_json(control/'plan.json',dict(phase=phase,jobs=jobs,reuse=reuse,predicted_gpu_seconds=predicted,gpus=[0,1,2,3],timing='four-GPU concurrent non-isolated accuracy',created_at=infra.now(),frozen_sha256={p:adapter.digest(ROOT/p) for p in frozen}),exclusive=True)
 state(control,status='prepared',jobs_total=len(jobs),reused=len(reuse),new_jobs=len(jobs)-len(reuse));print('Prepared',len(jobs),'jobs,',len(reuse),'accepted baseline reuses; no training launched')

def run(phase):
 control=STUDY/'campaigns'/phase;plan=verify(control);errors=[]
 with infra.lock_file(control/'supervisor.lock'):
  while infra.gpu_processes():state(control,status='waiting_machine_idle');time.sleep(15)
  state(control,status='running')
  def worker(gpu):
   with infra.lock_file(control/f'gpu_{gpu}.lock') as fd:
    uuid=infra.gpu_uuid(gpu)
    for i,job in enumerate(plan['jobs']):
     if i%4!=gpu:continue
     directory,artifact,log,context=paths(job)
     try:
      verify(control)
      if (control/'DISPATCH_HOLD.json').exists():raise RuntimeError('Explicit dispatch hold; preserve outputs')
      if directory.exists():
       complete=infra.read_json(directory/'complete.json')
       checked=validate(job,artifact,log,None if complete.get('reused') else context)
       assert checked['artifact_sha256']==complete['artifact_sha256'] and checked['log_sha256']==complete['log_sha256'];continue
      while any(row[0].strip()==uuid for row in infra.gpu_processes()):time.sleep(15)
      directory.mkdir(parents=True,exist_ok=False)
      infra.write_json(directory/'job.json',job,exclusive=True)
      record=dict(job=job,gpu=gpu,started_at=infra.now())
      if key(job) in plan['reuse']:
       source=plan['reuse'][key(job)]
       for src,dst in [(Path(source['artifact']),artifact),(Path(source['log']),log)]:
        with src.open('rb') as f,dst.open('xb') as g:shutil.copyfileobj(f,g)
       checked=validate(job,artifact,log)
       assert checked['artifact_sha256']==source['artifact_sha256'] and checked['log_sha256']==source['log_sha256']
       record.update(reused=True,reuse_source=source,**checked)
      else:
       staged=directory/'staged.npz'
       command=[PYTHON,'-B','-u',str(ROOT/'scripts/run_controlled_ex8_float64_v2.py'),'--job-json',str(directory/'job.json'),'--artifact-path',str(staged)]
       env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),JAX_PLATFORMS='cuda',JAX_ENABLE_X64='false',CONDA_DEFAULT_ENV='arff-sde',CONDA_PREFIX=str(Path(PYTHON).parent.parent));env['PATH']=str(Path(PYTHON).parent)+os.pathsep+env.get('PATH','')
       while any(row[0].strip()==uuid for row in infra.gpu_processes()):time.sleep(15)
       with log.open('x') as f:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,pass_fds=(fd,))
        record.update(command=command,pid=p.pid,status='running');infra.write_json(directory/'running.json',record,exclusive=True)
        print(infra.now(),'START',key(job),'gpu',gpu,'pid',p.pid,flush=True);code=p.wait()
       infra.write_json(directory/'exit.json',dict(record,returncode=code,finished_at=infra.now()),exclusive=True)
       if code:raise RuntimeError('Numerical process failed '+str(code))
       staged_context=staged.with_suffix('.context.json');checked=validate(job,staged,log,staged_context)
       verify(control);os.link(staged,artifact);os.link(staged_context,context);record.update(reused=False,**checked)
      record.update(finished_at=infra.now(),status='complete');infra.write_json(directory/'complete.json',record,exclusive=True)
      print(infra.now(),'COMPLETE',key(job),'gpu',gpu,'reuse',record['reused'],flush=True)
     except Exception as e:
      errors.append(dict(job=job,gpu=gpu,error=repr(e)));state(control,errors=errors)
      print(infra.now(),'WORKER STOPPED',gpu,key(job),repr(e),flush=True);return
  with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(worker,range(4)))
  state(control,status='failed_workers' if errors else 'complete',errors=errors)
  print('Phase ended',phase,'errors',len(errors),flush=True)
if __name__=='__main__':
 action,phase=sys.argv[1:3]
 if action=='prepare':prepare(phase)
 elif action=='run':run(phase)
 elif action=='status':print(json.dumps(infra.read_json(STUDY/'campaigns'/phase/'state.json'),indent=2))
 else:raise ValueError(action)
