"""Final descriptive extension; waits for Phase2 and stops after its summary."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,time,json,subprocess,traceback
from pathlib import Path
import regime_ex8_final_h as r
import run_regime_ex8_final_h_campaign as scheduler
ROOT,OUT,c=r.ROOT,r.STUDY,r.campaign
PARENT=scheduler.PARENT;CONTROL=OUT/'pipeline';PYTHON=c.PYTHON
def verify():
 r.adapter.verify_registration()
 for p,digest in c.infra.read_json(CONTROL/'plan.json')['source_sha256'].items():assert r.adapter.digest(ROOT/p)==digest,p
 if (CONTROL/'DISPATCH_HOLD.json').exists():raise RuntimeError('Explicit dispatch hold')
def state(**v):
 old=c.infra.read_json(CONTROL/'state.json') if (CONTROL/'state.json').exists() else {}
 old.update(v,updated_at=c.infra.now());c.infra.write_json(CONTROL/'state.json',old);print(c.infra.now(),v,flush=True)
def wait_parent():
 state(status='waiting_parent_phase2',phase='parent_phase2',session='capacity_regime_ex8_v2_capacity')
 while True:
  verify();s=c.infra.read_json(PARENT/'pipeline/state.json')
  workers=c.infra.read_json(PARENT/'campaigns/capacity/state.json')
  if workers.get('errors') or workers['status']=='failed_workers':raise RuntimeError('Parent Phase2 worker failure')
  if s.get('errors') or s['status'] in ['failed_workers','stopped']:raise RuntimeError('Parent campaign worker failure; amendment stopped')
  if s['status']=='complete':return
  if s['status'] not in ['prepared','running','waiting_machine_idle','waiting_phase','analyzing']:raise RuntimeError('Unexpected parent state '+str(s))
  time.sleep(15)
def data():
 done=OUT/'data/complete.json'
 if done.exists():assert c.infra.read_json(done)['status']=='passed';return
 log=CONTROL/'data.log';assert not log.exists(),'Preserve incomplete data attempt'
 state(status='waiting_machine_idle',phase='data')
 while c.infra.gpu_processes():verify();time.sleep(15)
 verify();state(status='generating_coupled_extension',phase='data',gpu=0)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cuda',JAX_ENABLE_X64='true',CONDA_DEFAULT_ENV='arff-sde')
 command=[PYTHON,'-B','-u',str(ROOT/'scripts/create_regime_ex8_final_h_data.py')]
 with log.open('x') as f:code=subprocess.run(command,cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT).returncode
 c.infra.write_json(CONTROL/'data.exit.json',dict(command=command,returncode=code),exclusive=True)
 if code:raise RuntimeError('Coupled data validation failed; preserve outputs, no dispatch')
 assert c.infra.read_json(done)['status']=='passed'
def phase(name):
 verify();control=OUT/'campaigns'/name
 if not control.exists():scheduler.prepare()
 s=c.infra.read_json(control/'state.json')['status'];session='capacity_regime_ex8_final_h_jobs'
 if s=='prepared':
  assert not (control/'supervisor.log').exists()
  cmd=f'JAX_PLATFORMS=cpu {PYTHON} -B -u {ROOT}/scripts/run_regime_ex8_final_h_campaign.py run > {control}/supervisor.log 2>&1'
  subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),cmd],check=True)
  subprocess.run(['tmux','set-option','-t',session,'remain-on-exit','on'],check=True)
 state(status='waiting_phase',phase=name,session=session)
 while True:
  verify();s=c.infra.read_json(control/'state.json')
  if s.get('errors') or s['status'] in ['failed_workers','stopped']:raise RuntimeError('Failed worker in '+name)
  if s['status']=='complete':return
  time.sleep(15)
def analyze(action):
 verify();log=CONTROL/(action+'.log');done=CONTROL/(action+'.complete.json')
 if done.exists():assert c.infra.read_json(done)['log_sha256']==r.adapter.digest(log);return
 assert not log.exists(),'Preserve failed/incomplete analysis'
 command=[PYTHON,'-B','-u',str(ROOT/'scripts/summarize_regime_ex8_final_h.py')]
 state(status='analyzing',phase=action)
 with log.open('x') as f:code=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
 c.infra.write_json(CONTROL/(action+'.exit.json'),dict(command=command,returncode=code),exclusive=True)
 if code:raise RuntimeError('Analysis failed '+action)
 c.infra.write_json(done,dict(command=command,log_sha256=r.adapter.digest(log)),exclusive=True)
def main():
 with c.infra.lock_file(CONTROL/'driver.lock'):
  try:
   verify();wait_parent();data();phase('h');analyze('final_h_summary')
   state(status='complete',phase='complete')
  except Exception as e:state(status='stopped',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
