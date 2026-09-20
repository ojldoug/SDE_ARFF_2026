#!/usr/bin/env python3
"""Server-owned Phase1 -> validation selection -> frozen Phase2 workflow."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,time,json,subprocess,traceback
from pathlib import Path
import regime_ex8 as r
import run_regime_ex8_campaign as scheduler
ROOT,OUT,c=r.ROOT,r.STUDY,r.campaign
CONTROL=OUT/'pipeline';PYTHON=c.PYTHON
SOURCES=scheduler.SOURCES+['scripts/oracle_regime_ex8.py','scripts/create_regime_ex8_data.py','scripts/run_regime_ex8_pipeline.py','scripts/analyze_regime_ex8.py']
def verify():
 r.adapter.verify_registration()
 for p,h in c.infra.read_json(CONTROL/'plan.json')['source_sha256'].items():assert r.adapter.digest(ROOT/p)==h,p
 if (CONTROL/'DISPATCH_HOLD.json').exists():raise RuntimeError('Explicit dispatch hold')
def state(**v):
 old=c.infra.read_json(CONTROL/'state.json') if (CONTROL/'state.json').exists() else {}
 old.update(v,updated_at=c.infra.now());c.infra.write_json(CONTROL/'state.json',old);print(c.infra.now(),v,flush=True)
def wait_phase(phase):
 while True:
  verify();s=c.infra.read_json(OUT/'campaigns'/phase/'state.json')['status']
  if s=='complete':return
  if s=='failed_workers':raise RuntimeError('Failed worker in '+phase+'; no automatic numerical patch')
  time.sleep(15)
def phase(name):
 verify();control=OUT/'campaigns'/name
 if not control.exists():scheduler.prepare(name)
 s=c.infra.read_json(control/'state.json')['status'];session='capacity_regime_ex8_'+name
 if s=='prepared':
  assert not (control/'supervisor.log').exists()
  cmd=f'JAX_PLATFORMS=cpu {PYTHON} -B -u {ROOT}/scripts/run_regime_ex8_campaign.py run {name} > {control}/supervisor.log 2>&1'
  subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),cmd],check=True)
  subprocess.run(['tmux','set-option','-t',session,'remain-on-exit','on'],check=True)
 state(status='waiting_phase',phase=name,session=session);wait_phase(name)
def analyze(action):
 verify();log=CONTROL/(action+'.log');done=CONTROL/(action+'.complete.json')
 if done.exists():assert c.infra.read_json(done)['log_sha256']==r.adapter.digest(log);return
 assert not log.exists(),'Preserve failed/incomplete analysis'
 command=[PYTHON,'-B','-u',str(ROOT/'scripts/analyze_regime_ex8.py'),action]
 with log.open('x') as f:code=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT).returncode
 c.infra.write_json(CONTROL/(action+'.exit.json'),dict(command=command,returncode=code),exclusive=True)
 if code:raise RuntimeError('Analysis failed '+action)
 c.infra.write_json(done,dict(command=command,log_sha256=r.adapter.digest(log)),exclusive=True)
def main(action):
 if action=='prepare':
  CONTROL.mkdir(exist_ok=False)
  c.infra.write_json(CONTROL/'plan.json',dict(source_sha256={p:r.adapter.digest(ROOT/p) for p in SOURCES},order=['N','h','select_validation_only','N_report','h_report','capacity','capacity_report'],test_used_for_selection=False,seeds=list(range(10)),poster_edits=False),exclusive=True)
  state(status='prepared');return
 with c.infra.lock_file(CONTROL/'driver.lock'):
  try:
   verify();assert c.infra.read_json(OUT/'data/complete.json')['status']=='passed'
   assert c.infra.read_json(OUT/'oracle_moments.json')['status']=='passed'
   phase('N');phase('h')
   # Selection precedes even generation of aggregate test reports.
   analyze('select');analyze('N');analyze('h')
   phase('capacity');analyze('capacity');state(status='complete')
  except Exception as e:state(status='stopped',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main(sys.argv[1])
