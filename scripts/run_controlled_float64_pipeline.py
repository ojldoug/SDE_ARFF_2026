#!/usr/bin/env python3
"""Persistent baseline-first workflow. No scientific decisions from test rankings."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,json,subprocess,time,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_campaign_float64_v2_extended as c
ROOT,STUDY,PYTHON=c.ROOT,c.STUDY,c.PYTHON
CONTROL=STUDY/'pipeline'
FROZEN_SCRIPTS=['scripts/run_controlled_float64_pipeline.py','scripts/create_controlled_ex8_float64_lags.py','scripts/select_controlled_arff_float64_v2.py','scripts/summarize_controlled_ex8_float64_v2.py','scripts/summarize_controlled_float64_curves.py','scripts/run_controlled_campaign_float64_v2_extended.py']

def state(**values):
 old=c.infra.read_json(CONTROL/'state.json') if (CONTROL/'state.json').exists() else {}
 old.update(values,updated_at=c.infra.now());c.infra.write_json(CONTROL/'state.json',old)
 print(c.infra.now(),json.dumps(values),flush=True)
def verify():
 p=c.infra.read_json(CONTROL/'plan.json')
 for name,h in p['source_sha256'].items():assert c.adapter.digest(ROOT/name)==h,name
 c.adapter.verify_registration()
 if (CONTROL/'DISPATCH_HOLD.json').exists():raise RuntimeError('Explicit pipeline dispatch hold')
 return p

def run_command(name,command,env=None):
 verify();record=CONTROL/(name+'.complete.json');log=CONTROL/(name+'.log')
 if record.exists():
  done=c.infra.read_json(record);assert done['command']==command and done['log_sha256']==c.adapter.digest(log);return
 if log.exists():raise RuntimeError('Incomplete/suspicious operation; preserve and stop: '+name)
 state(status='running_operation',operation=name,command=command)
 with log.open('x') as f:
  p=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,env=env)
  c.infra.write_json(CONTROL/(name+'.running.json'),dict(command=command,pid=p.pid,started_at=c.infra.now()),exclusive=True)
  code=p.wait()
 c.infra.write_json(CONTROL/(name+'.exit.json'),dict(command=command,returncode=code,finished_at=c.infra.now()),exclusive=True)
 if code:raise RuntimeError('Operation failed; no automatic patch/retry: '+name)
 c.infra.write_json(record,dict(command=command,log_sha256=c.adapter.digest(log)),exclusive=True)

def wait_phase(phase):
 path=STUDY/'campaigns'/phase/'state.json';last=None
 while True:
  verify();status=c.infra.read_json(path)['status']
  if status=='complete':return
  if status=='failed_workers':raise RuntimeError('Worker failure in '+phase+'; outputs preserved')
  if status!=last:state(status='waiting_phase',phase=phase,campaign_status=status);last=status
  time.sleep(15)

def phase(phase):
 verify();path=STUDY/'campaigns'/phase
 if not path.exists():
  try:c.prepare(phase)
  except RuntimeError as e:
   if phase.startswith('stage2_') and '192 GPU-hour extension allowance exceeded' in str(e):
    c.infra.write_json(CONTROL/(phase+'.deferred.json'),dict(reason=str(e),decision='Predetermined timing-only budget; whole curve deferred, no test-based filtering',recorded_at=c.infra.now()),exclusive=True)
    state(status='curve_deferred',phase=phase,reason=str(e));return
   raise
 session='controlled_f64_'+phase
 if c.infra.read_json(path/'state.json')['status']=='prepared':
  # Named sessions have their own immutable supervisor logs and state.
  command=f'JAX_PLATFORMS=cpu {PYTHON} -B -u {ROOT}/scripts/run_controlled_campaign_float64_v2_extended.py run {phase} > {path}/supervisor.log 2>&1'
  assert not (path/'supervisor.log').exists()
  subprocess.run(['tmux','new-session','-d','-s',session,'-c',str(ROOT),command],check=True)
  subprocess.run(['tmux','set-option','-t',session,'remain-on-exit','on'],check=True)
  state(status='launched_phase',phase=phase,tmux_session=session)
 wait_phase(phase)

def main(action):
 if action=='prepare':
  CONTROL.mkdir(exist_ok=False)
  c.infra.write_json(CONTROL/'plan.json',dict(order=['baseline','baseline_summary','lag_data','stage1_capacity_N','stage1_h','calibration','calibration_selection','stage1_adaptation','stage1_summary','stage2_capacity','stage2_N','stage2_h','stage2_adaptation','final_summary'],baseline_jobs=150,source_sha256={p:c.adapter.digest(ROOT/p) for p in FROZEN_SCRIPTS},timing='Concurrent non-isolated accuracy only',no_manuscript_or_poster_edits=True,scientific_selection='Only preregistered validation-only ARFF calibration; no performance-based pruning',created_at=c.infra.now()),exclusive=True)
  state(status='prepared');return
 assert action=='run'
 with c.infra.lock_file(CONTROL/'driver.lock'):
  try:
   wait_phase('baseline')
   run_command('baseline_summary',[PYTHON,'-B','-u',str(ROOT/'scripts/summarize_controlled_ex8_float64_v2.py')])
   assert (STUDY/'baseline_summary/ex8_capacity_matched_five_methods_float64_v2_rmse_two_panel.pdf').exists()
   state(status='baseline_validated_and_reported',report=str(STUDY/'baseline_summary/report.md'))
   # No broader GPU use until that explicit validated baseline result exists.
   while c.infra.gpu_processes():time.sleep(15)
   gpu_env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',JAX_PLATFORMS='cuda',JAX_ENABLE_X64='true')
   run_command('lag_data',[PYTHON,'-B','-u',str(ROOT/'scripts/create_controlled_ex8_float64_lags.py')],gpu_env)
   phase('stage1_capacity_N');phase('stage1_h');phase('calibration')
   run_command('calibration_selection',[PYTHON,'-B','-u',str(ROOT/'scripts/select_controlled_arff_float64_v2.py')])
   phase('stage1_adaptation')
   run_command('stage1_summary',[PYTHON,'-B','-u',str(ROOT/'scripts/summarize_controlled_float64_curves.py'),'stage1'])
   for p in ['stage2_capacity','stage2_N','stage2_h','stage2_adaptation']:phase(p)
   run_command('final_summary',[PYTHON,'-B','-u',str(ROOT/'scripts/summarize_controlled_float64_curves.py'),'final'])
   state(status='complete')
  except Exception as e:
   state(status='stopped',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main(sys.argv[1])
