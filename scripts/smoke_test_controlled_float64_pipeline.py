"""Pure workflow fixtures: baseline gate, failure stop and validation-only ties."""
import os
os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import tempfile,sys,itertools
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_float64_pipeline as p
import select_controlled_arff_float64_v2 as calibration
rows=[]
for metro,resample in [(True,False),(False,True),(True,True)]:
 for delta,lam in itertools.product([.1,.2],[1e-4,1e-3]):
  rows.append(dict(variant=f'{metro}_{resample}_{delta}_{lam}',metropolis_test=metro,resampling=resample,delta=delta,lambda_reg=lam,mean_validation_nll=1.))
selected=calibration.select(rows)
assert len(selected)==3 and all(v['delta']==.1 and v['lambda_reg']==1e-4 for v in selected)
print('PASS preregistered validation-only tie ordering')
for fail in [False,True]:
 with tempfile.TemporaryDirectory() as temp:
  root=Path(temp);control=root/'pipeline';control.mkdir();summary=root/'baseline_summary';summary.mkdir()
  (summary/'ex8_capacity_matched_five_methods_float64_v2_rmse_two_panel.pdf').write_bytes(b'fixture')
  events=[]
  def wait(phase):events.append('wait_'+phase)
  def command(name,args,env=None):
   events.append(name)
   if fail and name=='baseline_summary':raise RuntimeError('fixture failed baseline summary')
  def phase(name):events.append(name)
  with patch.object(p,'STUDY',root),patch.object(p,'CONTROL',control),patch.object(p,'wait_phase',side_effect=wait),patch.object(p,'run_command',side_effect=command),patch.object(p,'phase',side_effect=phase),patch.object(p.c.infra,'gpu_processes',return_value=[]):
   try:p.main('run')
   except RuntimeError:
    assert fail
  assert events[:2]==['wait_baseline','baseline_summary']
  if fail:assert events==['wait_baseline','baseline_summary']
  else:assert events[2:]==['lag_data','stage1_capacity_N','stage1_h','calibration','calibration_selection','stage1_adaptation','stage1_summary','stage2_capacity','stage2_N','stage2_h','stage2_adaptation','final_summary']
print('PASS all downstream work gated on complete validated baseline; failure stops without dispatch. No numerical subprocesses launched.')
