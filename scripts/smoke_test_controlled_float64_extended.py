"""No training: complete-grid scheduling/reuse and five-seed timing-policy fixtures."""
import os
os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import tempfile,sys,json
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_campaign_float64_v2_extended as c
m=c.adapter.verify_registration()
def write(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v))
def valid(job,a,l,ctx=None):
 assert a.read_bytes()==b'fixture artifact' and l.read_bytes()==b'fixture log'
 return dict(artifact_sha256=c.adapter.digest(a),log_sha256=c.adapter.digest(l),algorithm_time=1.,parameter_count=1792)
def complete(job):
 directory,a,l,ctx=c.paths(job);directory.mkdir(parents=True)
 a.write_bytes(b'fixture artifact');l.write_bytes(b'fixture log');ctx.write_text('{}')
 write(directory/'complete.json',valid(job,a,l))
with tempfile.TemporaryDirectory() as t,patch.object(c,'validate',side_effect=valid):
 out=Path(t)
 with patch.object(c,'STUDY',out):
  write(out/'dietrich/gate.json',dict(broad_controlled_training_allowed=True))
  write(out/'campaigns/baseline/state.json',dict(status='complete'))
  write(out/'baseline_summary/summary.json',{})
  for n in m['N_grid']:write(out/f'dataset_roots/N_{n}/dataset_view.json',{})
  roots={format(h,'.8g'):'lags/'+format(h,'.8g') for h in m['h_grid']}
  for folder in roots.values():write(out/folder/'dataset_view.json',{})
  write(out/'lag_datasets/complete.json',dict(dataset_roots=roots))
  variants=[dict(variant=f'mode{i}',metropolis_test=a,resampling=b,delta=.1,lambda_reg=.0001) for i,(a,b) in enumerate([(True,False),(False,True),(True,True)])]
  write(out/'calibration_selection.json',dict(selected_modes=variants))
  for method in m['methods']:
   for seed in range(30):complete(dict(study='baseline',method=method,K=128,N=80000,h=.0001,seed=seed))
  expected={'stage1_capacity_N':(225,25),'stage1_h':(125,25),'calibration':(36,0),'stage1_adaptation':(15,0),'stage2_capacity':(625,125),'stage2_N':(500,0),'stage2_h':(625,125),'stage2_adaptation':(75,0)}
  for phase,(n,reused) in expected.items():
   c.prepare(phase);plan=c.infra.read_json(out/'campaigns'/phase/'plan.json')
   assert (len(plan['jobs']),len(plan['reuse']))==(n,reused),(phase,len(plan['jobs']),len(plan['reuse']))
   for job in plan['jobs']:complete(job)
   write(out/'campaigns'/phase/'state.json',dict(status='complete'))
   if phase.startswith('stage2_'):assert abs(plan['predicted_gpu_seconds']-(n-reused)*121.)<1e-8
   print('PASS',phase,'jobs',n,'corrected baseline reuse',reused)
print('PASS all prescribed grids, no old-data reuse, five-seed cost accounting excluding reused fits. No GPU/numerical dispatch.')
