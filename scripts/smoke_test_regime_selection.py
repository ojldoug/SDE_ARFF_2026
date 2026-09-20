"""Synthetic validation-only selection proves no test metric is accessed."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import analyze_regime_ex8 as a
import run_regime_ex8_campaign as scheduler
class Archive:
 def __init__(self,h):self.h=h
 def __enter__(self):return self
 def __exit__(self,*args):pass
 def __getitem__(self,k):
  assert k in ['validation_nll','validation_drift_rmse','validation_covariance_rmse'],k
  return .5+np.log(self.h) if k=='validation_nll' else 1.
def jobs(phase):
 for j in scheduler.plans(phase):yield j,Path(str(j['h'])),Path(str(j['h'])),dict(artifact_sha256='fixture')
def logs(path,label):
 assert label=='validation'
 return dict(nll=.5+np.log(float(str(path))),drift_rmse=1.,covariance_rmse=1.)
with tempfile.TemporaryDirectory() as t:
 out=Path(t);(out/'dataset_roots/N_80000/h_0.0004').mkdir(parents=True)
 with patch.object(a,'OUT',out),patch.object(a,'checked_jobs',side_effect=jobs),patch.object(a,'log_metrics',side_effect=logs),patch.object(a.np,'load',side_effect=lambda p,**kw:Archive(float(str(p)))):
  a.select()
 result=json.loads((out/'selection.json').read_text())
 assert result['N_star']==80000 and result['h_star']==.0004 and not result['test_used']
 assert result['N_equivalence_demonstrated'] and result['h_equivalence_demonstrated']
print('PASS: selection only reads validation fields; known log(h) offset removed; stable fixture chooses smallest N and largest admissible h without rankings. No numerical training.')
