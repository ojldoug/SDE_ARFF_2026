"""Route accepted tiny synthetic split-MLP smoke through the versioned adapter."""
import os
os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import sys,runpy
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_ex8_float64_v2 as r
import run_ex8_split_mlp_w27 as w
import run_adam_split_mlp_experiment as native

def main(argv):
 job=dict(study='baseline',method='split_mlp',K=128,N=80000,h=.0001,seed=int(argv[argv.index('--seed')+1]))
 path=Path(argv[argv.index('--artifact-path')+1]);original=native.get_config;root=native.REPO_ROOT
 try:
  # The existing fixture deliberately uses 60 synthetic rows and two epochs.
  # It inspects all seven regressions and initialization PRNG identity.
  with patch.object(r.base,'configuration',return_value=w.effective_config('ex8')),patch.object(r.base,'check_assigned_gpu'):
   r.run_job(job,path,allow_cpu=True)
 finally:native.get_config=original;native.REPO_ROOT=root
with patch.object(w,'main',side_effect=main):
 runpy.run_path(str(Path(__file__).with_name('smoke_test_ex8_split_mlp_w27.py')),run_name='__main__')
