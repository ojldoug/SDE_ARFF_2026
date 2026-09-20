"""Non-training adapter tests using real native initialization and serialization."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import sys, tempfile, json, unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import asdict
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import run_controlled_ex8_float64_v2 as r
import run_controlled_campaign_float64_v2 as c
import smoke_test_ex8_k128 as fourier
import smoke_test_ex8_mlp_w27 as mlp
from src.experiments.config import get_config
base_config=asdict(get_config('ex8'))
r.verify_registration()

def proxy(method):
 def main(argv):
  seed=int(argv[argv.index('--seed')+1]);path=Path(argv[argv.index('--artifact-path')+1])
  module=__import__(r.RUNNERS[method]);original=module.get_config;root=module.REPO_ROOT
  job=dict(study='baseline',method=method,K=128,N=80000,h=.0001,seed=seed)
  try:
   with patch.object(r.base,'check_assigned_gpu'):r.run_job(job,path,allow_cpu=True)
  finally:module.get_config=original;module.REPO_ROOT=root
 return main
for tests,method,name in [(fourier,'joint_fourier','test_joint_width_reaches_initialization_warmup_fit_and_real_archive'),(fourier,'split_fourier','test_split_width_reaches_warmup_fit_and_real_archive'),(mlp,'joint_mlp','test_joint_width_reaches_initialization_warmup_fit_and_real_archive')]:
 original=tests.dataset
 def dataset():
  d=original();d.r=d.r.astype(np.float64);return d
 with patch.object(tests,'dataset',side_effect=dataset),patch.object(tests.wrapper,'main',side_effect=proxy(method)):
  result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([tests.WidthTests(name)]))
  assert result.wasSuccessful()
print('PASS new data adapter reaches native initialization, warmup, fit entry and real serialization; float64 input fixture and unchanged training settings')
# Validate actual corrected input conversion outside training.
from src.experiments.dataset import load_dataset
import jax.numpy as jnp
actual=load_dataset(r.STUDY/'dataset_roots/N_80000/data/ex8.npz')
assert actual.r.dtype==np.float64
assert str(jnp.asarray(actual.r).dtype)=='float32'
for method in r.RUNNERS:
 job=dict(study='baseline',method=method,K=128,N=80000,h=.0001,seed=0)
 assert r.configuration(job).fourier_frequencies==128
assert asdict(get_config('ex8'))==base_config
# Mock only subprocess/GPU: verify new supervisor invokes v2 and refuses failed/valid path overwrite.
for code in [0,2]:
 with tempfile.TemporaryDirectory() as tmp:
  out=Path(tmp);control=out/'campaigns/fixture';control.mkdir(parents=True)
  job=dict(study='baseline',method='joint_fourier',K=128,N=80000,h=.0001,seed=0);launched=[]
  def popen(command,**kw):
   launched.append(command);assert command[3].endswith('run_controlled_ex8_float64_v2.py')
   assert kw['env']['CUDA_VISIBLE_DEVICES']=='0' and kw['env']['JAX_ENABLE_X64']=='false'
   p=Path(command[-1]);p.write_bytes(b'fixture');p.with_suffix('.context.json').write_text('{}');kw['stdout'].write('fixture\n')
   return type('P',(),{'pid':123,'wait':lambda self:code})()
  def valid(j,a,l,*args):return dict(artifact_sha256=r.digest(a),log_sha256=r.digest(l),algorithm_time=1.,parameter_count=1792)
  with patch.object(c,'STUDY',out),patch.object(c,'verify',return_value={'jobs':[job],'reuse':{}}),patch.object(c.infra,'gpu_processes',return_value=[]),patch.object(c.infra,'gpu_uuid',return_value='fixtureGPU'),patch.object(c,'validate',side_effect=valid),patch.object(c.subprocess,'Popen',side_effect=popen):
   c.run('fixture');directory,a,l,ctx=c.paths(job);assert a.exists()==(code==0)
   c.run('fixture');assert len(launched)==1
print('PASS exclusive v2 worker, failure preservation, no overwrites. No benchmark fitting.')
