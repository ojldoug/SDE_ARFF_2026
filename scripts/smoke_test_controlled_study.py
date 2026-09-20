"""No benchmark fitting: configuration/model-size, source freezing and supervisor fixtures."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,json,tempfile
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import run_controlled_ex8 as r
import run_controlled_campaign as c
from src.experiments.config import get_config
from src.adam.mlp import initialize_model
from src.experiments.mlp_size import matched_two_layer_width
import jax
m=r.verify_registration();base=asdict(get_config('ex8'))
for size in m['capacity']:
 w,count,target=matched_two_layer_width(state_dimension=2,covariance_output_dimension=3,n_frequencies=size['K'])
 assert (w,count,target)==(size['mlp_width'],size['mlp_parameters'],size['fourier_parameters'])
 model=initialize_model(jax.random.PRNGKey(0),input_dimension=2,output_dimension=2,diff_type='symmetric',hidden_sizes=(w,w))
 assert sum(a.size for a in jax.tree_util.tree_leaves(model))==count
 for method in m['methods']:
  job=dict(study='capacity',method=method,K=size['K'],N=80000,h=.0001,seed=0)
  cfg=r.configuration(job);assert cfg.fourier_frequencies==size['K']
  assert cfg.adam==get_config('ex8').adam and cfg.adam_mlp==get_config('ex8').adam_mlp
assert asdict(get_config('ex8'))==base
print('PASS all5 capacities, matched architecture/count, accepted Adam settings, unchanged global config')
for code in [0,2]:
 with tempfile.TemporaryDirectory() as tmp:
  out=Path(tmp);control=out/'campaigns'/'fixture';control.mkdir(parents=True);job=dict(study='capacity',method='joint_fourier',K=32,N=80000,h=.0001,seed=0);launched=[]
  def popen(command,**kw):
   launched.append(command);assert kw['env']['CUDA_VISIBLE_DEVICES']=='0' and kw['env']['JAX_PLATFORMS']=='cuda'
   path=Path(command[-1]);path.write_bytes(b'fixture');path.with_suffix('.context.json').write_text('{}');kw['stdout'].write('fixture console\n')
   return type('P',(),{'pid':123,'wait':lambda self:code})()
  def valid(j,a,l,*args):return dict(artifact_sha256=r.digest(a),log_sha256=r.digest(l),algorithm_time=1.,parameter_count=448)
  with patch.object(c,'STUDY',out),patch.object(c,'verify',return_value={'jobs':[job],'reuse':{}}),patch.object(c.infra,'gpu_processes',return_value=[]),patch.object(c.infra,'gpu_uuid',return_value='testGPU'),patch.object(c,'validate',side_effect=valid),patch.object(c.subprocess,'Popen',side_effect=popen):
   c.run('fixture');p,a,l,ctx=c.paths(job)
   assert a.exists()==(code==0) and l.read_text()=='fixture console\n'
   c.run('fixture');assert len(launched)==1
  print('PASS supervisor preserves output and prevents relaunch; exit',code)
print('No benchmark processes launched.')
