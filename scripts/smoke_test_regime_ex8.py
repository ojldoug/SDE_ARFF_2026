"""Non-training width1024 adapter and workflow checks; no benchmark dispatch."""
import os
os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import sys,ast,types,unittest,tempfile,json
from unittest.mock import patch
from dataclasses import asdict
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import regime_ex8 as r
import run_regime_ex8_campaign as scheduler
import analyze_regime_ex8 as analysis
from src.experiments.config import get_config
from src.experiments.mlp_size import matched_two_layer_width
from src.adam.mlp import initialize_model
import jax
original_config=asdict(get_config('ex8'));r.adapter.verify_registration()
assert matched_two_layer_width(state_dimension=2,covariance_output_dimension=3,n_frequencies=1024)==(81,14180,14336)
model=initialize_model(jax.random.PRNGKey(0),input_dimension=2,output_dimension=2,diff_type='symmetric',hidden_sizes=(81,81))
assert sum(v.size for v in jax.tree_util.tree_leaves(model))==14180
class Width(ast.NodeTransformer):
 def __init__(self,mlp=False):self.mlp=mlp
 def visit_Constant(self,node):
  if isinstance(node.value,int) and not isinstance(node.value,bool):
   mapping={128:1024};mapping.update({27:81,1814:14180} if self.mlp else {})
   if node.value in mapping:return ast.copy_location(ast.Constant(mapping[node.value]),node)
  return node
 def visit_Tuple(self,node):
  if not self.mlp and len(node.elts)==2 and all(isinstance(e,ast.Constant) for e in node.elts) and [e.value for e in node.elts]==[256,3]:node.elts[0]=ast.Constant(2048)
  return self.generic_visit(node)
def proxy(method):
 def main(argv):
  path=Path(argv[argv.index('--artifact-path')+1]);seed=int(argv[argv.index('--seed')+1]);module=__import__(r.adapter.RUNNERS[method]);config=module.get_config;root=module.REPO_ROOT
  j=scheduler.job('regime_N',method,seed,1024,80000,.0001)
  try:
   with patch.object(r.adapter.base,'check_assigned_gpu'):r.adapter.run_job(j,path,allow_cpu=True)
  finally:module.get_config=config;module.REPO_ROOT=root
 return main
for filename,methods in [('smoke_test_ex8_k128.py',[('joint_fourier','test_joint_width_reaches_initialization_warmup_fit_and_real_archive'),('split_fourier','test_split_width_reaches_warmup_fit_and_real_archive')]),('smoke_test_ex8_mlp_w27.py',[('joint_mlp','test_joint_width_reaches_initialization_warmup_fit_and_real_archive')])]:
 file=Path(__file__).with_name(filename);module=types.ModuleType('regime_fixture');module.__file__=str(file)
 tree=ast.fix_missing_locations(Width('mlp' in filename).visit(ast.parse(file.read_text())))
 exec(compile(tree,str(file),'exec'),module.__dict__)
 original=module.dataset
 def dataset():
  data=original();data.r=data.r.astype(np.float64);return data
 for method,name in methods:
  with patch.object(module,'dataset',side_effect=dataset),patch.object(module.wrapper,'main',side_effect=proxy(method)):
   result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([module.WidthTests(name)]));assert result.wasSuccessful()
assert asdict(get_config('ex8'))==original_config
for method in r.adapter.RUNNERS:
 cfg=r.adapter.configuration(scheduler.job('regime_N',method,0,1024,640000,.0001));assert cfg.fourier_frequencies==1024
 assert cfg.adam==get_config('ex8').adam and cfg.adam_mlp==get_config('ex8').adam_mlp
assert analysis.equivalent(np.arange(10.)+10,np.arange(10.)+10,'drift_rmse')['passes']
assert not analysis.equivalent(np.ones(10),np.ones(10)*2,'drift_rmse')['passes']
assert len(scheduler.plans('N'))==200 and len(scheduler.plans('h'))==250
# Verify actual entrypoint substitution also updates the recorded command.
commands=[]
def fake_worker(phase):
 command=[r.campaign.PYTHON,'-B','-u',str(r.ROOT/'scripts/run_controlled_ex8_float64_v2.py')]
 r.campaign.subprocess.Popen(command);assert command[-1].endswith('/run_regime_ex8.py')
with patch.object(r.campaign,'run',side_effect=fake_worker),patch.object(r.campaign.subprocess,'Popen',side_effect=lambda cmd,*a,**k:commands.append(cmd)):
 scheduler.run('fixture')
assert len(commands)==1
print('PASS: K1024/MLP81, initialization/warmup/fit entry/archive, unchanged global numerical settings, equivalence rule, complete10-seed grids, exact worker command routing. No benchmark fits.')
