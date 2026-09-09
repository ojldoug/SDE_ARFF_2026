"""CPU-only tiny synthetic integration; never fits canonical Experiment 8."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,tempfile
from pathlib import Path
from dataclasses import replace,asdict
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_ex8_split_mlp_w27 as w
import run_adam_split_mlp_experiment as r
import src.adam.split_mlp as s
from src.adam.mlp import initialize_model
from src.experiments.config import get_config
import jax
cfg=w.effective_config('ex8');assert cfg.adam_mlp.epochs==300 and cfg.adam_mlp.batch_size==256 and cfg.adam_mlp.learning_rate==.001
base=asdict(get_config('ex8'));effective=asdict(cfg)
assert {k for k in base if base[k]!=effective[k]}=={'fourier_frequencies'}
rng=np.random.default_rng(71)
data=SimpleNamespace(x=rng.normal(size=(60,2)).astype('f'),r=rng.normal(scale=.01,size=(60,2)).astype('f'),h=np.full((60,1),.0001,dtype='f'),train_idx=np.arange(40),validation_idx=np.arange(40,50),test_idx=np.arange(50,60))
fixture=replace(cfg,adam_mlp=replace(cfg.adam_mlp,epochs=2,batch_size=8))
seen=[];inits=[];original_init=s.initialize_model;original_warm=r.prepare_compiled_functions;original_fit=s._fit_one_regression

def init(*a,**k):
 assert k['hidden_sizes']==(27,27)
 model=original_init(*a,**k);inits.append(model);return model

def warm(*a,**k):
 assert k['hidden_sizes']==(27,27) and k['n_folds']==5
 seen.append('warmup');return original_warm(*a,**k)

def fit(*a,**k):
 assert k['epochs']==2 and k['batch_size']==8
 x=np.asarray(a[2]);assert all(any(np.array_equal(row,t) for t in data.x[:40]) for row in x)
 np.testing.assert_array_equal(a[5],data.x[40:50]);seen.append(len(x))
 return original_fit(*a,**k)
with tempfile.TemporaryDirectory() as tmp:
 path=Path(tmp)/'fixture.npz'
 with patch.object(w,'effective_config',return_value=fixture),patch.object(w,'check_assigned_gpu'),patch.object(r,'load_dataset',return_value=data),patch.object(r,'prepare_compiled_functions',side_effect=warm),patch.object(s,'initialize_model',side_effect=init),patch.object(s,'_fit_one_regression',side_effect=fit):
  w.main(['--seed','9','--artifact-path',str(path)])
 with np.load(path,allow_pickle=False) as z:
  a={k:z[k] for k in z.files};r.validate_artifact(a)
  assert int(a['actual_mlp_parameter_count'])==1814 and int(a['hidden_width'])==27
  assert a['drift_weight_0'].shape==(2,27) and a['covariance_weight_2'].shape==(27,3)
  assert len(a['fold_id'])==40 and len(a['final_drift_validation_mse'])==2
  assert 'test_nll' in a
assert seen==['warmup',32,32,32,32,32,40,40],seen
_,key=jax.random.split(jax.random.PRNGKey(9));joint=initialize_model(key,input_dimension=2,output_dimension=2,diff_type='symmetric',hidden_sizes=(27,27))
for x,y in zip(jax.tree_util.tree_leaves(inits[0]),jax.tree_util.tree_leaves(joint)):np.testing.assert_array_equal(x,y)
assert get_config('ex8').fourier_frequencies==512
print('PASS: unchanged global config; width27 initialization/warmup/seven regressions/archive; same joint initial arrays; training excludes test; 1814 parameters.')
