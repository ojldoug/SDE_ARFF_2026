"""Non-training final-extension safety/configuration smoke checks."""
import os
os.environ.update(JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
from pathlib import Path
import sys,ast,json
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import run_regime_ex8_final_h_campaign as s
import run_regime_ex8_final_h_pipeline as p
from oracle_regime_ex8 import oracle
from src.experiments.mlp_size import matched_two_layer_width
for source in ROOT.glob('scripts/*regime_ex8_final_h*.py'):ast.parse(source.read_text())
p.verify();m=s.r.adapter.verify_registration();old=json.loads((s.PARENT/'manifest.json').read_text())
for key in ['settings','methods','stage1_seeds','stage2_seeds','dataset','capacity','N_grid','execution','frozen_source_sha256']:assert m[key]==old[key],key
assert m['diagnostic']==old['diagnostic']
assert not m['amendment']['selection_allowed'] and not m['amendment']['further_extensions_allowed']
assert m['amendment']['hard_maximum_h']==.002
for h in [.0015,.002]:assert oracle(h)['relative_discrete_drift_bias']<=.001
jobs=s.plans();new=[j for j in jobs if j['h'] not in old['h_grid']]
assert len(jobs)==450 and len(new)==100 and max(j['h'] for j in jobs)==.002
assert all(j['K']==1024 and j['N']==640000 for j in jobs)
for h in [.0015,.002]:
 for method in m['methods']:
  assert {j['seed'] for j in new if j['method']==method and j['h']==h}==set(range(10))
  cfg=s.r.adapter.configuration(dict(K=1024,N=640000,h=h,method=method))
  assert cfg.fourier_frequencies==1024 and cfg.data.em_substeps==round(h/1e-7)
assert matched_two_layer_width(state_dimension=2,covariance_output_dimension=3,n_frequencies=1024)==(81,14180,14336)
# Wait gate: no numerical subprocess, no control operation, no writes.
def states(path):
 return dict(status='complete',errors=[])
with patch.object(p,'verify'),patch.object(p,'state'),patch.object(p.c.infra,'read_json',side_effect=states):p.wait_parent()
def failed(path):return dict(status='stopped',error='fixture') if path==s.PARENT/'pipeline/state.json' else dict(status='complete',errors=[])
with patch.object(p,'verify'),patch.object(p,'state'),patch.object(p.c.infra,'read_json',side_effect=failed):
 try:p.wait_parent()
 except RuntimeError:pass
 else:raise AssertionError('Parent scientific stop was not propagated')
main=ast.parse((ROOT/'scripts/run_regime_ex8_final_h_pipeline.py').read_text())
main=next(v for v in main.body if isinstance(v,ast.FunctionDef) and v.name=='main')
assert [n.args[0].value for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['phase','analyze']]==['h','final_h_summary']
captured=[]
def fake(command,*a,**kw):captured.append(command.copy())
def fake_run(phase):
 assert phase=='h'
 s.c.subprocess.Popen([s.c.PYTHON,'-B','-u',str(ROOT/'scripts/run_controlled_ex8_float64_v2.py'),'--job-json','fixture','--artifact-path','fixture.npz'])
with patch.object(s.c.subprocess,'Popen',fake),patch.object(s.c,'run',fake_run):s.run()
assert captured[0][3]==str(ROOT/'scripts/run_regime_ex8_final_h.py')
# Numerical data fixture only: no model, real dataset or GPU touched.
import jax,jax.numpy as jnp
from create_regime_ex8_data import advance
with jax.enable_x64():
 x=jnp.asarray([[.4,.8],[-.7,.3]],dtype=jnp.float64);z=jnp.zeros((20,2,2),dtype=jnp.float32)
 np.testing.assert_array_equal(advance(x,z),advance(advance(x,z[:10]),z[10:]))
 np.testing.assert_allclose(advance(x,z),x*(1-1e-7)**20,rtol=0,atol=3e-15)
assert not (s.STUDY/'data').exists() and not (s.STUDY/'campaigns').exists()
print('PASS: static/source hashes, unchanged settings,450 total/350 reuse/100 new points,oracle-only inclusion,hard h cap,seeds/width81 counts,parent success/failure gates,no selection/capacity follow-on,worker dispatch,synthetic float64 prefix and zero-noise fixture. No training or real-data generation.')
