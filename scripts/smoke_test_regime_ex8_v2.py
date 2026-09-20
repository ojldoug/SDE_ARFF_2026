"""No fitting: amendment plans, selection identity, dispatch and data arithmetic."""
import os
os.environ.update(JAX_PLATFORMS='cpu',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
from pathlib import Path
import sys,ast,json
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import run_regime_ex8_v2_campaign as s
import run_regime_ex8_v2_pipeline as pipeline
from src.experiments.mlp_size import matched_two_layer_width
for p in [*ROOT.glob('scripts/*regime_ex8_v2*.py')]:ast.parse(p.read_text())
pipeline.verify()
m=s.r.adapter.verify_registration();old=json.loads((s.PARENT/'manifest.json').read_text())
for k in ['settings','stage1_seeds','stage2_seeds','methods','dataset','N_grid','execution','frozen_source_sha256']:assert m[k]==old[k],k
for k in ['N_selection','stability_rule','h_selection','h_covariance_caveat','N_h_interaction']:assert m['diagnostic'][k]==old['diagnostic'][k]
assert len(s.plans('N'))==200 and len(s.plans('h'))==350
actual=Path.read_text
def read(self,*args,**kwargs):
 if self==s.STUDY/'selection.json':return json.dumps(dict(N_star=320000,h_star=.001))
 return actual(self,*args,**kwargs)
with patch.object(Path,'read_text',read):jobs=s.plans('capacity')
assert len(jobs)==250 and sorted(set(j['K'] for j in jobs))==[64,128,256,512,1024]
assert all(j['N']==320000 and j['h']==.001 for j in jobs)
for K in [64,128,256,512,1024]:
 assert {j['seed'] for j in jobs if j['K']==K}==set(range(10))
 w,mp,fp=matched_two_layer_width(state_dimension=2,covariance_output_dimension=3,n_frequencies=K)
 expected=next(v for v in m['capacity'] if v['K']==K)
 assert (w,mp,fp)==(expected['mlp_width'],expected['mlp_parameters'],expected['fourier_parameters'])
 for method in m['methods']:
  cfg=s.r.adapter.configuration(dict(K=K,N=320000,h=.001,method=method))
  assert cfg.fourier_frequencies==K and cfg.data.em_substeps==10000
before=(ROOT/'scripts/analyze_regime_ex8.py').read_text();after=(ROOT/'scripts/analyze_regime_ex8_v2.py').read_text()
assert before.split('def equivalent(')[1].split(' # Materialize')[0]==after.split('def equivalent(')[1].split(' # Materialize')[0], 'Selection implementation changed'
# Prove the supervisor changes only the native process entrypoint.
captured=[]
def fake(command,*a,**kw):captured.append(command.copy())
def fake_run(phase):
 s.c.subprocess.Popen([s.c.PYTHON,'-B','-u',str(ROOT/'scripts/run_controlled_ex8_float64_v2.py'),'--job-json','fixture.json','--artifact-path','fixture.npz'])
with patch.object(s.c.subprocess,'Popen',fake),patch.object(s.c,'run',fake_run):s.run('capacity')
assert captured[0][3]==str(ROOT/'scripts/run_regime_ex8_v2.py') and captured[0][4:]==['--job-json','fixture.json','--artifact-path','fixture.npz']
# Synthetic integration check only, no study rows or trained model evaluated.
import jax,jax.numpy as jnp
import create_regime_ex8_data as b
with jax.enable_x64():
 x=jnp.asarray([[.4,.8],[-.7,.3]],dtype=jnp.float64);z=jnp.zeros((12,2,2),dtype=jnp.float32)
 whole=b.advance(x,z);segmented=b.advance(b.advance(x,z[:5]),z[5:])
 np.testing.assert_array_equal(whole,segmented)
 np.testing.assert_allclose(whole,x*(1-1e-7)**12,rtol=0,atol=2e-15)
assert not (s.STUDY/'selection.json').exists() and not (s.STUDY/'campaigns').exists()
print('PASS: static/registration; unchanged settings and exact selection code; 200/350/250 job plans; K128 width27/1814; all capacity counts; frozen-regime propagation; worker command dispatch; synthetic float64 prefix/zero-noise fixture. No training or data generation dispatched.')
