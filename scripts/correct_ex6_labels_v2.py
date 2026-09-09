#!/usr/bin/env python3
"""Verify original ex6 trajectory replay and publish exclusively a label-only v2."""
from pathlib import Path
import sys,json,hashlib,subprocess
from unittest.mock import patch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiments import wave_data as wave
OUT=ROOT/'results/final_reproduction'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(a):return dict(max_abs=float(np.max(abs(a))),rmse=float(np.sqrt(np.mean(a*a))))
def main():
 original=ROOT/'data/ex6.npz';target=ROOT/'data/ex6_labels_v2.npz'
 meta_path=target.with_suffix('.json');change_path=ROOT/'data/ex6_labels_v2_changes.npz'
 report_path=OUT/'evidence/ex6_v2_validation.json'
 for p in (target,meta_path,change_path,report_path):
  if p.exists():raise FileExistsError(p)
 meta=json.loads(original.with_suffix('.json').read_text());oldsha=sha(original)
 assert oldsha==meta['dataset_sha256']
 source=ROOT/'src/experiments/wave_data.py'
 oldcode=subprocess.check_output(['git','show',meta['git_commit']+':src/experiments/wave_data.py'],cwd=ROOT)
 assert oldcode==source.read_bytes(),'Generator differs from source recorded at data generation'
 with np.load(original,allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
 delta=.001;space=np.arange(1000)*delta;time=np.arange(2000)*delta
 f=lambda x,t,u:5.*np.sin(4.*np.pi*x)
 g=lambda x,t,u:.05*(1.+np.exp(-150.*(x-.5)**2))
 u0=np.exp(-150.*(space-.5)**2)/20.;v0=-2.*np.gradient(u0,delta)
 factory=np.random.default_rng
 class Capture:
  def __init__(self,zero):self.rng=factory(1);self.zero=zero;self.draws=[]
  def normal(self,loc,scale,size):
   w=np.zeros(size) if self.zero else self.rng.normal(loc=loc,scale=scale,size=size)
   self.draws.append(w.copy());return w
 def replay(zero):
  rng=Capture(zero)
  with patch.object(wave.np.random,'default_rng',return_value=rng):
   u=wave.integrate_stochastic_wave(u0,v0,f,g,time,space,seed=1,periodic_boundary=False)
  start,x,end=wave.split_wave_learning_data(u,space[::2],time)
  return u,x,end-start,np.stack(rng.draws)[:,1:-1].reshape(-1,1)
 trajectory,x,r,w=replay(False)
 np.testing.assert_array_equal(x,a['x_data']);np.testing.assert_array_equal(r,a['r_data'])
 n=len(r);row=np.arange(n);block=row//498;j=block+1
 lag_rows=row[block==0];spatial_rows=row[j%2==0]
 assert len(lag_rows)==498 and len(spatial_rows)==497502 and n==995004
 b={k:v.copy() for k,v in a.items()}
 b['step_sizes'][lag_rows,0]=delta**2/4
 b['x_data'][spatial_rows,1]=np.tile(space[1::2][1:-1],999)
 for key in ('r_data','train_idx','validation_idx','test_idx'):np.testing.assert_array_equal(a[key],b[key])
 np.testing.assert_array_equal(a['x_data'][:,0],b['x_data'][:,0])
 def identity_error(arr,realized_r,draw):
  x=arr['x_data'][:,1:2];h=arr['step_sizes']
  return realized_r/h-f(x,0,0)-g(x,0,0)*draw/(4*h)
 raw=identity_error(a,r,w);fixed=identity_error(b,r,w)
 _,zx,zr,zw=replay(True);np.testing.assert_array_equal(zx,a['x_data'])
 zraw=identity_error(a,zr,zw);zfixed=identity_error(b,zr,zw)
 assert np.max(abs(fixed))<1e-7 and np.max(abs(zfixed))<1e-7
 # Exact variance identity: Var[(g/4) W]=h*(g/2)^2.
 varw=np.where(block[:,None]==0,delta**2,2*delta**2)
 ratio_old=varw/(4*a['step_sizes']);ratio_new=varw/(4*b['step_sizes'])
 assert np.array_equal(ratio_new,np.ones_like(ratio_new))
 report=dict(original_sha256=oldsha,source_revision=meta['git_commit'],source_sha256=sha(source),
  replay=dict(x_bitwise_identical=True,r_bitwise_identical=True,trajectory_sha256=hashlib.sha256(trajectory.tobytes()).hexdigest()),
  changed_rows=dict(step_sizes=498,x_data_space=497502,x_data_time=0,r_data=0,split_indices=0),
  stochastic_noise_subtracted_drift_identity=dict(before=stats(raw),after=stats(fixed)),
  zero_noise_actual_grid_and_forcing=dict(before=stats(zraw),after=stats(zfixed)),
  initialization_noise_variance_ratio=dict(before=float(ratio_old[0,0]),after=float(ratio_new[0,0])),
  tolerance_drift_units=1e-7,
  identity='r = h*f(x) + (g(x)/4)*W; h0=Delta^2/4, h_interior=Delta^2/2; Var(W0)=Delta^2, Var(Wj)=2*Delta^2; sigma_eff=g/2',
  scientific_choice='None for autonomous coefficient-learning labels under the existing generator. Time column is retained as the original buffer-index covariate, not reinterpreted as physical observation time. No claim of repairing/redefining the historical SPDE spatial stencil.')
 changes=dict(step_sizes_row=lag_rows,step_sizes_column=np.zeros(len(lag_rows),dtype=int),step_sizes_old=a['step_sizes'][lag_rows,0],step_sizes_new=b['step_sizes'][lag_rows,0],
  x_data_row=spatial_rows,x_data_column=np.ones(len(spatial_rows),dtype=int),x_data_old=a['x_data'][spatial_rows,1],x_data_new=b['x_data'][spatial_rows,1])
 with change_path.open('xb') as handle:np.savez_compressed(handle,**changes)
 with target.open('xb') as handle:np.savez(handle,**b)
 assert sha(original)==oldsha
 with np.load(target,allow_pickle=False) as z:
  for key in b:np.testing.assert_array_equal(z[key],b[key])
 report['corrected_sha256']=sha(target);report['change_record_sha256']=sha(change_path)
 newmeta=dict(meta,dataset_version='ex6_labels_v2',dataset_sha256=report['corrected_sha256'],parent_dataset=str(original),parent_dataset_sha256=oldsha,
  correction='historical dataset -> confirmed labeling defect -> blocker-required label correction -> corrected canonical dataset',
  correction_script_sha256=sha(Path(__file__)),changed_fields_record=str(change_path),changed_fields_record_sha256=sha(change_path),
  h_convention='first 498 rows 2.5e-7; remaining rows 5e-7; inherited data_config.observation_lag describes interior lag',
  original_data_config=meta['data_config'],data_config_scope='Inherited generator configuration; per-row corrected lags are in step_sizes.',
  validation=report)
 with meta_path.open('x') as f:json.dump(newmeta,f,indent=2)
 with report_path.open('x') as f:json.dump(report,f,indent=2)
 print(json.dumps(report,indent=2))
if __name__=='__main__':main()
