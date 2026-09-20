"""Exact discrete drift and covariance trace; no closed full-matrix oracle assumed."""
from pathlib import Path
import json,numpy as np,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiments.dataset import load_dataset
OUT=ROOT/'results/capacity_regime_ex8_v1'
def oracle(h):
 dt=1e-7;m=round(h/dt);logA=np.log1p(-dt)
 drift_coefficient=np.expm1(m*logA)/h
 geometric=np.expm1(2*m*logA)/np.expm1(2*logA)
 covariance_trace=dt/h*geometric*(1.+1e-8)
 return dict(h=h,fine_steps=m,discrete_drift_coefficient=float(drift_coefficient),continuous_drift_coefficient=float(np.expm1(-h)/h),relative_discrete_drift_bias=float(abs(drift_coefficient+1)),oracle_effective_covariance_trace=float(covariance_trace),instantaneous_covariance_trace=1.+1e-8,oracle_drift_target_noise_rms_per_coordinate=float(np.sqrt(covariance_trace/(2*h))))
def main():
 manifest=json.loads((OUT/'manifest.json').read_text());records=[]
 for h in manifest['h_grid']:
  d=load_dataset(OUT/f'dataset_roots/N_640000/h_{h:.8g}/data/ex8.npz');v=oracle(h)
  # Conditional mean A^m x and trace(Cov(r|x))/h are known exactly even
  # though the conditional covariance orientation is state dependent.
  idx=d.train_idx;residual=d.r[idx]-h*v['discrete_drift_coefficient']*d.x[idx].astype(np.float64)
  squared=np.sum(residual**2,axis=1)/h;se=squared.std(ddof=1)/np.sqrt(len(idx))
  v.update(training_empirical_covariance_trace=float(squared.mean()),training_trace_standard_error=float(se),validation_drift_oracle_rmse=float(abs(v['discrete_drift_coefficient']+1)*np.sqrt(np.mean(d.x[d.validation_idx].astype(float)**2))))
  assert abs(squared.mean()-v['oracle_effective_covariance_trace'])<max(6*se,1e-4),v
  records.append(v)
 with (OUT/'oracle_moments.json').open('x') as f:json.dump(dict(status='passed',records=records,identity='A=1-dt; E[r|x]=(A^m-1)x; tr Cov(r|x)=dt*sum(A^(2j),j=0..m-1)*(1+1e-8). Follows from linear drift, independent centered increments and constant tr(Sigma(x)).',limitation='Trace identity does not determine covariance orientation, Frobenius error, or log determinant. No full covariance finite-lag oracle or covariance bias bound is claimed.'),f,indent=2)
 print(json.dumps(records,indent=2))
if __name__=='__main__':main()
