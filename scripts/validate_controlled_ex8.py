"""CPU archival integrity checks, never an accuracy acceptance threshold."""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
from pathlib import Path
import sys,json,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import run_controlled_ex8 as runner

def validate(job,path,log,context=None):
 with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 method=job['method'];K=job['K'];N=job['N'];cal=job.get('study')=='calibration'
 assert int(a['seed'])==job['seed'] and str(a['experiment'])=='ex8'
 if method.endswith('mlp'):assert int(a['fourier_parameter_count'])==14*K
 else:assert int(a['fourier_frequencies'])==K
 for k,v in a.items():
  assert not v.dtype.hasobject,k
  if np.issubdtype(v.dtype,np.number):
   if method=='arff' and k.endswith(('_validation_mse','_moving_average')) and v.ndim==1:assert not np.any(np.isnan(v)|np.isneginf(v)),k
   else:assert np.all(np.isfinite(v)),k
 assert float(a['algorithm_time'])>0 and float(a['compilation_time'])>=0
 assert np.isclose(float(a['end_to_end_time']),float(a['algorithm_time'])+float(a['compilation_time']))
 if method.endswith('mlp'):
  w=min(range(1,300),key=lambda w:abs(2*w*w+13*w+5-14*K));count=0
  for prefix,q in [('drift',2),('covariance',3)]:
   for i,shape in enumerate([(2,w),(w,w),(w,q)]):
    assert a[f'{prefix}_weight_{i}'].shape==shape and a[f'{prefix}_bias_{i}'].shape==(shape[1],)
    count+=a[f'{prefix}_weight_{i}'].size+a[f'{prefix}_bias_{i}'].size
  assert count==int(a['mlp_parameter_count'])==2*w*w+13*w+5
 else:
  for prefix,q in [('drift',2),('covariance',3)]:assert a[prefix+'_omega'].shape==(2,K) and a[prefix+'_amp'].shape==(2*K,q)
  count=14*K
 if method.startswith('joint'):
  assert int(a['epochs'])==300
  stages=[('','validation_nll','best_epoch','best_validation_nll','cumulative_time')]
 elif method.startswith('split'):
  assert int(a['epochs_per_regression'])==300
  stages=[('','final_drift_validation_mse','final_drift_best_epoch','final_drift_best_validation_mse','final_drift_cumulative_time'),('','covariance_validation_nll','covariance_best_epoch','covariance_best_validation_nll','covariance_cumulative_time')]
 else:
  stages=[('',f'{stage}_validation_mse',f'{stage}_best_iteration',f'{stage}_best_validation_mse',f'{stage}_cumulative_time') for stage in [f'fold_{i}' for i in range(5)]+['final_drift','covariance']]
  for k,v in [('M_min',300),('M_max',300),('lambda_reg',job.get('lambda_reg',.001)),('delta',job.get('delta',.2)),('gamma',1),('resampling',job.get('resampling',False)),('metropolis_test',job.get('metropolis_test',True))]:assert a[k].item()==v,k
 for _,hist,best,val,clock in stages:
  assert a[hist].shape==a[clock].shape==(300,)
  i=int(np.argmin(a[hist]));assert int(a[best])==i+(1 if method=='arff' else 0)
  assert np.isfinite(a[hist][i]) and np.isclose(float(a[val]),a[hist][i])
  assert np.all(np.diff(a[clock])>=0) and a[clock][0]>=0
 if not method.startswith('joint'):
  from src.arff.two_stage import make_folds
  f=np.full(N,-1,dtype=int)
  for i,idx in enumerate(make_folds(N,5,2026)):f[idx]=i
  np.testing.assert_array_equal(a['fold_id'],f)
  assert int(a['n_train'])==N and int(a['n_validation'])==10000 and int(a['n_test'])==10000
 if method!='arff':assert int(a['batch_size'])==256 and float(a['learning_rate'])==(.001 if method.endswith('mlp') else .0001)
 if cal:
  assert str(a['method'])=='controlled_arff_calibration' and not bool(a['test_evaluated'])
  assert 'test_nll' not in a
 else:
  assert 'backend    : gpu' in log.read_text()
  if method.startswith('joint'):
   assert re.search(r'train N\s*:\s*'+str(N)+r'\b',log.read_text())
   import summarize_ex8_capacity_matched as old
   old.parse_test_log(log)
  else:
   for key in ['test_nll','test_drift_rmse','test_covariance_rmse']:assert key in a
 if context is not None:
  c=json.loads(Path(context).read_text());assert c['job']==job
  expected=json.loads((runner.STUDY/job.get('dataset_root',f'dataset_roots/N_{N}')/'dataset_view.json').read_text())
  assert c['dataset_view']==expected
 return dict(artifact_sha256=runner.digest(path),log_sha256=runner.digest(log),algorithm_time=float(a['algorithm_time']),parameter_count=count)
