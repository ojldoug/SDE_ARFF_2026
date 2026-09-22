#!/usr/bin/env python3
"""Evaluate one published selected model using existing repository evaluators.
No fitting. --inspect checks bytes/metadata only. Requires explicit data/artifact,
a new output file and an identified historical or modern protocol.
"""
import argparse,hashlib,importlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'GPU')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',required=True,type=Path);p.add_argument('--dataset',required=True,type=Path);p.add_argument('--protocol',choices=['historical','modern'],required=True);p.add_argument('--method',choices=['arff','fourier','mlp_shallow','mlp_deep','joint_fourier','split_fourier','joint_mlp','split_mlp'],required=True);p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--output',type=Path);p.add_argument('--new-run',action='store_true',help='Explicitly evaluate a new native ARFF artifact after schema validation; other fresh runners evaluate at serialization');p.add_argument('--inspect',action='store_true');a=p.parse_args()
 if a.protocol=='historical' and a.split!='validation':p.error('Retained historical evidence is validation-selected, not test')
 if not a.inspect and (not a.output or a.output.exists()):p.error('Evaluation requires a new --output')
 import numpy as np
 with np.load(a.artifact,allow_pickle=False) as z:d={k:z[k] for k in z.files}
 expected=str(d.get('dataset_sha256',''))
 if expected:assert sha(a.dataset)==expected,'Dataset identity differs'
 manifest=json.loads((ROOT/'docs/independent_reproduction_v1/artifact_manifest.json').read_text())
 # The manifest provides file identity even for runners whose native artifact stores a view path.
 hashes={r['sha256'] for r in manifest['files']};assert sha(a.dataset) in hashes,'Not an inventoried dataset'
 if not a.new_run:assert sha(a.artifact) in hashes,'Not an inventoried published checkpoint'
 elif a.method!='arff' or a.protocol!='modern':p.error('--new-run is limited to modern native ARFF; other runners already evaluate before serialization')
 elif not a.inspect:
  import run_ex8_arff_validation_selected_crossfit as validator
  validator.K=int(d['fourier_frequencies']);validator.FIT_SETTINGS=dict(validator.FIT_SETTINGS,K=validator.K);validator.validate_artifact(d)
 record=dict(artifact_sha256=sha(a.artifact),dataset_sha256=sha(a.dataset),seed=int(d['seed']),method=a.method,protocol=a.protocol,split=a.split)
 if a.inspect:print(json.dumps(record,indent=2));return
 import jax,jax.numpy as jnp
 from src.experiments.dataset import load_dataset
 from src.experiments.definitions import get_experiment
 data=load_dataset(a.dataset);definition=get_experiment(str(d['experiment']));idx=d[a.split+'_idx'] if a.protocol=='historical' else getattr(data,a.split+'_idx');x,r,h=(jnp.asarray(v[idx]) for v in [data.x,data.r,data.h])
 if a.method=='arff':
  import run_ex8_arff_validation_selected_crossfit as mod
  # Historical floor differs from Ex8, as archived. No alternative floor is tested.
  from src.arff.evaluation import gaussian_nll,true_function_errors
  model=mod.reconstruct_model(d);score=gaussian_nll(model,x,r,h,spd_epsilon=float(d['spd_epsilon']));f,c=true_function_errors(model,x,true_drift=definition.drift,true_diffusion_factor=definition.diffusion_factor)
  values=dict(drift_rmse=f,covariance_rmse=c,nll=score.nll,raw_spd_violation_rate=score.spd_violation_rate,min_raw_eigenvalue=score.min_raw_eigenvalue)
 elif a.protocol=='historical':
  lib=importlib.import_module('lib.lib_Adam_FF' if a.method=='fourier' else 'lib.lib_Adam_tanh');assert sha(Path(lib.__file__))==str(d['historical_source_sha256'])
  names=['omega','amp'] if a.method=='fourier' else ['W1','W2','W3','b1','b2','b3','amp']
  def params(pre):return {k:jnp.asarray(d[pre+'_'+k]) for k in names if pre+'_'+k in d}
  from run_final_historical_adam import metrics
  values=metrics(lib,params('drift'),params('covariance'),x,r,h,definition)
 else:
  from src.adam import fourier as ff,mlp as ml
  mod=ml if 'mlp' in a.method else ff
  def params(pre):
   if mod is ml:return ml.MLPParams(tuple(jnp.asarray(d[f'{pre}_weight_{i}']) for i in range(3)),tuple(jnp.asarray(d[f'{pre}_bias_{i}']) for i in range(3)))
   return ff.FourierParams(jnp.asarray(d[pre+'_omega']),jnp.asarray(d[pre+'_amp']))
  model=(ml.AdamMLPModel if mod is ml else ff.AdamFourierModel)(params('drift'),params('covariance'),str(d['diff_type']))
  if a.method.startswith('joint_'):
   native=importlib.import_module('run_adam_mlp_experiment' if mod is ml else 'run_adam_fourier_experiment')
   f,c=native.true_function_errors(model,x,true_drift=definition.drift,true_diffusion_factor=definition.diffusion_factor)
   values=dict(drift_rmse=f,covariance_rmse=c,nll=mod.gaussian_nll(model,x,r,h),min_covariance_eig=np.linalg.eigvalsh(np.asarray(mod.predict_covariance(model,x))).min())
  elif mod is ml:
   from run_adam_split_mlp_experiment import evaluate_model
   values=evaluate_model(model,x,r,h,definition)
  else:
   from run_adam_split_fourier_experiment import evaluate_split
   values=evaluate_split(model,x,r,h,true_drift=definition.drift,true_diffusion_factor=definition.diffusion_factor)
 values={k:float(v) for k,v in values.items()};record.update(metrics=values,backend=jax.default_backend(),jax=jax.__version__,comparison={})
 for k,v in values.items():
  key=a.split+'_'+k
  if key in d:record['comparison'][k]=dict(archived=float(d[key]),difference=v-float(d[key]),within_predeclared_tolerance=bool(np.isclose(v,float(d[key]),rtol=1e-5,atol=1e-6)))
 record['comparison_scope']='Same-source/backend target rtol=1e-5 atol=1e-6; a cross-backend failure is reported, never silently accepted or refitted.'
 with a.output.open('x') as f:json.dump(record,f,indent=2,allow_nan=False)
 print(json.dumps(record,indent=2))
 if not all(v['within_predeclared_tolerance'] for v in record['comparison'].values()):raise SystemExit('Reconstruction mismatch retained in output; stop, do not relax tolerances')
if __name__=='__main__':main()
