#!/usr/bin/env python3
"""Descriptive comparison at preregistered tolerances; never fits or selects."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--new',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 old=dict(np.load(a.reference,allow_pickle=False));new=dict(np.load(a.new,allow_pickle=False));rows={}
 exact=['seed','fourier_frequencies','n_train','n_validation','n_test','dataset_sha256','source_sha256_json','config_json','fold_id','train_idx','validation_idx','test_idx','final_prng_key']
 exact += [k for k in old if k.endswith(('_fit_train_positions','_internal_validation_train_positions','_validation_seed'))]
 identity={k:bool(np.array_equal(old[k],new[k])) for k in exact}
 for k in old:
  if k.endswith(('_validation_mse','_moving_average','_best_iteration','_best_validation_mse','_rmse','_nll','_spd_violation_rate','_min_eigenvalue','_min_raw_eigenvalue')):
   x,y=old[k],new[k]
   if not np.issubdtype(x.dtype,np.number):continue
   same=np.isclose(x,y,rtol=1e-5,atol=1e-6,equal_nan=False)
   finite=np.isfinite(x)&np.isfinite(y)
   row=dict(within_predeclared_tolerance=bool(same.all()),number_outside=int((~same).sum()),shape=list(x.shape),max_abs_difference=float(np.max(np.abs(y[finite]-x[finite]))) if np.any(finite) else None)
   if x.ndim==0:row.update(reference=x.item(),new=y.item(),difference=(y-x).item())
   rows[k]=row
 models={}
 for k in old:
  if k.endswith(('_omega','_amp')) or k=='crossfit_covariance_targets':
   x,y=old[k],new[k]; models[k]=dict(exact=bool(np.array_equal(x,y)),max_abs_difference=float(np.max(np.abs(y-x))),rms_difference=float(np.sqrt(np.mean((y.astype(float)-x.astype(float))**2))))
 result=dict(model_differences=models,rtol=1e-5,atol=1e-6,reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),new_sha256=hashlib.sha256(a.new.read_bytes()).hexdigest(),schema_equal=set(old)==set(new),exact_identity=identity,comparisons=rows,timings={k:dict(reference=float(old[k]),new=float(new[k])) for k in ['algorithm_time','compilation_time','end_to_end_time']})
 result['all_compared_values_within_tolerance']=all(r['within_predeclared_tolerance'] for r in rows.values());result['all_identity_checks_pass']=all(identity.values())
 with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({k:v for k,v in result.items() if k not in ['comparisons','exact_identity']},indent=2))
if __name__=='__main__':main()
