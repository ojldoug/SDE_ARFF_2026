#!/usr/bin/env python3
"""Analyze saved diagnostic arrays; no JAX execution, adaptation or fits."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--verification',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compare(x,y):
 d=np.asarray(y,dtype=float)-np.asarray(x,dtype=float)
 return dict(exact=bool(np.array_equal(x,y)),within_original_tolerance=bool(np.allclose(x,y,rtol=1e-5,atol=1e-6)),max_abs=float(np.max(np.abs(d))),rms=float(np.sqrt(np.mean(d*d))))
ref=a.verification/'checkout/results/controlled_study_2026/float64_v2/production/baseline/K_128_N_80000/arff/seed_0/artifact.npz';new=a.verification/'run/artifact.npz';old=dict(np.load(ref));fresh=dict(np.load(new));records={};arrays={}
for env in ['isolated','current_research']:
 folder=a.workspace/env;records[env]=json.loads((folder/'record.json').read_text());arrays[env]=dict(np.load(folder/'arrays.npz'))
stages={}
for stage in ['fold_0','fold_1','fold_2','fold_3','fold_4','final_drift','covariance']:
 x,y=old[stage+'_validation_mse'],fresh[stage+'_validation_mse'];idx=np.flatnonzero(x!=y);out=np.flatnonzero(~np.isclose(x,y,rtol=1e-5,atol=1e-6))
 stages[stage]=dict(first_unequal_recorded_loss_iteration=int(idx[0])+1 if len(idx) else None,first_loss_outside_tolerance=int(out[0])+1 if len(out) else None,selected_iterations=[int(old[stage+'_best_iteration']),int(fresh[stage+'_best_iteration'])],omega=compare(old[stage+'_omega'],fresh[stage+'_omega']),amplitudes=compare(old[stage+'_amp'],fresh[stage+'_amp']))
fixed={}
for env,z in arrays.items():
 fixed[env]=dict(repeated_amplitudes=compare(z['amp_0'],z['amp_1']),repeated_predictions=compare(z['pred_0'],z['pred_1']),vs_archived_amp=compare(old['final_drift_amp'],z['amp_0']),vs_new_amp=compare(fresh['final_drift_amp'],z['amp_0']),vs_archived_prediction=compare(z['pred_archived'],z['pred_0']),residuals=records[env]['residuals'])
lib=lambda r:{Path(x['path']).name:x['sha256'] for x in r['loaded_libraries']}
l0,l1=map(lib,[records['isolated'],records['current_research']]);cross={k:compare(arrays['isolated'][k],arrays['current_research'][k]) for k in arrays['isolated']}
envkeys=['python_version','platform','hostname','jax_version','numpy_version','package_versions_json','jax_default_matmul_precision','devices_json','environment_json','training_dtype']
result=dict(rtol=1e-5,atol=1e-6,solve_calls=sum(r['solve_calls'] for r in records.values()),historical_vs_new_environment={k:dict(archived=str(old[k]),new=str(fresh[k])) for k in envkeys},stages=stages,oof_targets=compare(old['crossfit_covariance_targets'],fresh['crossfit_covariance_targets']),fixed_basis=fixed,cross_environment=cross,input_hashes={k:r['input_hashes'] for k,r in records.items()},current_loaded_libraries={k:dict(isolated=l0.get(k),research=l1.get(k),match=l0.get(k)==l1.get(k)) for k in sorted(l0.keys()|l1.keys())},package_differences={k:[records['isolated']['packages'].get(k),records['current_research']['packages'].get(k)] for k in records['isolated']['packages'].keys()|records['current_research']['packages'].keys() if records['isolated']['packages'].get(k)!=records['current_research']['packages'].get(k)},source_sha256={k:r['source_sha256'] for k,r in records.items()},file_hashes={str(p):sha(p) for p in [ref,new,*[a.workspace/e/'arrays.npz' for e in records]]})
A=arrays['isolated']['regularized_gram'].astype(float);rhs=arrays['isolated']['rhs'].astype(float);beta=fresh['final_drift_amp'].astype(float);res=A@beta-rhs
result['new_saved_amplitude_residual']=dict(absolute_frobenius=float(np.linalg.norm(res)),relative_rhs=float(np.linalg.norm(res)/np.linalg.norm(rhs)),normwise_backward_error=float(np.linalg.norm(res)/(np.linalg.norm(A)*np.linalg.norm(beta)+np.linalg.norm(rhs))))
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({k:result[k] for k in ['solve_calls','fixed_basis','cross_environment','package_differences']},indent=2))
