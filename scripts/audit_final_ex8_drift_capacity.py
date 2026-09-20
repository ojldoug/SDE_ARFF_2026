"""Read-only model/provenance audit; writes new audit outputs, never fits."""
import os
os.environ.update(JAX_PLATFORMS='cpu',JAX_ENABLE_X64='false',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:]);os.nice(10)
from pathlib import Path
import sys,json,csv,hashlib,subprocess
import numpy as np
import jax,jax.numpy as jnp
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.experiments.dataset import load_dataset
from src.experiments.definitions import ex8_drift
from src.arff.regression import ARFFModel,predict
from src.arff.validation_selected import _split_indices
BASE=ROOT/'results/capacity_regime_ex8_v2'
JSON=BASE/'ARFF_DRIFT_K_CONSISTENCY_AUDIT.json'
MD=JSON.with_suffix('.md')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while z:=f.read(4*1024*1024):h.update(z)
 return h.hexdigest()
def ah(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
def main():
 assert not JSON.exists() and not MD.exists()
 plan=json.loads((BASE/'campaigns/capacity/plan.json').read_text())
 manifest=json.loads((BASE/'manifest.json').read_text());protected={}
 for p,h in manifest['frozen_source_sha256'].items():assert sha(ROOT/p)==h;protected[str(ROOT/p)]=h
 for f in [BASE/'manifest.json',BASE/'selection.json',BASE/'capacity_summary/summary.json',BASE/'capacity_summary/per_seed_metrics.csv',BASE/'campaigns/capacity/plan.json']:
  protected[str(f)]=sha(f)
 dataset=BASE/'dataset_roots/N_640000/h_0.0001/data/ex8.npz';d=load_dataset(dataset);dh=sha(dataset);protected[str(dataset)]=dh
 assert d.x.shape==d.r.shape==(660000,2) and d.h.shape==(660000,1)
 assert np.all(d.h==np.float32(.0001)) and d.r.dtype==np.float64
 x=jnp.asarray(d.x[d.test_idx]);truth=jnp.asarray(ex8_drift(x));np.testing.assert_array_equal(truth,-d.x[d.test_idx])
 ids={k:ah(getattr(d,k)) for k in ['train_idx','validation_idx','test_idx']}
 summary_rows=list(csv.DictReader((BASE/'capacity_summary/per_seed_metrics.csv').open()))
 records=[];reuse=[]
 for K in [64,128,256,512,1024]:
  for seed in range(10):
   folder=BASE/f'production/regime_capacity/h_0.0001/K_{K}_N_640000/arff/seed_{seed}'
   artifact=folder/'artifact.npz';done=json.loads((folder/'complete.json').read_text())
   digest=sha(artifact);assert digest==done['artifact_sha256'];protected[str(artifact)]=digest
   with np.load(artifact,allow_pickle=False) as z:a={k:z[k] for k in z.files}
   assert int(a['seed'])==seed and int(a['K'])==K and int(a['fourier_frequencies'])==K
   assert str(a['dataset_sha256'])==dh and sha(str(a['dataset_path']))==dh
   for field in ids:np.testing.assert_array_equal(a[field],getattr(d,field))
   assert [int(a[k]) for k in ['n_train','n_validation','n_test']]==[640000,10000,10000]
   for key,expected in [('lambda_reg',.001),('delta',.2),('gamma',1),('M_min',300),('M_max',300),('metropolis_test',True),('resampling',False)]:assert a[key].item()==expected
   cfg=json.loads(str(a['config_json']));assert cfg['data']['observation_lag']==.0001
   stage_counts={}
   for stage in [*(f'fold_{i}' for i in range(5)),'final_drift','covariance']:
    fit=a[stage+'_fit_train_positions'];val=a[stage+'_internal_validation_train_positions']
    inputs=np.flatnonzero(a['fold_id']!=int(stage[5:])) if stage.startswith('fold_') else np.arange(640000)
    ff,vv=_split_indices(len(inputs),validation_fraction=.1,seed=int(a[stage+'_validation_seed']))
    np.testing.assert_array_equal(fit,inputs[ff]);np.testing.assert_array_equal(val,inputs[vv])
    stage_counts[stage]=dict(fit=len(fit),internal_validation=len(val),fit_positions_sha256=ah(fit),internal_validation_positions_sha256=ah(val))
   models={}
   for name,q in [('drift',2),('covariance',3)]:
    omega,amp=a[name+'_omega'],a[name+'_amp'];assert omega.shape==(2,K) and amp.shape==(2*K,q)
    np.testing.assert_array_equal(omega,a[('final_drift' if name=='drift' else name)+'_omega'])
    np.testing.assert_array_equal(amp,a[('final_drift' if name=='drift' else name)+'_amp'])
    models[name]=dict(omega_sha256=ah(omega),amp_sha256=ah(amp),omega_shape=list(omega.shape),amp_shape=list(amp.shape),zero_frequency_columns=int(np.sum(np.all(omega==0,axis=0))),zero_amplitude_rows=int(np.sum(np.all(amp==0,axis=1))),zero_amplitude_scalars=int(np.sum(amp==0)))
   hist=a['final_drift_validation_mse'];best=int(a['final_drift_best_iteration']);assert len(hist)==300 and best==int(np.argmin(hist))+1
   assert float(hist[best-1])==float(a['final_drift_best_validation_mse'])
   sr=next(v for v in summary_rows if v['method']=='arff' and int(v['K'])==K and int(v['seed'])==seed)
   assert float(sr['test_drift_rmse'])==float(a['test_drift_rmse'])
   record=dict(K=K,seed=seed,path=str(artifact.relative_to(ROOT)),artifact_sha256=digest,dataset_recorded_path=str(a['dataset_path']),dataset_sha256=dh,selected_iteration=best,selected_validation_mse=float(hist[best-1]),initial_validation_mse=float(hist[0]),last_validation_mse=float(hist[-1]),train_coefficient_rmse=float(a['train_drift_rmse']),test_drift_rmse=float(a['test_drift_rmse']),models=models,stage_counts=stage_counts,reused=bool(done.get('reused')))
   if K==128:
    assert not done.get('reused')
    ctx=json.loads((folder/'artifact.context.json').read_text());assert ctx['job']['N']==640000 and ctx['job']['K']==128 and ctx['dataset_view']['dataset_sha256']==dh
   if K==1024:
    peers=[ROOT/f'results/{study}/production/regime_h/h_0.0001/K_1024_N_640000/arff/seed_{seed}/artifact.npz' for study in ['capacity_regime_ex8_v2','capacity_regime_ex8_final_h_v1']]
    assert done['reused']
    for peer in peers:assert sha(peer)==digest
    reuse.append(dict(seed=seed,artifact_sha256=digest,models=models,capacity=str(artifact),lag_paths=list(map(str,peers)),all_metrics_bit_identical=True,source=done['reuse_source']['artifact']))
   if seed==0:
    omega=jnp.asarray(a['drift_omega']);amp=jnp.asarray(a['drift_amp']);normal=predict(ARFFModel(omega,amp),x)
    # Independent explicit cosine/sine reconstruction; no repository feature helper.
    proj=x@omega
    independent=jnp.cos(proj)@amp[:K]+jnp.sin(proj)@amp[K:]
    concat=jnp.concatenate([jnp.cos(proj),jnp.sin(proj)],axis=1)@amp
    np.testing.assert_array_equal(concat,normal)
    native_rmse=float(jnp.sqrt(jnp.mean((normal-truth)**2)))
    rmse=float(jnp.sqrt(jnp.mean((independent-truth)**2)))
    # Float64 NumPy is a precision sensitivity check, not a model conversion.
    projected=d.x[d.test_idx].astype(float)@a['drift_omega'].astype(float)
    numpy_prediction=np.cos(projected)@a['drift_amp'][:K].astype(float)+np.sin(projected)@a['drift_amp'][K:].astype(float)
    record['reconstruction']=dict(seed=seed,independent_vs_native_max_absolute=float(jnp.max(jnp.abs(independent-normal))),concat_vs_native_max_absolute=0.,independent_rmse=rmse,native_cpu_rmse=native_rmse,saved_gpu_rmse=float(a['test_drift_rmse']),cpu_minus_saved=native_rmse-float(a['test_drift_rmse']),numpy64_rmse=float(np.sqrt(np.mean((numpy_prediction+d.x[d.test_idx])**2))),numpy64_vs_native_max_absolute=float(np.max(np.abs(numpy_prediction-np.asarray(normal)))))
   records.append(record)
  print('Audited all10 ARFF seeds at K',K,flush=True)
 for p,h in protected.items():assert sha(p)==h,p
 out=dict(classification='pending_backend_precision_confirmation',dataset=dict(path=str(dataset),resolved_path=str(dataset.resolve()),sha256=dh,step_sizes_unique=[float(v) for v in np.unique(d.h)],dtype=str(d.h.dtype),train=640000,validation=10000,test=10000,index_sha256=ids,test_states_sha256=ah(d.x[d.test_idx])),records=records,reuse_checks=reuse,protected_sha256=protected,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),rmse_definition='sqrt(mean over Ntest and2 coordinates); vector-norm definition divided by sqrt(2)',truth='instantaneous f(x)=-x; no h in coefficient evaluator',target='float32(r)/float32(h); r archive is float64, model arithmetic unchanged float32; no state or response standardization',source_paths=['scripts/run_controlled_ex8.py::run_job','scripts/run_ex8_arff_validation_selected_crossfit.py::main/learn/evaluate_split','src/arff/validation_selected.py::fit_validation_selected_arff','src/arff/regression.py::fourier_features/predict','src/arff/evaluation.py::true_function_errors'],training_target_mse_saved=False,conditioning_or_rank_diagnostics_saved=False)
 with (BASE/'ARFF_DRIFT_K_AUDIT_CPU_EVIDENCE.json').open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps([r['reconstruction']|{'K':r['K']} for r in records if 'reconstruction' in r],indent=2),flush=True)
if __name__=='__main__':main()
