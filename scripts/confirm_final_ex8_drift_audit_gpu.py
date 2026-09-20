"""Saved-model evaluation only: confirm GPU precision, then finalize the audit."""
import os
os.environ.update(JAX_PLATFORMS='cuda',JAX_ENABLE_X64='false',CUDA_VISIBLE_DEVICES='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',XLA_PYTHON_CLIENT_PREALLOCATE='false')
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from run_ex8_split_mlp_w27 import check_assigned_gpu
check_assigned_gpu(allow_cpu=False)
import jax,jax.numpy as jnp
from src.arff.regression import predict,ARFFModel
from src.experiments.dataset import load_dataset
from src.experiments.definitions import ex8_drift
BASE=ROOT/'results/capacity_regime_ex8_v2'
s=json.loads((BASE/'ARFF_DRIFT_K_AUDIT_CPU_EVIDENCE.json').read_text())
d=load_dataset(s['dataset']['path']);x=jnp.asarray(d.x[d.test_idx]);truth=ex8_drift(x)
for rec in s['records']:
 if 'reconstruction' not in rec:continue
 with np.load(ROOT/rec['path'],allow_pickle=False) as z:w=jnp.asarray(z['drift_omega']);a=jnp.asarray(z['drift_amp'])
 K=rec['K'];normal=predict(ARFFModel(w,a),x);angle=x@w
 independent=jnp.cos(angle)@a[:K]+jnp.sin(angle)@a[K:]
 concat=jnp.concatenate([jnp.cos(angle),jnp.sin(angle)],axis=-1)@a
 np.testing.assert_array_equal(concat,normal)
 value=float(jnp.sqrt(jnp.mean((normal-truth)**2)))
 split_value=float(jnp.sqrt(jnp.mean((independent-truth)**2)))
 np.testing.assert_allclose(value,rec['test_drift_rmse'],rtol=1e-5,atol=2e-6)
 # Split versus concatenated accumulation can differ; bound float32 summation.
 discrepancy=float(jnp.max(jnp.abs(independent-normal)))
 bound=float(8*np.finfo(np.float32).eps*jnp.max(jnp.sum(jnp.abs(a),axis=0)))
 assert discrepancy<=bound,(K,discrepancy,bound)
 rec['reconstruction'].update(native_gpu_rmse=value,gpu_minus_saved=value-rec['test_drift_rmse'],independent_gpu_rmse=split_value,independent_vs_native_gpu_max_absolute=discrepancy,float32_accumulation_bound=bound)
 print('GPU saved-model reconstruction passed',K,value,flush=True)
for path,digest in s['protected_sha256'].items():
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  while z:=f.read(4*1024*1024):h.update(z)
 assert h.hexdigest()==digest,path
s['classification']='verified scientific result'
s['audit_environment']=dict(jax=jax.__version__,gpu=str(jax.devices()[0]),no_training=True)
s['notes']=['Config retains base generation target_samples100000 and 80/10/10 defaults; actual explicit dataset view is660000 rows with640000/10000/10000 split. Runner consumes stored indices; it does not regenerate or resplit from these defaults.','Generic two_stage.py all-training fit docstring is not the executed validation-selected fitter: final parameter fitting uses576000 of nominal640000, internal validation64000.','No full training-target MSE or numerical rank/condition histories were recorded. Existing train coefficient RMSE is not target MSE. No such expensive diagnostics were added.']
with (BASE/'ARFF_DRIFT_K_CONSISTENCY_AUDIT.json').open('x') as f:json.dump(s,f,indent=2)
lines=['# ARFF drift-versus-capacity consistency audit','', '**Classification: verified scientific result.** All50 saved ARFF capacity runs passed provenance, configuration, shape, target/split/checkpoint and summary checks. Seed0 at each of five K values additionally passed independent saved-model reconstruction on CPU and GPU. No training or original artifact modification occurred.','',
'## Dataset, target and sample accounting','',f"Dataset view: `{s['dataset']['path']}`. Resolved canonical file: `{s['dataset']['resolved_path']}`. SHA-256: `{s['dataset']['sha256']}`.",
f"All660000 stored step sizes equal float32(1e-4) = {s['dataset']['step_sizes_unique'][0]:.18g}; dtype float32, shape(660000,1). Increments are archived float64. All50 artifact dataset hashes and recorded source paths match this exact dataset.",'',
'Executed path: controlled adapter sets native runner REPO_ROOT to the registered dataset view; production main loads that file and indexes stored train_idx. jnp.asarray casts archived increments to accepted float32 training arithmetic. learn passes r/h to final drift and r[fit_idx]/h[fit_idx] to every OOF drift fit. No state/response centering, standardization or additional h scaling. The covariance target instead uses OOF residual outer products divided by h. Sources: scripts/run_controlled_ex8.py::run_job; scripts/run_ex8_arff_validation_selected_crossfit.py::main/learn; src/arff/validation_selected.py.','',
'Nominal training640000; external validation10000; test10000. Final drift and covariance parameter fitting576000 with64000 internal validation. Each of five OOF regressions has512000 nominal inputs,460800 fitting and51200 internal validation, with128000 outer holdout. All stored internal split arrays were compared against the frozen deterministic split helper; every K shares the same seed-specific construction. No post-selection refit on all rows.','',
'Split-array SHA-256 values (array bytes):','']
for k,v in s['dataset']['index_sha256'].items():lines.append(f'- {k}: `{v}`')
lines+=['',f"Test-state SHA-256: `{s['dataset']['test_states_sha256']}`.",'',
'## Drift truth and error convention','',
'src/arff/evaluation.py::true_function_errors calls predict(model.drift,x) and instantaneous ex8_drift(x)=-x, then sqrt(mean((prediction−truth)^2)). Mean covers10000 rows AND2 coordinates: RMSE=sqrt(sum_n||error_n||²/(2 Ntest)). Thus the vector-norm formula in the request would be sqrt(2) times the saved values. This is the same established coordinate-averaged convention across all K; it is not a summary bug. The coefficient evaluator does not use h. Predictions are already drift rates; no multiplication/division by h follows prediction. NLL separately uses residual r−h*f.','',
'## Saved-model reconstruction','',
'omega shape(2,K), amplitude shape(2K,2), predictions shape(10000,2). Rows0:K are cosine amplitudes; rowsK:2K are sine amplitudes. Independently constructed [cos(X omega),sin(X omega)]@amp exactly matches native predict. Separate cosine and sine matrix products differ only by floating-point accumulation; measured errors are below the recorded float32 bounds. GPU checks reproduce saved metrics; CPU differences are explicitly reported rather than silently changing tolerances.','',
'| K | Saved RMSE | Native CPU−saved | Native GPU−saved | Independent GPU prediction max error | NumPy64 prediction max error vs CPU |','|---|---:|---:|---:|---:|---:|']
for r in s['records']:
 if 'reconstruction' in r:
  v=r['reconstruction'];lines.append(f"| {r['K']} | {r['test_drift_rmse']:.10g} | {v['cpu_minus_saved']:.5g} | {v['gpu_minus_saved']:.5g} | {v['independent_vs_native_gpu_max_absolute']:.5g} | {v['numpy64_vs_native_max_absolute']:.5g} |")
lines+=['','## Capacity/checkpoint checks','',
'Every fit uses300 adaptations, ridge lambda0.001, Metropolis delta0.2/gamma1, metropolis_test=True, resampling=False, internal-validation fraction0.1. Checkpoint indices remain one-based: first argmin(validation_mse)+1. All saved final drift model arrays exactly equal the selected-stage arrays; neither truncation nor a fixed-width model was found.','',
'| K | Real features | Drift/final total parameters | Selected drift iterations (seeds0–9) | Selected target-validation MSE mean / min / max |','|---|---:|---:|---|---:|']
for K in [64,128,256,512,1024]:
 rr=[r for r in s['records'] if r['K']==K];v=np.array([r['selected_validation_mse'] for r in rr])
 lines.append(f"| {K} | {2*K} | {6*K} / {14*K} | {','.join(str(r['selected_iteration']) for r in rr)} | {v.mean():.8g} / {v.min():.8g} / {v.max():.8g} |")
active={str(K):{name:sum(r['models'][name]['zero_amplitude_scalars'] for r in s['records'] if r['K']==K) for name in ['drift','covariance']} for K in [64,128,256,512,1024]}
lines+=['','Exact zero-amplitude counts across ten models per K: '+json.dumps(active)+'. Per-model zero rows/frequency columns and array hashes are in JSON. Nonzero coefficients do not certify numerical rank or independent features. No saved conditioning/rank diagnostics or training-target MSE histories exist; no new fits or per-iteration predictions were added. All300 validation-MSE values remain in each source artifact; initial, selected and last values are archived in audit JSON.','',
'## Reuse and provenance','',
'All10 K1024 capacity artifacts are byte-identical to both the seven-lag and final nine-lag h=1e-4 anchors; artifact equality implies exact model and metric equality. The JSON includes every source path, artifact hash and drift/covariance array hash. All10 K128 capacity runs are newly fitted at nominalN640000; complete records say reused=False and dataset/context checks reject the N80000 baseline.','',
'Metadata caveat: base get_config generation defaults still say target_samples100000 and80/10/10. They are not consumed to regenerate/resplit these registered views. The actual saved arrays, source hashes, dataset_view record and n_train/n_validation/n_test confirm640000/10000/10000. This is a documented template-metadata limitation, not a data mismatch.','',
'All snapshotted artifacts and frozen numerical source files retain their SHA-256 values. The flat/noisy drift-versus-K result remains after direct reconstruction; raw covariance recovery can improve independently. This audit verifies consistency, not an explanation of optimization/statistical mechanisms or a capacity-dominated regime. Source commit: '+s['source_commit']+'.']
with (BASE/'ARFF_DRIFT_K_CONSISTENCY_AUDIT.md').open('x') as f:f.write('\n'.join(lines)+'\n')
print('CLASSIFICATION:',s['classification'],flush=True)
