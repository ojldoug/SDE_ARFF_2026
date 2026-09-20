#!/usr/bin/env python3
"""Evaluation only: unchanged Joint-MLP drift + unchanged ARFF covariance."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_ENABLE_X64']='false'
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MPLBACKEND']='Agg'
# Bound this cheap diagnostic to two CPU cores at low priority; never use GPUs.
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(19)
from pathlib import Path
import sys,json,csv,hashlib,time,platform
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import jax,jax.numpy as jnp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'))
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment
from src.adam.mlp import MLPParams,predict_mlp
from src.arff import evaluation as evaluation
from src.arff.covariance import raw_covariance
from scripts.run_ex8_arff_validation_selected_crossfit import reconstruct_model,SPD_EPSILON
BASE=ROOT/'results/controlled_study_2026/float64_v2'
OUT=BASE/'hybrid_jointmlp_arff'
LABELS=dict(joint_mlp='Joint MLP Adam',arff='ARFF',split_mlp='Split MLP Adam',split_fourier='Split Fourier Adam',joint_fourier='Joint Fourier Adam')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def desc(values):
 x=np.asarray(values,dtype=float);assert x.shape==(30,) and np.isfinite(x).all()
 return dict(n=30,mean=float(x.mean()),sd=float(x.std(ddof=1)),median=float(np.median(x)),min=float(x.min()),max=float(x.max()))
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
def main():
 start=time.monotonic();assert not OUT.exists();OUT.mkdir()
 assert jax.default_backend()=='cpu' and not jax.config.jax_enable_x64
 baseline=json.loads((BASE/'baseline_summary/summary.json').read_text())
 dataset=ROOT/'data/ex8_float64_v2.npz';expected='6fb009e3d6fa5f241cb1a15f9a9815cbd16ea19c3a9e4ca546ae99bdae091a72'
 assert sha(dataset)==expected==baseline['dataset']['sha256']
 d=load_dataset(dataset);assert len(d.test_idx)==10000 and d.r.dtype==np.float64
 x,r,h=(jnp.asarray(v[d.test_idx]) for v in [d.x,d.r,d.h]);definition=get_experiment('ex8')
 with np.load(BASE/'baseline_summary/test_distributions.npz') as z:reference={k:z[k] for k in z.files}
 assert np.array_equal(reference['seeds'],np.arange(30))
 # Verify accepted source bytes and snapshot only immutable existing files.
 protected={str(dataset):sha(dataset)}
 for path,digest in baseline['source_sha256'].items():
  assert sha(ROOT/path)==digest,path
  protected[str(ROOT/path)]=digest
 for directory in [BASE/'baseline_summary',ROOT/'results/capacity_matched_ex8_w27',ROOT/'results/ex8_joint_split_mlp_w27']:
  for p in directory.rglob('*'):
   if p.is_file():protected[str(p)]=sha(p)
 for p in [ROOT/'src/arff/evaluation.py',ROOT/'src/arff/covariance.py',ROOT/'src/adam/mlp.py',ROOT/'scripts/run_ex8_arff_validation_selected_crossfit.py',ROOT/'data/ex8.npz',Path('/home/kammonaa/projects/SDE_NN_overleaf/arXiv_2026/RaulCom_ARFFSDELearning.tex')]:
  if p.exists():protected[str(p)]=sha(p)
 write(OUT/'protocol.json',dict(seeds=list(range(30)),dataset_sha256=expected,test_idx_sha256=sha_bytes(d.test_idx),method='Post-hoc same-seed component composition; not independently trained',spd_epsilon=SPD_EPSILON,arithmetic='Accepted float32 evaluation on CPU; archive increments remain float64; native JAX input conversion unchanged',rmse_tolerance=dict(rtol=1e-5,atol=1e-6),spd_cpu_comparison_tolerance='at most one test point for rate; minimum eigenvalue rtol1e-4 atol1e-6',retraining=False,tuning=False,test_selection=False,script_sha256=sha(__file__)))
 rows=[];predictions_f=[];predictions_cov=[];checks=[]
 original_predict=evaluation.predict
 def dispatch(model,xx):
  return predict_mlp(model,xx) if isinstance(model,MLPParams) else original_predict(model,xx)
 for seed in range(30):
  loaded={}
  for method in ['joint_mlp','arff']:
   p=BASE/f'production/baseline/K_128_N_80000/{method}/seed_{seed}'
   with np.load(p/'artifact.npz',allow_pickle=False) as z:loaded[method]={k:z[k] for k in z.files}
   a=loaded[method];ctx=json.loads((p/'artifact.context.json').read_text())
   assert int(a['seed'])==ctx['job']['seed']==seed
   assert ctx['dataset_view']['dataset_sha256']==expected
   assert ctx['dataset_view']['test_original_row_ids_sha256']==sha_bytes(d.test_idx)
   if 'test_idx' in a:np.testing.assert_array_equal(a['test_idx'],d.test_idx)
  aa,ma=loaded['arff'],loaded['joint_mlp'];assert float(aa['spd_epsilon'])==SPD_EPSILON==.001
  assert int(ma['hidden_width'])==27 and str(ma['diff_type'])==str(aa['diff_type'])=='symmetric'
  drift=MLPParams(tuple(jnp.asarray(ma[f'drift_weight_{i}']) for i in range(3)),tuple(jnp.asarray(ma[f'drift_bias_{i}']) for i in range(3)))
  arff=reconstruct_model(aa)
  hybrid=SimpleNamespace(drift=drift,covariance=arff.covariance,diff_type=arff.diff_type)
  # Only the drift prediction dispatch changes. The native ARFF residual,
  # covariance reconstruction, symmetrization, projection and NLL code runs intact.
  with patch.object(evaluation,'predict',side_effect=dispatch):
   likelihood=evaluation.gaussian_nll(hybrid,x,r,h,spd_epsilon=SPD_EPSILON)
   drift_error,cov_error=evaluation.true_function_errors(hybrid,x,true_drift=definition.drift,true_diffusion_factor=definition.diffusion_factor)
  arff_cpu=evaluation.gaussian_nll(arff,x,r,h,spd_epsilon=SPD_EPSILON)
  f=np.asarray(predict_mlp(drift,x));cov=np.asarray(raw_covariance(arff.covariance,x,arff.diff_type))
  predictions_f.append(f);predictions_cov.append(cov)
  np.testing.assert_allclose(drift_error,reference['joint_mlp_drift_rmse'][seed],rtol=1e-5,atol=1e-6)
  np.testing.assert_allclose(cov_error,reference['arff_covariance_rmse'][seed],rtol=1e-5,atol=1e-6)
  assert likelihood.spd_violation_rate==arff_cpu.spd_violation_rate
  assert likelihood.min_raw_eigenvalue==arff_cpu.min_raw_eigenvalue
  assert abs(likelihood.spd_violation_rate-float(aa['test_raw_spd_violation_rate']))<=1/len(d.test_idx)+1e-7
  np.testing.assert_allclose(likelihood.min_raw_eigenvalue,float(aa['test_min_raw_eigenvalue']),rtol=1e-4,atol=1e-6)
  check=dict(seed=seed,drift_rmse_minus_baseline=float(drift_error-reference['joint_mlp_drift_rmse'][seed]),covariance_rmse_minus_baseline=float(cov_error-reference['arff_covariance_rmse'][seed]),raw_spd_rate_cpu_minus_baseline=float(likelihood.spd_violation_rate-float(aa['test_raw_spd_violation_rate'])),min_raw_eigenvalue_cpu_minus_baseline=float(likelihood.min_raw_eigenvalue-float(aa['test_min_raw_eigenvalue'])),arff_nll_cpu_minus_baseline=float(arff_cpu.nll-reference['arff_nll'][seed]))
  checks.append(check)
  row=dict(seed=seed,drift_rmse=drift_error,covariance_rmse=cov_error,nll=likelihood.nll,raw_spd_violation_rate=float(aa['test_raw_spd_violation_rate']),min_raw_eigenvalue=float(aa['test_min_raw_eigenvalue']),cpu_raw_spd_violation_rate=likelihood.spd_violation_rate,cpu_min_raw_eigenvalue=likelihood.min_raw_eigenvalue,min_projected_eigenvalue=likelihood.min_projected_eigenvalue)
  for method in LABELS:row['nll_difference_vs_'+method]=likelihood.nll-float(reference[method+'_nll'][seed])
  rows.append(row);print('Validated paired seed',seed,flush=True)
 statistics={k:desc([v[k] for v in rows]) for k in ['drift_rmse','covariance_rmse','nll','raw_spd_violation_rate','min_raw_eigenvalue']}
 differences={}
 for method in LABELS:
  delta=np.array([v['nll_difference_vs_'+method] for v in rows])
  differences[method]=dict(label=LABELS[method],**desc(delta),lower=int(np.sum(delta<0)),higher=int(np.sum(delta>0)),equal=int(np.sum(delta==0)),definition='hybrid NLL minus frozen corrected-baseline comparator NLL; negative favors hybrid')
 for p,digest in protected.items():assert sha(p)==digest,p
 consistency={k:float(max(abs(c[k]) for c in checks)) for k in checks[0] if k!='seed'}
 write(OUT/'summary.json',dict(statistics=statistics,paired_nll_differences=differences,consistency_max_absolute_differences=consistency,spd_violating_seeds=sum(v['raw_spd_violation_rate']>0 for v in rows),protected_sha256=protected,environment=dict(python=platform.python_version(),jax=jax.__version__,numpy=np.__version__,backend=jax.default_backend(),cpu_affinity=sorted(os.sched_getaffinity(0))),elapsed_seconds=time.monotonic()-start,interpretation='Post-hoc composition of two components chosen after observing baseline performance. No further seed/checkpoint/config selection. Not an independent sixth trained method. Lower coefficient RMSE need not imply lower NLL; ARFF projection and covariance-weighted residuals matter.',spd_reporting='Inherited native ARFF diagnostics; CPU recomputation additionally archived for backend-precision checks.'))
 write(OUT/'consistency_checks.json',checks)
 with (OUT/'per_seed.csv').open('x',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 with (OUT/'test_predictions.npz').open('xb') as f:np.savez_compressed(f,seeds=np.arange(30),test_idx=d.test_idx,drift=np.stack(predictions_f),raw_covariance=np.stack(predictions_cov))
 lines=['# Post-hoc Joint MLP drift + ARFF covariance: corrected Ex8','', 'Evaluation only, all30 matching seeds and the fixed10000-point corrected test split. This is a composition of previously trained components, not an independently trained sixth estimator.','', '| Metric | Mean ± sample SD | Median | Range |','|---|---|---|---|']
 for k,v in statistics.items():lines.append(f"| {k} | {v['mean']:.9g} ± {v['sd']:.7g} | {v['median']:.9g} | [{v['min']:.9g}, {v['max']:.9g}] |")
 lines+=['','| Comparator | Paired NLL difference mean ± SD | Median | Range | Hybrid lower / higher / equal |','|---|---|---|---|---|']
 for method,v in differences.items():lines.append(f"| {v['label']} | {v['mean']:.9g} ± {v['sd']:.7g} | {v['median']:.9g} | [{v['min']:.9g}, {v['max']:.9g}] | {v['lower']} / {v['higher']} / {v['equal']} |")
 lines+=['','Difference = hybrid minus comparator; lower NLL is better. Comparators retain their exact frozen native baseline values. No NLL is inferred by combining existing scores.','',f"Raw SPD violations occur in {sum(v['raw_spd_violation_rate']>0 for v in rows)}/30 seeds. These raw diagnostics are inherited from the ARFF component; projection is used only for likelihood, not covariance RMSE.",'', 'The accepted ARFF code reconstructs the symmetric raw covariance, symmetrizes it, floors eigenvalues at1e-3, and evaluates mean full Gaussian NLL with covariance h*Sigma, dimension2 and the Gaussian constant. That code is unchanged. Only its drift-prediction dispatch is supplied by the saved MLP network; no estimator is converted or refitted.','', 'Consistency checks (maximum absolute discrepancy from frozen GPU/serialized values):','',json.dumps(consistency,indent=2),'', 'All component artifacts are paired by seed and verified against baseline hashes. Dataset SHA-256: '+expected+'. Test-index SHA-256: '+sha_bytes(d.test_idx)+'. All snapshotted source artifacts, datasets, baseline reports/figures and manuscript hashes remain unchanged. No training, tuning, checkpoint/seed selection, GPU work, or capacity-campaign control operation occurred. CPU evaluation used two cores at low priority.','', 'Interpretation: selecting these components was motivated by observed baseline coefficient errors, so this remains a post-hoc diagnostic on the same test set. It supplies no independently validated model-selection claim. Joint likelihood depends on covariance-weighted residuals and log determinants; stronger individual coefficient RMSE does not guarantee better NLL.']
 (OUT/'report.md').write_text('\n'.join(lines)+'\n')
 print('\n'.join(lines),flush=True)
def sha_bytes(a):return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()
if __name__=='__main__':main()
