"""Cheap supplemental CPU NLL comparisons for the fixed30-seed hybrid diagnostic."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['JAX_ENABLE_X64']='false';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1'
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[-2:]);os.nice(19)
from pathlib import Path
import sys,json,csv,hashlib
import numpy as np,jax.numpy as jnp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.adam import mlp,fourier
from src.arff import evaluation
from scripts.run_ex8_arff_validation_selected_crossfit import reconstruct_model,SPD_EPSILON
from src.experiments.dataset import load_dataset
BASE=ROOT/'results/controlled_study_2026/float64_v2';OUT=BASE/'hybrid_jointmlp_arff'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def describe(x):
 x=np.asarray(x,dtype=float)
 return dict(mean=float(x.mean()),sd=float(x.std(ddof=1)),median=float(np.median(x)),min=float(x.min()),max=float(x.max()),lower=int(np.sum(x<0)),higher=int(np.sum(x>0)),equal=int(np.sum(x==0)))
def main():
 primary=json.loads((OUT/'evaluation_v2/summary.json').read_text())
 rows=list(csv.DictReader((OUT/'evaluation_v2/per_seed.csv').open()))
 d=load_dataset(ROOT/'data/ex8_float64_v2.npz');x,r,h=(jnp.asarray(v[d.test_idx]) for v in [d.x,d.r,d.h]);methods=list(primary['paired_nll_differences']);records=[]
 for seed,row in enumerate(rows):
  assert int(row['seed'])==seed
  for method in methods:
   path=BASE/f'production/baseline/K_128_N_80000/{method}/seed_{seed}/artifact.npz'
   assert digest(path)==primary['protected_sha256'][str(path)]
   with np.load(path,allow_pickle=False) as z:a={k:z[k] for k in z.files}
   assert int(a['seed'])==seed
   if method=='arff':
    assert float(a['spd_epsilon'])==SPD_EPSILON
    nll=evaluation.gaussian_nll(reconstruct_model(a),x,r,h,spd_epsilon=SPD_EPSILON).nll
   elif method.endswith('mlp'):
    def params(prefix):return mlp.MLPParams(tuple(jnp.asarray(a[f'{prefix}_weight_{i}']) for i in range(3)),tuple(jnp.asarray(a[f'{prefix}_bias_{i}']) for i in range(3)))
    model=mlp.AdamMLPModel(params('drift'),params('covariance'),str(a['diff_type']))
    nll=float(mlp.gaussian_nll(model,x,r,h))
   else:
    def params(prefix):return fourier.FourierParams(jnp.asarray(a[prefix+'_omega']),jnp.asarray(a[prefix+'_amp']))
    model=fourier.AdamFourierModel(params('drift'),params('covariance'),str(a['diff_type']))
    nll=float(fourier.gaussian_nll(model,x,r,h))
   assert np.isfinite(nll)
   records.append(dict(seed=seed,method=method,comparator_cpu_nll=nll,hybrid_cpu_nll=float(row['nll']),difference=float(row['nll'])-nll))
  print('CPU comparator checks',seed,flush=True)
 summary={m:describe([v['difference'] for v in records if v['method']==m]) for m in methods}
 with (OUT/'same_backend_comparisons.json').open('x') as f:json.dump(dict(statistics=summary,per_seed=records,script_sha256=digest(__file__),purpose='Precision sensitivity only. No models or configurations selected. Hybrid NLL is the previously directly evaluated CPU value.'),f,indent=2)
 for path,sha in primary['protected_sha256'].items():assert digest(path)==sha,path
 text=(OUT/'evaluation_v2/report.md').read_text()
 text+='\n## Same-backend precision check\n\nAll five fixed comparator models were additionally evaluated on CPU with their native NLL functions. The direct hybrid NLL was unchanged. This avoids mixing CPU hybrid arithmetic with frozen GPU comparator arithmetic.\n\n| Comparator | CPU paired difference mean ± SD | Hybrid lower / higher / equal |\n|---|---|---|\n'
 for m,v in summary.items():text+=f"| {primary['paired_nll_differences'][m]['label']} | {v['mean']:.9g} ± {v['sd']:.7g} | {v['lower']} / {v['higher']} / {v['equal']} |\n"
 consistent=all((v['lower'],v['higher'],v['equal'])==tuple(primary['paired_nll_differences'][m][k] for k in ['lower','higher','equal']) for m,v in summary.items())
 text+=f'\nThe paired lower/higher counts agree with all frozen-GPU comparisons: **{consistent}**. Exact GPU numerical reproduction is not claimed. Maximum CPU/GPU differences were drift RMSE8.1211e-5, covariance RMSE2.4498e-5, raw SPD rate0.0034 (0.34 percentage points), minimum raw eigenvalue0.0035941, and ARFF NLL0.004077. These are reported rather than hidden by a relaxed tolerance. Same-backend hybrid/component differences were2.98e-8 for drift RMSE and exactly0 for covariance RMSE and both SPD diagnostics.\n\nThe hybrid improves ARFF likelihood modestly but retains ARFF covariance and its projection behavior. Better raw covariance RMSE than an MLP does not make it a better likelihood model. This remains a post-hoc composition and supplies no independent test-set validation of the choice of components.\n'
 with (OUT/'report.md').open('x') as f:f.write(text)
 with (OUT/'per_seed.csv').open('xb') as f:f.write((OUT/'evaluation_v2/per_seed.csv').read_bytes())
 with (OUT/'summary.json').open('x') as f:json.dump(dict(primary, same_backend_nll_differences=summary,same_backend_pair_counts_agree=consistent,final_protocol='evaluation_v2/protocol.json',failed_preflight='attempt_1_console.txt'),f,indent=2)
 print(json.dumps(summary,indent=2));print('Counts agree:',consistent)
if __name__=='__main__':main()
