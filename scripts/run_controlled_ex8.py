#!/usr/bin/env python3
"""Preregistered process-local adapters; accepted numerical modules stay unchanged."""
from pathlib import Path
from dataclasses import replace,asdict
import argparse,sys,json,os,hashlib,importlib,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
STUDY=ROOT/'results/controlled_study_2026'
from src.experiments.config import get_config
from run_ex8_split_mlp_w27 import check_assigned_gpu
RUNNERS=dict(joint_fourier='run_adam_fourier_experiment',split_fourier='run_adam_split_fourier_experiment',joint_mlp='run_adam_mlp_experiment',split_mlp='run_adam_split_mlp_experiment',arff='run_ex8_arff_validation_selected_crossfit')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify_registration():
 for p,h in json.loads((STUDY/'registration_sha256.json').read_text()).items():
  if digest(ROOT/p)!=h:raise RuntimeError('Preregistration changed: '+p)
 m=json.loads((STUDY/'manifest.json').read_text())
 for p,h in m['frozen_source_sha256'].items():
  if digest(ROOT/p)!=h:raise RuntimeError('Accepted numerical source changed: '+p)
 return m

def configuration(job):
 cfg=replace(get_config('ex8'),fourier_frequencies=job['K'])
 if job['method']=='arff':
  cfg=replace(cfg,arff=replace(cfg.arff,M_min=300,M_max=300,lambda_reg=job.get('lambda_reg',.001),gamma=1.,delta=job.get('delta',.2),resampling=job.get('resampling',False),metropolis_test=job.get('metropolis_test',True)))
 if job.get('h',.0001)!=.0001:
  cfg=replace(cfg,data=replace(cfg.data,observation_lag=job['h'],trajectory_time=job['h'],em_substeps=round(job['h']/1e-7)))
 return cfg

def calibration(r,job,path):
 import jax,jax.numpy as jnp,numpy as np
 d=r.load_dataset(r.REPO_ROOT/'data/ex8.npz');definition=r.get_experiment('ex8')
 metadata=r.provenance(r.REPO_ROOT/'data/ex8.npz')
 x,inc,h=(jnp.asarray(v[d.train_idx]) for v in [d.x,d.r,d.h]);r.block_until_ready((x,inc,h))
 step,comp=r.prepare_compiled_functions(x,inc,h,seed=job['seed'],diff_type=definition.diff_type)
 result=r.learn(jax.random.PRNGKey(job['seed']),x,inc,h,seed=job['seed'],diff_type=definition.diff_type,compiled_step=step)
 metrics={label:r.evaluate_split(result.model,d.x[idx],d.r[idx],d.h[idx],definition) for label,idx in [('train',d.train_idx),('validation',d.validation_idx)]}
 arrays=r.build_artifact(result,d,metrics,seed=job['seed'],compilation_time=comp,metadata=metadata,other_times={})
 arrays.update(artifact_version=np.asarray(1),method=np.asarray('controlled_arff_calibration'),test_evaluated=np.asarray(False))
 for label in ('train','validation'):
  for key in ('nll','drift_rmse','covariance_rmse'):
   assert np.isfinite(arrays[label+'_'+key])
 assert not any(k.startswith('test_') and k!='test_evaluated' and k!='test_idx' for k in arrays)
 with path.open('xb') as f:np.savez_compressed(f,**arrays)
 print(json.dumps(metrics,indent=2),flush=True)

def run_job(job,path,allow_cpu=False):
 m=verify_registration();assert job['method'] in RUNNERS and job['K'] in [v['K'] for v in m['capacity']]
 assert job['N'] in m['N_grid'] or job.get('study')=='equal_fitting_count'
 if job.get('study')=='calibration':
  assert job['method']=='arff' and job['seed'] in m['calibration']['seeds']
  assert job['delta'] in m['calibration']['delta'] and job['lambda_reg'] in m['calibration']['lambda_reg']
  assert dict(metropolis_test=job['metropolis_test'],resampling=job['resampling']) in m['calibration']['modes']
 else:assert job['seed'] in m['stage2_seeds']
 assert not path.exists() and not path.with_suffix('.context.json').exists()
 view=STUDY/job.get('dataset_root',f"dataset_roots/N_{job['N']}")
 record=json.loads((view/'dataset_view.json').read_text());assert digest(view/'data/ex8.npz')==record['dataset_sha256']
 check_assigned_gpu(allow_cpu=allow_cpu)
 cfg=configuration(job);r=importlib.import_module(RUNNERS[job['method']]);r.REPO_ROOT=view;r.get_config=lambda name:cfg
 if hasattr(r,'check_machine_idle'):r.check_machine_idle=lambda allow_cpu=False:check_assigned_gpu(allow_cpu=allow_cpu)
 if job['method']=='arff':
  r.K=job['K'];r.FIT_SETTINGS=dict(r.FIT_SETTINGS,K=job['K'])
  for key in ('delta','lambda_reg','metropolis_test','resampling'):
   if key in job:r.FIT_SETTINGS[key]=job[key]
 context=dict(job=job,effective_config=asdict(cfg),dataset_view=record,wrapper_sha256=digest(__file__),registered_manifest_sha256=digest(STUDY/'manifest.json'),timing_context='non-isolated concurrent accuracy',native_runner=RUNNERS[job['method']],artifact_path=str(path),source_commit=m['source_commit'])
 path.parent.mkdir(parents=True,exist_ok=True)
 with path.with_suffix('.context.json').open('x') as f:json.dump(context,f,indent=2)
 print('CONTROLLED JOB',json.dumps(job,sort_keys=True),flush=True)
 if job.get('study')=='calibration':calibration(r,job,path)
 elif job['method']=='arff':r.main(['--seed',str(job['seed']),'--artifact-path',str(path)])
 else:
  sys.argv=[r.__file__,'ex8','--seed',str(job['seed']),'--artifact-path',str(path)]
  if allow_cpu and job['method']=='split_mlp':sys.argv.append('--allow-cpu')
  r.main()
 verify_registration();assert path.is_file()

def main():
 p=argparse.ArgumentParser();p.add_argument('--job-json',type=Path,required=True);p.add_argument('--artifact-path',type=Path,required=True);p.add_argument('--inspect',action='store_true');a=p.parse_args()
 job=json.loads(a.job_json.read_text())
 if a.inspect:print(json.dumps(dict(job=job,effective_config=asdict(configuration(job))),indent=2));return
 run_job(job,a.artifact_path.resolve())
if __name__=='__main__':main()
