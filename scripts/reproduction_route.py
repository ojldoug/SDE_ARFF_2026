#!/usr/bin/env python3
"""Portable single-job entry points; inspect is non-numerical, run is explicit.
Does not launch supervisors or alter accepted runners. Restored bundle is expected
at the repository root; all outputs must be outside accepted results/.
"""
import argparse,importlib,importlib.util,json,os,sys,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts'),str(ROOT/'GPU')]
STUDIES={'baseline':'results/controlled_study_2026/float64_v2','regime':'results/capacity_regime_ex8_v2','final-h':'results/capacity_regime_ex8_final_h_v1'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def relocate(v):
 if isinstance(v,str) and 'SDE_ARFF_2026/' in v and v.startswith('/'):
  return str(ROOT/v.split('SDE_ARFF_2026/',1)[1])
 if isinstance(v,list):return [relocate(x) for x in v]
 if isinstance(v,dict):return {relocate(k):relocate(x) for k,x in v.items()}
 return v
def load_module(path):
 spec=importlib.util.spec_from_file_location('archived_single_job',path);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('route',choices=['baseline','regime','final-h','historical','lag-ridge','coefficient-grids','fixed-basis','adaptive-ridge','resampling']);p.add_argument('--arm',choices=['M','MR'],default='M');p.add_argument('--execute',action='store_true');p.add_argument('--method',default='arff');p.add_argument('--experiment',default='ex1');p.add_argument('--seed',type=int,default=0);p.add_argument('--K',type=int,default=128);p.add_argument('--N',type=int,default=80000);p.add_argument('--h',type=float,default=.0001);p.add_argument('--drift-lambda',type=float,default=.001);p.add_argument('--output',type=Path);a=p.parse_args()
 if a.output and (a.output.resolve().is_relative_to(ROOT/'results') or a.output.exists()):p.error('Use a new output directory outside accepted results/')
 if a.execute and not a.output:p.error('--execute requires --output')
 if not a.execute:os.environ.setdefault('JAX_PLATFORMS','cpu')
 if a.route in STUDIES:
  folder=ROOT/STUDIES[a.route];plans=list((folder/'campaigns').glob('*/plan.json'))
  jobs=[]
  for plan in plans:
   for j in json.loads(plan.read_text()).get('jobs',[]):
    if all(j.get(k, .0001 if k=='h' else None)==v for k,v in dict(K=a.K,N=a.N,h=a.h,method=a.method,seed=a.seed).items()):jobs.append((j,plan))
  if not jobs:raise SystemExit('No registered matching job; no configuration synthesized')
  # Overlapping anchors have the same science; prefer baseline or capacity entry.
  job,plan=sorted(jobs,key=lambda z:(z[0].get('study') not in ['baseline','regime_capacity'],str(z[1])))[0]
  import run_controlled_ex8_float64_v2 as adapter
  adapter.STUDY=adapter.base.STUDY=folder
  def portable_registration():
   m=adapter.base.verify_registration()
   v=relocate(json.loads((ROOT/'results/controlled_study_2026/float64_v2_validation/validation.json').read_text()))
   assert v['status']=='passed'
   assert sha(ROOT/m['dataset']['path'])==m['dataset']['sha256']==v['corrected_sha256']
   assert sha(Path(v['original_dataset']))==v['original_sha256']
   return m
  adapter.verify_registration=portable_registration;adapter.verify_registration()
  view=folder/job.get('dataset_root',f"dataset_roots/N_{a.N}");record=json.loads((view/'dataset_view.json').read_text());assert sha(view/'data/ex8.npz')==record['dataset_sha256']
  from dataclasses import asdict
  print(json.dumps(dict(job=job,registered_plan=str(plan.relative_to(ROOT)),dataset=record,effective_config=asdict(adapter.configuration(job))),indent=2))
  if a.execute:
   a.output.mkdir(parents=True);(a.output/'job.json').write_text(json.dumps(job,indent=2));adapter.run_job(job,(a.output/'artifact.npz').resolve())
 elif a.route=='historical':
  if a.experiment not in ['ex1','ex2','ex3','ex5','ex6','ex7']:p.error('Only retained historical-compatible experiments')
  if a.method not in ['arff','fourier','mlp_shallow','mlp_deep'] or a.seed not in range(30):p.error('Unknown retained historical method or seed')
  if a.experiment=='ex3' and a.method not in ['arff','mlp_shallow']:p.error('Ex3 retains only ARFF and shallow MLP')
  arff=a.method=='arff';manifest=ROOT/f'results/final_reproduction/production_manifest.{"arff" if arff else "adam"}_v1.json';m=json.loads(manifest.read_text());record=m['datasets'][a.experiment];assert sha(ROOT/record['path'])==record['sha256']
  mod=importlib.import_module('run_final_historical_arff' if arff else 'run_final_historical_adam')
  print(json.dumps(dict(configuration=m['studies'][a.experiment],dataset=record,seed=a.seed,method=a.method),indent=2))
  if a.execute:
   from run_ex8_split_mlp_w27 import check_assigned_gpu
   check_assigned_gpu(allow_cpu=False);a.output.mkdir(parents=True)
   if arff:mod.run(a.experiment,a.seed,(a.output/'artifact.npz').resolve())
   else:mod.run(a.experiment,a.method,a.seed,(a.output/'artifact.npz').resolve())
 elif a.route=='lag-ridge':
  folder=ROOT/'results/ex8_h_lambda_drift_v1';raw=json.loads((folder/'manifest.json').read_text());m=relocate(raw)
  for file,h in m['sources'].items():assert sha(file)==h,file
  job=next(j for j in m['jobs'] if j['h']==a.h and j['lambda']==a.drift_lambda and j['seed']==a.seed)
  assert sha(Path(job['dataset_root'])/'data/ex8.npz')==job['dataset_sha256'];assert sha(job['baseline'])==job['baseline_sha256']
  module=load_module(folder/'study.py');print(json.dumps(job,indent=2))
  if a.execute:
   a.output.mkdir(parents=True);r=module.configure(job);r.main(['--seed',str(a.seed),'--artifact-path',str((a.output/'artifact.npz').resolve())]);module.validate(job,a.output/'artifact.npz')
 elif a.route in ['fixed-basis','adaptive-ridge','resampling']:
  import shutil
  name={'fixed-basis':'arff_fixed_basis_ridge_diagnostic_v1','adaptive-ridge':'arff_B_ridge_intervention_v1','resampling':'arff_resampling_isolation_v1'}[a.route]
  folder=ROOT/'results'/name;entry=folder/('diagnostic.py' if a.route=='fixed-basis' else 'study.py');mod=load_module(entry)
  # Rebase only archived path strings in memory. Original manifest bytes/hashes survive.
  original_read=mod.read
  def public_read(p):
   d=relocate(original_read(p))
   if isinstance(d,dict) and 'protected' in d:
    from package_independent_reproduction import private_reference
    def private(k):
     # Only manuscript-preservation checks; never computational source/data checks.
     return private_reference(k) or any(t in k for t in ['/SDE_NN_overleaf/','/reference/sde-identification/'])
    omitted=[k for k in d['protected'] if private(k)]
    d['protected']={k:v for k,v in d['protected'].items() if not private(k)}
    for k in d['protected']:
     if Path(k).is_absolute() and not Path(k).is_relative_to(ROOT):raise RuntimeError('Unresolved external numerical input: '+k)
    d['excluded_private_document_preservation_checks']=omitted
   return d
  mod.read=public_read
  if a.route=='fixed-basis':
   m=mod.read(folder/'provenance.json');assert sha(entry)==m['script_sha256']
   for f,h in m['protected'].items():assert sha(Path(f))==h,f
  else:m=mod.verify()
  print(json.dumps(dict(route=a.route,seed=a.seed,arm=a.arm,protected_files=len(m['protected']),private_document_preservation_checks_omitted=len(m.get('excluded_private_document_preservation_checks',[])),entry=str(entry.relative_to(ROOT))),indent=2))
  if a.execute:
   from run_ex8_split_mlp_w27 import check_assigned_gpu
   check_assigned_gpu(allow_cpu=False);a.output.mkdir(parents=True)
   for f in folder.iterdir():
    if f.is_file() and (f.suffix in ['.py','.npz'] or f.name in ['provenance.json','PREREGISTRATION.md','input_manifest.json','jobs.json','PREFLIGHT_PASSED.json']):shutil.copy2(f,a.output/f.name)
   mod.O=a.output.resolve();mod.PY=sys.executable
   if a.route=='fixed-basis':(a.output/'cases').mkdir();mod.run()
   elif a.route=='adaptive-ridge':mod.worker(a.seed)
   else:sys.path.insert(0,str(folder));mod.worker(a.seed,a.arm)
 elif a.route=='coefficient-grids':
  # Existing complete nine-model reconstruction gate; output isolation only.
  record=json.loads((ROOT/'results/conservative_manuscript_revision_v1/evaluation/RECONSTRUCTION_CHECKS_PUBLIC.json').read_text())
  print('Nine illustrative seed-0 models; original metric checks rtol=1e-5, atol=1e-6; GPU required.');print('Reconstruction record keys:',list(record))
  if a.execute:
   gpu=os.environ.get('CUDA_VISIBLE_DEVICES','0');mod=load_module(ROOT/'results/conservative_manuscript_revision_v1/evaluate_coefficients.py');os.environ['CUDA_VISIBLE_DEVICES']=gpu
   a.output.mkdir(parents=True);mod.E=a.output.resolve();mod.main()
if __name__=='__main__':main()
