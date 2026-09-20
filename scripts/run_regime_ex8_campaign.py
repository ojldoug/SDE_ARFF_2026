#!/usr/bin/env python3
"""Immutable phase plans; reuse accepted worker implementation and failure semantics."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,json
from pathlib import Path
import regime_ex8 as r
c=r.campaign;ROOT=r.ROOT;STUDY=r.STUDY
SOURCES=['scripts/regime_ex8.py','scripts/run_regime_ex8.py','scripts/run_regime_ex8_campaign.py','scripts/run_controlled_campaign_float64_v2.py','scripts/run_controlled_ex8_float64_v2.py','scripts/run_controlled_ex8.py','scripts/validate_controlled_ex8.py']
def job(study,method,seed,K,N,h):
 return dict(study=study,variant=f'h_{h:.8g}',method=method,seed=seed,K=K,N=N,h=h,dataset_root=f'dataset_roots/N_{N}/h_{h:.8g}')
def plans(phase):
 m=r.adapter.verify_registration();methods=m['methods'];seeds=m['stage1_seeds']
 if phase=='N':return [job('regime_N',method,seed,1024,N,.0001) for N in m['N_grid'] for method in methods for seed in seeds]
 if phase=='h':return [job('regime_h',method,seed,1024,640000,h) for h in m['h_grid'] for method in methods for seed in seeds]
 if phase=='capacity':
  selection=json.loads((STUDY/'selection.json').read_text())
  return [job('regime_capacity',method,seed,K,selection['N_star'],selection['h_star']) for K in [64,256,512,1024] for method in methods for seed in seeds]
 raise ValueError(phase)
def prepare(phase):
 assert (STUDY/'data/complete.json').exists()
 control=STUDY/'campaigns'/phase;assert not control.exists()
 jobs=plans(phase);reuse={}
 for j in jobs:
  assert not c.paths(j)[0].exists()
  assert (STUDY/j['dataset_root']/'dataset_view.json').exists()
  candidates=[]
  if phase=='h' and j['h']==.0001:candidates=[dict(j,study='regime_N')]
  if phase=='capacity' and j['K']==1024:
   if j['h']==.0001:candidates.append(dict(j,study='regime_N'))
   if j['N']==640000:candidates.append(dict(j,study='regime_h'))
  for source in candidates:
   d,a,l,ctx=c.paths(source)
   if not (d/'complete.json').exists():continue
   done=c.infra.read_json(d/'complete.json');checked=c.validate(source,a,l,None if done.get('reused') else ctx)
   assert checked['artifact_sha256']==done['artifact_sha256'] and checked['log_sha256']==done['log_sha256']
   reuse[c.key(j)]=dict(artifact=str(a),log=str(l),**checked);break
 control.mkdir(parents=True)
 c.infra.write_json(control/'plan.json',dict(phase=phase,jobs=jobs,reuse=reuse,gpus=[0,1,2,3],timing='four-GPU non-isolated accuracy',created_at=c.infra.now(),frozen_sha256={p:r.adapter.digest(ROOT/p) for p in SOURCES}),exclusive=True)
 c.state(control,status='prepared',jobs_total=len(jobs),reused=len(reuse),new_jobs=len(jobs)-len(reuse))
 print('Prepared',phase,len(jobs),'jobs;',len(reuse),'verified same-data anchor reuses')
def run(phase):
 # The accepted worker uses its original entrypoint string. Redirect only that
 # process entrypoint, without changing its arguments or native numerical code.
 original=c.subprocess.Popen
 def popen(command,*args,**kwargs):
  command[:]=[str(ROOT/'scripts/run_regime_ex8.py') if str(v)==str(ROOT/'scripts/run_controlled_ex8_float64_v2.py') else v for v in command]
  return original(command,*args,**kwargs)
 c.subprocess.Popen=popen
 try:c.run(phase)
 finally:c.subprocess.Popen=original
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare(sys.argv[2])
 else:run(sys.argv[2])
