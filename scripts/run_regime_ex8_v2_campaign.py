"""Versioned amendment; accepted numerical workers and selection semantics."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,json
import regime_ex8_v2 as r
c=r.campaign;ROOT=r.ROOT;STUDY=r.STUDY
PARENT=ROOT/'results/capacity_regime_ex8_v1'
SOURCES=['scripts/regime_ex8_v2.py','scripts/run_regime_ex8_v2.py','scripts/run_regime_ex8_v2_campaign.py','scripts/run_controlled_campaign_float64_v2.py','scripts/run_controlled_ex8_float64_v2.py','scripts/run_controlled_ex8.py','scripts/validate_controlled_ex8.py']
def job(study,method,seed,K,N,h):
 return dict(study=study,variant=f'h_{h:.8g}',method=method,seed=seed,K=K,N=N,h=h,dataset_root=f'dataset_roots/N_{N}/h_{h:.8g}')
def plans(phase):
 m=r.adapter.verify_registration()
 if phase=='N':return [job('regime_N',method,seed,1024,N,.0001) for N in m['N_grid'] for method in m['methods'] for seed in m['stage1_seeds']]
 if phase=='h':return [job('regime_h',method,seed,1024,640000,h) for h in m['h_grid'] for method in m['methods'] for seed in m['stage1_seeds']]
 if phase=='capacity':
  selected=json.loads((STUDY/'selection.json').read_text())
  return [job('regime_capacity',method,seed,v['K'],selected['N_star'],selected['h_star']) for v in m['capacity'] for method in m['methods'] for seed in m['stage2_seeds']]
 raise ValueError(phase)
def prepare(phase):
 assert json.loads((STUDY/'data/complete.json').read_text())['status']=='passed'
 control=STUDY/'campaigns'/phase;assert not control.exists()
 jobs=plans(phase);reuse={}
 for j in jobs:
  assert not c.paths(j)[0].exists()
  view=STUDY/j['dataset_root'];assert (view/'dataset_view.json').exists()
  candidates=[]
  if phase in ['N','h']:candidates.append((PARENT,j))
  if phase=='capacity' and j['K']==1024:
   if j['h']==.0001:candidates.append((STUDY,dict(j,study='regime_N')))
   if j['N']==640000:candidates.append((STUDY,dict(j,study='regime_h')))
  # K128 is deliberately always a new fit at the selected common regime;
  # no prior baseline is reinterpreted as configuration-identical.
  for parent,source in candidates:
   d=parent/'production'/c.key(source)
   if not (d/'complete.json').exists():continue
   done=c.infra.read_json(d/'complete.json')
   source_view=c.infra.read_json(parent/source['dataset_root']/'dataset_view.json')
   assert source_view==c.infra.read_json(view/'dataset_view.json')
   checked=c.validate(source,d/'artifact.npz',d/'console.log',None if done.get('reused') else d/'artifact.context.json')
   assert all(checked[k]==done[k] for k in ['artifact_sha256','log_sha256'])
   reuse[c.key(j)]=dict(artifact=str(d/'artifact.npz'),log=str(d/'console.log'),**checked);break
  if phase=='N' or (phase=='h' and j['h'] in c.infra.read_json(PARENT/'manifest.json')['h_grid']):
   assert c.key(j) in reuse,'Historical phase must be reused, never retrained'
 control.mkdir(parents=True)
 c.infra.write_json(control/'plan.json',dict(phase=phase,jobs=jobs,reuse=reuse,gpus=[0,1,2,3],timing='four-GPU non-isolated accuracy',created_at=c.infra.now(),frozen_sha256={p:r.adapter.digest(ROOT/p) for p in SOURCES}),exclusive=True)
 c.state(control,status='prepared',jobs_total=len(jobs),reused=len(reuse),new_jobs=len(jobs)-len(reuse))
 print('Prepared',phase,len(jobs),'jobs;',len(reuse),'exact reuses',flush=True)
def run(phase):
 original=c.subprocess.Popen
 def popen(command,*args,**kwargs):
  command[:]=[str(ROOT/'scripts/run_regime_ex8_v2.py') if str(v)==str(ROOT/'scripts/run_controlled_ex8_float64_v2.py') else v for v in command]
  return original(command,*args,**kwargs)
 c.subprocess.Popen=popen
 try:c.run(phase)
 finally:c.subprocess.Popen=original
if __name__=='__main__':
 if sys.argv[1]=='prepare':prepare(sys.argv[2])
 else:run(sys.argv[2])
