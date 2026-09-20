"""Final bounded descriptive lag campaign; no selection/capacity entrypoints."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,json
import regime_ex8_final_h as r
c=r.campaign;ROOT=r.ROOT;STUDY=r.STUDY
PARENT=ROOT/'results/capacity_regime_ex8_v2'
SOURCES=['scripts/regime_ex8_final_h.py','scripts/run_regime_ex8_final_h.py','scripts/run_regime_ex8_final_h_campaign.py','scripts/run_controlled_campaign_float64_v2.py','scripts/run_controlled_ex8_float64_v2.py','scripts/run_controlled_ex8.py','scripts/validate_controlled_ex8.py']
def plans():
 m=r.adapter.verify_registration()
 assert max(m['h_grid'])<=.002
 return [dict(study='regime_h',variant=f'h_{h:.8g}',method=method,seed=seed,K=1024,N=640000,h=h,dataset_root=f'dataset_roots/N_640000/h_{h:.8g}') for h in m['h_grid'] for method in m['methods'] for seed in m['stage1_seeds']]
def prepare():
 assert json.loads((PARENT/'pipeline/state.json').read_text())['status']=='complete'
 assert json.loads((STUDY/'data/complete.json').read_text())['status']=='passed'
 control=STUDY/'campaigns/h';assert not control.exists()
 jobs=plans();reuse={};old_lags=c.infra.read_json(PARENT/'manifest.json')['h_grid']
 for j in jobs:
  assert not c.paths(j)[0].exists()
  view=STUDY/j['dataset_root'];assert (view/'dataset_view.json').exists()
  if j['h'] not in old_lags:continue
  d=PARENT/'production'/c.key(j);done=c.infra.read_json(d/'complete.json')
  assert c.infra.read_json(PARENT/j['dataset_root']/'dataset_view.json')==c.infra.read_json(view/'dataset_view.json')
  checked=c.validate(j,d/'artifact.npz',d/'console.log',None if done.get('reused') else d/'artifact.context.json')
  assert all(checked[k]==done[k] for k in ['artifact_sha256','log_sha256'])
  reuse[c.key(j)]=dict(artifact=str(d/'artifact.npz'),log=str(d/'console.log'),**checked)
 assert len(reuse)==350 and len(jobs)-len(reuse)==50*len(r.adapter.verify_registration()['amendment']['included_h'])
 control.mkdir(parents=True)
 c.infra.write_json(control/'plan.json',dict(phase='h',jobs=jobs,reuse=reuse,gpus=[0,1,2,3],timing='four-GPU non-isolated accuracy',created_at=c.infra.now(),frozen_sha256={p:r.adapter.digest(ROOT/p) for p in SOURCES}),exclusive=True)
 c.state(control,status='prepared',jobs_total=len(jobs),reused=len(reuse),new_jobs=len(jobs)-len(reuse))
 print('Prepared final h campaign',len(jobs),'points;',len(reuse),'exact reuses',flush=True)
def run():
 original=c.subprocess.Popen
 def popen(command,*args,**kwargs):
  command[:]=[str(ROOT/'scripts/run_regime_ex8_final_h.py') if str(v)==str(ROOT/'scripts/run_controlled_ex8_float64_v2.py') else v for v in command]
  return original(command,*args,**kwargs)
 c.subprocess.Popen=popen
 try:c.run('h')
 finally:c.subprocess.Popen=original
if __name__=='__main__':
 assert len(sys.argv)==2 and sys.argv[1] in ['prepare','run']
 if sys.argv[1]=='prepare':prepare()
 else:run()
