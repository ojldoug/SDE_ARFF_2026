"""Freeze preregistered calibration decisions using validation NLL only."""
from pathlib import Path
import sys,json,itertools
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_controlled_campaign_float64_v2_extended as c

def select(candidates):
 selected=[]
 for mode in [(True,False),(False,True),(True,True)]:
  rows=[r for r in candidates if (r['metropolis_test'],r['resampling'])==mode]
  assert len(rows)==4
  best=min(rows,key=lambda r:(r['mean_validation_nll'],r['delta'],r['lambda_reg']))
  selected.append({k:best[k] for k in ['variant','metropolis_test','resampling','delta','lambda_reg']})
 return selected

def main():
 study=c.STUDY;target=study/'calibration_selection.json';assert not target.exists()
 plan=c.verify(study/'campaigns/calibration')
 assert c.infra.read_json(study/'campaigns/calibration/state.json')['status']=='complete'
 m=c.adapter.verify_registration();groups={};hashes={}
 for job in plan['jobs']:
  directory,a,l,ctx=c.paths(job);v=c.validate(job,a,l,ctx)
  done=c.infra.read_json(directory/'complete.json');assert v['artifact_sha256']==done['artifact_sha256']
  with np.load(a,allow_pickle=False) as z:
   assert not bool(z['test_evaluated']) and 'test_nll' not in z
   # Only this validation scalar participates in selection; no test file read.
   value=float(z['validation_nll']);assert np.isfinite(value)
  groups.setdefault(job['variant'],[]).append((job,value));hashes[str(a)]=v['artifact_sha256']
 rows=[]
 for variant,entries in groups.items():
  assert sorted(j['seed'] for j,v in entries)==m['calibration']['seeds']
  job=entries[0][0]
  rows.append(dict(variant=variant,metropolis_test=job['metropolis_test'],resampling=job['resampling'],delta=job['delta'],lambda_reg=job['lambda_reg'],validation_nll_by_seed={str(j['seed']):v for j,v in entries},mean_validation_nll=float(np.mean([v for j,v in entries]))))
 chosen=select(rows)
 c.infra.write_json(target,dict(selected_modes=chosen,candidates=rows,criterion=m['calibration']['criterion'],test_access=False,source_sha256=hashes,script_sha256=c.adapter.digest(__file__)),exclusive=True)
 print(json.dumps(chosen,indent=2))
if __name__=='__main__':main()
