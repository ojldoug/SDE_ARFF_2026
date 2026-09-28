"""Create an exclusive source/data/protocol envelope before any fitting."""
import json,subprocess,time
from .data import ROOT,P,sha

def main():
 if (P/'FROZEN.json').exists():raise FileExistsError('Already frozen')
 if (P/'models').exists():raise RuntimeError('Model directory exists before freeze')
 checks=json.loads((P/'PREFIT_CHECKS.json').read_text());assert checks['passed']
 jobs=[dict(method=k,configuration=v,grid_index=g,seed=s,budget=300) for k,grid in [('arff',[.001,.008,.064]),('neural',[.0003,.001,.003])] for g,v in enumerate(grid) for s in [0,1,2]]
 (P/'JOB_MANIFEST.json').write_text(json.dumps(dict(jobs=jobs,auxiliary='constant; EWMA3decays; DCC <=3deterministic starts per fitting objective',no_test_selection=True),indent=2)+'\n')
 files=list((ROOT/'scripts/financial_pilot').glob('*.py'))+list((ROOT/'src/arff').glob('*.py'))+list((ROOT/'src/adam').glob('*.py'))+[P/f for f in ['PROTOCOL.md','JOB_MANIFEST.json','development.npz','SPLIT_MANIFEST.json','DATA_MANIFEST.json','PREFIT_CHECKS.json']]
 identities={str(p.relative_to(ROOT)):sha(p) for p in files};record=dict(frozen_before_fitting=True,test_prices_opened=False,time=time.time(),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip()!='',identities=identities)
 with (P/'FROZEN.json').open('x') as f:json.dump(record,f,indent=2);f.write('\n')
 print('Frozen18candidate fits, no training or test-value access')
if __name__=='__main__':main()
