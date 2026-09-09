#!/usr/bin/env python3
"""Supervisor fixtures; benchmark subprocess launches are mocked."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,tempfile
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_final_arff_campaign as c

def main():
 for code in (0,2):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);control=root/'control';control.mkdir();(root/'adam_campaign_control').mkdir();(root/'adam_campaign_control/state.json').write_text('{"status":"complete"}');job=dict(experiment='ex1',method='arff_historical_corrected',seed=0)
   launches=[]
   def launch(command,**kw):
    launches.append(command);assert kw['env']['CUDA_VISIBLE_DEVICES']=='0';assert kw['env']['JAX_PLATFORMS']=='cuda'
    Path(command[-1]).write_bytes(b'preserved fixture');kw['stdout'].write('fixture log\n')
    return type('Process',(),dict(pid=123,wait=lambda self:code))()
   def validate(job,a,l):return dict(artifact_sha256=c.digest(a),log_sha256=c.digest(l),algorithm_time=1.)
   with patch.object(c,'OUT',root),patch.object(c,'CONTROL',control),patch.object(c,'verify',return_value={'jobs':[job]}),patch.object(c.infra,'gpu_processes',return_value=[]),patch.object(c.infra,'gpu_uuid',return_value='fixtureGPU'),patch.object(c,'validate',side_effect=validate),patch.object(c.subprocess,'Popen',side_effect=launch):
    c.run();a,l,r=c.paths(job)
    assert l.read_text()=='fixture log\n' and (r/'exit.json').exists()
    assert a.exists()==(code==0)
    c.run();assert len(launches)==1,'Existing or suspicious path was relaunched'
   print('PASS supervisor preservation/reuse case returncode',code)
 print('No numerical processes launched.')
if __name__=='__main__':main()
