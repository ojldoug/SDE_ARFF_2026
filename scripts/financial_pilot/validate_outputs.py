"""Read completed artifacts and validate identities/report consistency; no inference."""
import json,csv,pickle,time
import numpy as np
from .data import ROOT,P,sha
from .pipeline import verify

def main():
 verify();complete=json.loads((P/'COMPLETE.json').read_text());assert complete['execution_complete']
 assert complete['summary_sha256']==sha(P/'summary.json')
 gate=json.loads((P/'PRETEST_GATE.json').read_text());assert gate['passed'] and not gate['test_values_opened']
 assert gate['selection_sha256']==sha(P/'SELECTION.json') and gate['identity']==sha(P/'FROZEN.json')
 models=[]
 for path in sorted((P/'models').glob('*.pkl')):
  meta=json.loads(path.with_suffix('.json').read_text());assert meta['sha256']==sha(path)==gate['model_hashes'][path.name]
  assert meta['frozen_sha256']==sha(P/'FROZEN.json')
  models.append(dict(name=path.name,sha256=sha(path),seconds=meta['seconds']))
 assert len(models)==19
 candidates=json.loads((P/'candidate_validation.json').read_text());assert len(candidates)==18
 assert len({(r['method'],r['grid_index'],r['seed']) for r in candidates})==18
 for row in json.loads((P/'DATA_MANIFEST.json').read_text())['records']:assert sha(ROOT/row['path'])==row['sha256']
 summary=json.loads((P/'summary.json').read_text());selected=json.loads((P/'SELECTION.json').read_text());checkpoint=[]
 for method in ['arff','neural']:
  vals=[np.mean([r['validation_nll'] for r in candidates if r['method']==method and r['grid_index']==g]) for g in range(3)];assert int(np.argmin(vals))==selected[method]['grid_index']
  for g in range(3):
   for seed in range(3):
    path=P/'models'/f'{method}_g{g}_s{seed}.pkl'
    with path.open('rb') as f:model=pickle.load(f)
    key='validation_mse' if method=='arff' else 'validation_nll';hist=np.asarray(model[key]);assert len(hist)==300 and np.isfinite(hist).all()
    choice=model['selected_iteration']-1 if method=='arff' else model['selected_epoch'];assert choice==int(np.argmin(hist))
    count=int(model['omega'].size+model['amp'].size) if method=='arff' else int(sum(a.size for a in (*model['model'].covariance.weights,*model['model'].covariance.biases)))
    assert count==(3072 if method=='arff' else 1275)
    checkpoint.append(dict(method=method,grid_index=g,seed=seed,selected_native_index=choice+(method=='arff'),native_validation_loss=float(hist[choice]),native_loss=key,parameters=count,elapsed_seconds=json.loads(path.with_suffix('.json').read_text())['seconds']))
 for split in ['validation','test']:
  scores=np.load(P/f'{split}_score_arrays.npz');pred=np.load(P/f'selected_{split}_predictions.npz')
  assert len(pred['returns'])==(8784 if split=='validation' else 8760)
  for name,m in summary['metrics'][split].items():
   assert np.isfinite(scores[name]).all() and np.isfinite(pred[name]).all()
   np.testing.assert_allclose(scores[name].mean(),m['nll'],rtol=0,atol=1e-12)
  if split=='validation':
   for method in ['arff','neural']:
    for seed in range(3):
     with (P/'models'/f'{method}_g{selected[method]["grid_index"]}_s{seed}.pkl').open('rb') as f:model=pickle.load(f)
     np.testing.assert_array_equal(pred[f'{method}_s{seed}'],model['validation_prediction'])
 with (P/'forecast_timestamps_test.csv').open() as f:
  rows=list(csv.DictReader(f));assert len(rows)==8760
  for row in rows:
   assert int(row['forecast_origin'])==int(row['latest_predictor'])==int(row['target_start'])
   assert int(row['target_end'])-int(row['target_start'])==3600
 # Original colloidal inputs remain untouched.
 A=ROOT/'results/experimental_trajectory_evaluation_float64_v1';a=json.loads((A/'summary.json').read_text())
 for name,h in a['input_hashes'].items():assert sha(ROOT/'results/experimental_trajectory_pilot_v1'/name)==h
 with (P/'checkpoint_summary.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=checkpoint[0]);w.writeheader();w.writerows(checkpoint)
 record=dict(passed=True,time=time.time(),model_count=len(models),candidate_count=len(candidates),archive_checksums_verified=147,checks=['frozen source/data/protocol','all19stage envelopes and pre-test model identities','all18finite histories and earliest minimum native checkpoint','mean3seed validation configuration rule','serialized validation arrays identical','common score arrays and summary means','8760test forecast timestamp ordering','colloidal original input hashes preserved'],models=models,scope='Artifact audit only; no additional model inference/fitting; parameter counts checked from saved arrays')
 (P/'OUTPUT_VALIDATION.json').write_text(json.dumps(record,indent=2)+'\n');print('OUTPUT VALIDATION PASS')
if __name__=='__main__':main()
