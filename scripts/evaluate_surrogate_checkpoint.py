#!/usr/bin/env python3
"""Read an archived surrogate ARFF model; explicit --execute evaluates test rows.
No amplitude solve or adaptation. Shared data are not SDE trajectory observations.
"""
import argparse,json,hashlib,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--artifact',type=Path,required=True);p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
 import numpy as np
 data=ROOT/'results/arff_drift_K_mechanism_diagnostic_phase2_final_v1/common_data.npz';m=json.loads((ROOT/'docs/independent_reproduction_v1/artifact_manifest.json').read_text());allowed={v['sha256'] for v in m['files']}
 assert sha(data) in allowed and sha(a.artifact) in allowed
 with np.load(a.artifact,allow_pickle=False) as z:model={k:z[k] for k in z.files}
 assert 'omega' in model and 'amp' in model,'Select actual model, not spectral-analysis intermediates'
 out=dict(artifact_sha256=sha(a.artifact),data_sha256=sha(data),test_rows=[37768,42768],scope='shared-noise surrogate; existing selected/final model, no fitting')
 if a.execute:
  if not a.output or a.output.exists():p.error('New --output required')
  import jax.numpy as jnp
  from src.arff.regression import ARFFModel,predict
  with np.load(data,allow_pickle=False) as z:x,truth,y=[z[k][37768:42768] for k in ['x','truth','noisy_targets']]
  pred=predict(ARFFModel(jnp.asarray(model['omega']),jnp.asarray(model['amp'])),jnp.asarray(x))
  if 'prediction_test' in model:np.testing.assert_allclose(np.asarray(pred),model['prediction_test'],rtol=1e-5,atol=1e-6)
  mse=float(jnp.mean((pred-jnp.asarray(truth))**2));out.update(test_drift_mse=mse,test_drift_rmse=mse**.5,test_noisy_mse=float(jnp.mean((pred-jnp.asarray(y))**2)))
  with a.output.open('x') as f:json.dump(out,f,indent=2)
 print(json.dumps(out,indent=2))
if __name__=='__main__':main()
