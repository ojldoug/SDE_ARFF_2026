"""Bounded inference only: selected seed 0, original metric populations then fixed grids."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='0',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLBACKEND='Agg')
from pathlib import Path
import sys,json,csv,hashlib,importlib,traceback
import numpy as np
O=Path(__file__).resolve().parent;R=O.parents[1];sys.path[:0]=[str(R),str(R/'GPU')]
import jax,jax.numpy as jnp
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment
from src.arff.regression import ARFFModel,predict
from src.arff.two_stage import TwoStageARFFModel
from src.arff.covariance import raw_covariance,project_spd
from src.arff.evaluation import gaussian_nll as arff_nll
from src.adam import fourier as ff,mlp as ml
E=O/'evaluation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,x):
 with (E/name).open('x') as f:json.dump(x,f,indent=2)

def load_model(ex,m,p):
 with np.load(p,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 assert int(a['seed'])==0
 if ex=='ex1' and m!='arff_historical_corrected':
  lib=importlib.import_module('lib.lib_Adam_FF' if m=='fourier' else 'lib.lib_Adam_tanh')
  assert sha(lib.__file__)==str(a['historical_source_sha256']),'historical source mismatch'
  leaves=['omega','amp'] if m=='fourier' else ['W1','W2','W3','b1','b2','b3','amp']
  def params(pre):return {k:jnp.asarray(a[pre+'_'+k]) for k in leaves if pre+'_'+k in a}
  d,c=params('drift'),params('covariance')
  return a,lambda x:lib.Functions.drift(d,x),lambda x:lib.Functions.diffusion_cov(c,x,'diagonal'),lambda x,r,h:lib.AdamTrain.nll_loss(d,c,x,r,h,'diagonal')
 if m in ['arff','arff_historical_corrected']:
  model=TwoStageARFFModel(ARFFModel(jnp.asarray(a['drift_omega']),jnp.asarray(a['drift_amp'])),ARFFModel(jnp.asarray(a['covariance_omega']),jnp.asarray(a['covariance_amp'])),str(a['diff_type']))
  return a,lambda x:predict(model.drift,x),lambda x:raw_covariance(model.covariance,x,model.diff_type),lambda x,r,h:arff_nll(model,x,r,h,spd_epsilon=float(a['spd_epsilon'])).nll
 mod=ml if 'mlp' in m else ff
 def params(pre):
  if mod is ml:return ml.MLPParams(tuple(jnp.asarray(a[f'{pre}_weight_{i}']) for i in range(3)),tuple(jnp.asarray(a[f'{pre}_bias_{i}']) for i in range(3)))
  return ff.FourierParams(jnp.asarray(a[pre+'_omega']),jnp.asarray(a[pre+'_amp']))
 model=(ml.AdamMLPModel if mod is ml else ff.AdamFourierModel)(params('drift'),params('covariance'),str(a['diff_type']))
 pred=ml.predict_mlp if mod is ml else ff.predict_fourier
 return a,lambda x:pred(model.drift,x),lambda x:mod.predict_covariance(model,x),lambda x,r,h:mod.gaussian_nll(model,x,r,h)

def main():
 assert jax.default_backend()=='gpu','Original GPU backend required for validation; no tolerance relaxation'
 protocol={'seed':0,'rule':'fixed illustrative seed, no selection','backend':jax.default_backend(),'devices':str(jax.devices()),'rtol':1e-5,'atol':1e-6,'tolerance_source':'existing Ex8 component RMSE serialization check; same bounds predeclared for all metric checks, not enlarged after evaluation','grids':'Ex1 prescribed 201-point slices; Ex8 100x100 cell centres on [-2,2]^2','source_hashes':{},'checkpoint_hashes':{},'dataset_hashes':{}}
 for d in [R/'src',R/'GPU/lib']:
  for p in d.rglob('*.py'):protocol['source_hashes'][str(p)]=sha(p)
 write('PROTOCOL.json',protocol)
 hrows=list(csv.DictReader((R/'results/final_reproduction/resolved_accuracy_v1/per_seed_metrics.csv').open()))
 brows=list(csv.DictReader((R/'results/final_ex8_publication_bundle/summaries/baseline/per_seed_split_metrics.csv').open()))
 records=[];models={};datasets={}
 for ex,methods in [('ex1',['arff_historical_corrected','fourier','mlp_shallow','mlp_deep']),('ex8',['joint_fourier','split_fourier','arff','joint_mlp','split_mlp'])]:
  dp=R/'data'/('ex1.npz' if ex=='ex1' else 'ex8_float64_v2.npz');data=load_dataset(dp);datasets[ex]=data;protocol['dataset_hashes'][str(dp)]=sha(dp)
  for m in methods:
   p=(R/f'results/final_reproduction/production/accuracy/ex1/{m}/seed_0_artifacts.npz' if ex=='ex1' else R/f'results/controlled_study_2026/float64_v2/production/baseline/K_128_N_80000/{m}/seed_0/artifact.npz')
   protocol['checkpoint_hashes'][str(p)]=sha(p);a,fp,cp,nll=load_model(ex,m,p);models[(ex,m)]=(fp,cp,nll)
   idx=a['validation_idx'] if ex=='ex1' else data.test_idx
   if ex=='ex8':
    ctx=json.loads(p.with_name('artifact.context.json').read_text());assert ctx['dataset_view']['dataset_sha256']==sha(dp)
   else:assert str(a['dataset_sha256'])==sha(dp)
   x,r,h=(jnp.asarray(v[idx]) for v in [data.x,data.r,data.h]);definition=get_experiment(ex)
   f=fp(x);c=cp(x);truef=definition.drift(x);sig=definition.diffusion_factor(x);truec=sig@sig.swapaxes(-1,-2)
   vals={'drift_rmse':float(jnp.sqrt(jnp.mean((f-truef)**2))),'covariance_rmse':float(jnp.sqrt(jnp.mean((c-truec)**2))),'nll':float(nll(x,r,h))}
   ref=next(v for v in (hrows if ex=='ex1' else brows) if v['method']==m and int(v['seed'])==0 and (v.get('experiment')==ex if ex=='ex1' else v['split']=='test'))
   rec={'experiment':ex,'method':m,'seed':0,'population':'validation' if ex=='ex1' else 'test','metrics':{}}
   for k,v in vals.items():rec['metrics'][k]={'computed':v,'archived':float(ref[k]),'difference':v-float(ref[k]),'passed':bool(np.isclose(v,float(ref[k]),rtol=1e-5,atol=1e-6))}
   records.append(rec);print(json.dumps(rec),flush=True)
   write(f'validation_{ex}_{m}.json',rec)
   if not all(x['passed'] for x in rec['metrics'].values()):raise RuntimeError('Reconstruction mismatch: '+ex+'/'+m+'; grids not evaluated')
   with (E/f'validation_predictions_{ex}_{m}.npz').open('xb') as out:np.savez_compressed(out,idx=idx,drift=np.asarray(f),covariance_raw=np.asarray(c))
 # Only after ALL nine validations pass, evaluate new deterministic visualization coordinates.
 for ex,data in datasets.items():
  if ex=='ex1':
   t=np.linspace(-1,1,201,dtype=np.float32);coords=np.concatenate([np.stack([t,t*0],axis=1),np.stack([t*0,t],axis=1)])
  else:
   v=-2+(np.arange(100,dtype=np.float64)+.5)*4/100;xx,yy=np.meshgrid(v,v);coords=np.stack([xx.ravel(),yy.ravel()],axis=1).astype(np.float32)
  x=jnp.asarray(coords);definition=get_experiment(ex);sig=definition.diffusion_factor(x)
  arrays={'coordinates':coords,'true_drift':np.asarray(definition.drift(x)),'true_covariance':np.asarray(sig@sig.swapaxes(-1,-2))}
  for (e,m),(fp,cp,_) in models.items():
   if e!=ex:continue
   c=cp(x);arrays[m+'_drift']=np.asarray(fp(x));arrays[m+'_covariance_raw']=np.asarray(c)
   if m in ['arff','arff_historical_corrected']:arrays[m+'_covariance_projected']=np.asarray(project_spd(c,epsilon=.001 if ex=='ex8' else 1e-6))
  with (E/f'{ex}_seed0_coefficients.npz').open('xb') as f:np.savez_compressed(f,**arrays)
 write('COMPLETE.json',dict(status='validated',validation=records,provenance=protocol,training=False,new_observations=False))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  write('FAILURE.json',{'error':str(exc),'traceback':traceback.format_exc()});raise
