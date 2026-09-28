"""Finite, frozen financial pipeline. Test opening requires passed development gates."""
from pathlib import Path
import os,sys,time,json,pickle,csv,hashlib
import numpy as np
from .data import ROOT,P,sha,load,make_features,timestamp
from .evaluation import evaluate
from . import dcc

def dump(path,obj):
 def conv(o):
  if isinstance(o,np.ndarray):return o.tolist()
  if isinstance(o,np.generic):return o.item()
  raise TypeError(type(o))
 path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2,default=conv,allow_nan=False)+'\n');tmp.replace(path)
def verify():
 f=json.loads((P/'FROZEN.json').read_text())
 for name,h in f['identities'].items():
  if sha(ROOT/name)!=h:raise RuntimeError('IDENTITY FAILURE '+name)
 return f

def stage(name,fn):
 path=P/'models'/f'{name}.pkl';path.parent.mkdir(exist_ok=True)
 if path.exists():
  meta=json.loads(path.with_suffix('.json').read_text());assert meta['sha256']==sha(path) and meta['frozen_sha256']==sha(P/'FROZEN.json')
  with path.open('rb') as f:return pickle.load(f)
 print('START',name,flush=True);dump(P/'status.json',dict(status='running',stage=name,time=time.time()));t=time.monotonic();result=fn()
 with path.open('xb') as f:pickle.dump(result,f)
 dump(path.with_suffix('.json'),dict(name=name,seconds=time.monotonic()-t,sha256=sha(path),frozen_sha256=sha(P/'FROZEN.json')));print('DONE',name,flush=True);return result

def lower_targets(r):
 a,b=np.tril_indices(3);return (r[:,:,None]*r[:,None,:])[:,a,b]
def symmetric(v):
 a,b=np.tril_indices(3);H=np.zeros((len(v),3,3),dtype=v.dtype);H[:,a,b]=v;H[:,b,a]=v;return H

def prediction(model,X):
 import jax.numpy as jnp
 if model['kind']=='arff':
  from src.arff.regression import ARFFModel,predict
  return symmetric(np.asarray(predict(ARFFModel(jnp.asarray(model['omega']),jnp.asarray(model['amp'])),jnp.asarray(X,dtype=jnp.float32))))
 from src.adam.split_mlp import covariance_from_split_model
 return np.asarray(covariance_from_split_model(model['model'],jnp.asarray(X,dtype=jnp.float32)))

def fit_arff(X,Y,V,W,ridge,seed):
 import jax,jax.numpy as jnp
 from src.arff.regression import ARFFModel,fit_amplitudes,make_compiled_adaptation_step,predict
 x=jnp.asarray(X,dtype=jnp.float32);y=jnp.asarray(lower_targets(Y),dtype=jnp.float32);v=jnp.asarray(V,dtype=jnp.float32);w=jnp.asarray(lower_targets(W),dtype=jnp.float32)
 omega=jnp.zeros((12,128),dtype=jnp.float32);m=ARFFModel(omega,fit_amplitudes(x,y,omega,ridge));key=jax.random.PRNGKey(seed);step=make_compiled_adaptation_step(delta=.2,lambda_reg=ridge,gamma=1.,resampling=False,metropolis_test=True)
 best=np.inf;hist=[];times=[];t=time.monotonic()
 for iteration in range(1,301):
  key,m=step(key,m,x,y);loss=float(jnp.mean((predict(m,v)-w)**2));hist.append(loss);times.append(time.monotonic()-t)
  if not np.isfinite(loss):raise ValueError('Nonfinite ARFF validation')
  if loss<best:best=loss;selected=iteration;bm=m
 result=dict(kind='arff',ridge=ridge,seed=seed,omega=np.asarray(bm.omega),amp=np.asarray(bm.amp),selected_iteration=selected,validation_mse=hist,cumulative_seconds=times)
 result['validation_prediction']=prediction(result,V)
 np.testing.assert_allclose(result['validation_prediction'],symmetric(np.asarray(predict(bm,v))),rtol=1e-5,atol=1e-6)
 return result

def fit_neural(X,Y,V,W,lr,seed):
 import jax,jax.numpy as jnp
 from src.adam.mlp import initialize_model
 from src.adam.split_mlp import CovarianceMLPModel,covariance_gaussian_nll,covariance_from_split_model
 from src.adam.training import make_compiled_adam_functions,fit_adam
 initial=initialize_model(jax.random.PRNGKey(seed),input_dimension=12,output_dimension=3,diff_type='triangular',hidden_sizes=(27,27));m=CovarianceMLPModel(initial.covariance,'triangular',3);opt,step,loss=make_compiled_adam_functions(lr,nll_fn=covariance_gaussian_nll)
 _,fit=fit_adam(jax.random.PRNGKey(seed),m,jnp.asarray(X,dtype=jnp.float32),jnp.asarray(Y,dtype=jnp.float32),jnp.ones((len(X),1)),jnp.asarray(V,dtype=jnp.float32),jnp.asarray(W,dtype=jnp.float32),jnp.ones((len(V),1)),epochs=300,batch_size=256,optimizer=opt,compiled_train_step=step,compiled_nll=loss)
 result=dict(kind='neural',lr=lr,seed=seed,model=jax.tree_util.tree_map(np.asarray,fit.model),selected_epoch=int(fit.best_epoch),validation_nll=fit.validation_nll,training_nll=fit.training_nll,cumulative_seconds=fit.cumulative_time)
 result['validation_prediction']=prediction(result,V)
 np.testing.assert_allclose(result['validation_prediction'],np.asarray(covariance_from_split_model(fit.model,jnp.asarray(V,dtype=jnp.float32))),rtol=1e-5,atol=1e-6)
 return result

def warmup():
 import jax,jax.numpy as jnp
 from src.arff.regression import ARFFModel,fit_amplitudes,make_compiled_adaptation_step
 from src.adam.mlp import initialize_model
 from src.adam.split_mlp import CovarianceMLPModel,covariance_gaussian_nll
 from src.adam.training import make_compiled_adam_functions
 t=time.monotonic();x=jnp.zeros((256,12));r=jnp.ones((256,3))*.1;y=jnp.ones((256,6));w=jnp.zeros((12,128));m=ARFFModel(w,fit_amplitudes(x,y,w,.001));step=make_compiled_adaptation_step(delta=.2,lambda_reg=.001,gamma=1.,resampling=False,metropolis_test=True);_,m=step(jax.random.PRNGKey(99),m,x,y);m.amp.block_until_ready()
 n=initialize_model(jax.random.PRNGKey(99),input_dimension=12,output_dimension=3,diff_type='triangular',hidden_sizes=(27,27));n=CovarianceMLPModel(n.covariance,'triangular',3);opt,fn,_=make_compiled_adam_functions(.001,nll_fn=covariance_gaussian_nll);n,_,loss=fn(n,opt.init(n),x,r,jnp.ones((256,1)));loss.block_until_ready()
 dump(P/'WARMUP.json',dict(seconds=time.monotonic()-t,rows=256,discarded_initializations=True,real_jobs_reset_to_frozen_seed=True,device=str(jax.devices())))

def record_runtime():
 import platform,importlib.metadata as im,jax,jaxlib
 dump(P/'RUNTIME.json',dict(python=sys.version,executable=sys.executable,platform=platform.platform(),packages={k:im.version(k) for k in ['jax','jaxlib','numpy','scipy','optax','matplotlib']},devices=[str(d) for d in jax.devices()],jax_enable_x64=bool(jax.config.jax_enable_x64),default_matmul_precision=str(jax.config.jax_default_matmul_precision),settings={k:v for k,v in os.environ.items() if k.startswith(('JAX_','XLA_','CUDA_','OMP_','OPENBLAS_'))}))

def main():
 verify();record_runtime();d=dict(np.load(P/'development.npz'));idx=d['idx'];fit=d['fit'];val=d['validation'];X=(d['X']-d['mu'])/d['sd'];scale=d['D'];target=d['r'][idx]/scale
 # Recursions start at first target in fitting period and include genuine intervening observations.
 start=int(idx[fit][0]);end=int(idx[fit][-1])+1;r=d['r'][start:]/scale;train_r=r[:end-start];eligible=idx[fit]-start
 constant=np.mean(target[fit,:,None]*target[fit,None,:],axis=0)
 modeldcc=stage('dcc',lambda:dcc.fit(train_r,eligible));dump(P/'DCC_REPORT.json',modeldcc);warmup()
 candidates=[];ewma_scores=[]
 vi=idx[val]-start
 for decay in [.94,.97,.99]:
  H=dcc.ewma(r,decay,constant)[vi];m,_=evaluate(d['r'][idx[val]],H,scale=scale);ewma_scores.append(dict(decay=decay,validation_nll=m['nll']))
 chosen_ewma=min(ewma_scores,key=lambda z:z['validation_nll'])['decay']
 for kind,grid in [('arff',[.001,.008,.064]),('neural',[.0003,.001,.003])]:
  for g,value in enumerate(grid):
   for seed in [0,1,2]:
    name=f'{kind}_g{g}_s{seed}';function=fit_arff if kind=='arff' else fit_neural
    m=stage(name,lambda:function(X[fit],target[fit],X[val],target[val],value,seed))
    fresh=prediction(m,X[val]);np.testing.assert_allclose(fresh,m['validation_prediction'],rtol=1e-5,atol=1e-6)
    metric,_=evaluate(d['r'][idx[val]],fresh,scale=scale)
    candidates.append(dict(method=kind,grid_index=g,configuration=value,seed=seed,validation_nll=metric['nll'],selected=m.get('selected_iteration',m.get('selected_epoch')),validation_raw_spd_rate=metric['raw_spd_violation_rate']))
    dump(P/'candidate_validation.json',candidates)
 selection={}
 for kind in ['arff','neural']:
  means=[np.mean([a['validation_nll'] for a in candidates if a['method']==kind and a['grid_index']==g]) for g in range(3)];selection[kind]=dict(grid_index=int(np.argmin(means)),mean_validation_nll=means)
 selection['ewma']=dict(decay=chosen_ewma,candidates=ewma_scores);selection['dcc_valid']=modeldcc['valid'];dump(P/'SELECTION.json',selection)
 verify()
 # Recheck all serialized models/predictions, common evaluator and numeric gates before any test read.
 from .checks import run as numerical_checks
 numerical_checks();gate=dict(passed=True,identity=sha(P/'FROZEN.json'),selection_sha256=sha(P/'SELECTION.json'),model_hashes={p.name:sha(p) for p in sorted((P/'models').glob('*.pkl'))},checkpoint_reconstruction_rtol=1e-5,checkpoint_reconstruction_atol=1e-6,test_values_opened=False,time=time.time())
 dump(P/'PRETEST_GATE.json',gate)
 t,rr=load(test=True);ix,features,years=make_features(t,rr);test=years==2025;testidx=ix[test];testX=(features[test]-d['mu'])/d['sd'];ty=rr[testidx]
 assert len(testidx)/8760>=.95
 # Verify development data are a byte-equal prefix of final causal data build.
 np.testing.assert_array_equal(rr[:len(d['r'])],d['r']);np.testing.assert_allclose(features[years<=2024],d['X'],rtol=0,atol=0)
 rfull=rr[start:]/scale;raws={'constant':np.tile(constant,(len(testidx),1,1)),'ewma':dcc.ewma(rfull,chosen_ewma,constant)[testidx-start]}
 valraws={'constant':np.tile(constant,(val.sum(),1,1)),'ewma':dcc.ewma(r,chosen_ewma,constant)[vi]}
 if modeldcc['valid']:
  raws['dcc']=dcc.forecasts(rfull,modeldcc)[testidx-start];valraws['dcc']=dcc.forecasts(r,modeldcc)[vi]
 for kind in ['arff','neural']:
  g=selection[kind]['grid_index']
  for seed in [0,1,2]:
   with (P/'models'/f'{kind}_g{g}_s{seed}.pkl').open('rb') as f:m=pickle.load(f)
   name=f'{kind}_s{seed}';raws[name]=prediction(m,testX);valraws[name]=m['validation_prediction']
   if kind=='arff':
    raws[name+'_diagonal']=np.eye(3)[None]*np.diagonal(raws[name],axis1=1,axis2=2)[:,None,:]
    valraws[name+'_diagonal']=np.eye(3)[None]*np.diagonal(valraws[name],axis1=1,axis2=2)[:,None,:]
 # Predictions complete: reporting/evaluator handles all valid methods, no ranking-based filtering.
 np.savez_compressed(P/'selected_test_predictions.npz',**raws,returns=ty,ends=t[testidx],D=scale)
 np.savez_compressed(P/'selected_validation_predictions.npz',**valraws,returns=d['r'][idx[val]],ends=d['t'][idx[val]],D=scale)
 with (P/'forecast_timestamps_test.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['forecast_origin','latest_predictor','target_start','target_end'])
  for j in testidx:assert t[j]-t[j-1]==3600;w.writerow([t[j-1],t[j-1],t[j-1],t[j]])
 from .report import report
 report()
 dump(P/'COMPLETE.json',dict(execution_complete=True,scientific_comparison_complete=bool(modeldcc['valid']),time=time.time(),summary_sha256=sha(P/'summary.json'),test_price_exposure='Only after PRETEST_GATE; no substantive prior test inspection',no_further_jobs=True))
 dump(P/'status.json',dict(status='complete',time=time.time()))

if __name__=='__main__':
 try:main()
 except Exception as e:
  dump(P/'FAILURE.json',dict(error=repr(e),time=time.time(),test_gate_exists=(P/'PRETEST_GATE.json').exists()));dump(P/'status.json',dict(status='failed',error=repr(e)));raise
