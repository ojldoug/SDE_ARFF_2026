#!/usr/bin/env python3
"""Historical Adam compatibility runner: existing legacy numerical functions, new archival envelope."""
from pathlib import Path
import sys,os,json,time,hashlib,importlib,argparse
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'GPU'))
import jax
import jax.numpy as jnp
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment
from src.experiments.timing import timed_call,block_until_ready

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load_view(name):
    p=ROOT/'data'/('ex6_labels_v2.npz' if name=='ex6' else name+'.npz')
    d=load_dataset(p)
    # Reuse the existing deterministic permutation; historical last floor(.1*N) is validation.
    order=np.concatenate([d.train_idx,d.validation_idx,d.test_idx]);nv=int(.1*len(order))
    return p,d,order[:-nv],order[-nv:]

def metrics(lib,dp,cp,x,r,h,definition):
    dtype=definition.diff_type
    f=np.asarray(lib.Functions.drift(dp,x));sigma=np.asarray(lib.Functions.diffusion(cp,x,dtype))
    cov=sigma@np.swapaxes(sigma,-1,-2)
    true_sigma=np.asarray(definition.diffusion_factor(x));truth=true_sigma@np.swapaxes(true_sigma,-1,-2)
    eig=np.linalg.eigvalsh(cov)
    return dict(nll=float(lib.AdamTrain.nll_loss(dp,cp,x,r,h,dtype)),
        drift_rmse=float(np.sqrt(np.mean((f-np.asarray(definition.drift(x)))**2))),
        covariance_rmse=float(np.sqrt(np.mean((cov-truth)**2))),
        raw_spd_violation_rate=float(np.mean(eig[:,0]<=0)),min_raw_eigenvalue=float(eig.min()))

def run(name,method,seed,path):
    manifest_path=ROOT/'results/final_reproduction/production_manifest.adam_v1.json'
    manifest=json.loads(manifest_path.read_text());spec=manifest['studies'][name]
    settings=spec['historical_adam'];width=settings['K'];hidden=(width,) if method!='mlp_deep' else (width//2,width//2)
    lib=importlib.import_module('lib.lib_Adam_FF' if method=='fourier' else 'lib.lib_Adam_tanh')
    dataset_path,d,train,val=load_view(name)
    expected=manifest['datasets'][name]['sha256'];assert digest(dataset_path)==expected
    definition=get_experiment(name)
    order=np.concatenate([train,val]);x,r,h=(jnp.asarray(a[order]) for a in (d.x,d.r,d.h))
    block_until_ready((x,r,h))
    hp=dict(epochs=settings['epochs'],batch_size=settings['batch_size'],learning_rate=settings['learning_rate'],layer_widths=hidden)
    lib.AdamTrain.set_opt(hp)
    # Exact initialization key progression from historical Adam_training.ipynb repeated-run cell7.
    key,kd,kc=jax.random.split(jax.random.PRNGKey(seed),3)
    dp=lib.init_drift_params(kd,hidden,x.shape[1],r.shape[1]);cp=lib.init_diffusion_params(kc,hidden,x.shape[1],r.shape[1],definition.diff_type)
    block_until_ready((dp,cp))
    # Warm up original numerical functions using discarded states, without consuming production keys.
    od=lib.opt.init(dp);oc=lib.opt.init(cp);batch=hp['batch_size'];n=len(train)
    compilation=0.
    for size in sorted(set([min(batch,n),n%batch])-{0}):
        _,elapsed=timed_call(lib.AdamTrain.train_step,dp,cp,od,oc,x[:size],r[:size],h[:size],definition.diff_type);compilation+=elapsed
    _,elapsed=timed_call(lib.AdamTrain.nll_loss,dp,cp,x[n:],r[n:],h[n:],definition.diff_type);compilation+=elapsed
    original_loss=lib.AdamTrain.nll_loss
    best=dict(loss=np.inf,epoch=-1,drift=None,covariance=None);epoch=[0]
    def observe(dp,cp,xx,rr,hh,diff_type):
        value=original_loss(dp,cp,xx,rr,hh,diff_type)
        # Existing training_loop evaluates validation outside train_step once per epoch.
        # The jitted train_step was already traced with the original loss during warm-up.
        if not isinstance(value,jax.core.Tracer) and len(xx)==len(val):
            v=float(value)
            if v<best['loss']:best.update(loss=v,epoch=epoch[0],drift=dp,covariance=cp)
            epoch[0]+=1
        return value
    lib.AdamTrain.nll_loss=staticmethod(observe)
    print(f'Experiment: {name}; method: {method}; seed: {seed}; backend: {jax.default_backend()}; train={len(train)} validation={len(val)} test=NONE',flush=True)
    try:
        result,algorithm=timed_call(lib.AdamTrain.training_loop,hp,dp,cp,x,r,h,definition.diff_type,.1,plot=False)
    finally:lib.AdamTrain.nll_loss=original_loss
    _,_,times,losses,validation=result
    assert len(validation)==hp['epochs'] and epoch[0]==hp['epochs']
    assert best['epoch']==int(np.argmin(validation)) and np.all(np.isfinite(validation))
    dp,cp=best['drift'],best['covariance']
    final={label:metrics(lib,dp,cp,jnp.asarray(d.x[idx]),jnp.asarray(d.r[idx]),jnp.asarray(d.h[idx]),definition) for label,idx in [('train',train),('validation',val)]}
    arrays=dict(artifact_version=1,experiment=name,method=method,seed=seed,settings_json=json.dumps(hp,sort_keys=True),
        dataset_path=str(dataset_path),dataset_sha256=expected,production_manifest_sha256=digest(manifest_path),
        historical_source_path=lib.__file__,historical_source_sha256=digest(lib.__file__),
        runner_source_sha256=digest(__file__),train_idx=train,validation_idx=val,n_test=0,
        split_convention='same canonical permutation; last floor(.1*N) validation; no held-out test',
        checkpoint_convention='earliest minimum canonical validation NLL; retain immutable epoch parameters; no refit',
        initialization_convention='exact legacy initializer, PRNGKey(seed) split3; historical epoch PRNGKey(epoch) shuffling',
        covariance_convention='Sigma=(L L^T)^2' if definition.diff_type=='symmetric' else 'historical '+definition.diff_type,
        epochs=hp['epochs'],batch_size=hp['batch_size'],learning_rate=hp['learning_rate'],hidden_sizes=hidden,
        training_nll_history=losses,validation_nll_history=validation,cumulative_time=times,best_epoch=best['epoch'],best_validation_nll=best['loss'],
        algorithm_time=algorithm,compilation_time=compilation,end_to_end_time=algorithm+compilation,
        timing_scope='original full historical loop after discarded warmup; final evaluation/hash/serialization excluded',
        timing_context=os.environ.get('FINAL_TIMING_CONTEXT','four-GPU accuracy campaign; non-isolated'))
    for prefix,params in [('drift',dp),('covariance',cp)]:
        for k,v in params.items():arrays[prefix+'_'+k]=np.asarray(v)
        # Legacy tanh has an unused amp leaf; preserve it and distinguish active from stored parameters.
        arrays[prefix+'_active_parameters']=sum(v.size for k,v in params.items() if method=='fourier' or k!='amp')
        arrays[prefix+'_stored_parameters']=sum(v.size for v in params.values())
    arrays['total_active_parameters']=arrays['drift_active_parameters']+arrays['covariance_active_parameters']
    for label,values in final.items():
        arrays.update({label+'_'+k:v for k,v in values.items()});print(label,values,flush=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
    print('artifact:',path,flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('experiment',choices=['ex1','ex2','ex3','ex5','ex6','ex7']);p.add_argument('method',choices=['fourier','mlp_shallow','mlp_deep']);p.add_argument('--seed',type=int,required=True);p.add_argument('--artifact-path',type=Path,required=True)
    a=p.parse_args()
    if a.artifact_path.exists():raise FileExistsError(a.artifact_path)
    run(a.experiment,a.method,a.seed,a.artifact_path)
if __name__=='__main__':main()
