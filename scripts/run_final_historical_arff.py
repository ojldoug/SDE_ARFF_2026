#!/usr/bin/env python3
"""Historical settings and control flow, accepted OOF/ridge/SPD corrections."""
from pathlib import Path
import argparse,json,time,sys,os
import jax
import jax.numpy as jnp
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import final_historical_arff_compat as compat
import run_ex8_arff_validation_selected_crossfit as shared
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment
from src.experiments.timing import block_until_ready
MANIFEST=ROOT/'results/final_reproduction/production_manifest.arff_v1.json'

def run(experiment,seed,path):
    wall=time.perf_counter()
    manifest=json.loads(MANIFEST.read_text());spec=manifest['studies'][experiment]
    settings=spec['historical_arff'];dataset_path=ROOT/manifest['datasets'][experiment]['path']
    assert shared.sha256_file(dataset_path)==manifest['datasets'][experiment]['sha256']
    data=load_dataset(dataset_path);definition=get_experiment(experiment)
    order=np.concatenate([data.train_idx,data.validation_idx,data.test_idx]);nv=int(.1*len(order))
    train,val=order[:-nv],order[-nv:]
    x,r,h=(jnp.asarray(a[train]) for a in (data.x,data.r,data.h));block_until_ready((x,r,h))
    split_metadata=[]
    def fitting(key,xx,yy,*,validation_seed,compiled_step,warmup=False):
        key,result,fi,vi=compat.fit(key,xx,yy,settings=settings,step=compiled_step,warmup=warmup)
        if not warmup:split_metadata.append((fi,vi))
        return key,result
    # Only this process-local orchestration reference changes. No source file or
    # Experiment8 configuration/fitter is edited or used for historical stopping.
    shared.fit_selected=fitting
    shared.SPD_EPSILON=spec['current_config']['evaluation']['spd_epsilon']
    start=time.perf_counter();step=compat.make_step(settings)
    shared.learn(jax.random.PRNGKey(987654321),x,r,h,seed=seed,diff_type=definition.diff_type,compiled_step=step,warmup=True)
    compilation=time.perf_counter()-start
    print(f'Experiment: {experiment}; seed: {seed}; backend: {jax.default_backend()}; K={settings["K"]}; train={len(train)} validation={len(val)} test=NONE',flush=True)
    result=shared.learn(jax.random.PRNGKey(seed),x,r,h,seed=seed,diff_type=definition.diff_type,compiled_step=step)
    start=time.perf_counter()
    metrics={label:shared.evaluate_split(result.model,data.x[idx],data.r[idx],data.h[idx],definition) for label,idx in [('train',train),('validation',val)]}
    metric_time=time.perf_counter()-start
    meta=shared.provenance(dataset_path)
    sources=json.loads(meta['source_sha256_json'])
    for p in ['scripts/run_final_historical_arff.py','scripts/final_historical_arff_compat.py','GPU/lib/lib_ARFF.py']:
        sources[p]=shared.sha256_file(ROOT/p)
    meta['source_sha256_json']=json.dumps(sources,sort_keys=True)
    arrays=dict(**meta,artifact_version=1,experiment=experiment,method='arff_historical_corrected',seed=seed,**settings,
        production_manifest_sha256=shared.sha256_file(MANIFEST),settings_json=json.dumps(settings,sort_keys=True),
        train_idx=train,validation_idx=val,n_test=0,n_folds=5,fold_seed=2026,arff_validation_fraction=.1,
        spd_epsilon=shared.SPD_EPSILON,diff_type=definition.diff_type,fold_id=result.fold_id,
        crossfit_covariance_targets=result.targets,final_prng_key=result.final_key,
        algorithm_time=result.algorithm_time,crossfit_algorithm_time=result.crossfit_time,compilation_time=compilation,
        end_to_end_time=compilation+result.algorithm_time,metric_evaluation_time=metric_time,
        end_to_end_scope='warmup plus learning only; wall_before_serialization separately includes loading/metrics/provenance',
        timing_context=os.environ.get('FINAL_TIMING_CONTEXT','four-GPU accuracy; non-isolated'),
        algorithm_scope='seven fits with splitting, initialization, adaptation, validation, stopping, OOF prediction and covariance target construction; excludes final metrics/provenance/serialization',
        iteration_convention='historical zero-based; first adaptation 0; stop checked before candidate retention',
        split_convention='same canonical permutation; last floor(.1*N) outer validation; no test; historical key-based inner split of training-only regressions',
        checkpoint_convention='earliest minimum eligible internal validation MSE, no refit; terminal stopping candidate excluded as historically',
        numerical_corrections='existing HIGHEST float32 ridge solve; correct cosine/sine multi-output frequency norms; existing tiny safe weights/ratios; OOF targets; raw covariance diagnostics and SPD projection only for NLL',
        adaptation_order='resample then random walk then optional Metropolis then ridge fit',
        covariance_rmse_convention='raw Sigma before SPD projection')
    positions=np.arange(len(train))
    for (name,stage),(fi,vi) in zip(result.stages.items(),split_metadata,strict=True):
        training=stage.training
        for field in ['validation_mse','moving_average','cumulative_time','best_iteration','best_validation_mse','best_time','stopped_iteration']:
            arrays[name+'_'+field]=getattr(training,field)
        arrays.update({name+'_omega':training.model.omega,name+'_amp':training.model.amp,name+'_algorithm_time':stage.elapsed,name+'_start_offset':stage.start_offset})
        inputs=positions[result.fold_id!=int(name[5:])] if name.startswith('fold_') else positions
        arrays[name+'_fit_train_positions']=inputs[fi];arrays[name+'_internal_validation_train_positions']=inputs[vi]
    for prefix,model in [('drift',result.model.drift),('covariance',result.model.covariance)]:
        arrays[prefix+'_omega']=model.omega;arrays[prefix+'_amp']=model.amp
        arrays[prefix+'_active_parameters']=model.omega.size+model.amp.size
    arrays['total_active_parameters']=arrays['drift_active_parameters']+arrays['covariance_active_parameters']
    for label,values in metrics.items():
        arrays.update({label+'_'+k:v for k,v in values.items()});print(label,values,flush=True)
    arrays={k:np.asarray(jax.device_get(v)) for k,v in arrays.items()}
    arrays['wall_before_serialization']=np.asarray(time.perf_counter()-wall)
    validate(arrays,settings)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as f:np.savez_compressed(f,**arrays)
    print('artifact:',path,flush=True)

def validate(a,settings):
    for k,v in settings.items():assert a[k].item()==v,(k,'configuration mismatch')
    for k,v in a.items():
        assert not v.dtype.hasobject,k
        if np.issubdtype(v.dtype,np.number):assert np.all(np.isfinite(v)),k
    assert int(a['n_test'])==0 and int(a['n_folds'])==5
    n=len(a['train_idx']);all_positions=set(range(n));fold_id=a['fold_id']
    assert set(fold_id)==set(range(5))
    for name in [f'fold_{i}' for i in range(5)]+['final_drift','covariance']:
        mse=a[name+'_validation_mse'];best=int(a[name+'_best_iteration']);stop=int(a[name+'_stopped_iteration'])
        assert len(mse)==stop+1 and 0<=best<=stop
        stopped=int(np.argmin(a[name+'_moving_average']))+5<stop and stop>settings['M_min']
        eligible=mse[:-1] if stopped else mse
        assert best==int(np.argmin(eligible)) and float(mse[best])==float(a[name+'_best_validation_mse'])
        fi=set(a[name+'_fit_train_positions']);vi=set(a[name+'_internal_validation_train_positions'])
        inputs=all_positions-set(np.flatnonzero(fold_id==int(name[5:]))) if name.startswith('fold_') else all_positions
        assert not fi&vi and fi|vi==inputs
        assert np.all(np.diff(a[name+'_cumulative_time'])>=0)
    assert not set(a['train_idx'])&set(a['validation_idx'])

def main():
    p=argparse.ArgumentParser();p.add_argument('experiment',choices=['ex1','ex2','ex3','ex5','ex6','ex7']);p.add_argument('--seed',type=int,required=True);p.add_argument('--artifact-path',type=Path,required=True);a=p.parse_args()
    if a.artifact_path.exists():raise FileExistsError(a.artifact_path)
    run(a.experiment,a.seed,a.artifact_path)
if __name__=='__main__':main()
