"""Historical ARFF control flow with explicitly accepted numerical corrections.

Separate from the frozen Experiment 8 fitter. Preserves resample-before-mutate,
legacy split/key flow, float32 rolling sum and zero-based stopping/selection order.
Reuses corrected ridge solves, frequency norms, OOF targets and SPD evaluation.
"""
from pathlib import Path
import sys,time
import numpy as np
import jax
import jax.numpy as jnp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'GPU'))
from lib.lib_ARFF import split_data
from src.arff.regression import ARFFModel,fit_amplitudes,predict,frequency_amplitude_norm
from src.arff.validation_selected import ValidationSelectedResult
from src.experiments.timing import block_until_ready


def make_step(settings):
    @jax.jit
    def step(key,model,x,y):
        omega=model.omega;K=omega.shape[1]
        norm=frequency_amplitude_norm(model.amp,K)
        tiny=jnp.finfo(model.amp.dtype).tiny
        if settings['resampling']:
            weights=jnp.maximum(norm,tiny);weights=weights/jnp.sum(weights)
            key,subkey=jax.random.split(key)
            omega=omega[:,jax.random.choice(subkey,K,shape=(K,),p=weights)]
        key,subkey=jax.random.split(key)
        proposal=omega+settings['delta']*jax.random.normal(subkey,omega.shape)
        if settings['metropolis_test']:
            amp=fit_amplitudes(x,y,proposal,settings['lambda_reg'])
            new_norm=frequency_amplitude_norm(amp,K)
            key,subkey=jax.random.split(key)
            ratio=(new_norm/jnp.maximum(norm,tiny))**settings['gamma']
            omega=jnp.where(ratio>=jax.random.uniform(subkey,(K,)),proposal,omega)
        else:omega=proposal
        return key,ARFFModel(omega,fit_amplitudes(x,y,omega,settings['lambda_reg']))
    return step


def fit(key,x,y,*,settings,step,warmup=False):
    start=time.perf_counter()
    # Historical helper preserves input row order inside the two subsets.
    # Archive its deterministic indices by passing them through the same split.
    (xx,yy,fit_idx),(vx,vy,val_idx),key=split_data(key,.1,x,y,jnp.arange(len(x)))
    omega=jnp.zeros((x.shape[1],settings['K']),dtype=x.dtype)
    model=ARFFModel(omega,fit_amplitudes(xx,yy,omega,settings['lambda_reg']))
    maximum=1 if warmup else settings['M_max']
    minimum=0 if warmup else settings['M_min']
    histories=jnp.zeros(maximum);rolling=jnp.asarray(0.,dtype=x.dtype)
    minimum_average=jnp.inf;minimum_index=0;best_loss=jnp.inf;best=None
    moving=[];times=[];selected=-1;best_time=np.nan
    for i in range(maximum):
        key,model=step(key,model,xx,yy)
        error=jnp.mean(jnp.abs(predict(model,vx)-vy)**2)
        histories=histories.at[i].set(error)
        rolling=rolling+error
        if i>=5:rolling=rolling-histories[i-5]
        average=rolling/min(i+1,5)
        moving.append(float(average));times.append(time.perf_counter()-start)
        if not np.isfinite(float(error)):raise RuntimeError('Nonfinite historical ARFF validation MSE')
        if average<minimum_average:minimum_average=average;minimum_index=i
        # Preserve historical candidate exclusion at the stopping iteration.
        if minimum_index+5<i and i>minimum:break
        if error<best_loss:best_loss=error;best=model;selected=i;best_time=times[-1]
    if best is None:raise RuntimeError('No finite selected ARFF model')
    block_until_ready((key,best.omega,best.amp))
    result=ValidationSelectedResult(best,np.asarray(histories[:i+1]),np.asarray(moving),np.asarray(times),selected,float(best_loss),best_time,i)
    return key,result,np.asarray(fit_idx),np.asarray(val_idx)
