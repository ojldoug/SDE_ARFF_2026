"""CPU synthetic regression/control-flow checks; no production dataset or GPU."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import contextlib,io
import jax
import jax.numpy as jnp
import numpy as np
import final_historical_arff_compat as c
from lib import lib_ARFF as legacy

x=jax.random.normal(jax.random.PRNGKey(41),(40,2));y=x[:,:1]**2
s=dict(K=4,M_min=1,M_max=20,lambda_reg=.01,gamma=1.,delta=.1,resampling=True,metropolis_test=False)
# The old grouping is demonstrably wrong for q>1; the corrected norm pools
# matching cosine/sine rows across all output coordinates.
a=jnp.arange(24.).reshape(8,3)
expected=np.array([np.linalg.norm(np.asarray(a)[[k,k+4]]) for k in range(4)])
np.testing.assert_allclose(c.frequency_amplitude_norm(a,4),expected)
assert not np.allclose(np.linalg.norm(np.asarray(a).reshape(-1,4),axis=0),expected)
original_amp=legacy.ARFFTrain.get_amp;original_step=legacy.ARFFTrain.ARFF_one_step;original_beta=legacy.Functions.beta
try:
 legacy.ARFFTrain.get_amp=staticmethod(lambda x,y,lam,omega:c.fit_amplitudes(x,y,omega,lam))
 legacy.Functions.beta=staticmethod(lambda p,x:c.predict(c.ARFFModel(p['omega'],p['amp']),x))
 for mode in ['resample','metropolis','constant']:
  settings=dict(s,resampling=mode=='resample',metropolis_test=mode=='metropolis')
  yy=jnp.zeros_like(y) if mode=='constant' else y
  step=c.make_step(settings)
  if mode=='constant':step=lambda key,model,x,y:(key,model)
  def bridge(key,omega,amp,x,y,*args,**kwargs):
   key,model=step(key,c.ARFFModel(omega,amp),x,y)
   return model.omega,model.amp,key
  legacy.ARFFTrain.ARFF_one_step=staticmethod(bridge)
  hp=legacy.ARFFHyperparameters(**{k:settings[k] for k in ('K','M_min','M_max','lambda_reg','gamma','delta')})
  with contextlib.redirect_stdout(io.StringIO()):
   p,h,m,t,key=legacy.ARFFTrain(settings['resampling'],settings['metropolis_test']).ARFF_loop(jax.random.PRNGKey(7),hp,x,yy,.1)
  new_key,r,fit_idx,val_idx=c.fit(jax.random.PRNGKey(7),x,yy,settings=settings,step=step)
  np.testing.assert_array_equal(new_key,key)
  np.testing.assert_array_equal(r.model.omega,p['omega']);np.testing.assert_array_equal(r.model.amp,p['amp'])
  np.testing.assert_array_equal(r.validation_mse[:-1],h);np.testing.assert_array_equal(r.moving_average[:-1],m)
  assert len(fit_idx)==36 and len(val_idx)==4 and not set(fit_idx)&set(val_idx)
  if mode=='constant':assert r.stopped_iteration==6 and r.best_iteration==0
  print(mode,'PASS: legacy key, model and histories; complete terminal history archived')
finally:
 legacy.ARFFTrain.get_amp=original_amp;legacy.ARFFTrain.ARFF_one_step=original_step;legacy.Functions.beta=original_beta
