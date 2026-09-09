import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,'.')
from src.experiments.definitions import get_experiment
from src.arff.regression import ARFFModel,predict
import jax.numpy as j
out=Path('results/final_reproduction/metric_reconciliation_evidence');d=np.load('data/ex7.npz');a=np.load('results/final_reproduction/production/accuracy/ex7/fourier/seed_0_artifacts.npz');idx=a['validation_idx'];x=d['x_data'][idx];h=d['step_sizes'][idx];r=d['r_data'][idx];f=np.asarray(get_experiment('ex7').drift(x));q=np.sum((r-h*f)**2/(h*.01),axis=1)
values=dict(D=2,h=float(h[0,0]),gaussian_constant=float(np.log(2*np.pi)),half_logdet=float(np.mean(np.sum(np.log(np.broadcast_to(h*.01,r.shape)),axis=1))*.5),mean_half_quadratic=float(q.mean()*.5),oracle_nll=float(.5*q.mean()+np.log(2*np.pi)+np.mean(np.log(h*.01))*1),log10=float(np.log(10)),omit_h_shift=float(-np.log(.001)),density_r_over_sqrt_h_jacobian=float(-np.log(.001)),density_r_over_h_jacobian=float(-2*np.log(.001)))
print('EX7',values);(out/'ex7_nll_decomposition.json').write_text(json.dumps(values,indent=2)+'\n')
old=np.load('data/ex6.npz');new=np.load('data/ex6_labels_v2.npz');a=np.load('results/final_reproduction/production/accuracy/ex6/arff_historical_corrected/seed_0_artifacts.npz');model=ARFFModel(j.asarray(a['drift_omega']),j.asarray(a['drift_amp']));sums=[0.,0.,0.];count=0
for i in range(0,len(old['x_data']),4096):
 xo=old['x_data'][i:i+4096];xn=new['x_data'][i:i+4096];fo=np.asarray(get_experiment('ex6').drift(xo));fn=np.asarray(get_experiment('ex6').drift(xn));po=np.asarray(predict(model,j.asarray(xo)));pn=np.asarray(predict(model,j.asarray(xn)))
 for k,e in enumerate([po-fo,pn-fn,fo-fn]):sums[k]+=float(np.sum(e.astype('float64')**2))
 count+=fo.size
v=dict(seed=0,estimator='arff_historical_corrected',original_coordinates_rmse=(sums[0]/count)**.5,corrected_coordinates_rmse=(sums[1]/count)**.5,truth_label_shift_rms=(sums[2]/count)**.5,interpretation='Fixed current model evaluated at original/corrected labels; not a counterfactual retraining')
print('EX6',v);(out/'ex6_label_only_evaluation.json').write_text(json.dumps(v,indent=2)+'\n')
