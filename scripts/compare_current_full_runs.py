#!/usr/bin/env python3
"""Compare two existing complete fits; no training or checkpoint selection."""
import argparse,hashlib,json,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--checkout',type=Path,required=True);p.add_argument('--previous',type=Path,required=True);p.add_argument('--repeat',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();sys.path[:0]=[str(a.checkout.resolve()),str(a.checkout.resolve()/'scripts')]
import numpy as np
import jax.numpy as jnp
from src.experiments.dataset import load_dataset
from src.arff.regression import predict
from src.arff.covariance import raw_covariance
from run_ex8_arff_validation_selected_crossfit import reconstruct_model,validate_artifact
old=dict(np.load(a.previous,allow_pickle=False));new=dict(np.load(a.repeat,allow_pickle=False));validate_artifact(old);validate_artifact(new)
def compare(x,y):
 x,y=np.asarray(x),np.asarray(y);out=dict(bitwise_equal=x.dtype==y.dtype and x.shape==y.shape and x.tobytes()==y.tobytes(),array_equal=bool(np.array_equal(x,y)),shape=list(x.shape))
 if np.issubdtype(x.dtype,np.number):
  out['within_original_tolerance']=bool(np.allclose(x,y,rtol=1e-5,atol=1e-6));out['max_abs_difference']=float(np.max(np.abs(x.astype(float)-y.astype(float))))
  if x.ndim==0:out.update(previous=x.item(),repeat=y.item())
 return out
allfields={k:compare(v,new[k]) for k,v in old.items()};stages=['fold_0','fold_1','fold_2','fold_3','fold_4','final_drift','covariance'];retained={}
for s in stages:
 x,y=old[s+'_validation_mse'],new[s+'_validation_mse'];ii=np.flatnonzero(x!=y)
 retained[s]=dict(selected_previous=int(old[s+'_best_iteration']),selected_repeat=int(new[s+'_best_iteration']),first_unequal_loss_iteration=int(ii[0])+1 if len(ii) else None)
path=a.checkout/'data/ex8_float64_v2.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==str(old['dataset_sha256'])==str(new['dataset_sha256']);data=load_dataset(path);models=[reconstruct_model(v) for v in [old,new]];preds={};arrays={}
for split in ['train','validation','test']:
 x=jnp.asarray(data.x[getattr(data,split+'_idx')]);values=[]
 for model in models:values.append(dict(drift=np.asarray(predict(model.drift,x)),raw_covariance=np.asarray(raw_covariance(model.covariance,x,model.diff_type))))
 for k in values[0]:
  name=split+'_'+k;preds[name]=compare(values[0][k],values[1][k]);preds[name]['previous_sha256']=hashlib.sha256(values[0][k].tobytes()).hexdigest();preds[name]['repeat_sha256']=hashlib.sha256(values[1][k].tobytes()).hexdigest()
result=dict(rtol=1e-5,atol=1e-6,previous_sha256=hashlib.sha256(a.previous.read_bytes()).hexdigest(),repeat_sha256=hashlib.sha256(a.repeat.read_bytes()).hexdigest(),schema_equal=old.keys()==new.keys(),fields=allfields,stage_summary=retained,predictions=preds)
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(stages=retained,unequal_fields=[k for k,v in allfields.items() if not v['bitwise_equal']],prediction_checks=preds),indent=2))
