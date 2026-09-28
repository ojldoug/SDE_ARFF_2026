"""Compact zero-mean Gaussian GARCH/DCC; defining recursions checked independently."""
import numpy as np
from scipy.signal import lfilter
from scipy.optimize import minimize

def margins(r,par,initial):
 n,d=r.shape;v=np.empty_like(r,dtype=float);v[0]=initial
 for k in range(d):
  om,a,b=par[k]
  if np.isfinite(r[:,k]).all():
   v[1:,k]=lfilter([1.],[1.,-b],om+a*r[:-1,k]**2,zi=[b*initial[k]])[0]
  else:
   for t in range(1,n):v[t,k]=om+a*(r[t-1,k]**2 if np.isfinite(r[t-1,k]) else v[t-1,k])+b*v[t-1,k]
 return v

def correlations(z,a,b,qbar):
 n,d=z.shape;Q=np.empty((n,d,d));Q[0]=qbar
 if np.isfinite(z).all():
  u=(1-a-b)*qbar+a*np.einsum('ni,nj->nij',z[:-1],z[:-1]);Q[1:]=lfilter([1.],[1.,-b],u,axis=0,zi=(b*qbar)[None])[0]
 else:
  for t in range(1,n):
   scale=np.sqrt(np.diag(Q[t-1]));expected=Q[t-1]/scale[:,None]/scale[None,:]
   shock=np.outer(z[t-1],z[t-1]) if np.isfinite(z[t-1]).all() else expected
   Q[t]=(1-a-b)*qbar+a*shock+b*Q[t-1]
 diag=np.sqrt(np.diagonal(Q,axis1=1,axis2=2));return Q/diag[:,:,None]/diag[:,None,:],Q

def forecasts(r,model):
 v=margins(r,np.asarray(model['margins']),np.asarray(model['initial']));z=r/np.sqrt(v);R,Q=correlations(z,*model['dcc'],np.asarray(model['qbar']));return R*np.sqrt(v)[:,:,None]*np.sqrt(v)[:,None,:]

def ewma(r,decay,initial):
 n,d=r.shape;out=np.empty((n,d,d));H=initial.copy()
 for t in range(n):
  out[t]=H
  if np.isfinite(r[t]).all():H=decay*H+(1-decay)*np.outer(r[t],r[t])
 return out

def fit(r,eligible):
 initial=np.mean(r[eligible]**2,axis=0);pars=[];records=[]
 for k in range(r.shape[1]):
  def loss(p):
   if min(p)<=-1e-12 or p[1]+p[2]>.99900001:return 1e15
   v=margins(r[:,k:k+1],np.asarray([p]),initial[k:k+1])[eligible,0]
   if np.any(v<=0) or not np.isfinite(v).all():return 1e15
   return float(.5*np.mean(np.log(v)+r[eligible,k]**2/v))
  candidates=[]
  for frac,a,b in [(.02,.04,.94),(.04,.08,.88),(.1,.15,.75)]:
   res=minimize(loss,[frac*initial[k],a,b],method='SLSQP',bounds=[(1e-8,10),(0,.999),(0,.999)],constraints=[{'type':'ineq','fun':lambda p:.999-p[1]-p[2]}],options=dict(maxiter=300,ftol=1e-9))
   valid=bool(res.success and res.x[0]>=1e-8 and min(res.x[1:])>=0 and sum(res.x[1:])<=.999+1e-8 and np.isfinite(res.fun));row=dict(stage='margin',asset=k,start=[frac*initial[k],a,b],success=bool(res.success),valid=valid,parameters=res.x.tolist(),objective=float(res.fun),iterations=int(res.nit),message=str(res.message));records.append(row)
   if valid:candidates.append(res)
  if not candidates:return dict(valid=False,reason=f'No converged margin{k}',optimizer_records=records)
  best=min(candidates,key=lambda x:x.fun);pars.append(best.x)
 v=margins(r,np.array(pars),initial);z=r/np.sqrt(v);qbar=z[eligible].T@z[eligible]/len(eligible)
 if np.linalg.eigvalsh(qbar).min()<=0:return dict(valid=False,reason='non-SPD Qbar',optimizer_records=records)
 def lossq(p):
  a,b=p
  if min(p)<0 or a+b>.99900001:return 1e15
  R,_=correlations(z,a,b,qbar);RR=R[eligible];zz=z[eligible]
  sign,ld=np.linalg.slogdet(RR)
  if np.any(sign<=0):return 1e15
  return float(.5*np.mean(ld+np.sum(zz*np.linalg.solve(RR,zz[...,None])[...,0],axis=1)))
 candidates=[]
 for st in [(.02,.95),(.05,.90),(.10,.80)]:
  res=minimize(lossq,st,method='SLSQP',bounds=[(0,.999),(0,.999)],constraints=[{'type':'ineq','fun':lambda p:.999-p.sum()}],options=dict(maxiter=300,ftol=1e-9))
  valid=bool(res.success and min(res.x)>=0 and sum(res.x)<=.999+1e-8 and np.isfinite(res.fun));records.append(dict(stage='dcc',start=list(st),success=bool(res.success),valid=valid,parameters=res.x.tolist(),objective=float(res.fun),iterations=int(res.nit),message=str(res.message)))
  if valid:candidates.append(res)
 if not candidates:return dict(valid=False,reason='No converged DCC correlation',optimizer_records=records)
 return dict(valid=True,margins=np.array(pars).tolist(),initial=initial.tolist(),qbar=qbar.tolist(),dcc=min(candidates,key=lambda x:x.fun).x.tolist(),optimizer_records=records)
