"""Deterministic numerical fixtures and causal checks; no market test outcomes."""
from pathlib import Path
import sys,json
import numpy as np
from .evaluation import evaluate
from .dcc import margins,correlations,forecasts,ewma
from .data import make_features,P

def independent_dcc(r,pars,initial,a,b,qbar):
 v=initial.copy();q=qbar.copy();answer=[]
 for obs in r:
  R=q/np.sqrt(np.outer(np.diag(q),np.diag(q)));answer.append(np.sqrt(np.outer(v,v))*R)
  z=obs/np.sqrt(v)
  shock=np.outer(z,z) if np.isfinite(z).all() else R
  q=(1-a-b)*qbar+a*shock+b*q
  v=np.array([om+alpha*(x*x if np.isfinite(x) else vv)+beta*vv for x,vv,(om,alpha,beta) in zip(obs,v,pars)])
 return np.array(answer)

def run():
 r=np.reshape(np.sin(np.arange(1200)*.17),(400,3))*.01;t=np.arange(400)*3600
 idx,X,y=make_features(t,r)
 j=int(idx[25]);r2=r.copy();r2[j:]+=.2;idx2,X2,_=make_features(t,r2);np.testing.assert_allclose(X[25],X2[25],rtol=0,atol=0)
 w=r[j-168:j];cc=np.corrcoef(w,rowvar=False);ref=np.r_[w[-1],np.log(np.mean(w[-24:]**2,0)),np.log(np.mean(w**2,0)),cc[0,1],cc[0,2],cc[1,2]];np.testing.assert_allclose(X[25],ref,rtol=1e-10,atol=1e-10)
 _,zero,_=make_features(t,np.zeros_like(r));assert np.all(zero[:,9:]==0)
 raw=np.tile(np.array([[1,.2,.05],[.2,.8,.1],[.05,.1,.6]]),(400,1,1));raw[0,0,0]=-.1;D=np.array([.01,.02,.015])
 _,u=evaluate(r,raw,scale=D);_,v=evaluate(r/D,raw)
 np.testing.assert_allclose(u['nll'],v['nll']+np.log(D).sum(),rtol=1e-9,atol=1e-9);assert u['raw_eigenvalues'][0].min()<0;assert np.linalg.eigvalsh(u['projected_physical'][0]).min()>0
 pars=np.array([[.02,.05,.9]]*3);initial=np.array([1.,.9,1.1]);qbar=np.array([[1,.3,.1],[.3,1,.2],[.1,.2,1]])
 model=dict(margins=pars,initial=initial,dcc=[.04,.93],qbar=qbar)
 for gap in [False,True]:
  rr=r*100
  if gap:rr[125]=np.nan
  fast=forecasts(rr,model);ref=independent_dcc(rr,pars,initial,.04,.93,qbar);np.testing.assert_allclose(fast,ref,rtol=1e-11,atol=1e-11)
  perturb=rr.copy();perturb[200:]+=2;np.testing.assert_allclose(fast[:201],forecasts(perturb,model)[:201],rtol=0,atol=0)
  start=np.eye(3);result=ewma(rr,.97,start);manual=[];h=start.copy()
  for obs in rr:
   manual.append(h.copy())
   if np.isfinite(obs).all():h=.97*h+.03*np.outer(obs,obs)
  np.testing.assert_allclose(result,manual,rtol=1e-11,atol=1e-11)
  np.testing.assert_allclose(result[:201],ewma(perturb,.97,start)[:201],rtol=0,atol=0)
 record=dict(passed=True,fixtures='deterministic trigonometric arrays; not extra experimental observations',checks=['feature direct-window equality,168past only','future-target perturbation leaves current feature unchanged','zero-variance correlation rule','double eigen vs Cholesky NLL','scale Jacobian sum(log D)','negative eigenvalue flooring only in evaluation','DCC/GARCH against independent scalar-time reference, including missing observations','EWMA against independent update loop','EWMA/DCC forecast-before-observe future perturbation invariance'],tolerances=dict(gaussian=1e-9,recursion=1e-11,features=1e-10,causality='exact'),test_prices_opened=False)
 (P/'PREFIT_CHECKS.json').write_text(json.dumps(record,indent=2)+'\n');print('PREFIT CHECKS PASS')
if __name__=='__main__':run()
