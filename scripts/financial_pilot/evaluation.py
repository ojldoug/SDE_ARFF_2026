"""Common float64 Gaussian evaluator. No fitting or model inference."""
import numpy as np
from scipy.stats import chi2
RTOL=1e-9
ATOL=1e-9

def evaluate(y, raw, *, scale=None, h=1., mean=None, floor=1e-4, check=True):
 y=np.asarray(y,dtype=np.float64);raw=np.asarray(raw,dtype=np.float64)
 if mean is not None:y=y-np.asarray(mean,dtype=np.float64)*float(h)
 n,d=y.shape
 if raw.shape!=(n,d,d) or not np.isfinite(y).all() or not np.isfinite(raw).all():raise ValueError('Invalid/nonfinite evaluation inputs')
 D=np.ones(d,dtype=np.float64) if scale is None else np.asarray(scale,dtype=np.float64)
 if np.any(D<=0):raise ValueError('Invalid fitting scale')
 sym=(raw+raw.transpose(0,2,1))*.5;w,q=np.linalg.eigh(sym);wc=np.maximum(w,float(floor));delta=wc-w
 projected=np.einsum('nik,nk,njk->nij',q,wc,q)
 physical=projected*D[None,:,None]*D[None,None,:]*float(h)
 ys=y/D/np.sqrt(float(h));rot=np.einsum('nji,nj->ni',q,ys);wh=rot/np.sqrt(wc);mahal=np.sum(wh**2,axis=1)
 logdet=np.log(wc).sum(1)+2*np.log(D).sum()+d*np.log(float(h))
 eigen_nll=.5*(mahal+logdet+d*np.log(2*np.pi))
 # Independent factorization and triangular solve on physical matrix.
 L=np.linalg.cholesky(physical);u=np.linalg.solve(L,y[...,None])[...,0]
 chol_nll=.5*np.sum(u*u,axis=1)+np.log(np.diagonal(L,axis1=1,axis2=2)).sum(1)+d*.5*np.log(2*np.pi)
 if check:np.testing.assert_allclose(eigen_nll,chol_nll,rtol=RTOL,atol=ATOL)
 white=np.einsum('nij,nj->ni',q,wh)
 metrics={'nll':float(eigen_nll.mean()),'raw_spd_violation_rate':float(np.mean(w[:,0]<=0)),'min_raw_eigenvalue':float(w.min()),'projection_frequency':float(np.mean(np.any(delta>0,axis=1))),'floored_eigenvalue_fraction':float(np.mean(delta>0)),'mean_projection_frobenius':float(np.mean(np.linalg.norm(delta,axis=1))),'max_projection_frobenius':float(np.max(np.linalg.norm(delta,axis=1))),'mean_mahalanobis_squared':float(mahal.mean()),'whitened_mean':white.mean(0).tolist(),'whitened_second_moment':(white.T@white/n).tolist(),'coverage':{str(p):float(np.mean(mahal<=chi2.ppf(p,d))) for p in [.5,.9,.95]},'algebra_max_absolute_difference':float(np.max(np.abs(eigen_nll-chol_nll))),'count':n}
 return metrics,dict(nll=eigen_nll,chol_nll=chol_nll,mahal=mahal,projected_physical=physical,raw_eigenvalues=w,projection_norm=np.linalg.norm(delta,axis=1),white=white)
