from pathlib import Path
import numpy as np,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
O=Path(__file__).resolve().parent;E=O/'evaluation';F=O/'figures'
record=json.loads((E/'RECONSTRUCTION_CHECKS_PUBLIC.json').read_text())
assert record['status']=='validated' and all(
 all(metric['passed'] for metric in row['metrics'].values()) for row in record['validation']
)
import hashlib
for name, expected in record['grid_data_sha256'].items():
 assert hashlib.sha256((E/name).read_bytes()).hexdigest()==expected
plt.rcParams.update({'font.size':11,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
colors=['#009E73','#0072B2','#D55E00','#CC79A7']
def save(fig,name):
 for ext in ['pdf','png']:fig.savefig(F/(name+'.'+ext),dpi=200,bbox_inches='tight')
 plt.close(fig)
z=np.load(E/'ex1_seed0_coefficients.npz');t=z['coordinates'][:201,0];fig,axs=plt.subplots(2,2,figsize=(9,6),layout='constrained')
methods=['arff_historical_corrected','fourier','mlp_shallow','mlp_deep'];names=['ARFF','Fourier Adam','Shallow MLP Adam','Deep MLP Adam']
for j in range(2):
 sl=slice(j*201,(j+1)*201)
 for row,kind in enumerate(['drift','covariance_raw']):
  a=axs[row,j];truth=z['true_drift'][sl,j] if row==0 else z['true_covariance'][sl,j,j];a.plot(t,truth,color='black',lw=2,ls='--',label='True')
  for m,label,col in zip(methods,names,colors):
   v=z[m+'_'+kind];a.plot(t,v[sl,j] if row==0 else v[sl,j,j],lw=1.4,color=col,label=label)
  a.set_xlabel('(t, 0)' if j==0 else '(0, t)');a.set_ylabel(f'Drift $f_{j+1}$' if row==0 else rf'Raw covariance $\Sigma_{{{j+1}{j+1}}}$');a.grid(alpha=.2)
fig.legend(*axs[0,0].get_legend_handles_labels(),loc='outside upper center',ncol=3);save(fig,'ex1_seed0_recovery')
z=np.load(E/'ex8_seed0_coefficients.npz');methods=['joint_fourier','split_fourier','arff','joint_mlp','split_mlp'];labels=['True','Joint Fourier','Split Fourier','ARFF','Joint MLP','Split MLP'];coords=z['coordinates'];extent=[-2,2,-2,2]
def panel(ax,a,norm,cmap):
 im=ax.imshow(a.reshape(100,100),origin='lower',extent=extent,cmap=cmap,norm=norm,interpolation='nearest');ax.set_xticks([-2,0,2]);ax.set_yticks([-2,0,2]);ax.set_xlabel('$x_1$');ax.set_ylabel('$x_2$');return im
fig,axs=plt.subplots(4,6,figsize=(15,10),layout='constrained')
for d in range(2):
 vals=[z['true_drift'][:,d]]+[z[m+'_drift'][:,d] for m in methods];lims=max(np.max(np.abs(v)) for v in vals);errs=[v-vals[0] for v in vals];elim=max(np.max(np.abs(v)) for v in errs)
 for row,arrays,lm in [(2*d,vals,lims),(2*d+1,errs,elim)]:
  for col,v in enumerate(arrays):
   im=panel(axs[row,col],v,TwoSlopeNorm(0,-lm,lm),'RdBu_r');axs[row,col].set_title(labels[col] if row==0 else ('True: zero error' if col==0 and row%2 else ''))
  cb=fig.colorbar(im,ax=axs[row,:],shrink=.8);cb.set_label(f'$f_{d+1}$' if row%2==0 else f'$\\widehat f_{d+1}-f_{d+1}$')
save(fig,'ex8_seed0_drift')
fig,axs=plt.subplots(3,6,figsize=(15,8),layout='constrained')
for row,(a,b) in enumerate([(0,0),(0,1),(1,1)]):
 vals=[z['true_covariance'][:,a,b]]+[z[m+'_covariance_raw'][:,a,b] for m in methods];lm=max(np.max(np.abs(v)) for v in vals)
 for col,v in enumerate(vals):
  im=panel(axs[row,col],v,TwoSlopeNorm(0,-lm,lm),'RdBu_r')
  if row==0:axs[row,col].set_title(labels[col])
 cb=fig.colorbar(im,ax=axs[row,:],shrink=.8);cb.set_label(rf'Raw $\Sigma_{{{a+1}{b+1}}}$')
save(fig,'ex8_seed0_raw_covariance')
raw=z['arff_covariance_raw'];pr=z['arff_covariance_projected'];er=np.linalg.eigvalsh(raw.astype(float))[:,0];ep=np.linalg.eigvalsh(pr.astype(float))[:,0];delta=np.linalg.norm(pr-raw,axis=(1,2))
fig,axs=plt.subplots(2,3,figsize=(10,6.5),layout='constrained')
vals=[er,ep,delta,raw[:,0,1],pr[:,0,1],(er<=0).astype(float)];titles=['Raw minimum eigenvalue','Projected minimum eigenvalue','Projection change: Frobenius norm',r'Raw $\Sigma_{12}$',r'Projected $\Sigma_{12}$','Raw nonpositive-eigenvalue mask']
for i,(ax,v,title) in enumerate(zip(axs.ravel(),vals,titles)):
 if i in [0,1]:lm=max(abs(er).max(),abs(ep).max());norm=TwoSlopeNorm(0,-lm,lm);cm='RdBu_r'
 elif i in [3,4]:lm=max(abs(vals[3]).max(),abs(vals[4]).max());norm=TwoSlopeNorm(0,-lm,lm);cm='RdBu_r'
 else:norm=plt.Normalize(0,max(float(v.max()),1e-10));cm='viridis'
 im=panel(ax,v,norm,cm);fig.colorbar(im,ax=ax,shrink=.8);ax.set_title(title)
save(fig,'ex8_seed0_projection')
(E/'GRID_DIAGNOSTICS.json').write_text(json.dumps({'scope':'illustrative 100x100 cell-centre grid, not production test metrics','seed':0,'arff_raw_nonpositive_fraction':float(np.mean(er<=0)),'arff_raw_min_eigenvalue':float(er.min()),'arff_projected_min_eigenvalue':float(ep.min()),'floor':.001,'true_min_eigenvalue':1e-8,'projection_definition':'accepted project_spd; raw coefficient maps and production RMSE unchanged'},indent=2))
