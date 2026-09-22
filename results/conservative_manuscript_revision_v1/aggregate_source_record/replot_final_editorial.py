from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
O=Path(__file__).resolve().parents[1]
A=O/'archive_inputs'
methods=['joint_fourier','split_fourier','arff','joint_mlp','split_mlp']
names=dict(zip(methods,['Joint Fourier','Split Fourier','ARFF','Joint MLP','Split MLP']))
brows=list(csv.DictReader((A/'baseline_per_seed.csv').open()))
sens=list(csv.DictReader((A/'sensitivity_metrics.csv').open()))
for r in sens:
 for k in ['point','mean','sd']:r[k]=float(r[k])
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
colors=['#0072B2','#56B4E9','#009E73','#D55E00','#CC79A7']
fig,axs=plt.subplots(1,2,figsize=(10.3,4.0))
for ax,metric,title in zip(axs,['drift_rmse','covariance_rmse'],['Drift RMSE','Raw covariance RMSE']):
 data=[[float(x[metric]) for x in brows if x['method']==m and x['split']=='test'] for m in methods]
 bp=ax.boxplot(data,patch_artist=True,showfliers=False,widths=.58)
 for j,(a,col) in enumerate(zip(data,colors)):
  bp['boxes'][j].set(facecolor=col,alpha=.2);ax.scatter(j+1+np.linspace(-.15,.15,len(a)),a,s=8,color=col,alpha=.65);ax.scatter(j+1,np.mean(a),marker='D',color='black',s=22,zorder=4)
 ax.set_xticks(range(1,6),['Joint\nFourier\n1,792','Split\nFourier\n1,792','ARFF\n1,792','Joint\nMLP\n1,814','Split\nMLP\n1,814']);ax.set_ylabel(title);ax.grid(axis='y',alpha=.2)
fig.suptitle('Corrected Experiment 8: capacity-matched comparison');fig.text(.5,.015,'30 fitting seeds; fixed test set. Boxes: quartiles/median; points: seeds; diamonds: means.\nCounts are final parameters.',ha='center',fontsize=10);fig.tight_layout(rect=(0,.075,1,.94))
for ext in ['pdf','png']:fig.savefig(O/f'figures/baseline.{ext}',dpi=200)
plt.close(fig)
for study,xlabel in [('h','Observation lag h'),('capacity','Fourier width K (matched MLP widths in Table S1)'),('N','Nominal training observations N')]:
 third='nll_minus_log_h' if study=='h' else 'nll'
 available={x['metric'] for x in sens if x['study']==study}
 if third not in available and study=='h':third=next(k for k in available if k not in ['drift_rmse','covariance_rmse','nll'])
 fig,axs=plt.subplots(1,3,figsize=(10.3,3.5))
 for ax,metric,title in zip(axs,['drift_rmse','covariance_rmse',third],['Drift RMSE','Raw covariance RMSE','NLL − log h' if study=='h' else 'Gaussian NLL']):
  for m,col in zip(methods,colors):
   z=sorted([x for x in sens if x['study']==study and x['method']==m and x['metric']==metric],key=lambda x:x['point']);assert len(z)==({'h':9,'capacity':5,'N':4}[study])
   ax.errorbar([x['point'] for x in z],[x['mean'] for x in z],yerr=[x['sd'] for x in z],marker='o',markersize=3,lw=1,capsize=2,color=col,label=names[m])
  ax.set_xscale('log');ax.set_ylabel(title);ax.set_xlabel(xlabel if study!='capacity' else 'Fourier width K');ax.grid(alpha=.18)
  if metric!='nll' and metric!=third:ax.set_yscale('log')
  ax.xaxis.set_minor_formatter(NullFormatter())
  if study=='N':ax.set_xticks([80000,160000,320000,640000],['80k','160k','320k','640k'])
  if study=='capacity':ax.set_xticks([64,128,256,512,1024],['64','128','256','512','1024'])
 handles,labs=axs[0].get_legend_handles_labels();fig.legend(handles,labs,loc='upper center',ncol=3,frameon=False,fontsize=10);fig.tight_layout(rect=(0,0,1,.84))
 for ext in ['pdf','png']:fig.savefig(O/f'figures/{study}.{ext}',dpi=200)
 plt.close(fig)
