"""Replot only the recorded surrogate trajectories packaged in evidence/."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size':11, 'pdf.fonttype':42})
O=Path(__file__).resolve().parents[1]
A=O/'archive_inputs'
selected={(int(x['repeat']),x['arm']):int(x['selected_iteration']) for x in csv.DictReader((A/'mechanism_selections.csv').open())}
fig,axs=plt.subplots(3,3,figsize=(10,6.6),sharex=True,sharey='row')
for rep in range(3):
 titles=[]
 for arm,col in [('M','#2166ac'),('MR','#b35806')]:
  r=list(csv.DictReader((A/f'mechanism_{rep}_{arm}.csv').open()));it=selected[rep,arm];titles.append(f'{arm}: {it}')
  for j,k in enumerate(['training_noisy_mse','internal_noisy_mse','post_frequency_rms']):
   y=np.array([float(t[k]) for t in r]);a=axs[j,rep];a.plot(np.arange(1,301),y,color=col,lw=.8,label=arm);a.axvline(it,color=col,ls='--',lw=.7);a.scatter(it,y[it-1],color=col,s=18);a.grid(alpha=.18);a.set_xlim(0,300)
 axs[0,rep].set_title(f'Repeat {rep}\nselected '+', '.join(titles),fontsize=11);axs[-1,rep].set_xlabel('Recorded iteration')
for j,name in enumerate(['Training noisy MSE','Held-out noisy MSE','Frequency RMS']):axs[j,0].set_ylabel(name)
h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=2,frameon=False);fig.tight_layout(rect=(0,.065,1,.94));fig.text(.5,.012,'M: Metropolis; MR: Metropolis + resampling each iteration.\nDashed lines/dots: selected checkpoints; no smoothing.',ha='center',fontsize=10)
for ext in ['pdf','png']:fig.savefig(O/f'figures/mechanism.{ext}',dpi=200)
