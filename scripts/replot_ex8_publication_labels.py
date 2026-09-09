#!/usr/bin/env python3
"""Label-only rendering of the frozen capacity-matched Experiment8 observations."""
import os
os.environ['JAX_PLATFORMS']='cpu';os.environ['MPLBACKEND']='Agg'
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib.pyplot as plt
import summarize_ex8_capacity_matched as accepted

def main():
    records,hashes=accepted.collect()
    out=accepted.OUT;stem='ex8_capacity_matched_w27_rmse_two_panel_publication_labels'
    targets=[out/(stem+'.'+ext) for ext in ['pdf','png','json']]
    if any(p.exists() for p in targets):raise FileExistsError('Publication-label output already exists')
    with np.load(out/'test_distributions.npz',allow_pickle=False) as saved:
        for method in accepted.LABELS:
            for metric in accepted.METRICS:
                values=np.array([r[metric] for r in records if r['method']==method])
                np.testing.assert_array_equal(values,saved[method+'_'+metric])
    labels=['Joint Fourier Adam\n1,792 parameters','Split Fourier Adam\n1,792 parameters','ARFF\n1,792 parameters','Joint MLP Adam\n1,814 parameters']
    colors=['#4477AA','#66CCEE','#228833','#AA3377']
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.4))
    for ax,metric,logscale in [(axes[0],'covariance_rmse',True),(axes[1],'drift_rmse',False)]:
        data=[np.array([r[metric] for r in records if r['method']==m]) for m in accepted.LABELS]
        boxes=ax.boxplot(data,tick_labels=labels,showfliers=False,patch_artist=True,medianprops={'color':'black','linewidth':1.6})
        for box,color in zip(boxes['boxes'],colors):box.set_facecolor(color);box.set_alpha(.25)
        rng=np.random.default_rng(2026)
        for i,(values,color) in enumerate(zip(data,colors),1):
            ax.scatter(i+rng.normal(0,.045,len(values)),values,s=19,alpha=.65,color=color,zorder=3)
            ax.scatter(i,values.mean(),marker='D',s=48,color=color,edgecolor='black',zorder=4)
        if logscale:ax.set_yscale('log')
        ax.set_ylabel('Test '+('covariance RMSE' if metric=='covariance_rmse' else 'drift RMSE'))
        ax.grid(axis='y',alpha=.25);ax.tick_params(axis='x',labelsize=9)
    axes[0].set_title('(a) Covariance');axes[1].set_title('(b) Drift')
    fig.suptitle('Experiment 8: capacity-matched comparison',fontsize=14)
    fig.text(.5,.015,'30 seeds per method; points = runs, diamonds = means, boxes = quartiles and median. Fourier: 1792 parameters; MLP: 1814.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    for ext in ['pdf','png']:
        with (out/(stem+'.'+ext)).open('xb') as f:fig.savefig(f,format=ext,dpi=300,bbox_inches='tight')
    plt.close(fig)
    with targets[2].open('x') as f:json.dump(dict(source_sha256=hashes,distributions_sha256=hashlib.sha256((out/'test_distributions.npz').read_bytes()).hexdigest(),validation='All120 accepted runs validated; all12 metric vectors exactly equal frozen data',changes='X-axis labels only; data/statistics/colors/jitter/panel order/scales/mean/box rendering unchanged',outputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in targets[:2]}),f,indent=2)
    print('\n'.join(map(str,targets)))
if __name__=='__main__':main()
