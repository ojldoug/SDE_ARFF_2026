#!/usr/bin/env python3
"""Validate completed K=128 campaigns and create new summary/figure artifacts.

Reads recorded endpoint metrics; performs no fitting or model selection.
"""
from pathlib import Path
import csv
import json
import os
import re
import sys

os.environ.setdefault('JAX_PLATFORMS','cpu')
os.environ.setdefault('MPLBACKEND','Agg')
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_ex8_k128_campaigns as campaign
import run_ex8_mlp_w27_campaign as mlp_campaign
import run_ex8_campaigns as previous
import run_production_batch as batch

OUT=ROOT/'results/capacity_matched_ex8_w27'
LABELS={'joint':'Joint Fourier Adam K=128','split':'Split Fourier Adam K=128',
        'arff':'Corrected ARFF K=128','mlp':'Joint MLP Adam 27×27 — 1814 parameters'}
METRICS=('drift_rmse','covariance_rmse','nll')


def parse_test_log(path):
    text=path.read_text()
    block=re.search(r'^test\s*\n(.*?)(?=^artifact\s*:|\Z)',text,re.M|re.S)
    if block is None:raise ValueError(f'Missing final test block: {path}')
    values={}
    for key,label in [('drift_rmse','drift RMSE'),('covariance_rmse','covariance RMSE'),('nll','NLL')]:
        match=re.search(r'^\s*'+label+r'\s*:\s*([-+\d.eE]+)\s*$',block[1],re.M)
        if match is None:raise ValueError(f'Missing {key}: {path}')
        values[key]=float(match[1])
    if not np.all(np.isfinite(list(values.values()))):raise ValueError('Nonfinite test metrics')
    return values


def collect():
    campaign.verify_snapshot()
    records=[];provenance={}
    mlp_campaign.verify_snapshot()
    for method in LABELS:
        for seed in range(30):
            if method in ('joint','split'):
                artifact,log=campaign.paths(method,seed)
                campaign.validate_pair(method,seed,artifact,log)
            elif method=='arff':
                artifact,log=previous.paths('arff',seed)
                previous.validate_pair('arff',seed,artifact,log)
            else:
                artifact,log=mlp_campaign.paths('mlp',seed)
                mlp_campaign.validate_pair('mlp',seed,artifact,log)
            with np.load(artifact,allow_pickle=False) as z:
                if method=='joint':
                    values=parse_test_log(log)
                elif method=='mlp':
                    values=parse_test_log(log)
                else:
                    values={k:float(z['test_'+k]) for k in METRICS}
                records.append(dict(method=method,label=LABELS[method],seed=seed,**values,
                                     algorithm_time=float(z['algorithm_time'])))
            for p in (artifact,log):provenance[str(p.relative_to(ROOT))]=campaign.digest(p)
    return records,provenance


def describe(values):
    x=np.asarray(values,dtype=float)
    if x.shape!=(30,) or not np.all(np.isfinite(x)):raise ValueError('Expected 30 finite values')
    return dict(n=30,mean=float(x.mean()),sd=float(x.std(ddof=1)),median=float(np.median(x)),
                q25=float(np.quantile(x,.25)),q75=float(np.quantile(x,.75)),min=float(x.min()),max=float(x.max()))


def figures(records,out):
    import matplotlib.pyplot as plt
    labels=['Joint Fourier Adam\nK=128 · 1792 params','Split Fourier Adam\nK=128 · 1792 params','Corrected ARFF\nK=128 · 1792 params',
            'Joint MLP Adam\n27×27 · 1814 params']
    colors=['#4477AA','#66CCEE','#228833','#AA3377']
    def panel(ax,metric,logscale):
        data=[np.array([r[metric] for r in records if r['method']==m]) for m in LABELS]
        boxes=ax.boxplot(data,tick_labels=labels,showfliers=False,patch_artist=True,
                         medianprops={'color':'black','linewidth':1.6})
        for box,color in zip(boxes['boxes'],colors):box.set_facecolor(color);box.set_alpha(.25)
        rng=np.random.default_rng(2026)
        for i,(values,color) in enumerate(zip(data,colors),1):
            ax.scatter(i+rng.normal(0,.045,len(values)),values,s=19,alpha=.65,color=color,zorder=3)
            ax.scatter(i,values.mean(),marker='D',s=48,color=color,edgecolor='black',zorder=4)
        if logscale:ax.set_yscale('log')
        ax.set_ylabel('Test '+('covariance RMSE' if metric=='covariance_rmse' else 'drift RMSE'))
        ax.grid(axis='y',alpha=.25)
        ax.tick_params(axis='x',labelsize=9)
    def save(fig,name):
        for ext in ('pdf','png'):
            p=out/(name+'.'+ext)
            with p.open('xb') as handle:fig.savefig(handle,format=ext,dpi=300,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13.5,5.4))
    panel(axes[0],'covariance_rmse',True);panel(axes[1],'drift_rmse',False)
    axes[0].set_title('(a) Covariance');axes[1].set_title('(b) Drift')
    fig.suptitle('Experiment 8: capacity-matched comparison',fontsize=14)
    fig.text(.5,.015,'30 seeds per method; points = runs, diamonds = means, boxes = quartiles and median. Fourier: 1792 parameters; MLP: 1814.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    save(fig,'ex8_capacity_matched_w27_rmse_two_panel')
    fig,ax=plt.subplots(figsize=(8.6,5.6))
    panel(ax,'covariance_rmse',True)
    ax.set_title('Experiment 8: covariance recovery at matched parameter count')
    fig.text(.5,.015,'30 seeds per method; Fourier: 1792 parameters; MLP: 1814.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1))
    save(fig,'ex8_capacity_matched_w27_covariance_rmse_distribution')


def main():
    records,provenance=collect()
    summary={m:{k:describe([r[k] for r in records if r['method']==m]) for k in METRICS} for m in LABELS}
    OUT.mkdir(exist_ok=False)  # Never replace an earlier report or figure.
    with (OUT/'per_seed_test_metrics.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    with (OUT/'summary.json').open('x') as handle:
        json.dump(dict(labels=LABELS,statistics=summary,source_sha256=provenance,
            timing_note='Fourier, MLP and prior ARFF accuracy campaigns used concurrent GPU jobs. Timings are non-isolated.',
            metrics_note='Joint Fourier and MLP endpoint metrics from unchanged runner logs (8-digit scientific notation); Split/ARFF from artifacts.',
            interpretation='Fourier final models have 1792 parameters; MLP has 1814. Objectives, validation use and factor-vs-direct covariance definitions remain different.'),handle,indent=2)
    arrays={m+'_'+k:np.asarray([r[k] for r in records if r['method']==m]) for m in LABELS for k in METRICS}
    with (OUT/'test_distributions.npz').open('xb') as handle:np.savez_compressed(handle,seeds=np.arange(30),**arrays)
    lines=['Experiment 8 — capacity-matched comparison','', 'All 30 width-27 MLP artifacts, 60 K=128 Fourier artifacts and 30 corrected ARFF artifacts validated; 30 seeds per method.', '']
    for m in LABELS:
        lines.append(LABELS[m])
        for metric,stats in summary[m].items():
            lines.append(f"  {metric}: mean={stats['mean']:.8e}, SD={stats['sd']:.8e}, median={stats['median']:.8e}, range=[{stats['min']:.8e}, {stats['max']:.8e}]")
        lines.append('')
    (OUT/'summary.txt').write_text('\n'.join(lines)+'\n')
    figures(records,OUT)
    print('\n'.join(lines),flush=True)
    print('Saved summary and new figures:',OUT,flush=True)


if __name__=='__main__':main()
