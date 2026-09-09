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
import run_ex8_campaigns as previous
import run_production_batch as batch

OUT=ROOT/'results/width_controlled_ex8_k128'
LABELS={'joint':'Joint Fourier Adam K=128','split':'Split Fourier Adam K=128',
        'arff':'Corrected ARFF K=128','mlp':'Joint MLP Adam 57×57 (separate expressive baseline)'}
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
    mlp_summary=ROOT/'results/ex8_joint_production_summary.npz'
    with np.load(mlp_summary,allow_pickle=False) as z:
        np.testing.assert_array_equal(z['mlp_seed'],np.arange(30))
        mlp_values={k:np.asarray(z['mlp_'+k]) for k in METRICS}
    provenance[str(mlp_summary.relative_to(ROOT))]=campaign.digest(mlp_summary)
    for method in LABELS:
        for seed in range(30):
            if method in ('joint','split'):
                artifact,log=campaign.paths(method,seed)
                campaign.validate_pair(method,seed,artifact,log)
            elif method=='arff':
                artifact,log=previous.paths('arff',seed)
                previous.validate_pair('arff',seed,artifact,log)
            else:
                directory=ROOT/'results/production/mlp_ex8'
                artifact,log=directory/f'seed_{seed}_artifacts.npz',directory/f'seed_{seed}.txt'
                if not batch.seed_is_complete(log_path=log,artifact_path=artifact,method='mlp',experiment='ex8',seed=seed):
                    raise ValueError(f'Invalid MLP baseline seed {seed}')
            with np.load(artifact,allow_pickle=False) as z:
                if method=='joint':
                    values=parse_test_log(log)
                elif method=='mlp':
                    if int(z['hidden_width'])!=57 or int(z['hidden_layers'])!=2 or int(z['mlp_parameter_count'])!=7244:
                        raise ValueError('MLP baseline capacity changed')
                    values={k:float(mlp_values[k][seed]) for k in METRICS}
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
    labels=['Joint Fourier\nAdam K=128','Split Fourier\nAdam K=128','Corrected ARFF\nK=128',
            'Joint MLP 57×57\nexpressive baseline']
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
    fig.suptitle('Experiment 8: width-controlled Fourier comparison (K=128)',fontsize=14)
    fig.text(.5,.015,'30 seeds per method; points = runs, diamonds = means, boxes = quartiles and median. MLP is a separate expressive baseline.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    save(fig,'ex8_k128_rmse_two_panel')
    fig,ax=plt.subplots(figsize=(8.6,5.6))
    panel(ax,'covariance_rmse',True)
    ax.set_title('Experiment 8: covariance recovery at matched Fourier width')
    fig.text(.5,.015,'30 seeds per method; MLP remains a separate expressive baseline.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,1))
    save(fig,'ex8_k128_covariance_rmse_distribution')


def main():
    records,provenance=collect()
    summary={m:{k:describe([r[k] for r in records if r['method']==m]) for k in METRICS} for m in LABELS}
    OUT.mkdir(exist_ok=False)  # Never replace an earlier report or figure.
    with (OUT/'per_seed_test_metrics.csv').open('x',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    with (OUT/'summary.json').open('x') as handle:
        json.dump(dict(labels=LABELS,statistics=summary,source_sha256=provenance,
            timing_note='New Fourier and prior ARFF campaign timings are non-isolated; do not compare with isolated MLP timings.',
            metrics_note='Joint Fourier endpoint metrics from unchanged runner logs (8-digit scientific notation); Split/ARFF from artifacts; MLP from existing authoritative full-precision summary.',
            interpretation='Fourier final models have 1792 parameters; MLP has 7244. Objectives, validation use and factor-vs-direct covariance definitions remain different.'),handle,indent=2)
    arrays={m+'_'+k:np.asarray([r[k] for r in records if r['method']==m]) for m in LABELS for k in METRICS}
    with (OUT/'test_distributions.npz').open('xb') as handle:np.savez_compressed(handle,seeds=np.arange(30),**arrays)
    lines=['Experiment 8 — width-controlled comparison','', 'All 60 new Fourier artifacts validated at K=128; 30 seeds per method.', '']
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
