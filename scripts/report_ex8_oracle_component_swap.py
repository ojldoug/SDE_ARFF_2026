"""Read-only numerical cross-checks and new oracle-diagnostic report/figure."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MPLBACKEND='Agg')
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[-2:])
os.nice(19)
from pathlib import Path
import json, csv, hashlib
import numpy as np
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/controlled_study_2026/float64_v2/oracle_component_swap'
SOURCE=OUT/'evaluation_v2'
HYB=OUT.parent/'hybrid_jointmlp_arff'
s=json.loads((SOURCE/'summary.json').read_text())
assert not (OUT/'report.md').exists()
with np.load(ROOT/'data/ex8_float64_v2.npz') as z:
    ix=z['test_idx'];x=z['x_data'][ix].astype(float);r=z['r_data'][ix];h=z['step_sizes'][ix].astype(float)
theta=3*np.arctan2(x[:,1],x[:,0])
weak=np.stack([np.cos(theta),np.sin(theta)],axis=1)
strong=np.stack([-np.sin(theta),np.cos(theta)],axis=1)
residual=r+h*x
rw=np.sum(weak*residual,axis=1);rs=np.sum(strong*residual,axis=1)
qweak=.5*rw**2/(h[:,0]*1e-8);qstrong=.5*rs**2/h[:,0]
analytic=float(np.mean(qweak+qstrong)+np.log(2*np.pi)+np.mean(np.log(h[:,0]))+.5*np.log(1e-8))
cov=1e-8*weak[:,:,None]*weak[:,None,:]+strong[:,:,None]*strong[:,None,:]
v=cov*h[:,:,None]
sg,ld=np.linalg.slogdet(v);assert (sg>0).all()
direct=float(np.mean(.5*(np.sum(residual*np.linalg.solve(v,residual[:,:,None])[:,:,0],axis=1)+ld)+np.log(2*np.pi)))
np.testing.assert_allclose([analytic,direct],s['oracle']['nll'],rtol=1e-8,atol=1e-5)
oracle_checks=dict(analytic_rotation_nll=analytic, independent_matrix_solve_nll=direct,
    weak_quadratic_mean=float(qweak.mean()),strong_quadratic_mean=float(qstrong.mean()),
    weak_residual_second_moment_over_h=float(np.mean(rw**2/h[:,0])),
    nominal_weak_covariance=1e-8, weak_quadratic_top_one_percent_share=float(np.sort(qweak)[-100:].sum()/qweak.sum()),
    weak_quadratic_quantiles={str(p):float(np.quantile(qweak,p)) for p in [0,.5,.9,.95,.99,1]},
    h_min=float(h.min()),h_max=float(h.max()),test_count=len(ix))
s['independent_oracle_checks']=oracle_checks
projection=list(csv.DictReader((SOURCE/'projection_per_seed.csv').open()))
hybrid=list(csv.DictReader((HYB/'per_seed.csv').open()))
for seed,row in enumerate(projection):
    assert int(row['seed'])==int(hybrid[seed]['seed'])==seed
    for key in ['raw_spd_violation_rate','min_raw_eigenvalue','cpu_raw_spd_violation_rate','cpu_min_raw_eigenvalue']:
        row['prior_hybrid_'+key]=hybrid[seed][key]
s['source_files_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.iterdir() if p.is_file()}
s['final_report_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
with (OUT/'summary.json').open('x') as f:json.dump(s,f,indent=2,allow_nan=False)
for name in ['per_seed.csv','floor_sensitivity_per_seed.csv']:
    with (OUT/name).open('xb') as f:f.write((SOURCE/name).read_bytes())
with (OUT/'projection_per_seed.csv').open('x',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(projection[0]));w.writeheader();w.writerows(projection)

def fmt(v):return f"{v['mean']:.8g} ± {v['sd']:.5g}"
def table(scope):
    out=['| Components | Quadratic | Half log determinant | Total NLL |','|---|---:|---:|---:|']
    for key,value in s['combinations'].items():
        if key.startswith(scope+'/'):
            name=key.split('/')[1]
            out.append(f"| {name} | {fmt(value['quadratic'])} | {fmt(value['logdet'])} | {fmt(value['nll'])} |")
    return out

lines=['# Oracle component swaps: corrected Experiment 8','',
'All 30 existing seeds, paired by identity, evaluated on the unchanged corrected float64-v2 10,000-point test split. No fitting, tuning, selection, campaign control, or changes to existing models, data, figures or manuscript. This is a post-hoc diagnostic, not a new estimator. Sample SD describes seed variability on one fixed dataset; it does not describe dataset uncertainty.','',
'## Main finding','',
'ARFF is chiefly covariance/calibration-limited for the accepted Gaussian score. Replacing its drift with true drift or Joint MLP drift changes NLL little. The accepted floor is protective on these observations: lowering it makes NLL much worse, despite barely changing covariance RMSE. The instantaneous true covariance is not the finite-lag transition covariance, so the unprojected “oracle” Gaussian is not a best achievable NLL reference. This distinction is especially consequential at eigenvalue 1e-8.','',
'## Likelihood and precision','',
'Every score uses residual e = r − h f and covariance h Sigma, mean over test rows, no dimension normalization: NLL = mean(0.5 eᵀ(h Sigma)⁻¹e) + mean(0.5 log det(h Sigma)) + log(2π). The Gaussian constant is 1.8378770664 nats (1.83787704 in native float32). Covariance RMSE is sqrt(mean over all rows and all four matrix entries of squared error). Raw RMSE never projects.','',
'ARFF controls use unchanged project_spd: symmetrize, eigendecompose, floor eigenvalues at 1e-3, reconstruct, then the accepted float32 solve/logdet/reduction. The hybrid total is reused exactly from the prior CPU diagnostic and checked against the decomposition; all 30 original ARFF totals also exactly match the prior same-CPU evaluation. Component predictions were not changed.','',
'Unprojected oracle and predetermined floor sensitivity use float64 spectral likelihood arithmetic and archived float64 increments to resolve the 1e-8 eigenvalue. This is an explicitly separate evaluation-precision diagnostic, not a production change. All learned predictions remain the original float32 predictions. Native minus float64 hybrid NLL is '+fmt(s['paired_differences']['native_minus_float64_hybrid'])+'; this precision difference cannot explain the findings. The true covariance is unprojected in the requested oracle swaps; projected-true-covariance controls below isolate applying the production floor to truth.','',
'## Requested decomposition','',
'Native accepted ARFF floor, mean ± sample SD over 30 seeds:','']+table('native_float32_floor_1e-3')+['',
'Unprojected true covariance, stable float64. Oracle is one evaluation, repeated in the CSV only for paired differences; its zero across-seed SD is not an uncertainty estimate.','']+table('float64_unprojected_true_covariance')+['',
'Float64 accepted-floor controls for comparisons with the float64 oracle:','']+table('float64_floor_1e-3')+['',
'## Answers to the six interpretation questions','',
'1. **True covariance, learned drifts:** ARFF minus Joint MLP NLL is '+fmt(s['paired_differences']['arff_minus_joint_mlp_drift_under_true_covariance'])+' nats. The log determinant is identical; the entire difference is covariance-weighted residual error. These large values are amplified by the 1e-8 weak direction and should not be mistaken for comparable production-NLL differences.',
'2. **True drift, ARFF covariance:** NLL remains −8.916273 ± 0.074339. Relative to the unprojected instantaneous oracle it is about 16,026 nats lower, not a positive penalty. The meaningful floor-matched true-covariance control is −10.178364: learned ARFF covariance therefore adds about 1.262090 ± 0.074339 nats at the same floor and true drift. Of this, approximately 0.240427 is quadratic and 1.021664 is half-log-determinant. This is a descriptive reference contrast, not a unique causal allocation.',
'3. **Replacing ARFF drift:** Joint MLP reduces native ARFF NLL by 0.018816 ± 0.008209; all 30 seeds improve. Quadratic contribution changes 0.907441 → 0.888624, while log determinant is exactly unchanged. True drift gives 0.888404. Thus the hybrid is already close to true-drift performance when covariance is held to ARFF. The positive eigenvalue floor limits amplification of drift errors relative to using true covariance.',
'4. **Projection and RMSE:** raw → projected covariance RMSE is '+fmt(s['projection']['raw_covariance_rmse'])+' → '+fmt(s['projection']['projected_covariance_rmse'])+'. Paired change is '+fmt(s['paired_differences']['projection_rmse_change'])+' (about 1.07% lower). A small matrix-entry RMSE change coexists with a large likelihood change because inversion is sensitive to small eigenvalues.',
'5. **The 1e-3 floor:** on true covariance it raises the small eigenvalue 100,000-fold (additive change 0.00099999; weak-direction standard deviation rises sqrt(100000) ≈ 316.23-fold). Entrywise covariance RMSE introduced is exactly 0.000499995. Half-log-determinant rises by 0.5 log(100000) = 5.756463 nats, but the observed quadratic term drops by far more. On ARFF, lowering the floor to 1e-8 makes NLL tens of thousands, rather than recovering the MLP advantage. No positive fraction of ARFF’s disadvantage can be assigned to the accepted floor from these swaps; learned eigenvalues/orientations, finite-lag residual calibration and projection interact. Raw indefinite covariance has no valid unprojected Gaussian NLL.',
'6. **Classification:** covariance/calibration-limited under the accepted evaluation, with strong projection sensitivity; not primarily drift-limited. It is not defensible to call it solely projection-limited or assert that matching the instantaneous minimum eigenvalue would repair it. Better raw coefficient RMSE is insufficient for a calibrated transition likelihood.','',
'## Why the unprojected instantaneous oracle is large','',
f"Independent NumPy rotation-coordinate evaluation gives NLL {analytic:.9f}; independent float64 matrix solve gives {direct:.9f}. Both agree with the repository-definition spectral evaluation to relative tolerance 1e-8. Weak/strong directional quadratic means are {qweak.mean():.9f} and {qstrong.mean():.9f}. Mean weak residual squared divided by h is {oracle_checks['weak_residual_second_moment_over_h']:.9g}, versus instantaneous weak covariance 1e-8. The top 1% of test rows account for {100*oracle_checks['weak_quadratic_top_one_percent_share']:.3f}% of the weak quadratic term; no points are removed.",'',
'The dataset contains endpoints after 1000 fine EM steps, not a single Gaussian Euler increment frozen at x0. The diffusion eigendirections rotate with state (theta = 3 atan2(x2,x1)); variation during the lag can populate the initially weak direction. Hence h Sigma(x0) need not calibrate the finite-lag residual, particularly near singularity. These checks establish the score and the instantaneous-versus-transition distinction; they do not identify an exact finite-lag covariance or apportion all mismatch between finite-lag dynamics, discretization and sample variation. No dataset or estimator correction is made or inferred from this diagnostic.','',
'## Projection diagnostics','',
'Statistics below use float64 eigenanalysis of unchanged archived CPU raw predictions. Frozen GPU and prior CPU SPD statistics are additionally retained in projection_per_seed.csv; small backend differences documented by the hybrid study remain, and no historical values are overwritten. All 30 seeds contain violations.','',
'| Statistic | Mean ± sample SD | Median | Range across seeds |','|---|---:|---:|---:|']
for key in ['raw_spd_violation_rate','min_raw_eigenvalue','eigenvalue_fraction_changed','eigenvalue_change_mean','changed_only_mean']:
    v=s['projection'][key];lines.append(f"| {key} | {fmt(v)} | {v['median']:.8g} | [{v['min']:.8g}, {v['max']:.8g}] |")
lines+=['','Eigenvalue-change quantiles are computed over 20,000 individual eigenvalues within each seed, then summarized across seeds. Zeros are included in the all-eigenvalue distribution. Changed-only quantiles condition on a positive change.','',
'| Quantile | All eigenvalues: mean ± SD of quantile | Changed only: mean ± SD of quantile |','|---|---:|---:|']
for p in [0,.25,.5,.75,.9,.95,.99,1]:
    lines.append(f"| {p:g} | {fmt(s['projection'][f'change_q{p:g}'])} | {fmt(s['projection'][f'changed_only_q{p:g}'])} |")
lines+=['','## Predetermined floor sensitivity — not tuning','',
'Every grid point is reported, with no selected floor. The accepted result remains epsilon = 1e-3. These use float64 likelihood arithmetic.','',
'| Floor | ARFF drift + ARFF covariance NLL | True drift + ARFF covariance NLL | Hybrid NLL | Projected covariance RMSE |','|---|---:|---:|---:|---:|']
for floor,v in s['floor_sensitivity'].items():
    lines.append(f"| {floor} | {fmt(v['arff_original']['nll'])} | {fmt(v['true_drift_arff_covariance']['nll'])} | {fmt(v['hybrid']['nll'])} | {fmt(v['arff_original']['projected_covariance_rmse'])} |")
lines+=['','True drift + true covariance floor controls (one fixed test set, no seed SD):','',
'| Floor | Quadratic | Half log determinant | NLL | Covariance RMSE introduced |','|---|---:|---:|---:|---:|']
for floor,v in s['true_covariance_floor_controls'].items():
    lines.append(f"| {floor} | {v['quadratic']:.9g} | {v['logdet']:.9g} | {v['nll']:.9g} | {v['projected_covariance_rmse']:.9g} |")
lines+=['','## Provenance and checks','',
'Dataset: data/ex8_float64_v2.npz; SHA-256 '+s['dataset_sha256']+'. Test-index SHA-256 '+s['test_idx_sha256']+'. All 30 paired source artifacts and the hybrid directory passed hash checks, and protected historical data/model/report/figure/manuscript hashes remain unchanged. Baseline source hashes and protocol are archived in evaluation_v2/protocol.json.','',
'The first diagnostic attempt stopped before swaps because this installed JAX exposes jax.enable_x64 rather than jax.experimental.enable_x64. Its source/protocol/log are preserved. Only the new diagnostic was corrected; no accepted numerical runner was edited. AST and a Gaussian matrix-versus-spectral fixture passed; all 30 native hybrid and ARFF NLL integrity checks passed. No training functions, GPU work or campaign control operations were invoked.','',
'Files: per_seed.csv (decompositions), projection_per_seed.csv (including prior native SPD diagnostics), floor_sensitivity_per_seed.csv, summary.json, diagnostic.pdf and diagnostic.png. No poster or manuscript changes; no further jobs launched.']
with (OUT/'report.md').open('x') as f:f.write('\n'.join(lines)+'\n')
fig,ax=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
floor=np.array([float(k) for k in s['floor_sensitivity']])
for method,label in [('arff_original','ARFF drift'),('hybrid','Joint MLP drift'),('true_drift_arff_covariance','True drift')]:
    vals=[s['floor_sensitivity'][str(e)][method]['nll'] for e in floor]
    ax[0].errorbar(floor,[v['mean'] for v in vals],yerr=[v['sd'] for v in vals],marker='o',label=label,capsize=3)
ax[0].set(xscale='log',yscale='symlog',xlabel='Predetermined eigenvalue floor',ylabel='Gaussian NLL (symlog)',title='Fixed ARFF covariance; no floor selected')
ax[0].axvline(.001,color='gray',ls=':',label='Accepted floor');ax[0].legend(fontsize=8)
names=['ARFF drift','Joint MLP drift','True drift'];keys=['arff_original','hybrid','true_drift_arff_covariance']
v=[s['combinations']['native_float32_floor_1e-3/'+k]['quadratic'] for k in keys]
ax[1].bar(names,[t['mean'] for t in v],yerr=[t['sd'] for t in v],capsize=4,color=['#d69f41','#5396ba','#779e69'])
ax[1].set(ylabel='Mean quadratic contribution (nats)',title='Accepted floor: same log determinant')
fig.suptitle('Experiment 8: post-hoc oracle component swaps')
for ext in ['pdf','png']:
    with (OUT/('diagnostic.'+ext)).open('xb') as f:fig.savefig(f,format=ext,dpi=180)
plt.close(fig)
print(json.dumps(oracle_checks,indent=2))
print('Report and diagnostic figures saved:',OUT)
