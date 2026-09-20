"""Preregister grids, rank-independent validation rules, seeds and fixed estimators."""
import sys,json,hashlib,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from src.experiments.mlp_size import matched_two_layer_width,mlp_parameter_count
OUT=ROOT/'results/capacity_regime_ex8_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 m=json.loads((ROOT/'results/controlled_study_2026/float64_v2/manifest.json').read_text())
 m['study']='Capacity-dominated regime diagnostic, exploratory; validation-only common regime selection'
 m['source_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 m['capacity']=[]
 for k in [64,256,512,1024]:
  w,count,target=matched_two_layer_width(state_dimension=2,covariance_output_dimension=3,n_frequencies=k)
  m['capacity'].append(dict(K=k,mlp_width=w,mlp_parameters=count,fourier_parameters=target,mlp_drift=mlp_parameter_count(2,2,(w,w)),mlp_factor=mlp_parameter_count(2,3,(w,w)),mismatch_percent=100*(count-target)/target))
 m['N_grid']=[80000,160000,320000,640000]
 m['h_grid']=[.000025,.00005,.0001,.0002,.0004]
 m['stage1_seeds']=m['stage2_seeds']=list(range(10))
 m['diagnostic']=dict(K=1024,N_h_diagnostic=640000,h_N_diagnostic=.0001,seeds=list(range(10)),N_selection='Smallest N with every comparison against larger N passing for every method/metric; otherwise max N with sampling-unresolved flag',stability_rule='Paired-seed 90% confidence interval for mean difference entirely inside a practical equivalence band: RMSE band max(5% of reference mean, 1 reference seed SD); NLL band max(0.05 nats/observation, 1 reference seed SD). t9=1.8331129326536335. Validation only; no between-method rankings.',h_selection='Largest lag with exact discrete drift relative bias <=0.001 and passing the same practical equivalence rule against its immediate smaller lag for every method/metric at N=640000. Compare NLL-log(h) (d=2), not raw NLL across lags. If none passes choose minimum h, explicitly unresolved; no claim of a capacity-dominated regime.',h_covariance_caveat='No exact closed-form finite-lag covariance oracle is assumed for this state-dependent rotating factor. Analytic drift oracle is exact; covariance validation stability cannot certify absence of finite-lag covariance bias.',N_h_interaction='N plateau screened at h=1e-4 only. After choosing common N*,h*, run/validate the K1024 common anchor if not already available. Do not claim joint N/h resolution from marginal diagnostics alone.',phase2='K64,256,512,1024 at frozen common N*,h*, seeds0–9. Poor results retained; phase2 descriptive if diagnostic equivalence fails. No tuning or test selection.')
 m['data_extension']=dict(distribution='unchanged iid Uniform([-2,2]^2) initial states; unchanged ex8 drift/factor; dt1e-7 float64 EM; float32 normal draws cast64; r stored64',original_rows='Preserve all100000 corrected rows and original validation/test IDs. Train ordering: original train_idx followed by appended rows in generation order.',extension_blocks=7,rows_per_block=80000,root_seed=0,namespace=20260913,keys='extension block b: split(fold_in(fold_in(PRNGKey(0),20260913),b)); first key for x0, second for noise; time block t: fold_in(noise_key,t), full shape1000x80000x2 float32. Lags share exact prefixes.',h_source='Existing validated corrected coupled lag bundle for original100000 rows; new independent training rows append only. No resampling of evaluation sets.')
 m['stage2_policy']='Exactly10 seeds throughout; no automatic30-seed extension'
 m.pop('priority_baseline',None)
 with (OUT/'manifest.json').open('x') as f:json.dump(m,f,indent=2)
 plan='''# Capacity-dominated regime diagnostic v1

Preregistered before any new fitting/test evaluation. This is exploratory work for the paper; all completed studies, the poster and manuscript remain unchanged.

Phase1N: K1024, h=1e-4, N={80000,160000,320000,640000}. Phase1h: K1024, N=640000, h={2.5e-5,5e-5,1e-4,2e-4,4e-4}. All five methods, seeds0–9. The overlapping anchor is reused after integrity checks. All training sets are deterministic prefixes of an extended iid pool. Existing10000 validation and10000 test states/IDs remain fixed; observations at each lag are Brownian-coupled.

Phase2: common frozen N*,h*, K={64,256,512,1024},10 seeds. MLP width follows the existing parameter rule. Every accepted objective, optimizer, epoch/adaptation budget, regularization, checkpoint policy, cross-fit/internal-holdout rule and training precision remains unchanged. Training arithmetic remains float32; data integration and archived increments use float64. No new calibration of ARFF modes is performed; retain accepted Metropolis-only lambda=.001, delta=.2, gamma1,300 adaptations.

Selection uses only validation coefficient metrics and NLL, never test metrics or method ranking. Exact equivalence bands and fallback rules are in manifest.diagnostic. A paired-seed90% confidence interval must fit within a band of max(5% of reference mean, one reference seed SD) for RMSE and max(.05 nats, one reference seed SD) for NLL. NLL-log(h) removes the known two-dimensional Gaussian log-h offset for lag comparisons. This does not remove all changes in transition shape. Seed SD is optimizer/initialization uncertainty on one fixed dataset, not independent-dataset sampling uncertainty. A validation plateau therefore does not prove approximation-error dominance.

Use exact discrete oracle drift f_h=((1-dt)^(h/dt)-1)x/h, compared with -x; continuous-time oracle is (exp(-h)-1)x/h. Relative bias threshold.001. The rotating diffusion has no assumed exact finite-lag covariance formula; report that limitation explicitly, and do not equate validation stability with an oracle covariance-bias bound.

If all-method equivalence cannot be demonstrated, choose the preregistered practical fallback (max N; smallest h when no lag qualifies), retain an unresolved flag, and still produce the requested complete descriptive Phase2 curve. Do not claim this establishes a capacity-dominated regime. Do not add points, tune methods, or filter valid poor results. No O(1/K) slope is imposed. Interpret optimization, finite-N, finite-h and capacity effects separately.

Data extension changes sample count only: same initial distribution/SDE/fine step. PRNG namespaces and chunk shapes are fixed in the manifest. Reuse corrected baseline/coupled-lag rows exactly, append only new training rows, preserve original row identities. ARFF actual final fitting N is .9 nominal N; each OOF drift fit uses .72N; Adam final uses N and OOF uses .8N. This difference remains explicit.

Four exclusive A6000 workers, persistent tmux, non-isolated timing. Versioned immutable plans/job paths; validate before reuse; preserve failed output, stop worker and gate later phases. Never modify numerical runners automatically. Any data/definition validity failure stops the pipeline. No poster/manuscript edits.
'''
 with (OUT/'STUDY_PLAN.md').open('x') as f:f.write(plan)
 with (OUT/'registration_sha256.json').open('x') as f:json.dump({str((OUT/n).relative_to(ROOT)):sha(OUT/n) for n in ['manifest.json','STUDY_PLAN.md']},f,indent=2)
 print(json.dumps(m['capacity'],indent=2))
if __name__=='__main__':main()
