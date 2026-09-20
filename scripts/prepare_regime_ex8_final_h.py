"""Register the final bounded amendment; do not touch parent study state."""
import os
os.environ['JAX_PLATFORMS']='cpu'
import sys,json,datetime,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from oracle_regime_ex8 import oracle
import run_regime_ex8_final_h_campaign as s
def main():
 out=s.STUDY;parent=s.PARENT;assert not out.exists()
 m=json.loads((parent/'manifest.json').read_text());selected=json.loads((parent/'selection.json').read_text())
 assert selected['N_star']==640000 and selected['h_star']==.0001
 threshold=float(re.search(r'bias <=([\d.]+)',m['diagnostic']['h_selection'])[1]);assert threshold==.001
 screening=[dict(**oracle(h),included=oracle(h)['relative_discrete_drift_bias']<=threshold) for h in [.0015,.002]]
 included=[v['h'] for v in screening if v['included']]
 assert included==[.0015,.002],'Screening result differs from independent arithmetic; stop before registration'
 preserved={str((parent/n).relative_to(ROOT)):s.r.adapter.digest(parent/n) for n in ['manifest.json','registration_sha256.json','STUDY_PLAN.md','selection.json','validation_selection_inputs.json','data/complete.json','oracle_moments.json','h_summary/summary.json','h_summary/per_seed_metrics.csv','h_summary/report.md']}
 m['registered_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();m['h_grid']+=included
 m['lag_dataset']['steps']=[round(h/1e-7) for h in m['h_grid']]
 m['amendment']=dict(version='final_h_v1',parent=str(parent.relative_to(ROOT)),included_h=included,drift_bias_threshold=threshold,oracle_screening=screening,
  label='Final oracle-criterion-driven descriptive extension; not a search for ARFF optimal lag',
  hard_maximum_h=.002,further_extensions_allowed=False,selection_allowed=False,
  frozen_phase2=dict(N_star=640000,h_star=.0001,selection_sha256=s.r.adapter.digest(parent/'selection.json'),selection_reinterpreted=False),
  parent_sha256=preserved,phase2_priority='Wait until parent pipeline completes including final capacity summary, then wait for GPU idle; never control parent jobs',
  coupling='Replay identical initial states and full normal draw shapes; verify all ten prior noise-block hashes and 10000-step endpoints bit-for-bit; append time blocks10–19 with the existing fold_in rule',
  precision='Fine EM dt1e-7 float64 states and endpoint differences; float32 normal draws cast64; accepted float32 model arithmetic unchanged',
  refinement_namespace='fold_in(original noise key,2026091600+level), levels1,2; diagnostic bridge only',
  stop_after='Validated combined lag summary and candidate figures; no subsequent campaigns')
 out.mkdir()
 text='''# Final bounded oracle-criterion-driven h extension

This versioned amendment follows capacity_regime_ex8_v2 without changing its manifest, selection, Phase2 jobs or completed artifacts. This is descriptive, not a search for ARFF's optimal lag. No learned validation/test metric or ranking is consulted for inclusion.

The exact existing oracle uses A=1−dt, dt=1e−7, m=round(h/dt), f_h(x)=expm1(m*log1p(−dt))*x/h. Relative drift bias is abs(expm1(m*log1p(−dt))/h+1), with threshold0.001. Candidate h0.0015 gives0.0007495752155233237; h0.002 gives0.0009992837664303256. Both pass, so exactly100 new fits are registered: two lags, five methods, seeds0–9, K1024 and N640000. No h beyond0.002 is allowed. No iterative solver, retuning or calibration work.

Phase2 remains N*=640000,h*=0.0001 and its existing unresolved-equivalence flags are retained. This controller has no selection step and no capacity phase. It waits for the entire parent pipeline to complete, then for available GPUs. It cannot interrupt or displace valid Phase2 jobs.

Data replay verifies hashes of the original ten Brownian time blocks and bit-identical endpoints at10000 steps. Append blocks10–19, keeping root seed, namespace20260913 for added training pools, full draw shapes, initial states and split IDs. Integrate in float64 and form endpoint differences before any conversion. Existing float32 estimator arithmetic, epochs/adaptations, PRNG progression, folds, checkpoint rules, coefficient definitions, SPD and NLL conventions are unchanged. Validate split/count/dtype identity, zero-noise drift, coupled refinement, finite values and the existing covariance-trace check before any added fit. Drift-bias validity does not certify full covariance finite-lag bias.

Four exclusive GPU workers in persistent tmux; non-isolated timing. Validate existing artifacts before byte-for-byte reuse of350 earlier lag points. Preserve failures/logs and stop on numerical/data/hash errors; no automatic numerical patches. No overwrite, poster or manuscript changes.

Combined nine-lag reports separately show validation and test RMSE_f, raw RMSE_Sigma, NLL and NLL−log(h), with10-seed sample SD, medians/ranges and per-seed records. ARFF raw SPD rates and minimum eigenvalues are archived. Candidate PDF/PNG figures are new files. No optimum is selected and no future experiment is launched after summary completion.
'''
 s.c.infra.write_json(out/'manifest.json',m,exclusive=True)
 s.c.infra.write_json(out/'oracle_screening.json',dict(threshold=threshold,records=screening,learned_metrics_used=False),exclusive=True)
 with (out/'STUDY_PLAN.md').open('x') as f:f.write(text)
 hashes=dict(preserved)
 hashes.update({str((out/n).relative_to(ROOT)):s.r.adapter.digest(out/n) for n in ['manifest.json','oracle_screening.json','STUDY_PLAN.md']})
 s.c.infra.write_json(out/'registration_sha256.json',hashes,exclusive=True)
 for view in (parent/'dataset_roots').glob('N_640000/h_*'):
  target=out/'dataset_roots'/view.parent.name/view.name;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(view,target_is_directory=True)
 control=out/'pipeline';control.mkdir()
 sources=s.SOURCES+['scripts/create_regime_ex8_final_h_data.py','scripts/create_regime_ex8_data.py','scripts/oracle_regime_ex8.py','scripts/summarize_regime_ex8_final_h.py','scripts/run_regime_ex8_final_h_pipeline.py','scripts/prepare_regime_ex8_final_h.py','scripts/smoke_test_regime_ex8_final_h.py']
 s.c.infra.write_json(control/'plan.json',dict(source_sha256={p:s.r.adapter.digest(ROOT/p) for p in sources},order=['wait_parent_phase2_complete','data_validation','h_reuses_and_final_extension','final_lag_summary','stop']),exclusive=True)
 s.c.state(control,status='prepared')
 print(json.dumps(dict(out=str(out),screening=screening,new_fits=len(included)*50,frozen_phase2=m['amendment']['frozen_phase2']),indent=2))
if __name__=='__main__':main()
