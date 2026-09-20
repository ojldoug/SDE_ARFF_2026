"""Freeze the user-approved amendment before dispatch; no GPU or training."""
import os
os.environ['JAX_PLATFORMS']='cpu'
from pathlib import Path
import sys,json,datetime,re
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from oracle_regime_ex8 import oracle
from src.experiments.mlp_size import matched_two_layer_width,mlp_parameter_count
import run_regime_ex8_v2_campaign as scheduler
OUT=scheduler.STUDY;PARENT=scheduler.PARENT;r=scheduler.r;c=scheduler.c
def main():
 assert not OUT.exists()
 assert not (PARENT/'selection.json').exists() and not (PARENT/'campaigns/capacity').exists()
 m=json.loads((PARENT/'manifest.json').read_text())
 threshold=float(re.search(r'bias <=([\d.]+)',m['diagnostic']['h_selection'])[1])
 screening=[dict(**oracle(h),included=oracle(h)['relative_discrete_drift_bias']<=threshold) for h in [.0008,.001]]
 included=[v['h'] for v in screening if v['included']]
 assert included==[.0008,.001] # Independent oracle calculation; no learned results read.
 parent_hashes={str(p.relative_to(ROOT)):r.adapter.digest(p) for p in [PARENT/'manifest.json',PARENT/'STUDY_PLAN.md',PARENT/'registration_sha256.json']}
 m['capacity']=[]
 for K in [64,128,256,512,1024]:
  w,mp,fp=matched_two_layer_width(state_dimension=2,covariance_output_dimension=3,n_frequencies=K)
  m['capacity'].append(dict(K=K,mlp_width=w,mlp_parameters=mp,fourier_parameters=fp,mlp_drift=mlp_parameter_count(2,2,(w,w)),mlp_factor=mlp_parameter_count(2,3,(w,w)),mismatch_percent=100*(mp-fp)/fp))
 m['h_grid']+=included;m['lag_dataset']['steps']=[round(h/1e-7) for h in m['h_grid']]
 m['registered_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
 m['diagnostic']['phase2']='K64,128,256,512,1024 at frozen common N*,h*, seeds0–9. Existing validation-only selection/equivalence/fallback unchanged.'
 m['amendment']=dict(version=2,parent=str(PARENT.relative_to(ROOT)),parent_sha256=parent_hashes,
  approved='User amendment 2026-09-14: K128 plus only oracle-approved upward h extension; no method-performance criterion',
  included_h=included,drift_bias_threshold=threshold,oracle_screening=screening,
  old_artifacts='Read-only exact reuse after validation; no old numerical reruns',
  K128_reuse='Conservative policy: always fit new K128 under eventual selected regime; never reuse baseline without exact matching',
  coupling='Replay unchanged original blocks0–3 and verify r4000 bit-for-bit. Continue time blocks4–9 using existing fold_in key rule and full original draw shapes. Do not reconstruct state as x+r.',
  refinement_namespace='Original noise key fold_in(2026091400+level), levels1,2; diagnostic only',
  old_pipeline='Replace only idle waiting continuation controller; keep parent h supervisor and all numerical children running unchanged')
 OUT.mkdir()
 plan=(PARENT/'STUDY_PLAN.md').read_text()
 plan+='\n\n# Approved amendment v2 — 2026-09-14\n\nThis version supersedes the still-unstarted Phase2 grid with K={64,128,256,512,1024}. Parent artifacts/plans remain immutable. Extend Phase1h by h={0.0008,0.001}, solely because exact discrete drift relative biases 0.000399843394644 and 0.000499783424963 are below the existing 0.001 bound. No learned performance enters inclusion. Selection remains unfrozen until the complete extended h grid passes integrity checks; the same validation-only rule and fallback remain.\n\nOriginal normal keys and full draw shapes are replayed and SHA-256 verified through4000 substeps. Continue the same Brownian paths through8000/10000 substeps with time-block keys4–9. State arithmetic and endpoint differences remain float64; estimator arithmetic remains float32. Validate exact old-prefix replay, deterministic drift, Brownian-bridge refinement, sample/split/dtype identity and the existing oracle covariance-trace gate before added fits. Failure stops continuation; no automatic scientific repair.\n\nParent N and h results are copied byte-for-byte into versioned reuse slots only after validation; source artifacts remain untouched. K128 uses a new fit at the frozen common regime. All five methods and seeds0–9 remain unchanged. The existing equal-width two-hidden-layer MLP rule gives:\n\n| K | Fourier retained parameters | MLP hidden architecture | Drift parameters | Factor parameters | MLP total |\n|---|---|---|---|---|---|\n'
 for v in m['capacity']:plan+=f"| {v['K']} | {v['fourier_parameters']} | {v['mlp_width']}×{v['mlp_width']} | {v['mlp_drift']} | {v['mlp_factor']} | {v['mlp_parameters']} |\n"
 plan+='\nController order: wait unchanged parent h campaign; create/validate larger-lag data on an idle GPU; archive all exact N reuses; run amended h plan with exact old-h reuses and100 added fits; unchanged validation selection; reports; amended capacity250 points with exact K1024 reuse where possible; final report. Persistent tmux, four exclusive workers, non-isolated timing, original stop conditions. No poster/manuscript edits, tuning/calibration or unrelated campaigns. Parent excerpt above is historical; this amendment governs the new grid.\n'
 for name,value in [('manifest.json',m),('oracle_screening.json',dict(threshold=threshold,records=screening,learned_metrics_read=False))]:c.infra.write_json(OUT/name,value,exclusive=True)
 with (OUT/'STUDY_PLAN.md').open('x') as f:f.write(plan)
 reg={str((OUT/name).relative_to(ROOT)):r.adapter.digest(OUT/name) for name in ['manifest.json','STUDY_PLAN.md','oracle_screening.json']}
 c.infra.write_json(OUT/'registration_sha256.json',reg,exclusive=True)
 # Versioned symlinks are read-only inputs, never mutation targets.
 for view in (PARENT/'dataset_roots').glob('N_*/h_*'):
  target=OUT/'dataset_roots'/view.parent.name/view.name;target.parent.mkdir(parents=True,exist_ok=True);target.symlink_to(view,target_is_directory=True)
 control=OUT/'pipeline';control.mkdir()
 sources=scheduler.SOURCES+['scripts/create_regime_ex8_v2_data.py','scripts/create_regime_ex8_data.py','scripts/oracle_regime_ex8.py','scripts/analyze_regime_ex8_v2.py','scripts/run_regime_ex8_v2_pipeline.py','scripts/prepare_regime_ex8_v2.py']
 c.infra.write_json(control/'plan.json',dict(source_sha256={p:r.adapter.digest(ROOT/p) for p in sources},order=['wait_parent_h','data','N_reuse','h_extended','select_validation_only','N_report','h_report','capacity','capacity_report']),exclusive=True)
 c.state(control,status='prepared')
 assert r.adapter.verify_registration()==m
 print(json.dumps(dict(screening=screening,capacity=m['capacity'],out=str(OUT)),indent=2))
if __name__=='__main__':main()
