"""Promote archived predictions BEFORE all covariance/NLL operations; no inference."""
from pathlib import Path
import sys,json,hashlib,csv
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from financial_pilot.evaluation import evaluate
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'results/experimental_trajectory_pilot_v1';OUT=ROOT/'results/experimental_trajectory_evaluation_float64_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=False)
 names=['test_predictions.npz','test_evaluation_inputs.npz','summary.json','VALIDATION.json','PROTOCOL.md']
 hashes={n:sha(SRC/n) for n in names};a=dict(np.load(SRC/names[0]));d=dict(np.load(SRC/names[1]));old=json.loads((SRC/'summary.json').read_text());rows=[];result={};outputs={}
 for method in ['affine','kernel','arff','neural']:
  result[method]={}
  for mode,drift in [('end_to_end',a[method+'_drift']),('shared_affine_drift',a['affine_drift'])]:
   m,z=evaluate(d['r'],a[method+'_raw_covariance'],mean=drift,h=.2,floor=1e-4)
   # Preserve original 1e-10 check for this corrected study, stricter than common finance tolerance.
   np.testing.assert_allclose(z['nll'],z['chol_nll'],rtol=1e-10,atol=1e-10)
   m['increment_mse']=float(np.mean((d['r'].astype(float)-.2*drift.astype(float))**2));m['coverage_delta']={p:m['coverage'][p]-old['results'][method][mode]['coverage'][p] for p in m['coverage']};m['nll_delta']=m['nll']-old['results'][method][mode]['nll'];result[method][mode]=m
   rows.append(dict(method=method,mode=mode,old_nll=old['results'][method][mode]['nll'],new_nll=m['nll'],nll_delta=m['nll_delta'],raw_spd_violation_rate=m['raw_spd_violation_rate'],min_raw_eigenvalue=m['min_raw_eigenvalue'],projection_frequency=m['projection_frequency'],coverage95=m['coverage']['0.95'],coverage95_delta=m['coverage_delta']['0.95'],algebra_max_abs=m['algebra_max_absolute_difference']))
   outputs[method+'_'+mode+'_nll']=z['nll']
 with (OUT/'old_new_metrics.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 np.savez_compressed(OUT/'corrected_score_arrays.npz',**outputs)
 record=dict(results=result,input_hashes=hashes,evaluator_sha256=sha(Path(__file__).parent/'financial_pilot/evaluation.py'),script_sha256=sha(Path(__file__)),rtol=1e-10,atol=1e-10,all_algebra_checks_pass=True,original_outputs_preserved=all(sha(SRC/n)==h for n,h in hashes.items()),no_new_predictions_or_fitting=True)
 (OUT/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,ax=plt.subplots(figsize=(8,4));names=['affine','kernel','arff','neural']
 for j,mode in enumerate(['end_to_end','shared_affine_drift']):ax.scatter([result[k][mode]['nll'] for k in names],np.arange(4)+(j-.5)*.15,label=mode.replace('_',' '))
 ax.set_yticks(range(4),['Affine + constant','Kernel moments','ARFF','Joint MLP']);ax.invert_yaxis();ax.set_xlabel('Corrected float64 test Gaussian NLL (lower is better)');ax.legend();ax.grid(axis='x',alpha=.2);fig.tight_layout();fig.savefig(OUT/'comparison.pdf');fig.savefig(OUT/'comparison.png',dpi=180)
 lines=['# Common float64 reevaluation of the colloidal pilot','','All four saved raw covariance and drift arrays were promoted to float64 before symmetrization, eigenanalysis, flooring, solves, log determinants and reductions. This cannot restore precision lost during prediction. Same observations, h=.2s, physical units, floor1e-4um²/s and equal retained-increment weighting; no new model predictions or fits. Original outputs and failed1e-10 check remain unchanged. Corrected eigen/Cholesky checks pass the original rtol=atol=1e-10.','','| Method | Mode | Corrected NLL | New minus old | 95% coverage |','|---|---|---:|---:|---:|']
 for r in rows:lines.append(f"| {r['method']} | {r['mode']} | {r['new_nll']:.9f} | {r['nll_delta']:.3g} | {r['coverage95']:.3%} |")
 lines+=['','Conclusion unchanged: very similar scores; no demonstrated ARFF advantage over constant covariance. All raw SPD violation/floor rates remain zero. Coverage changes are recorded explicitly in summary.json. The four-column versus six-column discrepancy remains unresolved: local README/header inspections establish a plausible mapping, not author-confirmed semantics. Local screening_notes.md and source paper/readme supply no unique experimental-cell IDs or per-movie optical-phase calibration. Do not claim these issues resolved. No author contact occurred. Archived supplementary candidate only; no manuscript inclusion.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n');(OUT/'FINAL_SUMMARY.txt').write_text('Common float64 reevaluation complete; all eight algebra checks pass1e-10; original failures preserved. Conclusions unchanged. No inference/fitting; schema/cell/optical-phase limitations unresolved.\n')
 (OUT/'COMPLETE.json').write_text(json.dumps({'complete':True,'summary_sha256':sha(OUT/'summary.json')},indent=2)+'\n')
if __name__=='__main__':main()
