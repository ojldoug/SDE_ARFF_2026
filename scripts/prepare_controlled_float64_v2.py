"""Register approved data-only correction; preserve all original registration bytes."""
from pathlib import Path
import json, sys, hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
import prepare_controlled_dataset_views as views
OLD=ROOT/'results/controlled_study_2026';OUT=OLD/'float64_v2'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2)
def main():
 validation=json.loads((OLD/'float64_v2_validation/validation.json').read_text());assert validation['status']=='passed'
 original=json.loads((OLD/'manifest.json').read_text());m=json.loads(json.dumps(original))
 m['schema_version']=2
 m['dataset'].update(path='data/ex8_float64_v2.npz',sha256=validation['corrected_sha256'])
 m['correction_provenance']=dict(original_manifest_sha256=sha(OLD/'manifest.json'),original_dataset=original['dataset'],validation_file=str(OLD/'float64_v2_validation/validation.json'),validation_sha256=sha(OLD/'float64_v2_validation/validation.json'),change='float64 integration and endpoint difference, same noise/states/splits; estimator arithmetic unchanged',approval='User approved correction and all five baseline methods seeds0–29 before wider curves',historical_K32='All nine old-data K32 runs excluded from corrected-data curves')
 m['priority_baseline']=dict(K=128,N=80000,h=.0001,seeds=list(range(30)),methods=m['methods'],reuse_old_data=False)
 OUT.mkdir(exist_ok=False)
 write(OUT/'manifest.json',m)
 text=(OLD/'STUDY_PLAN.md').read_text()
 with (OUT/'STUDY_PLAN.md').open('x') as f:f.write('# Approved float64 dataset amendment\n\nOnly the dataset arithmetic changes. All original registered grids and scientific settings below remain in force. Complete and validate the corrected 150-run K128/width27 baseline before any broader GPU work. Never reuse original float32 runs as corrected-data evidence. Native estimator arithmetic and timing scopes remain unchanged; stored float64 increments convert through the native loader/JAX path.\n\n'+text)
 for name in ['sample_accounting.md','sample_accounting.csv']:
  with (OUT/name).open('xb') as f:f.write((OLD/name).read_bytes())
 write(OUT/'registration_sha256.json',{str((OUT/name).relative_to(ROOT)):sha(OUT/name) for name in ['manifest.json','STUDY_PLAN.md','sample_accounting.md','sample_accounting.csv']})
 (OUT/'dietrich').mkdir()
 write(OUT/'dietrich/gate.json',json.loads((OLD/'dietrich/gate.json').read_text()))
 views.OUT=OUT;views.main()
 print('Versioned registration and exact corrected nested data views complete')
if __name__=='__main__':main()
