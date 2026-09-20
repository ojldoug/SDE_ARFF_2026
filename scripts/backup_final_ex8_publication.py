"""Create a verified, internally deduplicated server backup; never delete sources."""
import hashlib,json,os,shutil,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[1]
B=R.parent/'backups/SDE_ARFF_2026_final_ex8_2026-09'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 B.mkdir(parents=True,exist_ok=False);paths=set();bundle=R/'results/final_ex8_publication_bundle'
 refs=json.loads((bundle/'large_file_references.json').read_text())
 for rel in refs['artifacts_and_authenticated_logs']:
  p=R/rel;paths.add(p)
  for q in p.parent.glob('*.json'):paths.add(q)
 for rel in refs['datasets']:
  paths.add(R/rel)
  prov=(R/rel).with_suffix('.provenance.json')
  if prov.exists():paths.add(prov)
 roots=[bundle,R/'results/capacity_regime_ex8_v1',R/'results/capacity_regime_ex8_v2',R/'results/capacity_regime_ex8_final_h_v1',R/'results/controlled_study_2026/float64_v2_validation',R/'results/controlled_study_2026/float64_v2/hybrid_jointmlp_arff',R/'results/controlled_study_2026/float64_v2/oracle_component_swap']
 for root in roots:
  for p in root.rglob('*'):
   if p.is_file() and 'production' not in p.parts and p.suffix in ['.md','.json','.csv','.pdf','.png','.npz']:
    if not any(x in str(p) for x in ['attempt','failed','staged','__pycache__']):paths.add(p)
 # Dataset roots may be directory symlinks: explicitly descend using os.walk.
 for study in roots[1:4]:
  for base,_,files in os.walk(study/'dataset_roots',followlinks=True):
   for name in files:
    if name.endswith(('.npz','.json')):paths.add(Path(base)/name)
 for p in bundle.rglob('*'):
  if p.is_file():paths.add(p)
 records={};objects={}
 for p in sorted(paths):
  rel=p.relative_to(R);d=sha(p);dest=B/'repository'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
  if d in objects:os.link(objects[d],dest)
  else:
   shutil.copy2(p,dest);assert sha(dest)==d;objects[d]=dest
  records[str(Path('repository')/rel)]={'bytes':p.stat().st_size,'sha256':d}
 subprocess.run(['git','bundle','create',str(B/'repository.git.bundle'),'--all'],cwd=R,check=True)
 subprocess.run(['git','bundle','verify',str(B/'repository.git.bundle')],cwd=R,check=True)
 head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=R,text=True).strip()
 (B/'README.md').write_text(f'''# Final Experiment 8 server backup — 2026-09-20

Source: {R}\nBranch: {branch}\nCommit: {head}

repository.git.bundle preserves all local Git refs and committed source/configuration/publication files. Clone with git clone repository.git.bundle restored, then overlay repository/ into restored/. The overlay preserves repository-relative paths for final frozen artifacts, authenticated console logs, contexts, corrected canonical datasets and all controlled data views, summaries, figures, provenance, hybrid/oracle diagnostics and publication bundle. Identical file contents are hard-linked only WITHIN this backup, never to source files. Ordinary copy remains valid; use rsync -aH to preserve deduplication.

Corrected float64 results supersede old float32 scientific conclusions. Original canonical float32 data is retained for provenance. Unrelated experiments, scratch logs/caches, staged duplicate archives, and superseded historical model campaigns are intentionally omitted. No source is deleted. Final native archives preserve selected models, histories and fold models; no expensive final fit needs regeneration. Omitted older diagnostic fits can be reconstructed from their original repository/protocol, but are not needed for the final authenticated results.

SHA256_MANIFEST.json records all payload sizes/hashes (excluding itself). The Mac/Synology destination is not mounted here; transfer this entire directory later.
''')
 for p in [B/'README.md',B/'repository.git.bundle']:records[p.name]={'bytes':p.stat().st_size,'sha256':sha(p)}
 out={'source_commit':head,'branch':branch,'files':records,'file_count':len(records),'logical_bytes':sum(v['bytes'] for v in records.values()),'unique_payload_bytes':sum(p.stat().st_size for p in objects.values())+(B/'repository.git.bundle').stat().st_size+(B/'README.md').stat().st_size}
 (B/'SHA256_MANIFEST.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='files'},indent=2));print(B)
if __name__=='__main__':main()
