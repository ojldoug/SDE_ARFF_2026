#!/usr/bin/env python3
"""Explicit data replay into new output; inspection is the default, no dispatch.
Native generators are unchanged. Coupled extensions replay from archived parents.
"""
import argparse,importlib,json,os,sys,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
SPECS={
 'ex8-float64':('create_ex8_float64_v2','results/controlled_study_2026/float64_v2_validation'),
 'coupled-lags':('create_controlled_ex8_float64_lags','results/controlled_study_2026/float64_v2'),
 'regime-v1':('create_regime_ex8_data','results/capacity_regime_ex8_v1'),
 'regime-v2':('create_regime_ex8_v2_data','results/capacity_regime_ex8_v2'),
 'final-h':('create_regime_ex8_final_h_data','results/capacity_regime_ex8_final_h_v1')}
def main():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['original','surrogate',*SPECS]);p.add_argument('--experiment',choices=['ex1','ex2','ex3','ex5','ex6','ex7','ex8'],default='ex1');p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true');a=p.parse_args()
 if a.output and (a.output.exists() or a.output.resolve().is_relative_to(ROOT/'results') or a.output.resolve().is_relative_to(ROOT/'data')):p.error('New external output directory required')
 if a.execute and not a.output:p.error('--output is required')
 if not a.execute:
  print(json.dumps(dict(stage=a.stage,source='scripts/generate_dataset.py' if a.stage=='original' else 'scripts/arff_drift_K_phase2_final.py' if a.stage=='surrogate' else 'scripts/'+SPECS[a.stage][0]+'.py',execute=False,warning='Fresh metadata/NPZ container bytes need not match published bytes. Exact archived input hashes are required by production replay; compare arrays/splits and numerical construction separately. Coupled stages use authenticated archived parents.'),indent=2));return
 from run_ex8_split_mlp_w27 import check_assigned_gpu
 check_assigned_gpu(allow_cpu=False)
 if a.stage=='original':
  mod=importlib.import_module('generate_dataset');sys.argv=[mod.__file__,a.experiment,'--output-dir',str(a.output.resolve())];mod.main();return
 if a.stage=='surrogate':
  mod=importlib.import_module('arff_drift_K_phase2_final');a.output.mkdir(parents=True)
  for f in ['seed_manifest.json','data_recipe.json']:shutil.copy2(ROOT/'results/arff_drift_K_mechanism_diagnostic_phase2_final_v1'/f,a.output/f)
  mod.O=a.output.resolve();mod.make_common_data();return
 name,source=SPECS[a.stage];mod=importlib.import_module(name);a.output.mkdir(parents=True)
 if a.stage=='ex8-float64':
  mod.OUT=a.output.resolve()/'validation';mod.TARGET=a.output.resolve()/'ex8_float64_v2.npz'
 elif a.stage=='coupled-lags':
  mod.STUDY=a.output.resolve();shutil.copy2(ROOT/source/'manifest.json',a.output/'manifest.json')
 else:
  mod.OUT=a.output.resolve();shutil.copy2(ROOT/source/'manifest.json',a.output/'manifest.json')
 mod.main()
if __name__=='__main__':main()
