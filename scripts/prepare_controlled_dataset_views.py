"""Immutable nested canonical views for the preregistered controlled N study."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.experiments.dataset import load_dataset
OUT=ROOT/'results/controlled_study_2026'
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 m=json.loads((OUT/'manifest.json').read_text());parent=ROOT/m['dataset']['path'];assert digest(parent)==m['dataset']['sha256'];d=load_dataset(parent)
 for n in m['N_grid']:
  p=OUT/'dataset_roots'/f'N_{n}';p.mkdir(parents=True,exist_ok=False);(p/'data').mkdir()
  for folder in ('src','scripts'):(p/folder).symlink_to(ROOT/folder,target_is_directory=True)
  ids=np.concatenate([d.train_idx[:n],d.validation_idx,d.test_idx])
  target=p/'data/ex8.npz'
  if n==80000:
   target.symlink_to(parent)
   split={'train_idx':d.train_idx,'validation_idx':d.validation_idx,'test_idx':d.test_idx};row_map=np.arange(len(d.x))
  else:
   split={'train_idx':np.arange(n),'validation_idx':np.arange(n,n+10000),'test_idx':np.arange(n+10000,n+20000)};row_map=ids
   with target.open('xb') as f:np.savez_compressed(f,x_data=d.x[ids],r_data=d.r[ids],step_sizes=d.h[ids],**split)
  with (p/'original_row_ids.npz').open('xb') as f:np.savez_compressed(f,original_row_ids=row_map)
  record=dict(n_train=n,n_validation=10000,n_test=10000,source_dataset=str(parent),source_sha256=digest(parent),dataset_sha256=digest(target),original_row_ids_sha256=digest(p/'original_row_ids.npz'),training_original_row_ids_sha256=hashlib.sha256(d.train_idx[:n].tobytes()).hexdigest(),validation_original_row_ids_sha256=hashlib.sha256(d.validation_idx.tobytes()).hexdigest(),test_original_row_ids_sha256=hashlib.sha256(d.test_idx.tobytes()).hexdigest(),subset_rule='canonical train_idx[:N], identical validation/test rows and ordering')
  (p/'dataset_view.json').write_text(json.dumps(record,indent=2))
  actual=load_dataset(target)
  for key in ('x','r','h'):
   np.testing.assert_array_equal(getattr(actual,key)[actual.train_idx],getattr(d,key)[d.train_idx[:n]])
   np.testing.assert_array_equal(getattr(actual,key)[actual.validation_idx],getattr(d,key)[d.validation_idx]);np.testing.assert_array_equal(getattr(actual,key)[actual.test_idx],getattr(d,key)[d.test_idx])
  print('Validated exact nested view',n,record['dataset_sha256'])
if __name__=='__main__':main()
