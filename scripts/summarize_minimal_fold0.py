#!/usr/bin/env python3
"""Read-only array/program comparison; no numerical fitting."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();dirs=sorted(a.workspace.glob('process_[0-9]'));records=[json.loads((d/'record.json').read_text()) for d in dirs];arrays=[dict(np.load(d/'arrays.npz')) for d in dirs];fixture=dict(np.load(a.workspace/'fixture.npz'))
def compare(x,y):
 delta=y.astype(float)-x.astype(float)
 return dict(bitwise_equal=x.dtype==y.dtype and x.tobytes()==y.tobytes(),within_original_tolerance=bool(np.allclose(x,y,rtol=1e-5,atol=1e-6)),max_abs=float(np.max(abs(delta))),rms=float(np.sqrt(np.mean(delta**2))))
r=dict(processes=len(dirs),comparisons=[{k:compare(arrays[0][k],z[k]) for k in arrays[0]} for z in arrays[1:]],checkpoint_comparisons=[{k:compare(fixture[k],z['amp']) for k in ['amp_archived','amp_new']} for z in arrays],identities={k:all(v[k]==records[0][k] for v in records) for k in ['gpu','fixture_sha256','source_sha256','affinity','environment','jax_config','loaded_libraries','cache_inventories']},program_hashes=[v['program_hashes'] for v in records],runtime_fingerprints=[v['runtime_fingerprint'] for v in records],preserved_files={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for d in dirs for f in d.iterdir() if f.is_file()})
with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
