#!/usr/bin/env python3
"""Extract authenticated prior captures only; no generation or fitting."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--capture',type=Path,required=True);p.add_argument('--record',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();z=dict(np.load(a.capture));r=json.loads(a.record.read_text());keys=['x','y','initial_key','initial_omega','omega','fit_positions','validation_positions','amp_archived','amp_new']
for k in keys:assert hashlib.sha256(z[k].tobytes()).hexdigest()==r['hashes'][k],k
with a.output.open('xb') as f:np.savez_compressed(f,**{k:z[k] for k in keys})
print(hashlib.sha256(a.output.read_bytes()).hexdigest())
