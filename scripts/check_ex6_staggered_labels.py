#!/usr/bin/env python3
"""No training or canonical-data mutation: deterministic proof of ex6 label mismatch."""
from pathlib import Path
import sys,json
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.experiments.wave_data import integrate_stochastic_wave,split_wave_learning_data
class ZeroNoise:
    def __init__(self):self.scales=[]
    def normal(self,loc,scale,size):self.scales.append(float(scale));return np.zeros(size)
def main():
    delta=.01;space=np.arange(20)*delta;time=np.arange(8)*delta;rng=ZeroNoise()
    f=lambda x,t,u:1.+x
    g=lambda x,t,u:np.ones_like(x)
    with patch('src.experiments.wave_data.np.random.default_rng',return_value=rng):
        u=integrate_stochastic_wave(np.zeros(20),np.zeros(20),f,g,time,space,seed=1)
    a,x,b=split_wave_learning_data(u,space[::2],time)
    residual=(b-a).reshape(6,8);declared_h=.5*delta**2
    declared_truth=(1+x[:,1]).reshape(6,8)
    y=residual/declared_h
    assert np.allclose(y[0],.5*declared_truth[0])
    assert np.allclose(y[1::2]-declared_truth[1::2],delta)
    assert np.allclose(y[2::2],declared_truth[2::2])
    corrected_h=np.full_like(residual,declared_h);corrected_h[0]*=.5
    corrected_truth=declared_truth.copy();corrected_truth[1::2]+=delta
    assert np.allclose(residual/corrected_h,corrected_truth,atol=1e-12)
    assert np.isclose(rng.scales[0],delta)
    assert np.allclose(rng.scales[1:],np.sqrt(2)*delta)
    print(json.dumps(dict(status='CONFIRMED_CANONICAL_LABEL_MISMATCH',
        fixture='zero initial state; f(x)=1+x; zero realized noise; dx=dt=0.01',
        current_max_drift_identity_error_per_time_row=np.max(abs(y-declared_truth),axis=1).tolist(),
        corrected_label_max_error=float(np.max(abs(residual/corrected_h-corrected_truth))),
        noise_draw_standard_deviations=rng.scales,
        canonical_affected_initialization_rows=498,canonical_affected_spatial_rows=999*498,
        canonical_total_rows=1998*498,
        minimal_candidate_correction='Preserve x/r observations: half h for first 498 rows; add grid_step to spatial coordinate in blocks with integration j even. Audit time-coordinate convention separately; current forcing autonomous.',
        scope='Audit fixture only. No canonical dataset modified or regenerated.'),indent=2))
if __name__=='__main__':main()
