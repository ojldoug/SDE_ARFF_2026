#!/usr/bin/env python3
"""Smoke test for Experiment 6 wave data generation."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.config import get_config
from src.experiments.wave_data import generate_wave_data


def main():
    config = get_config("ex6")

    # Shorter time interval for a quick smoke test.
    smoke_data = replace(
        config.data,
        trajectory_time=0.02,
        target_samples=1000,
    )
    smoke_config = replace(config, data=smoke_data)

    x, r, h = generate_wave_data(smoke_config)

    assert x.shape == (1000, 2)
    assert r.shape == (1000, 1)
    assert h.shape == (1000, 1)

    assert np.all(np.isfinite(x))
    assert np.all(np.isfinite(r))
    assert np.all(np.isfinite(h))

    expected_h = 0.5 * smoke_data.fine_step**2
    assert np.allclose(h, expected_h)

    print("Experiment 6 wave smoke test")
    print("----------------------------")
    print(f"x shape       : {x.shape}")
    print(f"r shape       : {r.shape}")
    print(f"h shape       : {h.shape}")
    print(f"h             : {h[0,0]:.8e}")
    print(f"time range    : [{x[:,0].min():.6f}, {x[:,0].max():.6f}]")
    print(f"space range   : [{x[:,1].min():.6f}, {x[:,1].max():.6f}]")
    print(f"mean |r|      : {np.mean(np.abs(r)):.8e}")
    print(f"max  |r|      : {np.max(np.abs(r)):.8e}")
    print()
    print("All wave-data checks passed.")


if __name__ == "__main__":
    main()
