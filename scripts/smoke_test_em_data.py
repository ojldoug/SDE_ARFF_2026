#!/usr/bin/env python3
"""Smoke test for the standard reproducible EM data generator."""

from dataclasses import replace
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.config import get_config
from src.experiments.definitions import get_experiment
from src.experiments.em_data import generate_standard_em_data


TEST_EXPERIMENTS = ["ex1", "ex2", "ex3", "ex7", "ex8"]


def main():
    for name in TEST_EXPERIMENTS:
        definition = get_experiment(name)
        config = get_config(name)

        # Small version of the final configuration.
        smoke_data = replace(
            config.data,
            n_trajectories=128,
            target_samples=128,
        )
        smoke_config = replace(config, data=smoke_data)

        x, r, h = generate_standard_em_data(
            definition,
            smoke_config,
        )

        assert x.shape == (128, definition.state_dimension)
        assert r.shape == (128, definition.n_dimensions)
        assert h.shape == (128, 1)

        assert np.all(np.isfinite(x))
        assert np.all(np.isfinite(r))
        assert np.all(np.isfinite(h))

        assert np.all(h > 0.0)
        assert np.allclose(
            h,
            smoke_config.data.observation_lag,
        )

        print(name)
        print(f"  x shape      : {x.shape}")
        print(f"  r shape      : {r.shape}")
        print(f"  h shape      : {h.shape}")
        print(f"  h            : {h[0,0]:.8e}")
        print(f"  mean |r|     : {np.mean(np.abs(r)):.8e}")
        print(f"  max  |r|     : {np.max(np.abs(r)):.8e}")
        print()

    print("All standard EM smoke tests passed.")


if __name__ == "__main__":
    main()
