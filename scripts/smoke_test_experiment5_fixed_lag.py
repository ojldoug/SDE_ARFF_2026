#!/usr/bin/env python3
"""Smoke test for the reproducible fixed-lag Experiment 5 generator."""

from pathlib import Path

from dataclasses import replace
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.config import get_config
from src.experiments.sir_ssa import (
    generate_sir_data,
    sample_initial_simplex,
    simulate_sir_ssa_fixed_observations,
)


N = 1024
K1 = 1.0
K2 = 1.0
K3 = 0.0

SEED = 1
N_TRAJECTORIES = 10
TRAJECTORY_TIME = 4.0
OBSERVATION_LAG = 1.0e-2


def main():
    rng = np.random.default_rng(SEED)

    # Include both t=0 and t=T.
    observation_times = np.arange(
        0.0,
        TRAJECTORY_TIME + 0.5 * OBSERVATION_LAG,
        OBSERVATION_LAG,
    )

    initial_states = sample_initial_simplex(N_TRAJECTORIES, rng)

    trajectories = []

    for initial_state in initial_states:
        trajectory = simulate_sir_ssa_fixed_observations(
            initial_state,
            observation_times,
            population_size=N,
            k1=K1,
            k2=K2,
            k3=K3,
            rng=rng,
        )
        trajectories.append(trajectory)

    trajectories = np.stack(trajectories)

    # Construct consecutive training pairs.
    x0 = trajectories[:, :-1, :].reshape(-1, 2)
    x1 = trajectories[:, 1:, :].reshape(-1, 2)

    h = np.diff(observation_times)
    h = np.tile(h, N_TRAJECTORIES)

    infected_x0 = 1.0 - x0[:, 0] - x0[:, 1]
    infected_x1 = 1.0 - x1[:, 0] - x1[:, 1]

    assert x0.shape == x1.shape
    assert len(h) == len(x0)

    assert np.all(h > 0.0)
    assert np.allclose(h, OBSERVATION_LAG)

    tol = 1.0e-12

    assert np.min(x0) >= -tol
    assert np.min(x1) >= -tol
    assert np.max(x0) <= 1.0 + tol
    assert np.max(x1) <= 1.0 + tol

    assert np.min(infected_x0) >= -tol
    assert np.min(infected_x1) >= -tol

    print("Fixed-lag Experiment 5 smoke test")
    print("---------------------------------")
    print(f"N                        : {N}")
    print(f"k1, k2, k3               : {K1}, {K2}, {K3}")
    print(f"seed                     : {SEED}")
    print(f"trajectories             : {N_TRAJECTORIES}")
    print(f"trajectory time          : {TRAJECTORY_TIME}")
    print(f"observation lag          : {OBSERVATION_LAG}")
    print(f"observations/trajectory  : {len(observation_times)}")
    print(f"training pairs           : {len(h)}")
    print()
    print("Observation intervals")
    print(f"min h                    : {h.min():.8e}")
    print(f"mean h                   : {h.mean():.8e}")
    print(f"max h                    : {h.max():.8e}")
    print()
    print("State ranges in (S,R)")
    print(f"S                        : [{x0[:,0].min():.6f}, {x0[:,0].max():.6f}]")
    print(f"R                        : [{x0[:,1].min():.6f}, {x0[:,1].max():.6f}]")
    print(
        "I = 1-S-R              : "
        f"[{infected_x0.min():.6f}, {infected_x0.max():.6f}]"
    )
    print()
    print("All checks passed.")

    # With k3 = 0, an extinct state is absorbing.
    extinct_initial = np.array([0.8, 0.0, 0.2])

    extinct_path = simulate_sir_ssa_fixed_observations(
        extinct_initial,
        np.array([0.0, 0.01, 0.02, 0.03]),
        population_size=N,
        k1=K1,
        k2=K2,
        k3=K3,
        rng=np.random.default_rng(123),
    )

    assert np.allclose(
        extinct_path,
        extinct_path[0],
    )

    config = get_config("ex5")

    smoke_data = replace(
        config.data,
        n_trajectories=5,
        trajectory_time=0.1,
        target_samples=None,
    )
    smoke_config = replace(config, data=smoke_data)

    x_data, r_data, step_sizes = generate_sir_data(smoke_config)

    assert x_data.ndim == 2
    assert r_data.ndim == 2
    assert step_sizes.ndim == 2

    assert x_data.shape[1] == 2
    assert r_data.shape[1] == 2
    assert step_sizes.shape[1] == 1

    infected_start = (
        1.0
        - x_data[:, 0]
        - x_data[:, 1]
    )

    assert np.all(infected_start > 0.0)

    assert len(x_data) == len(r_data)
    assert len(x_data) == len(step_sizes)

    assert 0 < len(x_data) <= 50

    assert np.all(np.isfinite(x_data))
    assert np.all(np.isfinite(r_data))
    assert np.allclose(step_sizes, 1e-2)

    print()
    print("Config-level Experiment 5 wrapper")
    print(f"x shape                   : {x_data.shape}")
    print(f"r shape                   : {r_data.shape}")
    print(f"h shape                   : {step_sizes.shape}")
    print(f"unique h                  : {np.unique(step_sizes)}")
    print("Wrapper checks passed.")


if __name__ == "__main__":
    main()
