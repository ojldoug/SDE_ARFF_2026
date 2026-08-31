#!/usr/bin/env python3
"""
Smoke test for Experiment 5 (SIR Gillespie benchmark).

This script deliberately uses the legacy Gillespie simulator without
modifying it. It reconstructs consecutive training pairs using the exact
elapsed time

    h_i = t_{i+1} - t_i,

rather than the np.gradient-based step-size construction used in the
legacy notebook/code.

The purpose is to inspect the data-generation protocol before building
the final reproducible Experiment 5 pipeline.
"""

from pathlib import Path
import sys

import numpy as np


# ---------------------------------------------------------------------
# Import the legacy simulator without modifying the legacy GPU/ tree.
# ---------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_DIR = REPO_ROOT / "GPU"
sys.path.insert(0, str(GPU_DIR))

from gillespie.sir import SIRG  # noqa: E402


# ---------------------------------------------------------------------
# Experiment 5 parameters from Owen's current generator.
# ---------------------------------------------------------------------
N = 1024
K1 = 1
K2 = 1
K3 = 0

N_TRAJECTORIES = 10       # smoke test only
TRAJECTORY_TIME = 4.0
FINE_TIME_STEP = 1.0e-3
N_SKIP_STEPS = 10
SEED = 1

NOMINAL_TRAINING_LAG = FINE_TIME_STEP * N_SKIP_STEPS


def main():
    sirg = SIRG(
        N=N,
        k1=K1,
        k2=K2,
        k3=K3,
        random_state=SEED,
    )

    # generate_trajectories already:
    # 1. simulates the SSA,
    # 2. records near the fine observation grid,
    # 3. keeps every N_SKIP_STEPS-th recorded point,
    # 4. transforms coordinates to (theta_0, theta_2) = (S, R).
    trajectories, times = sirg.generate_trajectories(
        n_trajectories=N_TRAJECTORIES,
        time_max=TRAJECTORY_TIME,
        time_step=FINE_TIME_STEP,
        n_skip_steps=N_SKIP_STEPS,
    )

    x0_all = []
    x1_all = []
    h_all = []

    gradient_h_all = []

    retained_trajectories = 0

    for trajectory, t in zip(trajectories, times):
        if len(t) < 2:
            continue

        retained_trajectories += 1

        # Exact elapsed time belonging to each consecutive pair.
        h = np.diff(t)

        # Legacy convention, retained here only for comparison.
        h_gradient = np.gradient(t)[:-1]

        x0_all.append(trajectory[:-1])
        x1_all.append(trajectory[1:])
        h_all.append(h)
        gradient_h_all.append(h_gradient)

    if not h_all:
        raise RuntimeError("No usable trajectories were generated.")

    x0 = np.concatenate(x0_all, axis=0)
    x1 = np.concatenate(x1_all, axis=0)
    h = np.concatenate(h_all)
    h_gradient = np.concatenate(gradient_h_all)

    # In (theta_0, theta_2) coordinates:
    # theta_1 = 1 - theta_0 - theta_2.
    infected_x0 = 1.0 - x0[:, 0] - x0[:, 1]
    infected_x1 = 1.0 - x1[:, 0] - x1[:, 1]

    if not np.all(h > 0):
        raise AssertionError("Found a non-positive observation interval.")

    tol = 1.0e-12
    if np.min(x0) < -tol or np.min(x1) < -tol:
        raise AssertionError("Found a negative S/R state.")

    if np.max(x0) > 1.0 + tol or np.max(x1) > 1.0 + tol:
        raise AssertionError("Found an S/R state greater than one.")

    if np.min(infected_x0) < -tol or np.min(infected_x1) < -tol:
        raise AssertionError("Found a negative infected fraction.")

    difference = h_gradient - h

    print("Experiment 5 smoke test")
    print("-----------------------")
    print(f"N                         : {N}")
    print(f"k1, k2, k3                : {K1}, {K2}, {K3}")
    print(f"seed                      : {SEED}")
    print(f"requested trajectories    : {N_TRAJECTORIES}")
    print(f"usable trajectories       : {retained_trajectories}")
    print(f"trajectory time           : {TRAJECTORY_TIME}")
    print(f"fine observation spacing  : {FINE_TIME_STEP:g}")
    print(f"subsampling factor        : {N_SKIP_STEPS}")
    print(f"nominal training lag      : {NOMINAL_TRAINING_LAG:g}")
    print(f"number of training pairs  : {len(h)}")
    print()

    print("Exact pairwise h = diff(t)")
    print(f"min                       : {np.min(h):.8e}")
    print(f"mean                      : {np.mean(h):.8e}")
    print(f"median                    : {np.median(h):.8e}")
    print(f"max                       : {np.max(h):.8e}")
    print(f"std                       : {np.std(h):.8e}")
    print()

    print("Deviation from nominal lag")
    abs_nominal_error = np.abs(h - NOMINAL_TRAINING_LAG)
    print(f"mean |h-h_nom|            : {np.mean(abs_nominal_error):.8e}")
    print(f"max  |h-h_nom|            : {np.max(abs_nominal_error):.8e}")
    print()

    print("Legacy gradient(t) versus exact diff(t)")
    print(f"mean difference           : {np.mean(difference):.8e}")
    print(f"RMSE difference           : {np.sqrt(np.mean(difference**2)):.8e}")
    print(f"max absolute difference   : {np.max(np.abs(difference)):.8e}")
    print()

    print("State ranges in (theta_0, theta_2) = (S, R)")
    print(f"theta_0 range             : [{x0[:,0].min():.6f}, {x0[:,0].max():.6f}]")
    print(f"theta_2 range             : [{x0[:,1].min():.6f}, {x0[:,1].max():.6f}]")
    print(
        "theta_1 = 1-theta_0-theta_2: "
        f"[{infected_x0.min():.6f}, {infected_x0.max():.6f}]"
    )
    print()
    print("All checks passed.")


if __name__ == "__main__":
    main()
