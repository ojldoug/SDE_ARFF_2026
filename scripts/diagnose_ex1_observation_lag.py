#!/usr/bin/env python3
"""Test Experiment 1 with the historical 1000-substep protocol."""

from dataclasses import replace
from pathlib import Path
import sys
import time

import jax
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.arff.evaluation import gaussian_nll, true_function_errors
from src.arff.two_stage import fit_two_stage_arff
from src.experiments.config import get_config
from src.experiments.definitions import get_experiment
from src.experiments.em_data import generate_standard_em_data


def main():
    definition = get_experiment("ex1")
    config = get_config("ex1")

    # Diagnostic only:
    #
    # Retained observation lag:
    #     h = 1e-2
    #
    # Generated using 1000 Euler--Maruyama substeps:
    #     h_fine = h / 1000 = 1e-5.
    diagnostic_data = replace(
        config.data,
        n_trajectories=10_000,
        trajectory_time=1e-2,
        observation_lag=1e-2,
        em_substeps=1000,
        target_samples=10_000,
    )

    # Use the experiment-level Fourier-feature count shared by
    # ARFF and Fourier-feature Adam.

    diagnostic_config = replace(
        config,
        data=diagnostic_data,
    )

    print("Generating diagnostic ex1 data...")
    x, r, h = generate_standard_em_data(
        definition,
        diagnostic_config,
    )

    print(f"x shape : {x.shape}")
    print(f"r shape : {r.shape}")
    print(f"h       : [{h.min():.8e}, {h.max():.8e}]")
    print()

    # Historical 90/10 train-validation split.
    rng = np.random.default_rng(config.split.seed)
    indices = rng.permutation(len(x))

    n_train = int(0.9 * len(x))

    train_idx = indices[:n_train]
    validation_idx = indices[n_train:]

    print(f"train N      : {len(train_idx)}")
    print(f"validation N : {len(validation_idx)}")
    print(f"K            : {diagnostic_config.K}")
    print(f"folds        : {diagnostic_config.arff.n_folds}")
    print()

    # Diagnose signal-to-noise scale of the drift regression target.
    target = r[train_idx] / h[train_idx]
    truth = np.asarray(
        definition.drift(x[train_idx])
    )
    noise = target - truth

    print("Drift target diagnostic")
    print(
        "  target RMS        : "
        f"{np.sqrt(np.mean(target**2)):.8e}"
    )
    print(
        "  true drift RMS    : "
        f"{np.sqrt(np.mean(truth**2)):.8e}"
    )
    print(
        "  target-truth RMSE : "
        f"{np.sqrt(np.mean(noise**2)):.8e}"
    )
    print()

    key = jax.random.PRNGKey(0)

    start = time.time()

    key, model, crossfit = fit_two_stage_arff(
        key,
        x[train_idx],
        r[train_idx],
        h[train_idx],
        K=diagnostic_config.K,
        diff_type=definition.diff_type,
        config=diagnostic_config.arff,
        fold_seed=config.split.seed,
    )

    print(
        f"training time      : {time.time() - start:.3f} s"
    )
    print()

    for label, idx in [
        ("train", train_idx),
        ("validation", validation_idx),
    ]:
        result = gaussian_nll(
            model,
            x[idx],
            r[idx],
            h[idx],
            spd_epsilon=config.evaluation.spd_epsilon,
        )

        drift_rmse, covariance_rmse = true_function_errors(
            model,
            x[idx],
            true_drift=definition.drift,
            true_diffusion_factor=definition.diffusion_factor,
        )

        print(label)
        print(f"  NLL                     : {result.nll:.8e}")
        print(f"  drift RMSE              : {drift_rmse:.8e}")
        print(f"  covariance RMSE         : {covariance_rmse:.8e}")
        print(
            "  raw SPD violation rate  : "
            f"{result.spd_violation_rate:.6f}"
        )
        print(
            "  min raw eigenvalue      : "
            f"{result.min_raw_eigenvalue:.8e}"
        )
        print()


if __name__ == "__main__":
    main()