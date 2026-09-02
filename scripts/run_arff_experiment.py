#!/usr/bin/env python3
"""Run one two-stage ARFF experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import jax


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.arff.evaluation import (
    gaussian_nll,
    true_function_errors,
)
from src.arff.two_stage import fit_two_stage_arff
from src.experiments.config import get_config
from src.experiments.definitions import get_experiment
from src.experiments.dataset import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        choices=[f"ex{i}" for i in range(1, 9)],
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    name = args.experiment

    config = get_config(name)
    definition = get_experiment(name)

    if config.K is None:
        raise ValueError(
            f"No Fourier-feature count K has been established for {name}."
        )

    data = load_dataset(
        REPO_ROOT / "data" / f"{name}.npz"
    )

    x = data.x
    r = data.r
    h = data.h

    train_idx = data.train_idx
    validation_idx = data.validation_idx
    test_idx = data.test_idx

    key = jax.random.PRNGKey(args.seed)

    print(f"Experiment : {name}")
    print(f"seed       : {args.seed}")
    print(f"backend    : {jax.default_backend()}")
    print(f"train N    : {len(train_idx)}")
    print(f"validation : {len(validation_idx)}")
    print(f"test       : {len(test_idx)}")
    print(f"folds      : {config.arff.n_folds}")
    print(f"K          : {config.K}")
    print(f"iterations : {config.arff.M_max}")
    print()

    start = time.time()

    key, model, crossfit = fit_two_stage_arff(
        key,
        x[train_idx],
        r[train_idx],
        h[train_idx],
        K=config.K,
        diff_type=definition.diff_type,
        config=config.arff,
        fold_seed=config.split.seed,
    )

    training_time = time.time() - start

    print(f"training time: {training_time:.3f} s")
    print()

    for label, idx in [
        ("train", train_idx),
        ("validation", validation_idx),
        ("test", test_idx),
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
        print(f"  NLL                    : {result.nll:.8e}")
        print(
            "  raw SPD violation rate : "
            f"{result.spd_violation_rate:.6f}"
        )
        print(
            "  min raw eigenvalue     : "
            f"{result.min_raw_eigenvalue:.8e}"
        )
        print(
            "  min projected eigenvalue: "
            f"{result.min_projected_eigenvalue:.8e}"
        )
        print(
            "  drift RMSE              : "
            f"{drift_rmse:.8e}"
        )
        print(
            "  covariance RMSE         : "
            f"{covariance_rmse:.8e}"
        )
        print()


if __name__ == "__main__":
    main()
