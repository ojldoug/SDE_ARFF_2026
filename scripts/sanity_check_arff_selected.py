#!/usr/bin/env python3
"""
Validation-only sanity check of the selected ARFF configurations.

Fits one ARFF model for each requested experiment using the selected
adaptation budget M and evaluates the canonical validation split.

The test split is never evaluated.

This is a pre-production sanity check, not a timing benchmark and not
additional hyperparameter tuning.

Selected adaptation budgets
----------------------------

    ex1:  25
    ex2:  25
    ex3:  50
    ex4:  50
    ex5:  50
    ex6: 200
    ex7:  25
    ex8:  25

Validated SPD projection floors
-------------------------------

    ex3: 1e-2
    ex8: 1e-3

Other experiments retain their currently configured evaluation epsilon.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.arff.evaluation import (
    gaussian_nll,
    true_function_errors,
)
from src.arff.two_stage import fit_two_stage_arff
from src.experiments.config import get_config
from src.experiments.dataset import (
    load_dataset,
    validate_split_indices,
)
from src.experiments.definitions import get_experiment


SELECTED_M = {
    "ex1": 25,
    "ex2": 25,
    "ex3": 50,
    "ex4": 50,
    "ex5": 50,
    "ex6": 200,
    "ex7": 25,
    "ex8": 25,
}

SELECTED_SPD_EPSILON = {
    "ex3": 1e-2,
    "ex8": 1e-3,
}

ALL_EXPERIMENTS = tuple(
    SELECTED_M.keys()
)


def run_experiment(
    experiment_name,
    *,
    seed,
):
    config = get_config(
        experiment_name
    )

    definition = get_experiment(
        experiment_name
    )

    data = load_dataset(
        REPO_ROOT
        / "data"
        / f"{experiment_name}.npz"
    )

    validate_split_indices(
        len(data.x),
        data.train_idx,
        data.validation_idx,
        data.test_idx,
    )

    M = SELECTED_M[
        experiment_name
    ]

    epsilon = (
        SELECTED_SPD_EPSILON.get(
            experiment_name,
            config.evaluation.spd_epsilon,
        )
    )

    arff_config = replace(
        config.arff,
        M_min=M,
        M_max=M,
        resampling=True,
        metropolis_test=False,
    )

    train_idx = (
        data.train_idx
    )

    validation_idx = (
        data.validation_idx
    )

    x_train = jnp.asarray(
        data.x[
            train_idx
        ]
    )

    r_train = jnp.asarray(
        data.r[
            train_idx
        ]
    )

    h_train = jnp.asarray(
        data.h[
            train_idx
        ]
    )

    x_validation = jnp.asarray(
        data.x[
            validation_idx
        ]
    )

    r_validation = jnp.asarray(
        data.r[
            validation_idx
        ]
    )

    h_validation = jnp.asarray(
        data.h[
            validation_idx
        ]
    )

    key = jax.random.PRNGKey(
        seed
    )

    start = time.perf_counter()

    key, model, _ = (
        fit_two_stage_arff(
            key,
            x_train,
            r_train,
            h_train,
            K=(
                config.fourier_frequencies
            ),
            diff_type=(
                definition.diff_type
            ),
            config=arff_config,
            fold_seed=(
                config.split.seed
            ),
        )
    )

    jax.block_until_ready(
        model
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    likelihood = gaussian_nll(
        model,
        x_validation,
        r_validation,
        h_validation,
        spd_epsilon=epsilon,
    )

    (
        drift_rmse,
        covariance_rmse,
    ) = true_function_errors(
        model,
        x_validation,
        true_drift=(
            definition.drift
        ),
        true_diffusion_factor=(
            definition.diffusion_factor
        ),
    )

    return {
        "experiment": experiment_name,
        "seed": seed,
        "K": (
            config.fourier_frequencies
        ),
        "M": M,
        "epsilon": epsilon,
        "validation_nll": (
            likelihood.nll
        ),
        "drift_rmse": (
            drift_rmse
        ),
        "covariance_rmse": (
            covariance_rmse
        ),
        "spd_violation_rate": (
            likelihood.spd_violation_rate
        ),
        "min_raw_eigenvalue": (
            likelihood.min_raw_eigenvalue
        ),
        "min_projected_eigenvalue": (
            likelihood.min_projected_eigenvalue
        ),
        "elapsed_seconds": elapsed,
    }


def print_result(
    result,
):
    print(
        f"{result['experiment']:>3s}  "
        f"K={result['K']:4d}  "
        f"M={result['M']:3d}  "
        f"eps={result['epsilon']:.1e}  "
        f"NLL={result['validation_nll']: .8e}  "
        f"drift={result['drift_rmse']:.8e}  "
        f"cov={result['covariance_rmse']:.8e}  "
        f"SPD={result['spd_violation_rate']:.6f}"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=[
            *ALL_EXPERIMENTS,
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.experiment == "all":
        experiments = (
            ALL_EXPERIMENTS
        )
    else:
        experiments = (
            args.experiment,
        )

    print(
        "Selected ARFF validation sanity check"
    )
    print(
        "====================================="
    )

    print(
        f"backend    : "
        f"{jax.default_backend()}"
    )

    print(
        f"seed       : "
        f"{args.seed}"
    )

    print(
        "resampling : True"
    )

    print(
        "Metropolis : False"
    )

    print(
        "test split : NOT USED"
    )

    print()

    results = []

    for experiment_name in experiments:
        print(
            f"running {experiment_name}..."
        )

        result = run_experiment(
            experiment_name,
            seed=args.seed,
        )

        results.append(
            result
        )

        print_result(
            result
        )

        print()

    print(
        "=" * 120
    )

    print(
        "Summary"
    )

    print(
        "=" * 120
    )

    for result in results:
        print_result(
            result
        )


if __name__ == "__main__":
    main()