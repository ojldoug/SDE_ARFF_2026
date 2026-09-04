#!/usr/bin/env python3
"""
Validation-only study of the ARFF random-walk iteration budget.

The ARFF adaptation regime is fixed to:

    resampling=False
    metropolis_test=False

Only the number of adaptation iterations M is varied.

Experiments:
    ex1, ex2, ex4, ex5

The canonical test split is never used.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.arff.covariance import (
    raw_covariance,
    spd_violation_mask,
)
from src.arff.evaluation import gaussian_nll
from src.arff.regression import (
    make_compiled_adaptation_step,
    predict,
)
from src.arff.two_stage import fit_two_stage_arff
from src.experiments.config import get_config
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment


EXPERIMENTS = (
    "ex1",
    "ex2",
    "ex4",
    "ex5",
)

ITERATIONS = (
    25,
    50,
    100,
    200,
    300,
)


def drift_rmse(
    model,
    x,
    *,
    true_drift,
):
    x = jnp.asarray(x)

    learned = predict(
        model.drift,
        x,
    )

    truth = jnp.asarray(
        true_drift(x)
    )

    return float(
        jnp.sqrt(
            jnp.mean(
                (learned - truth) ** 2
            )
        )
    )


def covariance_rmse(
    model,
    x,
    *,
    true_diffusion_factor,
):
    x = jnp.asarray(x)

    learned = raw_covariance(
        model.covariance,
        x,
        model.diff_type,
    )

    sigma = jnp.asarray(
        true_diffusion_factor(x)
    )

    truth = (
        sigma
        @ jnp.swapaxes(
            sigma,
            -1,
            -2,
        )
    )

    return float(
        jnp.sqrt(
            jnp.mean(
                (learned - truth) ** 2
            )
        )
    )


def fit_one(
    experiment_name,
    n_iterations,
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

    train_idx = data.train_idx
    validation_idx = (
        data.validation_idx
    )

    arff_config = replace(
        config.arff,
        M_min=n_iterations,
        M_max=n_iterations,
        resampling=False,
        metropolis_test=False,
    )

    x_train = jnp.asarray(
        data.x[train_idx]
    )

    r_train = jnp.asarray(
        data.r[train_idx]
    )

    h_train = jnp.asarray(
        data.h[train_idx]
    )

    compiled_step = (
        make_compiled_adaptation_step(
            delta=arff_config.delta,
            lambda_reg=(
                arff_config.lambda_reg
            ),
            gamma=arff_config.gamma,
            resampling=False,
            metropolis_test=False,
        )
    )

    key = jax.random.PRNGKey(
        0
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
            compiled_adaptation_step=(
                compiled_step
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

    x_validation = data.x[
        validation_idx
    ]

    r_validation = data.r[
        validation_idx
    ]

    h_validation = data.h[
        validation_idx
    ]

    evaluation = gaussian_nll(
        model,
        x_validation,
        r_validation,
        h_validation,
        spd_epsilon=(
            config.evaluation.spd_epsilon
        ),
    )

    covariance = raw_covariance(
        model.covariance,
        jnp.asarray(
            x_validation
        ),
        definition.diff_type,
    )

    violations = float(
        np.mean(
            np.asarray(
                spd_violation_mask(
                    covariance
                )
            )
        )
    )

    drift_error = drift_rmse(
        model,
        x_validation,
        true_drift=(
            definition.drift
        ),
    )

    covariance_error = (
        covariance_rmse(
            model,
            x_validation,
            true_diffusion_factor=(
                definition.diffusion_factor
            ),
        )
    )

    return {
        "experiment": (
            experiment_name
        ),
        "iterations": (
            n_iterations
        ),
        "nll": (
            evaluation.nll
        ),
        "spd": (
            violations
        ),
        "drift_rmse": (
            drift_error
        ),
        "covariance_rmse": (
            covariance_error
        ),
        "elapsed": (
            elapsed
        ),
    }


def print_result(
    result,
):
    print(
        f"M={result['iterations']:3d}  "
        f"NLL={result['nll']: .8e}  "
        f"SPD={result['spd']:.6f}  "
        f"drift={result['drift_rmse']:.8e}  "
        f"cov={result['covariance_rmse']:.8e}  "
        f"time={result['elapsed']:.2f}s"
    )


def main():
    print(
        "ARFF random-walk iteration study"
    )
    print(
        "================================"
    )

    print(
        "resampling : False"
    )

    print(
        "Metropolis : False"
    )

    print()

    all_results = []

    for experiment_name in EXPERIMENTS:
        config = get_config(
            experiment_name
        )

        print(
            experiment_name
        )

        print(
            "-" * len(
                experiment_name
            )
        )

        print(
            f"K      : "
            f"{config.fourier_frequencies}"
        )

        print(
            f"delta  : "
            f"{config.arff.delta:.8e}"
        )

        print()

        experiment_results = []

        for n_iterations in ITERATIONS:
            result = fit_one(
                experiment_name,
                n_iterations,
            )

            experiment_results.append(
                result
            )

            all_results.append(
                result
            )

            print_result(
                result
            )

        best = min(
            experiment_results,
            key=lambda item: item[
                "nll"
            ],
        )

        print()

        print(
            "best validation NLL:"
        )

        print_result(
            best
        )

        print()
        print()

    print(
        "=" * 100
    )

    print(
        "Compact summary"
    )

    print(
        "=" * 100
    )

    for experiment_name in EXPERIMENTS:
        subset = [
            result
            for result in all_results
            if result[
                "experiment"
            ]
            == experiment_name
        ]

        best = min(
            subset,
            key=lambda item: item[
                "nll"
            ],
        )

        print(
            f"{experiment_name}: "
            f"M={best['iterations']}  "
            f"NLL={best['nll']:.8e}  "
            f"SPD={best['spd']:.6f}  "
            f"drift={best['drift_rmse']:.8e}  "
            f"cov={best['covariance_rmse']:.8e}"
        )


if __name__ == "__main__":
    main()