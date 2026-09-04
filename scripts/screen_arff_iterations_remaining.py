#!/usr/bin/env python3
"""
Seed-0 validation screening of ARFF iteration budgets for the remaining
experiments.

The ARFF adaptation regime is fixed to

    resampling=True
    metropolis_test=False

Candidate iteration budgets:

    M in {25, 50, 100}

Experiments:

    ex3, ex6, ex7, ex8

The canonical test split is never used.

This is a screening step only. Once promising iteration budgets are
identified, they should be checked across multiple tuning seeds before
being frozen for the production experiments.
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
    "ex3",
    "ex6",
    "ex7",
    "ex8",
)

ITERATIONS = (
    25,
    50,
    100,
)

SEED = 0


def drift_rmse(
    model,
    x,
    *,
    true_drift,
):
    """
    Validation drift RMSE.

    Diagnostic only. Candidate M is ranked by validation NLL.
    """
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
    """
    Validation covariance RMSE before SPD projection.

    Diagnostic only. Candidate M is ranked by validation NLL.
    """
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


def run_candidate(
    *,
    experiment_name: str,
    n_iterations: int,
    config,
    definition,
    data,
):
    train_idx = data.train_idx
    validation_idx = data.validation_idx

    arff_config = replace(
        config.arff,
        M_min=n_iterations,
        M_max=n_iterations,
        resampling=True,
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

    x_validation = data.x[
        validation_idx
    ]

    r_validation = data.r[
        validation_idx
    ]

    h_validation = data.h[
        validation_idx
    ]

    compiled_step = (
        make_compiled_adaptation_step(
            delta=arff_config.delta,
            lambda_reg=arff_config.lambda_reg,
            gamma=arff_config.gamma,
            resampling=True,
            metropolis_test=False,
        )
    )

    key = jax.random.PRNGKey(
        SEED
    )

    start = time.perf_counter()

    key, model, _ = fit_two_stage_arff(
        key,
        x_train,
        r_train,
        h_train,
        K=config.fourier_frequencies,
        diff_type=definition.diff_type,
        config=arff_config,
        fold_seed=config.split.seed,
        compiled_adaptation_step=compiled_step,
    )

    jax.block_until_ready(
        model
    )

    elapsed = (
        time.perf_counter()
        - start
    )

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

    violation_rate = float(
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
        true_drift=definition.drift,
    )

    covariance_error = covariance_rmse(
        model,
        x_validation,
        true_diffusion_factor=(
            definition.diffusion_factor
        ),
    )

    return {
        "experiment": experiment_name,
        "iterations": n_iterations,
        "validation_nll": evaluation.nll,
        "spd_violation_rate": violation_rate,
        "drift_rmse": drift_error,
        "covariance_rmse": covariance_error,
        "elapsed": elapsed,
    }


def print_result(
    result,
):
    print(
        f"M={result['iterations']:3d}  "
        f"NLL={result['validation_nll']: .8e}  "
        f"SPD={result['spd_violation_rate']:.6f}  "
        f"drift={result['drift_rmse']:.8e}  "
        f"cov={result['covariance_rmse']:.8e}  "
        f"time={result['elapsed']:.2f}s"
    )


def main():
    print(
        "ARFF remaining-experiment iteration screen"
    )
    print(
        "=========================================="
    )
    print(
        "resampling : True"
    )
    print(
        "Metropolis : False"
    )
    print(
        f"seed       : {SEED}"
    )
    print(
        f"M values   : {ITERATIONS}"
    )
    print()

    all_results = []

    for experiment_name in EXPERIMENTS:
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
        print(
            f"gamma  : "
            f"{config.arff.gamma:.8e}"
        )
        print(
            f"lambda : "
            f"{config.arff.lambda_reg:.8e}"
        )
        print(
            f"train N: "
            f"{len(data.train_idx)}"
        )
        print(
            f"val N  : "
            f"{len(data.validation_idx)}"
        )
        print()

        experiment_results = []

        for n_iterations in ITERATIONS:
            result = run_candidate(
                experiment_name=experiment_name,
                n_iterations=n_iterations,
                config=config,
                definition=definition,
                data=data,
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
                "validation_nll"
            ],
        )

        print()
        print(
            "best seed-0 validation NLL:"
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
        "Screening summary"
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
                "validation_nll"
            ],
        )

        print(
            f"{experiment_name}: "
            f"M={best['iterations']}  "
            f"NLL={best['validation_nll']:.8e}  "
            f"SPD={best['spd_violation_rate']:.6f}  "
            f"drift={best['drift_rmse']:.8e}  "
            f"cov={best['covariance_rmse']:.8e}"
        )


if __name__ == "__main__":
    main()