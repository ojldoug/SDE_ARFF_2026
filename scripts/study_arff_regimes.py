#!/usr/bin/env python3
"""
Validation-only ARFF adaptation-regime study.

Experiments:
    ex1, ex2, ex4, ex5

Regimes:
    1. Metropolis only
    2. Resampling only
    3. Resampling + Metropolis
    4. Random walk only

All non-regime ARFF hyperparameters remain fixed at the experiment's
current configuration.

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


REGIMES = (
    ("metropolis_only", False, True),
    ("resampling_only", True, False),
    ("resampling_metropolis", True, True),
    ("random_walk_only", False, False),
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


def drift_rmse(
    model,
    x,
    *,
    true_drift,
):
    x = jnp.asarray(x)

    from src.arff.regression import predict

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


def fit_one_regime(
    *,
    experiment_name,
    regime_name,
    resampling,
    metropolis_test,
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
    validation_idx = data.validation_idx

    arff_config = replace(
        config.arff,
        resampling=resampling,
        metropolis_test=metropolis_test,
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
            resampling=arff_config.resampling,
            metropolis_test=(
                arff_config.metropolis_test
            ),
        )
    )

    key = jax.random.PRNGKey(
        0
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
        compiled_adaptation_step=(
            compiled_step
        ),
    )

    jax.block_until_ready(
        model
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    validation_result = gaussian_nll(
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

    validation_cov_rmse = (
        covariance_rmse(
            model,
            x_validation,
            true_diffusion_factor=(
                definition.diffusion_factor
            ),
        )
    )

    validation_drift_rmse = (
        drift_rmse(
            model,
            x_validation,
            true_drift=(
                definition.drift
            ),
        )
    )

    return {
        "experiment": experiment_name,
        "regime": regime_name,
        "resampling": resampling,
        "metropolis": metropolis_test,
        "validation_nll": (
            validation_result.nll
        ),
        "spd_violation_rate": (
            violation_rate
        ),
        "drift_rmse": (
            validation_drift_rmse
        ),
        "covariance_rmse": (
            validation_cov_rmse
        ),
        "elapsed": elapsed,
    }


def print_result(
    result,
):
    print(
        f"{result['experiment']:4s}  "
        f"{result['regime']:24s}  "
        f"NLL={result['validation_nll']: .8e}  "
        f"SPD={result['spd_violation_rate']:.6f}  "
        f"drift={result['drift_rmse']:.8e}  "
        f"cov={result['covariance_rmse']:.8e}  "
        f"time={result['elapsed']:.2f}s"
    )


def main():
    print(
        "ARFF adaptation-regime validation study"
    )
    print(
        "======================================="
    )
    print()

    all_results = []

    for experiment_name in EXPERIMENTS:
        config = get_config(
            experiment_name
        )

        print(
            f"{experiment_name}"
        )
        print(
            "-" * len(experiment_name)
        )
        print(
            f"K          : "
            f"{config.fourier_frequencies}"
        )
        print(
            f"iterations : "
            f"{config.arff.M_max}"
        )
        print(
            f"lambda     : "
            f"{config.arff.lambda_reg:.8e}"
        )
        print(
            f"delta      : "
            f"{config.arff.delta:.8e}"
        )
        print(
            f"gamma      : "
            f"{config.arff.gamma:.8e}"
        )
        print()

        experiment_results = []

        for (
            regime_name,
            resampling,
            metropolis_test,
        ) in REGIMES:
            result = fit_one_regime(
                experiment_name=(
                    experiment_name
                ),
                regime_name=(
                    regime_name
                ),
                resampling=resampling,
                metropolis_test=(
                    metropolis_test
                ),
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

        print()
        print(
            "ranking by validation NLL"
        )

        for result in sorted(
            experiment_results,
            key=lambda item: item[
                "validation_nll"
            ],
        ):
            print_result(
                result
            )

        print()
        print()

    print(
        "=" * 100
    )
    print(
        "Overall compact summary"
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
            f"{best['regime']}  "
            f"NLL={best['validation_nll']:.8e}  "
            f"SPD={best['spd_violation_rate']:.6f}  "
            f"drift={best['drift_rmse']:.8e}  "
            f"cov={best['covariance_rmse']:.8e}"
        )


if __name__ == "__main__":
    main()