#!/usr/bin/env python3
"""
Capacity diagnostic for Experiment 3.

Compare the current K=1024 ARFF model with the historically used
K=2048 model capacity.

This diagnostic uses only the canonical training and validation splits.
The test split is never evaluated.

The immediate question is whether increasing K from 1024 to 2048
restores Experiment 3 to the expected likelihood / drift-error scale.

ARFF regime:

    resampling=True
    metropolis_test=False

Candidate iteration budgets are intentionally short for the first
capacity check because K=2048 makes every ridge solve much more
expensive.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp


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
from src.experiments.dataset import (
    load_dataset,
    validate_split_indices,
)
from src.experiments.definitions import get_experiment


EXPERIMENT = "ex3"

FREQUENCY_COUNTS = (
    1024,
    2048,
)

ITERATIONS = (
    25,
    50,
)

SEED = 0


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
                (
                    learned
                    - truth
                )
                ** 2
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

    sigma_true = jnp.asarray(
        true_diffusion_factor(x)
    )

    covariance_true = (
        sigma_true
        @ jnp.swapaxes(
            sigma_true,
            -1,
            -2,
        )
    )

    return float(
        jnp.sqrt(
            jnp.mean(
                (
                    learned
                    - covariance_true
                )
                ** 2
            )
        )
    )


def run_candidate(
    *,
    K,
    n_iterations,
    config,
    definition,
    data,
):
    arff_config = replace(
        config.arff,
        M_min=n_iterations,
        M_max=n_iterations,
        resampling=True,
        metropolis_test=False,
    )

    train_idx = data.train_idx
    validation_idx = data.validation_idx

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
        K=K,
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

    likelihood = gaussian_nll(
        model,
        x_validation,
        r_validation,
        h_validation,
        spd_epsilon=(
            config.evaluation.spd_epsilon
        ),
    )

    covariance_raw = raw_covariance(
        model.covariance,
        x_validation,
        definition.diff_type,
    )

    violation_rate = float(
        jnp.mean(
            spd_violation_mask(
                covariance_raw
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
        "K": K,
        "M": n_iterations,
        "nll": likelihood.nll,
        "spd": violation_rate,
        "drift_rmse": drift_error,
        "covariance_rmse": covariance_error,
        "min_raw_eigenvalue": (
            likelihood.min_raw_eigenvalue
        ),
        "time": elapsed,
    }


def print_result(
    result,
):
    print(
        f"K={result['K']:4d}  "
        f"M={result['M']:3d}  "
        f"NLL={result['nll']: .8e}  "
        f"SPD={result['spd']:.6f}  "
        f"drift={result['drift_rmse']:.8e}  "
        f"cov={result['covariance_rmse']:.8e}  "
        f"min-eig={result['min_raw_eigenvalue']:.8e}  "
        f"time={result['time']:.2f}s"
    )


def main():
    config = get_config(
        EXPERIMENT
    )

    definition = get_experiment(
        EXPERIMENT
    )

    data = load_dataset(
        REPO_ROOT
        / "data"
        / f"{EXPERIMENT}.npz"
    )

    validate_split_indices(
        len(data.x),
        data.train_idx,
        data.validation_idx,
        data.test_idx,
    )

    print(
        "Experiment 3 ARFF capacity diagnostic"
    )
    print(
        "====================================="
    )

    print(
        f"backend      : "
        f"{jax.default_backend()}"
    )

    print(
        f"seed         : "
        f"{SEED}"
    )

    print(
        f"train N      : "
        f"{len(data.train_idx)}"
    )

    print(
        f"validation N : "
        f"{len(data.validation_idx)}"
    )

    print(
        f"K values     : "
        f"{FREQUENCY_COUNTS}"
    )

    print(
        f"M values     : "
        f"{ITERATIONS}"
    )

    print(
        "resampling   : True"
    )

    print(
        "Metropolis   : False"
    )

    print(
        "test split   : NOT USED"
    )

    print()

    results = []

    for K in FREQUENCY_COUNTS:
        print(
            f"K={K}"
        )
        print(
            "-" * (
                len(
                    str(K)
                )
                + 2
            )
        )

        for n_iterations in ITERATIONS:
            print(
                f"running "
                f"K={K}, "
                f"M={n_iterations}"
            )

            result = run_candidate(
                K=K,
                n_iterations=n_iterations,
                config=config,
                definition=definition,
                data=data,
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
        "Capacity comparison"
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