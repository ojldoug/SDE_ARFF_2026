#!/usr/bin/env python3
"""
Validation-only study of ARFF adaptation regimes for Experiment 5.

This compares the four combinations of:
    resampling in {False, True}
    metropolis_test in {False, True}

All other ARFF hyperparameters remain fixed at the canonical ex5 values.

Selection is based only on the canonical validation split.
The test set is not touched.
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


def run_regime(
    *,
    name,
    resampling,
    metropolis_test,
    base_config,
    definition,
    data,
):
    arff_config = replace(
        base_config.arff,
        resampling=resampling,
        metropolis_test=metropolis_test,
    )

    train_idx = data.train_idx
    validation_idx = data.validation_idx

    x_train = jnp.asarray(
        data.x[train_idx]
    )
    r_train = jnp.asarray(
        data.r[train_idx]
    )
    h_train = jnp.asarray(
        data.h[train_idx]
    )

    compiled_step = make_compiled_adaptation_step(
        delta=arff_config.delta,
        lambda_reg=arff_config.lambda_reg,
        gamma=arff_config.gamma,
        resampling=arff_config.resampling,
        metropolis_test=arff_config.metropolis_test,
    )

    key = jax.random.PRNGKey(0)

    start = time.perf_counter()

    key, model, crossfit = fit_two_stage_arff(
        key,
        x_train,
        r_train,
        h_train,
        K=base_config.fourier_frequencies,
        diff_type=definition.diff_type,
        config=arff_config,
        fold_seed=base_config.split.seed,
        compiled_adaptation_step=compiled_step,
    )

    jax.block_until_ready(model)

    elapsed = (
        time.perf_counter()
        - start
    )

    validation_x = data.x[validation_idx]
    validation_r = data.r[validation_idx]
    validation_h = data.h[validation_idx]

    validation_result = gaussian_nll(
        model,
        validation_x,
        validation_r,
        validation_h,
        spd_epsilon=(
            base_config.evaluation.spd_epsilon
        ),
    )

    covariance = raw_covariance(
        model.covariance,
        jnp.asarray(validation_x),
        definition.diff_type,
    )

    violations = np.asarray(
        spd_violation_mask(
            covariance
        )
    )

    cov_rmse = covariance_rmse(
        model,
        validation_x,
        true_diffusion_factor=(
            definition.diffusion_factor
        ),
    )

    covariance_diag = np.diagonal(
        np.asarray(covariance),
        axis1=-2,
        axis2=-1,
    )

    print(name)
    print("-" * len(name))
    print(
        f"resampling              : "
        f"{resampling}"
    )
    print(
        f"Metropolis              : "
        f"{metropolis_test}"
    )
    print(
        f"elapsed                 : "
        f"{elapsed:.3f} s"
    )
    print(
        "validation NLL          : "
        f"{validation_result.nll:.8e}"
    )
    print(
        "validation SPD violation: "
        f"{np.mean(violations):.8f}"
    )
    print(
        "validation covariance RMSE: "
        f"{cov_rmse:.8e}"
    )
    print(
        "min covariance diagonal : "
        f"{np.min(covariance_diag):.8e}"
    )
    print(
        "median covariance diag   : "
        f"{np.median(covariance_diag):.8e}"
    )
    print(
        "max covariance diagonal  : "
        f"{np.max(covariance_diag):.8e}"
    )
    print()

    return {
        "name": name,
        "resampling": resampling,
        "metropolis_test": metropolis_test,
        "validation_nll": validation_result.nll,
        "spd_violation_rate": float(
            np.mean(violations)
        ),
        "covariance_rmse": cov_rmse,
        "elapsed": elapsed,
    }


def main():
    experiment = "ex5"

    config = get_config(
        experiment
    )

    definition = get_experiment(
        experiment
    )

    data = load_dataset(
        REPO_ROOT
        / "data"
        / f"{experiment}.npz"
    )

    print(
        "Experiment 5 ARFF adaptation-regime study"
    )
    print(
        "========================================="
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

    results = []

    for (
        regime_name,
        resampling,
        metropolis_test,
    ) in REGIMES:
        result = run_regime(
            name=regime_name,
            resampling=resampling,
            metropolis_test=metropolis_test,
            base_config=config,
            definition=definition,
            data=data,
        )

        results.append(
            result
        )

    print(
        "Summary"
    )
    print(
        "-------"
    )

    for result in sorted(
        results,
        key=lambda item: item[
            "validation_nll"
        ],
    ):
        print(
            f"{result['name']:24s}  "
            f"NLL={result['validation_nll']: .8e}  "
            f"SPD={result['spd_violation_rate']:.6f}  "
            f"covRMSE={result['covariance_rmse']:.8e}  "
            f"time={result['elapsed']:.2f}s"
        )


if __name__ == "__main__":
    main()