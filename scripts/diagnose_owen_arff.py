#!/usr/bin/env python3
"""
Validation-only diagnostic using Owen's documented ARFF hyperparameters.

Purpose
-------
Hold the current reproducible data, train/validation split, corrected
two-stage implementation, cross-fitting, and SPD evaluation fixed while
restoring the ARFF hyperparameters reported in the February 2026 draft.

This is NOT production training and does not access the test split.
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

from src.arff.covariance import (
    raw_covariance,
    spd_violation_mask,
)
from src.arff.evaluation import (
    gaussian_nll,
    true_function_errors,
)
from src.arff.regression import (
    make_compiled_adaptation_step,
)
from src.arff.two_stage import (
    fit_two_stage_arff,
)
from src.experiments.config import (
    get_config,
)
from src.experiments.dataset import (
    load_dataset,
    validate_split_indices,
)
from src.experiments.definitions import (
    get_experiment,
)


# Table 10 of Owen's February 2026 manuscript.
#
# Experiment 8 is intentionally omitted here because Owen used
# DIFFERENT ARFF settings for drift and covariance there. We will
# implement that separately after establishing ex1/ex2 behavior.
OWEN = {
    "ex1": dict(
        delta=0.05,
        lambda_reg=1e-3,
        resampling=True,
        metropolis_test=False,
        M_min=30,
    ),
    "ex2": dict(
        delta=0.01,
        lambda_reg=1e-5,
        resampling=True,
        metropolis_test=False,
        M_min=20,
    ),
    "ex3": dict(
        delta=0.01,
        lambda_reg=1e-2,
        resampling=True,
        metropolis_test=False,
        M_min=50,
    ),
    "ex4": dict(
        delta=0.1,
        lambda_reg=1e-3,
        resampling=True,
        metropolis_test=False,
        M_min=20,
    ),
    "ex5": dict(
        delta=0.1,
        lambda_reg=2e-3,
        resampling=True,
        metropolis_test=False,
        M_min=20,
    ),
    "ex6": dict(
        delta=0.3,
        lambda_reg=1e-3,
        resampling=True,
        metropolis_test=False,
        M_min=50,
    ),
    "ex7": dict(
        delta=0.5,
        lambda_reg=1e-2,
        resampling=False,
        metropolis_test=True,
        M_min=50,
    ),
}


def run_candidate(
    name,
    *,
    seed,
    n_iterations,
):
    config = get_config(name)
    definition = get_experiment(name)

    historical = OWEN[name]

    if n_iterations < historical["M_min"]:
        raise ValueError(
            f"{name}: M={n_iterations} is below Owen's "
            f"reported M_min={historical['M_min']}."
        )

    data = load_dataset(
        REPO_ROOT
        / "data"
        / f"{name}.npz"
    )

    validate_split_indices(
        len(data.x),
        data.train_idx,
        data.validation_idx,
        data.test_idx,
    )

    # Keep all blocker-related/current implementation choices fixed.
    # Restore only Owen's documented ARFF hyperparameters.
    arff_config = replace(
        config.arff,
        M_min=n_iterations,
        M_max=n_iterations,
        delta=historical["delta"],
        lambda_reg=historical["lambda_reg"],
        resampling=historical["resampling"],
        metropolis_test=historical["metropolis_test"],
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

    x_validation = jnp.asarray(
        data.x[validation_idx]
    )
    r_validation = jnp.asarray(
        data.r[validation_idx]
    )
    h_validation = jnp.asarray(
        data.h[validation_idx]
    )

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

    key = jax.random.PRNGKey(seed)

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

    jax.block_until_ready(model)

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

    drift_rmse, covariance_rmse = (
        true_function_errors(
            model,
            x_validation,
            true_drift=definition.drift,
            true_diffusion_factor=(
                definition.diffusion_factor
            ),
        )
    )

    raw = raw_covariance(
        model.covariance,
        x_validation,
        model.diff_type,
    )

    violation_rate = float(
        jnp.mean(
            spd_violation_mask(raw)
        )
    )

    print(
        f"{name} "
        f"seed={seed:2d} "
        f"K={config.fourier_frequencies:4d} "
        f"M={n_iterations:4d} "
        f"delta={arff_config.delta:.3g} "
        f"lambda={arff_config.lambda_reg:.3g} "
        f"resample={arff_config.resampling} "
        f"metro={arff_config.metropolis_test}"
    )

    print(
        f"  validation NLL     : "
        f"{likelihood.nll:.8e}"
    )

    print(
        f"  drift RMSE         : "
        f"{drift_rmse:.8e}"
    )

    print(
        f"  covariance RMSE    : "
        f"{covariance_rmse:.8e}"
    )

    print(
        f"  raw SPD violations : "
        f"{violation_rate:.8e}"
    )

    print(
        f"  elapsed            : "
        f"{elapsed:.3f} s"
    )

    print()


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "experiment",
        choices=tuple(OWEN),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        required=True,
    )

    args = parser.parse_args()

    print(
        "OWEN-PARAMETER ARFF DIAGNOSTIC"
    )
    print(
        "TEST SPLIT IS NOT ACCESSED"
    )
    print()

    for M in args.iterations:
        run_candidate(
            args.experiment,
            seed=args.seed,
            n_iterations=M,
        )


if __name__ == "__main__":
    main()
