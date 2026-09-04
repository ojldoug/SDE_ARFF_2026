#!/usr/bin/env python3
"""
Compare ex6 ARFF ridge solves using:

1. default float32 matrix-multiplication precision,
2. highest float32 matrix-multiplication precision,
3. float64 normal equations.

This reproduces the current float32 ARFF frequency trajectory for
Experiment 6, seed 0, and inspects iterations 47--50.

The purpose is to determine whether the observed instability is caused
primarily by reduced GPU matmul precision or whether genuine float64
arithmetic is required.

No production code is modified.
"""

from __future__ import annotations

# Enable float64 for the reference calculation.
# This must happen before substantial JAX work.
import jax

jax.config.update(
    "jax_enable_x64",
    True,
)

from pathlib import Path
import sys

import jax.numpy as jnp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.arff.regression import (
    ARFFModel,
    frequency_amplitude_norm,
)
from src.experiments.config import get_config
from src.experiments.dataset import load_dataset


EXPERIMENT = "ex6"
SEED = 0

START_ITERATION = 47
END_ITERATION = 50


def features(
    omega,
    x,
):
    projection = (
        x @ omega
    )

    return jnp.concatenate(
        [
            jnp.cos(
                projection
            ),
            jnp.sin(
                projection
            ),
        ],
        axis=-1,
    )


def ridge_normal(
    x,
    y,
    omega,
    lambda_reg,
    *,
    precision=None,
):
    """
    Ridge regression via normal equations.

    The optional precision argument controls the precision used by the
    two matrix multiplications constructing Phi^T Phi and Phi^T y.
    """
    phi = features(
        omega,
        x,
    )

    gram = jnp.matmul(
        phi.T,
        phi,
        precision=precision,
    )

    rhs = jnp.matmul(
        phi.T,
        y,
        precision=precision,
    )

    matrix = (
        gram
        + x.shape[0]
        * lambda_reg
        * jnp.eye(
            phi.shape[1],
            dtype=phi.dtype,
        )
    )

    beta = jnp.linalg.solve(
        matrix,
        rhs,
    )

    return beta


def prediction_rmse(
    x,
    y,
    omega,
    beta,
):
    prediction = (
        features(
            omega,
            x,
        )
        @ beta
    )

    return float(
        jnp.sqrt(
            jnp.mean(
                (
                    prediction
                    - y
                )
                ** 2
            )
        )
    )


def ridge_objective(
    x,
    y,
    omega,
    beta,
    lambda_reg,
):
    prediction = (
        features(
            omega,
            x,
        )
        @ beta
    )

    residual = (
        prediction
        - y
    )

    return float(
        jnp.mean(
            residual**2
        )
        + lambda_reg
        * jnp.sum(
            beta**2
        )
    )


def amplitude_pmf(
    amp,
    K,
):
    norm = frequency_amplitude_norm(
        amp,
        K,
    )

    tiny = jnp.finfo(
        norm.dtype
    ).tiny

    weights = jnp.maximum(
        norm,
        tiny,
    )

    return (
        weights
        / jnp.sum(
            weights
        )
    )


def fit32(
    x,
    y,
    omega,
    lambda_reg,
):
    """
    Reproduce the current production float32 ridge solve.

    Deliberately uses the default matmul precision so that the ARFF
    frequency trajectory matches the current implementation.
    """
    return ridge_normal(
        x.astype(
            jnp.float32
        ),
        y.astype(
            jnp.float32
        ),
        omega.astype(
            jnp.float32
        ),
        lambda_reg,
        precision=None,
    )


def one_iteration_float32(
    key,
    model,
    x,
    y,
    *,
    delta,
    lambda_reg,
):
    """
    Reproduce the current production resampling-only ARFF iteration:

        random walk
        -> float32 amplitude fit
        -> amplitude-weighted resampling
        -> float32 amplitude refit

    This preserves the exact float32 stochastic trajectory used by the
    current implementation.
    """
    K = model.omega.shape[1]

    key, mutation_key = (
        jax.random.split(
            key
        )
    )

    proposal = (
        model.omega
        + delta
        * jax.random.normal(
            mutation_key,
            model.omega.shape,
            dtype=model.omega.dtype,
        )
    )

    proposal_amp = fit32(
        x,
        y,
        proposal,
        lambda_reg,
    )

    pmf = amplitude_pmf(
        proposal_amp,
        K,
    )

    key, resampling_key = (
        jax.random.split(
            key
        )
    )

    selected = jax.random.choice(
        resampling_key,
        K,
        shape=(K,),
        replace=True,
        p=pmf,
    )

    omega = proposal[
        :,
        selected,
    ]

    amp = fit32(
        x,
        y,
        omega,
        lambda_reg,
    )

    return (
        key,
        ARFFModel(
            omega=omega,
            amp=amp,
        ),
    )


def main():
    config = get_config(
        EXPERIMENT
    )

    data = load_dataset(
        REPO_ROOT
        / "data"
        / f"{EXPERIMENT}.npz"
    )

    train_idx = (
        data.train_idx
    )

    validation_idx = (
        data.validation_idx
    )

    x32 = jnp.asarray(
        data.x[
            train_idx
        ],
        dtype=jnp.float32,
    )

    r32 = jnp.asarray(
        data.r[
            train_idx
        ],
        dtype=jnp.float32,
    )

    h32 = jnp.asarray(
        data.h[
            train_idx
        ],
        dtype=jnp.float32,
    )

    y32 = (
        r32
        / h32
    )

    xv32 = jnp.asarray(
        data.x[
            validation_idx
        ],
        dtype=jnp.float32,
    )

    rv32 = jnp.asarray(
        data.r[
            validation_idx
        ],
        dtype=jnp.float32,
    )

    hv32 = jnp.asarray(
        data.h[
            validation_idx
        ],
        dtype=jnp.float32,
    )

    yv32 = (
        rv32
        / hv32
    )

    K = (
        config.fourier_frequencies
    )

    delta = (
        config.arff.delta
    )

    lambda_reg = (
        config.arff.lambda_reg
    )

    # Same zero-frequency initialization as production ARFF.
    omega = jnp.zeros(
        (
            x32.shape[1],
            K,
        ),
        dtype=jnp.float32,
    )

    amp = fit32(
        x32,
        y32,
        omega,
        lambda_reg,
    )

    model = ARFFModel(
        omega=omega,
        amp=amp,
    )

    key = jax.random.PRNGKey(
        SEED
    )

    print(
        "Ex6 matrix-multiplication precision diagnostic"
    )
    print(
        "==============================================="
    )

    print(
        f"x64 enabled : "
        f"{jax.config.x64_enabled}"
    )

    print(
        f"seed        : "
        f"{SEED}"
    )

    print(
        f"K           : "
        f"{K}"
    )

    print(
        f"lambda      : "
        f"{lambda_reg:.8e}"
    )

    print(
        f"iterations  : "
        f"{START_ITERATION}--"
        f"{END_ITERATION}"
    )

    print()

    for iteration in range(
        1,
        END_ITERATION + 1,
    ):
        key, model = (
            one_iteration_float32(
                key,
                model,
                x32,
                y32,
                delta=delta,
                lambda_reg=(
                    lambda_reg
                ),
            )
        )

        if (
            iteration
            < START_ITERATION
        ):
            continue

        omega32 = model.omega

        # --------------------------------------------------------
        # 1. Current/default float32 matmul precision.
        # --------------------------------------------------------
        beta32_default = ridge_normal(
            x32,
            y32,
            omega32,
            lambda_reg,
            precision=None,
        )

        # --------------------------------------------------------
        # 2. Highest float32 matmul precision.
        # --------------------------------------------------------
        beta32_highest = ridge_normal(
            x32,
            y32,
            omega32,
            lambda_reg,
            precision=jax.lax.Precision.HIGHEST,
        )

        # --------------------------------------------------------
        # 3. Float64 reference.
        #
        # This still materializes the large float64 feature matrix, but
        # only for four diagnostic iterations. We deliberately omit the
        # augmented lstsq calculation that previously caused the large
        # additional memory allocation.
        # --------------------------------------------------------
        x64 = x32.astype(
            jnp.float64
        )

        y64 = y32.astype(
            jnp.float64
        )

        xv64 = xv32.astype(
            jnp.float64
        )

        yv64 = yv32.astype(
            jnp.float64
        )

        omega64 = omega32.astype(
            jnp.float64
        )

        beta64 = ridge_normal(
            x64,
            y64,
            omega64,
            lambda_reg,
            precision=jax.lax.Precision.HIGHEST,
        )

        # Make sure all computations have completed before reporting.
        jax.block_until_ready(
            beta32_default
        )

        jax.block_until_ready(
            beta32_highest
        )

        jax.block_until_ready(
            beta64
        )

        val32_default = (
            prediction_rmse(
                xv32,
                yv32,
                omega32,
                beta32_default,
            )
        )

        val32_highest = (
            prediction_rmse(
                xv32,
                yv32,
                omega32,
                beta32_highest,
            )
        )

        val64 = (
            prediction_rmse(
                xv64,
                yv64,
                omega64,
                beta64,
            )
        )

        train32_default = (
            prediction_rmse(
                x32,
                y32,
                omega32,
                beta32_default,
            )
        )

        train32_highest = (
            prediction_rmse(
                x32,
                y32,
                omega32,
                beta32_highest,
            )
        )

        train64 = (
            prediction_rmse(
                x64,
                y64,
                omega64,
                beta64,
            )
        )

        objective32_default = (
            ridge_objective(
                x32,
                y32,
                omega32,
                beta32_default,
                lambda_reg,
            )
        )

        objective32_highest = (
            ridge_objective(
                x32,
                y32,
                omega32,
                beta32_highest,
                lambda_reg,
            )
        )

        objective64 = (
            ridge_objective(
                x64,
                y64,
                omega64,
                beta64,
                lambda_reg,
            )
        )

        norm32_default = float(
            jnp.linalg.norm(
                beta32_default
            )
        )

        norm32_highest = float(
            jnp.linalg.norm(
                beta32_highest
            )
        )

        norm64 = float(
            jnp.linalg.norm(
                beta64
            )
        )

        print(
            f"iteration {iteration}"
        )

        print(
            "  validation RMSE"
        )

        print(
            f"    f32 default : "
            f"{val32_default:.8e}"
        )

        print(
            f"    f32 highest : "
            f"{val32_highest:.8e}"
        )

        print(
            f"    f64         : "
            f"{val64:.8e}"
        )

        print(
            "  training RMSE"
        )

        print(
            f"    f32 default : "
            f"{train32_default:.8e}"
        )

        print(
            f"    f32 highest : "
            f"{train32_highest:.8e}"
        )

        print(
            f"    f64         : "
            f"{train64:.8e}"
        )

        print(
            "  coefficient norm"
        )

        print(
            f"    f32 default : "
            f"{norm32_default:.8e}"
        )

        print(
            f"    f32 highest : "
            f"{norm32_highest:.8e}"
        )

        print(
            f"    f64         : "
            f"{norm64:.8e}"
        )

        print(
            "  ridge objective"
        )

        print(
            f"    f32 default : "
            f"{objective32_default:.8e}"
        )

        print(
            f"    f32 highest : "
            f"{objective32_highest:.8e}"
        )

        print(
            f"    f64         : "
            f"{objective64:.8e}"
        )

        print()


if __name__ == "__main__":
    main()