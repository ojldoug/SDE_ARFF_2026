#!/usr/bin/env python3
"""
Diagnose Experiment 3 drift regression independently of covariance
learning.

Compare:

1. zero drift,
2. zero-frequency ridge regression,
3. fixed standard-normal random Fourier frequencies,
4. adapted ARFF frequencies.

Only canonical training and validation data are used.
"""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.arff.regression import (
    ARFFModel,
    fit_amplitudes,
    fit_arff,
    predict,
)
from src.experiments.config import get_config
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment


EXPERIMENT = "ex3"
K = 1024
SEED = 0
M = 50


def rmse(
    prediction,
    truth,
):
    return float(
        jnp.sqrt(
            jnp.mean(
                (prediction - truth) ** 2
            )
        )
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

    y_train = (
        r_train
        / h_train
    )

    drift_train_true = jnp.asarray(
        definition.drift(
            x_train
        )
    )

    drift_validation_true = jnp.asarray(
        definition.drift(
            x_validation
        )
    )

    print(
        "Experiment 3 drift diagnostic"
    )
    print(
        "============================="
    )

    print(
        f"train N      : {len(train_idx)}"
    )

    print(
        f"validation N : {len(validation_idx)}"
    )

    print(
        f"dimension    : {x_train.shape[1]}"
    )

    print(
        f"K            : {K}"
    )

    print(
        f"M            : {M}"
    )

    print()

    # ------------------------------------------------------------
    # Scale of the true drift and noisy regression target.
    # ------------------------------------------------------------
    zero_validation_rmse = rmse(
        jnp.zeros_like(
            drift_validation_true
        ),
        drift_validation_true,
    )

    target_noise_rmse = rmse(
        y_train,
        drift_train_true,
    )

    print(
        "problem scale"
    )
    print(
        "-------------"
    )

    print(
        "true drift RMS             : "
        f"{zero_validation_rmse:.8e}"
    )

    print(
        "RMSE(r/h, true drift)      : "
        f"{target_noise_rmse:.8e}"
    )

    print()

    # ------------------------------------------------------------
    # Zero-frequency ridge model.
    # ------------------------------------------------------------
    omega_zero = jnp.zeros(
        (
            x_train.shape[1],
            K,
        ),
        dtype=x_train.dtype,
    )

    amp_zero = fit_amplitudes(
        x_train,
        y_train,
        omega_zero,
        config.arff.lambda_reg,
    )

    zero_frequency_model = ARFFModel(
        omega=omega_zero,
        amp=amp_zero,
    )

    zero_frequency_prediction = predict(
        zero_frequency_model,
        x_validation,
    )

    zero_frequency_rmse = rmse(
        zero_frequency_prediction,
        drift_validation_true,
    )

    # ------------------------------------------------------------
    # Fixed standard-normal random Fourier features.
    # ------------------------------------------------------------
    key = jax.random.PRNGKey(
        SEED
    )

    omega_random = jax.random.normal(
        key,
        shape=(
            x_train.shape[1],
            K,
        ),
        dtype=x_train.dtype,
    )

    amp_random = fit_amplitudes(
        x_train,
        y_train,
        omega_random,
        config.arff.lambda_reg,
    )

    random_model = ARFFModel(
        omega=omega_random,
        amp=amp_random,
    )

    random_prediction = predict(
        random_model,
        x_validation,
    )

    random_rmse = rmse(
        random_prediction,
        drift_validation_true,
    )

    # ------------------------------------------------------------
    # Production ARFF drift regression.
    # ------------------------------------------------------------
    key = jax.random.PRNGKey(
        SEED
    )

    key, adapted_model = fit_arff(
        key,
        x_train,
        y_train,
        K=K,
        n_iterations=M,
        lambda_reg=config.arff.lambda_reg,
        gamma=config.arff.gamma,
        delta=config.arff.delta,
        resampling=True,
        metropolis_test=False,
    )

    jax.block_until_ready(
        adapted_model
    )

    adapted_prediction = predict(
        adapted_model,
        x_validation,
    )

    adapted_rmse = rmse(
        adapted_prediction,
        drift_validation_true,
    )

    # ------------------------------------------------------------
    # Frequency statistics.
    # ------------------------------------------------------------
    random_norms = jnp.linalg.norm(
        omega_random,
        axis=0,
    )

    adapted_norms = jnp.linalg.norm(
        adapted_model.omega,
        axis=0,
    )

    print(
        "validation drift RMSE"
    )
    print(
        "---------------------"
    )

    print(
        "zero predictor             : "
        f"{zero_validation_rmse:.8e}"
    )

    print(
        "zero-frequency ridge       : "
        f"{zero_frequency_rmse:.8e}"
    )

    print(
        "fixed N(0,I) frequencies   : "
        f"{random_rmse:.8e}"
    )

    print(
        "adapted ARFF frequencies   : "
        f"{adapted_rmse:.8e}"
    )

    print()

    print(
        "frequency norms"
    )
    print(
        "---------------"
    )

    print(
        "random median / mean / max : "
        f"{float(jnp.median(random_norms)):.6f} / "
        f"{float(jnp.mean(random_norms)):.6f} / "
        f"{float(jnp.max(random_norms)):.6f}"
    )

    print(
        "ARFF median / mean / max   : "
        f"{float(jnp.median(adapted_norms)):.6f} / "
        f"{float(jnp.mean(adapted_norms)):.6f} / "
        f"{float(jnp.max(adapted_norms)):.6f}"
    )


if __name__ == "__main__":
    main()