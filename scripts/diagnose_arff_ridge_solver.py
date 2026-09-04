#!/usr/bin/env python3
"""
Diagnose numerical stability of the ARFF ridge-regression amplitude solve.

The current ARFF implementation solves ridge regression through the
normal equations

    (Phi^T Phi + N lambda I) beta = Phi^T y.

This diagnostic compares that solution with a direct augmented
least-squares solve,

        [ Phi              ] beta ~= [ y ]
        [ sqrt(N lambda) I ]          [ 0 ]

which avoids explicitly squaring the condition number.

The experiment deliberately introduces duplicate and near-duplicate
Fourier frequencies, mimicking amplitude-weighted resampling.
"""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.arff.regression import (
    fit_amplitudes,
    fourier_features,
)


def direct_ridge_lstsq(
    x,
    y,
    omega,
    lambda_reg,
):
    features = fourier_features(
        omega,
        x,
    )

    n = features.shape[0]
    p = features.shape[1]

    ridge = (
        jnp.sqrt(
            n * lambda_reg
        )
        * jnp.eye(
            p,
            dtype=features.dtype,
        )
    )

    augmented_features = jnp.concatenate(
        [
            features,
            ridge,
        ],
        axis=0,
    )

    augmented_targets = jnp.concatenate(
        [
            y,
            jnp.zeros(
                (p, y.shape[1]),
                dtype=y.dtype,
            ),
        ],
        axis=0,
    )

    beta, _, _, _ = jnp.linalg.lstsq(
        augmented_features,
        augmented_targets,
        rcond=None,
    )

    return beta


def objective(
    x,
    y,
    omega,
    beta,
    lambda_reg,
):
    features = fourier_features(
        omega,
        x,
    )

    residual = (
        features @ beta
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


def prediction_mse(
    x,
    y,
    omega,
    beta,
):
    prediction = (
        fourier_features(
            omega,
            x,
        )
        @ beta
    )

    return float(
        jnp.mean(
            (
                prediction
                - y
            ) ** 2
        )
    )


def main():
    rng = np.random.default_rng(
        123
    )

    n = 2048
    d = 2
    K = 128

    lambda_reg = 1e-3

    x = rng.uniform(
        -1.0,
        1.0,
        size=(n, d),
    ).astype(
        np.float32
    )

    y = (
        np.sin(
            2.0 * x[:, 0]
        )
        + 0.5
        * np.cos(
            3.0 * x[:, 1]
        )
    )[:, None].astype(
        np.float32
    )

    # Start with a moderate set of independent frequencies.
    base_K = 16

    base_omega = rng.normal(
        size=(d, base_K),
    ).astype(
        np.float32
    )

    # Mimic severe particle duplication after resampling.
    selected = rng.integers(
        0,
        base_K,
        size=K,
    )

    omega_duplicate = (
        base_omega[
            :,
            selected
        ]
    )

    # Also construct a near-duplicate population by adding a tiny
    # mutation after resampling.
    omega_near = (
        omega_duplicate
        + 1e-4
        * rng.normal(
            size=omega_duplicate.shape
        ).astype(
            np.float32
        )
    )

    for label, omega in [
        (
            "exact duplicates",
            omega_duplicate,
        ),
        (
            "near duplicates",
            omega_near,
        ),
    ]:
        print()
        print(
            "=" * 72
        )
        print(label)
        print(
            "=" * 72
        )

        x_jax = jnp.asarray(
            x
        )

        y_jax = jnp.asarray(
            y
        )

        omega_jax = jnp.asarray(
            omega
        )

        features = np.asarray(
            fourier_features(
                omega_jax,
                x_jax,
            )
        )

        singular_values = (
            np.linalg.svd(
                features,
                compute_uv=False,
            )
        )

        gram = (
            features.T
            @ features
        )

        ridge_gram = (
            gram
            + n
            * lambda_reg
            * np.eye(
                gram.shape[0],
                dtype=gram.dtype,
            )
        )

        cond_features = (
            singular_values[0]
            / max(
                singular_values[-1],
                np.finfo(
                    singular_values.dtype
                ).tiny,
            )
        )

        cond_ridge_gram = (
            np.linalg.cond(
                ridge_gram
            )
        )

        beta_normal = (
            fit_amplitudes(
                x_jax,
                y_jax,
                omega_jax,
                lambda_reg,
            )
        )

        beta_direct = (
            direct_ridge_lstsq(
                x_jax,
                y_jax,
                omega_jax,
                lambda_reg,
            )
        )

        beta_normal = np.asarray(
            beta_normal
        )

        beta_direct = np.asarray(
            beta_direct
        )

        print(
            "feature condition estimate : "
            f"{cond_features:.8e}"
        )

        print(
            "regularized Gram condition : "
            f"{cond_ridge_gram:.8e}"
        )

        print(
            "normal-equation finite      : "
            f"{np.all(np.isfinite(beta_normal))}"
        )

        print(
            "direct-lstsq finite         : "
            f"{np.all(np.isfinite(beta_direct))}"
        )

        print(
            "normal-equation |beta|      : "
            f"{np.linalg.norm(beta_normal):.8e}"
        )

        print(
            "direct-lstsq |beta|         : "
            f"{np.linalg.norm(beta_direct):.8e}"
        )

        print(
            "relative beta difference    : "
            f"{np.linalg.norm(beta_normal - beta_direct) / max(np.linalg.norm(beta_direct), 1e-30):.8e}"
        )

        print(
            "normal prediction MSE       : "
            f"{prediction_mse(x_jax, y_jax, omega_jax, beta_normal):.8e}"
        )

        print(
            "direct prediction MSE       : "
            f"{prediction_mse(x_jax, y_jax, omega_jax, beta_direct):.8e}"
        )

        print(
            "normal ridge objective      : "
            f"{objective(x_jax, y_jax, omega_jax, beta_normal, lambda_reg):.8e}"
        )

        print(
            "direct ridge objective      : "
            f"{objective(x_jax, y_jax, omega_jax, beta_direct, lambda_reg):.8e}"
        )


if __name__ == "__main__":
    main()