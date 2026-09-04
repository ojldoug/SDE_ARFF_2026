#!/usr/bin/env python3
"""
Oracle diagnostic for Experiment 3.

Evaluate the canonical ex3 data using the known true drift and true
diffusion factor, without fitting ARFF or Adam.

This determines whether the current ex3 dataset and likelihood
evaluation are mutually consistent.

For correctly generated Euler--Maruyama data,

    r = h f(x) + sqrt(h) sigma(x) xi,

the oracle Mahalanobis quadratic should have mean approximately equal
to the state dimension D.

No training or test-set model selection is performed.
"""

from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.dataset import (
    load_dataset,
    validate_split_indices,
)
from src.experiments.definitions import get_experiment


EXPERIMENT = "ex3"


def oracle_statistics(
    x,
    r,
    h,
    *,
    true_drift,
    true_diffusion_factor,
):
    x = jnp.asarray(x)
    r = jnp.asarray(r)
    h = jnp.asarray(h)

    drift = jnp.asarray(
        true_drift(x)
    )

    sigma = jnp.asarray(
        true_diffusion_factor(x)
    )

    covariance = (
        sigma
        @ jnp.swapaxes(
            sigma,
            -1,
            -2,
        )
    )

    residual = (
        r
        - h * drift
    )

    variance = (
        h[:, :, None]
        * covariance
    )

    sign, logdet = jnp.linalg.slogdet(
        variance
    )

    solution = jnp.linalg.solve(
        variance,
        residual[:, :, None],
    )[:, :, 0]

    quadratic = jnp.sum(
        residual
        * solution,
        axis=1,
    )

    dimension = r.shape[1]

    nll_per_sample = (
        0.5
        * (
            quadratic
            + logdet
            + dimension
            * jnp.log(
                2.0
                * jnp.pi
            )
        )
    )

    covariance_eigenvalues = (
        jnp.linalg.eigvalsh(
            covariance
        )
    )

    return {
        "nll": float(
            jnp.mean(
                nll_per_sample
            )
        ),
        "quadratic_mean": float(
            jnp.mean(
                quadratic
            )
        ),
        "quadratic_std": float(
            jnp.std(
                quadratic
            )
        ),
        "logdet_mean": float(
            jnp.mean(
                logdet
            )
        ),
        "positive_determinant_fraction": float(
            jnp.mean(
                sign > 0
            )
        ),
        "true_covariance_min_eigenvalue": float(
            jnp.min(
                covariance_eigenvalues
            )
        ),
        "true_covariance_max_eigenvalue": float(
            jnp.max(
                covariance_eigenvalues
            )
        ),
        "residual_rmse": float(
            jnp.sqrt(
                jnp.mean(
                    residual**2
                )
            )
        ),
    }


def print_statistics(
    label,
    statistics,
    dimension,
):
    print(label)
    print("-" * len(label))

    print(
        f"NLL                         : "
        f"{statistics['nll']:.8e}"
    )

    print(
        f"mean Mahalanobis quadratic  : "
        f"{statistics['quadratic_mean']:.8e}"
    )

    print(
        f"expected quadratic mean     : "
        f"{dimension:.8e}"
    )

    print(
        f"quadratic std               : "
        f"{statistics['quadratic_std']:.8e}"
    )

    print(
        f"mean log determinant        : "
        f"{statistics['logdet_mean']:.8e}"
    )

    print(
        f"positive determinant frac.  : "
        f"{statistics['positive_determinant_fraction']:.8f}"
    )

    print(
        f"min true covariance eig     : "
        f"{statistics['true_covariance_min_eigenvalue']:.8e}"
    )

    print(
        f"max true covariance eig     : "
        f"{statistics['true_covariance_max_eigenvalue']:.8e}"
    )

    print(
        f"oracle residual RMSE        : "
        f"{statistics['residual_rmse']:.8e}"
    )

    print()


def main():
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
        "Experiment 3 oracle likelihood diagnostic"
    )
    print(
        "========================================="
    )

    print(
        f"state dimension : "
        f"{definition.n_dimensions}"
    )

    print(
        f"diffusion type  : "
        f"{definition.diff_type}"
    )

    print(
        f"total N         : "
        f"{len(data.x)}"
    )

    print(
        f"train N         : "
        f"{len(data.train_idx)}"
    )

    print(
        f"validation N    : "
        f"{len(data.validation_idx)}"
    )

    print(
        f"test N          : "
        f"{len(data.test_idx)}"
    )

    print(
        f"h min           : "
        f"{np.min(data.h):.8e}"
    )

    print(
        f"h mean          : "
        f"{np.mean(data.h):.8e}"
    )

    print(
        f"h max           : "
        f"{np.max(data.h):.8e}"
    )

    print()

    for label, idx in [
        (
            "train",
            data.train_idx,
        ),
        (
            "validation",
            data.validation_idx,
        ),
        (
            "test",
            data.test_idx,
        ),
    ]:
        statistics = oracle_statistics(
            data.x[idx],
            data.r[idx],
            data.h[idx],
            true_drift=(
                definition.drift
            ),
            true_diffusion_factor=(
                definition.diffusion_factor
            ),
        )

        print_statistics(
            label,
            statistics,
            definition.n_dimensions,
        )


if __name__ == "__main__":
    main()