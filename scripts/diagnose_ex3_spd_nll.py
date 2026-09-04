#!/usr/bin/env python3
"""
Sweep the SPD eigenvalue floor for the current Experiment-3 ARFF model.

The ARFF model is fitted exactly once using the current canonical
training procedure. The fitted model is then evaluated on the canonical
validation split using several projection floors

    Sigma_eps = Q diag(max(lambda_i, eps)) Q^T.

This diagnostic answers whether the catastrophic projected NLL is
caused primarily by choosing an unrealistically small covariance floor.

The test split is never evaluated.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import sys

import jax
import jax.numpy as jnp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.arff.covariance import raw_covariance
from src.arff.regression import predict
from src.arff.two_stage import fit_two_stage_arff
from src.experiments.config import get_config
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment


EXPERIMENT = "ex3"
SEED = 0
M = 50

EPSILONS = (
    1e-6,
    1e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
)


def project_spd(
    covariance,
    epsilon,
):
    eigenvalues, eigenvectors = (
        jnp.linalg.eigh(
            covariance
        )
    )

    projected_eigenvalues = (
        jnp.maximum(
            eigenvalues,
            epsilon,
        )
    )

    projected = (
        eigenvectors
        * projected_eigenvalues[
            :,
            None,
            :,
        ]
    ) @ jnp.swapaxes(
        eigenvectors,
        -1,
        -2,
    )

    return (
        projected,
        eigenvalues,
        projected_eigenvalues,
    )


def per_sample_nll(
    model,
    x,
    r,
    h,
    *,
    epsilon,
):
    x = jnp.asarray(x)
    r = jnp.asarray(r)
    h = jnp.asarray(h)

    drift = predict(
        model.drift,
        x,
    )

    covariance_raw = raw_covariance(
        model.covariance,
        x,
        model.diff_type,
    )

    (
        covariance,
        raw_eigenvalues,
        projected_eigenvalues,
    ) = project_spd(
        covariance_raw,
        epsilon,
    )

    residual = (
        r
        - h * drift
    )

    variance = (
        h[:, :, None]
        * covariance
    )

    sign, logdet = (
        jnp.linalg.slogdet(
            variance
        )
    )

    solution = (
        jnp.linalg.solve(
            variance,
            residual[:, :, None],
        )[
            :,
            :,
            0,
        ]
    )

    quadratic = jnp.sum(
        residual
        * solution,
        axis=1,
    )

    dimension = r.shape[1]

    losses = (
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

    raw_min_eigenvalue = (
        raw_eigenvalues[
            :,
            0,
        ]
    )

    projected_min_eigenvalue = (
        projected_eigenvalues[
            :,
            0,
        ]
    )

    violation = (
        raw_min_eigenvalue
        <= 0.0
    )

    if not bool(
        jnp.all(
            sign > 0.0
        )
    ):
        raise RuntimeError(
            "SPD projection produced "
            "a non-positive determinant."
        )

    return {
        "losses": losses,
        "quadratic": quadratic,
        "logdet": logdet,
        "violation": violation,
        "raw_min_eigenvalue": (
            raw_min_eigenvalue
        ),
        "projected_min_eigenvalue": (
            projected_min_eigenvalue
        ),
    }


def summarize(
    epsilon,
    terms,
):
    losses = terms[
        "losses"
    ]

    violation = terms[
        "violation"
    ]

    violating_losses = (
        losses[
            violation
        ]
    )

    valid_losses = (
        losses[
            ~violation
        ]
    )

    return {
        "epsilon": epsilon,
        "mean_nll": float(
            jnp.mean(
                losses
            )
        ),
        "median_nll": float(
            jnp.median(
                losses
            )
        ),
        "q95_nll": float(
            jnp.quantile(
                losses,
                0.95,
            )
        ),
        "q99_nll": float(
            jnp.quantile(
                losses,
                0.99,
            )
        ),
        "max_nll": float(
            jnp.max(
                losses
            )
        ),
        "mean_valid_nll": float(
            jnp.mean(
                valid_losses
            )
        ),
        "mean_violating_nll": float(
            jnp.mean(
                violating_losses
            )
        ),
        "violation_rate": float(
            jnp.mean(
                violation
            )
        ),
        "min_raw_eigenvalue": float(
            jnp.min(
                terms[
                    "raw_min_eigenvalue"
                ]
            )
        ),
        "min_projected_eigenvalue": float(
            jnp.min(
                terms[
                    "projected_min_eigenvalue"
                ]
            )
        ),
        "mean_quadratic": float(
            jnp.mean(
                terms[
                    "quadratic"
                ]
            )
        ),
        "mean_logdet": float(
            jnp.mean(
                terms[
                    "logdet"
                ]
            )
        ),
    }


def print_result(
    result,
):
    print(
        f"epsilon={result['epsilon']:.1e}"
    )

    print(
        "  mean NLL                 : "
        f"{result['mean_nll']:.8e}"
    )

    print(
        "  median NLL               : "
        f"{result['median_nll']:.8e}"
    )

    print(
        "  95% NLL quantile         : "
        f"{result['q95_nll']:.8e}"
    )

    print(
        "  99% NLL quantile         : "
        f"{result['q99_nll']:.8e}"
    )

    print(
        "  max NLL                  : "
        f"{result['max_nll']:.8e}"
    )

    print(
        "  mean NLL, raw SPD        : "
        f"{result['mean_valid_nll']:.8e}"
    )

    print(
        "  mean NLL, raw violating  : "
        f"{result['mean_violating_nll']:.8e}"
    )

    print(
        "  raw violation rate       : "
        f"{result['violation_rate']:.8f}"
    )

    print(
        "  min raw eigenvalue       : "
        f"{result['min_raw_eigenvalue']:.8e}"
    )

    print(
        "  min projected eigenvalue : "
        f"{result['min_projected_eigenvalue']:.8e}"
    )

    print(
        "  mean quadratic           : "
        f"{result['mean_quadratic']:.8e}"
    )

    print(
        "  mean log determinant     : "
        f"{result['mean_logdet']:.8e}"
    )

    print()


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

    arff_config = replace(
        config.arff,
        M_min=M,
        M_max=M,
        resampling=True,
        metropolis_test=False,
    )

    train_idx = data.train_idx
    validation_idx = (
        data.validation_idx
    )

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

    key = jax.random.PRNGKey(
        SEED
    )

    print(
        "Experiment 3 SPD-floor diagnostic"
    )
    print(
        "================================="
    )

    print(
        f"backend     : "
        f"{jax.default_backend()}"
    )

    print(
        f"K           : "
        f"{config.fourier_frequencies}"
    )

    print(
        f"M           : "
        f"{M}"
    )

    print(
        f"seed        : "
        f"{SEED}"
    )

    print(
        f"validation N: "
        f"{len(validation_idx)}"
    )

    print(
        "test split  : NOT USED"
    )

    print()

    print(
        "Fitting current ARFF model once..."
    )

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
        )
    )

    jax.block_until_ready(
        model
    )

    print(
        "fit complete."
    )

    print()

    results = []

    for epsilon in EPSILONS:
        terms = per_sample_nll(
            model,
            x_validation,
            r_validation,
            h_validation,
            epsilon=epsilon,
        )

        result = summarize(
            epsilon,
            terms,
        )

        results.append(
            result
        )

        print_result(
            result
        )

    best = min(
        results,
        key=lambda item: item[
            "mean_nll"
        ],
    )

    print(
        "=" * 80
    )

    print(
        "Best validation mean NLL"
    )

    print(
        "=" * 80
    )

    print_result(
        best
    )


if __name__ == "__main__":
    main()