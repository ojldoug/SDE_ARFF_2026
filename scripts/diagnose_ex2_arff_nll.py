#!/usr/bin/env python3
"""
Diagnose per-sample ARFF Gaussian NLL contributions for Experiment 2.

This script reproduces the canonical seed-0 ARFF fit and inspects the
test-set likelihood sample by sample. It does not modify the benchmark
implementation or select any hyperparameter.

The purpose is to determine whether a very small number of covariance
predictions requiring SPD projection dominate the mean test NLL.
"""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.arff.covariance import (
    project_spd,
    raw_covariance,
    spd_violation_mask,
)
from src.arff.regression import (
    make_compiled_adaptation_step,
    predict,
)
from src.arff.two_stage import fit_two_stage_arff
from src.experiments.config import get_config
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment


def per_sample_nll(
    model,
    x,
    r,
    h,
    *,
    spd_epsilon,
):
    """
    Return all intermediate quantities used in Gaussian NLL evaluation.
    """
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

    violations = spd_violation_mask(
        covariance_raw
    )

    covariance_projected = project_spd(
        covariance_raw,
        epsilon=spd_epsilon,
    )

    residual = (
        r
        - h * drift
    )

    variance = (
        covariance_projected
        * h[:, :, None]
    )

    sign, logdet = jnp.linalg.slogdet(
        variance
    )

    if not bool(
        jnp.all(sign > 0)
    ):
        raise RuntimeError(
            "Projected covariance produced a "
            "non-positive determinant."
        )

    solution = jnp.linalg.solve(
        variance,
        residual[:, :, None],
    )[:, :, 0]

    quadratic = jnp.sum(
        residual * solution,
        axis=1,
    )

    dimension = r.shape[1]

    losses = 0.5 * (
        quadratic
        + logdet
        + dimension
        * jnp.log(
            2.0 * jnp.pi
        )
    )

    raw_eigenvalues = jnp.linalg.eigvalsh(
        covariance_raw
    )

    projected_eigenvalues = jnp.linalg.eigvalsh(
        covariance_projected
    )

    return {
        "loss": np.asarray(losses),
        "quadratic": np.asarray(quadratic),
        "logdet": np.asarray(logdet),
        "violations": np.asarray(violations),
        "raw_eigenvalues": np.asarray(
            raw_eigenvalues
        ),
        "projected_eigenvalues": np.asarray(
            projected_eigenvalues
        ),
        "residual": np.asarray(residual),
        "covariance_raw": np.asarray(
            covariance_raw
        ),
        "covariance_projected": np.asarray(
            covariance_projected
        ),
    }


def summarize_split(
    label,
    diagnostics,
):
    losses = diagnostics["loss"]
    quadratic = diagnostics["quadratic"]
    violations = diagnostics["violations"]

    print(label)
    print("-" * len(label))

    print(
        f"N                       : "
        f"{len(losses)}"
    )

    print(
        f"mean NLL                : "
        f"{np.mean(losses):.8e}"
    )

    print(
        f"median NLL              : "
        f"{np.median(losses):.8e}"
    )

    print(
        f"min NLL                 : "
        f"{np.min(losses):.8e}"
    )

    print(
        f"max NLL                 : "
        f"{np.max(losses):.8e}"
    )

    print(
        f"mean quadratic          : "
        f"{np.mean(quadratic):.8e}"
    )

    print(
        f"max quadratic           : "
        f"{np.max(quadratic):.8e}"
    )

    print(
        f"SPD violations          : "
        f"{np.sum(violations)}"
    )

    print(
        f"SPD violation rate      : "
        f"{np.mean(violations):.8e}"
    )

    for percentile in [
        50.0,
        90.0,
        95.0,
        99.0,
        99.5,
        99.9,
        100.0,
    ]:
        value = np.percentile(
            losses,
            percentile,
        )

        print(
            f"NLL p{percentile:5.1f}             : "
            f"{value:.8e}"
        )

    print()


def print_worst_samples(
    diagnostics,
    *,
    n_worst=10,
):
    losses = diagnostics["loss"]
    quadratic = diagnostics["quadratic"]
    logdet = diagnostics["logdet"]
    violations = diagnostics["violations"]

    raw_eigenvalues = diagnostics[
        "raw_eigenvalues"
    ]

    projected_eigenvalues = diagnostics[
        "projected_eigenvalues"
    ]

    covariance_raw = diagnostics[
        "covariance_raw"
    ]

    residual = diagnostics[
        "residual"
    ]

    worst = np.argsort(
        losses
    )[::-1][:n_worst]

    print(
        f"Worst {n_worst} test samples"
    )

    print(
        "-----------------------------"
    )

    for rank, idx in enumerate(
        worst,
        start=1,
    ):
        print(
            f"rank {rank}, "
            f"index {idx}"
        )

        print(
            f"  NLL              : "
            f"{losses[idx]:.8e}"
        )

        print(
            f"  quadratic        : "
            f"{quadratic[idx]:.8e}"
        )

        print(
            f"  logdet           : "
            f"{logdet[idx]:.8e}"
        )

        print(
            f"  raw SPD violation: "
            f"{bool(violations[idx])}"
        )

        print(
            "  raw eigenvalues  : "
            f"{raw_eigenvalues[idx]}"
        )

        print(
            "  projected eigs   : "
            f"{projected_eigenvalues[idx]}"
        )

        print(
            "  residual         : "
            f"{residual[idx]}"
        )

        print(
            "  raw covariance   :"
        )

        print(
            covariance_raw[idx]
        )

        print()


def trimmed_mean_report(
    losses,
):
    """
    Diagnostic only: show how much the upper tail contributes.

    These trimmed means are NOT proposed benchmark metrics.
    """
    ordered = np.sort(
        losses
    )

    print(
        "Influence of upper NLL tail"
    )

    print(
        "---------------------------"
    )

    print(
        f"ordinary mean        : "
        f"{np.mean(ordered):.8e}"
    )

    for n_remove in [
        1,
        2,
        5,
        10,
    ]:
        if len(ordered) > n_remove:
            trimmed = ordered[
                :-n_remove
            ]

            print(
                f"mean without worst "
                f"{n_remove:2d}: "
                f"{np.mean(trimmed):.8e}"
            )

    print()


def epsilon_diagnostic(
    model,
    x,
    r,
    h,
):
    """
    Sensitivity diagnostic only.

    This does NOT select epsilon. It simply shows whether the observed
    likelihood pathology is caused by the eigenvalue projection floor.
    """
    print(
        "SPD epsilon sensitivity "
        "(diagnostic only)"
    )

    print(
        "----------------------------------------"
    )

    for epsilon in [
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
    ]:
        diagnostics = per_sample_nll(
            model,
            x,
            r,
            h,
            spd_epsilon=epsilon,
        )

        losses = diagnostics[
            "loss"
        ]

        print(
            f"epsilon={epsilon:.0e}  "
            f"mean={np.mean(losses): .8e}  "
            f"median={np.median(losses): .8e}  "
            f"max={np.max(losses): .8e}"
        )

    print()


def main():
    name = "ex2"
    seed = 0

    config = get_config(
        name
    )

    definition = get_experiment(
        name
    )

    data = load_dataset(
        REPO_ROOT
        / "data"
        / f"{name}.npz"
    )

    train_idx = data.train_idx
    validation_idx = (
        data.validation_idx
    )
    test_idx = data.test_idx

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

    key = jax.random.PRNGKey(
        seed
    )

    compiled_adaptation_step = (
        make_compiled_adaptation_step(
            delta=config.arff.delta,
            lambda_reg=(
                config.arff.lambda_reg
            ),
            gamma=config.arff.gamma,
            resampling=(
                config.arff.resampling
            ),
            metropolis_test=(
                config.arff.metropolis_test
            ),
        )
    )

    print(
        "Fitting canonical ex2 ARFF model..."
    )

    key, model, _ = fit_two_stage_arff(
        key,
        x_train,
        r_train,
        h_train,
        K=config.fourier_frequencies,
        diff_type=(
            definition.diff_type
        ),
        config=config.arff,
        fold_seed=(
            config.split.seed
        ),
        compiled_adaptation_step=(
            compiled_adaptation_step
        ),
    )

    # Ensure training is complete before diagnostics.
    jax.block_until_ready(
        model
    )

    epsilon = (
        config.evaluation.spd_epsilon
    )

    print()
    print(
        f"benchmark SPD epsilon: "
        f"{epsilon:.8e}"
    )

    print()

    train_diagnostics = per_sample_nll(
        model,
        data.x[train_idx],
        data.r[train_idx],
        data.h[train_idx],
        spd_epsilon=epsilon,
    )

    validation_diagnostics = (
        per_sample_nll(
            model,
            data.x[
                validation_idx
            ],
            data.r[
                validation_idx
            ],
            data.h[
                validation_idx
            ],
            spd_epsilon=epsilon,
        )
    )

    test_diagnostics = per_sample_nll(
        model,
        data.x[test_idx],
        data.r[test_idx],
        data.h[test_idx],
        spd_epsilon=epsilon,
    )

    summarize_split(
        "train",
        train_diagnostics,
    )

    summarize_split(
        "validation",
        validation_diagnostics,
    )

    summarize_split(
        "test",
        test_diagnostics,
    )

    print_worst_samples(
        test_diagnostics,
        n_worst=10,
    )

    trimmed_mean_report(
        test_diagnostics[
            "loss"
        ]
    )

    epsilon_diagnostic(
        model,
        data.x[test_idx],
        data.r[test_idx],
        data.h[test_idx],
    )


if __name__ == "__main__":
    main()