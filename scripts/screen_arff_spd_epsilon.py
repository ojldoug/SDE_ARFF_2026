#!/usr/bin/env python3
"""
Validation-only screening of the SPD eigenvalue floor for ARFF.

For a fixed experiment, ARFF configuration, and seed, the model is fitted
exactly once. The same fitted model is then evaluated using several
projection floors

    Sigma_eps = Q diag(max(lambda_i, eps)) Q^T.

The test split is never evaluated.

Typical use
-----------

    CUDA_VISIBLE_DEVICES=0 \
    python scripts/screen_arff_spd_epsilon.py \
      --experiment ex8 \
      --seed 0 \
      --iterations 25 \
      --epsilons 1e-3 3e-3 1e-2 3e-2 1e-1 \
      --output results/ex8_spd_epsilon_seed0.csv

Selection criterion
-------------------

For a multi-seed screen, select epsilon using mean validation Gaussian NLL
across the prespecified tuning seeds.

Notes
-----

This script does not modify or retrain the model for each epsilon.
The expensive ARFF fit is performed exactly once per invocation.

The reported raw SPD violation rate is independent of epsilon.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.arff.covariance import raw_covariance
from src.arff.regression import predict
from src.arff.two_stage import fit_two_stage_arff
from src.experiments.config import get_config
from src.experiments.dataset import (
    load_dataset,
    validate_split_indices,
)
from src.experiments.definitions import get_experiment


ALL_EXPERIMENTS = tuple(
    f"ex{i}"
    for i in range(1, 9)
)

DEFAULT_EPSILONS = (
    1e-3,
    3e-3,
    1e-2,
    3e-2,
    1e-1,
)


def project_spd(
    covariance,
    epsilon,
):
    """
    Project symmetric covariance matrices onto

        Sigma >= epsilon I

    by eigenvalue flooring.
    """
    eigenvalues, eigenvectors = (
        jnp.linalg.eigh(
            covariance
        )
    )

    projected_eigenvalues = jnp.maximum(
        eigenvalues,
        epsilon,
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


def validation_statistics(
    model,
    x,
    r,
    h,
    *,
    epsilon,
):
    """
    Evaluate validation Gaussian NLL after SPD projection.
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

    if not bool(
        jnp.all(
            sign > 0.0
        )
    ):
        raise RuntimeError(
            "Projected covariance produced "
            "a non-positive determinant."
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

    n_violating = int(
        jnp.sum(
            violation
        )
    )

    n_valid = (
        len(violation)
        - n_violating
    )

    if n_valid > 0:
        mean_nll_raw_spd = float(
            jnp.mean(
                losses[
                    ~violation
                ]
            )
        )
    else:
        mean_nll_raw_spd = (
            float("nan")
        )

    if n_violating > 0:
        mean_nll_raw_violating = float(
            jnp.mean(
                losses[
                    violation
                ]
            )
        )
    else:
        mean_nll_raw_violating = (
            float("nan")
        )

    return {
        "validation_nll": float(
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
        "mean_quadratic": float(
            jnp.mean(
                quadratic
            )
        ),
        "mean_logdet": float(
            jnp.mean(
                logdet
            )
        ),
        "spd_violation_rate": float(
            jnp.mean(
                violation
            )
        ),
        "mean_nll_raw_spd": (
            mean_nll_raw_spd
        ),
        "mean_nll_raw_violating": (
            mean_nll_raw_violating
        ),
        "min_raw_eigenvalue": float(
            jnp.min(
                raw_min_eigenvalue
            )
        ),
        "min_projected_eigenvalue": float(
            jnp.min(
                projected_min_eigenvalue
            )
        ),
    }


def run_seed(
    *,
    experiment_name,
    seed,
    n_iterations,
    epsilons,
):
    config = get_config(
        experiment_name
    )

    definition = get_experiment(
        experiment_name
    )

    if (
        definition.diff_type
        == "diagonal"
    ):
        raise ValueError(
            f"{experiment_name} uses diagonal "
            "covariance. This eigenvalue-floor "
            "screen is intended for full/symmetric "
            "covariance experiments."
        )

    if (
        config.fourier_frequencies
        is None
    ):
        raise ValueError(
            "No Fourier frequency count "
            f"has been established for "
            f"{experiment_name}."
        )

    data = load_dataset(
        REPO_ROOT
        / "data"
        / f"{experiment_name}.npz"
    )

    validate_split_indices(
        len(data.x),
        data.train_idx,
        data.validation_idx,
        data.test_idx,
    )

    arff_config = replace(
        config.arff,
        M_min=n_iterations,
        M_max=n_iterations,
        resampling=True,
        metropolis_test=False,
    )

    train_idx = (
        data.train_idx
    )

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
        seed
    )

    print(
        "ARFF SPD-epsilon validation screen"
    )
    print(
        "=================================="
    )

    print(
        f"experiment   : "
        f"{experiment_name}"
    )

    print(
        f"backend      : "
        f"{jax.default_backend()}"
    )

    print(
        f"seed         : "
        f"{seed}"
    )

    print(
        f"K            : "
        f"{config.fourier_frequencies}"
    )

    print(
        f"M            : "
        f"{n_iterations}"
    )

    print(
        f"diffusion    : "
        f"{definition.diff_type}"
    )

    print(
        f"train N      : "
        f"{len(train_idx)}"
    )

    print(
        f"validation N : "
        f"{len(validation_idx)}"
    )

    print(
        f"epsilons     : "
        f"{epsilons}"
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

    print(
        "Fitting ARFF model once..."
    )

    start = (
        time.perf_counter()
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

    fit_time = (
        time.perf_counter()
        - start
    )

    print(
        f"fit complete in "
        f"{fit_time:.2f} s"
    )

    print()

    results = []

    for epsilon in epsilons:
        statistics = (
            validation_statistics(
                model,
                x_validation,
                r_validation,
                h_validation,
                epsilon=epsilon,
            )
        )

        result = {
            "experiment": (
                experiment_name
            ),
            "seed": (
                seed
            ),
            "K": (
                config.fourier_frequencies
            ),
            "M": (
                n_iterations
            ),
            "epsilon": (
                epsilon
            ),
            "fit_seconds": (
                fit_time
            ),
            **statistics,
        }

        results.append(
            result
        )

        print(
            f"epsilon={epsilon:.1e}"
        )

        print(
            "  validation NLL          : "
            f"{result['validation_nll']:.8e}"
        )

        print(
            "  median NLL              : "
            f"{result['median_nll']:.8e}"
        )

        print(
            "  95% NLL quantile        : "
            f"{result['q95_nll']:.8e}"
        )

        print(
            "  99% NLL quantile        : "
            f"{result['q99_nll']:.8e}"
        )

        print(
            "  max NLL                 : "
            f"{result['max_nll']:.8e}"
        )

        print(
            "  mean quadratic          : "
            f"{result['mean_quadratic']:.8e}"
        )

        print(
            "  mean logdet             : "
            f"{result['mean_logdet']:.8e}"
        )

        print(
            "  raw SPD violation       : "
            f"{result['spd_violation_rate']:.8f}"
        )

        print(
            "  NLL on raw-SPD points   : "
            f"{result['mean_nll_raw_spd']:.8e}"
        )

        print(
            "  NLL on violating points : "
            f"{result['mean_nll_raw_violating']:.8e}"
        )

        print(
            "  min raw eig             : "
            f"{result['min_raw_eigenvalue']:.8e}"
        )

        print(
            "  min projected eig       : "
            f"{result['min_projected_eigenvalue']:.8e}"
        )

        print()

    best = min(
        results,
        key=lambda item: item[
            "validation_nll"
        ],
    )

    print(
        "best epsilon for this seed:"
    )

    print(
        f"  epsilon = "
        f"{best['epsilon']:.1e}"
    )

    print(
        f"  NLL     = "
        f"{best['validation_nll']:.8e}"
    )

    print()

    return results


def write_csv(
    path,
    results,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "experiment",
        "seed",
        "K",
        "M",
        "epsilon",
        "fit_seconds",
        "validation_nll",
        "median_nll",
        "q95_nll",
        "q99_nll",
        "max_nll",
        "mean_quadratic",
        "mean_logdet",
        "spd_violation_rate",
        "mean_nll_raw_spd",
        "mean_nll_raw_violating",
        "min_raw_eigenvalue",
        "min_projected_eigenvalue",
    ]

    with path.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            results
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Screen ARFF SPD projection "
            "epsilon on validation data."
        )
    )

    parser.add_argument(
        "--experiment",
        choices=ALL_EXPERIMENTS,
        required=True,
    )

    parser.add_argument(
        "--seed",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        required=True,
        help=(
            "Fixed ARFF adaptation iteration "
            "budget M."
        ),
    )

    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=list(
            DEFAULT_EPSILONS
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if (
        args.iterations
        <= 0
    ):
        raise ValueError(
            "--iterations must be positive."
        )

    epsilons = tuple(
        sorted(
            set(
                args.epsilons
            )
        )
    )

    if not epsilons:
        raise ValueError(
            "At least one epsilon must "
            "be supplied."
        )

    if any(
        value <= 0.0
        for value in epsilons
    ):
        raise ValueError(
            "All epsilon values must "
            "be positive."
        )

    results = run_seed(
        experiment_name=(
            args.experiment
        ),
        seed=args.seed,
        n_iterations=(
            args.iterations
        ),
        epsilons=epsilons,
    )

    if (
        args.output
        is not None
    ):
        write_csv(
            args.output,
            results,
        )

        print(
            f"results written to "
            f"{args.output}"
        )


if __name__ == "__main__":
    main()