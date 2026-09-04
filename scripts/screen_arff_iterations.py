#!/usr/bin/env python3
"""
Validation-only screening of ARFF adaptation-iteration budgets.

The production ARFF implementation is used without modification.

For every requested experiment, seed, and iteration budget M, this script

    1. fits the two-stage ARFF estimator using the canonical training split,
    2. evaluates it only on the canonical validation split,
    3. reports validation Gaussian NLL,
    4. reports diagnostic drift/covariance RMSE and raw SPD violations.

The canonical test split is never evaluated.

Typical use
-----------
Screen one experiment:

    python scripts/screen_arff_iterations.py \
        --experiment ex6 \
        --seed 0 \
        --iterations 25 50 100 200 300

Screen all experiments:

    python scripts/screen_arff_iterations.py \
        --experiment all \
        --seed 0 \
        --iterations 25 50 100 200 300

Write machine-readable results:

    python scripts/screen_arff_iterations.py \
        --experiment ex6 \
        --seed 0 \
        --iterations 25 50 100 200 300 \
        --output results/arff_screen_ex6_seed0.csv

Notes
-----
This is a hyperparameter-screening script, not a timing benchmark.

The wall-clock times printed here include ordinary execution effects and
possibly compilation/cache effects. Final publication timing must be done
separately with the dedicated timing protocol.

ARFF is forced here to the current paper candidate:

    resampling=True
    metropolis_test=False

The number of adaptation iterations M is the screened quantity.
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
    predict,
)
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

DEFAULT_ITERATIONS = (
    25,
    50,
    100,
    200,
    300,
)


def drift_rmse(
    model,
    x,
    *,
    true_drift,
) -> float:
    """
    RMSE of the learned drift against the known drift.

    Diagnostic only. Iteration-budget selection is based on validation
    Gaussian NLL.
    """
    x = jnp.asarray(x)

    learned = predict(
        model.drift,
        x,
    )

    truth = jnp.asarray(
        true_drift(x)
    )

    value = jnp.sqrt(
        jnp.mean(
            (
                learned
                - truth
            )
            ** 2
        )
    )

    return float(
        value
    )


def covariance_rmse(
    model,
    x,
    *,
    true_diffusion_factor,
) -> float:
    """
    RMSE of the raw learned covariance against the known covariance.

    The learned covariance is evaluated before SPD projection, matching
    the diagnostic convention used elsewhere in the repository.

    Diagnostic only. Iteration-budget selection is based on validation
    Gaussian NLL.
    """
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

    value = jnp.sqrt(
        jnp.mean(
            (
                learned
                - covariance_true
            )
            ** 2
        )
    )

    return float(
        value
    )


def evaluate_validation(
    model,
    *,
    x,
    r,
    h,
    definition,
    spd_epsilon,
):
    """
    Evaluate one fitted model on the validation split only.
    """
    result = gaussian_nll(
        model,
        x,
        r,
        h,
        spd_epsilon=spd_epsilon,
    )

    covariance_raw = raw_covariance(
        model.covariance,
        jnp.asarray(x),
        model.diff_type,
    )

    violations = spd_violation_mask(
        covariance_raw
    )

    drift_error = drift_rmse(
        model,
        x,
        true_drift=definition.drift,
    )

    covariance_error = covariance_rmse(
        model,
        x,
        true_diffusion_factor=(
            definition.diffusion_factor
        ),
    )

    return {
        "validation_nll": (
            result.nll
        ),
        "spd_violation_rate": float(
            jnp.mean(
                violations
            )
        ),
        "min_raw_eigenvalue": (
            result.min_raw_eigenvalue
        ),
        "min_projected_eigenvalue": (
            result.min_projected_eigenvalue
        ),
        "drift_rmse": (
            drift_error
        ),
        "covariance_rmse": (
            covariance_error
        ),
    }


def run_candidate(
    *,
    experiment_name: str,
    seed: int,
    n_iterations: int,
    config,
    definition,
    data,
):
    """
    Fit and evaluate one (experiment, seed, M) candidate.
    """
    arff_config = replace(
        config.arff,
        M_min=n_iterations,
        M_max=n_iterations,
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

    # Construct the production ARFF kernel using exactly the
    # hyperparameters for this candidate.
    compiled_step = (
        make_compiled_adaptation_step(
            delta=arff_config.delta,
            lambda_reg=(
                arff_config.lambda_reg
            ),
            gamma=arff_config.gamma,
            resampling=True,
            metropolis_test=False,
        )
    )

    key = jax.random.PRNGKey(
        seed
    )

    start = time.perf_counter()

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
            compiled_adaptation_step=(
                compiled_step
            ),
        )
    )

    # Ensure all asynchronous GPU work is complete before stopping
    # the screening wall-clock timer.
    jax.block_until_ready(
        model
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    evaluation = evaluate_validation(
        model,
        x=x_validation,
        r=r_validation,
        h=h_validation,
        definition=definition,
        spd_epsilon=(
            config.evaluation.spd_epsilon
        ),
    )

    return {
        "experiment": (
            experiment_name
        ),
        "seed": (
            seed
        ),
        "iterations": (
            n_iterations
        ),
        "frequencies": (
            config.fourier_frequencies
        ),
        "train_samples": (
            len(train_idx)
        ),
        "validation_samples": (
            len(validation_idx)
        ),
        "resampling": True,
        "metropolis_test": False,
        "lambda_reg": (
            arff_config.lambda_reg
        ),
        "gamma": (
            arff_config.gamma
        ),
        "delta": (
            arff_config.delta
        ),
        "elapsed_seconds": (
            elapsed
        ),
        **evaluation,
    }


def print_result(
    result,
):
    print(
        f"M={result['iterations']:4d}  "
        f"NLL={result['validation_nll']: .8e}  "
        f"SPD={result['spd_violation_rate']:.6f}  "
        f"drift={result['drift_rmse']:.8e}  "
        f"cov={result['covariance_rmse']:.8e}  "
        f"time={result['elapsed_seconds']:.2f}s"
    )


def write_csv(
    path: Path,
    results,
):
    """
    Write all completed screening results to CSV.
    """
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "experiment",
        "seed",
        "iterations",
        "frequencies",
        "train_samples",
        "validation_samples",
        "resampling",
        "metropolis_test",
        "lambda_reg",
        "gamma",
        "delta",
        "elapsed_seconds",
        "validation_nll",
        "spd_violation_rate",
        "min_raw_eigenvalue",
        "min_projected_eigenvalue",
        "drift_rmse",
        "covariance_rmse",
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


def resolve_experiments(
    value: str,
):
    if value == "all":
        return ALL_EXPERIMENTS

    return (
        value,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Screen ARFF iteration budgets "
            "using canonical validation data."
        )
    )

    parser.add_argument(
        "--experiment",
        choices=[
            *ALL_EXPERIMENTS,
            "all",
        ],
        required=True,
        help=(
            "Experiment to screen, or 'all' "
            "for ex1--ex8."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "ARFF random seed. "
            "Default: 0."
        ),
    )

    parser.add_argument(
        "--iterations",
        type=int,
        nargs="+",
        default=list(
            DEFAULT_ITERATIONS
        ),
        help=(
            "Candidate ARFF iteration budgets. "
            "Default: 25 50 100 200 300."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional CSV output path."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    iterations = tuple(
        sorted(
            set(
                args.iterations
            )
        )
    )

    if not iterations:
        raise ValueError(
            "At least one iteration budget "
            "must be supplied."
        )

    if any(
        value <= 0
        for value in iterations
    ):
        raise ValueError(
            "All iteration budgets must be positive."
        )

    experiments = (
        resolve_experiments(
            args.experiment
        )
    )

    print(
        "ARFF iteration-budget validation screen"
    )
    print(
        "======================================="
    )
    print(
        f"backend      : "
        f"{jax.default_backend()}"
    )
    print(
        f"devices      : "
        f"{jax.devices()}"
    )
    print(
        f"seed         : "
        f"{args.seed}"
    )
    print(
        f"experiments  : "
        f"{experiments}"
    )
    print(
        f"M candidates : "
        f"{iterations}"
    )
    print(
        "resampling   : True"
    )
    print(
        "Metropolis   : False"
    )
    print(
        "selection    : validation NLL"
    )
    print(
        "test split   : NOT USED"
    )
    print()

    all_results = []

    for experiment_name in experiments:
        config = get_config(
            experiment_name
        )

        definition = get_experiment(
            experiment_name
        )

        dataset_path = (
            REPO_ROOT
            / "data"
            / f"{experiment_name}.npz"
        )

        data = load_dataset(
            dataset_path
        )

        validate_split_indices(
            len(data.x),
            data.train_idx,
            data.validation_idx,
            data.test_idx,
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

        print(
            experiment_name
        )
        print(
            "-" * len(
                experiment_name
            )
        )

        print(
            f"dataset      : "
            f"{dataset_path}"
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
            f"K            : "
            f"{config.fourier_frequencies}"
        )
        print(
            f"lambda       : "
            f"{config.arff.lambda_reg:.8e}"
        )
        print(
            f"gamma        : "
            f"{config.arff.gamma:.8e}"
        )
        print(
            f"delta        : "
            f"{config.arff.delta:.8e}"
        )
        print()

        experiment_results = []

        for n_iterations in iterations:
            print(
                f"running "
                f"{experiment_name}, "
                f"seed={args.seed}, "
                f"M={n_iterations}"
            )

            result = run_candidate(
                experiment_name=(
                    experiment_name
                ),
                seed=args.seed,
                n_iterations=(
                    n_iterations
                ),
                config=config,
                definition=definition,
                data=data,
            )

            experiment_results.append(
                result
            )

            all_results.append(
                result
            )

            print_result(
                result
            )

            # If CSV output was requested, update it after each
            # completed candidate. This preserves completed work if a
            # later long-running candidate is interrupted.
            if (
                args.output
                is not None
            ):
                write_csv(
                    args.output,
                    all_results,
                )

            print()

        best = min(
            experiment_results,
            key=lambda item: item[
                "validation_nll"
            ],
        )

        print(
            "best validation NLL:"
        )
        print_result(
            best
        )
        print()
        print()

    print(
        "=" * 100
    )
    print(
        "Screening summary"
    )
    print(
        "=" * 100
    )

    for experiment_name in experiments:
        subset = [
            result
            for result in all_results
            if result[
                "experiment"
            ]
            == experiment_name
        ]

        best = min(
            subset,
            key=lambda item: item[
                "validation_nll"
            ],
        )

        print(
            f"{experiment_name}: "
            f"M={best['iterations']}  "
            f"NLL={best['validation_nll']:.8e}  "
            f"SPD={best['spd_violation_rate']:.6f}  "
            f"drift={best['drift_rmse']:.8e}  "
            f"cov={best['covariance_rmse']:.8e}"
        )

    if (
        args.output
        is not None
    ):
        print()
        print(
            f"results written to: "
            f"{args.output}"
        )


if __name__ == "__main__":
    main()