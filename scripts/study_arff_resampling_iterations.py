#!/usr/bin/env python3
"""
Validation-only multi-seed study of the ARFF resampling iteration budget.

The ARFF adaptation regime is fixed to

    resampling=True
    metropolis_test=False

so that fitted Fourier amplitudes drive resampling of the frequencies,
followed by Gaussian random-walk mutation.

Only the number of ARFF adaptation iterations M is varied.

Hyperparameter selection rule
-----------------------------
For each experiment and each candidate M, evaluate the canonical
validation NLL over a fixed set of ARFF tuning seeds

    seeds = {0, 1, 2}.

Select

    M* = argmin_M mean_seed validation_NLL(M, seed).

The canonical test split is never used.

Experiments:
    ex1, ex2, ex4, ex5
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
    predict,
)
from src.arff.two_stage import fit_two_stage_arff
from src.experiments.config import get_config
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment


EXPERIMENTS = (
    "ex1",
    "ex2",
    "ex4",
    "ex5",
)

ITERATIONS = (
    25,
    50,
    100,
    200,
    300,
)

TUNING_SEEDS = (
    0,
    1,
    2,
)


def drift_rmse(
    model,
    x,
    *,
    true_drift,
):
    """
    Validation drift RMSE.

    Diagnostic only. M is selected using validation NLL.
    """
    x = jnp.asarray(x)

    learned = predict(
        model.drift,
        x,
    )

    truth = jnp.asarray(
        true_drift(x)
    )

    return float(
        jnp.sqrt(
            jnp.mean(
                (learned - truth) ** 2
            )
        )
    )


def covariance_rmse(
    model,
    x,
    *,
    true_diffusion_factor,
):
    """
    Validation covariance RMSE before SPD projection.

    Diagnostic only. M is selected using validation NLL.
    """
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


def fit_one(
    *,
    experiment_name: str,
    n_iterations: int,
    seed: int,
    config,
    definition,
    data,
):
    """
    Fit one ARFF candidate and evaluate it on the validation split only.
    """
    train_idx = data.train_idx
    validation_idx = data.validation_idx

    arff_config = replace(
        config.arff,
        M_min=n_iterations,
        M_max=n_iterations,
        resampling=True,
        metropolis_test=False,
    )

    x_train = jnp.asarray(
        data.x[train_idx]
    )

    r_train = jnp.asarray(
        data.r[train_idx]
    )

    h_train = jnp.asarray(
        data.h[train_idx]
    )

    x_validation = data.x[
        validation_idx
    ]

    r_validation = data.r[
        validation_idx
    ]

    h_validation = data.h[
        validation_idx
    ]

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

    jax.block_until_ready(
        model
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    evaluation = gaussian_nll(
        model,
        x_validation,
        r_validation,
        h_validation,
        spd_epsilon=(
            config.evaluation.spd_epsilon
        ),
    )

    covariance = raw_covariance(
        model.covariance,
        jnp.asarray(
            x_validation
        ),
        definition.diff_type,
    )

    violation_rate = float(
        np.mean(
            np.asarray(
                spd_violation_mask(
                    covariance
                )
            )
        )
    )

    drift_error = drift_rmse(
        model,
        x_validation,
        true_drift=(
            definition.drift
        ),
    )

    covariance_error = (
        covariance_rmse(
            model,
            x_validation,
            true_diffusion_factor=(
                definition.diffusion_factor
            ),
        )
    )

    return {
        "experiment": experiment_name,
        "iterations": n_iterations,
        "seed": seed,
        "nll": evaluation.nll,
        "spd": violation_rate,
        "drift_rmse": drift_error,
        "covariance_rmse": covariance_error,
        "elapsed": elapsed,
    }


def summarize_candidate(
    results,
):
    """
    Aggregate one experiment/M candidate over tuning seeds.
    """
    nll = np.array(
        [
            result["nll"]
            for result in results
        ],
        dtype=float,
    )

    spd = np.array(
        [
            result["spd"]
            for result in results
        ],
        dtype=float,
    )

    drift = np.array(
        [
            result["drift_rmse"]
            for result in results
        ],
        dtype=float,
    )

    covariance = np.array(
        [
            result["covariance_rmse"]
            for result in results
        ],
        dtype=float,
    )

    elapsed = np.array(
        [
            result["elapsed"]
            for result in results
        ],
        dtype=float,
    )

    return {
        "experiment": results[0]["experiment"],
        "iterations": results[0]["iterations"],
        "mean_nll": float(
            np.mean(nll)
        ),
        "std_nll": float(
            np.std(
                nll,
                ddof=1,
            )
        ),
        "mean_spd": float(
            np.mean(spd)
        ),
        "std_spd": float(
            np.std(
                spd,
                ddof=1,
            )
        ),
        "mean_drift_rmse": float(
            np.mean(drift)
        ),
        "std_drift_rmse": float(
            np.std(
                drift,
                ddof=1,
            )
        ),
        "mean_covariance_rmse": float(
            np.mean(covariance)
        ),
        "std_covariance_rmse": float(
            np.std(
                covariance,
                ddof=1,
            )
        ),
        "mean_elapsed": float(
            np.mean(elapsed)
        ),
    }


def print_seed_result(
    result,
):
    print(
        f"    seed={result['seed']}  "
        f"NLL={result['nll']: .8e}  "
        f"SPD={result['spd']:.6f}  "
        f"drift={result['drift_rmse']:.8e}  "
        f"cov={result['covariance_rmse']:.8e}  "
        f"time={result['elapsed']:.2f}s"
    )


def print_summary(
    summary,
):
    print(
        f"M={summary['iterations']:3d}  "
        f"NLL={summary['mean_nll']: .8e} "
        f"+/- {summary['std_nll']:.3e}  "
        f"SPD={summary['mean_spd']:.6f} "
        f"+/- {summary['std_spd']:.3e}  "
        f"drift={summary['mean_drift_rmse']:.8e} "
        f"+/- {summary['std_drift_rmse']:.3e}  "
        f"cov={summary['mean_covariance_rmse']:.8e} "
        f"+/- {summary['std_covariance_rmse']:.3e}"
    )


def main():
    print(
        "ARFF resampling iteration study"
    )
    print(
        "================================"
    )

    print(
        "resampling   : True"
    )

    print(
        "Metropolis   : False"
    )

    print(
        f"tuning seeds : {TUNING_SEEDS}"
    )

    print(
        f"M candidates : {ITERATIONS}"
    )

    print()

    all_summaries = []

    for experiment_name in EXPERIMENTS:
        config = get_config(
            experiment_name
        )

        definition = get_experiment(
            experiment_name
        )

        data = load_dataset(
            REPO_ROOT
            / "data"
            / f"{experiment_name}.npz"
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
            f"K      : "
            f"{config.fourier_frequencies}"
        )

        print(
            f"delta  : "
            f"{config.arff.delta:.8e}"
        )

        print(
            f"gamma  : "
            f"{config.arff.gamma:.8e}"
        )

        print(
            f"lambda : "
            f"{config.arff.lambda_reg:.8e}"
        )

        print()

        experiment_summaries = []

        for n_iterations in ITERATIONS:
            print(
                f"M={n_iterations}"
            )

            seed_results = []

            for seed in TUNING_SEEDS:
                result = fit_one(
                    experiment_name=(
                        experiment_name
                    ),
                    n_iterations=(
                        n_iterations
                    ),
                    seed=seed,
                    config=config,
                    definition=definition,
                    data=data,
                )

                seed_results.append(
                    result
                )

                print_seed_result(
                    result
                )

            summary = (
                summarize_candidate(
                    seed_results
                )
            )

            experiment_summaries.append(
                summary
            )

            all_summaries.append(
                summary
            )

            print(
                "  aggregate:"
            )

            print(
                "  ",
                end="",
            )

            print_summary(
                summary
            )

            print()

        # Formal selection criterion:
        # lowest mean validation NLL across tuning seeds.
        best = min(
            experiment_summaries,
            key=lambda item: item[
                "mean_nll"
            ],
        )

        print(
            "selected by mean validation NLL:"
        )

        print_summary(
            best
        )

        print()
        print()

    print(
        "=" * 110
    )

    print(
        "Final tuning summary"
    )

    print(
        "=" * 110
    )

    for experiment_name in EXPERIMENTS:
        subset = [
            summary
            for summary in all_summaries
            if summary[
                "experiment"
            ]
            == experiment_name
        ]

        best = min(
            subset,
            key=lambda item: item[
                "mean_nll"
            ],
        )

        print(
            f"{experiment_name}: "
            f"M={best['iterations']}  "
            f"mean validation NLL="
            f"{best['mean_nll']:.8e} "
            f"+/- {best['std_nll']:.3e}  "
            f"mean SPD="
            f"{best['mean_spd']:.6f}"
        )


if __name__ == "__main__":
    main()