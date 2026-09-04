#!/usr/bin/env python3
"""
Diagnose ARFF resampling dynamics for the Experiment 6 drift regression.

This reproduces the final ex6 drift regression used in the two-stage
pipeline:

    x_train
    y_train = r_train / h_train

with the current ARFF settings

    resampling=True
    metropolis_test=False.

The production implementation is not modified.

At every iteration this diagnostic reports:

    - effective sample size of the amplitude-based resampling PMF,
    - maximum resampling probability,
    - entropy-based effective population size,
    - number of unique ancestors selected by multinomial resampling,
    - maximum ancestor multiplicity,
    - frequency norm statistics before and after mutation,
    - amplitude concentration,
    - training prediction RMSE,
    - validation prediction RMSE.

The purpose is to determine whether the bad ex6 seeds are associated
with particle impoverishment / resampling collapse.
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
    ARFFModel,
    fit_amplitudes,
    fourier_features,
    frequency_amplitude_norm,
    predict,
)
from src.experiments.config import get_config
from src.experiments.dataset import load_dataset


EXPERIMENT = "ex6"

SEEDS = (
    0,
    1,
    2,
)

N_ITERATIONS = 100


def rmse(
    model,
    x,
    y,
):
    prediction = predict(
        model,
        x,
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


def pmf_diagnostics(
    amp,
    K,
):
    """
    Compute diagnostics for the amplitude-based resampling distribution.
    """
    amp_norm = frequency_amplitude_norm(
        amp,
        K,
    )

    tiny = jnp.finfo(
        amp_norm.dtype
    ).tiny

    weights = jnp.maximum(
        amp_norm,
        tiny,
    )

    pmf = (
        weights
        / jnp.sum(
            weights
        )
    )

    ess = (
        1.0
        / jnp.sum(
            pmf**2
        )
    )

    entropy = (
        -jnp.sum(
            pmf
            * jnp.log(
                jnp.maximum(
                    pmf,
                    tiny,
                )
            )
        )
    )

    entropy_ess = jnp.exp(
        entropy
    )

    max_probability = jnp.max(
        pmf
    )

    sorted_pmf = jnp.sort(
        pmf
    )[::-1]

    top1_mass = sorted_pmf[0]
    top5_mass = jnp.sum(
        sorted_pmf[
            : min(
                5,
                K,
            )
        ]
    )
    top10_mass = jnp.sum(
        sorted_pmf[
            : min(
                10,
                K,
            )
        ]
    )

    return {
        "pmf": pmf,
        "ess": float(
            ess
        ),
        "entropy_ess": float(
            entropy_ess
        ),
        "max_probability": float(
            max_probability
        ),
        "top1_mass": float(
            top1_mass
        ),
        "top5_mass": float(
            top5_mass
        ),
        "top10_mass": float(
            top10_mass
        ),
        "amp_norm_min": float(
            jnp.min(
                amp_norm
            )
        ),
        "amp_norm_median": float(
            jnp.median(
                amp_norm
            )
        ),
        "amp_norm_max": float(
            jnp.max(
                amp_norm
            )
        ),
    }


def frequency_diagnostics(
    omega,
):
    """
    Euclidean norm statistics for the Fourier frequencies.
    """
    norms = jnp.linalg.norm(
        omega,
        axis=0,
    )

    return {
        "omega_min": float(
            jnp.min(
                norms
            )
        ),
        "omega_median": float(
            jnp.median(
                norms
            )
        ),
        "omega_mean": float(
            jnp.mean(
                norms
            )
        ),
        "omega_max": float(
            jnp.max(
                norms
            )
        ),
    }


def resampling_diagnostics(
    selected,
    K,
):
    """
    Diagnostic statistics of one multinomial-resampling draw.
    """
    selected_np = np.asarray(
        selected
    )

    counts = np.bincount(
        selected_np,
        minlength=K,
    )

    unique = int(
        np.count_nonzero(
            counts
        )
    )

    maximum_multiplicity = int(
        counts.max()
    )

    return {
        "unique_ancestors": unique,
        "unique_fraction": (
            unique
            / K
        ),
        "max_multiplicity": (
            maximum_multiplicity
        ),
    }


def one_iteration(
    key,
    model,
    x_train,
    y_train,
    *,
    delta,
    lambda_reg,
):
    """
    Perform exactly the current resampling-only ARFF iteration:

        random walk
        -> amplitude fit
        -> amplitude-weighted resampling
        -> amplitude refit

    Diagnostics from the intermediate proposal and resampling step are
    returned as ordinary Python values.
    """
    K = model.omega.shape[1]

    omega_old = model.omega

    # ------------------------------------------------------------
    # Mutation.
    # ------------------------------------------------------------
    key, mutation_key = jax.random.split(
        key
    )

    proposal = (
        omega_old
        + delta
        * jax.random.normal(
            mutation_key,
            omega_old.shape,
        )
    )

    proposal_amp = fit_amplitudes(
        x_train,
        y_train,
        proposal,
        lambda_reg,
    )

    proposal_weight_stats = (
        pmf_diagnostics(
            proposal_amp,
            K,
        )
    )

    proposal_frequency_stats = (
        frequency_diagnostics(
            proposal
        )
    )

    pmf = proposal_weight_stats[
        "pmf"
    ]

    # ------------------------------------------------------------
    # Resampling.
    # ------------------------------------------------------------
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

    selection_stats = (
        resampling_diagnostics(
            selected,
            K,
        )
    )

    omega = proposal[
        :,
        selected,
    ]

    amp = fit_amplitudes(
        x_train,
        y_train,
        omega,
        lambda_reg,
    )

    model = ARFFModel(
        omega=omega,
        amp=amp,
    )

    selected_frequency_stats = (
        frequency_diagnostics(
            omega
        )
    )

    return (
        key,
        model,
        proposal_weight_stats,
        proposal_frequency_stats,
        selection_stats,
        selected_frequency_stats,
    )


def print_header():
    print(
        "iter  "
        "ESS      "
        "eESS     "
        "pmax      "
        "top5      "
        "uniq   "
        "maxrep  "
        "|w|med   "
        "|w|max   "
        "trainRMSE   "
        "valRMSE"
    )


def print_iteration(
    iteration,
    weight_stats,
    selection_stats,
    frequency_stats,
    train_error,
    validation_error,
):
    print(
        f"{iteration:4d}  "
        f"{weight_stats['ess']:7.1f}  "
        f"{weight_stats['entropy_ess']:7.1f}  "
        f"{weight_stats['max_probability']:.5f}  "
        f"{weight_stats['top5_mass']:.5f}  "
        f"{selection_stats['unique_ancestors']:4d}  "
        f"{selection_stats['max_multiplicity']:6d}  "
        f"{frequency_stats['omega_median']:8.3f}  "
        f"{frequency_stats['omega_max']:8.3f}  "
        f"{train_error:11.4e}  "
        f"{validation_error:11.4e}"
    )


def run_seed(
    seed,
    x_train,
    y_train,
    x_validation,
    y_validation,
    *,
    K,
    delta,
    lambda_reg,
):
    print()
    print(
        "=" * 120
    )
    print(
        f"seed {seed}"
    )
    print(
        "=" * 120
    )

    key = jax.random.PRNGKey(
        seed
    )

    # Same zero-frequency initialization as fit_arff.
    omega = jnp.zeros(
        (
            x_train.shape[1],
            K,
        ),
        dtype=x_train.dtype,
    )

    amp = fit_amplitudes(
        x_train,
        y_train,
        omega,
        lambda_reg,
    )

    model = ARFFModel(
        omega=omega,
        amp=amp,
    )

    initial_train_rmse = rmse(
        model,
        x_train,
        y_train,
    )

    initial_validation_rmse = rmse(
        model,
        x_validation,
        y_validation,
    )

    print(
        f"initial train RMSE      : "
        f"{initial_train_rmse:.8e}"
    )

    print(
        f"initial validation RMSE : "
        f"{initial_validation_rmse:.8e}"
    )

    print()

    print_header()

    records = []

    for iteration in range(
        1,
        N_ITERATIONS + 1,
    ):
        (
            key,
            model,
            weight_stats,
            proposal_frequency_stats,
            selection_stats,
            selected_frequency_stats,
        ) = one_iteration(
            key,
            model,
            x_train,
            y_train,
            delta=delta,
            lambda_reg=lambda_reg,
        )

        # Synchronize before diagnostics so failures are associated with
        # the correct iteration.
        jax.block_until_ready(
            model
        )

        train_error = rmse(
            model,
            x_train,
            y_train,
        )

        validation_error = rmse(
            model,
            x_validation,
            y_validation,
        )

        record = {
            "iteration": iteration,
            "ess": weight_stats[
                "ess"
            ],
            "entropy_ess": weight_stats[
                "entropy_ess"
            ],
            "max_probability": (
                weight_stats[
                    "max_probability"
                ]
            ),
            "top5_mass": (
                weight_stats[
                    "top5_mass"
                ]
            ),
            "top10_mass": (
                weight_stats[
                    "top10_mass"
                ]
            ),
            "unique_ancestors": (
                selection_stats[
                    "unique_ancestors"
                ]
            ),
            "unique_fraction": (
                selection_stats[
                    "unique_fraction"
                ]
            ),
            "max_multiplicity": (
                selection_stats[
                    "max_multiplicity"
                ]
            ),
            "proposal_omega_median": (
                proposal_frequency_stats[
                    "omega_median"
                ]
            ),
            "proposal_omega_max": (
                proposal_frequency_stats[
                    "omega_max"
                ]
            ),
            "selected_omega_median": (
                selected_frequency_stats[
                    "omega_median"
                ]
            ),
            "selected_omega_max": (
                selected_frequency_stats[
                    "omega_max"
                ]
            ),
            "train_rmse": (
                train_error
            ),
            "validation_rmse": (
                validation_error
            ),
        }

        records.append(
            record
        )

        # Print early iterations densely, then every 5 iterations, plus
        # the iteration budgets we have been studying.
        if (
            iteration <= 10
            or iteration % 5 == 0
            or iteration
            in {
                25,
                50,
                100,
            }
        ):
            print_iteration(
                iteration,
                weight_stats,
                selection_stats,
                selected_frequency_stats,
                train_error,
                validation_error,
            )

    print()
    print(
        "Worst / most concentrated iterations"
    )
    print(
        "------------------------------------"
    )

    lowest_ess = sorted(
        records,
        key=lambda item: item[
            "ess"
        ],
    )[:10]

    for record in lowest_ess:
        print(
            f"iter={record['iteration']:3d}  "
            f"ESS={record['ess']:8.2f}  "
            f"eESS={record['entropy_ess']:8.2f}  "
            f"pmax={record['max_probability']:.6f}  "
            f"top5={record['top5_mass']:.6f}  "
            f"unique={record['unique_ancestors']:3d}  "
            f"maxrep={record['max_multiplicity']:3d}  "
            f"valRMSE={record['validation_rmse']:.6e}"
        )

    print()
    print(
        "Worst validation-RMSE iterations"
    )
    print(
        "--------------------------------"
    )

    worst_validation = sorted(
        records,
        key=lambda item: item[
            "validation_rmse"
        ],
        reverse=True,
    )[:10]

    for record in worst_validation:
        print(
            f"iter={record['iteration']:3d}  "
            f"valRMSE={record['validation_rmse']:.6e}  "
            f"ESS={record['ess']:8.2f}  "
            f"pmax={record['max_probability']:.6f}  "
            f"unique={record['unique_ancestors']:3d}  "
            f"|w|max={record['selected_omega_max']:.3f}"
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

    # Exact drift targets used by two_stage.py.
    y_train = (
        r_train
        / h_train
    )

    y_validation = (
        r_validation
        / h_validation
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

    print(
        "Experiment 6 ARFF resampling diagnostic"
    )
    print(
        "========================================"
    )

    print(
        f"train N       : "
        f"{len(train_idx)}"
    )

    print(
        f"validation N  : "
        f"{len(validation_idx)}"
    )

    print(
        f"input dim     : "
        f"{x_train.shape[1]}"
    )

    print(
        f"output dim    : "
        f"{y_train.shape[1]}"
    )

    print(
        f"K             : "
        f"{K}"
    )

    print(
        f"delta         : "
        f"{delta:.8e}"
    )

    print(
        f"lambda        : "
        f"{lambda_reg:.8e}"
    )

    print(
        f"iterations    : "
        f"{N_ITERATIONS}"
    )

    print(
        f"seeds         : "
        f"{SEEDS}"
    )

    for seed in SEEDS:
        run_seed(
            seed,
            x_train,
            y_train,
            x_validation,
            y_validation,
            K=K,
            delta=delta,
            lambda_reg=lambda_reg,
        )


if __name__ == "__main__":
    main()