#!/usr/bin/env python3
"""
Multi-seed validation tuning of ARFF iteration budgets for ex3, ex6,
and ex7.

ARFF regime is fixed to

    resampling=True
    metropolis_test=False

Selection criterion
-------------------
For each experiment and candidate M,

    M* = argmin_M mean_seed validation_NLL(M, seed),

using fixed tuning seeds {0, 1, 2}.

The canonical test split is never used.

Candidate grids
---------------
ex3: {25, 50, 100}
ex6: {25, 50, 100}
ex7: {25, 50, 100, 200}
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


CANDIDATES = {
    "ex3": (25, 50, 100),
    "ex6": (25, 50, 100),
    "ex7": (25, 50, 100, 200),
}

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


def run_candidate(
    *,
    experiment_name,
    n_iterations,
    seed,
    config,
    definition,
    data,
):
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

    compiled_step = (
        make_compiled_adaptation_step(
            delta=arff_config.delta,
            lambda_reg=arff_config.lambda_reg,
            gamma=arff_config.gamma,
            resampling=True,
            metropolis_test=False,
        )
    )

    key = jax.random.PRNGKey(
        seed
    )

    start = time.perf_counter()

    key, model, _ = fit_two_stage_arff(
        key,
        x_train,
        r_train,
        h_train,
        K=config.fourier_frequencies,
        diff_type=definition.diff_type,
        config=arff_config,
        fold_seed=config.split.seed,
        compiled_adaptation_step=compiled_step,
    )

    jax.block_until_ready(
        model
    )

    elapsed = (
        time.perf_counter()
        - start
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

    spd_rate = float(
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
        true_drift=definition.drift,
    )

    covariance_error = covariance_rmse(
        model,
        x_validation,
        true_diffusion_factor=(
            definition.diffusion_factor
        ),
    )

    return {
        "experiment": experiment_name,
        "M": n_iterations,
        "seed": seed,
        "nll": evaluation.nll,
        "spd": spd_rate,
        "drift": drift_error,
        "covariance": covariance_error,
        "time": elapsed,
    }


def aggregate(
    results,
):
    def values(name):
        return np.asarray(
            [
                result[name]
                for result in results
            ],
            dtype=float,
        )

    nll = values("nll")
    spd = values("spd")
    drift = values("drift")
    covariance = values(
        "covariance"
    )

    return {
        "experiment": results[0][
            "experiment"
        ],
        "M": results[0]["M"],
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
        "mean_drift": float(
            np.mean(drift)
        ),
        "std_drift": float(
            np.std(
                drift,
                ddof=1,
            )
        ),
        "mean_covariance": float(
            np.mean(covariance)
        ),
        "std_covariance": float(
            np.std(
                covariance,
                ddof=1,
            )
        ),
    }


def print_run(
    result,
):
    print(
        f"    seed={result['seed']}  "
        f"NLL={result['nll']: .8e}  "
        f"SPD={result['spd']:.6f}  "
        f"drift={result['drift']:.8e}  "
        f"cov={result['covariance']:.8e}  "
        f"time={result['time']:.2f}s"
    )


def print_aggregate(
    result,
):
    print(
        f"M={result['M']:3d}  "
        f"NLL={result['mean_nll']: .8e} "
        f"+/- {result['std_nll']:.3e}  "
        f"SPD={result['mean_spd']:.6f} "
        f"+/- {result['std_spd']:.3e}  "
        f"drift={result['mean_drift']:.8e} "
        f"+/- {result['std_drift']:.3e}  "
        f"cov={result['mean_covariance']:.8e} "
        f"+/- {result['std_covariance']:.3e}"
    )


def main():
    print(
        "ARFF remaining-experiment multi-seed tuning"
    )
    print(
        "==========================================="
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
    print()

    final_results = []

    for experiment_name, iterations in (
        CANDIDATES.items()
    ):
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

        print(experiment_name)
        print(
            "-" * len(
                experiment_name
            )
        )
        print(
            f"K            : "
            f"{config.fourier_frequencies}"
        )
        print(
            f"M candidates : {iterations}"
        )
        print(
            f"train N      : "
            f"{len(data.train_idx)}"
        )
        print(
            f"validation N : "
            f"{len(data.validation_idx)}"
        )
        print()

        summaries = []

        for n_iterations in iterations:
            print(
                f"M={n_iterations}"
            )

            runs = []

            for seed in TUNING_SEEDS:
                result = run_candidate(
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

                runs.append(
                    result
                )

                print_run(
                    result
                )

            summary = aggregate(
                runs
            )

            summaries.append(
                summary
            )

            print(
                "  aggregate:"
            )
            print(
                "  ",
                end="",
            )
            print_aggregate(
                summary
            )
            print()

        best = min(
            summaries,
            key=lambda item: item[
                "mean_nll"
            ],
        )

        final_results.append(
            best
        )

        print(
            "selected by mean validation NLL:"
        )
        print_aggregate(
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

    for result in final_results:
        print(
            f"{result['experiment']}: "
            f"M={result['M']}  "
            f"mean validation NLL="
            f"{result['mean_nll']:.8e} "
            f"+/- {result['std_nll']:.3e}  "
            f"mean SPD="
            f"{result['mean_spd']:.6f}"
        )


if __name__ == "__main__":
    main()