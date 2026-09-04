#!/usr/bin/env python3
"""
Diagnose Experiment 5 ARFF covariance learning.

This script fits the canonical seed-0 two-stage ARFF model and compares:

1. cross-fitted stage-2 covariance targets;
2. raw learned covariance predictions;
3. the known effective SIR covariance.

No hyperparameters are changed and no test information is used for
model selection.
"""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.arff.covariance import raw_covariance
from src.arff.regression import make_compiled_adaptation_step
from src.arff.two_stage import fit_two_stage_arff
from src.experiments.config import get_config
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment


def describe(name, values):
    values = np.asarray(values)

    print(name)
    print("-" * len(name))
    print(f"shape       : {values.shape}")
    print(f"min         : {np.min(values):.8e}")
    print(f"p01         : {np.percentile(values, 1):.8e}")
    print(f"p05         : {np.percentile(values, 5):.8e}")
    print(f"median      : {np.median(values):.8e}")
    print(f"mean        : {np.mean(values):.8e}")
    print(f"p95         : {np.percentile(values, 95):.8e}")
    print(f"p99         : {np.percentile(values, 99):.8e}")
    print(f"max         : {np.max(values):.8e}")
    print(f"negative    : {np.mean(values < 0.0):.8f}")
    print(f"nonpositive : {np.mean(values <= 0.0):.8f}")
    print()


def true_covariance(
    definition,
    x,
):
    sigma = np.asarray(
        definition.diffusion_factor(
            jnp.asarray(x)
        )
    )

    return (
        sigma
        @ np.swapaxes(
            sigma,
            -1,
            -2,
        )
    )


def report_split(
    label,
    model,
    definition,
    x,
):
    x = np.asarray(x)

    learned = np.asarray(
        raw_covariance(
            model.covariance,
            jnp.asarray(x),
            model.diff_type,
        )
    )

    truth = true_covariance(
        definition,
        x,
    )

    print()
    print("=" * 72)
    print(label)
    print("=" * 72)
    print()

    # Experiment 5 is diagonal, but keep this check explicit.
    learned_diag = np.diagonal(
        learned,
        axis1=-2,
        axis2=-1,
    )

    truth_diag = np.diagonal(
        truth,
        axis1=-2,
        axis2=-1,
    )

    for j in range(
        learned_diag.shape[1]
    ):
        describe(
            f"learned covariance diagonal {j}",
            learned_diag[:, j],
        )

        describe(
            f"true covariance diagonal {j}",
            truth_diag[:, j],
        )

        rmse = np.sqrt(
            np.mean(
                (
                    learned_diag[:, j]
                    - truth_diag[:, j]
                ) ** 2
            )
        )

        print(
            f"component {j} RMSE: "
            f"{rmse:.8e}"
        )
        print()

    overall_rmse = np.sqrt(
        np.mean(
            (
                learned
                - truth
            ) ** 2
        )
    )

    print(
        "overall raw covariance RMSE: "
        f"{overall_rmse:.8e}"
    )

    print(
        "fraction with any nonpositive "
        "diagonal entry: "
        f"{np.mean(np.any(learned_diag <= 0.0, axis=1)):.8f}"
    )


def main():
    name = "ex5"
    seed = 0

    config = get_config(name)
    definition = get_experiment(name)

    data = load_dataset(
        REPO_ROOT
        / "data"
        / f"{name}.npz"
    )

    train_idx = data.train_idx
    validation_idx = data.validation_idx
    test_idx = data.test_idx

    x_train = jnp.asarray(
        data.x[train_idx]
    )
    r_train = jnp.asarray(
        data.r[train_idx]
    )
    h_train = jnp.asarray(
        data.h[train_idx]
    )

    print("Experiment 5 covariance diagnostic")
    print("----------------------------------")
    print(f"train N       : {len(train_idx)}")
    print(f"validation N  : {len(validation_idx)}")
    print(f"test N        : {len(test_idx)}")
    print(f"K             : {config.fourier_frequencies}")
    print(f"iterations    : {config.arff.M_max}")
    print(f"lambda        : {config.arff.lambda_reg:.8e}")
    print(f"delta         : {config.arff.delta:.8e}")
    print(f"gamma         : {config.arff.gamma:.8e}")
    print(f"resampling    : {config.arff.resampling}")
    print(f"Metropolis    : {config.arff.metropolis_test}")
    print()

    compiled_step = make_compiled_adaptation_step(
        delta=config.arff.delta,
        lambda_reg=config.arff.lambda_reg,
        gamma=config.arff.gamma,
        resampling=config.arff.resampling,
        metropolis_test=config.arff.metropolis_test,
    )

    key = jax.random.PRNGKey(seed)

    print("Fitting canonical seed-0 ARFF model...")

    key, model, crossfit = fit_two_stage_arff(
        key,
        x_train,
        r_train,
        h_train,
        K=config.fourier_frequencies,
        diff_type=definition.diff_type,
        config=config.arff,
        fold_seed=config.split.seed,
        compiled_adaptation_step=compiled_step,
    )

    jax.block_until_ready(model)

    targets = np.asarray(
        crossfit.covariance_targets
    )

    print()
    print("=" * 72)
    print("CROSS-FITTED STAGE-2 TARGETS")
    print("=" * 72)
    print()

    for j in range(
        targets.shape[1]
    ):
        describe(
            f"covariance target component {j}",
            targets[:, j],
        )

    print(
        "fraction of targets with any "
        "negative component: "
        f"{np.mean(np.any(targets < 0.0, axis=1)):.8f}"
    )

    print(
        "fraction of targets with any "
        "zero component: "
        f"{np.mean(np.any(targets == 0.0, axis=1)):.8f}"
    )

    report_split(
        "TRAIN",
        model,
        definition,
        data.x[train_idx],
    )

    report_split(
        "VALIDATION",
        model,
        definition,
        data.x[validation_idx],
    )

    report_split(
        "TEST",
        model,
        definition,
        data.x[test_idx],
    )


if __name__ == "__main__":
    main()