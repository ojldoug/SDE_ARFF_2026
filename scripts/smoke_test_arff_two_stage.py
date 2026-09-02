#!/usr/bin/env python3
"""Smoke test for cross-fitted two-stage ARFF learning."""

from dataclasses import replace
from pathlib import Path
import sys

import jax
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.arff.covariance import (
    raw_covariance,
    spd_violation_mask,
)
from src.arff.regression import predict
from src.arff.two_stage import fit_two_stage_arff
from src.experiments.config import get_config
from src.experiments.definitions import get_experiment

from src.experiments.em_data import generate_standard_em_data


def main():

    definition = get_experiment("ex1")
    config = get_config("ex1")

    smoke_data = replace(
        config.data,
        n_trajectories=256,
        target_samples=256,
    )
    smoke_config = replace(
        config,
        data=smoke_data,
    )

    x, r, h = generate_standard_em_data(
        definition,
        smoke_config,
    )

    smoke_arff = replace(
        config.arff,
        M_min=10,
        M_max=10,
        n_folds=5,
    )

    key = jax.random.PRNGKey(1234)

    key, model, crossfit = fit_two_stage_arff(
        key,
        x,
        r,
        h,
        K=32,
        diff_type=definition.diff_type,
        config=smoke_arff,
        fold_seed=config.split.seed,
    )

    assert crossfit.covariance_targets.shape == (
        len(x),
        definition.n_dimensions,
    )
    assert crossfit.fold_id.shape == (len(x),)

    counts = np.bincount(
        crossfit.fold_id,
        minlength=smoke_arff.n_folds,
    )

    assert counts.sum() == len(x)

    # np.array_split distributes samples as evenly as possible.
    assert counts.max() - counts.min() <= 1

    drift = np.asarray(predict(model.drift, x))
    covariance = np.asarray(
        raw_covariance(
            model.covariance,
            x,
            definition.diff_type,
        )
    )

    assert drift.shape == (
        len(x),
        definition.n_dimensions,
    )

    assert covariance.shape == (
        len(x),
        definition.n_dimensions,
        definition.n_dimensions,
    )

    assert np.all(np.isfinite(drift))
    assert np.all(np.isfinite(covariance))

    violations = np.asarray(
        spd_violation_mask(covariance)
    )

    print("Two-stage ARFF smoke test")
    print("-------------------------")
    print(f"samples              : {len(x)}")
    print(f"fold counts          : {counts}")
    print(
        "covariance targets   :",
        crossfit.covariance_targets.shape,
    )
    print(f"drift prediction     : {drift.shape}")
    print(f"covariance prediction: {covariance.shape}")
    print(
        "raw SPD violations   : "
        f"{violations.mean():.4f}"
    )
    print()
    print("All two-stage ARFF checks passed.")


if __name__ == "__main__":
    main()
