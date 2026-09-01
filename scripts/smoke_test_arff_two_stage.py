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


def main():
    data = np.load(
        REPO_ROOT / "data" / "ex1.npz",
        allow_pickle=False,
    )

    definition = get_experiment("ex1")
    config = get_config("ex1")

    # Tiny training subset and cheap ARFF settings for plumbing test.
    idx = data["train_idx"][:500]

    x = data["x_data"][idx]
    r = data["r_data"][idx]
    h = data["step_sizes"][idx]

    smoke_arff = replace(
        config.arff,
        K=32,
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
        diff_type=definition.diff_type,
        config=smoke_arff,
        fold_seed=config.split.seed,
    )

    assert crossfit.covariance_targets.shape == (500, 2)
    assert crossfit.fold_id.shape == (500,)

    counts = np.bincount(
        crossfit.fold_id,
        minlength=smoke_arff.n_folds,
    )

    assert counts.sum() == 500
    assert np.all(counts == 100)

    drift = np.asarray(predict(model.drift, x))
    covariance = np.asarray(
        raw_covariance(
            model.covariance,
            x,
            definition.diff_type,
        )
    )

    assert drift.shape == (500, 2)
    assert covariance.shape == (500, 2, 2)

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
