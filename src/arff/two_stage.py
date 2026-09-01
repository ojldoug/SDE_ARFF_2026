"""
Two-stage ARFF learning for SDE drift and covariance.

Stage 1:
    Fit the drift from increment rates r / h.

Stage 2:
    Construct cross-fitted residual covariance targets

        C_n = e_n e_n^T / h_n,

    where e_n = r_n - h_n f_hat^(-fold)(x_n),

    and fit an ARFF covariance model to those targets.

A final drift model is fitted on all training observations after
cross-fitting.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from src.arff.covariance import covariance_targets
from src.arff.regression import ARFFModel, fit_arff, predict
from src.experiments.config import ARFFConfig


@dataclass(frozen=True)
class TwoStageARFFModel:
    drift: ARFFModel
    covariance: ARFFModel
    diff_type: str


@dataclass(frozen=True)
class CrossFitResult:
    covariance_targets: np.ndarray
    fold_id: np.ndarray


def _fit_regression(
    key,
    x,
    y,
    config: ARFFConfig,
):
    return fit_arff(
        key,
        x,
        y,
        K=config.K,
        n_iterations=config.M_max,
        lambda_reg=config.lambda_reg,
        gamma=config.gamma,
        delta=config.delta,
        resampling=config.resampling,
        metropolis_test=config.metropolis_test,
    )


def make_folds(
    n_samples: int,
    n_folds: int,
    seed: int,
):
    """
    Deterministically partition training-set positions into folds.
    """
    if n_folds < 2:
        raise ValueError("Cross-fitting requires at least two folds.")

    if n_folds > n_samples:
        raise ValueError(
            "Number of folds cannot exceed number of samples."
        )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n_samples)

    return np.array_split(permutation, n_folds)


def cross_fitted_covariance_targets(
    key,
    x,
    r,
    h,
    *,
    diff_type: str,
    config: ARFFConfig,
    fold_seed: int,
):
    """
    Construct covariance targets using out-of-fold drift predictions.

    Every training observation receives a drift prediction from a model
    fitted without that observation.
    """
    x = np.asarray(x)
    r = np.asarray(r)
    h = np.asarray(h)

    n = len(x)

    if len(r) != n or len(h) != n:
        raise ValueError("x, r, and h must have equal sample counts.")

    folds = make_folds(
        n,
        config.n_folds,
        fold_seed,
    )

    if diff_type == "diagonal":
        target_dimension = r.shape[1]
    else:
        d = r.shape[1]
        target_dimension = d * (d + 1) // 2

    targets = np.empty(
        (n, target_dimension),
        dtype=x.dtype,
    )
    fold_id = np.empty(n, dtype=np.int32)

    all_indices = np.arange(n)

    for k, holdout_idx in enumerate(folds):
        train_mask = np.ones(n, dtype=bool)
        train_mask[holdout_idx] = False
        fit_idx = all_indices[train_mask]

        drift_target = (
            r[fit_idx]
            / h[fit_idx]
        )

        key, drift_fold = _fit_regression(
            key,
            x[fit_idx],
            drift_target,
            config,
        )

        drift_holdout = np.asarray(
            predict(
                drift_fold,
                jnp.asarray(x[holdout_idx]),
            )
        )

        residual = (
            r[holdout_idx]
            - h[holdout_idx] * drift_holdout
        )

        fold_targets = np.asarray(
            covariance_targets(
                jnp.asarray(residual),
                jnp.asarray(h[holdout_idx]),
                diff_type,
            )
        )

        targets[holdout_idx] = fold_targets
        fold_id[holdout_idx] = k

    if not np.all(np.isfinite(targets)):
        raise RuntimeError(
            "Cross-fitted covariance targets contain non-finite values."
        )

    return key, CrossFitResult(
        covariance_targets=targets,
        fold_id=fold_id,
    )


def fit_two_stage_arff(
    key,
    x,
    r,
    h,
    *,
    diff_type: str,
    config: ARFFConfig,
    fold_seed: int,
):
    """
    Fit final drift and covariance ARFF models on one training set.
    """
    x = np.asarray(x)
    r = np.asarray(r)
    h = np.asarray(h)

    key, crossfit = cross_fitted_covariance_targets(
        key,
        x,
        r,
        h,
        diff_type=diff_type,
        config=config,
        fold_seed=fold_seed,
    )

    # Final drift uses all training observations.
    drift_target = r / h

    key, drift_model = _fit_regression(
        key,
        x,
        drift_target,
        config,
    )

    # Stage-2 covariance fit uses all training x paired with honest
    # out-of-fold covariance targets.
    key, covariance_model = _fit_regression(
        key,
        x,
        crossfit.covariance_targets,
        config,
    )

    return (
        key,
        TwoStageARFFModel(
            drift=drift_model,
            covariance=covariance_model,
            diff_type=diff_type,
        ),
        crossfit,
    )
