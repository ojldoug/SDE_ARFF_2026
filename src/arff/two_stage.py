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

from src.arff.covariance import (
    covariance_targets,
)
from src.arff.regression import (
    ARFFModel,
    fit_arff,
    make_compiled_adaptation_step,
    predict,
)
from src.experiments.config import (
    ARFFConfig,
)


Array = jax.Array


@dataclass(frozen=True)
class TwoStageARFFModel:
    drift: ARFFModel
    covariance: ARFFModel
    diff_type: str


@dataclass(frozen=True)
class CrossFitResult:
    covariance_targets: Array
    fold_id: np.ndarray


def _fit_regression(
    key,
    x,
    y,
    *,
    K: int,
    config: ARFFConfig,
    compiled_adaptation_step,
):
    """
    Fit one ARFF regression using a shared compiled adaptation kernel.
    """
    return fit_arff(
        key,
        x,
        y,
        K=K,
        n_iterations=config.M_max,
        lambda_reg=config.lambda_reg,
        gamma=config.gamma,
        delta=config.delta,
        resampling=config.resampling,
        metropolis_test=(
            config.metropolis_test
        ),
        compiled_adaptation_step=(
            compiled_adaptation_step
        ),
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
        raise ValueError(
            "Cross-fitting requires at least "
            "two folds."
        )

    if n_folds > n_samples:
        raise ValueError(
            "Number of folds cannot exceed "
            "number of samples."
        )

    rng = np.random.default_rng(
        seed
    )

    permutation = rng.permutation(
        n_samples
    )

    return np.array_split(
        permutation,
        n_folds,
    )


def cross_fitted_covariance_targets(
    key,
    x,
    r,
    h,
    *,
    K: int,
    diff_type: str,
    config: ARFFConfig,
    fold_seed: int,
    compiled_adaptation_step,
):
    """
    Construct covariance targets using out-of-fold drift predictions.

    Every training observation receives a drift prediction from a model
    fitted without that observation.

    Numerical arrays remain on the JAX device throughout cross-fitting.
    Only the small fold-index bookkeeping is maintained as NumPy arrays.
    """
    x = jnp.asarray(x)
    r = jnp.asarray(r)
    h = jnp.asarray(h)

    n = len(x)

    if (
        len(r) != n
        or len(h) != n
    ):
        raise ValueError(
            "x, r, and h must have equal "
            "sample counts."
        )

    folds = make_folds(
        n,
        config.n_folds,
        fold_seed,
    )

    if diff_type == "diagonal":
        target_dimension = (
            r.shape[1]
        )

    else:
        d = r.shape[1]

        target_dimension = (
            d
            * (d + 1)
            // 2
        )

    targets = jnp.zeros(
        (
            n,
            target_dimension,
        ),
        dtype=x.dtype,
    )

    fold_id = np.empty(
        n,
        dtype=np.int32,
    )

    all_indices = np.arange(
        n
    )

    for k, holdout_idx in enumerate(
        folds
    ):
        train_mask = np.ones(
            n,
            dtype=bool,
        )

        train_mask[
            holdout_idx
        ] = False

        fit_idx = all_indices[
            train_mask
        ]

        # Index arrays are small host-side bookkeeping. The gathered
        # numerical arrays themselves remain JAX device arrays.
        x_fit = x[fit_idx]
        r_fit = r[fit_idx]
        h_fit = h[fit_idx]

        drift_target = (
            r_fit
            / h_fit
        )

        key, drift_fold = (
            _fit_regression(
                key,
                x_fit,
                drift_target,
                K=K,
                config=config,
                compiled_adaptation_step=(
                    compiled_adaptation_step
                ),
            )
        )

        x_holdout = x[
            holdout_idx
        ]

        r_holdout = r[
            holdout_idx
        ]

        h_holdout = h[
            holdout_idx
        ]

        drift_holdout = predict(
            drift_fold,
            x_holdout,
        )

        residual = (
            r_holdout
            - h_holdout
            * drift_holdout
        )

        fold_targets = (
            covariance_targets(
                residual,
                h_holdout,
                diff_type,
            )
        )

        targets = targets.at[
            holdout_idx
        ].set(
            fold_targets
        )

        fold_id[
            holdout_idx
        ] = k

    targets_finite = bool(
        jax.device_get(
            jnp.all(
                jnp.isfinite(
                    targets
                )
            )
        )
    )

    if not targets_finite:
        raise RuntimeError(
            "Cross-fitted covariance targets "
            "contain non-finite values."
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
    K: int,
    diff_type: str,
    config: ARFFConfig,
    fold_seed: int,
    compiled_adaptation_step=None,
):
    """
    Fit final drift and covariance ARFF models on one training set.

    One compiled ARFF adaptation kernel is reused across all cross-fit
    drift regressions and the final drift/covariance regressions. JAX
    automatically caches separate executables when distinct sample
    shapes occur.
    """
    x = jnp.asarray(x)
    r = jnp.asarray(r)
    h = jnp.asarray(h)

    if compiled_adaptation_step is None:
        compiled_adaptation_step = (
            make_compiled_adaptation_step(
                delta=config.delta,
                lambda_reg=(
                    config.lambda_reg
                ),
                gamma=config.gamma,
                resampling=(
                    config.resampling
                ),
                metropolis_test=(
                    config.metropolis_test
                ),
            )
        )

    key, crossfit = (
        cross_fitted_covariance_targets(
            key,
            x,
            r,
            h,
            K=K,
            diff_type=diff_type,
            config=config,
            fold_seed=fold_seed,
            compiled_adaptation_step=(
                compiled_adaptation_step
            ),
        )
    )

    # Final drift uses all training observations.
    drift_target = (
        r
        / h
    )

    key, drift_model = (
        _fit_regression(
            key,
            x,
            drift_target,
            K=K,
            config=config,
            compiled_adaptation_step=(
                compiled_adaptation_step
            ),
        )
    )

    # Stage-2 covariance fit uses all training x paired with honest
    # out-of-fold covariance targets.
    key, covariance_model = (
        _fit_regression(
            key,
            x,
            crossfit.covariance_targets,
            K=K,
            config=config,
            compiled_adaptation_step=(
                compiled_adaptation_step
            ),
        )
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