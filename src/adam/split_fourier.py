"""
Two-stage Fourier-feature SDE learning with Adam.

The split formulation retains the same Fourier representation,
intrinsically-SPD covariance parameterization, Gaussian covariance
likelihood, and Adam optimizer as the joint Fourier-Adam baseline.

Stage 1
-------
Fit the drift from increment rates

    y_n = r_n / h_n

with Adam and mean squared error.

Cross-fitting
-------------
Partition the canonical training set into deterministic folds.
For every training observation, obtain a drift prediction from a model
fitted without that observation and construct

    e_n = r_n - h_n f_hat^(-fold)(x_n).

Stage 2
-------
Freeze those honest out-of-fold residuals and train only the
covariance parameters with

    1/2 log det Sigma(x_n)
    + 1/(2 h_n) e_n^T Sigma(x_n)^(-1) e_n.

Finally, refit the drift model on all canonical training observations.

All JIT-compiled functions are supplied by the runner after warm-up.
The timing information stored here therefore describes actual algorithm
work, excluding one-time JAX/XLA compilation.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import numpy as np

from src.adam.fourier import (
    AdamFourierModel,
    EPS,
    FourierParams,
    initialize_fourier_params,
    initialize_model,
    predict_fourier,
)
from src.adam.training import (
    TrainingResult,
    fit_adam,
)
from src.arff.two_stage import (
    make_folds,
)


Array = jax.Array


def _block_until_ready(tree) -> None:
    for leaf in jax.tree_util.tree_leaves(
        tree
    ):
        if hasattr(
            leaf,
            "block_until_ready",
        ):
            leaf.block_until_ready()


@dataclass(frozen=True)
class CovarianceFourierModel:
    covariance: FourierParams
    diff_type: str
    output_dimension: int

    def tree_flatten(self):
        children = (
            self.covariance,
        )

        aux_data = (
            self.diff_type,
            self.output_dimension,
        )

        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data,
        children,
    ):
        (
            diff_type,
            output_dimension,
        ) = aux_data

        (covariance,) = children

        return cls(
            covariance=covariance,
            diff_type=diff_type,
            output_dimension=(
                output_dimension
            ),
        )


jax.tree_util.register_pytree_node_class(
    CovarianceFourierModel
)


@dataclass(frozen=True)
class CrossFitAdamResult:
    residuals: Array

    fold_id: np.ndarray
    fold_best_epochs: np.ndarray
    fold_best_validation_losses: np.ndarray

    # Complete wall-clock cost of each fold after JIT warm-up.
    fold_algorithm_times: np.ndarray

    # Complete cross-fitting wall-clock cost, including fold setup,
    # predictions, and residual assembly.
    crossfit_algorithm_time: float


@dataclass(frozen=True)
class SplitFourierTrainingResult:
    model: AdamFourierModel

    final_drift_training: TrainingResult
    covariance_training: TrainingResult
    crossfit: CrossFitAdamResult

    # Offsets are measured from the beginning of the complete split
    # algorithm call.
    final_drift_start_offset: float
    final_drift_algorithm_time: float

    covariance_start_offset: float
    covariance_algorithm_time: float

    internal_algorithm_time: float


# ------------------------------------------------------------------
# Stage 1: drift objective
# ------------------------------------------------------------------


def drift_mse(
    params: FourierParams,
    x: Array,
    target: Array,
    h: Array,
) -> Array:
    del h

    prediction = predict_fourier(
        params,
        x,
    )

    return jnp.mean(
        (
            prediction
            - target
        ) ** 2
    )


# ------------------------------------------------------------------
# Covariance representation
# ------------------------------------------------------------------


def covariance_factor_from_split_model(
    model: CovarianceFourierModel,
    x: Array,
) -> Array:
    """
    Return L such that Sigma = L L^T.

    This matches the SPD covariance construction used by joint
    Fourier Adam.
    """
    raw = predict_fourier(
        model.covariance,
        x,
    )

    d = model.output_dimension

    if model.diff_type == "diagonal":
        diagonal_variance = (
            jax.nn.softplus(
                raw
            )
            + EPS
        )

        diagonal_std = jnp.sqrt(
            diagonal_variance
        )

        return jax.vmap(
            jnp.diag
        )(
            diagonal_std
        )

    expected = (
        d
        * (d + 1)
        // 2
    )

    if raw.shape[1] != expected:
        raise ValueError(
            "Unexpected covariance-output "
            "dimension."
        )

    rows, cols = jnp.tril_indices(
        d
    )

    L = jnp.zeros(
        (
            len(x),
            d,
            d,
        ),
        dtype=x.dtype,
    )

    L = L.at[
        :,
        rows,
        cols,
    ].set(
        raw
    )

    diagonal = jnp.diagonal(
        L,
        axis1=-2,
        axis2=-1,
    )

    diagonal = (
        jax.nn.softplus(
            diagonal
        )
        + EPS
    )

    idx = jnp.arange(
        d
    )

    L = L.at[
        :,
        idx,
        idx,
    ].set(
        diagonal
    )

    return L


def covariance_from_split_model(
    model: CovarianceFourierModel,
    x: Array,
) -> Array:
    L = covariance_factor_from_split_model(
        model,
        x,
    )

    return (
        L
        @ jnp.swapaxes(
            L,
            -1,
            -2,
        )
    )


# ------------------------------------------------------------------
# Stage 2: covariance-only Gaussian likelihood
# ------------------------------------------------------------------


def covariance_gaussian_nll(
    model: CovarianceFourierModel,
    x: Array,
    residual: Array,
    h: Array,
) -> Array:
    """
    Gaussian NLL with frozen drift residuals.

    residual_n = r_n - h_n f_hat(x_n).

    Only covariance parameters are trainable during this stage.
    """
    x = jnp.asarray(
        x
    )

    residual = jnp.asarray(
        residual
    )

    h = jnp.asarray(
        h
    )

    d = residual.shape[1]

    L = covariance_factor_from_split_model(
        model,
        x,
    )

    # Cov(residual | x) = h Sigma(x).
    scale = (
        jnp.sqrt(
            h
        )[
            :,
            :,
            None,
        ]
        * L
    )

    def solve_one(
        scale_i,
        residual_i,
    ):
        return (
            jax.scipy.linalg.solve_triangular(
                scale_i,
                residual_i,
                lower=True,
            )
        )

    whitened = jax.vmap(
        solve_one
    )(
        scale,
        residual,
    )

    quadratic = jnp.sum(
        whitened**2,
        axis=1,
    )

    diagonal = jnp.diagonal(
        scale,
        axis1=-2,
        axis2=-1,
    )

    logdet = (
        2.0
        * jnp.sum(
            jnp.log(
                diagonal
            ),
            axis=1,
        )
    )

    nll = 0.5 * (
        quadratic
        + logdet
        + d
        * jnp.log(
            2.0
            * jnp.pi
        )
    )

    return jnp.mean(
        nll
    )


# ------------------------------------------------------------------
# Generic Adam regression helper
# ------------------------------------------------------------------


def _fit_one_regression(
    key,
    initial_model,
    x_train,
    target_train,
    h_train,
    x_validation,
    target_validation,
    h_validation,
    *,
    epochs: int,
    batch_size: int,
    optimizer,
    compiled_train_step,
    compiled_loss,
):
    return fit_adam(
        key,
        initial_model,
        x_train,
        target_train,
        h_train,
        x_validation,
        target_validation,
        h_validation,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        compiled_train_step=(
            compiled_train_step
        ),
        compiled_nll=(
            compiled_loss
        ),
    )


# ------------------------------------------------------------------
# Cross-fitting
# ------------------------------------------------------------------


def cross_fitted_residuals_adam(
    key,
    x_train,
    r_train,
    h_train,
    x_validation,
    r_validation,
    h_validation,
    *,
    input_dimension: int,
    output_dimension: int,
    n_frequencies: int,
    n_folds: int,
    fold_seed: int,
    epochs: int,
    batch_size: int,
    drift_optimizer,
    drift_compiled_train_step,
    drift_compiled_loss,
):
    x_train = jnp.asarray(
        x_train
    )

    r_train = jnp.asarray(
        r_train
    )

    h_train = jnp.asarray(
        h_train
    )

    x_validation = jnp.asarray(
        x_validation
    )

    r_validation = jnp.asarray(
        r_validation
    )

    h_validation = jnp.asarray(
        h_validation
    )

    crossfit_start = (
        time.perf_counter()
    )

    n = len(
        x_train
    )

    folds = make_folds(
        n,
        n_folds,
        fold_seed,
    )

    residuals = jnp.zeros(
        r_train.shape,
        dtype=r_train.dtype,
    )

    fold_id = np.empty(
        n,
        dtype=np.int32,
    )

    fold_best_epochs = np.empty(
        n_folds,
        dtype=np.int32,
    )

    fold_best_validation_losses = (
        np.empty(
            n_folds,
            dtype=np.float64,
        )
    )

    fold_algorithm_times = np.empty(
        n_folds,
        dtype=np.float64,
    )

    all_indices = np.arange(
        n
    )

    validation_target = (
        r_validation
        / h_validation
    )

    for fold_number, holdout_idx in enumerate(
        folds
    ):
        fold_start = (
            time.perf_counter()
        )

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

        (
            key,
            initialization_key,
        ) = jax.random.split(
            key
        )

        initial_drift = (
            initialize_fourier_params(
                initialization_key,
                input_dimension=(
                    input_dimension
                ),
                output_dimension=(
                    output_dimension
                ),
                n_frequencies=(
                    n_frequencies
                ),
            )
        )

        fit_target = (
            r_train[
                fit_idx
            ]
            / h_train[
                fit_idx
            ]
        )

        (
            key,
            training,
        ) = _fit_one_regression(
            key,
            initial_drift,
            x_train[
                fit_idx
            ],
            fit_target,
            h_train[
                fit_idx
            ],
            x_validation,
            validation_target,
            h_validation,
            epochs=epochs,
            batch_size=batch_size,
            optimizer=(
                drift_optimizer
            ),
            compiled_train_step=(
                drift_compiled_train_step
            ),
            compiled_loss=(
                drift_compiled_loss
            ),
        )

        drift_holdout = (
            predict_fourier(
                training.model,
                x_train[
                    holdout_idx
                ],
            )
        )

        fold_residual = (
            r_train[
                holdout_idx
            ]
            - h_train[
                holdout_idx
            ]
            * drift_holdout
        )

        residuals = residuals.at[
            holdout_idx
        ].set(
            fold_residual
        )

        _block_until_ready(
            residuals
        )

        fold_id[
            holdout_idx
        ] = fold_number

        fold_best_epochs[
            fold_number
        ] = (
            training.best_epoch
        )

        fold_best_validation_losses[
            fold_number
        ] = (
            training.best_validation_nll
        )

        fold_algorithm_times[
            fold_number
        ] = (
            time.perf_counter()
            - fold_start
        )

    _block_until_ready(
        residuals
    )

    finite = bool(
        jax.device_get(
            jnp.all(
                jnp.isfinite(
                    residuals
                )
            )
        )
    )

    if not finite:
        raise RuntimeError(
            "Cross-fitted residuals contain "
            "non-finite values."
        )

    crossfit_algorithm_time = (
        time.perf_counter()
        - crossfit_start
    )

    return (
        key,
        CrossFitAdamResult(
            residuals=residuals,
            fold_id=fold_id,
            fold_best_epochs=(
                fold_best_epochs
            ),
            fold_best_validation_losses=(
                fold_best_validation_losses
            ),
            fold_algorithm_times=(
                fold_algorithm_times
            ),
            crossfit_algorithm_time=float(
                crossfit_algorithm_time
            ),
        ),
    )


# ------------------------------------------------------------------
# Complete split estimator
# ------------------------------------------------------------------


def fit_split_fourier_adam(
    key,
    x_train,
    r_train,
    h_train,
    x_validation,
    r_validation,
    h_validation,
    *,
    input_dimension: int,
    output_dimension: int,
    n_frequencies: int,
    diff_type: str,
    n_folds: int,
    fold_seed: int,
    epochs: int,
    batch_size: int,
    drift_optimizer,
    drift_compiled_train_step,
    drift_compiled_loss,
    covariance_optimizer,
    covariance_compiled_train_step,
    covariance_compiled_loss,
):
    """
    Fit the complete split Fourier-Adam estimator.

    Stage offsets use a common algorithm clock beginning at entry to
    this function.
    """
    algorithm_start = (
        time.perf_counter()
    )

    x_train = jnp.asarray(
        x_train
    )

    r_train = jnp.asarray(
        r_train
    )

    h_train = jnp.asarray(
        h_train
    )

    x_validation = jnp.asarray(
        x_validation
    )

    r_validation = jnp.asarray(
        r_validation
    )

    h_validation = jnp.asarray(
        h_validation
    )

    (
        key,
        final_initialization_key,
    ) = jax.random.split(
        key
    )

    canonical_initial_model = (
        initialize_model(
            final_initialization_key,
            input_dimension=(
                input_dimension
            ),
            output_dimension=(
                output_dimension
            ),
            n_frequencies=(
                n_frequencies
            ),
            diff_type=diff_type,
        )
    )

    # --------------------------------------------------------------
    # Cross-fitted drift residuals.
    # --------------------------------------------------------------

    (
        key,
        crossfit,
    ) = cross_fitted_residuals_adam(
        key,
        x_train,
        r_train,
        h_train,
        x_validation,
        r_validation,
        h_validation,
        input_dimension=(
            input_dimension
        ),
        output_dimension=(
            output_dimension
        ),
        n_frequencies=(
            n_frequencies
        ),
        n_folds=n_folds,
        fold_seed=fold_seed,
        epochs=epochs,
        batch_size=batch_size,
        drift_optimizer=(
            drift_optimizer
        ),
        drift_compiled_train_step=(
            drift_compiled_train_step
        ),
        drift_compiled_loss=(
            drift_compiled_loss
        ),
    )

    # --------------------------------------------------------------
    # Final all-training drift.
    # --------------------------------------------------------------

    final_drift_start_offset = (
        time.perf_counter()
        - algorithm_start
    )

    final_drift_start = (
        time.perf_counter()
    )

    drift_train_target = (
        r_train
        / h_train
    )

    drift_validation_target = (
        r_validation
        / h_validation
    )

    (
        key,
        final_drift_training,
    ) = _fit_one_regression(
        key,
        canonical_initial_model.drift,
        x_train,
        drift_train_target,
        h_train,
        x_validation,
        drift_validation_target,
        h_validation,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=(
            drift_optimizer
        ),
        compiled_train_step=(
            drift_compiled_train_step
        ),
        compiled_loss=(
            drift_compiled_loss
        ),
    )

    final_drift = (
        final_drift_training.model
    )

    _block_until_ready(
        final_drift
    )

    final_drift_algorithm_time = (
        time.perf_counter()
        - final_drift_start
    )

    # --------------------------------------------------------------
    # Fixed validation residuals for covariance checkpoint selection.
    # --------------------------------------------------------------

    validation_drift = (
        predict_fourier(
            final_drift,
            x_validation,
        )
    )

    validation_residual = (
        r_validation
        - h_validation
        * validation_drift
    )

    _block_until_ready(
        validation_residual
    )

    # --------------------------------------------------------------
    # Covariance likelihood stage.
    # --------------------------------------------------------------

    covariance_start_offset = (
        time.perf_counter()
        - algorithm_start
    )

    covariance_start = (
        time.perf_counter()
    )

    covariance_initial_model = (
        CovarianceFourierModel(
            covariance=(
                canonical_initial_model
                .covariance
            ),
            diff_type=diff_type,
            output_dimension=(
                output_dimension
            ),
        )
    )

    (
        key,
        covariance_training,
    ) = _fit_one_regression(
        key,
        covariance_initial_model,
        x_train,
        crossfit.residuals,
        h_train,
        x_validation,
        validation_residual,
        h_validation,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=(
            covariance_optimizer
        ),
        compiled_train_step=(
            covariance_compiled_train_step
        ),
        compiled_loss=(
            covariance_compiled_loss
        ),
    )

    _block_until_ready(
        covariance_training.model
    )

    covariance_algorithm_time = (
        time.perf_counter()
        - covariance_start
    )

    final_model = AdamFourierModel(
        drift=final_drift,
        covariance=(
            covariance_training
            .model
            .covariance
        ),
        diff_type=diff_type,
    )

    # --------------------------------------------------------------
    # Representation consistency check.
    # --------------------------------------------------------------

    n_check = min(
        16,
        len(x_validation),
    )

    check_x = x_validation[
        :n_check
    ]

    covariance_a = (
        covariance_from_split_model(
            covariance_training.model,
            check_x,
        )
    )

    L = (
        covariance_factor_from_split_model(
            covariance_training.model,
            check_x,
        )
    )

    covariance_b = (
        L
        @ jnp.swapaxes(
            L,
            -1,
            -2,
        )
    )

    consistent = bool(
        jax.device_get(
            jnp.allclose(
                covariance_a,
                covariance_b,
                rtol=1e-6,
                atol=1e-7,
            )
        )
    )

    if not consistent:
        raise RuntimeError(
            "Split covariance representation "
            "failed consistency check."
        )

    _block_until_ready(
        final_model
    )

    internal_algorithm_time = (
        time.perf_counter()
        - algorithm_start
    )

    return (
        key,
        SplitFourierTrainingResult(
            model=final_model,
            final_drift_training=(
                final_drift_training
            ),
            covariance_training=(
                covariance_training
            ),
            crossfit=crossfit,
            final_drift_start_offset=float(
                final_drift_start_offset
            ),
            final_drift_algorithm_time=float(
                final_drift_algorithm_time
            ),
            covariance_start_offset=float(
                covariance_start_offset
            ),
            covariance_algorithm_time=float(
                covariance_algorithm_time
            ),
            internal_algorithm_time=float(
                internal_algorithm_time
            ),
        ),
    )
