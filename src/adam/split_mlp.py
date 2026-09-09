"""Complete MLP split estimator; does not modify joint/Fourier production code.

Drift: MSE on r/h. Cross-fitting: deterministic training-only folds, with
checkpoint selection on the canonical validation split. Covariance: Gaussian
NLL on frozen out-of-fold residuals. Final drift: full-training refit.

Cross-fitting and stage-specific checkpoint selection are additional differences
from joint fitting, not merely an alternative order of identical joint updates.
Every regression runs the prescribed epoch budget; selecting an early best
checkpoint does NOT stop training early.

Compiled update/validation functions are supplied by the runner and reused.
The stage-aligned history timestamps follow the existing Fourier-split archive
convention. They add the stage-call offset to the generic trainer's local clock;
a small setup interval before that local clock starts is not resolved exactly.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from src.adam.mlp import AdamMLPModel, EPS, MLPParams, initialize_model, predict_mlp
from src.adam.training import TrainingResult, fit_adam
from src.arff.two_stage import make_folds


def _block_until_ready(tree: Any) -> None:
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, 'block_until_ready'):
            leaf.block_until_ready()


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class CovarianceMLPModel:
    covariance: MLPParams
    diff_type: str
    output_dimension: int

    def tree_flatten(self):
        return (self.covariance,), (self.diff_type, self.output_dimension)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(children[0], aux_data[0], aux_data[1])


@dataclass(frozen=True)
class CrossFitMLPResult:
    residuals: Any
    fold_id: np.ndarray
    fold_best_epochs: np.ndarray
    fold_best_validation_losses: np.ndarray
    fold_algorithm_times: np.ndarray
    crossfit_algorithm_time: float


@dataclass(frozen=True)
class SplitMLPTrainingResult:
    model: AdamMLPModel
    final_drift_training: TrainingResult
    covariance_training: TrainingResult
    crossfit: CrossFitMLPResult
    final_drift_start_offset: float
    final_drift_algorithm_time: float
    covariance_start_offset: float
    covariance_algorithm_time: float
    internal_algorithm_time: float


def drift_mse(params, x, target, h):
    del h
    return jnp.mean((predict_mlp(params, x) - target) ** 2)


def covariance_factor_from_split_model(model: CovarianceMLPModel, x):
    raw = predict_mlp(model.covariance, x)
    d = model.output_dimension
    if model.diff_type == 'diagonal':
        if raw.shape[-1] != d:
            raise ValueError('Unexpected diagonal covariance output size')
        return jax.vmap(jnp.diag)(jnp.sqrt(jax.nn.softplus(raw) + EPS))
    if raw.shape[-1] != d * (d + 1) // 2:
        raise ValueError('Unexpected full covariance output size')
    rows, cols = jnp.tril_indices(d)
    factor = jnp.zeros((len(x), d, d), dtype=x.dtype)
    factor = factor.at[:, rows, cols].set(raw)
    diagonal = jax.nn.softplus(jnp.diagonal(factor, axis1=-2, axis2=-1)) + EPS
    idx = jnp.arange(d)
    return factor.at[:, idx, idx].set(diagonal)


def covariance_from_split_model(model, x):
    factor = covariance_factor_from_split_model(model, x)
    return factor @ jnp.swapaxes(factor, -1, -2)


def covariance_gaussian_nll(model, x, residual, h):
    x, residual, h = jnp.asarray(x), jnp.asarray(residual), jnp.asarray(h)
    if h.ndim != 2 or h.shape[1] != 1:
        raise ValueError('h must have shape (N, 1)')
    factor = covariance_factor_from_split_model(model, x)
    scale = jnp.sqrt(h)[:, :, None] * factor
    whitened = jax.vmap(
        lambda a, b: jax.scipy.linalg.solve_triangular(a, b, lower=True)
    )(scale, residual)
    quadratic = jnp.sum(whitened ** 2, axis=1)
    logdet = 2 * jnp.sum(jnp.log(jnp.diagonal(scale, axis1=-2, axis2=-1)), axis=1)
    return jnp.mean(.5 * (quadratic + logdet + residual.shape[1] * jnp.log(2 * jnp.pi)))


def _fit_one_regression(key, initial_model, x_train, target_train, h_train,
                        x_validation, target_validation, h_validation, *, epochs,
                        batch_size, optimizer, compiled_train_step, compiled_loss):
    return fit_adam(
        key, initial_model, x_train, target_train, h_train,
        x_validation, target_validation, h_validation,
        epochs=epochs, batch_size=batch_size, optimizer=optimizer,
        compiled_train_step=compiled_train_step, compiled_nll=compiled_loss,
    )


def cross_fitted_residuals_mlp(
    key, x_train, r_train, h_train, x_validation, r_validation, h_validation, *,
    input_dimension, output_dimension, diff_type, hidden_sizes, n_folds, fold_seed,
    epochs, batch_size, drift_optimizer, drift_compiled_train_step, drift_compiled_loss,
):
    start = time.perf_counter()
    n = len(x_train)
    folds = make_folds(n, n_folds, fold_seed)
    residuals = jnp.zeros_like(r_train)
    fold_id = np.full(n, -1, dtype=np.int32)
    best_epochs = np.empty(n_folds, dtype=np.int32)
    best_losses = np.empty(n_folds, dtype=np.float64)
    fold_times = np.empty(n_folds, dtype=np.float64)
    all_indices = np.arange(n)
    validation_target = r_validation / h_validation
    for k, holdout in enumerate(folds):
        fold_start = time.perf_counter()
        mask = np.ones(n, dtype=bool)
        mask[holdout] = False
        fit_idx = all_indices[mask]
        key, initialization_key = jax.random.split(key)
        # Use the canonical MODEL constructor, not an assumed output_scale
        # for standalone drift initialization. This preserves its settings.
        fold_model = initialize_model(
            initialization_key, input_dimension=input_dimension,
            output_dimension=output_dimension, diff_type=diff_type,
            hidden_sizes=tuple(hidden_sizes),
        )
        key, training = _fit_one_regression(
            key, fold_model.drift, x_train[fit_idx], r_train[fit_idx] / h_train[fit_idx],
            h_train[fit_idx], x_validation, validation_target, h_validation,
            epochs=epochs, batch_size=batch_size, optimizer=drift_optimizer,
            compiled_train_step=drift_compiled_train_step, compiled_loss=drift_compiled_loss,
        )
        error = r_train[holdout] - h_train[holdout] * predict_mlp(training.model, x_train[holdout])
        residuals = residuals.at[holdout].set(error)
        _block_until_ready(residuals)
        fold_id[holdout] = k
        best_epochs[k], best_losses[k] = training.best_epoch, training.best_validation_nll
        fold_times[k] = time.perf_counter() - fold_start
    if np.any(fold_id < 0) or not bool(jax.device_get(jnp.all(jnp.isfinite(residuals)))):
        raise RuntimeError('Invalid out-of-fold residuals or incomplete fold assignment')
    return key, CrossFitMLPResult(residuals, fold_id, best_epochs, best_losses,
                                  fold_times, time.perf_counter() - start)


def fit_split_mlp_adam(
    key, x_train, r_train, h_train, x_validation, r_validation, h_validation, *,
    input_dimension, output_dimension, diff_type, hidden_sizes, n_folds, fold_seed,
    epochs, batch_size, drift_optimizer, drift_compiled_train_step, drift_compiled_loss,
    covariance_optimizer, covariance_compiled_train_step, covariance_compiled_loss,
):
    start = time.perf_counter()
    hidden_sizes = tuple(hidden_sizes)
    x_train, r_train, h_train = map(jnp.asarray, (x_train, r_train, h_train))
    x_validation, r_validation, h_validation = map(jnp.asarray, (x_validation, r_validation, h_validation))
    if epochs <= 0 or batch_size <= 0 or not len(x_validation):
        raise ValueError('Positive epochs/batch size and nonempty validation data required')
    for x, r, h in [(x_train, r_train, h_train), (x_validation, r_validation, h_validation)]:
        if x.ndim != 2 or r.shape != (len(x), output_dimension) or h.shape != (len(x), 1):
            raise ValueError('Expected x=(N,input), r=(N,output), h=(N,1)')
    key, initialization_key = jax.random.split(key)
    initial = initialize_model(initialization_key, input_dimension=input_dimension,
                               output_dimension=output_dimension, diff_type=diff_type,
                               hidden_sizes=hidden_sizes)
    key, crossfit = cross_fitted_residuals_mlp(
        key, x_train, r_train, h_train, x_validation, r_validation, h_validation,
        input_dimension=input_dimension, output_dimension=output_dimension,
        diff_type=diff_type, hidden_sizes=hidden_sizes, n_folds=n_folds, fold_seed=fold_seed,
        epochs=epochs, batch_size=batch_size, drift_optimizer=drift_optimizer,
        drift_compiled_train_step=drift_compiled_train_step, drift_compiled_loss=drift_compiled_loss,
    )
    drift_start = time.perf_counter()
    key, drift_training = _fit_one_regression(
        key, initial.drift, x_train, r_train / h_train, h_train,
        x_validation, r_validation / h_validation, h_validation,
        epochs=epochs, batch_size=batch_size, optimizer=drift_optimizer,
        compiled_train_step=drift_compiled_train_step, compiled_loss=drift_compiled_loss,
    )
    _block_until_ready(drift_training.model)
    drift_time = time.perf_counter() - drift_start
    validation_residual = r_validation - h_validation * predict_mlp(drift_training.model, x_validation)
    _block_until_ready(validation_residual)
    covariance_initial = CovarianceMLPModel(initial.covariance, diff_type, output_dimension)
    covariance_start = time.perf_counter()
    key, covariance_training = _fit_one_regression(
        key, covariance_initial, x_train, crossfit.residuals, h_train,
        x_validation, validation_residual, h_validation,
        epochs=epochs, batch_size=batch_size, optimizer=covariance_optimizer,
        compiled_train_step=covariance_compiled_train_step, compiled_loss=covariance_compiled_loss,
    )
    _block_until_ready(covariance_training.model)
    covariance_time = time.perf_counter() - covariance_start
    final_model = AdamMLPModel(drift=drift_training.model,
                               covariance=covariance_training.model.covariance, diff_type=diff_type)
    _block_until_ready(final_model)
    # Compare against the real joint implementation AFTER benchmark timing
    # in the runner; do not time a duplicated, tautological consistency check.
    result = SplitMLPTrainingResult(
        model=final_model, final_drift_training=drift_training,
        covariance_training=covariance_training, crossfit=crossfit,
        final_drift_start_offset=drift_start-start, final_drift_algorithm_time=drift_time,
        covariance_start_offset=covariance_start-start, covariance_algorithm_time=covariance_time,
        internal_algorithm_time=time.perf_counter()-start,
    )
    return key, result
