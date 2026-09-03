"""
Fourier-feature Gaussian-likelihood baseline trained with Adam.

This baseline uses the same Fourier model capacity as ARFF, but learns
drift and covariance parameters jointly by minimizing the Gaussian
negative log-likelihood.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import optax

from src.experiments.model_size import covariance_output_dimension


Array = jax.Array
EPS = 1e-8


@dataclass(frozen=True)
class FourierParams:
    omega: Array
    amp: Array

    def tree_flatten(self):
        children = (
            self.omega,
            self.amp,
        )
        return children, None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        omega, amp = children
        return cls(
            omega=omega,
            amp=amp,
        )


jax.tree_util.register_pytree_node_class(
    FourierParams
)


@dataclass(frozen=True)
class AdamFourierModel:
    drift: FourierParams
    covariance: FourierParams
    diff_type: str

    def tree_flatten(self):
        children = (
            self.drift,
            self.covariance,
        )

        # diff_type is static metadata, not a trainable parameter.
        aux_data = self.diff_type

        return children, aux_data

    @classmethod
    def tree_unflatten(
        cls,
        aux_data,
        children,
    ):
        drift, covariance = children

        return cls(
            drift=drift,
            covariance=covariance,
            diff_type=aux_data,
        )


jax.tree_util.register_pytree_node_class(
    AdamFourierModel
)


def fourier_features(
    omega: Array,
    x: Array,
) -> Array:
    projection = x @ omega

    return jnp.concatenate(
        [
            jnp.cos(projection),
            jnp.sin(projection),
        ],
        axis=-1,
    )


def predict_fourier(
    params: FourierParams,
    x: Array,
) -> Array:
    return (
        fourier_features(params.omega, x)
        @ params.amp
    )



def initialize_fourier_params(
    key: Array,
    *,
    input_dimension: int,
    output_dimension: int,
    n_frequencies: int,
    omega_scale: float = 1.0,
) -> FourierParams:
    key_omega, key_amp = jax.random.split(key)

    # Trainable Fourier frequencies are initialized independently
    # from the standard normal distribution.
    omega = omega_scale * jax.random.normal(
        key_omega,
        shape=(
            input_dimension,
            n_frequencies,
        ),
    )

    amp = 1e-2 * jax.random.normal(
        key_amp,
        shape=(
            2 * n_frequencies,
            output_dimension,
        ),
    )

    return FourierParams(
        omega=omega,
        amp=amp,
    )


def initialize_model(
    key: Array,
    *,
    input_dimension: int,
    output_dimension: int,
    n_frequencies: int,
    diff_type: str,
    omega_scale: float = 1.0,
) -> AdamFourierModel:
    key_drift, key_covariance = jax.random.split(key)

    covariance_dimension = (
        covariance_output_dimension(
            output_dimension,
            diff_type,
        )
    )

    drift = initialize_fourier_params(
        key_drift,
        input_dimension=input_dimension,
        output_dimension=output_dimension,
        n_frequencies=n_frequencies,
        omega_scale=omega_scale,
    )

    covariance = initialize_fourier_params(
        key_covariance,
        input_dimension=input_dimension,
        output_dimension=covariance_dimension,
        n_frequencies=n_frequencies,
        omega_scale=omega_scale,
    )

    return AdamFourierModel(
        drift=drift,
        covariance=covariance,
        diff_type=diff_type,
    )


def covariance_factor(
    model: AdamFourierModel,
    x: Array,
) -> Array:
    """
    Return a factor L such that Sigma = L L^T.

    For diagonal covariance, L is diagonal with positive entries.
    For full covariance, the learned outputs populate a lower-triangular
    matrix whose diagonal is made positive by softplus.
    """
    raw = predict_fourier(
        model.covariance,
        x,
    )

    d = model.drift.amp.shape[1]

    if model.diff_type == "diagonal":
        diagonal_variance = (
            jax.nn.softplus(raw)
            + EPS
        )

        diagonal_std = jnp.sqrt(
            diagonal_variance
        )

        return jax.vmap(jnp.diag)(
            diagonal_std
        )

    expected = d * (d + 1) // 2

    if raw.shape[1] != expected:
        raise ValueError(
            "Unexpected covariance-output dimension."
        )

    rows, cols = jnp.tril_indices(d)

    L = jnp.zeros(
        (len(x), d, d),
        dtype=x.dtype,
    )

    L = L.at[:, rows, cols].set(raw)

    diagonal = jnp.diagonal(
        L,
        axis1=-2,
        axis2=-1,
    )

    diagonal = (
        jax.nn.softplus(diagonal)
        + EPS
    )

    idx = jnp.arange(d)

    L = L.at[:, idx, idx].set(
        diagonal
    )

    return L


def predict_covariance(
    model: AdamFourierModel,
    x: Array,
) -> Array:
    L = covariance_factor(model, x)

    return (
        L
        @ jnp.swapaxes(L, -1, -2)
    )


def gaussian_nll(
    model: AdamFourierModel,
    x: Array,
    r: Array,
    h: Array,
) -> Array:
    """
    Mean finite-sample Gaussian negative log-likelihood.
    """
    x = jnp.asarray(x)
    r = jnp.asarray(r)
    h = jnp.asarray(h)

    drift = predict_fourier(
        model.drift,
        x,
    )

    residual = (
        r
        - h * drift
    )

    d = r.shape[1]

    L = covariance_factor(
        model,
        x,
    )

    # Cov(r | x) = h Sigma(x), so a Cholesky factor is sqrt(h) L.
    scale = (
        jnp.sqrt(h)[:, :, None]
        * L
    )

    def solve_one(scale_i, residual_i):
        return jax.scipy.linalg.solve_triangular(
            scale_i,
            residual_i,
            lower=True,
        )

    whitened = jax.vmap(solve_one)(
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
            jnp.log(diagonal),
            axis=1,
        )
    )

    nll = 0.5 * (
        quadratic
        + logdet
        + d * jnp.log(2.0 * jnp.pi)
    )

    return jnp.mean(nll)


def make_optimizer(
    learning_rate: float,
):
    return optax.adam(
        learning_rate,
        b1=0.9,
        b2=0.999,
        eps=1e-7,
    )


def train_step(
    model: AdamFourierModel,
    optimizer,
    opt_state,
    x: Array,
    r: Array,
    h: Array,
):
    """
    Perform one joint Adam update of drift and covariance parameters.
    """
    loss, gradients = jax.value_and_grad(
        gaussian_nll
    )(
        model,
        x,
        r,
        h,
    )

    updates, opt_state = optimizer.update(
        gradients,
        opt_state,
        model,
    )

    model = optax.apply_updates(
        model,
        updates,
    )

    return (
        model,
        opt_state,
        loss,
    )