"""
MLP Gaussian-likelihood baseline trained with Adam.

This baseline provides a conventional neural-network comparison to the
Fourier models. Drift and covariance-factor parameters are learned
jointly by minimizing the same finite-sample Gaussian negative
log-likelihood used by the Adam Fourier baseline.

The default architecture uses two tanh hidden layers of width 50.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import jax
import jax.numpy as jnp

from src.experiments.model_size import (
    covariance_output_dimension,
)


Array = jax.Array
EPS = 1e-8


@dataclass(frozen=True)
class MLPParams:
    weights: Tuple[Array, ...]
    biases: Tuple[Array, ...]

    def tree_flatten(self):
        return (
            self.weights,
            self.biases,
        ), None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data,
        children,
    ):
        weights, biases = children

        return cls(
            weights=weights,
            biases=biases,
        )


jax.tree_util.register_pytree_node_class(
    MLPParams
)


@dataclass(frozen=True)
class AdamMLPModel:
    drift: MLPParams
    covariance: MLPParams
    diff_type: str

    def tree_flatten(self):
        children = (
            self.drift,
            self.covariance,
        )

        return children, self.diff_type

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
    AdamMLPModel
)


def initialize_mlp_params(
    key: Array,
    *,
    input_dimension: int,
    output_dimension: int,
    hidden_sizes: Tuple[int, ...],
    output_scale: float = 1.0,
) -> MLPParams:
    """
    Initialize a fully connected tanh network with Glorot-uniform
    weights and zero biases.
    """
    layer_sizes = (
        input_dimension,
        *hidden_sizes,
        output_dimension,
    )

    keys = jax.random.split(
        key,
        len(layer_sizes) - 1,
    )

    weights = []
    biases = []

    for (
        layer_key,
        fan_in,
        fan_out,
    ) in zip(
        keys,
        layer_sizes[:-1],
        layer_sizes[1:],
    ):
        limit = jnp.sqrt(
            6.0
            / (
                fan_in
                + fan_out
            )
        )

        weight = jax.random.uniform(
            layer_key,
            shape=(
                fan_in,
                fan_out,
            ),
            minval=-limit,
            maxval=limit,
        )

        weights.append(
            weight
        )

        biases.append(
            jnp.zeros(
                (fan_out,),
                dtype=weight.dtype,
            )
        )

    if output_scale != 1.0:
        weights[-1] = (
            output_scale
            * weights[-1]
        )

    return MLPParams(
        weights=tuple(
            weights
        ),
        biases=tuple(
            biases
        ),
    )


def predict_mlp(
    params: MLPParams,
    x: Array,
) -> Array:
    """
    Evaluate the tanh MLP.

    Hidden layers use tanh; the output layer is linear.
    """
    z = jnp.asarray(
        x
    )

    n_layers = len(
        params.weights
    )

    for i in range(
        n_layers - 1
    ):
        z = jnp.tanh(
            z
            @ params.weights[i]
            + params.biases[i]
        )

    return (
        z
        @ params.weights[-1]
        + params.biases[-1]
    )


def initialize_model(
    key: Array,
    *,
    input_dimension: int,
    output_dimension: int,
    diff_type: str,
    hidden_sizes: Tuple[int, ...] = (
        50,
        50,
    ),
) -> AdamMLPModel:
    """
    Initialize separate drift and covariance MLPs.
    """
    (
        key_drift,
        key_covariance,
    ) = jax.random.split(
        key
    )

    covariance_dimension = (
        covariance_output_dimension(
            output_dimension,
            diff_type,
        )
    )

    drift = initialize_mlp_params(
        key_drift,
        input_dimension=(
            input_dimension
        ),
        output_dimension=(
            output_dimension
        ),
        hidden_sizes=(
            hidden_sizes
        ),
    )

    # A smaller final-layer initialization for the covariance network
    # mirrors the conservative initialization used by the legacy MLP
    # baseline and avoids extreme initial covariance factors.
    covariance = initialize_mlp_params(
        key_covariance,
        input_dimension=(
            input_dimension
        ),
        output_dimension=(
            covariance_dimension
        ),
        hidden_sizes=(
            hidden_sizes
        ),
        output_scale=1e-2,
    )

    return AdamMLPModel(
        drift=drift,
        covariance=covariance,
        diff_type=diff_type,
    )


def covariance_factor(
    model: AdamMLPModel,
    x: Array,
) -> Array:
    """
    Return L such that Sigma = L L^T.

    Diagonal models predict positive variances through softplus.

    Full models predict a lower-triangular factor whose diagonal is
    made strictly positive through softplus.
    """
    raw = predict_mlp(
        model.covariance,
        x,
    )

    d = (
        model.drift.biases[-1].shape[0]
    )

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

    rows, cols = (
        jnp.tril_indices(
            d
        )
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


def predict_covariance(
    model: AdamMLPModel,
    x: Array,
) -> Array:
    L = covariance_factor(
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


def gaussian_nll(
    model: AdamMLPModel,
    x: Array,
    r: Array,
    h: Array,
) -> Array:
    """
    Mean finite-sample Gaussian negative log-likelihood.
    """
    x = jnp.asarray(
        x
    )

    r = jnp.asarray(
        r
    )

    h = jnp.asarray(
        h
    )

    drift = predict_mlp(
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

    scale = (
        jnp.sqrt(
            h
        )[:, :, None]
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