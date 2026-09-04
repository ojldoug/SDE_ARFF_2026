"""
Canonical simulation utilities for learned SDE models.

Learned covariance matrices are handled using the same canonical ARFF
covariance utilities as likelihood evaluation:

    raw covariance
        -> SPD eigenvalue-floor projection
        -> covariance factor

The Euler--Maruyama update is

    X_{n+1}
      = X_n
      + h f(X_n)
      + sqrt(h) sigma_epsilon(X_n) xi_n,

where

    sigma_epsilon sigma_epsilon^T
      = Pi_epsilon(Sigma(X_n)).

This is particularly important for ARFF, whose unconstrained covariance
regression may produce indefinite matrices at some states.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

from src.arff.covariance import (
    covariance_factor,
    project_spd,
)


Array = jax.Array


def euler_maruyama(
    key: Array,
    x0: Array,
    *,
    drift_fn: Callable[[Array], Array],
    covariance_fn: Callable[[Array], Array],
    diff_type: str,
    step_size: float,
    n_steps: int,
    spd_epsilon: float,
) -> Array:
    """
    Simulate parallel learned-SDE trajectories with Euler--Maruyama.

    Parameters
    ----------
    key:
        JAX PRNG key.

    x0:
        Initial states with shape (N, D), or one state with shape (D,).

    drift_fn:
        Maps states with shape (N, D) to drift values with shape
        (N, D).

    covariance_fn:
        Maps states with shape (N, D) to raw covariance matrices with
        shape (N, D, D).

    diff_type:
        Covariance representation used by the learned model.

    step_size:
        Euler--Maruyama time step.

    n_steps:
        Number of Euler updates.

    spd_epsilon:
        Eigenvalue floor applied before covariance factorization.

    Returns
    -------
    trajectories:
        Array with shape (n_steps + 1, N, D), or (n_steps + 1, D)
        when a single initial state is supplied.
    """
    if step_size <= 0.0:
        raise ValueError(
            "step_size must be positive."
        )

    if n_steps < 0:
        raise ValueError(
            "n_steps must be non-negative."
        )

    if spd_epsilon <= 0.0:
        raise ValueError(
            "spd_epsilon must be positive."
        )

    x0 = jnp.asarray(
        x0
    )

    squeeze_trajectory = False

    if x0.ndim == 1:
        x0 = x0[None, :]
        squeeze_trajectory = True

    if x0.ndim != 2:
        raise ValueError(
            "x0 must have shape (D,) "
            "or (N, D)."
        )

    n_trajectories = (
        x0.shape[0]
    )

    dimension = (
        x0.shape[1]
    )

    noise = jax.random.normal(
        key,
        shape=(
            n_steps,
            n_trajectories,
            dimension,
        ),
        dtype=x0.dtype,
    )

    sqrt_step_size = jnp.sqrt(
        jnp.asarray(
            step_size,
            dtype=x0.dtype,
        )
    )

    def step(
        x,
        xi,
    ):
        drift = jnp.asarray(
            drift_fn(
                x
            )
        )

        covariance_raw = jnp.asarray(
            covariance_fn(
                x
            )
        )

        if drift.shape != x.shape:
            raise ValueError(
                "drift_fn returned an "
                "unexpected shape."
            )

        expected_covariance_shape = (
            n_trajectories,
            dimension,
            dimension,
        )

        if (
            covariance_raw.shape
            != expected_covariance_shape
        ):
            raise ValueError(
                "covariance_fn returned "
                "an unexpected shape."
            )

        covariance_spd = project_spd(
            covariance_raw,
            epsilon=spd_epsilon,
        )

        diffusion = covariance_factor(
            covariance_spd,
            diff_type,
        )

        stochastic_increment = (
            sqrt_step_size
            * jnp.einsum(
                "nij,nj->ni",
                diffusion,
                xi,
            )
        )

        x_next = (
            x
            + step_size * drift
            + stochastic_increment
        )

        return (
            x_next,
            x_next,
        )

    _, simulated = jax.lax.scan(
        step,
        x0,
        noise,
    )

    trajectories = jnp.concatenate(
        (
            x0[None, ...],
            simulated,
        ),
        axis=0,
    )

    if squeeze_trajectory:
        trajectories = (
            trajectories[:, 0, :]
        )

    return trajectories