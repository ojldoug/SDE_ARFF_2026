"""
Adaptive random Fourier feature (ARFF) regression.

This module contains only the generic regression machinery. Experiment
or SDE-specific two-stage logic lives elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


Array = jax.Array


@dataclass(frozen=True)
class ARFFModel:
    omega: Array
    amp: Array


def fourier_features(omega: Array, x: Array) -> Array:
    """Real Fourier feature matrix [cos(X omega), sin(X omega)]."""
    projection = x @ omega
    return jnp.concatenate(
        [jnp.cos(projection), jnp.sin(projection)],
        axis=-1,
    )


def predict(model: ARFFModel, x: Array) -> Array:
    return fourier_features(model.omega, x) @ model.amp


def fit_amplitudes(
    x: Array,
    y: Array,
    omega: Array,
    lambda_reg: float,
) -> Array:
    """
    Ridge-regression amplitudes for fixed Fourier frequencies.
    """
    features = fourier_features(omega, x)

    gram = features.T @ features
    rhs = features.T @ y

    regularized_gram = (
        gram
        + x.shape[0]
        * lambda_reg
        * jnp.eye(features.shape[1], dtype=features.dtype)
    )

    return jnp.linalg.solve(regularized_gram, rhs)


def adaptation_step(
    key: Array,
    model: ARFFModel,
    x: Array,
    y: Array,
    *,
    delta: float,
    lambda_reg: float,
    gamma: float,
    resampling: bool,
    metropolis_test: bool,
):
    """
    Perform one ARFF frequency-adaptation step.
    """
    omega = model.omega
    amp = model.amp

    # One importance weight per Fourier frequency. The cosine and sine
    # amplitudes corresponding to the same omega are combined.
    K = omega.shape[1]
    amp_by_frequency = amp.reshape(2, K, -1)
    amp_norm = jnp.linalg.norm(amp_by_frequency, axis=(0, 2))

    tiny = jnp.finfo(amp_norm.dtype).tiny
    amp_norm_safe = jnp.maximum(amp_norm, tiny)

    if resampling:
        pmf = amp_norm_safe / jnp.sum(amp_norm_safe)

        key, subkey = jax.random.split(key)
        selected = jax.random.choice(
            subkey,
            K,
            shape=(K,),
            replace=True,
            p=pmf,
        )
        omega = omega[:, selected]

        # Refit after resampling before using amplitudes in the
        # Metropolis step.
        amp = fit_amplitudes(x, y, omega, lambda_reg)

        amp_by_frequency = amp.reshape(2, K, -1)
        amp_norm = jnp.linalg.norm(
            amp_by_frequency,
            axis=(0, 2),
        )
        amp_norm_safe = jnp.maximum(amp_norm, tiny)

    key, subkey = jax.random.split(key)
    proposal = omega + delta * jax.random.normal(
        subkey,
        omega.shape,
    )

    if metropolis_test:
        proposal_amp = fit_amplitudes(
            x,
            y,
            proposal,
            lambda_reg,
        )
        proposal_by_frequency = proposal_amp.reshape(
            2,
            K,
            -1,
        )
        proposal_norm = jnp.linalg.norm(
            proposal_by_frequency,
            axis=(0, 2),
        )

        ratio = (
            jnp.maximum(proposal_norm, tiny)
            / amp_norm_safe
        ) ** gamma

        key, subkey = jax.random.split(key)
        accept = ratio >= jax.random.uniform(
            subkey,
            shape=(K,),
        )

        omega = jnp.where(
            accept[None, :],
            proposal,
            omega,
        )
    else:
        omega = proposal

    amp = fit_amplitudes(
        x,
        y,
        omega,
        lambda_reg,
    )

    return key, ARFFModel(omega=omega, amp=amp)


def fit_arff(
    key: Array,
    x: Array,
    y: Array,
    *,
    K: int,
    n_iterations: int,
    lambda_reg: float,
    gamma: float,
    delta: float,
    resampling: bool,
    metropolis_test: bool,
):
    """
    Fit one ARFF regression model.

    No train/validation split is performed here. The caller decides
    which observations constitute the training data.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    if x.ndim != 2:
        raise ValueError("x must have shape (N, d).")

    if y.ndim == 1:
        y = y[:, None]

    if y.ndim != 2:
        raise ValueError("y must have shape (N, q).")

    if len(x) != len(y):
        raise ValueError("x and y must contain the same samples.")

    omega = jnp.zeros(
        (x.shape[1], K),
        dtype=x.dtype,
    )
    amp = fit_amplitudes(
        x,
        y,
        omega,
        lambda_reg,
    )

    model = ARFFModel(
        omega=omega,
        amp=amp,
    )

    for _ in range(n_iterations):
        key, model = adaptation_step(
            key,
            model,
            x,
            y,
            delta=delta,
            lambda_reg=lambda_reg,
            gamma=gamma,
            resampling=resampling,
            metropolis_test=metropolis_test,
        )

    return key, model
