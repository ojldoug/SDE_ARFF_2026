"""
Euler--Maruyama data generation for the reproducible experiments.

For the standard SDE experiments, each training observation is generated
by starting from an independently sampled initial state, evolving the SDE
with a fine Euler--Maruyama step, and retaining only the endpoint pair

    (X_0, X_h),

where h is the prescribed observation lag.

Intermediate fine-grid states are not stored.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from src.experiments.config import ExperimentConfig
from src.experiments.definitions import ExperimentDefinition


def _validate_standard_config(config: ExperimentConfig) -> int:
    data = config.data

    if data.n_trajectories is None:
        raise ValueError("n_trajectories must be specified.")

    if data.fine_step is None:
        raise ValueError("fine_step must be specified.")

    if data.observation_lag is None:
        raise ValueError("observation_lag must be specified.")

    ratio = data.observation_lag / data.fine_step
    n_fine_steps = int(round(ratio))

    if not np.isclose(
        n_fine_steps * data.fine_step,
        data.observation_lag,
    ):
        raise ValueError(
            "observation_lag must be an integer multiple of fine_step."
        )

    if n_fine_steps <= 0:
        raise ValueError("Number of fine EM steps must be positive.")

    return n_fine_steps


def generate_standard_em_data(
    definition: ExperimentDefinition,
    config: ExperimentConfig,
):
    """
    Generate independent endpoint pairs for a standard first-order SDE.

    Returns
    -------
    x_data
        Initial states, shape (N, state_dimension).
    r_data
        Endpoint increments X_h - X_0, shape (N, n_dimensions).
    step_sizes
        Observation lags, shape (N, 1).
    """
    n_fine_steps = _validate_standard_config(config)
    data = config.data

    if definition.xlim is None:
        raise ValueError(
            f"{definition.name} has no rectangular xlim for standard EM."
        )

    if definition.state_dimension != definition.n_dimensions:
        raise ValueError(
            "Standard EM generator requires state_dimension == "
            "n_dimensions. Use a coupled generator otherwise."
        )

    n = data.n_trajectories
    d = definition.n_dimensions
    h_fine = data.fine_step
    h_obs = data.observation_lag

    key = jax.random.PRNGKey(data.seed)
    key_x0, key_noise = jax.random.split(key)

    xlim = jnp.asarray(definition.xlim)

    x0 = jax.random.uniform(
        key_x0,
        shape=(n, d),
        minval=xlim[:, 0],
        maxval=xlim[:, 1],
    )

    def em_step(x, noise):
        drift = definition.drift(x)
        sigma = definition.diffusion_factor(x)

        dW = jnp.sqrt(h_fine) * noise
        diffusion_increment = jnp.einsum(
            "nij,nj->ni",
            sigma,
            dW,
        )

        x_next = x + h_fine * drift + diffusion_increment
        return x_next, None

    # Generate one noise array for each fine step. For the largest current
    # experiment this is manageable on the available GPU, while avoiding
    # storage of the full state trajectory.
    noise = jax.random.normal(
        key_noise,
        shape=(n_fine_steps, n, d),
    )

    x_final, _ = jax.lax.scan(em_step, x0, noise)

    r_data = x_final - x0
    step_sizes = jnp.full((n, 1), h_obs)

    return (
        np.asarray(x0),
        np.asarray(r_data),
        np.asarray(step_sizes),
    )
