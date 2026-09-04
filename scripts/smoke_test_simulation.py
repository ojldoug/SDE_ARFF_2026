#!/usr/bin/env python3
"""Smoke tests for canonical learned-SDE simulation."""

from __future__ import annotations

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from src.arff.covariance import (
    covariance_factor,
    project_spd,
)
from src.experiments.simulation import (
    euler_maruyama,
)


def main():
    epsilon = 1e-2

    # ------------------------------------------------------------
    # 1. Deliberately indefinite covariance.
    #
    # Verify that the canonical projection/factorization pair:
    #
    #   * floors eigenvalues at epsilon;
    #   * produces sigma such that
    #         sigma sigma^T = projected covariance.
    # ------------------------------------------------------------

    raw = jnp.asarray(
        [
            [
                [2.0, 0.5],
                [0.5, -0.2],
            ],
            [
                [1.0, 0.3],
                [0.3, 0.5],
            ],
        ],
        dtype=jnp.float32,
    )

    projected = project_spd(
        raw,
        epsilon=epsilon,
    )

    factor = covariance_factor(
        projected,
        "symmetric",
    )

    reconstructed = (
        factor
        @ jnp.swapaxes(
            factor,
            -1,
            -2,
        )
    )

    projected_np = np.asarray(
        projected
    )

    reconstructed_np = np.asarray(
        reconstructed
    )

    eigenvalues = np.linalg.eigvalsh(
        projected_np
    )

    factorization_error = np.max(
        np.abs(
            reconstructed_np
            - projected_np
        )
    )

    print(
        "minimum projected eigenvalue:",
        eigenvalues.min(),
    )

    print(
        "maximum factorization error:",
        factorization_error,
    )

    assert (
        eigenvalues.min()
        >= epsilon - 1e-6
    )

    assert np.allclose(
        reconstructed_np,
        projected_np,
        rtol=1e-5,
        atol=1e-6,
    )

    # ------------------------------------------------------------
    # 2. Check the Cholesky path as well.
    # ------------------------------------------------------------

    factor_cholesky = covariance_factor(
        projected,
        "triangular",
    )

    reconstructed_cholesky = (
        factor_cholesky
        @ jnp.swapaxes(
            factor_cholesky,
            -1,
            -2,
        )
    )

    cholesky_error = np.max(
        np.abs(
            np.asarray(
                reconstructed_cholesky
            )
            - projected_np
        )
    )

    print(
        "maximum Cholesky error:",
        cholesky_error,
    )

    assert np.allclose(
        np.asarray(
            reconstructed_cholesky
        ),
        projected_np,
        rtol=1e-5,
        atol=1e-6,
    )

    # ------------------------------------------------------------
    # 3. Zero-drift, identity-covariance simulation.
    #
    # Verify:
    #
    #   * expected trajectory shape;
    #   * finite values;
    #   * deterministic reproducibility for a fixed PRNG key.
    # ------------------------------------------------------------

    x0 = jnp.zeros(
        (4, 2),
        dtype=jnp.float32,
    )

    def drift_fn(x):
        return jnp.zeros_like(
            x
        )

    def covariance_fn(x):
        identity = jnp.eye(
            x.shape[-1],
            dtype=x.dtype,
        )

        return jnp.broadcast_to(
            identity,
            (
                x.shape[0],
                x.shape[-1],
                x.shape[-1],
            ),
        )

    key = jax.random.PRNGKey(
        123
    )

    trajectories_1 = euler_maruyama(
        key,
        x0,
        drift_fn=drift_fn,
        covariance_fn=covariance_fn,
        diff_type="symmetric",
        step_size=1e-3,
        n_steps=10,
        spd_epsilon=1e-6,
    )

    trajectories_2 = euler_maruyama(
        key,
        x0,
        drift_fn=drift_fn,
        covariance_fn=covariance_fn,
        diff_type="symmetric",
        step_size=1e-3,
        n_steps=10,
        spd_epsilon=1e-6,
    )

    trajectories_1_np = np.asarray(
        trajectories_1
    )

    trajectories_2_np = np.asarray(
        trajectories_2
    )

    print(
        "trajectory shape:",
        trajectories_1.shape,
    )

    print(
        "finite:",
        np.all(
            np.isfinite(
                trajectories_1_np
            )
        ),
    )

    print(
        "fixed-key reproducible:",
        np.array_equal(
            trajectories_1_np,
            trajectories_2_np,
        ),
    )

    assert (
        trajectories_1.shape
        == (11, 4, 2)
    )

    assert np.all(
        np.isfinite(
            trajectories_1_np
        )
    )

    assert np.array_equal(
        trajectories_1_np,
        trajectories_2_np,
    )

    print()
    print(
        "Simulation smoke tests passed."
    )


if __name__ == "__main__":
    main()