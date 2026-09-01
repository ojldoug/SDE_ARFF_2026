#!/usr/bin/env python3
"""Smoke tests for the clean ARFF regression/covariance core."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.arff.regression import fit_arff, predict
from src.arff.covariance import (
    covariance_targets,
    project_spd,
    raw_covariance,
    spd_violation_mask,
)


def main():
    # ---------------------------------------------------------------
    # Regression test on a function exactly representable by low
    # Fourier frequencies.
    # ---------------------------------------------------------------
    x = jnp.linspace(-1.0, 1.0, 256)[:, None]

    y = (
        0.7 * jnp.cos(2.0 * x)
        - 0.3 * jnp.sin(3.0 * x)
    )

    key = jax.random.PRNGKey(123)

    key, model = fit_arff(
        key,
        x,
        y,
        K=32,
        n_iterations=20,
        lambda_reg=1e-6,
        gamma=1.0,
        delta=0.2,
        resampling=False,
        metropolis_test=True,
    )

    prediction = predict(model, x)
    mse = float(jnp.mean((prediction - y) ** 2))

    assert np.isfinite(mse)
    assert prediction.shape == y.shape

    print("ARFF regression")
    print(f"  prediction shape : {prediction.shape}")
    print(f"  MSE              : {mse:.8e}")

    # ---------------------------------------------------------------
    # Covariance-target test.
    # ---------------------------------------------------------------
    residual = jnp.array(
        [
            [1.0, 2.0],
            [2.0, -1.0],
        ]
    )
    h = jnp.array([[0.5], [0.25]])

    target = covariance_targets(
        residual,
        h,
        "symmetric",
    )

    assert target.shape == (2, 3)

    print()
    print("Covariance targets")
    print(f"  shape            : {target.shape}")

    # ---------------------------------------------------------------
    # SPD projection test.
    # ---------------------------------------------------------------
    raw = jnp.array(
        [
            [[1.0, 2.0], [2.0, 1.0]],
            [[2.0, 0.0], [0.0, 3.0]],
        ]
    )

    violations = spd_violation_mask(raw)
    projected = project_spd(raw, epsilon=1e-6)
    projected_eigs = jnp.linalg.eigvalsh(projected)

    assert bool(violations[0])
    assert not bool(violations[1])
    projected_eigs_np = np.asarray(projected_eigs)

    # In float32 the reconstructed smallest eigenvalue can differ slightly
    # from the requested floor because of eigendecomposition/reconstruction
    # roundoff. It must nevertheless be numerically positive.
    assert np.all(projected_eigs_np > 0.0)

    # An already-SPD covariance should be unchanged.
    assert np.allclose(
        np.asarray(projected[1]),
        np.asarray(raw[1]),
        rtol=1e-6,
        atol=1e-7,
    )

    assert np.isclose(
        projected_eigs_np[0, 0],
        1e-6,
        rtol=0.1,
        atol=1e-7,
    )

    print()
    print("SPD projection")
    print(f"  violations       : {np.asarray(violations)}")
    print(
        "  projected min eig:",
        np.asarray(projected_eigs[:, 0]),
    )

    print()
    print("All ARFF core smoke tests passed.")


if __name__ == "__main__":
    main()
