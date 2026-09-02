#!/usr/bin/env python3
"""Smoke test for the Adam Fourier likelihood baseline."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adam.fourier import (
    gaussian_nll,
    initialize_model,
    make_optimizer,
    predict_covariance,
    predict_fourier,
    train_step,
)


def main():
    rng = np.random.default_rng(123)

    n = 256
    d = 2
    K = 32

    x = rng.uniform(
        -1.0,
        1.0,
        size=(n, d),
    ).astype(np.float32)

    true_drift = -x

    h = np.full(
        (n, 1),
        0.01,
        dtype=np.float32,
    )

    noise = rng.normal(
        size=(n, d),
    ).astype(np.float32)

    r = (
        h * true_drift
        + np.sqrt(0.01 * 0.2) * noise
    )

    key = jax.random.PRNGKey(0)

    model = initialize_model(
        key,
        input_dimension=d,
        output_dimension=d,
        n_frequencies=K,
        diff_type="diagonal",
    )

    drift = predict_fourier(
        model.drift,
        jnp.asarray(x),
    )

    covariance = predict_covariance(
        model,
        jnp.asarray(x),
    )

    assert drift.shape == (n, d)
    assert covariance.shape == (n, d, d)

    assert np.all(np.isfinite(drift))
    assert np.all(np.isfinite(covariance))

    eigenvalues = np.linalg.eigvalsh(
        np.asarray(covariance)
    )

    assert np.all(eigenvalues > 0.0)

    loss_before = float(
        gaussian_nll(
            model,
            x,
            r,
            h,
        )
    )

    optimizer = make_optimizer(
        learning_rate=1e-3,
    )

    opt_state = optimizer.init(model)

    # A few full-batch updates are enough to smoke-test differentiation
    # through both drift and covariance models.
    for _ in range(10):
        model, opt_state, loss = train_step(
            model,
            optimizer,
            opt_state,
            x,
            r,
            h,
        )

    loss_after = float(
        gaussian_nll(
            model,
            x,
            r,
            h,
        )
    )

    assert np.isfinite(loss_before)
    assert np.isfinite(loss_after)

    print("Adam Fourier smoke test")
    print("-----------------------")
    print(f"samples       : {n}")
    print(f"frequencies   : {K}")
    print(f"loss before   : {loss_before:.8e}")
    print(f"loss after    : {loss_after:.8e}")
    print(
        "min covariance eigenvalue: "
        f"{eigenvalues.min():.8e}"
    )
    print()
    print("All Adam Fourier checks passed.")

    # Smoke-test the full covariance parameterization too.
    key_full = jax.random.PRNGKey(1)

    full_model = initialize_model(
        key_full,
        input_dimension=2,
        output_dimension=2,
        n_frequencies=16,
        diff_type="symmetric",
    )

    full_covariance = predict_covariance(
        full_model,
        jnp.asarray(x),
    )

    assert full_covariance.shape == (n, 2, 2)

    full_eigenvalues = np.linalg.eigvalsh(
        np.asarray(full_covariance)
    )

    assert np.all(np.isfinite(full_covariance))
    assert np.all(full_eigenvalues > 0.0)

    print(
        "min full-covariance eigenvalue: "
        f"{full_eigenvalues.min():.8e}"
    )


if __name__ == "__main__":
    main()