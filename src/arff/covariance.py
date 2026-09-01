"""
Covariance representations and SPD diagnostics for ARFF SDE learning.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from src.arff.regression import ARFFModel, predict


def covariance_targets(
    residual: jax.Array,
    h: jax.Array,
    diff_type: str,
):
    """
    Pointwise covariance targets e e^T / h.

    For diagonal diffusion only diagonal entries are returned.
    Otherwise the lower triangle of the symmetric target is returned.
    """
    residual = jnp.asarray(residual)
    h = jnp.asarray(h)

    if diff_type == "diagonal":
        return residual**2 / h

    full = (
        residual[:, :, None]
        * residual[:, None, :]
        / h[:, :, None]
    )

    i, j = np.tril_indices(residual.shape[1])
    return full[:, i, j]


def raw_covariance(
    model: ARFFModel,
    x: jax.Array,
    diff_type: str,
):
    """
    Reconstruct the raw learned covariance matrix.

    No positivity correction is performed.
    """
    values = predict(model, x)

    if diff_type == "diagonal":
        return jax.vmap(jnp.diag)(values)

    q = values.shape[1]
    d = int((math.sqrt(1 + 8 * q) - 1) / 2)

    if d * (d + 1) // 2 != q:
        raise ValueError(
            f"{q} outputs cannot represent a symmetric matrix."
        )

    i, j = np.tril_indices(d)
    i = jnp.asarray(i)
    j = jnp.asarray(j)

    def reconstruct(v):
        lower = jnp.zeros(
            (d, d),
            dtype=v.dtype,
        ).at[i, j].set(v)

        return (
            lower
            + lower.T
            - jnp.diag(jnp.diag(lower))
        )

    return jax.vmap(reconstruct)(values)


def spd_violation_mask(covariance: jax.Array):
    """True where the raw covariance is not positive definite."""
    eigenvalues = jnp.linalg.eigvalsh(covariance)
    return eigenvalues[:, 0] <= 0.0


def project_spd(
    covariance: jax.Array,
    epsilon: float,
):
    """
    Eigenvalue-floor projection onto Sigma - epsilon I >= 0.
    """
    covariance = 0.5 * (
        covariance
        + jnp.swapaxes(covariance, -1, -2)
    )

    eigenvalues, eigenvectors = jnp.linalg.eigh(covariance)
    eigenvalues = jnp.maximum(eigenvalues, epsilon)

    return (
        eigenvectors
        * eigenvalues[:, None, :]
    ) @ jnp.swapaxes(eigenvectors, -1, -2)


def covariance_factor(
    covariance_spd: jax.Array,
    diff_type: str,
):
    """
    Construct sigma such that sigma sigma^T = covariance_spd.
    """
    if diff_type in {"diagonal", "triangular"}:
        return jax.vmap(jnp.linalg.cholesky)(
            covariance_spd
        )

    eigenvalues, eigenvectors = jnp.linalg.eigh(
        covariance_spd
    )

    return (
        eigenvectors
        * jnp.sqrt(eigenvalues)[:, None, :]
    ) @ jnp.swapaxes(eigenvectors, -1, -2)
