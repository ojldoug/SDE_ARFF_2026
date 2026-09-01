"""
Evaluation utilities for two-stage ARFF SDE models.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from src.arff.covariance import (
    project_spd,
    raw_covariance,
    spd_violation_mask,
)
from src.arff.regression import predict
from src.arff.two_stage import TwoStageARFFModel


@dataclass(frozen=True)
class EvaluationResult:
    nll: float
    spd_violation_rate: float
    min_raw_eigenvalue: float
    min_projected_eigenvalue: float


def gaussian_nll(
    model: TwoStageARFFModel,
    x,
    r,
    h,
    *,
    spd_epsilon: float,
):
    """
    Mean Gaussian Euler--Maruyama negative log-likelihood.

    The learned raw covariance is projected to

        Sigma - epsilon I >= 0

    before inversion and log-determinant evaluation.
    """
    x = jnp.asarray(x)
    r = jnp.asarray(r)
    h = jnp.asarray(h)

    drift = predict(model.drift, x)

    covariance_raw = raw_covariance(
        model.covariance,
        x,
        model.diff_type,
    )

    violations = spd_violation_mask(covariance_raw)

    covariance = project_spd(
        covariance_raw,
        epsilon=spd_epsilon,
    )

    residual = r - h * drift
    variance = covariance * h[:, :, None]

    sign, logdet = jnp.linalg.slogdet(variance)

    if not bool(jnp.all(sign > 0)):
        raise RuntimeError(
            "Projected covariance produced a non-positive determinant."
        )

    solution = jnp.linalg.solve(
        variance,
        residual[:, :, None],
    )[:, :, 0]

    quadratic = jnp.sum(
        residual * solution,
        axis=1,
    )

    dimension = r.shape[1]

    losses = 0.5 * (
        quadratic
        + logdet
        + dimension * jnp.log(2.0 * jnp.pi)
    )

    raw_eigenvalues = jnp.linalg.eigvalsh(
        covariance_raw
    )
    projected_eigenvalues = jnp.linalg.eigvalsh(
        covariance
    )

    return EvaluationResult(
        nll=float(jnp.mean(losses)),
        spd_violation_rate=float(jnp.mean(violations)),
        min_raw_eigenvalue=float(
            jnp.min(raw_eigenvalues)
        ),
        min_projected_eigenvalue=float(
            jnp.min(projected_eigenvalues)
        ),
    )
