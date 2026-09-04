#!/usr/bin/env python3
"""Audit mathematical consistency of Experiment 5."""

from __future__ import annotations

from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


from src.experiments.definitions import (
    ex5_diffusion_factor,
    ex5_drift,
)


POPULATION_SIZE = 1024


def reaction_drift_and_covariance(x):
    """
    Drift and covariance derived directly from the Experiment 5
    reaction system in normalized (S, R) coordinates.

    Active reactions:

        S + I -> 2 I
        I     -> R

    with I = 1 - S - R and count-space propensities

        a1 = N * 4 S I,
        a2 = N * I.

    The normalized jump vectors are

        nu1 / N = (-1, 0) / N,
        nu2 / N = ( 0, 1) / N.
    """
    x = np.asarray(
        x,
        dtype=float,
    )

    S = x[:, 0]
    R = x[:, 1]
    I = 1.0 - S - R

    if np.any(I < -1e-12):
        raise ValueError(
            "Test states leave the SIR simplex."
        )

    I = np.maximum(
        I,
        0.0,
    )

    a1 = (
        POPULATION_SIZE
        * 4.0
        * S
        * I
    )

    a2 = (
        POPULATION_SIZE
        * I
    )

    nu1 = np.array(
        [-1.0, 0.0]
    ) / POPULATION_SIZE

    nu2 = np.array(
        [0.0, 1.0]
    ) / POPULATION_SIZE

    drift = (
        a1[:, None] * nu1
        + a2[:, None] * nu2
    )

    covariance = (
        a1[:, None, None]
        * np.outer(
            nu1,
            nu1,
        )[None, :, :]
        + a2[:, None, None]
        * np.outer(
            nu2,
            nu2,
        )[None, :, :]
    )

    return (
        drift,
        covariance,
    )


def main():
    # Interior, boundary and near-extinction states.
    x = np.asarray(
        [
            [0.70, 0.10],
            [0.20, 0.30],
            [0.01, 0.49],
            [0.90, 0.09],
            [0.80, 0.20],
        ],
        dtype=float,
    )

    (
        reaction_drift,
        reaction_covariance,
    ) = reaction_drift_and_covariance(
        x
    )

    implemented_drift = np.asarray(
        ex5_drift(
            jnp.asarray(x)
        )
    )

    sigma = np.asarray(
        ex5_diffusion_factor(
            jnp.asarray(x)
        )
    )

    implemented_covariance = (
        sigma
        @ np.swapaxes(
            sigma,
            -1,
            -2,
        )
    )

    drift_error = np.max(
        np.abs(
            implemented_drift
            - reaction_drift
        )
    )

    covariance_error = np.max(
        np.abs(
            implemented_covariance
            - reaction_covariance
        )
    )

    print(
        "Experiment 5 SIR consistency audit"
    )
    print(
        "=================================="
    )
    print(
        "population size        :",
        POPULATION_SIZE,
    )
    print(
        "maximum drift error    :",
        f"{drift_error:.12e}",
    )
    print(
        "maximum covariance error:",
        f"{covariance_error:.12e}",
    )

    assert np.allclose(
        implemented_drift,
        reaction_drift,
        rtol=1e-6,
        atol=1e-7,
    )

    assert np.allclose(
        implemented_covariance,
        reaction_covariance,
        rtol=1e-6,
        atol=1e-9,
    )

    # With k3 = 0:
    #
    #   dS <= 0,
    #   dR >= 0.
    assert np.all(
        implemented_drift[:, 0]
        <= 1e-12
    )

    assert np.all(
        implemented_drift[:, 1]
        >= -1e-12
    )

    # The transformed (S,R) covariance must be diagonal.
    assert np.allclose(
        implemented_covariance[
            :, 0, 1
        ],
        0.0,
        atol=1e-12,
    )

    assert np.allclose(
        implemented_covariance[
            :, 1, 0
        ],
        0.0,
        atol=1e-12,
    )

    # At extinction I = 0, both drift and covariance vanish.
    extinct = np.asarray(
        [[0.8, 0.2]],
        dtype=float,
    )

    extinct_drift = np.asarray(
        ex5_drift(
            jnp.asarray(
                extinct
            )
        )
    )

    extinct_sigma = np.asarray(
        ex5_diffusion_factor(
            jnp.asarray(
                extinct
            )
        )
    )

    extinct_covariance = (
        extinct_sigma
        @ np.swapaxes(
            extinct_sigma,
            -1,
            -2,
        )
    )

    assert np.allclose(
        extinct_drift,
        0.0,
        atol=1e-7,
    )

    assert np.allclose(
        extinct_covariance,
        0.0,
        atol=1e-9,
    )

    print(
        "extinction drift       :",
        extinct_drift[0],
    )

    print(
        "extinction covariance  :"
    )
    print(
        extinct_covariance[0]
    )

    print()
    print(
        "Experiment 5 consistency audit passed."
    )


if __name__ == "__main__":
    main()