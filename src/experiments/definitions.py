"""
Authoritative mathematical definitions for Experiments 1--8.

All functions return the diffusion factor sigma(x), not the covariance,
unless explicitly stated otherwise. The covariance is always

    Sigma(x) = sigma(x) sigma(x)^T.

The definitions are migrated from the legacy true_functions notebook.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import jax.numpy as jnp
import numpy as np


Array = jnp.ndarray


@dataclass(frozen=True)
class ExperimentDefinition:
    name: str
    n_dimensions: int
    state_dimension: int
    diff_type: str
    xlim: Optional[np.ndarray]
    drift: Callable[[Array], Array]
    diffusion_factor: Callable[[Array], Array]
    coupled_fn: Optional[Callable] = None

    def covariance(self, x: Array) -> Array:
        sigma = self.diffusion_factor(x)
        return sigma @ jnp.swapaxes(sigma, -1, -2)


# ---------------------------------------------------------------------
# Experiment 1: 2D cubic
# ---------------------------------------------------------------------

def ex1_drift(x):
    x = jnp.atleast_2d(x)
    return -1.5 + 8.0 * x - 16.0 * x**3


def ex1_diffusion_factor(x):
    x = jnp.atleast_2d(x)
    s11 = 0.5 + 0.1 * x[:, 0]
    s22 = 0.5 + 0.1 * x[:, 1]
    zeros = jnp.zeros_like(s11)

    return jnp.stack(
        [
            jnp.stack([s11, zeros], axis=1),
            jnp.stack([zeros, s22], axis=1),
        ],
        axis=1,
    )


# ---------------------------------------------------------------------
# Experiment 2: 3D linear drift with constant triangular diffusion
# ---------------------------------------------------------------------

_EX2_SIGMA = jnp.array(
    [
        [0.09506174, 0.0,        0.0],
        [0.04639143, 0.15817465, 0.0],
        [0.04337843, 0.07506578, 0.00852886],
    ],
    dtype=jnp.float32,
)


def ex2_drift(x):
    x = jnp.atleast_2d(x)
    return -x


def ex2_diffusion_factor(x):
    x = jnp.atleast_2d(x)
    return jnp.broadcast_to(_EX2_SIGMA, (x.shape[0], 3, 3))


# ---------------------------------------------------------------------
# Experiment 3: 10D cubic drift with constant symmetric diffusion
# ---------------------------------------------------------------------

_EX3_SIGMA = jnp.array(
    [
        [ 0.22832221,  0.01086492,  0.09000350, -0.06087793,  0.01299195, -0.00934674, -0.02900455,  0.05489025, -0.09580944, -0.00783704],
        [ 0.01086492,  0.13756094,  0.00381470,  0.01211626, -0.00305770,  0.01440211,  0.00520558,  0.00759839,  0.04953970, -0.02442084],
        [ 0.09000350,  0.00381470,  0.27686390, -0.07324967, -0.05737787,  0.07795476, -0.06766098,  0.01155383, -0.12393842,  0.03276011],
        [-0.06087793,  0.01211626, -0.07324967,  0.27916460,  0.02096533,  0.01732130,  0.05402171, -0.07720050,  0.07261467, -0.06355727],
        [ 0.01299195, -0.00305770, -0.05737787,  0.02096533,  0.22741540, -0.05719331,  0.00331460, -0.01310485,  0.02251231, -0.00977777],
        [-0.00934674,  0.01440211,  0.07795476,  0.01732130, -0.05719331,  0.21695731, -0.02379200, -0.00335828, -0.01139305, -0.01274912],
        [-0.02900455,  0.00520558, -0.06766098,  0.05402171,  0.00331460, -0.02379200,  0.14159400, -0.00478786,  0.05316278, -0.02923832],
        [ 0.05489025,  0.00759839,  0.01155383, -0.07720050, -0.01310485, -0.00335828, -0.00478786,  0.20364688, -0.02160159,  0.00850617],
        [-0.09580944,  0.04953970, -0.12393842,  0.07261467,  0.02251231, -0.01139305,  0.05316278, -0.02160159,  0.29428910, -0.04313131],
        [-0.00783704, -0.02442084,  0.03276011, -0.06355727, -0.00977777, -0.01274912, -0.02923832,  0.00850617, -0.04313131,  0.19487537],
    ],
    dtype=jnp.float32,
)


def ex3_drift(x):
    x = jnp.atleast_2d(x)
    return -(32.0 * x**3 - 16.0 * x + 3.0) / 2.0


def ex3_diffusion_factor(x):
    x = jnp.atleast_2d(x)
    return jnp.broadcast_to(_EX3_SIGMA, (x.shape[0], 10, 10))


# ---------------------------------------------------------------------
# Experiment 4: underdamped Langevin
# Input coordinates are (v, x); learned output is dv.
# ---------------------------------------------------------------------

def ex4_drift(z):
    z = jnp.atleast_2d(z)
    v = z[:, 0]
    x = z[:, 1]
    return (-x**3 - x - 0.5 * v)[:, None]


def ex4_diffusion_factor(z):
    z = jnp.atleast_2d(z)
    value = jnp.sqrt(0.1)
    return jnp.ones((z.shape[0], 1, 1)) * value


def ex4_coupled_fn(y0, x0, h):
    y0_flat = jnp.ravel(y0)
    return x0.at[:, 1].add(y0_flat * h)


# ---------------------------------------------------------------------
# Experiment 5: SIR mean-field diffusion approximation
# Coordinates are (theta_0, theta_2) = (S, R).
# ---------------------------------------------------------------------

def ex5_drift(x):
    x = jnp.atleast_2d(x)

    theta0 = jnp.clip(x[:, 0], 0.0, 1.0)
    theta2 = jnp.clip(x[:, 1], 0.0, 1.0 - theta0)
    theta1 = jnp.clip(1.0 - theta0 - theta2, 0.0, 1.0)

    r1 = 4.0 * theta0 * theta1
    r2 = theta1

    return jnp.stack([-r1, r2], axis=1)


def ex5_diffusion_factor(x):
    x = jnp.atleast_2d(x)

    population_size = 1024

    theta0 = jnp.clip(x[:, 0], 0.0, 1.0)
    theta2 = jnp.clip(x[:, 1], 0.0, 1.0 - theta0)
    theta1 = jnp.clip(1.0 - theta0 - theta2, 0.0, 1.0)

    r1 = 4.0 * theta0 * theta1
    r2 = theta1

    s11 = jnp.sqrt(r1 / population_size)
    s22 = jnp.sqrt(r2 / population_size)
    zeros = jnp.zeros_like(s11)

    return jnp.stack(
        [
            jnp.stack([s11, zeros], axis=1),
            jnp.stack([zeros, s22], axis=1),
        ],
        axis=1,
    )


# ---------------------------------------------------------------------
# Experiment 6: stochastic wave effective forcing
# Input coordinates are (t, x).
#
# IMPORTANT:
# The factor 1/2 inherited by the effective diffusion is retained here
# provisionally and will be checked against the discretization derivation
# before final experiments are frozen.
# ---------------------------------------------------------------------

def ex6_drift(x):
    x = jnp.atleast_2d(x)
    space = x[:, 1]
    return (5.0 * jnp.sin(4.0 * jnp.pi * space))[:, None]


def ex6_diffusion_factor(x):
    x = jnp.atleast_2d(x)
    space = x[:, 1]

    g = 0.05 * (1.0 + jnp.exp(-150.0 * (space - 0.5) ** 2))
    effective_sigma = 0.5 * g

    return effective_sigma[:, None, None]


# ---------------------------------------------------------------------
# Experiment 7: localized broad-spectrum drift
# ---------------------------------------------------------------------

def ex7_drift(x):
    x = jnp.atleast_2d(x)

    x1 = x[:, 0]
    x2 = x[:, 1]

    f1 = jnp.exp(-jnp.abs(x1) / 0.1) * jnp.exp(
        -0.5 * (x1**2 + x2**2)
    )
    f2 = jnp.exp(-jnp.abs(x2) / 0.2) * jnp.exp(
        -0.5 * (x1**2 + x2**2)
    )

    return jnp.stack([f1, f2], axis=1)


def ex7_diffusion_factor(x):
    x = jnp.atleast_2d(x)
    return 0.1 * jnp.broadcast_to(jnp.eye(2), (x.shape[0], 2, 2))


# ---------------------------------------------------------------------
# Experiment 8: near-singular rotated diffusion
# ---------------------------------------------------------------------

def ex8_drift(x):
    x = jnp.atleast_2d(x)
    return -x


def ex8_diffusion_factor(x):
    x = jnp.atleast_2d(x)

    eps = 1.0e-4
    k = 3.0

    theta = k * jnp.atan2(x[:, 1], x[:, 0])

    c = jnp.cos(theta)
    s = jnp.sin(theta)

    rotation = jnp.stack(
        [
            jnp.stack([c, -s], axis=1),
            jnp.stack([s,  c], axis=1),
        ],
        axis=1,
    )

    diagonal = jnp.zeros((x.shape[0], 2, 2))
    diagonal = diagonal.at[:, 0, 0].set(eps)
    diagonal = diagonal.at[:, 1, 1].set(1.0)

    return rotation @ diagonal @ jnp.swapaxes(rotation, 1, 2)


# ---------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------

EXPERIMENTS = {
    "ex1": ExperimentDefinition(
        name="ex1",
        n_dimensions=2,
        state_dimension=2,
        diff_type="diagonal",
        xlim=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        drift=ex1_drift,
        diffusion_factor=ex1_diffusion_factor,
    ),
    "ex2": ExperimentDefinition(
        name="ex2",
        n_dimensions=3,
        state_dimension=3,
        diff_type="triangular",
        xlim=np.array([[-1.0, 1.0]] * 3),
        drift=ex2_drift,
        diffusion_factor=ex2_diffusion_factor,
    ),
    "ex3": ExperimentDefinition(
        name="ex3",
        n_dimensions=10,
        state_dimension=10,
        diff_type="symmetric",
        xlim=np.array([[-1.0, 1.0]] * 10),
        drift=ex3_drift,
        diffusion_factor=ex3_diffusion_factor,
    ),
    "ex4": ExperimentDefinition(
        name="ex4",
        n_dimensions=1,
        state_dimension=2,
        diff_type="diagonal",
        xlim=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        drift=ex4_drift,
        diffusion_factor=ex4_diffusion_factor,
        coupled_fn=ex4_coupled_fn,
    ),
    "ex5": ExperimentDefinition(
        name="ex5",
        n_dimensions=2,
        state_dimension=2,
        diff_type="diagonal",
        xlim=None,
        drift=ex5_drift,
        diffusion_factor=ex5_diffusion_factor,
    ),
    "ex6": ExperimentDefinition(
        name="ex6",
        n_dimensions=1,
        state_dimension=2,
        diff_type="diagonal",
        xlim=None,
        drift=ex6_drift,
        diffusion_factor=ex6_diffusion_factor,
    ),
    "ex7": ExperimentDefinition(
        name="ex7",
        n_dimensions=2,
        state_dimension=2,
        diff_type="diagonal",
        xlim=np.array([[-1.0, 1.0], [-1.0, 1.0]]),
        drift=ex7_drift,
        diffusion_factor=ex7_diffusion_factor,
    ),
    "ex8": ExperimentDefinition(
        name="ex8",
        n_dimensions=2,
        state_dimension=2,
        diff_type="symmetric",
        xlim=np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        drift=ex8_drift,
        diffusion_factor=ex8_diffusion_factor,
    ),
}


def get_experiment(name: str) -> ExperimentDefinition:
    try:
        return EXPERIMENTS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown experiment: {name}") from exc
