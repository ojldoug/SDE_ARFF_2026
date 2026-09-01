"""
Data generation for Experiment 6: stochastically forced wave equation.

This implements the staggered-grid scheme used in the legacy experiment.
The learning data are reformulated as one-step EM-type observations with

    h_eff = dt**2 / 2.

Under this transformation the effective diffusion factor is g(x)/2.
"""

from __future__ import annotations

import numpy as np


def integrate_stochastic_wave(
    u0,
    v0,
    f,
    g,
    time,
    space,
    *,
    seed,
    periodic_boundary=False,
):
    rng = np.random.default_rng(seed)

    dt = time[1] - time[0]
    noise_std = np.sqrt(2.0 * dt**2)

    idx_even = np.arange(len(space))[::2]
    idx_odd = np.arange(len(space))[1::2]

    buffer = np.zeros(len(idx_even) + 2)

    u0_large = buffer.copy()
    u0_large[1:-1] = u0[idx_even]

    if periodic_boundary:
        u0_large[0] = u0[idx_even][-1]
        u0_large[-1] = u0[idx_even][0]

    u0_m1 = u0_large[:-2]
    u0_p1 = u0_large[2:]

    um1_i = (
        0.5 * (u0_m1 + u0_p1)
        - dt * v0[idx_odd]
    )

    f0 = f(
        space[idx_even],
        0.0,
        0.5 * (u0_m1 + u0_p1),
    )
    g0 = g(
        space[idx_even],
        0.0,
        0.5 * (u0_m1 + u0_p1),
    )

    # The initialization step has half the interior noise variance,
    # matching the legacy staggered-grid scheme.
    W0 = rng.normal(
        loc=0.0,
        scale=noise_std / np.sqrt(2.0),
        size=len(idx_even),
    )

    u1_i = (
        u0_m1
        + u0_p1
        - um1_i
        + 0.5 * dt**2 * f0
        + 0.5 * g0 * W0
    )

    u = np.zeros((len(time), len(idx_even)))
    u[0, :] = um1_i
    u[1, :] = u0[idx_even]
    u[2, :] = u1_i

    for j in range(2, len(time) - 1):
        if j % 2 == 0:
            idx_jm1 = idx_even
            idx_j = idx_odd
        else:
            idx_jm1 = idx_odd
            idx_j = idx_even

        u_large = buffer.copy()
        u_large[1:-1] = u[j, :]

        if periodic_boundary:
            u_large[0] = u[j, -1]
            u_large[-1] = u[j, 0]

        uim1 = u_large[:-2]
        uip1 = u_large[2:]

        ft = f(
            space[idx_j],
            time[j],
            0.5 * (uim1 + uip1),
        )
        gt = g(
            space[idx_j],
            time[j],
            0.5 * (uim1 + uip1),
        )

        Wt = rng.normal(
            loc=0.0,
            scale=noise_std,
            size=len(idx_jm1),
        )

        u[j + 1, :] = (
            uim1
            + uip1
            - u[j - 1, :]
            + dt**2 * ft
            + 0.5 * gt * Wt
        )

    return u


def split_wave_learning_data(u, space_half, time):
    """
    Reformulate the staggered-grid solution into EM-type observations.
    """
    u_jm1 = []
    u_j = []
    p_n = []
    u_np1 = []

    for j in range(1, u.shape[0] - 1):
        u_jm1.append(u[j - 1, 1:-1])
        u_j.append([u[j, :-2] + u[j, 2:]])
        u_np1.append(u[j + 1, 1:-1])

        p_j = np.zeros((len(space_half) - 2, 2))
        p_j[:, 0] = time[j]
        p_j[:, 1] = space_half[1:-1]
        p_n.append(p_j)

    u_jm1 = np.vstack(u_jm1).ravel()
    u_j = np.vstack(u_j).ravel()
    u_np1 = np.vstack(u_np1).ravel()
    p_n = np.vstack(p_n)

    u_n = 0.5 * u_j.reshape(-1, 1)
    u_next = 0.5 * (u_np1 + u_jm1).reshape(-1, 1)

    return u_n, p_n, u_next


def generate_wave_data(config):
    """Generate Experiment 6 learning data."""
    data = config.data

    if data.grid_step is None:
        raise ValueError("Experiment 6 requires grid_step.")

    if data.trajectory_time is None:
        raise ValueError("Experiment 6 requires trajectory_time.")

    dt = data.grid_step
    effective_h = 0.5 * dt**2

    if data.observation_lag is None:
        raise ValueError(
            "Experiment 6 requires observation_lag."
        )

    if not np.isclose(
        data.observation_lag,
        effective_h,
    ):
        raise ValueError(
            "Experiment 6 observation_lag must equal "
            "0.5 * grid_step**2."
        )

    space_size = 1.0
    time_size = data.trajectory_time

    space = np.arange(int(space_size / dt)) * dt
    time = np.arange(int(time_size / dt)) * dt

    f = lambda x, t, u: 5.0 * np.sin(4.0 * np.pi * x)
    g = lambda x, t, u: 0.05 * (
        1.0 + np.exp(-150.0 * (x - 0.5) ** 2)
    )

    u0 = (
        np.exp(-150.0 * (space - 0.5) ** 2)
        / 20.0
    )

    v0 = -2.0 * np.gradient(u0, dt)

    u = integrate_stochastic_wave(
        u0,
        v0,
        f,
        g,
        time,
        space,
        seed=data.seed,
        periodic_boundary=False,
    )

    u_n, x_data, u_next = split_wave_learning_data(
        u,
        space[::2],
        time,
    )

    r_data = u_next - u_n

    step_sizes = np.full(
        (len(x_data), 1),
        data.observation_lag,
    )

    if data.target_samples is not None and len(x_data) < data.target_samples:
        raise RuntimeError(
            f"Experiment 6 generated only {len(x_data)} samples, "
            f"fewer than requested {data.target_samples}."
        )

    # Optionally select a deterministic subset. For the final Experiment 6
    # configuration, target_samples equals the full number of generated
    # interior learning points, so all samples are retained.
    if (
        data.target_samples is not None
        and data.target_samples < len(x_data)
    ):
        rng = np.random.default_rng(data.seed)
        indices = rng.choice(
            len(x_data),
            size=data.target_samples,
            replace=False,
        )

        x_data = x_data[indices]
        r_data = r_data[indices]
        step_sizes = step_sizes[indices]

    return x_data, r_data, step_sizes
