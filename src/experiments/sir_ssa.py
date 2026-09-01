"""
SIR stochastic simulation algorithm (SSA) used for Experiment 5.

Benchmark provenance
--------------------
The reaction system and SIR benchmark are based on the Gillespie
experiment of Dietrich et al., "Learning effective stochastic
differential equations from microscopic simulations: linking
stochastic numerics to deep learning."

This implementation is written independently for the present repository.
In particular, observations are taken at prescribed fixed times rather
than by subsampling event-triggered recordings.

State convention
----------------
Internally the integer-valued state is

    (I, R),

with

    S = N - I - R.

The returned normalized state is

    (S/N, R/N),

matching the (theta_0, theta_2) coordinates used in the manuscript.

Reactions
---------
1. Infection:
       S + I -> 2 I

2. Recovery:
       I -> R

3. Loss of immunity:
       R -> S

For Experiment 5, k3 = 0.

The propensities are

    a1 = 4 k1 I S / N,
    a2 = k2 I,
    a3 = k3 R.

Equivalently, in normalized coordinates,

    a1 = N * 4 k1 theta_0 theta_1,
    a2 = N * k2 theta_1,
    a3 = N * k3 theta_2.
"""

from __future__ import annotations

import numpy as np


def _validate_initial_state(initial_state: np.ndarray) -> np.ndarray:
    """Validate a normalized (S, I, R) initial condition."""
    initial_state = np.asarray(initial_state, dtype=float)

    if initial_state.shape != (3,):
        raise ValueError("initial_state must have shape (3,) = (S, I, R).")

    if np.any(initial_state < 0.0):
        raise ValueError("Initial fractions must be non-negative.")

    if not np.isclose(initial_state.sum(), 1.0):
        raise ValueError("Initial fractions must sum to one.")

    return initial_state


def simulate_sir_ssa_fixed_observations(
    initial_state: np.ndarray,
    observation_times: np.ndarray,
    *,
    population_size: int = 1024,
    k1: float = 1.0,
    k2: float = 1.0,
    k3: float = 0.0,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate the SIR jump process and observe it at prescribed times.

    Parameters
    ----------
    initial_state
        Normalized fractions (S, I, R), summing to one.
    observation_times
        Strictly increasing observation times. The first time must be
        non-negative. The initial state is understood to hold at t=0.
    population_size
        Total population N.
    k1, k2, k3
        Reaction-rate parameters.
    rng
        NumPy random-number generator.

    Returns
    -------
    observations
        Array of shape (len(observation_times), 2), containing
        (S/N, R/N) at the requested observation times.
    """
    initial_state = _validate_initial_state(initial_state)
    observation_times = np.asarray(observation_times, dtype=float)

    if observation_times.ndim != 1 or len(observation_times) == 0:
        raise ValueError("observation_times must be a non-empty 1D array.")

    if np.any(observation_times < 0.0):
        raise ValueError("Observation times must be non-negative.")

    if np.any(np.diff(observation_times) <= 0.0):
        raise ValueError("Observation times must be strictly increasing.")

    if population_size <= 0:
        raise ValueError("population_size must be positive.")

    # Convert normalized fractions to integer counts.
    #
    # Convert normalized fractions to integer counts using the
    # largest-remainder method so that the counts sum exactly to N
    # while staying as close as possible to the sampled fractions.
    expected_counts = initial_state * population_size
    counts = np.floor(expected_counts).astype(int)

    remainder = population_size - counts.sum()

    if remainder > 0:
        fractional_parts = expected_counts - counts
        indices = np.argsort(fractional_parts)[::-1][:remainder]
        counts[indices] += 1

    S, I, R = counts

    if min(S, I, R) < 0:
        raise ValueError("Rounded initial counts are not admissible.")

    observations = np.empty((len(observation_times), 2), dtype=float)

    current_time = 0.0

    # Draw the first event time. We maintain the next event explicitly so
    # that the state can be sampled exactly between events.
    next_event_time = None
    next_reaction = None

    def draw_next_event():
        nonlocal next_event_time, next_reaction

        if I == 0 and (k3 == 0.0 or R == 0):
            next_event_time = np.inf
            next_reaction = None
            return

        a1 = 4.0 * k1 * I * S / population_size
        a2 = k2 * I
        a3 = k3 * R

        rates = np.array([a1, a2, a3], dtype=float)
        total_rate = rates.sum()

        if total_rate <= 0.0:
            next_event_time = np.inf
            next_reaction = None
            return

        waiting_time = rng.exponential(scale=1.0 / total_rate)
        next_event_time = current_time + waiting_time

        u = rng.uniform(0.0, total_rate)
        next_reaction = int(np.searchsorted(np.cumsum(rates), u, side="right"))

    draw_next_event()

    for j, observation_time in enumerate(observation_times):

        # Execute every SSA event occurring on or before the requested
        # observation time. Between events, the jump process is constant.
        while next_event_time <= observation_time:
            current_time = next_event_time

            if next_reaction == 0:
                # S + I -> 2I
                if S <= 0:
                    raise RuntimeError("Invalid infection reaction.")
                S -= 1
                I += 1

            elif next_reaction == 1:
                # I -> R
                if I <= 0:
                    raise RuntimeError("Invalid recovery reaction.")
                I -= 1
                R += 1

            elif next_reaction == 2:
                # R -> S
                if R <= 0:
                    raise RuntimeError("Invalid immunity-loss reaction.")
                R -= 1
                S += 1

            else:
                raise RuntimeError("Invalid SSA reaction index.")

            if S + I + R != population_size:
                raise RuntimeError("Population conservation failed.")

            draw_next_event()

        observations[j, 0] = S / population_size
        observations[j, 1] = R / population_size

    return observations


def sample_initial_simplex(
    n_trajectories: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample initial (S, I, R) fractions uniformly by normalizing three
    independent U(0,1) variables, matching the legacy experiment setup.
    """
    raw = rng.uniform(0.0, 1.0, size=(n_trajectories, 3))
    return raw / raw.sum(axis=1, keepdims=True)


def generate_sir_data(config):
    """
    Generate the final fixed-lag Experiment 5 dataset.

    Each SSA trajectory is observed at the prescribed uniform times

        0, h, 2h, ..., T,

    and every consecutive observation pair is retained.

    Returns
    -------
    x_data
        Initial state of each transition in (S/N, R/N) coordinates.
    r_data
        State increment over one observation interval.
    step_sizes
        Exact observation lag for each transition.
    """
    data = config.data

    if data.n_trajectories is None:
        raise ValueError("Experiment 5 requires n_trajectories.")

    if data.trajectory_time is None:
        raise ValueError("Experiment 5 requires trajectory_time.")

    if data.observation_lag is None:
        raise ValueError("Experiment 5 requires observation_lag.")

    rng = np.random.default_rng(data.seed)

    observation_times = np.arange(
        0.0,
        data.trajectory_time + 0.5 * data.observation_lag,
        data.observation_lag,
    )

    initial_states = sample_initial_simplex(
        data.n_trajectories,
        rng,
    )

    trajectories = []

    for initial_state in initial_states:
        trajectory = simulate_sir_ssa_fixed_observations(
            initial_state,
            observation_times,
            population_size=1024,
            k1=1.0,
            k2=1.0,
            k3=0.0,
            rng=rng,
        )
        trajectories.append(trajectory)

    trajectories = np.stack(trajectories)

    x_data = trajectories[:, :-1, :].reshape(-1, 2)
    y_data = trajectories[:, 1:, :].reshape(-1, 2)
    r_data = y_data - x_data

    step_sizes = np.full(
        (len(x_data), 1),
        data.observation_lag,
        dtype=float,
    )

    if data.target_samples is not None:
        if len(x_data) != data.target_samples:
            raise RuntimeError(
                f"Expected {data.target_samples} Experiment 5 samples, "
                f"but generated {len(x_data)}."
            )

    return x_data, r_data, step_sizes