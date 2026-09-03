"""
Adaptive random Fourier feature (ARFF) regression.

This module contains only the generic regression machinery. Experiment
or SDE-specific two-stage logic lives elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


Array = jax.Array


@dataclass(frozen=True)
class ARFFModel:
    omega: Array
    amp: Array

    def tree_flatten(self):
        children = (
            self.omega,
            self.amp,
        )
        return children, None

    @classmethod
    def tree_unflatten(
        cls,
        aux_data,
        children,
    ):
        omega, amp = children

        return cls(
            omega=omega,
            amp=amp,
        )


jax.tree_util.register_pytree_node_class(
    ARFFModel
)


def fourier_features(
    omega: Array,
    x: Array,
) -> Array:
    """
    Real Fourier feature matrix [cos(X omega), sin(X omega)].
    """
    projection = x @ omega

    return jnp.concatenate(
        [
            jnp.cos(projection),
            jnp.sin(projection),
        ],
        axis=-1,
    )


def predict(
    model: ARFFModel,
    x: Array,
) -> Array:
    return (
        fourier_features(
            model.omega,
            x,
        )
        @ model.amp
    )


def fit_amplitudes(
    x: Array,
    y: Array,
    omega: Array,
    lambda_reg: float,
) -> Array:
    """
    Ridge-regression amplitudes for fixed Fourier frequencies.

    This solves the normal equations corresponding to

        mean_j |Phi(x_j) beta - y_j|^2
        + lambda_reg |beta|^2.
    """
    features = fourier_features(
        omega,
        x,
    )

    # The ARFF feature matrices can become strongly correlated after
    # resampling. On GPU, JAX's default float32 matrix-multiplication
    # precision can be insufficient for the resulting normal equations.
    #
    # Request full float32 accumulation precision explicitly. This avoids
    # the severe coefficient instability observed with the default GPU
    # matmul precision while retaining float32 storage and computation.
    gram = jnp.matmul(
        features.T,
        features,
        precision=jax.lax.Precision.HIGHEST,
    )

    rhs = jnp.matmul(
        features.T,
        y,
        precision=jax.lax.Precision.HIGHEST,
    )

    regularized_gram = (
        gram
        + x.shape[0]
        * lambda_reg
        * jnp.eye(
            features.shape[1],
            dtype=features.dtype,
        )
    )

    return jnp.linalg.solve(
        regularized_gram,
        rhs,
    )


def frequency_amplitude_norm(
    amp: Array,
    K: int,
) -> Array:
    """
    Return one amplitude magnitude per Fourier frequency.

    The cosine and sine amplitudes associated with the same frequency
    are combined. For vector-valued regression outputs, the norm also
    combines all output components.
    """
    amp_by_frequency = amp.reshape(
        2,
        K,
        -1,
    )

    return jnp.linalg.norm(
        amp_by_frequency,
        axis=(0, 2),
    )


def adaptation_step(
    key: Array,
    model: ARFFModel,
    x: Array,
    y: Array,
    *,
    delta: float,
    lambda_reg: float,
    gamma: float,
    resampling: bool,
    metropolis_test: bool,
):
    """
    Perform one ARFF frequency-adaptation step.

    The ordering follows the adaptive-resampling algorithm:

        1. random-walk mutation,
        2. least-squares amplitude fit,
        3. optional Metropolis accept/reject,
        4. amplitude-weighted resampling,
        5. final amplitude refit.

    In particular, when resampling=True and metropolis_test=False,
    the iteration is

        random walk -> least squares -> resampling -> least squares.

    Thus the model returned by an iteration is based on the resampled
    frequency population, rather than on an unfiltered random-walk
    proposal.
    """
    omega_old = model.omega
    amp_old = model.amp

    K = omega_old.shape[1]

    tiny = jnp.finfo(
        amp_old.dtype
    ).tiny

    # ------------------------------------------------------------
    # 1. Random-walk mutation.
    # ------------------------------------------------------------
    key, subkey = jax.random.split(
        key
    )

    proposal = (
        omega_old
        + delta
        * jax.random.normal(
            subkey,
            omega_old.shape,
        )
    )

    # ------------------------------------------------------------
    # 2. Fit amplitudes at the proposed frequencies.
    # ------------------------------------------------------------
    proposal_amp = fit_amplitudes(
        x,
        y,
        proposal,
        lambda_reg,
    )

    # ------------------------------------------------------------
    # 3. Optional Metropolis correction.
    #
    # The current paper experiments are expected to use
    #
    #     metropolis_test=False,
    #
    # but this branch is retained for completeness.
    # ------------------------------------------------------------
    if metropolis_test:
        old_norm = frequency_amplitude_norm(
            amp_old,
            K,
        )

        proposal_norm = frequency_amplitude_norm(
            proposal_amp,
            K,
        )

        old_norm_safe = jnp.maximum(
            old_norm,
            tiny,
        )

        proposal_norm_safe = jnp.maximum(
            proposal_norm,
            tiny,
        )

        ratio = (
            proposal_norm_safe
            / old_norm_safe
        ) ** gamma

        key, subkey = jax.random.split(
            key
        )

        accept = (
            ratio
            >= jax.random.uniform(
                subkey,
                shape=(K,),
            )
        )

        omega = jnp.where(
            accept[None, :],
            proposal,
            omega_old,
        )

        # Because different frequencies can be accepted independently,
        # amplitudes must be refitted for the resulting mixed population.
        amp = fit_amplitudes(
            x,
            y,
            omega,
            lambda_reg,
        )

    else:
        omega = proposal
        amp = proposal_amp

    # ------------------------------------------------------------
    # 4. Amplitude-weighted resampling.
    #
    # Resampling is performed AFTER mutation and amplitude fitting,
    # matching the adaptive-resampling algorithm.
    # ------------------------------------------------------------
    if resampling:
        amp_norm = frequency_amplitude_norm(
            amp,
            K,
        )

        amp_norm_safe = jnp.maximum(
            amp_norm,
            tiny,
        )

        pmf = (
            amp_norm_safe
            / jnp.sum(
                amp_norm_safe
            )
        )

        key, subkey = jax.random.split(
            key
        )

        selected = jax.random.choice(
            subkey,
            K,
            shape=(K,),
            replace=True,
            p=pmf,
        )

        omega = omega[
            :,
            selected,
        ]

        # --------------------------------------------------------
        # 5. Refit amplitudes on the selected frequency population.
        #
        # This is important: the model returned by the iteration is
        # the resampled model, not the unfiltered proposal model.
        # --------------------------------------------------------
        amp = fit_amplitudes(
            x,
            y,
            omega,
            lambda_reg,
        )

    return (
        key,
        ARFFModel(
            omega=omega,
            amp=amp,
        ),
    )


def make_compiled_adaptation_step(
    *,
    delta: float,
    lambda_reg: float,
    gamma: float,
    resampling: bool,
    metropolis_test: bool,
):
    """
    Construct one JIT-compiled ARFF adaptation step.

    The scalar hyperparameters and algorithmic switches are captured in
    the closure, so they are static from JAX's point of view. The
    resulting function can be reused for every adaptation iteration
    having the same input/output array shapes.
    """

    @jax.jit
    def compiled_step(
        key,
        model,
        x,
        y,
    ):
        return adaptation_step(
            key,
            model,
            x,
            y,
            delta=delta,
            lambda_reg=lambda_reg,
            gamma=gamma,
            resampling=resampling,
            metropolis_test=metropolis_test,
        )

    return compiled_step


def fit_arff(
    key: Array,
    x: Array,
    y: Array,
    *,
    K: int,
    n_iterations: int,
    lambda_reg: float,
    gamma: float,
    delta: float,
    resampling: bool,
    metropolis_test: bool,
    compiled_adaptation_step=None,
):
    """
    Fit one ARFF regression model.

    No train/validation split is performed here. The caller decides
    which observations constitute the training data.

    Frequencies are initialized at zero. Each adaptation iteration
    performs random-walk mutation followed by amplitude fitting and,
    when enabled, amplitude-weighted resampling.

    By default the repeated adaptation step is JIT-compiled once and
    reused across all iterations. A preconstructed compiled step may be
    supplied by the caller when explicit compilation warm-up or timing
    control is required.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    if x.ndim != 2:
        raise ValueError(
            "x must have shape (N, d)."
        )

    if y.ndim == 1:
        y = y[:, None]

    if y.ndim != 2:
        raise ValueError(
            "y must have shape (N, q)."
        )

    if len(x) != len(y):
        raise ValueError(
            "x and y must contain the same samples."
        )

    if K <= 0:
        raise ValueError(
            "K must be positive."
        )

    if n_iterations < 0:
        raise ValueError(
            "n_iterations must be non-negative."
        )

    omega = jnp.zeros(
        (
            x.shape[1],
            K,
        ),
        dtype=x.dtype,
    )

    amp = fit_amplitudes(
        x,
        y,
        omega,
        lambda_reg,
    )

    model = ARFFModel(
        omega=omega,
        amp=amp,
    )

    if compiled_adaptation_step is None:
        compiled_adaptation_step = (
            make_compiled_adaptation_step(
                delta=delta,
                lambda_reg=lambda_reg,
                gamma=gamma,
                resampling=resampling,
                metropolis_test=metropolis_test,
            )
        )

    for _ in range(
        n_iterations
    ):
        key, model = (
            compiled_adaptation_step(
                key,
                model,
                x,
                y,
            )
        )

    return key, model