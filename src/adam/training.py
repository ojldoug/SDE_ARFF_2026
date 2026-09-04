"""
Training utilities for Adam SDE baselines.

Training and validation data are supplied explicitly. This module never
constructs or modifies the canonical train/validation/test split.

The training loop is model-agnostic: Fourier and MLP baselines use the
same optimizer, minibatching, shuffling, validation checkpoint
selection, synchronization, and timing convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import optax


@dataclass(frozen=True)
class TrainingResult:
    model: Any
    best_epoch: int
    best_validation_nll: float
    training_nll: np.ndarray
    validation_nll: np.ndarray
    cumulative_time: np.ndarray


def make_optimizer(
    learning_rate: float,
):
    """
    Canonical Adam optimizer used by all Adam baselines.
    """
    if learning_rate <= 0.0:
        raise ValueError(
            "learning_rate must be positive."
        )

    return optax.adam(
        learning_rate,
        b1=0.9,
        b2=0.999,
        eps=1e-7,
    )


def make_compiled_adam_functions(
    learning_rate: float,
    *,
    nll_fn: Callable,
):
    """
    Construct the optimizer and JIT-compiled update/NLL functions for
    one Adam model family.

    Parameters
    ----------
    learning_rate:
        Adam learning rate.

    nll_fn:
        Model-specific function with signature

            nll_fn(model, x, r, h) -> scalar mean NLL.

        Fourier and MLP baselines therefore share all optimization
        machinery while retaining their own model representations.
    """
    optimizer = make_optimizer(
        learning_rate
    )

    @jax.jit
    def compiled_train_step(
        model,
        opt_state,
        x_batch,
        r_batch,
        h_batch,
    ):
        loss, gradients = jax.value_and_grad(
            nll_fn
        )(
            model,
            x_batch,
            r_batch,
            h_batch,
        )

        updates, new_opt_state = optimizer.update(
            gradients,
            opt_state,
            model,
        )

        new_model = optax.apply_updates(
            model,
            updates,
        )

        return (
            new_model,
            new_opt_state,
            loss,
        )

    compiled_nll = jax.jit(
        nll_fn
    )

    return (
        optimizer,
        compiled_train_step,
        compiled_nll,
    )


def fit_adam(
    key,
    initial_model,
    x_train,
    r_train,
    h_train,
    x_validation,
    r_validation,
    h_validation,
    *,
    epochs: int,
    batch_size: int,
    optimizer,
    compiled_train_step,
    compiled_nll,
):
    """
    Train one Adam SDE model and select the checkpoint with the smallest
    canonical validation NLL.

    All supplied data are kept as JAX arrays. Each epoch performs one
    device-side random permutation of the training observations and then
    uses contiguous minibatch slices of the shuffled arrays.

    Minibatch losses are accumulated on-device. Host synchronization is
    performed only once per epoch, together with validation evaluation.

    Cumulative algorithm time is recorded immediately after that
    existing epoch-boundary synchronization. Recording this history does
    not introduce any additional device synchronization relative to the
    canonical training implementation.
    """
    if epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    x_train = jnp.asarray(
        x_train
    )

    r_train = jnp.asarray(
        r_train
    )

    h_train = jnp.asarray(
        h_train
    )

    x_validation = jnp.asarray(
        x_validation
    )

    r_validation = jnp.asarray(
        r_validation
    )

    h_validation = jnp.asarray(
        h_validation
    )

    n_train = len(
        x_train
    )

    if n_train == 0:
        raise ValueError(
            "Training set must not be empty."
        )

    if len(x_validation) == 0:
        raise ValueError(
            "Validation set must not be empty."
        )

    if (
        len(r_train) != n_train
        or len(h_train) != n_train
    ):
        raise ValueError(
            "Training arrays have inconsistent "
            "sample counts."
        )

    if (
        len(r_validation)
        != len(x_validation)
        or len(h_validation)
        != len(x_validation)
    ):
        raise ValueError(
            "Validation arrays have inconsistent "
            "sample counts."
        )

    # Start real training from a fresh optimizer state. Any optimizer
    # state used during compilation warm-up in the runner is discarded.
    opt_state = optimizer.init(
        initial_model
    )

    model = initial_model

    training_history = np.empty(
        epochs,
        dtype=float,
    )

    validation_history = np.empty(
        epochs,
        dtype=float,
    )

    cumulative_time = np.empty(
        epochs,
        dtype=float,
    )

    best_model = model
    best_epoch = -1
    best_validation_nll = np.inf

    algorithm_start = (
        time.perf_counter()
    )

    for epoch in range(
        epochs
    ):
        # --------------------------------------------------------
        # Shuffle once per epoch, entirely on-device.
        # --------------------------------------------------------
        (
            key,
            permutation_key,
        ) = jax.random.split(
            key
        )

        permutation = (
            jax.random.permutation(
                permutation_key,
                n_train,
            )
        )

        x_epoch = x_train[
            permutation
        ]

        r_epoch = r_train[
            permutation
        ]

        h_epoch = h_train[
            permutation
        ]

        # Accumulate the sample-weighted minibatch loss on-device.
        weighted_loss_sum = (
            jnp.asarray(
                0.0,
                dtype=x_train.dtype,
            )
        )

        for start in range(
            0,
            n_train,
            batch_size,
        ):
            end = min(
                start + batch_size,
                n_train,
            )

            current_batch_size = (
                end - start
            )

            (
                model,
                opt_state,
                batch_loss,
            ) = compiled_train_step(
                model,
                opt_state,
                x_epoch[
                    start:end
                ],
                r_epoch[
                    start:end
                ],
                h_epoch[
                    start:end
                ],
            )

            weighted_loss_sum = (
                weighted_loss_sum
                + current_batch_size
                * batch_loss
            )

        training_nll_device = (
            weighted_loss_sum
            / n_train
        )

        validation_nll_device = (
            compiled_nll(
                model,
                x_validation,
                r_validation,
                h_validation,
            )
        )

        # One explicit synchronization per epoch, required for
        # validation-based checkpoint selection.
        (
            training_nll,
            validation_nll,
        ) = jax.device_get(
            (
                training_nll_device,
                validation_nll_device,
            )
        )

        # Timestamp immediately after the synchronization already
        # required by checkpoint selection.
        cumulative_time[
            epoch
        ] = (
            time.perf_counter()
            - algorithm_start
        )

        training_nll = float(
            training_nll
        )

        validation_nll = float(
            validation_nll
        )

        training_history[
            epoch
        ] = training_nll

        validation_history[
            epoch
        ] = validation_nll

        if not np.isfinite(
            training_nll
        ):
            raise RuntimeError(
                "Non-finite training NLL at "
                f"epoch {epoch}."
            )

        if not np.isfinite(
            validation_nll
        ):
            raise RuntimeError(
                "Non-finite validation NLL at "
                f"epoch {epoch}."
            )

        if not np.isfinite(
            cumulative_time[
                epoch
            ]
        ):
            raise RuntimeError(
                "Non-finite cumulative time at "
                f"epoch {epoch}."
            )

        if (
            validation_nll
            < best_validation_nll
        ):
            best_validation_nll = (
                validation_nll
            )

            best_epoch = epoch
            best_model = model

    if best_epoch < 0:
        raise RuntimeError(
            "No valid Adam checkpoint "
            "was selected."
        )

    if not np.all(
        np.diff(
            cumulative_time
        )
        >= 0.0
    ):
        raise RuntimeError(
            "Adam cumulative-time history "
            "is not monotonically "
            "non-decreasing."
        )

    return key, TrainingResult(
        model=best_model,
        best_epoch=best_epoch,
        best_validation_nll=(
            best_validation_nll
        ),
        training_nll=(
            training_history
        ),
        validation_nll=(
            validation_history
        ),
        cumulative_time=(
            cumulative_time
        ),
    )


# ------------------------------------------------------------------
# Backward-compatible alias.
#
# Existing Fourier runners can continue importing fit_adam_fourier
# while the generic trainer is also used by the MLP baseline.
# ------------------------------------------------------------------

fit_adam_fourier = fit_adam