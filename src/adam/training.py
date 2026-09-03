"""
Training utilities for Adam SDE baselines.

Training and validation data are supplied explicitly. This module never
constructs or modifies the canonical train/validation/test split.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import optax

from src.adam.fourier import (
    AdamFourierModel,
    gaussian_nll,
    make_optimizer,
)


@dataclass(frozen=True)
class TrainingResult:
    model: AdamFourierModel
    best_epoch: int
    best_validation_nll: float
    training_nll: np.ndarray
    validation_nll: np.ndarray


def make_compiled_adam_functions(
    learning_rate: float,
):
    """
    Construct the optimizer and JIT-compiled Adam update/NLL functions.

    The returned functions can be explicitly warmed up by the experiment
    runner before benchmark timing begins.
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
            gaussian_nll
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
        gaussian_nll
    )

    return (
        optimizer,
        compiled_train_step,
        compiled_nll,
    )


def fit_adam_fourier(
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
    Train an Adam Fourier model and select the checkpoint with the
    smallest canonical validation NLL.

    All supplied data are kept as JAX arrays. Each epoch performs one
    device-side random permutation of the training observations and then
    uses contiguous minibatch slices of the shuffled arrays.

    Minibatch losses are accumulated on-device. Host synchronization is
    performed only once per epoch, together with validation evaluation.
    """
    if epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    x_train = jnp.asarray(x_train)
    r_train = jnp.asarray(r_train)
    h_train = jnp.asarray(h_train)

    x_validation = jnp.asarray(
        x_validation
    )
    r_validation = jnp.asarray(
        r_validation
    )
    h_validation = jnp.asarray(
        h_validation
    )

    n_train = len(x_train)

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
        len(r_validation) != len(x_validation)
        or len(h_validation) != len(x_validation)
    ):
        raise ValueError(
            "Validation arrays have inconsistent "
            "sample counts."
        )

    # Start real training from a fresh optimizer state. Any state
    # produced during compilation warm-up in the runner is discarded.
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

    best_model = model
    best_epoch = -1
    best_validation_nll = np.inf

    for epoch in range(epochs):
        # --------------------------------------------------------
        # Shuffle once per epoch, entirely on-device.
        # --------------------------------------------------------
        key, permutation_key = jax.random.split(
            key
        )

        permutation = jax.random.permutation(
            permutation_key,
            n_train,
        )

        x_epoch = x_train[permutation]
        r_epoch = r_train[permutation]
        h_epoch = h_train[permutation]

        # Accumulate the sample-weighted minibatch loss on-device.
        weighted_loss_sum = jnp.asarray(
            0.0,
            dtype=x_train.dtype,
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

            model, opt_state, batch_loss = (
                compiled_train_step(
                    model,
                    opt_state,
                    x_epoch[start:end],
                    r_epoch[start:end],
                    h_epoch[start:end],
                )
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

        # One explicit synchronization per epoch.
        (
            training_nll,
            validation_nll,
        ) = jax.device_get(
            (
                training_nll_device,
                validation_nll_device,
            )
        )

        training_nll = float(
            training_nll
        )

        validation_nll = float(
            validation_nll
        )

        training_history[epoch] = (
            training_nll
        )

        validation_history[epoch] = (
            validation_nll
        )

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
            "No valid Adam checkpoint was selected."
        )

    return key, TrainingResult(
        model=best_model,
        best_epoch=best_epoch,
        best_validation_nll=(
            best_validation_nll
        ),
        training_nll=training_history,
        validation_nll=validation_history,
    )