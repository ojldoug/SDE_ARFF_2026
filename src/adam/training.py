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

from src.adam.fourier import (
    AdamFourierModel,
    gaussian_nll,
    make_optimizer,
    train_step,
)


@dataclass(frozen=True)
class TrainingResult:
    model: AdamFourierModel
    best_epoch: int
    best_validation_nll: float
    training_nll: np.ndarray
    validation_nll: np.ndarray


def fit_adam_fourier(
    key,
    initial_model: AdamFourierModel,
    x_train,
    r_train,
    h_train,
    x_validation,
    r_validation,
    h_validation,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
):
    """
    Train an Adam Fourier model and select the checkpoint with the
    smallest canonical validation NLL.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    x_train = jnp.asarray(x_train)
    r_train = jnp.asarray(r_train)
    h_train = jnp.asarray(h_train)

    x_validation = jnp.asarray(x_validation)
    r_validation = jnp.asarray(r_validation)
    h_validation = jnp.asarray(h_validation)

    n_train = len(x_train)

    if n_train == 0:
        raise ValueError("Training set must not be empty.")

    if len(x_validation) == 0:
        raise ValueError("Validation set must not be empty.")

    if len(r_train) != n_train or len(h_train) != n_train:
        raise ValueError(
            "Training arrays have inconsistent sample counts."
        )

    if (
        len(r_validation) != len(x_validation)
        or len(h_validation) != len(x_validation)
    ):
        raise ValueError(
            "Validation arrays have inconsistent sample counts."
        )

    optimizer = make_optimizer(learning_rate)
    opt_state = optimizer.init(initial_model)

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
        # Shuffle only the supplied training observations.
        key, permutation_key = jax.random.split(key)

        permutation = np.asarray(
            jax.random.permutation(
                permutation_key,
                n_train,
            )
        )

        batch_losses = []
        batch_sizes = []

        for start in range(
            0,
            n_train,
            batch_size,
        ):
            batch_idx = permutation[
                start:start + batch_size
            ]

            model, opt_state, batch_loss = train_step(
                model,
                optimizer,
                opt_state,
                x_train[batch_idx],
                r_train[batch_idx],
                h_train[batch_idx],
            )

            batch_losses.append(
                float(batch_loss)
            )
            batch_sizes.append(
                len(batch_idx)
            )

        # Weighted average because the final minibatch may be smaller.
        training_nll = np.average(
            batch_losses,
            weights=batch_sizes,
        )

        validation_nll = float(
            gaussian_nll(
                model,
                x_validation,
                r_validation,
                h_validation,
            )
        )

        training_history[epoch] = training_nll
        validation_history[epoch] = validation_nll

        if not np.isfinite(training_nll):
            raise RuntimeError(
                f"Non-finite training NLL at epoch {epoch}."
            )

        if not np.isfinite(validation_nll):
            raise RuntimeError(
                f"Non-finite validation NLL at epoch {epoch}."
            )

        if validation_nll < best_validation_nll:
            best_validation_nll = validation_nll
            best_epoch = epoch
            best_model = model

    if best_epoch < 0:
        raise RuntimeError(
            "No valid Adam checkpoint was selected."
        )

    return key, TrainingResult(
        model=best_model,
        best_epoch=best_epoch,
        best_validation_nll=best_validation_nll,
        training_nll=training_history,
        validation_nll=validation_history,
    )