"""
Validation-selected ARFF regression.

Implements the intended legacy ARFF model-selection rule:

  * reserve an internal validation subset;
  * perform ARFF adaptation on the remaining fitting subset;
  * evaluate validation regression MSE after every adaptation step;
  * retain the model with the smallest raw validation MSE;
  * use a short moving-average stagnation rule only for stopping.

The validation target is the regression target itself. True coefficient
functions and the SDE test split are never used for model selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import jax
import jax.numpy as jnp
import numpy as np

from src.arff.regression import (
    ARFFModel,
    fit_amplitudes,
    make_compiled_adaptation_step,
    predict,
)


Array = jax.Array


@dataclass(frozen=True)
class ValidationSelectedResult:
    model: ARFFModel
    validation_mse: np.ndarray
    moving_average: np.ndarray
    cumulative_time: np.ndarray
    best_iteration: int
    best_validation_mse: float
    best_time: float
    stopped_iteration: int


def _split_indices(
    n: int,
    *,
    validation_fraction: float,
    seed: int,
):
    if not (
        0.0
        < validation_fraction
        < 1.0
    ):
        raise ValueError(
            "validation_fraction must lie in (0, 1)."
        )

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(n)

    n_validation = max(
        1,
        int(
            round(
                validation_fraction
                * n
            )
        ),
    )

    if n_validation >= n:
        raise ValueError(
            "Internal validation split leaves no fitting data."
        )

    validation_idx = permutation[
        :n_validation
    ]

    fit_idx = permutation[
        n_validation:
    ]

    return (
        fit_idx,
        validation_idx,
    )


def fit_validation_selected_arff(
    key,
    x,
    y,
    *,
    K: int,
    M_min: int,
    M_max: int,
    lambda_reg: float,
    gamma: float,
    delta: float,
    resampling: bool,
    metropolis_test: bool,
    validation_fraction: float = 0.1,
    validation_seed: int = 0,
    moving_average_length: int = 5,
    patience: int = 5,
    compiled_adaptation_step=None,
):
    """
    Fit ARFF and return the minimum-validation-MSE checkpoint.

    Iterations are reported one-based:
        best_iteration = 1 means the model after the first adaptation.
    """
    x = jnp.asarray(x)
    y = jnp.asarray(y)

    if y.ndim == 1:
        y = y[:, None]

    if x.ndim != 2 or y.ndim != 2:
        raise ValueError(
            "x and y must be matrices."
        )

    if len(x) != len(y):
        raise ValueError(
            "x and y must have equal sample counts."
        )

    if not (
        0 <= M_min <= M_max
    ):
        raise ValueError(
            "Require 0 <= M_min <= M_max."
        )

    if moving_average_length <= 0:
        raise ValueError(
            "moving_average_length must be positive."
        )

    if patience <= 0:
        raise ValueError(
            "patience must be positive."
        )

    (
        fit_idx,
        validation_idx,
    ) = _split_indices(
        len(x),
        validation_fraction=(
            validation_fraction
        ),
        seed=validation_seed,
    )

    x_fit = x[fit_idx]
    y_fit = y[fit_idx]

    x_validation = x[
        validation_idx
    ]

    y_validation = y[
        validation_idx
    ]

    omega = jnp.zeros(
        (
            x.shape[1],
            K,
        ),
        dtype=x.dtype,
    )

    amp = fit_amplitudes(
        x_fit,
        y_fit,
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
                metropolis_test=(
                    metropolis_test
                ),
            )
        )

    validation_history = []
    moving_history = []
    time_history = []

    best_model = None
    best_validation_mse = np.inf
    best_iteration = -1
    best_time = np.nan

    minimum_moving_average = np.inf
    minimum_moving_iteration = -1

    start = time.perf_counter()

    for iteration_zero_based in range(
        M_max
    ):
        key, model = (
            compiled_adaptation_step(
                key,
                model,
                x_fit,
                y_fit,
            )
        )

        prediction = predict(
            model,
            x_validation,
        )

        validation_mse_device = (
            jnp.mean(
                (
                    prediction
                    - y_validation
                )
                ** 2
            )
        )

        validation_mse = float(
            jax.device_get(
                validation_mse_device
            )
        )

        elapsed = (
            time.perf_counter()
            - start
        )

        if not np.isfinite(
            validation_mse
        ):
            validation_mse = np.inf

        validation_history.append(
            validation_mse
        )

        time_history.append(
            elapsed
        )

        window = validation_history[
            -moving_average_length:
        ]

        moving_average = float(
            np.mean(
                window
            )
        )

        moving_history.append(
            moving_average
        )

        iteration = (
            iteration_zero_based
            + 1
        )

        if (
            validation_mse
            < best_validation_mse
        ):
            best_validation_mse = (
                validation_mse
            )

            best_iteration = iteration
            best_time = elapsed

            # ARFFModel is immutable. Retaining these arrays retains
            # the selected checkpoint even as later model variables
            # are rebound to new arrays.
            best_model = ARFFModel(
                omega=model.omega,
                amp=model.amp,
            )

        if (
            moving_average
            < minimum_moving_average
        ):
            minimum_moving_average = (
                moving_average
            )

            minimum_moving_iteration = (
                iteration
            )

        # Historical intent:
        # only stop after exceeding M_min and after the moving-average
        # optimum has failed to improve for `patience` iterations.
        if (
            iteration > M_min
            and (
                iteration
                - minimum_moving_iteration
            )
            > patience
        ):
            break

    if best_model is None:
        raise RuntimeError(
            "No finite ARFF validation checkpoint was found."
        )

    return (
        key,
        ValidationSelectedResult(
            model=best_model,
            validation_mse=np.asarray(
                validation_history,
                dtype=np.float64,
            ),
            moving_average=np.asarray(
                moving_history,
                dtype=np.float64,
            ),
            cumulative_time=np.asarray(
                time_history,
                dtype=np.float64,
            ),
            best_iteration=int(
                best_iteration
            ),
            best_validation_mse=float(
                best_validation_mse
            ),
            best_time=float(
                best_time
            ),
            stopped_iteration=int(
                len(
                    validation_history
                )
            ),
        ),
    )
