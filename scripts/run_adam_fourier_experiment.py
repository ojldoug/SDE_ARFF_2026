#!/usr/bin/env python3
"""Run one canonical Adam Fourier SDE experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adam.fourier import (
    gaussian_nll,
    initialize_model,
    predict_covariance,
    predict_fourier,
)
from src.adam.training import (
    fit_adam_fourier,
    make_compiled_adam_functions,
)
from src.experiments.config import (
    get_config,
)
from src.experiments.dataset import (
    load_dataset,
)
from src.experiments.definitions import (
    get_experiment,
)
from src.experiments.timing import (
    TimingResult,
    block_until_ready,
    timed_call,
)


def true_function_errors(
    model,
    x,
    *,
    true_drift,
    true_diffusion_factor,
):
    """
    RMSE of learned drift and covariance against the true functions.
    """
    x = np.asarray(x)

    learned_drift = np.asarray(
        predict_fourier(
            model.drift,
            x,
        )
    )

    learned_covariance = np.asarray(
        predict_covariance(
            model,
            x,
        )
    )

    drift_truth = np.asarray(
        true_drift(
            x
        )
    )

    sigma_truth = np.asarray(
        true_diffusion_factor(
            x
        )
    )

    covariance_truth = (
        sigma_truth
        @ np.swapaxes(
            sigma_truth,
            -1,
            -2,
        )
    )

    drift_rmse = np.sqrt(
        np.mean(
            (
                learned_drift
                - drift_truth
            ) ** 2
        )
    )

    covariance_rmse = np.sqrt(
        np.mean(
            (
                learned_covariance
                - covariance_truth
            ) ** 2
        )
    )

    return (
        drift_rmse,
        covariance_rmse,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "experiment",
        choices=[
            f"ex{i}"
            for i in range(1, 9)
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    name = args.experiment

    config = get_config(
        name
    )

    definition = get_experiment(
        name
    )

    if (
        config.fourier_frequencies
        is None
    ):
        raise ValueError(
            "No Fourier frequency count has "
            f"been established for {name}."
        )

    data = load_dataset(
        REPO_ROOT
        / "data"
        / f"{name}.npz"
    )

    train_idx = data.train_idx
    validation_idx = (
        data.validation_idx
    )
    test_idx = data.test_idx

    # ------------------------------------------------------------
    # Move benchmark training/validation data to the device before
    # compilation or algorithm timing begins.
    #
    # Dataset loading and host-to-device transfer are therefore not
    # counted as method training time.
    # ------------------------------------------------------------

    x_train = jnp.asarray(
        data.x[train_idx]
    )
    r_train = jnp.asarray(
        data.r[train_idx]
    )
    h_train = jnp.asarray(
        data.h[train_idx]
    )

    x_validation = jnp.asarray(
        data.x[validation_idx]
    )
    r_validation = jnp.asarray(
        data.r[validation_idx]
    )
    h_validation = jnp.asarray(
        data.h[validation_idx]
    )

    block_until_ready(
        (
            x_train,
            r_train,
            h_train,
            x_validation,
            r_validation,
            h_validation,
        )
    )

    key = jax.random.PRNGKey(
        args.seed
    )

    key, initialization_key = (
        jax.random.split(
            key
        )
    )

    initial_model = initialize_model(
        initialization_key,
        input_dimension=(
            definition.state_dimension
        ),
        output_dimension=(
            definition.n_dimensions
        ),
        n_frequencies=(
            config.fourier_frequencies
        ),
        diff_type=(
            definition.diff_type
        ),
    )

    block_until_ready(
        initial_model
    )

    print(
        f"Experiment : {name}"
    )
    print(
        f"seed       : {args.seed}"
    )
    print(
        "backend    : "
        f"{jax.default_backend()}"
    )
    print(
        f"train N    : {len(train_idx)}"
    )
    print(
        "validation : "
        f"{len(validation_idx)}"
    )
    print(
        f"test       : {len(test_idx)}"
    )
    print(
        "frequencies: "
        f"{config.fourier_frequencies}"
    )
    print(
        f"epochs     : {config.adam.epochs}"
    )
    print(
        "batch size : "
        f"{config.adam.batch_size}"
    )
    print(
        "learning rate: "
        f"{config.adam.learning_rate:.8e}"
    )
    print()

    # ------------------------------------------------------------
    # Construct optimizer and JIT-compiled functions.
    # ------------------------------------------------------------

    (
        optimizer,
        compiled_train_step,
        compiled_nll,
    ) = make_compiled_adam_functions(
        config.adam.learning_rate
    )

    initial_opt_state = optimizer.init(
        initial_model
    )

    # ------------------------------------------------------------
    # First-call/JIT warm-up.
    #
    # Compile every minibatch shape used by the real training run,
    # together with the validation NLL shape.
    #
    # All warm-up updates are discarded.
    # ------------------------------------------------------------

    batch_size = (
        config.adam.batch_size
    )

    n_train = len(
        x_train
    )

    full_batch_size = min(
        batch_size,
        n_train,
    )

    _, compile_full_batch = (
        timed_call(
            compiled_train_step,
            initial_model,
            initial_opt_state,
            x_train[
                :full_batch_size
            ],
            r_train[
                :full_batch_size
            ],
            h_train[
                :full_batch_size
            ],
        )
    )

    remainder = (
        n_train
        % batch_size
    )

    compile_remainder = 0.0

    if (
        remainder > 0
        and remainder
        != full_batch_size
    ):
        _, compile_remainder = (
            timed_call(
                compiled_train_step,
                initial_model,
                initial_opt_state,
                x_train[
                    :remainder
                ],
                r_train[
                    :remainder
                ],
                h_train[
                    :remainder
                ],
            )
        )

    _, compile_validation = (
        timed_call(
            compiled_nll,
            initial_model,
            x_validation,
            r_validation,
            h_validation,
        )
    )

    first_call_overhead = (
        compile_full_batch
        + compile_remainder
        + compile_validation
    )

    print(
        "first-call/JIT overhead: "
        f"{first_call_overhead:.3f} s"
    )

    print(
        "  full batch       : "
        f"{compile_full_batch:.3f} s"
    )

    if compile_remainder > 0.0:
        print(
            "  remainder batch  : "
            f"{compile_remainder:.3f} s"
        )

    print(
        "  validation       : "
        f"{compile_validation:.3f} s"
    )

    print()

    # ------------------------------------------------------------
    # Real training.
    #
    # The optimizer starts from a fresh state, while all required
    # compiled kernels have already been warmed up.
    # ------------------------------------------------------------

    (
        key,
        training,
    ), algorithm_time = timed_call(
        fit_adam_fourier,
        key,
        initial_model,
        x_train,
        r_train,
        h_train,
        x_validation,
        r_validation,
        h_validation,
        epochs=(
            config.adam.epochs
        ),
        batch_size=(
            config.adam.batch_size
        ),
        optimizer=optimizer,
        compiled_train_step=(
            compiled_train_step
        ),
        compiled_nll=(
            compiled_nll
        ),
    )

    timing = TimingResult(
        compilation_seconds=(
            first_call_overhead
        ),
        algorithm_seconds=(
            algorithm_time
        ),
    )

    model = training.model

    print(
        "algorithm time       : "
        f"{timing.algorithm_seconds:.3f} s"
    )

    print(
        "first-call/JIT time  : "
        f"{timing.compilation_seconds:.3f} s"
    )

    print(
        "end-to-end time      : "
        f"{timing.end_to_end_seconds:.3f} s"
    )

    print(
        "best epoch           : "
        f"{training.best_epoch}"
    )

    print(
        "best validation NLL  : "
        f"{training.best_validation_nll:.8e}"
    )

    print()

    # ------------------------------------------------------------
    # Final evaluation.
    #
    # This is outside benchmark training time. The test set is first
    # used here.
    # ------------------------------------------------------------

    for label, idx in [
        (
            "train",
            train_idx,
        ),
        (
            "validation",
            validation_idx,
        ),
        (
            "test",
            test_idx,
        ),
    ]:
        nll = float(
            gaussian_nll(
                model,
                data.x[idx],
                data.r[idx],
                data.h[idx],
            )
        )

        (
            drift_rmse,
            covariance_rmse,
        ) = true_function_errors(
            model,
            data.x[idx],
            true_drift=(
                definition.drift
            ),
            true_diffusion_factor=(
                definition.diffusion_factor
            ),
        )

        covariance = np.asarray(
            predict_covariance(
                model,
                data.x[idx],
            )
        )

        min_eigenvalue = np.min(
            np.linalg.eigvalsh(
                covariance
            )
        )

        print(label)

        print(
            "  NLL                : "
            f"{nll:.8e}"
        )

        print(
            "  drift RMSE         : "
            f"{drift_rmse:.8e}"
        )

        print(
            "  covariance RMSE    : "
            f"{covariance_rmse:.8e}"
        )

        print(
            "  min covariance eig : "
            f"{min_eigenvalue:.8e}"
        )

        print()


if __name__ == "__main__":
    main()