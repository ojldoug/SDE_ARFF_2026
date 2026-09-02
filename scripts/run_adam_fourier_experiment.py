#!/usr/bin/env python3
"""Run one canonical Adam Fourier SDE experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
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
from src.experiments.config import get_config
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment
from src.experiments.timing import (
    TimingResult,
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
        true_drift(x)
    )

    sigma_truth = np.asarray(
        true_diffusion_factor(x)
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
            (learned_drift - drift_truth) ** 2
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

    return drift_rmse, covariance_rmse


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "experiment",
        choices=[f"ex{i}" for i in range(1, 9)],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    name = args.experiment

    config = get_config(name)
    definition = get_experiment(name)

    if config.fourier_frequencies is None:
        raise ValueError(
            "No Fourier frequency count has been "
            f"established for {name}."
        )

    data = load_dataset(
        REPO_ROOT / "data" / f"{name}.npz"
    )

    train_idx = data.train_idx
    validation_idx = data.validation_idx
    test_idx = data.test_idx

    key = jax.random.PRNGKey(args.seed)

    key, initialization_key = jax.random.split(
        key
    )

    initial_model = initialize_model(
        initialization_key,
        input_dimension=definition.state_dimension,
        output_dimension=definition.n_dimensions,
        n_frequencies=config.fourier_frequencies,
        diff_type=definition.diff_type,
    )

    print(f"Experiment : {name}")
    print(f"seed       : {args.seed}")
    print(f"backend    : {jax.default_backend()}")
    print(f"train N    : {len(train_idx)}")
    print(f"validation : {len(validation_idx)}")
    print(f"test       : {len(test_idx)}")
    print(
        "frequencies: "
        f"{config.fourier_frequencies}"
    )
    print(f"epochs     : {config.adam.epochs}")
    print(f"batch size : {config.adam.batch_size}")
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
    # Warm up every array shape used during the real training run.
    # Warm-up parameter updates are discarded.
    # ------------------------------------------------------------

    batch_size = config.adam.batch_size
    n_train = len(train_idx)

    full_batch_size = min(
        batch_size,
        n_train,
    )

    full_idx = train_idx[
        :full_batch_size
    ]

    _, compile_full_batch = timed_call(
        compiled_train_step,
        initial_model,
        initial_opt_state,
        data.x[full_idx],
        data.r[full_idx],
        data.h[full_idx],
    )

    remainder = n_train % batch_size
    compile_remainder = 0.0

    if (
        remainder > 0
        and remainder != full_batch_size
    ):
        remainder_idx = train_idx[
            :remainder
        ]

        _, compile_remainder = timed_call(
            compiled_train_step,
            initial_model,
            initial_opt_state,
            data.x[remainder_idx],
            data.r[remainder_idx],
            data.h[remainder_idx],
        )

    _, compile_validation = timed_call(
        compiled_nll,
        initial_model,
        data.x[validation_idx],
        data.r[validation_idx],
        data.h[validation_idx],
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
        f"  full batch       : {compile_full_batch:.3f} s"
    )

    if compile_remainder > 0.0:
        print(
            f"  remainder batch  : {compile_remainder:.3f} s"
        )

    print(
        f"  validation       : {compile_validation:.3f} s"
    )
    print()

    # ------------------------------------------------------------
    # Real training.
    #
    # fit_adam_fourier starts from the untouched initial model and
    # creates a fresh optimizer state. The compiled functions have
    # already been warmed up above.
    # ------------------------------------------------------------

    (key, training), algorithm_time = timed_call(
        fit_adam_fourier,
        key,
        initial_model,
        data.x[train_idx],
        data.r[train_idx],
        data.h[train_idx],
        data.x[validation_idx],
        data.r[validation_idx],
        data.h[validation_idx],
        epochs=config.adam.epochs,
        batch_size=config.adam.batch_size,
        optimizer=optimizer,
        compiled_train_step=compiled_train_step,
        compiled_nll=compiled_nll,
    )

    timing = TimingResult(
        compilation_seconds=first_call_overhead,
        algorithm_seconds=algorithm_time,
    )

    model = training.model

    print(
        f"algorithm time       : "
        f"{timing.algorithm_seconds:.3f} s"
    )
    print(
        f"first-call/JIT time  : "
        f"{timing.compilation_seconds:.3f} s"
    )
    print(
        f"end-to-end time      : "
        f"{timing.end_to_end_seconds:.3f} s"
    )
    print(
        f"best epoch           : {training.best_epoch}"
    )
    print(
        "best validation NLL  : "
        f"{training.best_validation_nll:.8e}"
    )
    print()

    # ------------------------------------------------------------
    # Final evaluation.
    #
    # This happens after timing and does not contribute to training
    # time. The test set is first used here.
    # ------------------------------------------------------------

    for label, idx in [
        ("train", train_idx),
        ("validation", validation_idx),
        ("test", test_idx),
    ]:
        nll = float(
            gaussian_nll(
                model,
                data.x[idx],
                data.r[idx],
                data.h[idx],
            )
        )

        drift_rmse, covariance_rmse = (
            true_function_errors(
                model,
                data.x[idx],
                true_drift=definition.drift,
                true_diffusion_factor=(
                    definition.diffusion_factor
                ),
            )
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
            f"  NLL                : {nll:.8e}"
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