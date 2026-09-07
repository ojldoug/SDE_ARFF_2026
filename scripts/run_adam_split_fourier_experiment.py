#!/usr/bin/env python3
"""
Run one canonical split Fourier-Adam SDE experiment.

Compilation timing is separated from algorithm timing.

Before real training starts, the runner executes the drift and
covariance JIT functions once for every distinct minibatch shape that
the split procedure can encounter, and once on the canonical validation
shape.

Real split training then starts from fresh model initializations and
fresh Adam optimizer states.

Reported algorithm time therefore excludes one-time JAX/XLA compilation
but includes all work required by the split procedure:

    five cross-fit drift regressions,
    one final all-training drift regression,
    one covariance-likelihood regression,
    construction of out-of-fold residuals,
    and validation-based checkpoint selection.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(REPO_ROOT),
)

from src.adam.fourier import (
    gaussian_nll,
    initialize_model,
    predict_covariance,
    predict_fourier,
)
from src.adam.split_fourier import (
    CovarianceFourierModel,
    covariance_gaussian_nll,
    drift_mse,
    fit_split_fourier_adam,
)
from src.adam.training import (
    make_compiled_adam_functions,
)
from src.arff.two_stage import (
    make_folds,
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


# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------


def true_function_errors(
    model,
    x,
    *,
    true_drift,
    true_diffusion_factor,
):
    x = np.asarray(
        x
    )

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
        float(
            drift_rmse
        ),
        float(
            covariance_rmse
        ),
    )


# ------------------------------------------------------------------
# Compilation warm-up
# ------------------------------------------------------------------


def required_drift_batch_sizes(
    *,
    n_train: int,
    n_folds: int,
    fold_seed: int,
    batch_size: int,
):
    """
    Return every distinct minibatch size encountered by cross-fit and
    final drift training.
    """
    folds = make_folds(
        n_train,
        n_folds,
        fold_seed,
    )

    fit_sizes = [
        n_train
        - len(
            holdout
        )
        for holdout in folds
    ]

    training_sizes = (
        fit_sizes
        + [
            n_train,
        ]
    )

    batch_sizes = set()

    for n_samples in training_sizes:
        full = min(
            batch_size,
            n_samples,
        )

        batch_sizes.add(
            full
        )

        remainder = (
            n_samples
            % batch_size
        )

        if remainder > 0:
            batch_sizes.add(
                remainder
            )

    return sorted(
        batch_sizes
    )


def required_covariance_batch_sizes(
    *,
    n_train: int,
    batch_size: int,
):
    """
    Stage-2 covariance training uses the complete canonical train set.
    """
    sizes = {
        min(
            batch_size,
            n_train,
        ),
    }

    remainder = (
        n_train
        % batch_size
    )

    if remainder > 0:
        sizes.add(
            remainder
        )

    return sorted(
        sizes
    )


def compile_one_train_shape(
    *,
    compiled_train_step,
    optimizer,
    initial_model,
    x,
    target,
    h,
):
    opt_state = optimizer.init(
        initial_model
    )

    _, elapsed = timed_call(
        compiled_train_step,
        initial_model,
        opt_state,
        x,
        target,
        h,
    )

    return elapsed


def prepare_compiled_functions(
    *,
    warmup_key,
    x_train,
    r_train,
    h_train,
    x_validation,
    r_validation,
    h_validation,
    input_dimension: int,
    output_dimension: int,
    n_frequencies: int,
    diff_type: str,
    n_folds: int,
    fold_seed: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
):
    """
    Compile every distinct update/loss executable needed by the split
    procedure.

    The warm-up model and optimizer states are discarded afterward.
    """
    del epochs

    (
        warmup_key,
        initialization_key,
    ) = jax.random.split(
        warmup_key
    )

    warmup_joint_model = initialize_model(
        initialization_key,
        input_dimension=(
            input_dimension
        ),
        output_dimension=(
            output_dimension
        ),
        n_frequencies=(
            n_frequencies
        ),
        diff_type=diff_type,
    )

    warmup_covariance_model = (
        CovarianceFourierModel(
            covariance=(
                warmup_joint_model
                .covariance
            ),
            diff_type=diff_type,
            output_dimension=(
                output_dimension
            ),
        )
    )

    block_until_ready(
        (
            warmup_joint_model,
            warmup_covariance_model,
        )
    )

    (
        drift_optimizer,
        drift_compiled_train_step,
        drift_compiled_loss,
    ) = make_compiled_adam_functions(
        learning_rate,
        nll_fn=drift_mse,
    )

    (
        covariance_optimizer,
        covariance_compiled_train_step,
        covariance_compiled_loss,
    ) = make_compiled_adam_functions(
        learning_rate,
        nll_fn=(
            covariance_gaussian_nll
        ),
    )

    compilation_seconds = 0.0

    # --------------------------------------------------------------
    # Drift update shapes.
    # --------------------------------------------------------------

    drift_sizes = (
        required_drift_batch_sizes(
            n_train=len(
                x_train
            ),
            n_folds=n_folds,
            fold_seed=fold_seed,
            batch_size=batch_size,
        )
    )

    for size in drift_sizes:
        dummy_target = jnp.zeros_like(
            r_train[
                :size
            ]
        )

        elapsed = (
            compile_one_train_shape(
                compiled_train_step=(
                    drift_compiled_train_step
                ),
                optimizer=(
                    drift_optimizer
                ),
                initial_model=(
                    warmup_joint_model
                    .drift
                ),
                x=x_train[
                    :size
                ],
                target=(
                    dummy_target
                ),
                h=h_train[
                    :size
                ],
            )
        )

        compilation_seconds += (
            elapsed
        )

        print(
            "  drift train shape "
            f"N={size}: "
            f"{elapsed:.3f} s"
        )

    # --------------------------------------------------------------
    # Drift validation shape.
    # --------------------------------------------------------------

    drift_validation_target = (
        r_validation
        / h_validation
    )

    _, elapsed = timed_call(
        drift_compiled_loss,
        warmup_joint_model.drift,
        x_validation,
        drift_validation_target,
        h_validation,
    )

    compilation_seconds += (
        elapsed
    )

    print(
        "  drift validation "
        f"N={len(x_validation)}: "
        f"{elapsed:.3f} s"
    )

    # --------------------------------------------------------------
    # Covariance update shapes.
    #
    # Zero residuals are sufficient for compilation and avoid using a
    # noisy warm-up target. Warm-up states are discarded.
    # --------------------------------------------------------------

    covariance_sizes = (
        required_covariance_batch_sizes(
            n_train=len(
                x_train
            ),
            batch_size=batch_size,
        )
    )

    for size in covariance_sizes:
        dummy_residual = (
            jnp.zeros_like(
                r_train[
                    :size
                ]
            )
        )

        elapsed = (
            compile_one_train_shape(
                compiled_train_step=(
                    covariance_compiled_train_step
                ),
                optimizer=(
                    covariance_optimizer
                ),
                initial_model=(
                    warmup_covariance_model
                ),
                x=x_train[
                    :size
                ],
                target=(
                    dummy_residual
                ),
                h=h_train[
                    :size
                ],
            )
        )

        compilation_seconds += (
            elapsed
        )

        print(
            "  covariance train shape "
            f"N={size}: "
            f"{elapsed:.3f} s"
        )

    # --------------------------------------------------------------
    # Covariance validation shape.
    # --------------------------------------------------------------

    dummy_validation_residual = (
        jnp.zeros_like(
            r_validation
        )
    )

    _, elapsed = timed_call(
        covariance_compiled_loss,
        warmup_covariance_model,
        x_validation,
        dummy_validation_residual,
        h_validation,
    )

    compilation_seconds += (
        elapsed
    )

    print(
        "  covariance validation "
        f"N={len(x_validation)}: "
        f"{elapsed:.3f} s"
    )

    return (
        drift_optimizer,
        drift_compiled_train_step,
        drift_compiled_loss,
        covariance_optimizer,
        covariance_compiled_train_step,
        covariance_compiled_loss,
        compilation_seconds,
    )


# ------------------------------------------------------------------
# Artifact
# ------------------------------------------------------------------


def save_artifact(
    path: Path,
    *,
    experiment: str,
    seed: int,
    diff_type: str,
    result,
    timing,
    config,
):
    path = (
        path
        .expanduser()
        .resolve()
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    drift_training = (
        result.final_drift_training
    )

    covariance_training = (
        result.covariance_training
    )

    np.savez_compressed(
        path,

        artifact_version=np.asarray(
            3,
            dtype=np.int64,
        ),

        method=np.asarray(
            "adam_split_fourier"
        ),

        experiment=np.asarray(
            experiment
        ),

        seed=np.asarray(
            seed,
            dtype=np.int64,
        ),

        diff_type=np.asarray(
            diff_type
        ),

        fourier_frequencies=np.asarray(
            config.fourier_frequencies,
            dtype=np.int64,
        ),

        n_folds=np.asarray(
            config.arff.n_folds,
            dtype=np.int64,
        ),

        fold_seed=np.asarray(
            config.split.seed,
            dtype=np.int64,
        ),

        epochs_per_regression=np.asarray(
            config.adam.epochs,
            dtype=np.int64,
        ),

        batch_size=np.asarray(
            config.adam.batch_size,
            dtype=np.int64,
        ),

        learning_rate=np.asarray(
            config.adam.learning_rate,
            dtype=np.float64,
        ),

        algorithm_time=np.asarray(
            timing.algorithm_seconds,
            dtype=np.float64,
        ),

        compilation_time=np.asarray(
            timing.compilation_seconds,
            dtype=np.float64,
        ),

        end_to_end_time=np.asarray(
            timing.end_to_end_seconds,
            dtype=np.float64,
        ),

        final_drift_best_epoch=np.asarray(
            drift_training.best_epoch,
            dtype=np.int64,
        ),

        final_drift_best_validation_mse=(
            np.asarray(
                drift_training
                .best_validation_nll,
                dtype=np.float64,
            )
        ),

        final_drift_training_mse=np.asarray(
            drift_training.training_nll,
            dtype=np.float64,
        ),

        final_drift_validation_mse=np.asarray(
            drift_training.validation_nll,
            dtype=np.float64,
        ),

        final_drift_cumulative_time=np.asarray(
            drift_training.cumulative_time,
            dtype=np.float64,
        ),

        covariance_best_epoch=np.asarray(
            covariance_training.best_epoch,
            dtype=np.int64,
        ),

        covariance_best_validation_nll=(
            np.asarray(
                covariance_training
                .best_validation_nll,
                dtype=np.float64,
            )
        ),

        covariance_training_nll=np.asarray(
            covariance_training.training_nll,
            dtype=np.float64,
        ),

        covariance_validation_nll=np.asarray(
            covariance_training.validation_nll,
            dtype=np.float64,
        ),

        covariance_cumulative_time=np.asarray(
            covariance_training.cumulative_time,
            dtype=np.float64,
        ),

        fold_best_epochs=np.asarray(
            result
            .crossfit
            .fold_best_epochs,
            dtype=np.int64,
        ),

        fold_best_validation_mse=np.asarray(
            result
            .crossfit
            .fold_best_validation_losses,
            dtype=np.float64,
        ),

        fold_id=np.asarray(
            result.crossfit.fold_id,
            dtype=np.int32,
        ),

        drift_omega=np.asarray(
            jax.device_get(
                result.model.drift.omega
            )
        ),

        drift_amp=np.asarray(
            jax.device_get(
                result.model.drift.amp
            )
        ),

        covariance_omega=np.asarray(
            jax.device_get(
                result
                .model
                .covariance
                .omega
            )
        ),

        covariance_amp=np.asarray(
            jax.device_get(
                result
                .model
                .covariance
                .amp
            )
        ),
    )

    print(
        "artifact   : "
        f"{path}"
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "experiment",
        choices=[
            f"ex{i}"
            for i in range(
                1,
                9,
            )
        ],
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=None,
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
            "No Fourier frequency count "
            f"for {name}."
        )

    data = load_dataset(
        REPO_ROOT
        / "data"
        / f"{name}.npz"
    )

    train_idx = (
        data.train_idx
    )

    validation_idx = (
        data.validation_idx
    )

    test_idx = (
        data.test_idx
    )

    # --------------------------------------------------------------
    # Transfer canonical train/validation arrays before either
    # compilation timing or algorithm timing.
    # --------------------------------------------------------------

    x_train = jnp.asarray(
        data.x[
            train_idx
        ]
    )

    r_train = jnp.asarray(
        data.r[
            train_idx
        ]
    )

    h_train = jnp.asarray(
        data.h[
            train_idx
        ]
    )

    x_validation = jnp.asarray(
        data.x[
            validation_idx
        ]
    )

    r_validation = jnp.asarray(
        data.r[
            validation_idx
        ]
    )

    h_validation = jnp.asarray(
        data.h[
            validation_idx
        ]
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
        "folds      : "
        f"{config.arff.n_folds}"
    )

    print(
        "fold seed  : "
        f"{config.split.seed}"
    )

    print(
        "epochs/regression: "
        f"{config.adam.epochs}"
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

    # --------------------------------------------------------------
    # Compile every required executable with a completely independent
    # warm-up key.
    # --------------------------------------------------------------

    print(
        "Split Adam first-call/JIT warm-up"
    )

    warmup_key = jax.random.PRNGKey(
        987654321
    )

    (
        drift_optimizer,
        drift_compiled_train_step,
        drift_compiled_loss,
        covariance_optimizer,
        covariance_compiled_train_step,
        covariance_compiled_loss,
        compilation_time,
    ) = prepare_compiled_functions(
        warmup_key=warmup_key,
        x_train=x_train,
        r_train=r_train,
        h_train=h_train,
        x_validation=x_validation,
        r_validation=r_validation,
        h_validation=h_validation,
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
        n_folds=(
            config.arff.n_folds
        ),
        fold_seed=(
            config.split.seed
        ),
        epochs=(
            config.adam.epochs
        ),
        batch_size=(
            config.adam.batch_size
        ),
        learning_rate=(
            config.adam.learning_rate
        ),
    )

    print(
        "first-call/JIT overhead: "
        f"{compilation_time:.3f} s"
    )

    print()

    # --------------------------------------------------------------
    # Real training starts from the requested seed and from fresh Adam
    # states. All required JIT shapes have already been compiled.
    # --------------------------------------------------------------

    key = jax.random.PRNGKey(
        args.seed
    )

    (
        (
            key,
            result,
        ),
        algorithm_time,
    ) = timed_call(
        fit_split_fourier_adam,
        key,
        x_train,
        r_train,
        h_train,
        x_validation,
        r_validation,
        h_validation,
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
        n_folds=(
            config.arff.n_folds
        ),
        fold_seed=(
            config.split.seed
        ),
        epochs=(
            config.adam.epochs
        ),
        batch_size=(
            config.adam.batch_size
        ),
        drift_optimizer=(
            drift_optimizer
        ),
        drift_compiled_train_step=(
            drift_compiled_train_step
        ),
        drift_compiled_loss=(
            drift_compiled_loss
        ),
        covariance_optimizer=(
            covariance_optimizer
        ),
        covariance_compiled_train_step=(
            covariance_compiled_train_step
        ),
        covariance_compiled_loss=(
            covariance_compiled_loss
        ),
    )

    timing = TimingResult(
        compilation_seconds=(
            compilation_time
        ),
        algorithm_seconds=(
            algorithm_time
        ),
    )

    model = result.model

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
        "final drift best epoch: "
        f"{result.final_drift_training.best_epoch}"
    )

    print(
        "final drift best validation MSE: "
        f"{result.final_drift_training.best_validation_nll:.8e}"
    )

    print(
        "covariance best epoch : "
        f"{result.covariance_training.best_epoch}"
    )

    print(
        "covariance best validation NLL: "
        f"{result.covariance_training.best_validation_nll:.8e}"
    )

    print(
        "fold best epochs      : "
        f"{result.crossfit.fold_best_epochs.tolist()}"
    )

    print()

    # --------------------------------------------------------------
    # Final evaluation. Test is first used here.
    # --------------------------------------------------------------

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
                data.x[
                    idx
                ],
                data.r[
                    idx
                ],
                data.h[
                    idx
                ],
            )
        )

        (
            drift_rmse,
            covariance_rmse,
        ) = true_function_errors(
            model,
            data.x[
                idx
            ],
            true_drift=(
                definition.drift
            ),
            true_diffusion_factor=(
                definition
                .diffusion_factor
            ),
        )

        covariance = np.asarray(
            predict_covariance(
                model,
                data.x[
                    idx
                ],
            )
        )

        min_eigenvalue = float(
            np.min(
                np.linalg.eigvalsh(
                    covariance
                )
            )
        )

        print(
            label
        )

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

    if (
        args.artifact_path
        is not None
    ):
        save_artifact(
            args.artifact_path,
            experiment=name,
            seed=args.seed,
            diff_type=(
                definition.diff_type
            ),
            result=result,
            timing=timing,
            config=config,
        )


if __name__ == "__main__":
    main()
