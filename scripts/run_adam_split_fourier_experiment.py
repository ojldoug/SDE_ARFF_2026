#!/usr/bin/env python3
"""
Run one canonical split Fourier-Adam SDE experiment.

Compilation timing is separated from algorithm timing.

The artifact is deliberately archival-quality. It stores the final
model, endpoint metrics, optimization histories, cross-fit diagnostics,
stage timings, and metadata needed for later plotting and analysis.

Expensive diagnostic work is not inserted into training. Final
evaluation and artifact serialization occur only after algorithm timing
has stopped.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import platform
import subprocess
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


def evaluate_split(
    model,
    x,
    r,
    h,
    *,
    true_drift,
    true_diffusion_factor,
):
    """
    Evaluate one fitted model.

    This is called only after algorithm timing has stopped.
    """
    x = np.asarray(
        x
    )

    r = np.asarray(
        r
    )

    h = np.asarray(
        h
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

    nll = float(
        gaussian_nll(
            model,
            x,
            r,
            h,
        )
    )

    drift_rmse = float(
        np.sqrt(
            np.mean(
                (
                    learned_drift
                    - drift_truth
                ) ** 2
            )
        )
    )

    covariance_rmse = float(
        np.sqrt(
            np.mean(
                (
                    learned_covariance
                    - covariance_truth
                ) ** 2
            )
        )
    )

    eigenvalues = np.linalg.eigvalsh(
        learned_covariance
    )

    min_covariance_eig = float(
        np.min(
            eigenvalues
        )
    )

    max_covariance_eig = float(
        np.max(
            eigenvalues
        )
    )

    return {
        "nll": nll,
        "drift_rmse": drift_rmse,
        "covariance_rmse": (
            covariance_rmse
        ),
        "min_covariance_eig": (
            min_covariance_eig
        ),
        "max_covariance_eig": (
            max_covariance_eig
        ),
    }


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

    sizes = set()

    for n_samples in training_sizes:
        sizes.add(
            min(
                batch_size,
                n_samples,
            )
        )

        remainder = (
            n_samples
            % batch_size
        )

        if remainder > 0:
            sizes.add(
                remainder
            )

    return sorted(
        sizes
    )


def required_covariance_batch_sizes(
    *,
    n_train: int,
    batch_size: int,
):
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
    batch_size: int,
    learning_rate: float,
):
    """
    Compile every distinct executable shape needed by real training.

    Warm-up models and optimizer states are discarded afterward.
    """
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
# Cheap reproducibility metadata
# ------------------------------------------------------------------


def git_metadata():
    try:
        commit = subprocess.run(
            [
                "git",
                "rev-parse",
                "HEAD",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()

        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout

        dirty = bool(
            status.strip()
        )

        return (
            commit,
            dirty,
        )

    except (
        OSError,
        subprocess.SubprocessError,
    ):
        return (
            "unknown",
            True,
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
    train_metrics,
    validation_metrics,
    test_metrics,
    n_train: int,
    n_validation: int,
    n_test: int,
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

    crossfit = (
        result.crossfit
    )

    final_drift_local_time = (
        np.asarray(
            drift_training.cumulative_time,
            dtype=np.float64,
        )
    )

    covariance_local_time = (
        np.asarray(
            covariance_training.cumulative_time,
            dtype=np.float64,
        )
    )

    final_drift_global_time = (
        result.final_drift_start_offset
        + final_drift_local_time
    )

    covariance_global_time = (
        result.covariance_start_offset
        + covariance_local_time
    )

    (
        git_commit,
        git_dirty,
    ) = git_metadata()

    np.savez_compressed(
        path,

        # ----------------------------------------------------------
        # Identity and schema
        # ----------------------------------------------------------

        artifact_version=np.asarray(
            4,
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

        # ----------------------------------------------------------
        # Environment / provenance
        # ----------------------------------------------------------

        git_commit=np.asarray(
            git_commit
        ),

        git_dirty=np.asarray(
            git_dirty
        ),

        hostname=np.asarray(
            platform.node()
        ),

        python_version=np.asarray(
            platform.python_version()
        ),

        numpy_version=np.asarray(
            np.__version__
        ),

        jax_version=np.asarray(
            jax.__version__
        ),

        jax_backend=np.asarray(
            jax.default_backend()
        ),

        # ----------------------------------------------------------
        # Dataset sizes
        # ----------------------------------------------------------

        n_train=np.asarray(
            n_train,
            dtype=np.int64,
        ),

        n_validation=np.asarray(
            n_validation,
            dtype=np.int64,
        ),

        n_test=np.asarray(
            n_test,
            dtype=np.int64,
        ),

        # ----------------------------------------------------------
        # Hyperparameters
        # ----------------------------------------------------------

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

        # ----------------------------------------------------------
        # Top-level timing
        # ----------------------------------------------------------

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

        internal_algorithm_time=np.asarray(
            result.internal_algorithm_time,
            dtype=np.float64,
        ),

        # ----------------------------------------------------------
        # Cross-fit timing / diagnostics
        # ----------------------------------------------------------

        fold_algorithm_times=np.asarray(
            crossfit.fold_algorithm_times,
            dtype=np.float64,
        ),

        crossfit_algorithm_time=np.asarray(
            crossfit.crossfit_algorithm_time,
            dtype=np.float64,
        ),

        fold_best_epochs=np.asarray(
            crossfit.fold_best_epochs,
            dtype=np.int64,
        ),

        fold_best_validation_mse=np.asarray(
            crossfit
            .fold_best_validation_losses,
            dtype=np.float64,
        ),

        fold_id=np.asarray(
            crossfit.fold_id,
            dtype=np.int32,
        ),

        # ----------------------------------------------------------
        # Final drift stage
        # ----------------------------------------------------------

        final_drift_start_offset=np.asarray(
            result.final_drift_start_offset,
            dtype=np.float64,
        ),

        final_drift_algorithm_time=np.asarray(
            result.final_drift_algorithm_time,
            dtype=np.float64,
        ),

        final_drift_end_offset=np.asarray(
            (
                result.final_drift_start_offset
                + result.final_drift_algorithm_time
            ),
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

        final_drift_cumulative_time=(
            final_drift_local_time
        ),

        final_drift_global_cumulative_time=(
            final_drift_global_time
        ),

        # ----------------------------------------------------------
        # Covariance stage
        # ----------------------------------------------------------

        covariance_start_offset=np.asarray(
            result.covariance_start_offset,
            dtype=np.float64,
        ),

        covariance_algorithm_time=np.asarray(
            result.covariance_algorithm_time,
            dtype=np.float64,
        ),

        covariance_end_offset=np.asarray(
            (
                result.covariance_start_offset
                + result.covariance_algorithm_time
            ),
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

        covariance_cumulative_time=(
            covariance_local_time
        ),

        covariance_global_cumulative_time=(
            covariance_global_time
        ),

        # ----------------------------------------------------------
        # Endpoint train metrics
        # ----------------------------------------------------------

        train_nll=np.asarray(
            train_metrics[
                "nll"
            ],
            dtype=np.float64,
        ),

        train_drift_rmse=np.asarray(
            train_metrics[
                "drift_rmse"
            ],
            dtype=np.float64,
        ),

        train_covariance_rmse=np.asarray(
            train_metrics[
                "covariance_rmse"
            ],
            dtype=np.float64,
        ),

        train_min_covariance_eig=np.asarray(
            train_metrics[
                "min_covariance_eig"
            ],
            dtype=np.float64,
        ),

        train_max_covariance_eig=np.asarray(
            train_metrics[
                "max_covariance_eig"
            ],
            dtype=np.float64,
        ),

        # ----------------------------------------------------------
        # Endpoint validation metrics
        # ----------------------------------------------------------

        validation_nll=np.asarray(
            validation_metrics[
                "nll"
            ],
            dtype=np.float64,
        ),

        validation_drift_rmse=np.asarray(
            validation_metrics[
                "drift_rmse"
            ],
            dtype=np.float64,
        ),

        validation_covariance_rmse=np.asarray(
            validation_metrics[
                "covariance_rmse"
            ],
            dtype=np.float64,
        ),

        validation_min_covariance_eig=np.asarray(
            validation_metrics[
                "min_covariance_eig"
            ],
            dtype=np.float64,
        ),

        validation_max_covariance_eig=np.asarray(
            validation_metrics[
                "max_covariance_eig"
            ],
            dtype=np.float64,
        ),

        # ----------------------------------------------------------
        # Endpoint test metrics
        # ----------------------------------------------------------

        test_nll=np.asarray(
            test_metrics[
                "nll"
            ],
            dtype=np.float64,
        ),

        test_drift_rmse=np.asarray(
            test_metrics[
                "drift_rmse"
            ],
            dtype=np.float64,
        ),

        test_covariance_rmse=np.asarray(
            test_metrics[
                "covariance_rmse"
            ],
            dtype=np.float64,
        ),

        test_min_covariance_eig=np.asarray(
            test_metrics[
                "min_covariance_eig"
            ],
            dtype=np.float64,
        ),

        test_max_covariance_eig=np.asarray(
            test_metrics[
                "max_covariance_eig"
            ],
            dtype=np.float64,
        ),

        # ----------------------------------------------------------
        # Complete final model parameters
        # ----------------------------------------------------------

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
    # JIT warm-up.
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
    # Real split training.
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
        "internal algorithm   : "
        f"{result.internal_algorithm_time:.3f} s"
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
        "cross-fit time       : "
        f"{result.crossfit.crossfit_algorithm_time:.3f} s"
    )

    print(
        "fold times           : "
        f"{result.crossfit.fold_algorithm_times.tolist()}"
    )

    print(
        "final drift start    : "
        f"{result.final_drift_start_offset:.3f} s"
    )

    print(
        "final drift time     : "
        f"{result.final_drift_algorithm_time:.3f} s"
    )

    print(
        "covariance start     : "
        f"{result.covariance_start_offset:.3f} s"
    )

    print(
        "covariance time      : "
        f"{result.covariance_algorithm_time:.3f} s"
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
    # Final evaluation occurs after benchmark timing has stopped.
    # --------------------------------------------------------------

    metrics = {}

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
        values = evaluate_split(
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
            true_drift=(
                definition.drift
            ),
            true_diffusion_factor=(
                definition
                .diffusion_factor
            ),
        )

        metrics[
            label
        ] = values

        print(
            label
        )

        print(
            "  NLL                : "
            f"{values['nll']:.8e}"
        )

        print(
            "  drift RMSE         : "
            f"{values['drift_rmse']:.8e}"
        )

        print(
            "  covariance RMSE    : "
            f"{values['covariance_rmse']:.8e}"
        )

        print(
            "  min covariance eig : "
            f"{values['min_covariance_eig']:.8e}"
        )

        print(
            "  max covariance eig : "
            f"{values['max_covariance_eig']:.8e}"
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
            train_metrics=(
                metrics[
                    "train"
                ]
            ),
            validation_metrics=(
                metrics[
                    "validation"
                ]
            ),
            test_metrics=(
                metrics[
                    "test"
                ]
            ),
            n_train=len(
                train_idx
            ),
            n_validation=len(
                validation_idx
            ),
            n_test=len(
                test_idx
            ),
        )


if __name__ == "__main__":
    main()
