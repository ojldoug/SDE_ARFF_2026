#!/usr/bin/env python3
"""Run one canonical capacity-matched Adam MLP SDE experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adam.mlp import (
    gaussian_nll,
    initialize_model,
    predict_covariance,
    predict_mlp,
)
from src.adam.training import (
    fit_adam,
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
from src.experiments.mlp_size import (
    matched_two_layer_width,
)
from src.experiments.model_size import (
    covariance_output_dimension,
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
        predict_mlp(
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


def flatten_mlp_params(
    params,
    *,
    prefix: str,
):
    """
    Convert MLP weights/biases to named NumPy arrays for artifact saving.
    """
    arrays = {}

    for i, weight in enumerate(
        params.weights
    ):
        arrays[
            f"{prefix}_weight_{i}"
        ] = np.asarray(
            jax.device_get(
                weight
            )
        )

    for i, bias in enumerate(
        params.biases
    ):
        arrays[
            f"{prefix}_bias_{i}"
        ] = np.asarray(
            jax.device_get(
                bias
            )
        )

    return arrays


def save_artifact(
    path: Path,
    *,
    experiment: str,
    seed: int,
    diff_type: str,
    hidden_width: int,
    mlp_parameter_count: int,
    fourier_parameter_count: int,
    learning_rate: float,
    epochs: int,
    batch_size: int,
    model,
    training,
    timing,
):
    """
    Save the selected Adam MLP model and full loss-versus-time history.

    Serialization is outside benchmark timing.
    """
    path = path.expanduser().resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    training_nll = np.asarray(
        training.training_nll,
        dtype=np.float64,
    )

    validation_nll = np.asarray(
        training.validation_nll,
        dtype=np.float64,
    )

    cumulative_time = np.asarray(
        training.cumulative_time,
        dtype=np.float64,
    )

    if training_nll.shape != (
        epochs,
    ):
        raise RuntimeError(
            "Unexpected Adam MLP training-NLL "
            "history shape."
        )

    if validation_nll.shape != (
        epochs,
    ):
        raise RuntimeError(
            "Unexpected Adam MLP validation-NLL "
            "history shape."
        )

    if cumulative_time.shape != (
        epochs,
    ):
        raise RuntimeError(
            "Unexpected Adam MLP cumulative-time "
            "history shape."
        )

    for name, array in (
        (
            "training NLL",
            training_nll,
        ),
        (
            "validation NLL",
            validation_nll,
        ),
        (
            "cumulative time",
            cumulative_time,
        ),
    ):
        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise RuntimeError(
                f"{name} contains non-finite values."
            )

    if not np.all(
        np.diff(
            cumulative_time
        )
        >= 0.0
    ):
        raise RuntimeError(
            "Cumulative-time history is not "
            "monotonically non-decreasing."
        )

    if not (
        0
        <= training.best_epoch
        < epochs
    ):
        raise RuntimeError(
            "Best Adam MLP epoch is outside "
            "the stored history."
        )

    history_best_epoch = int(
        np.argmin(
            validation_nll
        )
    )

    if (
        history_best_epoch
        != training.best_epoch
    ):
        raise RuntimeError(
            "Stored validation history and "
            "selected best epoch disagree."
        )

    history_best_nll = float(
        validation_nll[
            training.best_epoch
        ]
    )

    if not np.isclose(
        history_best_nll,
        training.best_validation_nll,
        rtol=1e-7,
        atol=1e-7,
    ):
        raise RuntimeError(
            "Stored validation history and "
            "selected best NLL disagree."
        )

    parameter_arrays = {}

    parameter_arrays.update(
        flatten_mlp_params(
            model.drift,
            prefix="drift",
        )
    )

    parameter_arrays.update(
        flatten_mlp_params(
            model.covariance,
            prefix="covariance",
        )
    )

    for key, array in (
        parameter_arrays.items()
    ):
        if array.size == 0:
            raise RuntimeError(
                f"Empty MLP parameter array: {key}"
            )

        if not np.all(
            np.isfinite(
                array
            )
        ):
            raise RuntimeError(
                f"Non-finite MLP parameter array: {key}"
            )

    np.savez_compressed(
        path,
        artifact_version=np.asarray(
            1,
            dtype=np.int64,
        ),
        method=np.asarray(
            "adam_mlp"
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
        hidden_width=np.asarray(
            hidden_width,
            dtype=np.int64,
        ),
        hidden_layers=np.asarray(
            2,
            dtype=np.int64,
        ),
        mlp_parameter_count=np.asarray(
            mlp_parameter_count,
            dtype=np.int64,
        ),
        fourier_parameter_count=np.asarray(
            fourier_parameter_count,
            dtype=np.int64,
        ),
        epochs=np.asarray(
            epochs,
            dtype=np.int64,
        ),
        batch_size=np.asarray(
            batch_size,
            dtype=np.int64,
        ),
        learning_rate=np.asarray(
            learning_rate,
            dtype=np.float64,
        ),
        best_epoch=np.asarray(
            training.best_epoch,
            dtype=np.int64,
        ),
        best_validation_nll=np.asarray(
            training.best_validation_nll,
            dtype=np.float64,
        ),
        training_nll=training_nll,
        validation_nll=validation_nll,
        cumulative_time=cumulative_time,
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
        **parameter_arrays,
    )

    print(
        "artifact   : "
        f"{path}"
    )


def evaluate_split(
    *,
    label,
    idx,
    model,
    data,
    definition,
):
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
            definition.diffusion_factor
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

    min_eigenvalue = np.min(
        np.linalg.eigvalsh(
            covariance
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

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help=(
            "Optional Adam learning-rate override. "
            "Default: experiment config value."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Optional epoch-count override. "
            "Default: experiment config value."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help=(
            "Optional minibatch-size override. "
            "Default: experiment config value."
        ),
    )

    parser.add_argument(
        "--validation-only",
        action="store_true",
        help=(
            "Evaluate only training and validation "
            "splits after fitting. The test split is "
            "not accessed. Intended for hyperparameter "
            "screening."
        ),
    )

    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=None,
        help=(
            "Optional .npz path for the selected "
            "MLP model and Adam training history. "
            "Artifacts are only permitted for final "
            "non-validation-only runs."
        ),
    )

    args = parser.parse_args()

    if (
        args.validation_only
        and args.artifact_path is not None
    ):
        raise ValueError(
            "--artifact-path cannot be used with "
            "--validation-only."
        )

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

    learning_rate = (
        config.adam_mlp.learning_rate
        if args.learning_rate is None
        else args.learning_rate
    )

    epochs = (
        config.adam_mlp.epochs
        if args.epochs is None
        else args.epochs
    )

    batch_size = (
        config.adam_mlp.batch_size
        if args.batch_size is None
        else args.batch_size
    )

    if learning_rate <= 0.0:
        raise ValueError(
            "learning rate must be positive."
        )

    if epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch size must be positive."
        )

    covariance_dimension = (
        covariance_output_dimension(
            definition.n_dimensions,
            definition.diff_type,
        )
    )

    (
        hidden_width,
        mlp_parameter_count,
        fourier_parameter_count,
    ) = matched_two_layer_width(
        state_dimension=(
            definition.n_dimensions
        ),
        covariance_output_dimension=(
            covariance_dimension
        ),
        n_frequencies=(
            config.fourier_frequencies
        ),
    )

    hidden_sizes = (
        hidden_width,
        hidden_width,
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

    key = jax.random.PRNGKey(
        args.seed
    )

    (
        key,
        initialization_key,
    ) = jax.random.split(
        key
    )

    initial_model = initialize_model(
        initialization_key,
        input_dimension=(
            definition.state_dimension
        ),
        output_dimension=(
            definition.n_dimensions
        ),
        diff_type=(
            definition.diff_type
        ),
        hidden_sizes=(
            hidden_sizes
        ),
    )

    block_until_ready(
        initial_model
    )

    relative_parameter_difference = (
        100.0
        * (
            mlp_parameter_count
            - fourier_parameter_count
        )
        / fourier_parameter_count
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

    if args.validation_only:
        print(
            "test       : NOT USED"
        )
    else:
        print(
            f"test       : {len(test_idx)}"
        )

    print(
        "hidden     : "
        f"{hidden_width}-{hidden_width}"
    )

    print(
        "MLP params : "
        f"{mlp_parameter_count}"
    )

    print(
        "Fourier target params: "
        f"{fourier_parameter_count}"
    )

    print(
        "parameter difference : "
        f"{relative_parameter_difference:+.2f}%"
    )

    print(
        f"epochs     : {epochs}"
    )

    print(
        f"batch size : {batch_size}"
    )

    print(
        "learning rate: "
        f"{learning_rate:.8e}"
    )

    print()

    (
        optimizer,
        compiled_train_step,
        compiled_nll,
    ) = make_compiled_adam_functions(
        learning_rate,
        nll_fn=gaussian_nll,
    )

    initial_opt_state = optimizer.init(
        initial_model
    )

    n_train = len(
        x_train
    )

    full_batch_size = min(
        batch_size,
        n_train,
    )

    _, compile_full_batch = timed_call(
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
        _, compile_remainder = timed_call(
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

    _, compile_validation = timed_call(
        compiled_nll,
        initial_model,
        x_validation,
        r_validation,
        h_validation,
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

    (
        (
            key,
            training,
        ),
        algorithm_time,
    ) = timed_call(
        fit_adam,
        key,
        initial_model,
        x_train,
        r_train,
        h_train,
        x_validation,
        r_validation,
        h_validation,
        epochs=epochs,
        batch_size=batch_size,
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
    # During hyperparameter screening, the test split is never touched.
    # ------------------------------------------------------------

    evaluate_split(
        label="train",
        idx=train_idx,
        model=model,
        data=data,
        definition=definition,
    )

    evaluate_split(
        label="validation",
        idx=validation_idx,
        model=model,
        data=data,
        definition=definition,
    )

    if not args.validation_only:
        evaluate_split(
            label="test",
            idx=test_idx,
            model=model,
            data=data,
            definition=definition,
        )

    if args.artifact_path is not None:
        save_artifact(
            args.artifact_path,
            experiment=name,
            seed=args.seed,
            diff_type=(
                definition.diff_type
            ),
            hidden_width=(
                hidden_width
            ),
            mlp_parameter_count=(
                mlp_parameter_count
            ),
            fourier_parameter_count=(
                fourier_parameter_count
            ),
            learning_rate=(
                learning_rate
            ),
            epochs=epochs,
            batch_size=batch_size,
            model=model,
            training=training,
            timing=timing,
        )


if __name__ == "__main__":
    main()