#!/usr/bin/env python3
"""Run one canonical two-stage ARFF SDE experiment."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.arff.evaluation import (
    gaussian_nll,
    true_function_errors,
)
from src.arff.regression import (
    fit_arff,
    make_compiled_adaptation_step,
)
from src.arff.two_stage import (
    fit_two_stage_arff,
    make_folds,
)
from src.experiments.config import get_config
from src.experiments.dataset import (
    load_dataset,
    validate_split_indices,
)
from src.experiments.definitions import get_experiment
from src.experiments.model_size import (
    covariance_output_dimension,
)
from src.experiments.timing import (
    TimingResult,
    block_until_ready,
    timed_call,
)


def warm_arff_shape(
    key,
    compiled_adaptation_step,
    x,
    *,
    output_dimension: int,
    K: int,
    config,
):
    """
    Warm the ARFF regression path for one (N, output_dimension) shape.

    This executes a one-iteration ARFF fit. The returned model and key
    are discarded by the benchmark runner.

    Warming fit_arff rather than only the compiled adaptation step also
    exercises the initial ridge fit that precedes adaptation.
    """
    y = jnp.zeros(
        (
            len(x),
            output_dimension,
        ),
        dtype=x.dtype,
    )

    return fit_arff(
        key,
        x,
        y,
        K=K,
        n_iterations=1,
        lambda_reg=config.lambda_reg,
        gamma=config.gamma,
        delta=config.delta,
        resampling=config.resampling,
        metropolis_test=config.metropolis_test,
        compiled_adaptation_step=(
            compiled_adaptation_step
        ),
    )


def block_two_stage_result(result):
    """
    Synchronize all numerical outputs of a two-stage ARFF fit.
    """
    key, model, crossfit = result

    block_until_ready(
        (
            key,
            model.drift.omega,
            model.drift.amp,
            model.covariance.omega,
            model.covariance.amp,
            crossfit.covariance_targets,
        )
    )


def run_two_stage_synchronized(
    key,
    x,
    r,
    h,
    *,
    K,
    diff_type,
    config,
    fold_seed,
    compiled_adaptation_step,
):
    """
    Execute and synchronize one complete two-stage ARFF training run.
    """
    result = fit_two_stage_arff(
        key,
        x,
        r,
        h,
        K=K,
        diff_type=diff_type,
        config=config,
        fold_seed=fold_seed,
        compiled_adaptation_step=(
            compiled_adaptation_step
        ),
    )

    block_two_stage_result(
        result
    )

    return result


def save_artifact(
    path: Path,
    *,
    experiment: str,
    seed: int,
    diff_type: str,
    model,
    timing,
    config,
):
    """
    Save the final two-stage ARFF model and its reproducibility metadata.

    ARFF contributes one final loss/time point per run to the canonical
    paper comparison, so no artificial per-iteration history is stored.

    Artifact serialization is performed outside benchmark timing.
    """
    path = path.expanduser().resolve()

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    drift_omega = np.asarray(
        jax.device_get(
            model.drift.omega
        )
    )

    drift_amp = np.asarray(
        jax.device_get(
            model.drift.amp
        )
    )

    covariance_omega = np.asarray(
        jax.device_get(
            model.covariance.omega
        )
    )

    covariance_amp = np.asarray(
        jax.device_get(
            model.covariance.amp
        )
    )

    arrays = (
        drift_omega,
        drift_amp,
        covariance_omega,
        covariance_amp,
    )

    if not all(
        np.all(np.isfinite(array))
        for array in arrays
    ):
        raise RuntimeError(
            "ARFF model artifact contains "
            "non-finite parameters."
        )

    if drift_omega.shape[1] != (
        config.fourier_frequencies
    ):
        raise RuntimeError(
            "Unexpected ARFF drift frequency count."
        )

    if covariance_omega.shape[1] != (
        config.fourier_frequencies
    ):
        raise RuntimeError(
            "Unexpected ARFF covariance frequency count."
        )

    np.savez_compressed(
        path,
        artifact_version=np.asarray(
            1,
            dtype=np.int64,
        ),
        method=np.asarray(
            "arff"
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
        iterations=np.asarray(
            config.arff.M_max,
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
        lambda_reg=np.asarray(
            config.arff.lambda_reg,
            dtype=np.float64,
        ),
        gamma=np.asarray(
            config.arff.gamma,
            dtype=np.float64,
        ),
        delta=np.asarray(
            config.arff.delta,
            dtype=np.float64,
        ),
        resampling=np.asarray(
            config.arff.resampling,
            dtype=np.bool_,
        ),
        metropolis_test=np.asarray(
            config.arff.metropolis_test,
            dtype=np.bool_,
        ),
        spd_epsilon=np.asarray(
            config.evaluation.spd_epsilon,
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
        drift_omega=drift_omega,
        drift_amp=drift_amp,
        covariance_omega=covariance_omega,
        covariance_amp=covariance_amp,
    )

    print(
        "artifact   : "
        f"{path}"
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

    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=None,
        help=(
            "Optional .npz path for the final "
            "two-stage ARFF model and metadata. "
            "Artifact writing is outside benchmark "
            "training time."
        ),
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

    K = config.fourier_frequencies

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

    validate_split_indices(
        len(data.x),
        train_idx,
        validation_idx,
        test_idx,
    )

    # ------------------------------------------------------------
    # Move the complete ARFF training set to the device before any
    # compilation or algorithm timing begins.
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

    block_until_ready(
        (
            x_train,
            r_train,
            h_train,
        )
    )

    key = jax.random.PRNGKey(
        args.seed
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
        f"folds      : {config.arff.n_folds}"
    )
    print(
        f"frequencies: {K}"
    )
    print(
        f"iterations : {config.arff.M_max}"
    )
    print()

    # ------------------------------------------------------------
    # Construct one compiled adaptation kernel.
    # ------------------------------------------------------------

    compiled_adaptation_step = (
        make_compiled_adaptation_step(
            delta=config.arff.delta,
            lambda_reg=(
                config.arff.lambda_reg
            ),
            gamma=config.arff.gamma,
            resampling=(
                config.arff.resampling
            ),
            metropolis_test=(
                config.arff.metropolis_test
            ),
        )
    )

    # ------------------------------------------------------------
    # Determine every regression shape used by the actual procedure.
    # ------------------------------------------------------------

    folds = make_folds(
        len(x_train),
        config.arff.n_folds,
        config.split.seed,
    )

    drift_dimension = (
        definition.n_dimensions
    )

    covariance_dimension = (
        covariance_output_dimension(
            state_dimension=(
                definition.n_dimensions
            ),
            diff_type=(
                definition.diff_type
            ),
        )
    )

    warm_shapes = set()

    for holdout_idx in folds:
        fit_size = (
            len(x_train)
            - len(holdout_idx)
        )

        warm_shapes.add(
            (
                fit_size,
                drift_dimension,
            )
        )

    warm_shapes.add(
        (
            len(x_train),
            drift_dimension,
        )
    )

    warm_shapes.add(
        (
            len(x_train),
            covariance_dimension,
        )
    )

    warm_shapes = sorted(
        warm_shapes
    )

    # ------------------------------------------------------------
    # First-call/JIT warm-up.
    # ------------------------------------------------------------

    first_call_overhead = 0.0

    print(
        "ARFF first-call/JIT warm-up"
    )

    for (
        sample_count,
        output_dimension,
    ) in warm_shapes:
        x_warm = x_train[
            :sample_count
        ]

        (
            _,
            shape_time,
        ) = timed_call(
            warm_arff_shape,
            key,
            compiled_adaptation_step,
            x_warm,
            output_dimension=(
                output_dimension
            ),
            K=K,
            config=config.arff,
        )

        first_call_overhead += (
            shape_time
        )

        print(
            "  "
            f"N={sample_count}, "
            f"outputs={output_dimension}: "
            f"{shape_time:.3f} s"
        )

    print(
        "first-call/JIT overhead: "
        f"{first_call_overhead:.3f} s"
    )
    print()

    # ------------------------------------------------------------
    # Real two-stage ARFF training.
    # ------------------------------------------------------------

    (
        (
            key,
            model,
            crossfit,
        ),
        algorithm_time,
    ) = timed_call(
        run_two_stage_synchronized,
        key,
        x_train,
        r_train,
        h_train,
        K=K,
        diff_type=(
            definition.diff_type
        ),
        config=config.arff,
        fold_seed=(
            config.split.seed
        ),
        compiled_adaptation_step=(
            compiled_adaptation_step
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

    print()

    # ------------------------------------------------------------
    # Final evaluation.
    #
    # Evaluation is outside benchmark training time. The test split is
    # first used here.
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
        result = gaussian_nll(
            model,
            data.x[idx],
            data.r[idx],
            data.h[idx],
            spd_epsilon=(
                config.evaluation.spd_epsilon
            ),
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

        print(label)

        print(
            "  NLL                    : "
            f"{result.nll:.8e}"
        )

        print(
            "  raw SPD violation rate : "
            f"{result.spd_violation_rate:.6f}"
        )

        print(
            "  min raw eigenvalue     : "
            f"{result.min_raw_eigenvalue:.8e}"
        )

        print(
            "  min projected eigenvalue: "
            f"{result.min_projected_eigenvalue:.8e}"
        )

        print(
            "  drift RMSE              : "
            f"{drift_rmse:.8e}"
        )

        print(
            "  covariance RMSE         : "
            f"{covariance_rmse:.8e}"
        )

        print()

    # ------------------------------------------------------------
    # Optional archival artifact.
    #
    # Saving occurs strictly after benchmark timing and evaluation.
    # ------------------------------------------------------------

    if args.artifact_path is not None:
        save_artifact(
            args.artifact_path,
            experiment=name,
            seed=args.seed,
            diff_type=(
                definition.diff_type
            ),
            model=model,
            timing=timing,
            config=config,
        )


if __name__ == "__main__":
    main()