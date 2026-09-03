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
    #
    # Dataset loading, split indexing, and host-to-device transfer are
    # therefore not counted as method training time.
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
    #
    # The same function object is reused for every fold fit, final
    # drift fit, and final covariance fit. JAX compiles separate
    # executables only when array shapes differ.
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
    #
    # Cross-fitting can produce two neighboring training-set sizes
    # when N is not divisible by the number of folds.
    #
    # Output dimension also matters for JIT compilation: the final
    # covariance regression may have a different number of outputs
    # from the drift regression.
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

    # Final drift fit.
    warm_shapes.add(
        (
            len(x_train),
            drift_dimension,
        )
    )

    # Final covariance fit.
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
    #
    # Every distinct regression shape is exercised once with a
    # one-iteration fit. The result and advanced PRNG key are discarded.
    #
    # We intentionally use the original benchmark key independently for
    # every warm-up so warm-up cannot alter the stochastic state of the
    # real training run.
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
    #
    # Start from the untouched PRNG key. Compilation warm-up above
    # therefore has no effect on frequencies, resampling, or Metropolis
    # decisions in this fitted model.
    #
    # Cross-fitting is included in algorithm time because it is part of
    # the corrected two-stage estimator.
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


if __name__ == "__main__":
    main()