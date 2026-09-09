#!/usr/bin/env python3
"""
Experiment 8: validation-selected, cross-fitted ARFF diagnostic.

Goal
----
Test whether the Experiment-8 split-learning story survives when we use

  * the current canonical reproducible ex8 dataset,
  * blocker-compliant five-fold cross-fitting,
  * Owen's audited historical ex8 ARFF hyperparameters,
  * internal ARFF validation after every adaptation step,
  * the minimum-validation-MSE checkpoint from each ARFF regression.

The canonical test split is NOT used here. This is a validation-only
diagnostic before deciding on the final poster/production configuration.

Historical ex8 ARFF settings from the audited notebook
-------------------------------------------------------
    K                 = 128
    M_min = M_max     = 300
    lambda            = 1e-3
    gamma             = 1
    delta             = 0.2
    resampling        = False
    Metropolis        = True
    ARFF val fraction = 0.1

Current blocker-related changes retained
----------------------------------------
    * five-fold cross-fitted residual covariance targets;
    * deterministic canonical dataset split;
    * SPD diagnostics / projection convention;
    * untouched canonical test split.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0,
    str(REPO_ROOT),
)

from src.arff.covariance import (
    covariance_targets,
    raw_covariance,
    spd_violation_mask,
)
from src.arff.evaluation import (
    gaussian_nll,
    true_function_errors,
)
from src.arff.regression import (
    ARFFModel,
    make_compiled_adaptation_step,
    predict,
)
from src.arff.two_stage import (
    CrossFitResult,
    TwoStageARFFModel,
    make_folds,
)
from src.arff.validation_selected import (
    ValidationSelectedResult,
    fit_validation_selected_arff,
)
from src.experiments.config import (
    get_config,
)
from src.experiments.dataset import (
    load_dataset,
    validate_split_indices,
)
from src.experiments.definitions import (
    get_experiment,
)


# ------------------------------------------------------------------
# Historical ex8 ARFF configuration
# ------------------------------------------------------------------

K = 128

M_MIN = 300
M_MAX = 300

LAMBDA_REG = 1e-3
GAMMA = 1.0
DELTA = 0.2

RESAMPLING = True
METROPOLIS_TEST = False

ARFF_VALIDATION_FRACTION = 0.1

N_FOLDS = 5

MOVING_AVERAGE_LENGTH = 5
PATIENCE = 5


def block_until_ready(tree):
    for leaf in jax.tree_util.tree_leaves(
        tree
    ):
        if hasattr(
            leaf,
            "block_until_ready",
        ):
            leaf.block_until_ready()


def fit_selected(
    key,
    x,
    y,
    *,
    validation_seed: int,
    compiled_step,
):
    return fit_validation_selected_arff(
        key,
        x,
        y,
        K=K,
        M_min=M_MIN,
        M_max=M_MAX,
        lambda_reg=LAMBDA_REG,
        gamma=GAMMA,
        delta=DELTA,
        resampling=RESAMPLING,
        metropolis_test=METROPOLIS_TEST,
        validation_fraction=(
            ARFF_VALIDATION_FRACTION
        ),
        validation_seed=(
            validation_seed
        ),
        moving_average_length=(
            MOVING_AVERAGE_LENGTH
        ),
        patience=PATIENCE,
        compiled_adaptation_step=(
            compiled_step
        ),
    )


def build_cross_fitted_targets(
    key,
    x,
    r,
    h,
    *,
    diff_type: str,
    fold_seed: int,
    internal_validation_seed: int,
    compiled_step,
):
    """
    Build honest covariance targets using five validation-selected
    out-of-fold drift regressions.
    """
    x = jnp.asarray(
        x
    )

    r = jnp.asarray(
        r
    )

    h = jnp.asarray(
        h
    )

    n = len(
        x
    )

    folds = make_folds(
        n,
        N_FOLDS,
        fold_seed,
    )

    if diff_type == "diagonal":
        target_dimension = (
            r.shape[1]
        )
    else:
        d = r.shape[1]

        target_dimension = (
            d
            * (d + 1)
            // 2
        )

    targets = jnp.zeros(
        (
            n,
            target_dimension,
        ),
        dtype=x.dtype,
    )

    fold_id = np.empty(
        n,
        dtype=np.int32,
    )

    all_indices = np.arange(
        n
    )

    fold_results = []

    for fold_number, holdout_idx in enumerate(
        folds
    ):
        train_mask = np.ones(
            n,
            dtype=bool,
        )

        train_mask[
            holdout_idx
        ] = False

        fit_idx = all_indices[
            train_mask
        ]

        x_fit = x[
            fit_idx
        ]

        r_fit = r[
            fit_idx
        ]

        h_fit = h[
            fit_idx
        ]

        drift_target = (
            r_fit
            / h_fit
        )

        validation_seed = (
            internal_validation_seed
            + 1000
            + fold_number
        )

        fold_start = (
            time.perf_counter()
        )

        (
            key,
            fold_result,
        ) = fit_selected(
            key,
            x_fit,
            drift_target,
            validation_seed=(
                validation_seed
            ),
            compiled_step=(
                compiled_step
            ),
        )

        block_until_ready(
            fold_result.model
        )

        fold_elapsed = (
            time.perf_counter()
            - fold_start
        )

        x_holdout = x[
            holdout_idx
        ]

        r_holdout = r[
            holdout_idx
        ]

        h_holdout = h[
            holdout_idx
        ]

        drift_holdout = predict(
            fold_result.model,
            x_holdout,
        )

        residual = (
            r_holdout
            - h_holdout
            * drift_holdout
        )

        fold_targets = (
            covariance_targets(
                residual,
                h_holdout,
                diff_type,
            )
        )

        targets = targets.at[
            holdout_idx
        ].set(
            fold_targets
        )

        fold_id[
            holdout_idx
        ] = fold_number

        fold_results.append(
            (
                fold_result,
                fold_elapsed,
            )
        )

        print(
            f"fold {fold_number}: "
            f"best M="
            f"{fold_result.best_iteration}, "
            f"best val MSE="
            f"{fold_result.best_validation_mse:.8e}, "
            f"elapsed="
            f"{fold_elapsed:.3f}s"
        )

    block_until_ready(
        targets
    )

    finite = bool(
        jax.device_get(
            jnp.all(
                jnp.isfinite(
                    targets
                )
            )
        )
    )

    if not finite:
        raise RuntimeError(
            "Cross-fitted covariance targets "
            "contain non-finite values."
        )

    return (
        key,
        CrossFitResult(
            covariance_targets=targets,
            fold_id=fold_id,
        ),
        fold_results,
    )


def save_stage_result(
    prefix,
    result: ValidationSelectedResult,
    dictionary,
):
    dictionary[
        f"{prefix}_validation_mse"
    ] = result.validation_mse

    dictionary[
        f"{prefix}_moving_average"
    ] = result.moving_average

    dictionary[
        f"{prefix}_cumulative_time"
    ] = result.cumulative_time

    dictionary[
        f"{prefix}_best_iteration"
    ] = np.asarray(
        result.best_iteration,
        dtype=np.int64,
    )

    dictionary[
        f"{prefix}_best_validation_mse"
    ] = np.asarray(
        result.best_validation_mse,
        dtype=np.float64,
    )

    dictionary[
        f"{prefix}_best_time"
    ] = np.asarray(
        result.best_time,
        dtype=np.float64,
    )

    dictionary[
        f"{prefix}_stopped_iteration"
    ] = np.asarray(
        result.stopped_iteration,
        dtype=np.int64,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=(
            Path(
                "results/"
                "diagnose_ex8_validation_selected_"
                "crossfit_seed0.npz"
            )
        ),
    )

    args = parser.parse_args()

    config = get_config(
        "ex8"
    )

    definition = get_experiment(
        "ex8"
    )

    data = load_dataset(
        REPO_ROOT
        / "data"
        / "ex8.npz"
    )

    validate_split_indices(
        len(
            data.x
        ),
        data.train_idx,
        data.validation_idx,
        data.test_idx,
    )

    train_idx = (
        data.train_idx
    )

    validation_idx = (
        data.validation_idx
    )

    # --------------------------------------------------------------
    # IMPORTANT:
    # The canonical test split is deliberately never loaded below.
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

    print(
        "EXPERIMENT 8"
    )

    print(
        "VALIDATION-SELECTED "
        "CROSS-FITTED ARFF"
    )

    print(
        "TEST SPLIT IS NOT ACCESSED"
    )

    print()

    print(
        f"seed       : {args.seed}"
    )

    print(
        f"backend    : "
        f"{jax.default_backend()}"
    )

    print(
        f"train N    : "
        f"{len(train_idx)}"
    )

    print(
        f"validation : "
        f"{len(validation_idx)}"
    )

    print(
        f"K          : {K}"
    )

    print(
        f"M_min/max  : "
        f"{M_MIN}/{M_MAX}"
    )

    print(
        f"lambda     : "
        f"{LAMBDA_REG:.8e}"
    )

    print(
        f"delta      : "
        f"{DELTA:.8e}"
    )

    print(
        f"gamma      : "
        f"{GAMMA:.8e}"
    )

    print(
        f"resampling : "
        f"{RESAMPLING}"
    )

    print(
        f"Metropolis : "
        f"{METROPOLIS_TEST}"
    )

    print(
        f"folds      : "
        f"{N_FOLDS}"
    )

    print(
        "ARFF val   : "
        f"{ARFF_VALIDATION_FRACTION:.2f}"
    )

    print()

    key = jax.random.PRNGKey(
        args.seed
    )

    # --------------------------------------------------------------
    # One shared compiled adaptation kernel.
    # --------------------------------------------------------------

    compiled_step = (
        make_compiled_adaptation_step(
            delta=DELTA,
            lambda_reg=(
                LAMBDA_REG
            ),
            gamma=GAMMA,
            resampling=(
                RESAMPLING
            ),
            metropolis_test=(
                METROPOLIS_TEST
            ),
        )
    )

    total_start = (
        time.perf_counter()
    )

    # --------------------------------------------------------------
    # 1. Five-fold cross-fitted covariance targets.
    # --------------------------------------------------------------

    print(
        "Cross-fitted drift models"
    )

    crossfit_start = (
        time.perf_counter()
    )

    (
        key,
        crossfit,
        fold_results,
    ) = build_cross_fitted_targets(
        key,
        x_train,
        r_train,
        h_train,
        diff_type=(
            definition.diff_type
        ),
        fold_seed=(
            config.split.seed
        ),
        internal_validation_seed=(
            100000
            + args.seed
            * 10000
        ),
        compiled_step=(
            compiled_step
        ),
    )

    crossfit_time = (
        time.perf_counter()
        - crossfit_start
    )

    print(
        f"cross-fit total: "
        f"{crossfit_time:.3f}s"
    )

    print()

    # --------------------------------------------------------------
    # 2. Final drift fit on complete canonical training split.
    # --------------------------------------------------------------

    print(
        "Final drift model"
    )

    drift_target = (
        r_train
        / h_train
    )

    final_drift_start = (
        time.perf_counter()
    )

    (
        key,
        final_drift_result,
    ) = fit_selected(
        key,
        x_train,
        drift_target,
        validation_seed=(
            200000
            + args.seed
        ),
        compiled_step=(
            compiled_step
        ),
    )

    block_until_ready(
        final_drift_result.model
    )

    final_drift_time = (
        time.perf_counter()
        - final_drift_start
    )

    print(
        "  best iteration       : "
        f"{final_drift_result.best_iteration}"
    )

    print(
        "  stopped iteration    : "
        f"{final_drift_result.stopped_iteration}"
    )

    print(
        "  best validation MSE  : "
        f"{final_drift_result.best_validation_mse:.8e}"
    )

    print(
        "  elapsed              : "
        f"{final_drift_time:.3f}s"
    )

    print()

    # --------------------------------------------------------------
    # 3. Covariance regression on honest cross-fitted targets.
    # --------------------------------------------------------------

    print(
        "Covariance model"
    )

    covariance_start = (
        time.perf_counter()
    )

    (
        key,
        covariance_result,
    ) = fit_selected(
        key,
        x_train,
        crossfit.covariance_targets,
        validation_seed=(
            300000
            + args.seed
        ),
        compiled_step=(
            compiled_step
        ),
    )

    block_until_ready(
        covariance_result.model
    )

    covariance_time = (
        time.perf_counter()
        - covariance_start
    )

    print(
        "  best iteration       : "
        f"{covariance_result.best_iteration}"
    )

    print(
        "  stopped iteration    : "
        f"{covariance_result.stopped_iteration}"
    )

    print(
        "  best validation MSE  : "
        f"{covariance_result.best_validation_mse:.8e}"
    )

    print(
        "  elapsed              : "
        f"{covariance_time:.3f}s"
    )

    print()

    final_model = (
        TwoStageARFFModel(
            drift=(
                final_drift_result.model
            ),
            covariance=(
                covariance_result.model
            ),
            diff_type=(
                definition.diff_type
            ),
        )
    )

    block_until_ready(
        final_model
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    # --------------------------------------------------------------
    # 4. Canonical validation evaluation only.
    # --------------------------------------------------------------

    likelihood = gaussian_nll(
        final_model,
        x_validation,
        r_validation,
        h_validation,
        spd_epsilon=(
            config.evaluation.spd_epsilon
        ),
    )

    (
        drift_rmse,
        covariance_rmse,
    ) = true_function_errors(
        final_model,
        x_validation,
        true_drift=(
            definition.drift
        ),
        true_diffusion_factor=(
            definition.diffusion_factor
        ),
    )

    raw = raw_covariance(
        final_model.covariance,
        x_validation,
        final_model.diff_type,
    )

    raw_violations = (
        spd_violation_mask(
            raw
        )
    )

    raw_violation_rate = float(
        jnp.mean(
            raw_violations
        )
    )

    raw_eigenvalues = (
        jnp.linalg.eigvalsh(
            raw
        )
    )

    min_raw_eigenvalue = float(
        jnp.min(
            raw_eigenvalues
        )
    )

    print(
        "========================================"
    )

    print(
        "CANONICAL VALIDATION RESULT"
    )

    print(
        "========================================"
    )

    print(
        f"NLL                : "
        f"{likelihood.nll:.8e}"
    )

    print(
        f"drift RMSE         : "
        f"{drift_rmse:.8e}"
    )

    print(
        f"covariance RMSE    : "
        f"{covariance_rmse:.8e}"
    )

    print(
        f"raw SPD violations : "
        f"{raw_violation_rate:.8e}"
    )

    print(
        f"min raw eigenvalue : "
        f"{min_raw_eigenvalue:.8e}"
    )

    print(
        f"total elapsed      : "
        f"{total_time:.3f}s"
    )

    print(
        "========================================"
    )

    # --------------------------------------------------------------
    # 5. Archive everything useful.
    # --------------------------------------------------------------

    artifact = (
        args.artifact_path
        .expanduser()
        .resolve()
    )

    artifact.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    archive = {
        "artifact_version": np.asarray(
            1,
            dtype=np.int64,
        ),
        "method": np.asarray(
            "arff_validation_selected_crossfit"
        ),
        "experiment": np.asarray(
            "ex8"
        ),
        "seed": np.asarray(
            args.seed,
            dtype=np.int64,
        ),
        "K": np.asarray(
            K,
            dtype=np.int64,
        ),
        "M_min": np.asarray(
            M_MIN,
            dtype=np.int64,
        ),
        "M_max": np.asarray(
            M_MAX,
            dtype=np.int64,
        ),
        "lambda_reg": np.asarray(
            LAMBDA_REG,
            dtype=np.float64,
        ),
        "gamma": np.asarray(
            GAMMA,
            dtype=np.float64,
        ),
        "delta": np.asarray(
            DELTA,
            dtype=np.float64,
        ),
        "resampling": np.asarray(
            RESAMPLING
        ),
        "metropolis_test": np.asarray(
            METROPOLIS_TEST
        ),
        "n_folds": np.asarray(
            N_FOLDS,
            dtype=np.int64,
        ),
        "arff_validation_fraction": np.asarray(
            ARFF_VALIDATION_FRACTION,
            dtype=np.float64,
        ),
        "crossfit_time": np.asarray(
            crossfit_time,
            dtype=np.float64,
        ),
        "final_drift_time": np.asarray(
            final_drift_time,
            dtype=np.float64,
        ),
        "covariance_time": np.asarray(
            covariance_time,
            dtype=np.float64,
        ),
        "total_time": np.asarray(
            total_time,
            dtype=np.float64,
        ),
        "validation_nll": np.asarray(
            likelihood.nll,
            dtype=np.float64,
        ),
        "validation_drift_rmse": np.asarray(
            drift_rmse,
            dtype=np.float64,
        ),
        "validation_covariance_rmse": np.asarray(
            covariance_rmse,
            dtype=np.float64,
        ),
        "validation_raw_spd_violation_rate": np.asarray(
            raw_violation_rate,
            dtype=np.float64,
        ),
        "validation_min_raw_eigenvalue": np.asarray(
            min_raw_eigenvalue,
            dtype=np.float64,
        ),
        "fold_id": np.asarray(
            crossfit.fold_id,
            dtype=np.int32,
        ),
        "drift_omega": np.asarray(
            jax.device_get(
                final_model.drift.omega
            )
        ),
        "drift_amp": np.asarray(
            jax.device_get(
                final_model.drift.amp
            )
        ),
        "covariance_omega": np.asarray(
            jax.device_get(
                final_model.covariance.omega
            )
        ),
        "covariance_amp": np.asarray(
            jax.device_get(
                final_model.covariance.amp
            )
        ),
    }

    save_stage_result(
        "final_drift",
        final_drift_result,
        archive,
    )

    save_stage_result(
        "covariance",
        covariance_result,
        archive,
    )

    for fold_number, (
        fold_result,
        fold_elapsed,
    ) in enumerate(
        fold_results
    ):
        prefix = (
            f"fold_{fold_number}"
        )

        save_stage_result(
            prefix,
            fold_result,
            archive,
        )

        archive[
            f"{prefix}_elapsed"
        ] = np.asarray(
            fold_elapsed,
            dtype=np.float64,
        )

    np.savez_compressed(
        artifact,
        **archive,
    )

    print()

    print(
        f"artifact: {artifact}"
    )


if __name__ == "__main__":
    main()
