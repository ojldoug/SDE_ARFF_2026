#!/usr/bin/env python3
"""
Ex8 oracle-drift MLP covariance training trajectory.

Reproduces the canonical Adam epoch/minibatch loop exactly, while
recording true validation covariance RMSE after every epoch.

IMPORTANT:
True covariance is diagnostic only. Model selection remains based
exclusively on canonical validation NLL. The test split is never used.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adam.mlp import initialize_model
from src.adam.split_mlp import (
    CovarianceMLPModel,
    covariance_from_split_model,
    covariance_gaussian_nll,
)
from src.adam.training import make_compiled_adam_functions
from src.experiments.config import get_config
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment
from src.experiments.mlp_size import matched_two_layer_width
from src.experiments.model_size import covariance_output_dimension


def main():
    cfg = get_config("ex8")
    definition = get_experiment("ex8")
    hp = cfg.adam_mlp

    data = load_dataset(REPO_ROOT / "data" / "ex8.npz")
    tr = data.train_idx
    va = data.validation_idx

    x_train = jnp.asarray(data.x[tr])
    r_train = jnp.asarray(data.r[tr])
    h_train = jnp.asarray(data.h[tr])

    x_validation = jnp.asarray(data.x[va])
    r_validation = jnp.asarray(data.r[va])
    h_validation = jnp.asarray(data.h[va])

    # Oracle-drift residuals.
    residual_train = (
        r_train
        - h_train * definition.drift(x_train)
    )
    residual_validation = (
        r_validation
        - h_validation * definition.drift(x_validation)
    )

    true_covariance_validation = definition.covariance(
        x_validation
    )

    cov_output = covariance_output_dimension(
        definition.n_dimensions,
        definition.diff_type,
    )

    width, declared_count, target_count = matched_two_layer_width(
        state_dimension=definition.n_dimensions,
        covariance_output_dimension=cov_output,
        n_frequencies=cfg.fourier_frequencies,
    )

    hidden_sizes = (width, width)

    print("EXPERIMENT 8")
    print("ORACLE-DRIFT MLP NLL-vs-RMSE TRAJECTORY")
    print("TRUE RMSE IS DIAGNOSTIC ONLY")
    print("TEST SPLIT IS NOT ACCESSED")
    print()
    print(f"backend      : {jax.default_backend()}")
    print(f"train N      : {len(tr)}")
    print(f"validation N : {len(va)}")
    print(f"hidden       : {width}-{width}")
    print(f"MLP params   : {declared_count}")
    print(f"Fourier target params: {target_count}")
    print(f"epochs       : {hp.epochs}")
    print(f"batch size   : {hp.batch_size}")
    print(f"learning rate: {hp.learning_rate:.8e}")
    print()

    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)

    initial = initialize_model(
        init_key,
        input_dimension=definition.state_dimension,
        output_dimension=definition.n_dimensions,
        diff_type=definition.diff_type,
        hidden_sizes=hidden_sizes,
    )

    model = CovarianceMLPModel(
        initial.covariance,
        definition.diff_type,
        definition.n_dimensions,
    )

    (
        optimizer,
        compiled_train_step,
        compiled_nll,
    ) = make_compiled_adam_functions(
        hp.learning_rate,
        nll_fn=covariance_gaussian_nll,
    )

    opt_state = optimizer.init(model)

    epochs = hp.epochs
    batch_size = hp.batch_size
    n_train = len(x_train)

    training_history = np.empty(epochs, dtype=np.float64)
    validation_history = np.empty(epochs, dtype=np.float64)
    covariance_rmse_history = np.empty(epochs, dtype=np.float64)
    cumulative_time = np.empty(epochs, dtype=np.float64)

    # Initial, pre-training diagnostic.
    initial_covariance = covariance_from_split_model(
        model,
        x_validation,
    )

    initial_nll_device = compiled_nll(
        model,
        x_validation,
        residual_validation,
        h_validation,
    )

    initial_rmse_device = jnp.sqrt(
        jnp.mean(
            (
                initial_covariance
                - true_covariance_validation
            ) ** 2
        )
    )

    initial_nll, initial_rmse = jax.device_get(
        (
            initial_nll_device,
            initial_rmse_device,
        )
    )

    initial_nll = float(initial_nll)
    initial_rmse = float(initial_rmse)

    best_nll = np.inf
    best_nll_epoch = -1
    best_nll_rmse = np.nan

    best_rmse = np.inf
    best_rmse_epoch = -1
    best_rmse_nll = np.nan

    start_time = time.perf_counter()

    for epoch in range(epochs):
        # ------------------------------------------------------
        # Identical shuffling/minibatching logic to fit_adam.
        # ------------------------------------------------------
        key, permutation_key = jax.random.split(key)

        permutation = jax.random.permutation(
            permutation_key,
            n_train,
        )

        x_epoch = x_train[permutation]
        r_epoch = residual_train[permutation]
        h_epoch = h_train[permutation]

        weighted_loss_sum = jnp.asarray(
            0.0,
            dtype=x_train.dtype,
        )

        for batch_start in range(
            0,
            n_train,
            batch_size,
        ):
            batch_end = min(
                batch_start + batch_size,
                n_train,
            )

            current_batch_size = (
                batch_end - batch_start
            )

            (
                model,
                opt_state,
                batch_loss,
            ) = compiled_train_step(
                model,
                opt_state,
                x_epoch[batch_start:batch_end],
                r_epoch[batch_start:batch_end],
                h_epoch[batch_start:batch_end],
            )

            weighted_loss_sum = (
                weighted_loss_sum
                + current_batch_size * batch_loss
            )

        training_nll_device = (
            weighted_loss_sum / n_train
        )

        validation_nll_device = compiled_nll(
            model,
            x_validation,
            residual_validation,
            h_validation,
        )

        # Diagnostic only: evaluate coefficient error of this same
        # post-epoch checkpoint against known simulation truth.
        covariance_prediction = covariance_from_split_model(
            model,
            x_validation,
        )

        covariance_rmse_device = jnp.sqrt(
            jnp.mean(
                (
                    covariance_prediction
                    - true_covariance_validation
                ) ** 2
            )
        )

        (
            training_nll,
            validation_nll,
            covariance_rmse,
        ) = jax.device_get(
            (
                training_nll_device,
                validation_nll_device,
                covariance_rmse_device,
            )
        )

        cumulative_time[epoch] = (
            time.perf_counter() - start_time
        )

        training_nll = float(training_nll)
        validation_nll = float(validation_nll)
        covariance_rmse = float(covariance_rmse)

        training_history[epoch] = training_nll
        validation_history[epoch] = validation_nll
        covariance_rmse_history[epoch] = covariance_rmse

        if not np.isfinite(training_nll):
            raise RuntimeError(
                f"Non-finite training NLL at epoch {epoch}"
            )

        if not np.isfinite(validation_nll):
            raise RuntimeError(
                f"Non-finite validation NLL at epoch {epoch}"
            )

        if not np.isfinite(covariance_rmse):
            raise RuntimeError(
                f"Non-finite covariance RMSE at epoch {epoch}"
            )

        if validation_nll < best_nll:
            best_nll = validation_nll
            best_nll_epoch = epoch
            best_nll_rmse = covariance_rmse

        if covariance_rmse < best_rmse:
            best_rmse = covariance_rmse
            best_rmse_epoch = epoch
            best_rmse_nll = validation_nll

        if (
            epoch == 0
            or (epoch + 1) % 25 == 0
            or epoch == epochs - 1
        ):
            print(
                f"epoch {epoch:3d} | "
                f"val NLL {validation_nll: .8e} | "
                f"cov RMSE {covariance_rmse:.8e}",
                flush=True,
            )

    elapsed = time.perf_counter() - start_time

    print()
    print("==============================================")
    print("TRAJECTORY SUMMARY")
    print("==============================================")
    print(
        f"initial validation NLL       : "
        f"{initial_nll:.8e}"
    )
    print(
        f"initial covariance RMSE      : "
        f"{initial_rmse:.8e}"
    )
    print()
    print("minimum validation NLL checkpoint")
    print(
        f"  epoch                      : "
        f"{best_nll_epoch}"
    )
    print(
        f"  validation NLL             : "
        f"{best_nll:.8e}"
    )
    print(
        f"  covariance RMSE            : "
        f"{best_nll_rmse:.8e}"
    )
    print()
    print("minimum diagnostic covariance-RMSE checkpoint")
    print(
        f"  epoch                      : "
        f"{best_rmse_epoch}"
    )
    print(
        f"  covariance RMSE            : "
        f"{best_rmse:.8e}"
    )
    print(
        f"  validation NLL             : "
        f"{best_rmse_nll:.8e}"
    )
    print()
    print("final epoch")
    print(
        f"  epoch                      : "
        f"{epochs - 1}"
    )
    print(
        f"  validation NLL             : "
        f"{validation_history[-1]:.8e}"
    )
    print(
        f"  covariance RMSE            : "
        f"{covariance_rmse_history[-1]:.8e}"
    )
    print()
    print(f"elapsed                      : {elapsed:.3f}s")
    print("==============================================")

    output = (
        REPO_ROOT
        / "results"
        / "diagnose_ex8_oracle_drift_mlp_curve_seed0.npz"
    )

    np.savez_compressed(
        output,
        epoch=np.arange(epochs, dtype=np.int64),
        cumulative_time=cumulative_time,
        training_nll=training_history,
        validation_nll=validation_history,
        validation_covariance_rmse=covariance_rmse_history,
        initial_validation_nll=np.asarray(initial_nll),
        initial_covariance_rmse=np.asarray(initial_rmse),
        best_nll_epoch=np.asarray(best_nll_epoch),
        best_nll=np.asarray(best_nll),
        rmse_at_best_nll=np.asarray(best_nll_rmse),
        best_rmse_epoch=np.asarray(best_rmse_epoch),
        best_rmse=np.asarray(best_rmse),
        nll_at_best_rmse=np.asarray(best_rmse_nll),
    )

    print(f"artifact: {output}")


if __name__ == "__main__":
    main()
