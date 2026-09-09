#!/usr/bin/env python3
"""
Experiment 8 oracle-drift covariance diagnostic.

Fit exactly the split-MLP covariance model, but construct residuals using
the known true drift rather than an estimated/cross-fitted drift.

Purpose:
    determine how much of split MLP's covariance error is caused by
    imperfect drift residuals versus covariance representation/optimization.

The canonical test split is not accessed.
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adam.mlp import initialize_model
from src.adam.split_mlp import (
    CovarianceMLPModel,
    _fit_one_regression,
    covariance_from_split_model,
    covariance_gaussian_nll,
)
from src.adam.training import make_compiled_adam_functions
from src.experiments.config import get_config
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment
from src.experiments.mlp_size import matched_two_layer_width
from src.experiments.model_size import covariance_output_dimension


def block(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def main():
    cfg = get_config("ex8")
    definition = get_experiment("ex8")
    hp = cfg.adam_mlp

    data = load_dataset(
        REPO_ROOT / "data" / "ex8.npz"
    )

    tr = data.train_idx
    va = data.validation_idx

    x = jnp.asarray(data.x[tr])
    r = jnp.asarray(data.r[tr])
    h = jnp.asarray(data.h[tr])

    xv = jnp.asarray(data.x[va])
    rv = jnp.asarray(data.r[va])
    hv = jnp.asarray(data.h[va])

    # ----------------------------------------------------------
    # Oracle residuals: true drift, not an estimated drift.
    # ----------------------------------------------------------

    residual = (
        r
        - h * definition.drift(x)
    )

    residual_validation = (
        rv
        - hv * definition.drift(xv)
    )

    block(
        (
            residual,
            residual_validation,
        )
    )

    cov_output = covariance_output_dimension(
        definition.n_dimensions,
        definition.diff_type,
    )

    width, declared_count, target_count = (
        matched_two_layer_width(
            state_dimension=(
                definition.n_dimensions
            ),
            covariance_output_dimension=(
                cov_output
            ),
            n_frequencies=(
                cfg.fourier_frequencies
            ),
        )
    )

    hidden_sizes = (width, width)

    print("EXPERIMENT 8")
    print("ORACLE-DRIFT MLP COVARIANCE DIAGNOSTIC")
    print("TEST SPLIT IS NOT ACCESSED")
    print()
    print(f"backend    : {jax.default_backend()}")
    print(f"train N    : {len(tr)}")
    print(f"validation : {len(va)}")
    print(f"hidden     : {width}-{width}")
    print(f"MLP params : {declared_count}")
    print(f"Fourier target params: {target_count}")
    print(f"epochs     : {hp.epochs}")
    print(f"batch size : {hp.batch_size}")
    print(f"learning rate: {hp.learning_rate:.8e}")
    print()

    key = jax.random.PRNGKey(0)
    key, init_key = jax.random.split(key)

    initial = initialize_model(
        init_key,
        input_dimension=(
            definition.state_dimension
        ),
        output_dimension=(
            definition.n_dimensions
        ),
        diff_type=(
            definition.diff_type
        ),
        hidden_sizes=hidden_sizes,
    )

    covariance_initial = CovarianceMLPModel(
        initial.covariance,
        definition.diff_type,
        definition.n_dimensions,
    )

    optimizer = optax.adam(
        hp.learning_rate
    )

    compiled = make_compiled_adam_functions(
        hp.learning_rate,
        nll_fn=covariance_gaussian_nll,
    )

    # The helper returns:
    # optimizer, compiled_train_step, compiled_loss
    covariance_optimizer = compiled[0]
    compiled_train_step = compiled[1]
    compiled_loss = compiled[2]

    # Use the optimizer supplied by the canonical helper.
    del optimizer

    start = time.perf_counter()

    key, training = _fit_one_regression(
        key,
        covariance_initial,
        x,
        residual,
        h,
        xv,
        residual_validation,
        hv,
        epochs=hp.epochs,
        batch_size=hp.batch_size,
        optimizer=covariance_optimizer,
        compiled_train_step=(
            compiled_train_step
        ),
        compiled_loss=compiled_loss,
    )

    block(training.model)

    elapsed = time.perf_counter() - start

    covariance_prediction = (
        covariance_from_split_model(
            training.model,
            xv,
        )
    )

    covariance_true = (
        definition.covariance(xv)
    )

    covariance_rmse = float(
        jnp.sqrt(
            jnp.mean(
                (
                    covariance_prediction
                    - covariance_true
                )
                ** 2
            )
        )
    )

    eigenvalues = jnp.linalg.eigvalsh(
        covariance_prediction
    )

    min_eigenvalue = float(
        jnp.min(eigenvalues)
    )

    max_eigenvalue = float(
        jnp.max(eigenvalues)
    )

    validation_nll = float(
        covariance_gaussian_nll(
            training.model,
            xv,
            residual_validation,
            hv,
        )
    )

    print(
        "best epoch          : "
        f"{training.best_epoch}"
    )

    print(
        "best validation NLL : "
        f"{training.best_validation_nll:.8e}"
    )

    print()
    print("========================================")
    print("ORACLE-DRIFT VALIDATION RESULT")
    print("========================================")
    print(
        f"NLL                 : "
        f"{validation_nll:.8e}"
    )
    print(
        f"covariance RMSE     : "
        f"{covariance_rmse:.8e}"
    )
    print(
        f"min covariance eig  : "
        f"{min_eigenvalue:.8e}"
    )
    print(
        f"max covariance eig  : "
        f"{max_eigenvalue:.8e}"
    )
    print(
        f"elapsed             : "
        f"{elapsed:.3f}s"
    )
    print("========================================")

    np.savez_compressed(
        REPO_ROOT
        / "results"
        / "diagnose_ex8_oracle_drift_mlp_seed0.npz",
        method=np.asarray(
            "oracle_drift_mlp"
        ),
        experiment=np.asarray("ex8"),
        seed=np.asarray(0),
        best_epoch=np.asarray(
            training.best_epoch
        ),
        best_validation_nll=np.asarray(
            training.best_validation_nll
        ),
        validation_nll=np.asarray(
            validation_nll
        ),
        validation_covariance_rmse=np.asarray(
            covariance_rmse
        ),
        validation_min_covariance_eig=np.asarray(
            min_eigenvalue
        ),
        validation_max_covariance_eig=np.asarray(
            max_eigenvalue
        ),
        validation_history=np.asarray(
            training.validation_nll
        ),
        training_history=np.asarray(
            training.training_nll
        ),
        covariance_weights=np.asarray(
            [
                np.asarray(w)
                for w in training.model.covariance.weights
            ],
            dtype=object,
        ),
        covariance_biases=np.asarray(
            [
                np.asarray(b)
                for b in training.model.covariance.biases
            ],
            dtype=object,
        ),
    )


if __name__ == "__main__":
    main()
