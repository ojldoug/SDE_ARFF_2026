#!/usr/bin/env python3
"""Compare eager and JIT-compiled ARFF adaptation."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.arff.regression import (
    ARFFModel,
    adaptation_step,
    fit_amplitudes,
    make_compiled_adaptation_step,
)


def block_model(model):
    model.omega.block_until_ready()
    model.amp.block_until_ready()


def run_eager(
    key,
    model,
    x,
    y,
    *,
    n_iterations,
    delta,
    lambda_reg,
    gamma,
    resampling,
    metropolis_test,
):
    for _ in range(n_iterations):
        key, model = adaptation_step(
            key,
            model,
            x,
            y,
            delta=delta,
            lambda_reg=lambda_reg,
            gamma=gamma,
            resampling=resampling,
            metropolis_test=metropolis_test,
        )

    block_model(model)

    return key, model


def run_compiled(
    key,
    model,
    x,
    y,
    *,
    n_iterations,
    compiled_step,
):
    for _ in range(n_iterations):
        key, model = compiled_step(
            key,
            model,
            x,
            y,
        )

    block_model(model)

    return key, model


def main():
    rng = np.random.default_rng(123)

    n = 512
    d = 2
    q = 2
    K = 32
    n_iterations = 30

    lambda_reg = 1e-3
    gamma = 1.0
    delta = 0.2
    resampling = False
    metropolis_test = True

    x = jnp.asarray(
        rng.uniform(
            -1.0,
            1.0,
            size=(n, d),
        ).astype(np.float32)
    )

    y = jnp.asarray(
        np.column_stack(
            [
                np.sin(np.asarray(x[:, 0])),
                np.cos(np.asarray(x[:, 1])),
            ]
        ).astype(np.float32)
    )

    omega = jnp.zeros(
        (d, K),
        dtype=x.dtype,
    )

    amp = fit_amplitudes(
        x,
        y,
        omega,
        lambda_reg,
    )

    amp.block_until_ready()

    initial_model = ARFFModel(
        omega=omega,
        amp=amp,
    )

    initial_key = jax.random.PRNGKey(2026)

    compiled_step = make_compiled_adaptation_step(
        delta=delta,
        lambda_reg=lambda_reg,
        gamma=gamma,
        resampling=resampling,
        metropolis_test=metropolis_test,
    )

    # ------------------------------------------------------------
    # Compile/warm the JIT kernel.
    # Discard the result.
    # ------------------------------------------------------------
    warm_key, warm_model = compiled_step(
        initial_key,
        initial_model,
        x,
        y,
    )

    warm_key.block_until_ready()
    block_model(warm_model)

    # ------------------------------------------------------------
    # Eager reference.
    # ------------------------------------------------------------
    start = time.perf_counter()

    eager_key, eager_model = run_eager(
        initial_key,
        initial_model,
        x,
        y,
        n_iterations=n_iterations,
        delta=delta,
        lambda_reg=lambda_reg,
        gamma=gamma,
        resampling=resampling,
        metropolis_test=metropolis_test,
    )

    eager_seconds = time.perf_counter() - start

    # ------------------------------------------------------------
    # Compiled implementation.
    # ------------------------------------------------------------
    start = time.perf_counter()

    compiled_key, compiled_model = run_compiled(
        initial_key,
        initial_model,
        x,
        y,
        n_iterations=n_iterations,
        compiled_step=compiled_step,
    )

    compiled_seconds = time.perf_counter() - start

    eager_omega = np.asarray(
        eager_model.omega
    )
    compiled_omega = np.asarray(
        compiled_model.omega
    )

    eager_amp = np.asarray(
        eager_model.amp
    )
    compiled_amp = np.asarray(
        compiled_model.amp
    )

    omega_max_abs = np.max(
        np.abs(
            eager_omega
            - compiled_omega
        )
    )

    amp_max_abs = np.max(
        np.abs(
            eager_amp
            - compiled_amp
        )
    )

    prediction_eager = np.asarray(
        (
            jnp.concatenate(
                [
                    jnp.cos(x @ eager_model.omega),
                    jnp.sin(x @ eager_model.omega),
                ],
                axis=-1,
            )
            @ eager_model.amp
        )
    )

    prediction_compiled = np.asarray(
        (
            jnp.concatenate(
                [
                    jnp.cos(x @ compiled_model.omega),
                    jnp.sin(x @ compiled_model.omega),
                ],
                axis=-1,
            )
            @ compiled_model.amp
        )
    )

    prediction_max_abs = np.max(
        np.abs(
            prediction_eager
            - prediction_compiled
        )
    )

    eager_mse = np.mean(
        (
            prediction_eager
            - np.asarray(y)
        ) ** 2
    )

    compiled_mse = np.mean(
        (
            prediction_compiled
            - np.asarray(y)
        ) ** 2
    )

    print("ARFF JIT benchmark")
    print("------------------")
    print(f"backend                : {jax.default_backend()}")
    print(f"N                      : {n}")
    print(f"K                      : {K}")
    print(f"iterations             : {n_iterations}")
    print()
    print(f"eager time             : {eager_seconds:.6f} s")
    print(f"compiled time          : {compiled_seconds:.6f} s")
    print(
        "speedup                : "
        f"{eager_seconds / compiled_seconds:.3f}x"
    )
    print()
    print(f"eager MSE              : {eager_mse:.8e}")
    print(f"compiled MSE           : {compiled_mse:.8e}")
    print(
        "max |omega difference| : "
        f"{omega_max_abs:.8e}"
    )
    print(
        "max |amp difference|   : "
        f"{amp_max_abs:.8e}"
    )
    print(
        "max |pred difference|  : "
        f"{prediction_max_abs:.8e}"
    )

    if not np.allclose(
        eager_omega,
        compiled_omega,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError(
            "JIT and eager ARFF frequencies differ materially."
        )

    if not np.allclose(
        prediction_eager,
        prediction_compiled,
        rtol=1e-4,
        atol=1e-5,
    ):
        raise RuntimeError(
            "JIT and eager ARFF predictions differ materially."
        )

    print()
    print("JIT and eager ARFF agree.")


if __name__ == "__main__":
    main()