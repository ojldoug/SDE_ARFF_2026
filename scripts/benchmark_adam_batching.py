#!/usr/bin/env python3
"""Compare old and optimized Adam minibatch execution."""

from __future__ import annotations

from pathlib import Path
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adam.fourier import initialize_model
from src.adam.training import make_compiled_adam_functions


def block_tree(tree):
    for leaf in jax.tree_util.tree_leaves(tree):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def run_old_epoch(
    key,
    model,
    opt_state,
    x,
    r,
    h,
    *,
    batch_size,
    compiled_train_step,
):
    """
    Reproduce the previous host-index/per-batch-synchronization path.
    """
    n = len(x)

    key, permutation_key = jax.random.split(key)

    permutation = np.asarray(
        jax.random.permutation(
            permutation_key,
            n,
        )
    )

    losses = []
    sizes = []

    for start in range(
        0,
        n,
        batch_size,
    ):
        batch_idx = permutation[
            start:start + batch_size
        ]

        model, opt_state, batch_loss = (
            compiled_train_step(
                model,
                opt_state,
                x[batch_idx],
                r[batch_idx],
                h[batch_idx],
            )
        )

        # This is the expensive synchronization used by the old path.
        losses.append(
            float(batch_loss)
        )

        sizes.append(
            len(batch_idx)
        )

    training_nll = np.average(
        losses,
        weights=sizes,
    )

    block_tree(
        (
            model,
            opt_state,
        )
    )

    return (
        key,
        model,
        opt_state,
        float(training_nll),
    )


def run_new_epoch(
    key,
    model,
    opt_state,
    x,
    r,
    h,
    *,
    batch_size,
    compiled_train_step,
):
    """
    Optimized device-resident shuffle and loss accumulation.
    """
    n = len(x)

    key, permutation_key = jax.random.split(key)

    permutation = jax.random.permutation(
        permutation_key,
        n,
    )

    x_epoch = x[permutation]
    r_epoch = r[permutation]
    h_epoch = h[permutation]

    weighted_loss_sum = jnp.asarray(
        0.0,
        dtype=x.dtype,
    )

    for start in range(
        0,
        n,
        batch_size,
    ):
        end = min(
            start + batch_size,
            n,
        )

        current_batch_size = (
            end - start
        )

        model, opt_state, batch_loss = (
            compiled_train_step(
                model,
                opt_state,
                x_epoch[start:end],
                r_epoch[start:end],
                h_epoch[start:end],
            )
        )

        weighted_loss_sum = (
            weighted_loss_sum
            + current_batch_size
            * batch_loss
        )

    training_nll = (
        weighted_loss_sum
        / n
    )

    (
        model,
        opt_state,
        training_nll,
    ) = jax.device_get(
        (
            model,
            opt_state,
            training_nll,
        )
    )

    return (
        key,
        model,
        opt_state,
        float(training_nll),
    )


def main():
    rng = np.random.default_rng(123)

    n = 8192
    d = 2
    K = 32
    batch_size = 256
    learning_rate = 1e-3

    x = rng.uniform(
        -1.0,
        1.0,
        size=(n, d),
    ).astype(np.float32)

    true_drift = -x

    h = np.full(
        (n, 1),
        0.01,
        dtype=np.float32,
    )

    noise = rng.normal(
        size=(n, d),
    ).astype(np.float32)

    r = (
        h * true_drift
        + np.sqrt(0.01 * 0.2) * noise
    )

    x = jnp.asarray(x)
    r = jnp.asarray(r)
    h = jnp.asarray(h)

    block_tree(
        (
            x,
            r,
            h,
        )
    )

    initialization_key = (
        jax.random.PRNGKey(7)
    )

    initial_model = initialize_model(
        initialization_key,
        input_dimension=d,
        output_dimension=d,
        n_frequencies=K,
        diff_type="diagonal",
    )

    (
        optimizer,
        compiled_train_step,
        _,
    ) = make_compiled_adam_functions(
        learning_rate
    )

    initial_opt_state = optimizer.init(
        initial_model
    )

    # ------------------------------------------------------------
    # Warm both minibatch shapes used by this benchmark.
    # Here n is divisible by batch_size, so there is one shape.
    # ------------------------------------------------------------
    warm_model, warm_state, warm_loss = (
        compiled_train_step(
            initial_model,
            initial_opt_state,
            x[:batch_size],
            r[:batch_size],
            h[:batch_size],
        )
    )

    block_tree(
        (
            warm_model,
            warm_state,
            warm_loss,
        )
    )

    epoch_key = jax.random.PRNGKey(
        2026
    )

    # ------------------------------------------------------------
    # Old path.
    # ------------------------------------------------------------
    start = time.perf_counter()

    (
        old_key,
        old_model,
        old_state,
        old_loss,
    ) = run_old_epoch(
        epoch_key,
        initial_model,
        initial_opt_state,
        x,
        r,
        h,
        batch_size=batch_size,
        compiled_train_step=compiled_train_step,
    )

    old_seconds = (
        time.perf_counter()
        - start
    )

    # ------------------------------------------------------------
    # New path.
    # ------------------------------------------------------------
    start = time.perf_counter()

    (
        new_key,
        new_model,
        new_state,
        new_loss,
    ) = run_new_epoch(
        epoch_key,
        initial_model,
        initial_opt_state,
        x,
        r,
        h,
        batch_size=batch_size,
        compiled_train_step=compiled_train_step,
    )

    new_seconds = (
        time.perf_counter()
        - start
    )

    old_leaves = (
        jax.tree_util.tree_leaves(
            old_model
        )
    )

    new_leaves = (
        jax.tree_util.tree_leaves(
            new_model
        )
    )

    max_parameter_difference = 0.0

    for old_leaf, new_leaf in zip(
        old_leaves,
        new_leaves,
    ):
        difference = np.max(
            np.abs(
                np.asarray(old_leaf)
                - np.asarray(new_leaf)
            )
        )

        max_parameter_difference = max(
            max_parameter_difference,
            float(difference),
        )

    print("Adam batching benchmark")
    print("-----------------------")
    print(
        f"backend               : "
        f"{jax.default_backend()}"
    )
    print(
        f"N                     : {n}"
    )
    print(
        f"K                     : {K}"
    )
    print(
        f"batch size            : {batch_size}"
    )
    print(
        f"batches               : "
        f"{n // batch_size}"
    )
    print()

    print(
        f"old epoch time        : "
        f"{old_seconds:.6f} s"
    )
    print(
        f"new epoch time        : "
        f"{new_seconds:.6f} s"
    )
    print(
        "speedup               : "
        f"{old_seconds / new_seconds:.3f}x"
    )
    print()

    print(
        f"old training NLL      : "
        f"{old_loss:.8e}"
    )
    print(
        f"new training NLL      : "
        f"{new_loss:.8e}"
    )
    print(
        "NLL difference        : "
        f"{abs(old_loss - new_loss):.8e}"
    )
    print(
        "max parameter diff    : "
        f"{max_parameter_difference:.8e}"
    )

    if not np.isclose(
        old_loss,
        new_loss,
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError(
            "Old and optimized Adam losses differ materially."
        )

    if (
        max_parameter_difference
        > 1e-5
    ):
        raise RuntimeError(
            "Old and optimized Adam parameters differ materially."
        )

    print()
    print(
        "Old and optimized Adam paths agree."
    )


if __name__ == "__main__":
    main()