#!/usr/bin/env python3
"""Smoke test for Adam training and validation checkpointing."""

from pathlib import Path
import sys

import jax
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.adam.fourier import (
    gaussian_nll,
    initialize_model,
)
from src.adam.training import (
    fit_adam_fourier,
    make_compiled_adam_functions,
)


def main():
    rng = np.random.default_rng(321)

    n_train = 192
    n_validation = 64
    d = 2

    x = rng.uniform(
        -1.0,
        1.0,
        size=(n_train + n_validation, d),
    ).astype(np.float32)

    h = np.full(
        (len(x), 1),
        0.01,
        dtype=np.float32,
    )

    true_drift = -x

    noise = rng.normal(
        size=x.shape,
    ).astype(np.float32)

    r = (
        h * true_drift
        + np.sqrt(0.01 * 0.2) * noise
    )

    x_train = x[:n_train]
    r_train = r[:n_train]
    h_train = h[:n_train]

    x_validation = x[n_train:]
    r_validation = r[n_train:]
    h_validation = h[n_train:]

    key = jax.random.PRNGKey(7)

    model = initialize_model(
        key,
        input_dimension=d,
        output_dimension=d,
        n_frequencies=16,
        diff_type="diagonal",
    )

    initial_validation_nll = float(
        gaussian_nll(
            model,
            x_validation,
            r_validation,
            h_validation,
        )
    )

    (
        optimizer,
        compiled_train_step,
        compiled_nll,
    ) = make_compiled_adam_functions(
        learning_rate=1e-3,
    )

    key, result = fit_adam_fourier(
        key,
        model,
        x_train,
        r_train,
        h_train,
        x_validation,
        r_validation,
        h_validation,
        epochs=5,
        batch_size=32,
        optimizer=optimizer,
        compiled_train_step=compiled_train_step,
        compiled_nll=compiled_nll,
    )

    assert result.training_nll.shape == (5,)
    assert result.validation_nll.shape == (5,)

    assert np.all(
        np.isfinite(result.training_nll)
    )
    assert np.all(
        np.isfinite(result.validation_nll)
    )

    assert 0 <= result.best_epoch < 5

    assert np.isclose(
        result.best_validation_nll,
        np.min(result.validation_nll),
    )

    selected_validation_nll = float(
        gaussian_nll(
            result.model,
            x_validation,
            r_validation,
            h_validation,
        )
    )

    assert np.isclose(
        selected_validation_nll,
        result.best_validation_nll,
    )

    print("Adam training smoke test")
    print("------------------------")
    print(f"train N                : {n_train}")
    print(f"validation N           : {n_validation}")
    print(f"initial validation NLL : {initial_validation_nll:.8e}")
    print(f"best epoch             : {result.best_epoch}")
    print(
        "best validation NLL    : "
        f"{result.best_validation_nll:.8e}"
    )
    print(
        "validation history     : "
        f"{result.validation_nll}"
    )
    print()
    print("All Adam training checks passed.")


if __name__ == "__main__":
    main()