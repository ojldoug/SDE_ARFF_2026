#!/usr/bin/env python3
"""Compare new experiment definitions against legacy dill functions."""

from pathlib import Path
import sys

import dill
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.definitions import get_experiment  # noqa: E402


SEED = 12345
N_TEST = 1000


def max_abs(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    rng = np.random.default_rng(SEED)

    print("Legacy/new true-function comparison")
    print("===================================")

    for i in range(1, 9):
        name = f"ex{i}"
        definition = get_experiment(name)

        p = REPO_ROOT / "GPU" / "true_functions" / f"{name}.pkl"
        with p.open("rb") as f:
            legacy = dill.load(f)

        # Experiments 5 and 6 do not carry xlim in the legacy pickle.
        if definition.xlim is not None:
            low = definition.xlim[:, 0]
            high = definition.xlim[:, 1]
            x = rng.uniform(low=low, high=high, size=(N_TEST, len(low)))

        elif name == "ex5":
            # Sample admissible (S,R) points from the simplex.
            z = rng.uniform(size=(N_TEST, 3))
            z /= z.sum(axis=1, keepdims=True)
            x = z[:, [0, 2]]

        elif name == "ex6":
            # Inputs are (time, space).
            t = rng.uniform(0.0, 2.0, size=N_TEST)
            space = rng.uniform(0.0, 1.0, size=N_TEST)
            x = np.column_stack([t, space])

        else:
            raise RuntimeError(name)

        xj = jnp.asarray(x)

        legacy_drift = legacy["drift"](xj)
        new_drift = definition.drift(xj)

        legacy_sigma = legacy["diffusion"](xj)
        new_sigma = definition.diffusion_factor(xj)

        drift_error = max_abs(legacy_drift, new_drift)
        sigma_error = max_abs(legacy_sigma, new_sigma)

        print()
        print(name)
        print(f"  drift max abs error : {drift_error:.8e}")
        print(f"  sigma max abs error : {sigma_error:.8e}")

        # ex5 deliberately removes the legacy numerical epsilon.
        if name == "ex5":
            print("  note                : sigma differs intentionally because")
            print("                        legacy true_diffusion adds epsilon=1e-8")
        else:
            if drift_error > 1e-6:
                raise AssertionError(f"{name}: drift mismatch")

            if sigma_error > 1e-6:
                raise AssertionError(f"{name}: diffusion mismatch")

    print()
    print("All non-intentional comparisons passed.")


if __name__ == "__main__":
    main()
