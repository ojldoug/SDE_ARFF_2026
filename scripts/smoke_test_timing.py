#!/usr/bin/env python3
"""Smoke test for synchronized JAX timing utilities."""

from pathlib import Path
import sys

import jax
import jax.numpy as jnp
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.timing import (
    TimingResult,
    timed_call,
)


def main():
    @jax.jit
    def operation(x):
        return x @ x.T

    x = jnp.ones(
        (128, 32),
        dtype=jnp.float32,
    )

    result, first_time = timed_call(
        operation,
        x,
    )

    result2, second_time = timed_call(
        operation,
        x,
    )

    assert result.shape == (128, 128)
    assert result2.shape == (128, 128)

    assert np.isfinite(first_time)
    assert np.isfinite(second_time)

    assert first_time >= 0.0
    assert second_time >= 0.0

    timing = TimingResult(
        compilation_seconds=first_time,
        algorithm_seconds=second_time,
    )

    assert np.isclose(
        timing.end_to_end_seconds,
        first_time + second_time,
    )

    print("Timing smoke test")
    print("-----------------")
    print(f"first call  : {first_time:.6f} s")
    print(f"second call : {second_time:.6f} s")
    print(
        "combined    : "
        f"{timing.end_to_end_seconds:.6f} s"
    )
    print()
    print("All timing checks passed.")


if __name__ == "__main__":
    main()