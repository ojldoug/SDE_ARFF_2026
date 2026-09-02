"""
Timing utilities for reproducible JAX experiments.

The main benchmark distinguishes

    compilation time

from

    algorithm time,

where algorithm time excludes one-time JAX/XLA compilation but includes
all computations required by the stated training procedure.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

import jax


@dataclass(frozen=True)
class TimingResult:
    compilation_seconds: float
    algorithm_seconds: float

    @property
    def end_to_end_seconds(self) -> float:
        return (
            self.compilation_seconds
            + self.algorithm_seconds
        )


def block_until_ready(tree) -> None:
    """
    Wait until all JAX array leaves in a pytree have completed execution.
    """
    leaves = jax.tree_util.tree_leaves(tree)

    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def timed_call(function, *args, **kwargs):
    """
    Execute a function and return (result, elapsed wall time).

    JAX outputs are synchronized before the timer stops so GPU execution
    is included in the measured wall time.
    """
    start = time.perf_counter()

    result = function(
        *args,
        **kwargs,
    )

    block_until_ready(result)

    elapsed = time.perf_counter() - start

    return result, elapsed