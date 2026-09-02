#!/usr/bin/env python3
"""Smoke test for deterministic canonical data splits."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.config import SplitConfig

from src.experiments.dataset import (
    load_dataset,
    make_split_indices,
    validate_split_indices,
)


def main():
    config = SplitConfig(
        seed=2026,
        train_fraction=0.8,
        validation_fraction=0.1,
        test_fraction=0.1,
    )

    n = 1003

    train_idx, validation_idx, test_idx = (
        make_split_indices(n, config)
    )

    validate_split_indices(
        n,
        train_idx,
        validation_idx,
        test_idx,
    )

    # Determinism.
    train2, validation2, test2 = (
        make_split_indices(n, config)
    )

    assert np.array_equal(train_idx, train2)
    assert np.array_equal(validation_idx, validation2)
    assert np.array_equal(test_idx, test2)

    assert len(train_idx) == 802
    assert len(validation_idx) == 100
    assert len(test_idx) == 101

    # Test the canonical loader on a temporary dataset.
    x = np.zeros((n, 2))
    r = np.ones((n, 2))
    h = np.full((n, 1), 0.01)

    with TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.npz"

        np.savez(
            path,
            x_data=x,
            r_data=r,
            step_sizes=h,
            train_idx=train_idx,
            validation_idx=validation_idx,
            test_idx=test_idx,
        )

        dataset = load_dataset(path)

        assert dataset.x.shape == (n, 2)
        assert dataset.r.shape == (n, 2)
        assert dataset.h.shape == (n, 1)

        assert np.array_equal(
            dataset.train_idx,
            train_idx,
        )
        assert np.array_equal(
            dataset.validation_idx,
            validation_idx,
        )
        assert np.array_equal(
            dataset.test_idx,
            test_idx,
        )

    print("Canonical split smoke test")
    print("--------------------------")
    print(f"N          : {n}")
    print(f"train      : {len(train_idx)}")
    print(f"validation : {len(validation_idx)}")
    print(f"test       : {len(test_idx)}")
    print(f"seed       : {config.seed}")
    print()
    print("All canonical split checks passed.")


if __name__ == "__main__":
    main()