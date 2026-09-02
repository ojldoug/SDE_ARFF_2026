"""
Canonical dataset utilities for the reproducible experiments.
"""

from __future__ import annotations

import numpy as np

from src.experiments.config import SplitConfig


def make_split_indices(
    n_samples: int,
    config: SplitConfig,
):
    """
    Create one deterministic train/validation/test partition.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    fractions = np.array(
        [
            config.train_fraction,
            config.validation_fraction,
            config.test_fraction,
        ],
        dtype=float,
    )

    if np.any(fractions <= 0.0):
        raise ValueError("All split fractions must be positive.")

    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError(
            "Train/validation/test fractions must sum to one."
        )

    rng = np.random.default_rng(config.seed)
    permutation = rng.permutation(n_samples)

    n_train = int(
        np.floor(config.train_fraction * n_samples)
    )
    n_validation = int(
        np.floor(config.validation_fraction * n_samples)
    )

    n_test = n_samples - n_train - n_validation

    train_idx = permutation[:n_train]
    validation_idx = permutation[
        n_train:n_train + n_validation
    ]
    test_idx = permutation[
        n_train + n_validation:
    ]

    if len(test_idx) != n_test:
        raise RuntimeError("Split construction failed.")

    return train_idx, validation_idx, test_idx


def validate_split_indices(
    n_samples: int,
    train_idx,
    validation_idx,
    test_idx,
):
    """
    Verify that the three index sets form an exact partition.
    """
    train_idx = np.asarray(train_idx)
    validation_idx = np.asarray(validation_idx)
    test_idx = np.asarray(test_idx)

    combined = np.concatenate(
        [train_idx, validation_idx, test_idx]
    )

    if len(combined) != n_samples:
        raise ValueError(
            "Split sizes do not add up to the dataset size."
        )

    if len(np.unique(combined)) != n_samples:
        raise ValueError(
            "Split indices overlap or contain duplicates."
        )

    if np.min(combined) < 0 or np.max(combined) >= n_samples:
        raise ValueError("Split index out of range.")