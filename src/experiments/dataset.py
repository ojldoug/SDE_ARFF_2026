"""
Canonical dataset utilities for the reproducible experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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


@dataclass(frozen=True)
class ExperimentDataset:
    x: np.ndarray
    r: np.ndarray
    h: np.ndarray
    train_idx: np.ndarray
    validation_idx: np.ndarray
    test_idx: np.ndarray


def load_dataset(path: Path | str) -> ExperimentDataset:
    """
    Load and validate one canonical experiment dataset.

    The stored train/validation/test indices are treated as immutable.
    No split is constructed here.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Canonical dataset not found: {path}"
        )

    with np.load(path, allow_pickle=False) as data:
        required = {
            "x_data",
            "r_data",
            "step_sizes",
            "train_idx",
            "validation_idx",
            "test_idx",
        }

        missing = required.difference(data.files)

        if missing:
            raise ValueError(
                f"Dataset is missing arrays: {sorted(missing)}"
            )

        x = np.asarray(data["x_data"])
        r = np.asarray(data["r_data"])
        h = np.asarray(data["step_sizes"])

        train_idx = np.asarray(data["train_idx"])
        validation_idx = np.asarray(data["validation_idx"])
        test_idx = np.asarray(data["test_idx"])

    if x.ndim != 2:
        raise ValueError("x_data must be two-dimensional.")

    if r.ndim != 2:
        raise ValueError("r_data must be two-dimensional.")

    if h.ndim != 2 or h.shape[1] != 1:
        raise ValueError(
            "step_sizes must have shape (N, 1)."
        )

    n = len(x)

    if len(r) != n or len(h) != n:
        raise ValueError(
            "x_data, r_data, and step_sizes have "
            "inconsistent sample counts."
        )

    if not np.all(np.isfinite(x)):
        raise ValueError("x_data contains non-finite values.")

    if not np.all(np.isfinite(r)):
        raise ValueError("r_data contains non-finite values.")

    if not np.all(np.isfinite(h)):
        raise ValueError(
            "step_sizes contains non-finite values."
        )

    if np.any(h <= 0.0):
        raise ValueError("All step sizes must be positive.")

    validate_split_indices(
        n,
        train_idx,
        validation_idx,
        test_idx,
    )

    return ExperimentDataset(
        x=x,
        r=r,
        h=h,
        train_idx=train_idx,
        validation_idx=validation_idx,
        test_idx=test_idx,
    )