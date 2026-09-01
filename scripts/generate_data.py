#!/usr/bin/env python3
"""
Generate a reproducible dataset for one ARFF SDE experiment.

Examples
--------
    python scripts/generate_data.py ex1
    python scripts/generate_data.py ex5
    python scripts/generate_data.py ex8

Each output contains the common learning representation

    x_data, r_data, step_sizes

together with deterministic train/validation/test indices and metadata.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.config import get_config
from src.experiments.definitions import get_experiment
from src.experiments.em_data import (
    generate_langevin_em_data,
    generate_standard_em_data,
)
from src.experiments.sir_ssa import generate_sir_data
from src.experiments.wave_data import generate_wave_data


STANDARD_EM = {"ex1", "ex2", "ex3", "ex7", "ex8"}


def git_commit():
    """Return the current Git commit if available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return "unknown"


def generate(name):
    config = get_config(name)
    definition = get_experiment(name)

    if name in STANDARD_EM:
        return generate_standard_em_data(definition, config)

    if name == "ex4":
        return generate_langevin_em_data(definition, config)

    if name == "ex5":
        return generate_sir_data(config)

    if name == "ex6":
        return generate_wave_data(config)

    raise ValueError(f"No data generator registered for {name}.")


def validate(name, x, r, h, target_samples):
    if x.ndim != 2:
        raise RuntimeError(f"{name}: x_data must be two-dimensional.")

    if r.ndim != 2:
        raise RuntimeError(f"{name}: r_data must be two-dimensional.")

    if h.ndim != 2 or h.shape[1] != 1:
        raise RuntimeError(
            f"{name}: step_sizes must have shape (N, 1)."
        )

    n = len(x)

    if len(r) != n or len(h) != n:
        raise RuntimeError(
            f"{name}: inconsistent sample counts: "
            f"x={len(x)}, r={len(r)}, h={len(h)}."
        )

    if target_samples is not None and n != target_samples:
        raise RuntimeError(
            f"{name}: expected {target_samples} samples, got {n}."
        )

    if not np.all(np.isfinite(x)):
        raise RuntimeError(f"{name}: non-finite values in x_data.")

    if not np.all(np.isfinite(r)):
        raise RuntimeError(f"{name}: non-finite values in r_data.")

    if not np.all(np.isfinite(h)):
        raise RuntimeError(f"{name}: non-finite values in step_sizes.")

    if not np.all(h > 0.0):
        raise RuntimeError(f"{name}: all step sizes must be positive.")


def make_split(n, split_config):
    fractions = np.array(
        [
            split_config.train_fraction,
            split_config.validation_fraction,
            split_config.test_fraction,
        ]
    )

    if not np.isclose(fractions.sum(), 1.0):
        raise ValueError(
            "Train/validation/test fractions must sum to one."
        )

    rng = np.random.default_rng(split_config.seed)
    indices = rng.permutation(n)

    n_train = int(split_config.train_fraction * n)
    n_validation = int(split_config.validation_fraction * n)

    train_idx = indices[:n_train]
    validation_idx = indices[
        n_train:n_train + n_validation
    ]
    test_idx = indices[n_train + n_validation:]

    return train_idx, validation_idx, test_idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "experiment",
        choices=[f"ex{i}" for i in range(1, 9)],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )
    args = parser.parse_args()

    name = args.experiment
    config = get_config(name)
    definition = get_experiment(name)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{name}.npz"

    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_path} already exists. "
            "Use --overwrite to replace it."
        )

    print(f"Generating {name}...")

    x_data, r_data, step_sizes = generate(name)

    validate(
        name,
        x_data,
        r_data,
        step_sizes,
        config.data.target_samples,
    )

    train_idx, validation_idx, test_idx = make_split(
        len(x_data),
        config.split,
    )

    metadata = {
        "experiment": name,
        "git_commit": git_commit(),
        "definition": {
            "n_dimensions": definition.n_dimensions,
            "state_dimension": definition.state_dimension,
            "diff_type": definition.diff_type,
            "xlim": (
                None
                if definition.xlim is None
                else np.asarray(definition.xlim).tolist()
            ),
        },
        "config": asdict(config),
        "n_samples": len(x_data),
        "x_shape": list(x_data.shape),
        "r_shape": list(r_data.shape),
        "step_sizes_shape": list(step_sizes.shape),
    }

    np.savez_compressed(
        output_path,
        x_data=x_data,
        r_data=r_data,
        step_sizes=step_sizes,
        train_idx=train_idx,
        validation_idx=validation_idx,
        test_idx=test_idx,
        metadata=np.array(json.dumps(metadata)),
    )

    print(f"Saved: {output_path}")
    print(f"samples    : {len(x_data)}")
    print(f"x shape    : {x_data.shape}")
    print(f"r shape    : {r_data.shape}")
    print(f"h shape    : {step_sizes.shape}")
    print(
        "h range    : "
        f"[{step_sizes.min():.8e}, "
        f"{step_sizes.max():.8e}]"
    )
    print(f"train      : {len(train_idx)}")
    print(f"validation : {len(validation_idx)}")
    print(f"test       : {len(test_idx)}")


if __name__ == "__main__":
    main()
