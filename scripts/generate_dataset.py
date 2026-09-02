#!/usr/bin/env python3
"""Generate one canonical reproducible experiment dataset."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.config import get_config
from src.experiments.dataset import (
    make_split_indices,
    validate_split_indices,
)
from src.experiments.definitions import get_experiment
from src.experiments.em_data import (
    generate_langevin_em_data,
    generate_standard_em_data,
)
from src.experiments.sir_ssa import generate_sir_data
from src.experiments.wave_data import generate_wave_data


def git_commit() -> str:
    """Return the current Git commit hash."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def sha256(path: Path) -> str:
    """Compute the SHA-256 digest of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def generate(name: str):
    config = get_config(name)
    definition = get_experiment(name)

    if name == "ex4":
        return generate_langevin_em_data(
            definition,
            config,
        )

    if name == "ex5":
        return generate_sir_data(config)

    if name == "ex6":
        return generate_wave_data(config)

    return generate_standard_em_data(
        definition,
        config,
    )


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
    args = parser.parse_args()

    name = args.experiment
    config = get_config(name)

    print(f"Generating canonical dataset for {name}...")

    x_data, r_data, step_sizes = generate(name)

    n = len(x_data)

    if len(r_data) != n or len(step_sizes) != n:
        raise RuntimeError(
            "Generated arrays have inconsistent sample counts."
        )

    if not np.all(np.isfinite(x_data)):
        raise RuntimeError("x_data contains non-finite values.")

    if not np.all(np.isfinite(r_data)):
        raise RuntimeError("r_data contains non-finite values.")

    if not np.all(np.isfinite(step_sizes)):
        raise RuntimeError("step_sizes contains non-finite values.")

    if np.any(step_sizes <= 0.0):
        raise RuntimeError("step_sizes must be positive.")

    train_idx, validation_idx, test_idx = (
        make_split_indices(
            n,
            config.split,
        )
    )

    validate_split_indices(
        n,
        train_idx,
        validation_idx,
        test_idx,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / f"{name}.npz"
    metadata_path = output_dir / f"{name}.json"

    np.savez(
        dataset_path,
        x_data=x_data,
        r_data=r_data,
        step_sizes=step_sizes,
        train_idx=train_idx,
        validation_idx=validation_idx,
        test_idx=test_idx,
    )

    digest = sha256(dataset_path)

    metadata = {
        "experiment": name,
        "git_commit": git_commit(),
        "dataset_sha256": digest,
        "n_samples": n,
        "x_shape": list(x_data.shape),
        "r_shape": list(r_data.shape),
        "step_sizes_shape": list(step_sizes.shape),
        "train_samples": len(train_idx),
        "validation_samples": len(validation_idx),
        "test_samples": len(test_idx),
        "data_config": asdict(config.data),
        "split_config": asdict(config.split),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    print()
    print(f"dataset    : {dataset_path}")
    print(f"metadata   : {metadata_path}")
    print(f"SHA-256    : {digest}")
    print(f"N          : {n}")
    print(f"train      : {len(train_idx)}")
    print(f"validation : {len(validation_idx)}")
    print(f"test       : {len(test_idx)}")


if __name__ == "__main__":
    main()