"""
Summarize canonical production results.

Reads the per-seed text logs under results/production and writes:
    results/production_summary.csv

The summary contains mean, sample standard deviation, standard error,
median, minimum, and maximum over the canonical 30 seeds for:
    test NLL,
    test drift RMSE,
    test covariance RMSE,
    algorithm time.

This script is read-only with respect to production run directories.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOT = ROOT / "results" / "production"
OUTPUT_PATH = ROOT / "results" / "production_summary.csv"

METHODS = (
    ("arff", "ARFF split"),
    ("adam", "Fourier Adam joint"),
    ("mlp", "MLP Adam joint"),
)

EXPERIMENTS = tuple(
    f"ex{i}" for i in range(1, 9)
)

EXPECTED_SEEDS = tuple(range(30))


def extract_float(
    text: str,
    pattern: str,
    *,
    label: str,
    path: Path,
) -> float:
    match = re.search(
        pattern,
        text,
        flags=re.MULTILINE,
    )

    if match is None:
        raise RuntimeError(
            f"Could not find {label!r} in {path}"
        )

    value = float(match.group(1))

    if not np.isfinite(value):
        raise RuntimeError(
            f"Non-finite {label!r} in {path}: {value}"
        )

    return value


def parse_log(path: Path) -> dict[str, float]:
    text = path.read_text()

    algorithm_time = extract_float(
        text,
        r"^algorithm time\s*:\s*"
        r"([-+0-9.eE]+)\s*s\s*$",
        label="algorithm time",
        path=path,
    )

    test_match = re.search(
        r"^test\s*$"
        r"(.*?)(?="
        r"^artifact\s*:|\Z)",
        text,
        flags=(
            re.MULTILINE
            | re.DOTALL
        ),
    )

    if test_match is None:
        raise RuntimeError(
            f"Could not find test block in {path}"
        )

    test_text = test_match.group(1)

    test_nll = extract_float(
        test_text,
        r"^\s*NLL\s*:\s*"
        r"([-+0-9.eE]+)\s*$",
        label="test NLL",
        path=path,
    )

    drift_rmse = extract_float(
        test_text,
        r"^\s*drift RMSE\s*:\s*"
        r"([-+0-9.eE]+)\s*$",
        label="test drift RMSE",
        path=path,
    )

    covariance_rmse = extract_float(
        test_text,
        r"^\s*covariance RMSE\s*:\s*"
        r"([-+0-9.eE]+)\s*$",
        label="test covariance RMSE",
        path=path,
    )

    return {
        "test_nll": test_nll,
        "drift_rmse": drift_rmse,
        "covariance_rmse": covariance_rmse,
        "algorithm_time": algorithm_time,
    }


def stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if array.size != 30:
        raise RuntimeError(
            f"Expected 30 values, got {array.size}"
        )

    return {
        "mean": float(np.mean(array)),
        "std": float(np.std(array, ddof=1)),
        "sem": float(
            np.std(array, ddof=1)
            / np.sqrt(array.size)
        ),
        "median": float(np.median(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def main() -> None:
    rows = []

    print(
        "Canonical production summary"
    )
    print("=" * 110)

    for experiment in EXPERIMENTS:
        for method, method_label in METHODS:
            directory = (
                PRODUCTION_ROOT
                / f"{method}_{experiment}"
            )

            if not directory.is_dir():
                raise RuntimeError(
                    f"Missing directory: {directory}"
                )

            records = []

            for seed in EXPECTED_SEEDS:
                path = (
                    directory
                    / f"seed_{seed}.txt"
                )

                if not path.is_file():
                    raise RuntimeError(
                        f"Missing log: {path}"
                    )

                if path.stat().st_size == 0:
                    raise RuntimeError(
                        f"Empty log: {path}"
                    )

                records.append(
                    parse_log(path)
                )

            metric_stats = {
                metric: stats(
                    [
                        record[metric]
                        for record in records
                    ]
                )
                for metric in (
                    "test_nll",
                    "drift_rmse",
                    "covariance_rmse",
                    "algorithm_time",
                )
            }

            row = {
                "experiment": experiment,
                "method": method,
                "method_label": method_label,
                "n_runs": 30,
            }

            for metric, summary in metric_stats.items():
                for statistic, value in summary.items():
                    row[
                        f"{metric}_{statistic}"
                    ] = value

            rows.append(row)

            print(
                f"{experiment:3s}  "
                f"{method_label:20s}  "
                f"NLL "
                f"{metric_stats['test_nll']['mean']:+.6e}"
                f" ± "
                f"{metric_stats['test_nll']['std']:.2e}   "
                f"drift "
                f"{metric_stats['drift_rmse']['mean']:.3e}"
                f" ± "
                f"{metric_stats['drift_rmse']['std']:.2e}   "
                f"cov "
                f"{metric_stats['covariance_rmse']['mean']:.3e}"
                f" ± "
                f"{metric_stats['covariance_rmse']['std']:.2e}   "
                f"time "
                f"{metric_stats['algorithm_time']['mean']:.2f}"
                f" ± "
                f"{metric_stats['algorithm_time']['std']:.2f}s"
            )

        print("-" * 110)

    fieldnames = list(
        rows[0].keys()
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open(
        "w",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)

    print()
    print(
        f"Wrote: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
