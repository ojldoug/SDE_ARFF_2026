#!/usr/bin/env python3
"""
Summarize production experiment logs written by the canonical runners.

This parser is intended for logs such as

    results/production/adam_ex6/seed_0.txt
    ...
    results/production/adam_ex6/seed_29.txt

It extracts, when present:

    algorithm time
    first-call/JIT time
    end-to-end time
    best epoch
    best validation NLL
    train / validation / test NLL
    train / validation / test drift RMSE
    train / validation / test covariance RMSE
    train / validation / test minimum covariance eigenvalue

The script prints:

1. one row per seed;
2. aggregate mean, standard deviation, minimum, and maximum.

Typical use
-----------

    python scripts/summarize_production_logs.py \
        results/production/adam_ex6
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np


FLOAT_RE = (
    r"[-+]?"
    r"(?:"
    r"(?:\d+\.\d*)"
    r"|"
    r"(?:\.\d+)"
    r"|"
    r"(?:\d+)"
    r")"
    r"(?:[eE][-+]?\d+)?"
)


PATTERNS = {
    "algorithm_time": re.compile(
        rf"algorithm time\s*:\s*({FLOAT_RE})\s*s"
    ),
    "jit_time": re.compile(
        rf"first-call/JIT time\s*:\s*({FLOAT_RE})\s*s"
    ),
    "end_to_end_time": re.compile(
        rf"end-to-end time\s*:\s*({FLOAT_RE})\s*s"
    ),
    "best_epoch": re.compile(
        r"best epoch\s*:\s*(\d+)"
    ),
    "best_validation_nll": re.compile(
        rf"best validation NLL\s*:\s*({FLOAT_RE})"
    ),
}


METRIC_PATTERNS = {
    "nll": re.compile(
        rf"NLL\s*:\s*({FLOAT_RE})"
    ),
    "drift_rmse": re.compile(
        rf"drift RMSE\s*:\s*({FLOAT_RE})"
    ),
    "covariance_rmse": re.compile(
        rf"covariance RMSE\s*:\s*({FLOAT_RE})"
    ),
    "min_covariance_eig": re.compile(
        rf"min covariance eig\s*:\s*({FLOAT_RE})"
    ),
}


SPLITS = (
    "train",
    "validation",
    "test",
)


def seed_from_path(
    path: Path,
):
    match = re.search(
        r"seed_(\d+)",
        path.stem,
    )

    if match is None:
        return None

    return int(
        match.group(1)
    )


def parse_float(
    pattern,
    text,
):
    match = pattern.search(
        text
    )

    if match is None:
        return np.nan

    return float(
        match.group(1)
    )


def parse_int(
    pattern,
    text,
):
    match = pattern.search(
        text
    )

    if match is None:
        return np.nan

    return int(
        match.group(1)
    )


def split_blocks(
    text,
):
    """
    Extract train / validation / test sections.
    """
    blocks = {}

    for split in SPLITS:
        pattern = re.compile(
            rf"(?m)^{split}\s*$"
            rf"(.*?)(?="
            rf"^(?:train|validation|test)\s*$"
            rf"|\Z"
            rf")",
            re.DOTALL
            | re.MULTILINE,
        )

        match = pattern.search(
            text
        )

        if match is None:
            blocks[
                split
            ] = ""
        else:
            blocks[
                split
            ] = match.group(1)

    return blocks


def parse_log(
    path: Path,
):
    text = path.read_text()

    seed = seed_from_path(
        path
    )

    result = {
        "seed": (
            seed
            if seed is not None
            else np.nan
        ),
        "algorithm_time": parse_float(
            PATTERNS[
                "algorithm_time"
            ],
            text,
        ),
        "jit_time": parse_float(
            PATTERNS[
                "jit_time"
            ],
            text,
        ),
        "end_to_end_time": parse_float(
            PATTERNS[
                "end_to_end_time"
            ],
            text,
        ),
        "best_epoch": parse_int(
            PATTERNS[
                "best_epoch"
            ],
            text,
        ),
        "best_validation_nll": parse_float(
            PATTERNS[
                "best_validation_nll"
            ],
            text,
        ),
    }

    blocks = split_blocks(
        text
    )

    for split in SPLITS:
        block = blocks[
            split
        ]

        for (
            metric_name,
            pattern,
        ) in METRIC_PATTERNS.items():
            result[
                f"{split}_{metric_name}"
            ] = parse_float(
                pattern,
                block,
            )

    return result


def format_value(
    value,
):
    if isinstance(
        value,
        (
            int,
            np.integer,
        ),
    ):
        return str(
            value
        )

    if not np.isfinite(
        value
    ):
        return "nan"

    return (
        f"{value:.8e}"
    )


def print_runs(
    results,
):
    columns = [
        "seed",
        "algorithm_time",
        "best_epoch",
        "best_validation_nll",
        "validation_nll",
        "test_nll",
        "validation_drift_rmse",
        "validation_covariance_rmse",
    ]

    print(
        "Per-seed results"
    )

    print(
        "=" * 130
    )

    header = (
        f"{'seed':>4}  "
        f"{'alg[s]':>11}  "
        f"{'epoch':>6}  "
        f"{'best val NLL':>15}  "
        f"{'val NLL':>15}  "
        f"{'test NLL':>15}  "
        f"{'val drift':>15}  "
        f"{'val cov':>15}"
    )

    print(
        header
    )

    print(
        "-" * 130
    )

    for result in results:
        print(
            f"{int(result['seed']):4d}  "
            f"{result['algorithm_time']:11.3f}  "
            f"{int(result['best_epoch']):6d}  "
            f"{result['best_validation_nll']:15.8e}  "
            f"{result['validation_nll']:15.8e}  "
            f"{result['test_nll']:15.8e}  "
            f"{result['validation_drift_rmse']:15.8e}  "
            f"{result['validation_covariance_rmse']:15.8e}"
        )

    print()


def finite_values(
    results,
    key,
):
    values = np.asarray(
        [
            result[
                key
            ]
            for result in results
        ],
        dtype=float,
    )

    return values[
        np.isfinite(
            values
        )
    ]


def summary_statistics(
    values,
):
    if len(
        values
    ) == 0:
        return {
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
            "n": 0,
        }

    return {
        "mean": float(
            np.mean(
                values
            )
        ),
        "std": float(
            np.std(
                values,
                ddof=1,
            )
        )
        if len(values) > 1
        else 0.0,
        "min": float(
            np.min(
                values
            )
        ),
        "max": float(
            np.max(
                values
            )
        ),
        "n": int(
            len(
                values
            )
        ),
    }


def print_summary(
    results,
):
    metrics = [
        (
            "algorithm time [s]",
            "algorithm_time",
        ),
        (
            "first-call/JIT time [s]",
            "jit_time",
        ),
        (
            "end-to-end time [s]",
            "end_to_end_time",
        ),
        (
            "best epoch",
            "best_epoch",
        ),
        (
            "best validation NLL",
            "best_validation_nll",
        ),
        (
            "validation NLL",
            "validation_nll",
        ),
        (
            "test NLL",
            "test_nll",
        ),
        (
            "validation drift RMSE",
            "validation_drift_rmse",
        ),
        (
            "test drift RMSE",
            "test_drift_rmse",
        ),
        (
            "validation covariance RMSE",
            "validation_covariance_rmse",
        ),
        (
            "test covariance RMSE",
            "test_covariance_rmse",
        ),
        (
            "validation min covariance eig",
            "validation_min_covariance_eig",
        ),
        (
            "test min covariance eig",
            "test_min_covariance_eig",
        ),
    ]

    print(
        "Aggregate summary"
    )

    print(
        "=" * 112
    )

    print(
        f"{'metric':<38}"
        f"{'n':>5}"
        f"{'mean':>18}"
        f"{'std':>18}"
        f"{'min':>18}"
        f"{'max':>18}"
    )

    print(
        "-" * 112
    )

    for (
        label,
        key,
    ) in metrics:
        values = finite_values(
            results,
            key,
        )

        stats = summary_statistics(
            values
        )

        print(
            f"{label:<38}"
            f"{stats['n']:5d}"
            f"{stats['mean']:18.8e}"
            f"{stats['std']:18.8e}"
            f"{stats['min']:18.8e}"
            f"{stats['max']:18.8e}"
        )

    print()


def validate_results(
    results,
):
    if not results:
        raise RuntimeError(
            "No production logs were found."
        )

    seeds = [
        result[
            "seed"
        ]
        for result in results
    ]

    if any(
        not np.isfinite(
            seed
        )
        for seed in seeds
    ):
        raise RuntimeError(
            "Could not infer a seed from "
            "one or more filenames."
        )

    seeds = [
        int(seed)
        for seed in seeds
    ]

    if len(
        set(
            seeds
        )
    ) != len(
        seeds
    ):
        raise RuntimeError(
            "Duplicate seed logs detected."
        )

    required = (
        "algorithm_time",
        "best_epoch",
        "best_validation_nll",
        "validation_nll",
        "test_nll",
    )

    for result in results:
        for key in required:
            if not np.isfinite(
                result[
                    key
                ]
            ):
                raise RuntimeError(
                    "Incomplete production log: "
                    f"seed {int(result['seed'])} "
                    f"is missing {key}."
                )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "log_directory",
        type=Path,
        help=(
            "Directory containing "
            "seed_*.txt production logs."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    log_directory = (
        args.log_directory
    )

    if not (
        log_directory.is_dir()
    ):
        raise ValueError(
            f"Not a directory: "
            f"{log_directory}"
        )

    paths = sorted(
        log_directory.glob(
            "seed_*.txt"
        ),
        key=lambda path: (
            seed_from_path(
                path
            )
            if seed_from_path(
                path
            )
            is not None
            else 10**9
        ),
    )

    results = [
        parse_log(
            path
        )
        for path in paths
    ]

    validate_results(
        results
    )

    print(
        f"log directory : "
        f"{log_directory}"
    )

    print(
        f"runs found    : "
        f"{len(results)}"
    )

    print(
        f"seed range    : "
        f"{int(results[0]['seed'])}"
        f"--"
        f"{int(results[-1]['seed'])}"
    )

    print()

    print_runs(
        results
    )

    print_summary(
        results
    )


if __name__ == "__main__":
    main()