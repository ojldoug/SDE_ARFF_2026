#!/usr/bin/env python3
"""
Summarize canonical production experiment logs for Adam or ARFF.

The script auto-detects the method from the log contents.

Typical use
-----------

Adam:

    python scripts/summarize_production_logs.py \
        results/production/adam_ex6

ARFF:

    python scripts/summarize_production_logs.py \
        results/production/arff_ex6

The script prints per-seed values followed by aggregate mean, sample
standard deviation, minimum, and maximum.

For Adam it extracts:

    algorithm time
    first-call/JIT time
    end-to-end time
    best epoch
    best validation NLL
    train / validation / test:
        NLL
        drift RMSE
        covariance RMSE
        minimum covariance eigenvalue

For ARFF it extracts:

    algorithm time
    first-call/JIT time
    end-to-end time
    train / validation / test:
        NLL
        raw SPD violation rate
        minimum raw eigenvalue
        minimum projected eigenvalue
        drift RMSE
        covariance RMSE
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


COMMON_PATTERNS = {
    "algorithm_time": re.compile(
        rf"algorithm time\s*:\s*({FLOAT_RE})\s*s"
    ),
    "jit_time": re.compile(
        rf"first-call/JIT time\s*:\s*({FLOAT_RE})\s*s"
    ),
    "end_to_end_time": re.compile(
        rf"end-to-end time\s*:\s*({FLOAT_RE})\s*s"
    ),
}


ADAM_PATTERNS = {
    "best_epoch": re.compile(
        r"best epoch\s*:\s*(\d+)"
    ),
    "best_validation_nll": re.compile(
        rf"best validation NLL\s*:\s*({FLOAT_RE})"
    ),
}


ADAM_METRIC_PATTERNS = {
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


ARFF_METRIC_PATTERNS = {
    "nll": re.compile(
        rf"NLL\s*:\s*({FLOAT_RE})"
    ),
    "spd_violation_rate": re.compile(
        rf"raw SPD violation rate\s*:\s*({FLOAT_RE})"
    ),
    "min_raw_eigenvalue": re.compile(
        rf"min raw eigenvalue\s*:\s*({FLOAT_RE})"
    ),
    "min_projected_eigenvalue": re.compile(
        rf"min projected eigenvalue\s*:\s*({FLOAT_RE})"
    ),
    "drift_rmse": re.compile(
        rf"drift RMSE\s*:\s*({FLOAT_RE})"
    ),
    "covariance_rmse": re.compile(
        rf"covariance RMSE\s*:\s*({FLOAT_RE})"
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


def detect_method(
    text: str,
) -> str:
    if "best validation NLL" in text:
        return "adam"

    if "raw SPD violation rate" in text:
        return "arff"

    raise RuntimeError(
        "Could not determine whether log is Adam or ARFF."
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
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    method = detect_method(
        text
    )

    seed = seed_from_path(
        path
    )

    result = {
        "method": method,
        "seed": (
            seed
            if seed is not None
            else np.nan
        ),
        "algorithm_time": parse_float(
            COMMON_PATTERNS[
                "algorithm_time"
            ],
            text,
        ),
        "jit_time": parse_float(
            COMMON_PATTERNS[
                "jit_time"
            ],
            text,
        ),
        "end_to_end_time": parse_float(
            COMMON_PATTERNS[
                "end_to_end_time"
            ],
            text,
        ),
    }

    if method == "adam":
        result[
            "best_epoch"
        ] = parse_int(
            ADAM_PATTERNS[
                "best_epoch"
            ],
            text,
        )

        result[
            "best_validation_nll"
        ] = parse_float(
            ADAM_PATTERNS[
                "best_validation_nll"
            ],
            text,
        )

        metric_patterns = (
            ADAM_METRIC_PATTERNS
        )

    else:
        metric_patterns = (
            ARFF_METRIC_PATTERNS
        )

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
        ) in metric_patterns.items():
            result[
                f"{split}_{metric_name}"
            ] = parse_float(
                pattern,
                block,
            )

    return result


def finite_values(
    results,
    key,
):
    values = np.asarray(
        [
            result.get(
                key,
                np.nan,
            )
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
        "std": (
            float(
                np.std(
                    values,
                    ddof=1,
                )
            )
            if len(values) > 1
            else 0.0
        ),
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


def validate_results(
    results,
):
    if not results:
        raise RuntimeError(
            "No production logs were found."
        )

    methods = {
        result[
            "method"
        ]
        for result in results
    }

    if len(
        methods
    ) != 1:
        raise RuntimeError(
            "Mixed Adam and ARFF logs detected."
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

    method = results[
        0
    ][
        "method"
    ]

    if method == "adam":
        required = (
            "algorithm_time",
            "jit_time",
            "end_to_end_time",
            "best_epoch",
            "best_validation_nll",
            "validation_nll",
            "test_nll",
            "validation_drift_rmse",
            "validation_covariance_rmse",
        )

    else:
        required = (
            "algorithm_time",
            "jit_time",
            "end_to_end_time",
            "validation_nll",
            "test_nll",
            "validation_drift_rmse",
            "validation_covariance_rmse",
            "validation_spd_violation_rate",
            "test_spd_violation_rate",
            "validation_min_raw_eigenvalue",
            "test_min_raw_eigenvalue",
            "validation_min_projected_eigenvalue",
            "test_min_projected_eigenvalue",
        )

    for result in results:
        for key in required:
            value = result.get(
                key,
                np.nan,
            )

            if not np.isfinite(
                value
            ):
                raise RuntimeError(
                    "Incomplete production log: "
                    f"seed {int(result['seed'])} "
                    f"is missing {key}."
                )


def print_adam_runs(
    results,
):
    print(
        "Per-seed results"
    )

    print(
        "=" * 130
    )

    print(
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


def print_arff_runs(
    results,
):
    print(
        "Per-seed results"
    )

    print(
        "=" * 150
    )

    print(
        f"{'seed':>4}  "
        f"{'alg[s]':>11}  "
        f"{'JIT[s]':>10}  "
        f"{'val NLL':>15}  "
        f"{'test NLL':>15}  "
        f"{'val drift':>15}  "
        f"{'val cov':>15}  "
        f"{'val SPD':>10}  "
        f"{'test SPD':>10}"
    )

    print(
        "-" * 150
    )

    for result in results:
        print(
            f"{int(result['seed']):4d}  "
            f"{result['algorithm_time']:11.3f}  "
            f"{result['jit_time']:10.3f}  "
            f"{result['validation_nll']:15.8e}  "
            f"{result['test_nll']:15.8e}  "
            f"{result['validation_drift_rmse']:15.8e}  "
            f"{result['validation_covariance_rmse']:15.8e}  "
            f"{result['validation_spd_violation_rate']:10.6f}  "
            f"{result['test_spd_violation_rate']:10.6f}"
        )

    print()


def print_summary_table(
    results,
    metrics,
):
    print(
        "Aggregate summary"
    )

    print(
        "=" * 126
    )

    print(
        f"{'metric':<42}"
        f"{'n':>6}"
        f"{'mean':>20}"
        f"{'std':>20}"
        f"{'min':>20}"
        f"{'max':>20}"
    )

    print(
        "-" * 126
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
            f"{label:<42}"
            f"{stats['n']:6d}"
            f"{stats['mean']:20.8e}"
            f"{stats['std']:20.8e}"
            f"{stats['min']:20.8e}"
            f"{stats['max']:20.8e}"
        )

    print()


def print_adam_summary(
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

    print_summary_table(
        results,
        metrics,
    )


def print_arff_summary(
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
            "validation raw SPD violation rate",
            "validation_spd_violation_rate",
        ),
        (
            "test raw SPD violation rate",
            "test_spd_violation_rate",
        ),
        (
            "validation min raw eigenvalue",
            "validation_min_raw_eigenvalue",
        ),
        (
            "test min raw eigenvalue",
            "test_min_raw_eigenvalue",
        ),
        (
            "validation min projected eigenvalue",
            "validation_min_projected_eigenvalue",
        ),
        (
            "test min projected eigenvalue",
            "test_min_projected_eigenvalue",
        ),
    ]

    print_summary_table(
        results,
        metrics,
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

    method = results[
        0
    ][
        "method"
    ]

    print(
        f"log directory : "
        f"{log_directory}"
    )

    print(
        f"method        : "
        f"{method}"
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

    if method == "adam":
        print_adam_runs(
            results
        )

        print_adam_summary(
            results
        )

    else:
        print_arff_runs(
            results
        )

        print_arff_summary(
            results
        )


if __name__ == "__main__":
    main()