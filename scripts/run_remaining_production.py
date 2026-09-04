#!/usr/bin/env python3
"""
Run all remaining production batches sequentially.

This weekend runner executes the remaining ARFF, Adam Fourier, and
capacity-matched Adam MLP experiment batches one at a time on a single
CUDA device.

Completed seed log/artifact pairs are respected through --resume, so
the script can be restarted safely after interruption.

The script stops immediately if any batch fails.

Current intended order:
    ex1: ARFF, Adam, MLP
    ex2: ARFF, Adam, MLP
    ex4: ARFF, Adam, MLP
    ex5: ARFF, Adam, MLP
    ex7: ARFF, Adam, MLP
    ex8: ARFF, Adam, MLP
    ex3: ARFF, Adam, MLP
    ex6: MLP

ARFF/Adam batches that are already complete are retained in the list
because run_production_batch.py --resume will skip their completed
seeds safely.

MLP ex6 is placed last because Experiment 6 has the largest dataset.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]

BATCHES = (
    ("arff", "ex1"),
    ("adam", "ex1"),
    ("mlp", "ex1"),

    ("arff", "ex2"),
    ("adam", "ex2"),
    ("mlp", "ex2"),

    ("arff", "ex4"),
    ("adam", "ex4"),
    ("mlp", "ex4"),

    ("arff", "ex5"),
    ("adam", "ex5"),
    ("mlp", "ex5"),

    ("arff", "ex7"),
    ("adam", "ex7"),
    ("mlp", "ex7"),

    ("arff", "ex8"),
    ("adam", "ex8"),
    ("mlp", "ex8"),

    ("arff", "ex3"),
    ("adam", "ex3"),
    ("mlp", "ex3"),

    ("mlp", "ex6"),
)


def now():
    return datetime.now().astimezone().isoformat(
        timespec="seconds"
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--start-at",
        type=int,
        default=0,
        help=(
            "Start from this zero-based batch index "
            "in the predefined batch list."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device < 0:
        raise ValueError(
            "--device must be non-negative."
        )

    if not (
        0 <= args.start_at < len(BATCHES)
    ):
        raise ValueError(
            "--start-at is outside the batch list."
        )

    log_directory = (
        REPO_ROOT
        / "results"
        / "production"
    )

    log_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    master_log = (
        log_directory
        / "remaining_production_master.txt"
    )

    print(
        "=" * 80
    )
    print(
        "REMAINING PRODUCTION RUN"
    )
    print(
        f"started : {now()}"
    )
    print(
        f"device  : {args.device}"
    )
    print(
        f"start   : batch {args.start_at}"
    )
    print(
        "=" * 80
    )

    with master_log.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "\n"
            + "=" * 80
            + "\n"
        )
        handle.write(
            "REMAINING PRODUCTION RUN\n"
        )
        handle.write(
            f"started: {now()}\n"
        )
        handle.write(
            f"device: {args.device}\n"
        )
        handle.write(
            f"start_at: {args.start_at}\n"
        )
        handle.write(
            "=" * 80
            + "\n"
        )

    for index in range(
        args.start_at,
        len(BATCHES),
    ):
        method, experiment = (
            BATCHES[index]
        )

        print()
        print(
            "=" * 80
        )
        print(
            f"BATCH {index + 1}/{len(BATCHES)}"
        )
        print(
            f"{method.upper()} {experiment}"
        )
        print(
            f"started: {now()}"
        )
        print(
            "=" * 80
        )

        with master_log.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                "\n"
                + "=" * 80
                + "\n"
            )
            handle.write(
                f"BATCH {index + 1}/{len(BATCHES)}\n"
            )
            handle.write(
                f"{method.upper()} {experiment}\n"
            )
            handle.write(
                f"started: {now()}\n"
            )
            handle.write(
                "=" * 80
                + "\n"
            )

        command = [
            sys.executable,
            str(
                REPO_ROOT
                / "scripts"
                / "run_production_batch.py"
            ),
            method,
            experiment,
            "--device",
            str(args.device),
            "--resume",
        ]

        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None

        with master_log.open(
            "a",
            encoding="utf-8",
        ) as handle:
            for line in process.stdout:
                print(
                    line,
                    end="",
                    flush=True,
                )

                handle.write(
                    line
                )
                handle.flush()

        returncode = process.wait()

        if returncode != 0:
            message = (
                f"FAILED: {method} {experiment} "
                f"with return code {returncode}"
            )

            print(
                message
            )

            with master_log.open(
                "a",
                encoding="utf-8",
            ) as handle:
                handle.write(
                    message
                    + "\n"
                )
                handle.write(
                    f"time: {now()}\n"
                )

            raise SystemExit(
                returncode
            )

        print(
            f"completed: {now()}"
        )

        with master_log.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                f"completed: {now()}\n"
            )

    print()
    print(
        "=" * 80
    )
    print(
        "ALL REMAINING PRODUCTION BATCHES COMPLETE"
    )
    print(
        f"finished: {now()}"
    )
    print(
        "=" * 80
    )

    with master_log.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            "\n"
            + "=" * 80
            + "\n"
        )
        handle.write(
            "ALL REMAINING PRODUCTION BATCHES COMPLETE\n"
        )
        handle.write(
            f"finished: {now()}\n"
        )
        handle.write(
            "=" * 80
            + "\n"
        )


if __name__ == "__main__":
    main()