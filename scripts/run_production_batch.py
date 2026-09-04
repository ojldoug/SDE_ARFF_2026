#!/usr/bin/env python3
"""
Run one canonical production batch for ARFF or Adam.

The script executes independent seeds sequentially, using the canonical
experiment runners:

    scripts/run_arff_experiment.py
    scripts/run_adam_fourier_experiment.py

Production logs are written to

    results/production/{method}_{experiment}/seed_{seed}.txt

Features
--------
- Uses EvaluationConfig.n_runs by default.
- Runs seeds sequentially.
- Stops immediately if a seed fails.
- Refuses to overwrite nonempty existing logs by default.
- Can resume an interrupted batch by skipping completed logs.
- Records batch metadata and nvidia-smi snapshots.
- Uses one explicitly selected CUDA device.
- Does not duplicate training logic.
- Waits for the GPU machine to be idle before starting each seed.

Examples
--------

    python scripts/run_production_batch.py \
        arff ex2 \
        --device 0

    python scripts/run_production_batch.py \
        adam ex7 \
        --device 0

Resume an interrupted batch:

    python scripts/run_production_batch.py \
        arff ex2 \
        --device 0 \
        --resume
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.config import get_config


METHODS = (
    "arff",
    "adam",
)

EXPERIMENTS = tuple(
    f"ex{i}"
    for i in range(1, 9)
)


def timestamp():
    return datetime.now(
        timezone.utc
    ).astimezone().isoformat(
        timespec="seconds"
    )


def runner_path(
    method: str,
) -> Path:
    if method == "arff":
        return (
            REPO_ROOT
            / "scripts"
            / "run_arff_experiment.py"
        )

    if method == "adam":
        return (
            REPO_ROOT
            / "scripts"
            / "run_adam_fourier_experiment.py"
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


def run_nvidia_smi():
    """
    Return a complete human-readable nvidia-smi snapshot.
    """
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    except FileNotFoundError as exc:
        raise RuntimeError(
            "nvidia-smi was not found. "
            "Production timing requires an NVIDIA GPU machine."
        ) from exc

    output = (
        completed.stdout
        + completed.stderr
    )

    return (
        completed.returncode,
        output,
    )


def gpu_compute_processes():
    """
    Return currently active NVIDIA compute processes.

    Each returned line contains fields reported by nvidia-smi:

        pid, process_name, used_gpu_memory

    The query covers all GPUs on the machine. Therefore a production
    seed is started only when no existing NVIDIA compute process is
    present on any GPU.
    """
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps="
                "pid,process_name,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

    except FileNotFoundError as exc:
        raise RuntimeError(
            "nvidia-smi was not found while "
            "checking machine occupancy."
        ) from exc

    if completed.returncode != 0:
        raise RuntimeError(
            "nvidia-smi failed while checking "
            "whether the machine is idle:\n"
            + completed.stdout
            + completed.stderr
        )

    return [
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def write_text(
    path: Path,
    text: str,
):
    with path.open(
        "a",
        encoding="utf-8",
    ) as handle:
        handle.write(
            text
        )

        if (
            text
            and not text.endswith(
                "\n"
            )
        ):
            handle.write(
                "\n"
            )


def log_machine_snapshot(
    metadata_path: Path,
    *,
    label: str,
):
    returncode, output = (
        run_nvidia_smi()
    )

    text = (
        "\n"
        + "=" * 80
        + "\n"
        + f"{label}\n"
        + f"time: {timestamp()}\n"
        + f"nvidia-smi return code: {returncode}\n"
        + "=" * 80
        + "\n"
        + output
        + "\n"
    )

    write_text(
        metadata_path,
        text,
    )


def wait_for_machine_idle(
    metadata_path: Path,
    *,
    poll_seconds: int = 300,
):
    """
    Wait until no NVIDIA compute process is present anywhere on the
    machine.

    The check occurs only between production seeds. Polling therefore
    does not contribute to the measured algorithm time of a seed.

    If another user occupies a GPU, the production batch waits rather
    than starting a potentially contaminated timing run.
    """
    waiting = False

    while True:
        processes = (
            gpu_compute_processes()
        )

        if not processes:
            if waiting:
                message = (
                    "\n"
                    + "=" * 80
                    + "\n"
                    + "GPU MACHINE IDLE — RESUMING\n"
                    + f"time: {timestamp()}\n"
                    + "=" * 80
                    + "\n"
                )

                print(
                    message,
                    end="",
                    flush=True,
                )

                write_text(
                    metadata_path,
                    message,
                )

            return

        if not waiting:
            message = (
                "\n"
                + "=" * 80
                + "\n"
                + "GPU MACHINE BUSY — WAITING\n"
                + f"time: {timestamp()}\n"
                + f"poll interval: {poll_seconds} s\n"
                + "=" * 80
                + "\n"
                + "\n".join(
                    processes
                )
                + "\n"
            )

            print(
                message,
                end="",
                flush=True,
            )

            write_text(
                metadata_path,
                message,
            )

            waiting = True

        time.sleep(
            poll_seconds
        )


def log_is_complete(
    path: Path,
    method: str,
) -> bool:
    if not path.is_file():
        return False

    if path.stat().st_size == 0:
        return False

    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    if method == "adam":
        required = (
            "algorithm time",
            "validation",
            "test",
            "best validation NLL",
        )

    else:
        required = (
            "algorithm time",
            "validation",
            "test",
            "raw SPD violation rate",
        )

    return all(
        token in text
        for token in required
    )


def run_seed(
    *,
    method: str,
    experiment: str,
    seed: int,
    device: int,
    output_path: Path,
    metadata_path: Path,
):
    command = [
        sys.executable,
        str(
            runner_path(
                method
            )
        ),
        experiment,
        "--seed",
        str(
            seed
        ),
    ]

    environment = (
        os.environ.copy()
    )

    environment[
        "CUDA_VISIBLE_DEVICES"
    ] = str(
        device
    )

    write_text(
        metadata_path,
        (
            "\n"
            + "=" * 80
            + "\n"
            + f"seed {seed} start\n"
            + f"time: {timestamp()}\n"
            + f"command: {' '.join(command)}\n"
            + f"CUDA_VISIBLE_DEVICES={device}\n"
            + "=" * 80
            + "\n"
        ),
    )

    log_machine_snapshot(
        metadata_path,
        label=(
            f"GPU snapshot before seed {seed}"
        ),
    )

    print(
        "=" * 80
    )

    print(
        f"{method.upper()} "
        f"{experiment} "
        f"seed {seed}"
    )

    print(
        f"Started: {timestamp()}"
    )

    print(
        "=" * 80
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as output_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert (
            process.stdout
            is not None
        )

        for line in process.stdout:
            print(
                line,
                end="",
                flush=True,
            )

            output_handle.write(
                line
            )

            output_handle.flush()

        returncode = (
            process.wait()
        )

    log_machine_snapshot(
        metadata_path,
        label=(
            f"GPU snapshot after seed {seed}"
        ),
    )

    write_text(
        metadata_path,
        (
            f"seed {seed} end\n"
            f"time: {timestamp()}\n"
            f"return code: {returncode}\n"
        ),
    )

    if returncode != 0:
        raise RuntimeError(
            f"{method} {experiment} "
            f"seed {seed} failed "
            f"with return code "
            f"{returncode}."
        )

    if not log_is_complete(
        output_path,
        method,
    ):
        raise RuntimeError(
            f"{method} {experiment} "
            f"seed {seed} returned "
            "success but the output log "
            "does not look complete."
        )

    print(
        f"Completed: {timestamp()}"
    )

    print()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "method",
        choices=METHODS,
    )

    parser.add_argument(
        "experiment",
        choices=EXPERIMENTS,
    )

    parser.add_argument(
        "--device",
        type=int,
        required=True,
        help=(
            "Physical CUDA device index "
            "to expose to the runner."
        ),
    )

    parser.add_argument(
        "--start-seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--n-runs",
        type=int,
        default=None,
        help=(
            "Number of runs. Defaults to "
            "config.evaluation.n_runs."
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Skip logs that already appear "
            "complete. Incomplete logs are "
            "replaced."
        ),
    )

    parser.add_argument(
        "--idle-poll-seconds",
        type=int,
        default=300,
        help=(
            "Seconds between GPU-idle checks "
            "when another compute process is "
            "using the machine. Default: 300."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.device < 0:
        raise ValueError(
            "--device must be non-negative."
        )

    if args.start_seed < 0:
        raise ValueError(
            "--start-seed must be "
            "non-negative."
        )

    if args.idle_poll_seconds <= 0:
        raise ValueError(
            "--idle-poll-seconds must be positive."
        )

    config = get_config(
        args.experiment
    )

    if args.n_runs is None:
        n_runs = (
            config.evaluation.n_runs
        )
    else:
        n_runs = (
            args.n_runs
        )

    if n_runs <= 0:
        raise ValueError(
            "--n-runs must be positive."
        )

    first_seed = (
        args.start_seed
    )

    last_seed = (
        first_seed
        + n_runs
        - 1
    )

    output_directory = (
        REPO_ROOT
        / "results"
        / "production"
        / (
            f"{args.method}_"
            f"{args.experiment}"
        )
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata_path = (
        output_directory
        / "batch_metadata.txt"
    )

    write_text(
        metadata_path,
        (
            "\n"
            + "#" * 80
            + "\n"
            + "PRODUCTION BATCH\n"
            + f"method: {args.method}\n"
            + f"experiment: {args.experiment}\n"
            + f"device: {args.device}\n"
            + f"first seed: {first_seed}\n"
            + f"last seed: {last_seed}\n"
            + f"n runs: {n_runs}\n"
            + f"resume: {args.resume}\n"
            + (
                "idle poll seconds: "
                f"{args.idle_poll_seconds}\n"
            )
            + f"batch start: {timestamp()}\n"
            + "#" * 80
            + "\n"
        ),
    )

    for seed in range(
        first_seed,
        last_seed + 1,
    ):
        output_path = (
            output_directory
            / f"seed_{seed}.txt"
        )

        if output_path.exists():
            complete = (
                log_is_complete(
                    output_path,
                    args.method,
                )
            )

            if args.resume:
                if complete:
                    print(
                        f"Skipping completed "
                        f"seed {seed}: "
                        f"{output_path}"
                    )

                    continue

                print(
                    f"Replacing incomplete "
                    f"seed {seed}: "
                    f"{output_path}"
                )

            else:
                raise FileExistsError(
                    "Refusing to overwrite "
                    f"{output_path}. "
                    "Use --resume to skip "
                    "completed runs and "
                    "replace incomplete ones."
                )

        wait_for_machine_idle(
            metadata_path,
            poll_seconds=(
                args.idle_poll_seconds
            ),
        )

        run_seed(
            method=args.method,
            experiment=(
                args.experiment
            ),
            seed=seed,
            device=args.device,
            output_path=(
                output_path
            ),
            metadata_path=(
                metadata_path
            ),
        )

    write_text(
        metadata_path,
        (
            "\n"
            + "#" * 80
            + "\n"
            + "BATCH COMPLETE\n"
            + f"time: {timestamp()}\n"
            + "#" * 80
            + "\n"
        ),
    )

    print(
        "=" * 80
    )

    print(
        "PRODUCTION BATCH COMPLETE"
    )

    print(
        f"{args.method.upper()} "
        f"{args.experiment}"
    )

    print(
        f"seeds {first_seed}--{last_seed}"
    )

    print(
        f"Finished: {timestamp()}"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":
    main()