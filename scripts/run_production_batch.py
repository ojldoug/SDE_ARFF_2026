#!/usr/bin/env python3
"""
Run one canonical production batch for ARFF, Adam Fourier, or Adam MLP.

Each seed produces two archival outputs:

    seed_N.txt
    seed_N_artifacts.npz

A seed is considered complete only when both outputs pass basic
integrity checks.

The runner executes seeds sequentially on one selected CUDA device,
waits for the whole GPU machine to be idle before starting a new seed,
and stops immediately on failure.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import os
import subprocess
import sys
import time

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.config import get_config


METHODS = (
    "arff",
    "adam",
    "mlp",
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

    if method == "mlp":
        return (
            REPO_ROOT
            / "scripts"
            / "run_adam_mlp_experiment.py"
        )

    raise ValueError(
        f"Unknown method: {method}"
    )


def run_nvidia_smi():
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
            "Production timing requires an "
            "NVIDIA GPU machine."
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
    Return active NVIDIA compute processes on the machine.
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

    This check occurs only between production seeds.
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

    if method in ("adam", "mlp"):
        required = (
            "algorithm time",
            "validation",
            "test",
            "best validation NLL",
            "artifact",
        )

    else:
        required = (
            "algorithm time",
            "validation",
            "test",
            "raw SPD violation rate",
            "artifact",
        )

    return all(
        token in text
        for token in required
    )


def artifact_is_complete(
    path: Path,
    *,
    method: str,
    experiment: str,
    seed: int,
) -> bool:
    """
    Perform cheap integrity checks on one production artifact.

    This does not rerun model evaluation. Full model round-trip tests
    belong to the reproducibility tests, not the production scheduler.
    """
    if not path.is_file():
        return False

    if path.stat().st_size == 0:
        return False

    try:
        with np.load(
            path,
            allow_pickle=False,
        ) as data:
            if method == "adam":
                expected_method = (
                    "adam_fourier"
                )

                required = (
                    "artifact_version",
                    "method",
                    "experiment",
                    "seed",
                    "diff_type",
                    "fourier_frequencies",
                    "epochs",
                    "best_epoch",
                    "best_validation_nll",
                    "training_nll",
                    "validation_nll",
                    "cumulative_time",
                    "algorithm_time",
                    "drift_omega",
                    "drift_amp",
                    "covariance_omega",
                    "covariance_amp",
                )

            elif method == "mlp":
                expected_method = (
                    "adam_mlp"
                )

                required = (
                    "artifact_version",
                    "method",
                    "experiment",
                    "seed",
                    "diff_type",
                    "hidden_width",
                    "hidden_layers",
                    "mlp_parameter_count",
                    "fourier_parameter_count",
                    "epochs",
                    "batch_size",
                    "learning_rate",
                    "best_epoch",
                    "best_validation_nll",
                    "training_nll",
                    "validation_nll",
                    "cumulative_time",
                    "algorithm_time",
                    "drift_weight_0",
                    "drift_bias_0",
                    "covariance_weight_0",
                    "covariance_bias_0",
                )

            else:
                expected_method = (
                    "arff"
                )

                required = (
                    "artifact_version",
                    "method",
                    "experiment",
                    "seed",
                    "diff_type",
                    "fourier_frequencies",
                    "iterations",
                    "n_folds",
                    "fold_seed",
                    "resampling",
                    "metropolis_test",
                    "spd_epsilon",
                    "algorithm_time",
                    "drift_omega",
                    "drift_amp",
                    "covariance_omega",
                    "covariance_amp",
                )

            if not all(
                key in data.files
                for key in required
            ):
                return False

            if (
                data["method"].item()
                != expected_method
            ):
                return False

            if (
                data["experiment"].item()
                != experiment
            ):
                return False

            if (
                int(
                    data["seed"].item()
                )
                != seed
            ):
                return False

            if method == "mlp":
                parameter_keys = [
                    key
                    for key in data.files
                    if (
                        key.startswith(
                            "drift_weight_"
                        )
                        or key.startswith(
                            "drift_bias_"
                        )
                        or key.startswith(
                            "covariance_weight_"
                        )
                        or key.startswith(
                            "covariance_bias_"
                        )
                    )
                ]

                if not parameter_keys:
                    return False

            else:
                parameter_keys = (
                    "drift_omega",
                    "drift_amp",
                    "covariance_omega",
                    "covariance_amp",
                )

            for key in parameter_keys:
                array = np.asarray(
                    data[key]
                )

                if array.size == 0:
                    return False

                if not np.all(
                    np.isfinite(
                        array
                    )
                ):
                    return False

            algorithm_time = float(
                data[
                    "algorithm_time"
                ].item()
            )

            if (
                not np.isfinite(
                    algorithm_time
                )
                or algorithm_time <= 0.0
            ):
                return False

            if method in ("adam", "mlp"):
                training_nll = np.asarray(
                    data[
                        "training_nll"
                    ]
                )

                validation_nll = np.asarray(
                    data[
                        "validation_nll"
                    ]
                )

                cumulative_time = np.asarray(
                    data[
                        "cumulative_time"
                    ]
                )

                if (
                    training_nll.ndim != 1
                    or validation_nll.ndim != 1
                    or cumulative_time.ndim != 1
                ):
                    return False

                if not (
                    len(training_nll)
                    == len(validation_nll)
                    == len(cumulative_time)
                ):
                    return False

                if len(
                    training_nll
                ) == 0:
                    return False

                if not np.all(
                    np.isfinite(
                        training_nll
                    )
                ):
                    return False

                if not np.all(
                    np.isfinite(
                        validation_nll
                    )
                ):
                    return False

                if not np.all(
                    np.isfinite(
                        cumulative_time
                    )
                ):
                    return False

                if not np.all(
                    np.diff(
                        cumulative_time
                    )
                    >= 0.0
                ):
                    return False

                best_epoch = int(
                    data[
                        "best_epoch"
                    ].item()
                )

                if not (
                    0
                    <= best_epoch
                    < len(validation_nll)
                ):
                    return False

                if (
                    int(
                        np.argmin(
                            validation_nll
                        )
                    )
                    != best_epoch
                ):
                    return False

                stored_best = float(
                    data[
                        "best_validation_nll"
                    ].item()
                )

                if not np.isclose(
                    validation_nll[
                        best_epoch
                    ],
                    stored_best,
                    rtol=1e-7,
                    atol=1e-7,
                ):
                    return False

    except (
        OSError,
        ValueError,
        KeyError,
        EOFError,
    ):
        return False

    return True


def seed_is_complete(
    *,
    log_path: Path,
    artifact_path: Path,
    method: str,
    experiment: str,
    seed: int,
) -> bool:
    return (
        log_is_complete(
            log_path,
            method,
        )
        and artifact_is_complete(
            artifact_path,
            method=method,
            experiment=experiment,
            seed=seed,
        )
    )


def run_seed(
    *,
    method: str,
    experiment: str,
    seed: int,
    device: int,
    output_path: Path,
    artifact_path: Path,
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
        "--artifact-path",
        str(
            artifact_path
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

    # Remove a stale artifact before starting. This ensures a failed
    # run can never leave an older artifact paired with a new log.
    if artifact_path.exists():
        artifact_path.unlink()

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

    if not seed_is_complete(
        log_path=output_path,
        artifact_path=artifact_path,
        method=method,
        experiment=experiment,
        seed=seed,
    ):
        raise RuntimeError(
            f"{method} {experiment} "
            f"seed {seed} returned success "
            "but its production log/artifact "
            "pair is incomplete or invalid."
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
            "Skip seeds whose log and artifact "
            "both pass integrity checks. "
            "Incomplete seed outputs are replaced."
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

        artifact_path = (
            output_directory
            / f"seed_{seed}_artifacts.npz"
        )

        complete = seed_is_complete(
            log_path=output_path,
            artifact_path=artifact_path,
            method=args.method,
            experiment=args.experiment,
            seed=seed,
        )

        if (
            output_path.exists()
            or artifact_path.exists()
        ):
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
                    f"seed {seed} outputs in "
                    f"{output_directory}. "
                    "Use --resume to skip "
                    "complete seeds and replace "
                    "incomplete ones."
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
            artifact_path=(
                artifact_path
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