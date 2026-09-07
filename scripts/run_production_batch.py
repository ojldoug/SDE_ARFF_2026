#!/usr/bin/env python3
"""
Run one canonical production batch.

Supported methods:

    arff
    adam
    adam_split
    mlp

Each seed produces two archival outputs:

    seed_N.txt
    seed_N_artifacts.npz

A seed is considered complete only when both outputs pass integrity
checks.

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
sys.path.insert(
    0,
    str(REPO_ROOT),
)

from src.experiments.config import (
    get_config,
)


METHODS = (
    "arff",
    "adam",
    "adam_split",
    "mlp",
)

EXPERIMENTS = tuple(
    f"ex{i}"
    for i in range(
        1,
        9,
    )
)


def timestamp():
    return (
        datetime.now(
            timezone.utc
        )
        .astimezone()
        .isoformat(
            timespec="seconds"
        )
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

    if method == "adam_split":
        return (
            REPO_ROOT
            / "scripts"
            / "run_adam_split_fourier_experiment.py"
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
        + (
            "nvidia-smi return code: "
            f"{returncode}\n"
        )
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

    This occurs only between production seeds.
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
                + (
                    "poll interval: "
                    f"{poll_seconds} s\n"
                )
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

    if method in (
        "adam",
        "mlp",
    ):
        required = (
            "algorithm time",
            "validation",
            "test",
            "best validation NLL",
            "artifact",
        )

    elif method == "adam_split":
        required = (
            "algorithm time",
            "first-call/JIT time",
            "final drift best validation MSE",
            "covariance best validation NLL",
            "validation",
            "test",
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


def _finite_array(
    data,
    key: str,
) -> bool:
    array = np.asarray(
        data[
            key
        ]
    )

    return (
        array.size > 0
        and np.all(
            np.isfinite(
                array
            )
        )
    )


def _valid_history(
    values,
) -> bool:
    values = np.asarray(
        values
    )

    return (
        values.ndim == 1
        and len(values) > 0
        and np.all(
            np.isfinite(
                values
            )
        )
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

    Full model round-trip tests belong to the reproducibility tests,
    not the production scheduler.
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

            # ------------------------------------------------------
            # Method-specific schemas.
            # ------------------------------------------------------

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

            elif method == "adam_split":
                expected_method = (
                    "adam_split_fourier"
                )

                required = (
                    "artifact_version",
                    "method",
                    "experiment",
                    "seed",
                    "diff_type",
                    "fourier_frequencies",
                    "n_folds",
                    "fold_seed",
                    "epochs_per_regression",
                    "batch_size",
                    "learning_rate",
                    "git_commit",
                    "git_dirty",
                    "hostname",
                    "python_version",
                    "numpy_version",
                    "jax_version",
                    "jax_backend",
                    "n_train",
                    "n_validation",
                    "n_test",
                    "algorithm_time",
                    "compilation_time",
                    "end_to_end_time",
                    "internal_algorithm_time",
                    "fold_algorithm_times",
                    "crossfit_algorithm_time",
                    "final_drift_start_offset",
                    "final_drift_algorithm_time",
                    "final_drift_end_offset",
                    "final_drift_best_epoch",
                    "final_drift_best_validation_mse",
                    "final_drift_training_mse",
                    "final_drift_validation_mse",
                    "final_drift_cumulative_time",
                    "final_drift_global_cumulative_time",
                    "covariance_start_offset",
                    "covariance_algorithm_time",
                    "covariance_end_offset",
                    "covariance_best_epoch",
                    "covariance_best_validation_nll",
                    "covariance_training_nll",
                    "covariance_validation_nll",
                    "covariance_cumulative_time",
                    "covariance_global_cumulative_time",
                    "fold_best_epochs",
                    "fold_best_validation_mse",
                    "fold_id",
                    "train_nll",
                    "train_drift_rmse",
                    "train_covariance_rmse",
                    "validation_nll",
                    "validation_drift_rmse",
                    "validation_covariance_rmse",
                    "test_nll",
                    "test_drift_rmse",
                    "test_covariance_rmse",
                    "test_min_covariance_eig",
                    "test_max_covariance_eig",
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
                data[
                    "method"
                ].item()
                != expected_method
            ):
                return False

            if (
                data[
                    "experiment"
                ].item()
                != experiment
            ):
                return False

            if (
                int(
                    data[
                        "seed"
                    ].item()
                )
                != seed
            ):
                return False

            if method == "adam_split":
                if int(
                    data[
                        "artifact_version"
                    ].item()
                ) < 4:
                    return False

            # ------------------------------------------------------
            # Parameter integrity.
            # ------------------------------------------------------

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
                if not _finite_array(
                    data,
                    key,
                ):
                    return False

            # ------------------------------------------------------
            # Timing integrity.
            # ------------------------------------------------------

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

            if method == "adam_split":
                compilation_time = float(
                    data[
                        "compilation_time"
                    ].item()
                )

                end_to_end_time = float(
                    data[
                        "end_to_end_time"
                    ].item()
                )

                if (
                    not np.isfinite(
                        compilation_time
                    )
                    or compilation_time < 0.0
                ):
                    return False

                if (
                    not np.isfinite(
                        end_to_end_time
                    )
                    or end_to_end_time <= 0.0
                ):
                    return False

                if not np.isclose(
                    end_to_end_time,
                    algorithm_time
                    + compilation_time,
                    rtol=1e-7,
                    atol=1e-6,
                ):
                    return False

            # ------------------------------------------------------
            # Archival split-Adam timing integrity.
            # ------------------------------------------------------

            if method == "adam_split":
                fold_times = np.asarray(
                    data[
                        "fold_algorithm_times"
                    ],
                    dtype=np.float64,
                )

                if (
                    fold_times.ndim != 1
                    or len(fold_times) != 5
                    or not np.all(
                        np.isfinite(
                            fold_times
                        )
                    )
                    or not np.all(
                        fold_times > 0.0
                    )
                ):
                    return False

                crossfit_time = float(
                    data[
                        "crossfit_algorithm_time"
                    ].item()
                )

                final_drift_start = float(
                    data[
                        "final_drift_start_offset"
                    ].item()
                )

                final_drift_time = float(
                    data[
                        "final_drift_algorithm_time"
                    ].item()
                )

                final_drift_end = float(
                    data[
                        "final_drift_end_offset"
                    ].item()
                )

                covariance_start = float(
                    data[
                        "covariance_start_offset"
                    ].item()
                )

                covariance_time_value = float(
                    data[
                        "covariance_algorithm_time"
                    ].item()
                )

                covariance_end = float(
                    data[
                        "covariance_end_offset"
                    ].item()
                )

                internal_time = float(
                    data[
                        "internal_algorithm_time"
                    ].item()
                )

                scalar_times = np.asarray(
                    [
                        crossfit_time,
                        final_drift_start,
                        final_drift_time,
                        final_drift_end,
                        covariance_start,
                        covariance_time_value,
                        covariance_end,
                        internal_time,
                    ],
                    dtype=np.float64,
                )

                if not np.all(
                    np.isfinite(
                        scalar_times
                    )
                ):
                    return False

                if not (
                    0.0
                    < crossfit_time
                    <= final_drift_start
                    < covariance_start
                    < internal_time
                ):
                    return False

                if not (
                    final_drift_time > 0.0
                    and covariance_time_value > 0.0
                ):
                    return False

                if not np.isclose(
                    final_drift_end,
                    final_drift_start
                    + final_drift_time,
                    rtol=1e-7,
                    atol=1e-6,
                ):
                    return False

                if not np.isclose(
                    covariance_end,
                    covariance_start
                    + covariance_time_value,
                    rtol=1e-7,
                    atol=1e-6,
                ):
                    return False

                # Internal and external algorithm clocks surround
                # essentially the same fit. Allow a small scheduler /
                # Python bookkeeping difference without weakening the
                # integrity check.
                if not np.isclose(
                    internal_time,
                    algorithm_time,
                    rtol=1e-3,
                    atol=0.05,
                ):
                    return False

                drift_global_time = np.asarray(
                    data[
                        "final_drift_global_cumulative_time"
                    ],
                    dtype=np.float64,
                )

                covariance_global_time = np.asarray(
                    data[
                        "covariance_global_cumulative_time"
                    ],
                    dtype=np.float64,
                )

                for global_time in (
                    drift_global_time,
                    covariance_global_time,
                ):
                    if (
                        global_time.ndim != 1
                        or len(global_time) == 0
                        or not np.all(
                            np.isfinite(
                                global_time
                            )
                        )
                        or not np.all(
                            np.diff(
                                global_time
                            )
                            >= 0.0
                        )
                    ):
                        return False

                if (
                    drift_global_time[0]
                    < final_drift_start
                    or drift_global_time[-1]
                    > internal_time
                ):
                    return False

                if (
                    covariance_global_time[0]
                    < covariance_start
                    or covariance_global_time[-1]
                    > internal_time
                ):
                    return False

                endpoint_keys = (
                    "train_nll",
                    "train_drift_rmse",
                    "train_covariance_rmse",
                    "validation_nll",
                    "validation_drift_rmse",
                    "validation_covariance_rmse",
                    "test_nll",
                    "test_drift_rmse",
                    "test_covariance_rmse",
                    "test_min_covariance_eig",
                    "test_max_covariance_eig",
                )

                for key in endpoint_keys:
                    value = float(
                        data[
                            key
                        ].item()
                    )

                    if not np.isfinite(
                        value
                    ):
                        return False

                if (
                    float(
                        data[
                            "test_min_covariance_eig"
                        ].item()
                    )
                    <= 0.0
                ):
                    return False

            # ------------------------------------------------------
            # Joint Adam / MLP histories.
            # ------------------------------------------------------

            if method in (
                "adam",
                "mlp",
            ):
                training = np.asarray(
                    data[
                        "training_nll"
                    ]
                )

                validation = np.asarray(
                    data[
                        "validation_nll"
                    ]
                )

                cumulative = np.asarray(
                    data[
                        "cumulative_time"
                    ]
                )

                if not (
                    _valid_history(
                        training
                    )
                    and _valid_history(
                        validation
                    )
                    and _valid_history(
                        cumulative
                    )
                ):
                    return False

                if not (
                    len(training)
                    == len(validation)
                    == len(cumulative)
                ):
                    return False

                if not np.all(
                    np.diff(
                        cumulative
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
                    < len(
                        validation
                    )
                ):
                    return False

                if (
                    int(
                        np.argmin(
                            validation
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
                    validation[
                        best_epoch
                    ],
                    stored_best,
                    rtol=1e-7,
                    atol=1e-7,
                ):
                    return False

            # ------------------------------------------------------
            # Split Adam histories.
            # ------------------------------------------------------

            if method == "adam_split":
                drift_train = np.asarray(
                    data[
                        "final_drift_training_mse"
                    ]
                )

                drift_validation = np.asarray(
                    data[
                        "final_drift_validation_mse"
                    ]
                )

                drift_time = np.asarray(
                    data[
                        "final_drift_cumulative_time"
                    ]
                )

                covariance_train = np.asarray(
                    data[
                        "covariance_training_nll"
                    ]
                )

                covariance_validation = np.asarray(
                    data[
                        "covariance_validation_nll"
                    ]
                )

                covariance_time = np.asarray(
                    data[
                        "covariance_cumulative_time"
                    ]
                )

                histories = (
                    drift_train,
                    drift_validation,
                    drift_time,
                    covariance_train,
                    covariance_validation,
                    covariance_time,
                )

                if not all(
                    _valid_history(
                        history
                    )
                    for history in histories
                ):
                    return False

                if not (
                    len(
                        drift_train
                    )
                    == len(
                        drift_validation
                    )
                    == len(
                        drift_time
                    )
                ):
                    return False

                if not (
                    len(
                        covariance_train
                    )
                    == len(
                        covariance_validation
                    )
                    == len(
                        covariance_time
                    )
                ):
                    return False

                if not np.all(
                    np.diff(
                        drift_time
                    )
                    >= 0.0
                ):
                    return False

                if not np.all(
                    np.diff(
                        covariance_time
                    )
                    >= 0.0
                ):
                    return False

                drift_best_epoch = int(
                    data[
                        "final_drift_best_epoch"
                    ].item()
                )

                covariance_best_epoch = int(
                    data[
                        "covariance_best_epoch"
                    ].item()
                )

                if (
                    int(
                        np.argmin(
                            drift_validation
                        )
                    )
                    != drift_best_epoch
                ):
                    return False

                if (
                    int(
                        np.argmin(
                            covariance_validation
                        )
                    )
                    != covariance_best_epoch
                ):
                    return False

                if not np.isclose(
                    drift_validation[
                        drift_best_epoch
                    ],
                    float(
                        data[
                            "final_drift_best_validation_mse"
                        ].item()
                    ),
                    rtol=1e-7,
                    atol=1e-7,
                ):
                    return False

                if not np.isclose(
                    covariance_validation[
                        covariance_best_epoch
                    ],
                    float(
                        data[
                            "covariance_best_validation_nll"
                        ].item()
                    ),
                    rtol=1e-7,
                    atol=1e-7,
                ):
                    return False

                fold_best_epochs = np.asarray(
                    data[
                        "fold_best_epochs"
                    ]
                )

                fold_best_losses = np.asarray(
                    data[
                        "fold_best_validation_mse"
                    ]
                )

                if (
                    fold_best_epochs.ndim
                    != 1
                    or fold_best_losses.ndim
                    != 1
                    or len(
                        fold_best_epochs
                    )
                    != len(
                        fold_best_losses
                    )
                    or len(
                        fold_best_epochs
                    )
                    == 0
                ):
                    return False

                if not np.all(
                    np.isfinite(
                        fold_best_losses
                    )
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
            + (
                "command: "
                f"{' '.join(command)}\n"
            )
            + (
                "CUDA_VISIBLE_DEVICES="
                f"{device}\n"
            )
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

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Open first so an interrupted seed leaves an obvious incomplete
    # zero/partial log that --resume can safely replace.
    with output_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    returncode = (
        completed.returncode
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
            f"with return code {returncode}."
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
            "Incomplete outputs are replaced."
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
            + (
                f"method: "
                f"{args.method}\n"
            )
            + (
                f"experiment: "
                f"{args.experiment}\n"
            )
            + (
                f"device: "
                f"{args.device}\n"
            )
            + (
                f"first seed: "
                f"{first_seed}\n"
            )
            + (
                f"last seed: "
                f"{last_seed}\n"
            )
            + (
                f"n runs: "
                f"{n_runs}\n"
            )
            + (
                f"resume: "
                f"{args.resume}\n"
            )
            + (
                "idle poll seconds: "
                f"{args.idle_poll_seconds}\n"
            )
            + (
                f"batch start: "
                f"{timestamp()}\n"
            )
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
                        "Skipping completed "
                        f"seed {seed}: "
                        f"{output_path}"
                    )

                    continue

                print(
                    "Replacing incomplete "
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
