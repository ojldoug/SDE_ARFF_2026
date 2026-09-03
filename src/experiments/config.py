"""
Configuration for the reproducible ARFF SDE experiments.

These configurations define the final reproducible experiments.
They are not intended to reconstruct undocumented historical notebook
states exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DataConfig:
    seed: int
    n_trajectories: Optional[int] = None
    trajectory_time: Optional[float] = None
    observation_lag: Optional[float] = None
    em_substeps: Optional[int] = None
    target_samples: Optional[int] = None
    grid_step: Optional[float] = None


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 2026
    train_fraction: float = 0.8
    validation_fraction: float = 0.1
    test_fraction: float = 0.1


@dataclass(frozen=True)
class ARFFConfig:
    M_min: int = 300
    M_max: int = 300
    lambda_reg: float = 1e-3
    gamma: float = 1.0
    delta: float = 0.2
    resampling: bool = True
    metropolis_test: bool = False
    n_folds: int = 5


@dataclass(frozen=True)
class AdamConfig:
    epochs: int = 300
    learning_rate: float = 1e-4
    batch_size: int = 2**8


@dataclass(frozen=True)
class EvaluationConfig:
    n_runs: int = 30
    spd_epsilon: float = 1e-6
    covariance_rmse_before_projection: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    fourier_frequencies: Optional[int]
    data: DataConfig
    split: SplitConfig = SplitConfig()
    arff: ARFFConfig = ARFFConfig()
    adam: AdamConfig = AdamConfig()
    evaluation: EvaluationConfig = EvaluationConfig()


COMMON_SPLIT = SplitConfig()
COMMON_ARFF = ARFFConfig()
COMMON_ADAM = AdamConfig()
COMMON_EVALUATION = EvaluationConfig()


CONFIGS = {
    "ex1": ExperimentConfig(
        name="ex1",
        fourier_frequencies=2**8,
        data=DataConfig(
            seed=0,
            n_trajectories=10_000,
            trajectory_time=1e-2,
            observation_lag=1e-2,
            em_substeps=1000,
            target_samples=10_000,
        ),
        arff=ARFFConfig(
            M_min=25,
            M_max=25,
        ),
    ),

    "ex2": ExperimentConfig(
        name="ex2",
        fourier_frequencies=2**7,
        data=DataConfig(
            seed=0,
            n_trajectories=10_000,
            trajectory_time=1e-2,
            observation_lag=1e-2,
            em_substeps=1000,
            target_samples=10_000,
        ),
        arff=ARFFConfig(
            M_min=25,
            M_max=25,
        ),
    ),

    "ex3": ExperimentConfig(
        name="ex3",
        fourier_frequencies=2**10,
        data=DataConfig(
            seed=0,
            n_trajectories=100_000,
            trajectory_time=1e-2,
            observation_lag=1e-2,
            em_substeps=1000,
            target_samples=100_000,
        ),
        arff=ARFFConfig(
            M_min=50,
            M_max=50,
        ),
        evaluation=EvaluationConfig(
            spd_epsilon=1e-2,
        ),
    ),

    "ex4": ExperimentConfig(
        name="ex4",
        fourier_frequencies=2**7,
        data=DataConfig(
            seed=0,
            n_trajectories=10_000,
            trajectory_time=1e-2,
            observation_lag=1e-2,
            em_substeps=1000,
            target_samples=10_000,
        ),
        arff=ARFFConfig(
            M_min=50,
            M_max=50,
        ),
    ),

    "ex5": ExperimentConfig(
        name="ex5",
        fourier_frequencies=2**8,
        data=DataConfig(
            seed=1,
            n_trajectories=250,
            trajectory_time=4.0,
            observation_lag=1e-2,
            target_samples=None,
        ),
        arff=ARFFConfig(
            M_min=50,
            M_max=50,
        ),
    ),

    "ex6": ExperimentConfig(
        name="ex6",
        fourier_frequencies=2**9,
        data=DataConfig(
            seed=1,
            trajectory_time=2.0,
            grid_step=1e-3,
            observation_lag=5e-7,
            target_samples=995_004,
        ),
        arff=ARFFConfig(
            M_min=200,
            M_max=200,
        ),
    ),

    "ex7": ExperimentConfig(
        name="ex7",
        fourier_frequencies=2**9,
        data=DataConfig(
            seed=0,
            n_trajectories=100_000,
            trajectory_time=1e-3,
            observation_lag=1e-3,
            em_substeps=1000,
            target_samples=100_000,
        ),
        arff=ARFFConfig(
            M_min=25,
            M_max=25,
        ),
    ),

    "ex8": ExperimentConfig(
        name="ex8",
        fourier_frequencies=2**9,
        data=DataConfig(
            seed=0,
            n_trajectories=100_000,
            trajectory_time=1e-4,
            observation_lag=1e-4,
            em_substeps=1000,
            target_samples=100_000,
        ),
        arff=ARFFConfig(
            M_min=25,
            M_max=25,
        ),
        evaluation=EvaluationConfig(
            spd_epsilon=1e-3,
        ),
    ),
}


def get_config(name: str) -> ExperimentConfig:
    try:
        return CONFIGS[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown experiment: {name}"
        ) from exc