#!/usr/bin/env python3
"""Report capacity-matched model sizes for all experiments."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.config import get_config
from src.experiments.definitions import get_experiment
from src.experiments.model_size import (
    covariance_output_dimension,
    fourier_sde_model_size,
    matching_shallow_tanh_width,
    shallow_tanh_sde_model_size,
)


def main():
    print(
        f"{'Exp':<5}"
        f"{'K':>8}"
        f"{'FF params':>14}"
        f"{'tanh W':>10}"
        f"{'tanh params':>14}"
        f"{'diff':>10}"
    )
    print("-" * 61)

    for i in range(1, 9):
        name = f"ex{i}"

        config = get_config(name)
        definition = get_experiment(name)

        if config.K is None:
            raise ValueError(
                f"No Fourier frequency count defined for {name}."
            )

        input_dimension = definition.state_dimension
        drift_dimension = definition.n_dimensions

        covariance_dimension = covariance_output_dimension(
            state_dimension=definition.n_dimensions,
            diff_type=definition.diff_type,
        )

        ff = fourier_sde_model_size(
            input_dimension=input_dimension,
            drift_dimension=drift_dimension,
            covariance_dimension=covariance_dimension,
            n_frequencies=config.K,
        )

        width = matching_shallow_tanh_width(
            target_parameters=ff.total_parameters,
            input_dimension=input_dimension,
            drift_dimension=drift_dimension,
            covariance_dimension=covariance_dimension,
        )

        tanh = shallow_tanh_sde_model_size(
            input_dimension=input_dimension,
            drift_dimension=drift_dimension,
            covariance_dimension=covariance_dimension,
            width=width,
        )

        difference = (
            tanh.total_parameters
            - ff.total_parameters
        )

        print(
            f"{name:<5}"
            f"{config.K:>8}"
            f"{ff.total_parameters:>14}"
            f"{width:>10}"
            f"{tanh.total_parameters:>14}"
            f"{difference:>10}"
        )


if __name__ == "__main__":
    main()