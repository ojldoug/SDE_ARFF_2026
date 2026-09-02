#!/usr/bin/env python3
"""Smoke test for model-parameter matching utilities."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.model_size import (
    covariance_output_dimension,
    fourier_sde_model_size,
    matching_shallow_tanh_width,
    shallow_tanh_sde_model_size,
)


def main():
    # Experiment-1-like dimensions.
    input_dimension = 2
    drift_dimension = 2
    covariance_dimension = covariance_output_dimension(
        state_dimension=2,
        diff_type="diagonal",
    )
    K = 256

    ff = fourier_sde_model_size(
        input_dimension=input_dimension,
        drift_dimension=drift_dimension,
        covariance_dimension=covariance_dimension,
        n_frequencies=K,
    )

    # Each model has
    #   2*256 frequency coordinates
    #   + 2*256*2 amplitudes
    # = 1536 parameters.
    assert ff.drift_parameters == 1536
    assert ff.covariance_parameters == 1536
    assert ff.total_parameters == 3072

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

    # Widths 306 and 307 bracket the Fourier model size:
    #
    # P_tanh(W) = 10W + 4.
    assert width == 307
    assert tanh.total_parameters == 3074

    neighboring = shallow_tanh_sde_model_size(
        input_dimension=input_dimension,
        drift_dimension=drift_dimension,
        covariance_dimension=covariance_dimension,
        width=306,
    )

    assert abs(
        tanh.total_parameters - ff.total_parameters
    ) <= abs(
        neighboring.total_parameters - ff.total_parameters
    )

    print("Model-size smoke test")
    print("---------------------")
    print(f"Fourier frequencies   : {K}")
    print(f"Fourier parameters    : {ff.total_parameters}")
    print(f"matched tanh width    : {width}")
    print(f"tanh parameters       : {tanh.total_parameters}")
    print(
        "parameter difference : "
        f"{tanh.total_parameters - ff.total_parameters}"
    )
    print()
    print("All model-size checks passed.")


if __name__ == "__main__":
    main()