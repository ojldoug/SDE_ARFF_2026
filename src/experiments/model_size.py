"""
Model-size utilities for fair ARFF/Adam comparisons.

We compare methods using the number of scalar model parameters rather
than reusing an architecture-specific symbol such as K.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ModelSize:
    drift_parameters: int
    covariance_parameters: int

    @property
    def total_parameters(self) -> int:
        return (
            self.drift_parameters
            + self.covariance_parameters
        )


def covariance_output_dimension(
    state_dimension: int,
    diff_type: str,
) -> int:
    if diff_type == "diagonal":
        return state_dimension

    if diff_type in {"triangular", "symmetric"}:
        return (
            state_dimension
            * (state_dimension + 1)
            // 2
        )

    raise ValueError(
        f"Unknown diffusion type: {diff_type}"
    )


def fourier_parameter_count(
    input_dimension: int,
    output_dimension: int,
    n_frequencies: int,
) -> int:
    """
    Number of scalar parameters in

        f(x) = [cos(x Omega), sin(x Omega)] A,

    with Omega in R^{d x K} and A in R^{2K x q}.
    """
    if input_dimension <= 0:
        raise ValueError(
            "input_dimension must be positive."
        )

    if output_dimension <= 0:
        raise ValueError(
            "output_dimension must be positive."
        )

    if n_frequencies <= 0:
        raise ValueError(
            "n_frequencies must be positive."
        )

    return (
        input_dimension * n_frequencies
        + 2 * n_frequencies * output_dimension
    )


def fourier_sde_model_size(
    *,
    input_dimension: int,
    drift_dimension: int,
    covariance_dimension: int,
    n_frequencies: int,
) -> ModelSize:
    drift = fourier_parameter_count(
        input_dimension,
        drift_dimension,
        n_frequencies,
    )

    covariance = fourier_parameter_count(
        input_dimension,
        covariance_dimension,
        n_frequencies,
    )

    return ModelSize(
        drift_parameters=drift,
        covariance_parameters=covariance,
    )


def shallow_tanh_parameter_count(
    input_dimension: int,
    output_dimension: int,
    width: int,
) -> int:
    """
    Parameter count for

        x -> tanh(x W1 + b1) W2 + b2

    with one hidden layer of width W.
    """
    if width <= 0:
        raise ValueError("width must be positive.")

    return (
        input_dimension * width
        + width
        + width * output_dimension
        + output_dimension
    )


def shallow_tanh_sde_model_size(
    *,
    input_dimension: int,
    drift_dimension: int,
    covariance_dimension: int,
    width: int,
) -> ModelSize:
    return ModelSize(
        drift_parameters=shallow_tanh_parameter_count(
            input_dimension,
            drift_dimension,
            width,
        ),
        covariance_parameters=shallow_tanh_parameter_count(
            input_dimension,
            covariance_dimension,
            width,
        ),
    )


def matching_shallow_tanh_width(
    *,
    target_parameters: int,
    input_dimension: int,
    drift_dimension: int,
    covariance_dimension: int,
) -> int:
    """
    Return the positive integer shallow-network width whose total
    parameter count is closest to target_parameters.
    """
    if target_parameters <= 0:
        raise ValueError(
            "target_parameters must be positive."
        )

    # For two shallow networks with common width W:
    #
    # P(W)
    #   = W(2d + q_f + q_cov + 2)
    #     + q_f + q_cov.
    slope = (
        2 * input_dimension
        + drift_dimension
        + covariance_dimension
        + 2
    )

    intercept = (
        drift_dimension
        + covariance_dimension
    )

    continuous_width = (
        target_parameters - intercept
    ) / slope

    candidates = {
        max(1, math.floor(continuous_width)),
        max(1, math.ceil(continuous_width)),
    }

    def error(width: int) -> int:
        size = shallow_tanh_sde_model_size(
            input_dimension=input_dimension,
            drift_dimension=drift_dimension,
            covariance_dimension=covariance_dimension,
            width=width,
        )
        return abs(
            size.total_parameters
            - target_parameters
        )

    return min(
        candidates,
        key=lambda width: (error(width), width),
    )