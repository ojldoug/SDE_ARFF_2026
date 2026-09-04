"""Parameter-budget matching for the Adam MLP baseline."""

from __future__ import annotations

from typing import Sequence, Tuple


def mlp_parameter_count(
    input_dimension: int,
    output_dimension: int,
    hidden_sizes: Sequence[int],
) -> int:
    """
    Number of trainable weights and biases in a fully connected MLP.
    """
    sizes = (
        input_dimension,
        *hidden_sizes,
        output_dimension,
    )

    return sum(
        (n_in + 1) * n_out
        for n_in, n_out in zip(
            sizes[:-1],
            sizes[1:],
        )
    )


def fourier_parameter_count(
    input_dimension: int,
    output_dimension: int,
    n_frequencies: int,
) -> int:
    """
    Number of trainable parameters in the Adam Fourier model:

        omega : D x K
        amp   : 2K x q
    """
    return (
        n_frequencies
        * (
            input_dimension
            + 2 * output_dimension
        )
    )


def total_fourier_parameter_count(
    *,
    state_dimension: int,
    covariance_output_dimension: int,
    n_frequencies: int,
) -> int:
    """
    Combined drift + covariance parameter count.
    """
    return (
        fourier_parameter_count(
            state_dimension,
            state_dimension,
            n_frequencies,
        )
        + fourier_parameter_count(
            state_dimension,
            covariance_output_dimension,
            n_frequencies,
        )
    )


def total_mlp_parameter_count(
    *,
    state_dimension: int,
    covariance_output_dimension: int,
    hidden_sizes: Sequence[int],
) -> int:
    """
    Combined drift + covariance MLP parameter count.
    """
    return (
        mlp_parameter_count(
            state_dimension,
            state_dimension,
            hidden_sizes,
        )
        + mlp_parameter_count(
            state_dimension,
            covariance_output_dimension,
            hidden_sizes,
        )
    )


def matched_two_layer_width(
    *,
    state_dimension: int,
    covariance_output_dimension: int,
    n_frequencies: int,
    maximum_width: int = 4096,
) -> Tuple[int, int, int]:
    """
    Find the equal width of two hidden layers whose total parameter
    count is closest to the corresponding Adam Fourier parameter count.

    Returns
    -------
    width:
        Selected hidden-layer width.

    mlp_parameters:
        Total drift + covariance MLP parameter count.

    fourier_parameters:
        Target Adam Fourier parameter count.
    """
    if state_dimension <= 0:
        raise ValueError(
            "state_dimension must be positive."
        )

    if covariance_output_dimension <= 0:
        raise ValueError(
            "covariance_output_dimension must be positive."
        )

    if n_frequencies <= 0:
        raise ValueError(
            "n_frequencies must be positive."
        )

    if maximum_width <= 0:
        raise ValueError(
            "maximum_width must be positive."
        )

    target = total_fourier_parameter_count(
        state_dimension=state_dimension,
        covariance_output_dimension=(
            covariance_output_dimension
        ),
        n_frequencies=n_frequencies,
    )

    best_width = None
    best_count = None
    best_difference = None

    for width in range(
        1,
        maximum_width + 1,
    ):
        count = total_mlp_parameter_count(
            state_dimension=state_dimension,
            covariance_output_dimension=(
                covariance_output_dimension
            ),
            hidden_sizes=(
                width,
                width,
            ),
        )

        difference = abs(
            count - target
        )

        if (
            best_difference is None
            or difference < best_difference
        ):
            best_width = width
            best_count = count
            best_difference = difference

    assert best_width is not None
    assert best_count is not None

    return (
        best_width,
        best_count,
        target,
    )