#!/usr/bin/env python3
"""Poster-oriented Experiment 8 covariance recovery figure."""

from pathlib import Path
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adam.mlp import (
    AdamMLPModel,
    MLPParams,
    predict_covariance as predict_mlp_covariance,
)
from src.adam.split_mlp import (
    CovarianceMLPModel,
    covariance_from_split_model,
)
from src.arff.regression import (
    ARFFModel,
    predict as predict_arff,
)
from src.experiments.definitions import get_experiment


JOINT_MLP = (
    ROOT / "results" / "production" / "mlp_ex8"
    / "seed_16_artifacts.npz"
)

SPLIT_MLP = (
    ROOT / "results"
    / "mlp_split_ex8_seed0_finalcheck.npz"
)

ARFF = (
    ROOT / "results"
    / "diagnose_ex8_validation_selected_crossfit_seed0.npz"
)

OUTDIR = ROOT / "results" / "figures"


def mlp_params(z, prefix, n_layers):
    return MLPParams(
        weights=tuple(
            jnp.asarray(z[f"{prefix}_weight_{i}"])
            for i in range(n_layers)
        ),
        biases=tuple(
            jnp.asarray(z[f"{prefix}_bias_{i}"])
            for i in range(n_layers)
        ),
    )


def load_joint_mlp(path):
    with np.load(path, allow_pickle=False) as z:
        n_layers = int(z["hidden_layers"]) + 1

        return AdamMLPModel(
            drift=mlp_params(z, "drift", n_layers),
            covariance=mlp_params(z, "covariance", n_layers),
            diff_type=str(z["diff_type"].item()),
        )


def load_split_mlp(path):
    with np.load(path, allow_pickle=False) as z:
        n_layers = int(z["hidden_layers"]) + 1

        return CovarianceMLPModel(
            covariance=mlp_params(
                z, "covariance", n_layers
            ),
            diff_type=str(z["diff_type"].item()),
            output_dimension=int(z["output_dimension"]),
        )


def load_arff(path):
    with np.load(path, allow_pickle=False) as z:
        return ARFFModel(
            omega=jnp.asarray(z["covariance_omega"]),
            amp=jnp.asarray(z["covariance_amp"]),
        )


def symmetric_vector_to_matrix(values):
    """
    Convert the ex8 symmetric-vector convention to 2x2 matrices.

    Verify ordering from project helper before publication.
    Current project convention is expected to be:
        [Sigma_11, Sigma_21/Sigma_12, Sigma_22].
    """
    values = np.asarray(values)

    if values.shape[-1] != 3:
        raise ValueError(
            f"Expected 3 symmetric entries, got {values.shape}"
        )

    out = np.empty(
        values.shape[:-1] + (2, 2),
        dtype=values.dtype,
    )

    out[..., 0, 0] = values[..., 0]
    out[..., 0, 1] = values[..., 1]
    out[..., 1, 0] = values[..., 1]
    out[..., 1, 1] = values[..., 2]

    return out


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    definition = get_experiment("ex8")

    n = 181
    axis = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")

    points = np.stack(
        [xx.ravel(), yy.ravel()],
        axis=1,
    ).astype(np.float32)

    points_jax = jnp.asarray(points)

    truth = np.asarray(
        definition.covariance(points_jax)
    ).reshape(n, n, 2, 2)

    joint_model = load_joint_mlp(JOINT_MLP)
    joint = np.asarray(
        predict_mlp_covariance(
            joint_model,
            points_jax,
        )
    ).reshape(n, n, 2, 2)

    split_model = load_split_mlp(SPLIT_MLP)
    split = np.asarray(
        covariance_from_split_model(
            split_model,
            points_jax,
        )
    ).reshape(n, n, 2, 2)

    arff_model = load_arff(ARFF)
    arff_vector = np.asarray(
        predict_arff(
            arff_model,
            points_jax,
        )
    )

    arff = symmetric_vector_to_matrix(
        arff_vector
    ).reshape(n, n, 2, 2)

    methods = [
        ("Truth", truth),
        ("Joint MLP\n(seed 16, median run)", joint),
        ("Split MLP\n(seed 0)", split),
        ("ARFF split\n(seed 0)", arff),
    ]

    components = [
        (0, 0, r"$\Sigma_{11}$"),
        (0, 1, r"$\Sigma_{12}$"),
        (1, 1, r"$\Sigma_{22}$"),
    ]

    # Shared scale within each covariance component across all methods.
    scales = []

    for i, j, _ in components:
        values = np.concatenate(
            [a[..., i, j].ravel() for _, a in methods]
        )

        # Robust scale prevents a tiny number of wild pixels from making
        # every scientifically relevant structure invisible.
        lo, hi = np.quantile(values, [0.005, 0.995])

        # Ensure truth range is never clipped.
        truth_values = truth[..., i, j]
        lo = min(lo, float(truth_values.min()))
        hi = max(hi, float(truth_values.max()))

        scales.append((lo, hi))

    fig, axes = plt.subplots(
        len(methods),
        len(components),
        figsize=(9.2, 10.5),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )

    for row, (method_name, covariance) in enumerate(methods):
        for col, (i, j, component_name) in enumerate(components):
            ax = axes[row, col]
            lo, hi = scales[col]

            im = ax.imshow(
                covariance[..., i, j],
                origin="lower",
                extent=(-1, 1, -1, 1),
                vmin=lo,
                vmax=hi,
                interpolation="nearest",
                aspect="equal",
            )

            if row == 0:
                ax.set_title(component_name, fontsize=13)

            if col == 0:
                ax.set_ylabel(
                    method_name + "\n\n$x_2$",
                    fontsize=11,
                )

            if row == len(methods) - 1:
                ax.set_xlabel("$x_1$", fontsize=11)

            ax.set_xticks([-1, 0, 1])
            ax.set_yticks([-1, 0, 1])

            if col == len(components) - 1:
                fig.colorbar(
                    im,
                    ax=ax,
                    fraction=0.046,
                    pad=0.03,
                )

    fig.suptitle(
        "Experiment 8: recovery of a rotating near-singular covariance",
        fontsize=15,
    )

    png = OUTDIR / "ex8_covariance_recovery.png"
    pdf = OUTDIR / "ex8_covariance_recovery.pdf"

    fig.savefig(png, dpi=250, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")

    print(f"saved: {png}")
    print(f"saved: {pdf}")

    # Useful numerical sanity check on the plotting grid.
    print()
    print("GRID RMSE")
    for name, covariance in methods[1:]:
        rmse = np.sqrt(
            np.mean(
                (
                    covariance.astype(np.float64)
                    - truth.astype(np.float64)
                ) ** 2
            )
        )
        print(f"{name.replace(chr(10), ' '):35s}: {rmse:.8e}")

    eig = np.linalg.eigvalsh(arff)
    print()
    print(
        "ARFF raw grid SPD violation rate: "
        f"{np.mean(eig[..., 0] <= 0):.6f}"
    )
    print(
        "ARFF raw grid min eigenvalue     : "
        f"{eig[..., 0].min():.8e}"
    )


if __name__ == "__main__":
    main()
