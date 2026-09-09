#!/usr/bin/env python3
"""Re-evaluate all saved ex8 joint Adam production models."""

from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.adam.fourier import (
    AdamFourierModel,
    FourierParams,
    gaussian_nll as fourier_nll,
    predict_covariance as fourier_covariance,
    predict_fourier,
)
from src.adam.mlp import (
    AdamMLPModel,
    MLPParams,
    gaussian_nll as mlp_nll,
    predict_covariance as mlp_covariance,
    predict_mlp,
)
from src.experiments.dataset import load_dataset
from src.experiments.definitions import get_experiment


def reconstruct_fourier(z):
    return AdamFourierModel(
        drift=FourierParams(
            omega=jnp.asarray(z["drift_omega"]),
            amp=jnp.asarray(z["drift_amp"]),
        ),
        covariance=FourierParams(
            omega=jnp.asarray(z["covariance_omega"]),
            amp=jnp.asarray(z["covariance_amp"]),
        ),
        diff_type=str(z["diff_type"].item()),
    )


def reconstruct_mlp(z):
    n_layers = int(z["hidden_layers"]) + 1

    def params(prefix):
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

    return AdamMLPModel(
        drift=params("drift"),
        covariance=params("covariance"),
        diff_type=str(z["diff_type"].item()),
    )


def evaluate(model, x, r, h, definition, kind, chunk=8192):
    squared_drift = 0.0
    squared_covariance = 0.0
    nll_sum = 0.0
    n = len(x)

    if kind == "fourier":
        drift_fn = lambda m, xx: predict_fourier(m.drift, xx)
        covariance_fn = fourier_covariance
        nll_fn = fourier_nll
    else:
        drift_fn = lambda m, xx: predict_mlp(m.drift, xx)
        covariance_fn = mlp_covariance
        nll_fn = mlp_nll

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)

        xx = x[start:stop]
        rr = r[start:stop]
        hh = h[start:stop]

        f = np.asarray(drift_fn(model, xx), dtype=np.float64)
        a = np.asarray(covariance_fn(model, xx), dtype=np.float64)

        ft = np.asarray(definition.drift(xx), dtype=np.float64)
        at = np.asarray(definition.covariance(xx), dtype=np.float64)

        squared_drift += len(xx) * float(np.mean((f - ft) ** 2))
        squared_covariance += len(xx) * float(np.mean((a - at) ** 2))
        nll_sum += len(xx) * float(nll_fn(model, xx, rr, hh))

    return (
        np.sqrt(squared_drift / n),
        np.sqrt(squared_covariance / n),
        nll_sum / n,
    )


def summarize(kind, directory, x, r, h, definition):
    rows = []

    for p in sorted(directory.glob("seed_*_artifacts.npz")):
        with np.load(p, allow_pickle=False) as z:
            seed = int(z["seed"])
            model = (
                reconstruct_fourier(z)
                if kind == "fourier"
                else reconstruct_mlp(z)
            )
            drift, covariance, nll = evaluate(
                model, x, r, h, definition, kind
            )
            rows.append(
                (
                    seed,
                    drift,
                    covariance,
                    nll,
                    float(z["algorithm_time"]),
                )
            )

    rows = np.asarray(rows, dtype=float)
    rows = rows[np.argsort(rows[:, 0])]

    cov = rows[:, 2]
    median_value = np.median(cov)
    representative = rows[np.argmin(np.abs(cov - median_value))]

    print()
    print("=" * 72)
    print(kind.upper())
    print("=" * 72)
    print(f"runs: {len(rows)}")

    labels = [
        ("drift RMSE", rows[:, 1]),
        ("covariance RMSE", rows[:, 2]),
        ("NLL", rows[:, 3]),
        ("algorithm time [s]", rows[:, 4]),
    ]

    for label, values in labels:
        print(
            f"{label:20s}: "
            f"mean={np.mean(values):.8e}  "
            f"sd={np.std(values, ddof=1):.8e}  "
            f"median={np.median(values):.8e}"
        )

    print()
    print("representative covariance-median run")
    print(f"  seed            : {int(representative[0])}")
    print(f"  drift RMSE      : {representative[1]:.8e}")
    print(f"  covariance RMSE : {representative[2]:.8e}")
    print(f"  NLL             : {representative[3]:.8e}")
    print(f"  algorithm time  : {representative[4]:.3f} s")

    return rows, int(representative[0])


def main():
    definition = get_experiment("ex8")
    data = load_dataset(ROOT / "data" / "ex8.npz")

    te = data.test_idx
    x = jnp.asarray(data.x[te])
    r = jnp.asarray(data.r[te])
    h = jnp.asarray(data.h[te])

    fourier, fourier_seed = summarize(
        "fourier",
        ROOT / "results" / "production" / "adam_ex8",
        x, r, h, definition,
    )

    mlp, mlp_seed = summarize(
        "mlp",
        ROOT / "results" / "production" / "mlp_ex8",
        x, r, h, definition,
    )

    output = ROOT / "results" / "ex8_joint_production_summary.npz"

    np.savez_compressed(
        output,
        fourier_seed=fourier[:, 0].astype(np.int64),
        fourier_drift_rmse=fourier[:, 1],
        fourier_covariance_rmse=fourier[:, 2],
        fourier_nll=fourier[:, 3],
        fourier_algorithm_time=fourier[:, 4],
        fourier_representative_seed=np.asarray(fourier_seed),
        mlp_seed=mlp[:, 0].astype(np.int64),
        mlp_drift_rmse=mlp[:, 1],
        mlp_covariance_rmse=mlp[:, 2],
        mlp_nll=mlp[:, 3],
        mlp_algorithm_time=mlp[:, 4],
        mlp_representative_seed=np.asarray(mlp_seed),
    )

    print()
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
