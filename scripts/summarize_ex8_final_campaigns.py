#!/usr/bin/env python3

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

METHODS = {
    "ARFF": ROOT / "results/production/arff_ex8_corrected",
    "Split Fourier Adam": ROOT / "results/production/adam_split_ex8",
}

def scalar(z, key):
    if key not in z.files:
        raise KeyError(f"missing key {key}")
    return float(np.asarray(z[key]).item())

def load_campaign(name, directory):
    rows = []

    for p in sorted(directory.glob("seed_*_artifacts.npz")):
        with np.load(p, allow_pickle=False) as z:
            rows.append(
                dict(
                    seed=int(np.asarray(z["seed"]).item()),
                    drift=scalar(z, "test_drift_rmse"),
                    covariance=scalar(z, "test_covariance_rmse"),
                    nll=scalar(z, "test_nll"),
                    time=scalar(z, "algorithm_time"),
                )
            )

    rows = sorted(rows, key=lambda r: r["seed"])

    seeds = [r["seed"] for r in rows]

    if seeds != list(range(30)):
        raise RuntimeError(
            f"{name}: expected seeds 0..29, found {seeds}"
        )

    return rows

def describe(values):
    x = np.asarray(values, dtype=float)
    return {
        "mean": x.mean(),
        "sd": x.std(ddof=1),
        "median": np.median(x),
        "min": x.min(),
        "max": x.max(),
    }

def print_stat(label, values):
    s = describe(values)
    print(
        f"{label:22s} "
        f"mean={s['mean']:.8e}  "
        f"sd={s['sd']:.8e}  "
        f"median={s['median']:.8e}  "
        f"min={s['min']:.8e}  "
        f"max={s['max']:.8e}"
    )

campaigns = {}

for name, directory in METHODS.items():
    rows = load_campaign(name, directory)
    campaigns[name] = rows

    print()
    print("=" * 100)
    print(name)
    print("=" * 100)
    print(f"runs: {len(rows)}")
    print_stat("test drift RMSE", [r["drift"] for r in rows])
    print_stat("test covariance RMSE", [r["covariance"] for r in rows])
    print_stat("test NLL", [r["nll"] for r in rows])
    print_stat(
        "algorithm time [s]",
        [r["time"] for r in rows],
    )

# Existing authoritative joint summaries.
joint_path = ROOT / "results/ex8_joint_production_summary.npz"

with np.load(joint_path, allow_pickle=False) as z:
    joint_fourier_cov = np.asarray(
        z["fourier_covariance_rmse"], dtype=float
    )
    joint_fourier_drift = np.asarray(
        z["fourier_drift_rmse"], dtype=float
    )
    joint_fourier_nll = np.asarray(
        z["fourier_nll"], dtype=float
    )
    joint_fourier_time = np.asarray(
        z["fourier_algorithm_time"], dtype=float
    )

    joint_mlp_cov = np.asarray(
        z["mlp_covariance_rmse"], dtype=float
    )
    joint_mlp_drift = np.asarray(
        z["mlp_drift_rmse"], dtype=float
    )
    joint_mlp_nll = np.asarray(
        z["mlp_nll"], dtype=float
    )
    joint_mlp_time = np.asarray(
        z["mlp_algorithm_time"], dtype=float
    )

split_cov = np.asarray(
    [r["covariance"] for r in campaigns["Split Fourier Adam"]]
)

arff_cov = np.asarray(
    [r["covariance"] for r in campaigns["ARFF"]]
)

print()
print("=" * 100)
print("AUTHORITATIVE JOINT BASELINES")
print("=" * 100)

print("\nJoint Fourier Adam")
print_stat("test drift RMSE", joint_fourier_drift)
print_stat("test covariance RMSE", joint_fourier_cov)
print_stat("test NLL", joint_fourier_nll)
print_stat("algorithm time [s]", joint_fourier_time)

print("\nJoint MLP Adam")
print_stat("test drift RMSE", joint_mlp_drift)
print_stat("test covariance RMSE", joint_mlp_cov)
print_stat("test NLL", joint_mlp_nll)
print_stat("algorithm time [s]", joint_mlp_time)

print()
print("=" * 100)
print("CONTROLLED COMPARISONS")
print("=" * 100)

joint_mean = joint_fourier_cov.mean()
split_mean = split_cov.mean()

print(
    "Joint Fourier -> Split Fourier covariance RMSE:"
)
print(
    f"  {joint_mean:.8e} -> {split_mean:.8e}"
)
print(
    f"  absolute change = {split_mean - joint_mean:.8e}"
)
print(
    f"  relative change = "
    f"{100 * (split_mean / joint_mean - 1):.2f}%"
)

mlp_mean = joint_mlp_cov.mean()
arff_mean = arff_cov.mean()

print()
print(
    "Joint MLP -> ARFF covariance RMSE:"
)
print(
    f"  {mlp_mean:.8e} -> {arff_mean:.8e}"
)
print(
    f"  absolute change = {arff_mean - mlp_mean:.8e}"
)
print(
    f"  relative change = "
    f"{100 * (arff_mean / mlp_mean - 1):.2f}%"
)

out = ROOT / "results/ex8_final_campaign_summary.npz"

np.savez_compressed(
    out,
    joint_fourier_covariance_rmse=joint_fourier_cov,
    joint_fourier_drift_rmse=joint_fourier_drift,
    joint_fourier_nll=joint_fourier_nll,
    joint_fourier_algorithm_time=joint_fourier_time,
    joint_mlp_covariance_rmse=joint_mlp_cov,
    joint_mlp_drift_rmse=joint_mlp_drift,
    joint_mlp_nll=joint_mlp_nll,
    joint_mlp_algorithm_time=joint_mlp_time,
    split_fourier_seed=np.asarray(
        [r["seed"] for r in campaigns["Split Fourier Adam"]]
    ),
    split_fourier_covariance_rmse=np.asarray(
        [r["covariance"] for r in campaigns["Split Fourier Adam"]]
    ),
    split_fourier_drift_rmse=np.asarray(
        [r["drift"] for r in campaigns["Split Fourier Adam"]]
    ),
    split_fourier_nll=np.asarray(
        [r["nll"] for r in campaigns["Split Fourier Adam"]]
    ),
    split_fourier_algorithm_time=np.asarray(
        [r["time"] for r in campaigns["Split Fourier Adam"]]
    ),
    arff_seed=np.asarray(
        [r["seed"] for r in campaigns["ARFF"]]
    ),
    arff_covariance_rmse=np.asarray(
        [r["covariance"] for r in campaigns["ARFF"]]
    ),
    arff_drift_rmse=np.asarray(
        [r["drift"] for r in campaigns["ARFF"]]
    ),
    arff_nll=np.asarray(
        [r["nll"] for r in campaigns["ARFF"]]
    ),
    arff_algorithm_time=np.asarray(
        [r["time"] for r in campaigns["ARFF"]]
    ),
)

print()
print(f"saved: {out}")
