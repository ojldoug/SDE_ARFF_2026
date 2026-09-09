#!/usr/bin/env python3

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/ex8_final_campaign_summary.npz"
OUT = ROOT / "results/figures/ex8_final_candidates"
OUT.mkdir(parents=True, exist_ok=True)

with np.load(DATA, allow_pickle=False) as z:
    joint_fourier_cov = np.asarray(
        z["joint_fourier_covariance_rmse"], dtype=float
    )
    joint_fourier_drift = np.asarray(
        z["joint_fourier_drift_rmse"], dtype=float
    )

    split_fourier_cov = np.asarray(
        z["split_fourier_covariance_rmse"], dtype=float
    )
    split_fourier_drift = np.asarray(
        z["split_fourier_drift_rmse"], dtype=float
    )

    joint_mlp_cov = np.asarray(
        z["joint_mlp_covariance_rmse"], dtype=float
    )
    joint_mlp_drift = np.asarray(
        z["joint_mlp_drift_rmse"], dtype=float
    )

    arff_cov = np.asarray(
        z["arff_covariance_rmse"], dtype=float
    )
    arff_drift = np.asarray(
        z["arff_drift_rmse"], dtype=float
    )


names = [
    "Joint Fourier\nAdam",
    "Split Fourier\nAdam",
    "Joint MLP\nAdam",
    "ARFF",
]

covariance = [
    joint_fourier_cov,
    split_fourier_cov,
    joint_mlp_cov,
    arff_cov,
]

drift = [
    joint_fourier_drift,
    split_fourier_drift,
    joint_mlp_drift,
    arff_drift,
]


rng = np.random.default_rng(2026)


def scatter_distribution(ax, arrays, *, log=False):
    positions = np.arange(1, len(arrays) + 1)

    ax.boxplot(
        arrays,
        positions=positions,
        widths=0.50,
        showfliers=False,
        medianprops={"linewidth": 2},
    )

    for x, values in zip(positions, arrays):
        jitter = rng.normal(0.0, 0.055, size=len(values))

        ax.scatter(
            x + jitter,
            values,
            s=23,
            alpha=0.55,
            zorder=2,
        )

        mean = np.mean(values)
        sd = np.std(values, ddof=1)

        ax.errorbar(
            x,
            mean,
            yerr=sd,
            fmt="D",
            markersize=7,
            capsize=5,
            linewidth=2,
            zorder=4,
        )

    ax.set_xticks(positions)
    ax.set_xticklabels(names)

    if log:
        ax.set_yscale("log")

    ax.grid(axis="y", alpha=0.25)


def save(fig, stem):
    fig.savefig(
        OUT / f"{stem}.pdf",
        bbox_inches="tight",
    )

    fig.savefig(
        OUT / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


# ------------------------------------------------------------
# 1. Covariance RMSE: main poster candidate
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.2, 5.5))

scatter_distribution(
    ax,
    covariance,
    log=True,
)

ax.set_ylabel("Test covariance RMSE")
ax.set_title(
    "Experiment 8: near-singular covariance recovery"
)

ax.text(
    0.02,
    0.02,
    "30 independent runs per method; diamond = mean, bars = ±1 SD",
    transform=ax.transAxes,
    fontsize=9,
)

save(
    fig,
    "ex8_covariance_rmse_distribution",
)


# ------------------------------------------------------------
# 2. Drift RMSE
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8.2, 5.5))

scatter_distribution(
    ax,
    drift,
    log=False,
)

ax.set_ylabel("Test drift RMSE")
ax.set_title(
    "Experiment 8: drift recovery"
)

ax.text(
    0.02,
    0.97,
    "30 independent runs per method; diamond = mean, bars = ±1 SD",
    transform=ax.transAxes,
    va="top",
    fontsize=9,
)

save(
    fig,
    "ex8_drift_rmse_distribution",
)


# ------------------------------------------------------------
# 3. Two-panel RMSE summary
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.0, 5.2),
)

scatter_distribution(
    axes[0],
    covariance,
    log=True,
)

axes[0].set_ylabel("Test covariance RMSE")
axes[0].set_title("(a) Covariance")

scatter_distribution(
    axes[1],
    drift,
    log=False,
)

axes[1].set_ylabel("Test drift RMSE")
axes[1].set_title("(b) Drift")

fig.suptitle(
    "Experiment 8: coefficient recovery over 30 runs",
    fontsize=15,
)

fig.tight_layout()

save(
    fig,
    "ex8_rmse_two_panel",
)


# ------------------------------------------------------------
# 4. Controlled Fourier split diagnostic
# ------------------------------------------------------------

fig, ax = plt.subplots(figsize=(6.6, 5.2))

fourier_arrays = [
    joint_fourier_cov,
    split_fourier_cov,
]

fourier_names = [
    "Joint Fourier Adam",
    "Split Fourier Adam",
]

positions = [1, 2]

ax.boxplot(
    fourier_arrays,
    positions=positions,
    widths=0.5,
    showfliers=False,
    medianprops={"linewidth": 2},
)

for x, values in zip(
    positions,
    fourier_arrays,
):
    jitter = rng.normal(
        0.0,
        0.055,
        size=len(values),
    )

    ax.scatter(
        x + jitter,
        values,
        s=28,
        alpha=0.6,
    )

    ax.errorbar(
        x,
        np.mean(values),
        yerr=np.std(values, ddof=1),
        fmt="D",
        markersize=7,
        capsize=5,
        linewidth=2,
    )

ax.set_xticks(positions)
ax.set_xticklabels(fourier_names)
ax.set_ylabel("Test covariance RMSE")
ax.set_title(
    "Two-stage targets alone do not explain ARFF accuracy"
)
ax.grid(axis="y", alpha=0.25)

joint_mean = np.mean(joint_fourier_cov)
split_mean = np.mean(split_fourier_cov)

change = 100.0 * (
    split_mean / joint_mean - 1.0
)

ax.text(
    0.5,
    0.96,
    f"Mean change: {change:.1f}%",
    ha="center",
    va="top",
    transform=ax.transAxes,
)

save(
    fig,
    "ex8_joint_vs_split_fourier",
)


# ------------------------------------------------------------
# 5. Historical Owen ARFF vs corrected ARFF
# ------------------------------------------------------------

OWEN_ARFF_COV = 0.1417
OWEN_ARFF_DRIFT = 0.8622

fig, axes = plt.subplots(
    1,
    2,
    figsize=(9.5, 4.8),
)

# covariance
axes[0].scatter(
    np.ones_like(arff_cov) * 2
    + rng.normal(0, 0.035, len(arff_cov)),
    arff_cov,
    alpha=0.55,
    s=25,
)

axes[0].errorbar(
    2,
    np.mean(arff_cov),
    yerr=np.std(arff_cov, ddof=1),
    fmt="D",
    capsize=5,
)

axes[0].scatter(
    1,
    OWEN_ARFF_COV,
    marker="D",
    s=55,
)

axes[0].set_xticks([1, 2])
axes[0].set_xticklabels(
    ["Owen historical\nmean", "Corrected ARFF\n30 runs"]
)
axes[0].set_ylabel("Covariance RMSE")
axes[0].set_title("(a) Covariance")
axes[0].grid(axis="y", alpha=0.25)

# drift
axes[1].scatter(
    np.ones_like(arff_drift) * 2
    + rng.normal(0, 0.035, len(arff_drift)),
    arff_drift,
    alpha=0.55,
    s=25,
)

axes[1].errorbar(
    2,
    np.mean(arff_drift),
    yerr=np.std(arff_drift, ddof=1),
    fmt="D",
    capsize=5,
)

axes[1].scatter(
    1,
    OWEN_ARFF_DRIFT,
    marker="D",
    s=55,
)

axes[1].set_xticks([1, 2])
axes[1].set_xticklabels(
    ["Owen historical\nmean", "Corrected ARFF\n30 runs"]
)
axes[1].set_ylabel("Drift RMSE")
axes[1].set_title("(b) Drift")
axes[1].grid(axis="y", alpha=0.25)

fig.suptitle(
    "Experiment 8 ARFF: historical vs corrected pipeline"
)

fig.tight_layout()

save(
    fig,
    "ex8_arff_historical_vs_corrected",
)


print("Saved candidate figures to:")
print(OUT)

for p in sorted(OUT.glob("*.png")):
    print(" ", p)
