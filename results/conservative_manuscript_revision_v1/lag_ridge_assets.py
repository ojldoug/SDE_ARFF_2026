"""Render the archived Ex8 lag/ridge results for the isolated review copy.

This reads only compact copies of the completed study CSVs; it performs no model
evaluation or fitting. Run from any directory.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REVIEW = Path(__file__).resolve().parent
ARCHIVE = REVIEW / "archive_inputs"
H = (0.0001, 0.0004, 0.002)
LAMBDAS = (0.001, 0.008, 0.064)
SEEDS = (0, 1, 2)


def read_csv(name: str) -> list[dict[str, str]]:
    with (ARCHIVE / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def rows_by_key(rows: list[dict[str, str]]) -> dict[tuple[float, float, int], dict[str, str]]:
    indexed = {(float(r["h"]), float(r["lambda"]), int(r["seed"])): r for r in rows}
    expected = {(h, lam, seed) for h in H for lam in LAMBDAS for seed in SEEDS}
    assert set(indexed) == expected, (len(indexed), len(expected), set(indexed) ^ expected)
    return indexed


def fmt(x: float, digits: int = 4) -> str:
    return f"{x:.{digits}f}"


def main() -> None:
    runs = rows_by_key(read_csv("lag_ridge_per_run.csv"))
    pairs = rows_by_key(
        read_csv("lag_ridge_paired.csv")
        + [{"h": str(h), "lambda": "0.001", "seed": str(s)} for h in H for s in SEEDS]
    )
    for h in H:
        for lam in LAMBDAS[1:]:
            for s in SEEDS:
                for metric in ("drift_rmse", "covariance_rmse", "nll", "raw_spd_violation_rate"):
                    archived = float(pairs[h, lam, s][f"test_{metric}_change"])
                    from_runs = float(runs[h, lam, s][f"test_{metric}"]) - float(runs[h, 0.001, s][f"test_{metric}"])
                    assert math.isclose(archived, from_runs, rel_tol=0, abs_tol=1e-10)
    summary = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"$h$ & $\lambda_f$ & Drift RMSE & Raw cov. RMSE & NLL & Raw SPD viol. (\%)\\",
        r"\midrule",
    ]
    for h in H:
        for lam in LAMBDAS:
            arm = [runs[h, lam, s] for s in SEEDS]
            vals = [mean(float(r[key]) for r in arm) for key in (
                "test_drift_rmse", "test_covariance_rmse", "test_nll", "test_raw_spd_violation_rate"
            )]
            summary.append(
                f"{h:g} & {lam:g} & {fmt(vals[0])} & {fmt(vals[1])} & "
                f"{fmt(vals[2])} & {fmt(100 * vals[3], 2)}\\\\"
            )
        if h != H[-1]:
            summary.append(r"\addlinespace")
    summary.extend([r"\bottomrule", r"\end{tabular}"])
    (REVIEW / "sources/tables/lag_ridge_arms.tex").write_text("\n".join(summary) + "\n")

    paired = [
        r"\begin{tabular}{lllrrr}",
        r"\toprule",
        r"$h$ & $\lambda_f$ & Seed & $\Delta$ drift RMSE & $\Delta$ raw cov. RMSE & $\Delta$ NLL\\",
        r"\midrule",
    ]
    for h in H:
        for lam in LAMBDAS[1:]:
            for s in SEEDS:
                row = pairs[h, lam, s]
                vals = [float(row[key]) for key in (
                    "test_drift_rmse_change", "test_covariance_rmse_change", "test_nll_change"
                )]
                paired.append(
                    f"{h:g} & {lam:g} & {s} & {fmt(vals[0])} & {fmt(vals[1], 5)} & {fmt(vals[2])}\\\\"
                )
        if h != H[-1]:
            paired.append(r"\addlinespace")
    paired.extend([r"\bottomrule", r"\end{tabular}"])
    (REVIEW / "sources/tables/lag_ridge_paired.tex").write_text("\n".join(paired) + "\n")

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.2), sharey=True, constrained_layout=True)
    colors = ("#226699", "#ca6b31", "#5c8d3a")
    for ax, h in zip(axes, H):
        for s, color in zip(SEEDS, colors):
            yy = [0.0] + [float(pairs[h, lam, s]["test_nll_change"]) for lam in LAMBDAS[1:]]
            ax.plot(range(3), yy, marker="o", linewidth=1.6, markersize=4, color=color, label=f"Seed {s}")
        ax.axhline(0, color="0.45", linewidth=0.8)
        ax.set_xticks(range(3), ("0.001", "0.008", "0.064"))
        ax.set_xlabel(r"Drift ridge $\lambda_f$")
        ax.set_title(f"$h={h:g}$")
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel(r"Paired test NLL change vs $\lambda_f=0.001$")
    axes[-1].legend(loc="upper right", fontsize=8, frameon=False)
    for suffix in ("pdf", "png"):
        fig.savefig(REVIEW / "figures" / f"lag_ridge_paired_nll.{suffix}", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    main()
