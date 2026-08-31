#!/usr/bin/env python3
"""Extract experiment-related cells from the legacy notebooks."""

import json
from pathlib import Path


NOTEBOOKS = [
    Path("GPU/generate_training_data.ipynb"),
    Path("GPU/generate_gillespie_data.ipynb"),
    Path("GPU/generate_SPDE_data.ipynb"),
    Path("GPU/ARFF_training.ipynb"),
    Path("GPU/Adam_training.ipynb"),
    Path("GPU/plot_true_trained.ipynb"),
]

TERMS = [
    "experiment",
    "filename",
    "ex1", "ex2", "ex3", "ex4", "ex5", "ex6", "ex7", "ex8",
    "sample_rate",
    "n_trajectories",
    "trajectory_time",
    "step_size",
    "time_step",
    "n_skip_steps",
    "training_data",
    "validation",
    "test",
    "split",
    "random_seed",
    "random_state",
    "seed",
    "batch_size",
    "learning_rate",
    "epoch",
    "n_runs",
    "n_iterations",
    "K =",
    "M_min",
    "M_max",
    "lambda",
    "gamma",
    "delta",
    "enforce_spd",
    "n_folds",
    "diff_type",
    "drift_type",
    "true_drift",
    "true_diff",
    "rmse",
    "nll",
]


def main():
    for notebook in NOTEBOOKS:
        if not notebook.exists():
            continue

        nb = json.loads(notebook.read_text())

        print("\n" + "=" * 100)
        print(notebook)
        print("=" * 100)

        for i, cell in enumerate(nb.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue

            src = "".join(cell.get("source", []))

            if any(term.lower() in src.lower() for term in TERMS):
                print(f"\n--- CELL {i} ---")
                print(src.rstrip())


if __name__ == "__main__":
    main()
