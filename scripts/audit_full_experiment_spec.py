#!/usr/bin/env python3
"""
Audit the complete historical experiment specification.

This script does not decide which historical values are correct.
It extracts relevant evidence from the surviving legacy notebooks so that
conflicts between manuscript settings, data-generation settings, ARFF
settings, Adam settings, and evaluation settings can be resolved explicitly.

The resulting report is intended as forensic documentation for rebuilding
the reproducible experiment pipeline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))

NOTEBOOKS = [
    REPO_ROOT / "GPU/generate_training_data.ipynb",
    REPO_ROOT / "GPU/generate_gillespie_data.ipynb",
    REPO_ROOT / "GPU/generate_SPDE_data.ipynb",
    REPO_ROOT / "GPU/ARFF_training.ipynb",
    REPO_ROOT / "GPU/Adam_training.ipynb",
    REPO_ROOT / "GPU/plot_true_trained.ipynb",
    REPO_ROOT / "GPU/true_functions/true_functions.ipynb",
    REPO_ROOT / "GPU/saved_results/loss_time_data/min_loss_stats/read_min_data.ipynb",
    REPO_ROOT / "GPU/saved_results/true_trained_RMSE/read_RMSE_data.ipynb",
]


# Broad on purpose. We want evidence, not only currently understood parameters.
PARAMETER_PATTERNS = [
    # Experiment identity / data size
    r"\bex_name\b",
    r"\bex[1-8]\b",
    r"\bn_trajectories\b",
    r"\bn_samples\b",
    r"\bN\b",
    r"\btarget_samples\b",

    # Time discretization
    r"\btrajectory_time\b",
    r"\btime_step\b",
    r"\bstep_size\b",
    r"\bstep_sizes\b",
    r"\bgrid_resolution\b",
    r"\bsample_rate\b",
    r"\bn_skip_steps\b",
    r"\btime_max\b",
    r"\btime_size\b",
    r"\bstep_size_network\b",

    # ARFF
    r"\bK\b",
    r"\bM\b",
    r"\bM_min\b",
    r"\bM_max\b",
    r"\blambda\b",
    r"\blambda_reg\b",
    r"\bgamma\b",
    r"\bdelta\b",
    r"\bresampling\b",
    r"\bmetropolis\b",
    r"\bfold\b",
    r"\bn_folds\b",

    # Adam / NN
    r"\bepochs\b",
    r"\bN_EPOCHS\b",
    r"\blearning_rate\b",
    r"\bLEARNING_RATE\b",
    r"\bbatch_size\b",
    r"\bBATCH_SIZE\b",
    r"\bwidth\b",
    r"\bn_layers\b",
    r"\bn_dim_per_layer\b",
    r"\bactivation\b",
    r"\bACTIVATIONS\b",
    r"\bAdam_type\b",

    # Splits / repetitions / randomness
    r"\btrain_fraction\b",
    r"\bvalidation\b",
    r"\btest\b",
    r"\bsplit\b",
    r"\brandom_state\b",
    r"\bseed\b",
    r"\bn_runs\b",
    r"\bn_iterations\b",

    # Diffusion / covariance
    r"\bdiff_type\b",
    r"\bdiffusion\b",
    r"\bcovariance\b",
    r"\bSPD\b",
    r"\bepsilon\b",

    # Special experiments
    r"\bcoupled_fn\b",
    r"\bLangevin\b",
    r"\bGillespie\b",
    r"\bSIR\b",
    r"\bk1\b",
    r"\bk2\b",
    r"\bk3\b",
    r"\bpopulation\b",
    r"\bSPDE\b",

    # Evaluation / output
    r"\bRMSE\b",
    r"\bloss\b",
    r"\bval_loss\b",
    r"\btraining_time\b",
    r"\btrue_trained\b",
]


COMPILED_PATTERNS = [
    re.compile(pattern, flags=re.IGNORECASE)
    for pattern in PARAMETER_PATTERNS
]


def load_notebook(path: Path):
    return json.loads(path.read_text())


def relevant_lines(src: str):
    """
    Return matching lines with one line of context on either side.

    This avoids dumping huge notebook cells while preserving enough context
    to understand assignments.
    """
    lines = src.splitlines()
    selected = set()

    for i, line in enumerate(lines):
        if any(pattern.search(line) for pattern in COMPILED_PATTERNS):
            for j in range(max(0, i - 1), min(len(lines), i + 2)):
                selected.add(j)

    if not selected:
        return []

    result = []
    previous = None

    for i in sorted(selected):
        if previous is not None and i > previous + 1:
            result.append("    ...")

        result.append(f"{i + 1:4d}: {lines[i]}")
        previous = i

    return result


def print_notebook_evidence(path: Path):
    print()
    print("=" * 100)
    print(path.relative_to(REPO_ROOT))
    print("=" * 100)

    if not path.exists():
        print("MISSING")
        return

    nb = load_notebook(path)

    found_anything = False

    for cell_index, cell in enumerate(nb.get("cells", [])):
        if cell.get("cell_type") != "code":
            continue

        src = "".join(cell.get("source", []))
        lines = relevant_lines(src)

        if not lines:
            continue

        found_anything = True

        print()
        print(f"--- CELL {cell_index} ---")

        for line in lines:
            print(line)

    if not found_anything:
        print("No parameter-related code cells found.")


def print_current_reproducible_config():
    print()
    print("=" * 100)
    print("CURRENT REPRODUCIBLE CONFIG")
    print("=" * 100)

    # Import here so the historical-notebook audit still works even if the
    # current config is temporarily broken during development.
    try:
        from src.experiments.config import CONFIGS
    except Exception as exc:
        print(f"Could not import current config: {exc}")
        return

    for name, cfg in CONFIGS.items():
        print()
        print(name)
        print("-" * len(name))

        print(f"K                      : {cfg.K}")

        print("data")
        for field, value in vars(cfg.data).items():
            print(f"  {field:20s} : {value}")

        print("split")
        for field, value in vars(cfg.split).items():
            print(f"  {field:20s} : {value}")

        print("arff")
        for field, value in vars(cfg.arff).items():
            print(f"  {field:20s} : {value}")

        print("adam")
        for field, value in vars(cfg.adam).items():
            print(f"  {field:20s} : {value}")

        print("evaluation")
        for field, value in vars(cfg.evaluation).items():
            print(f"  {field:20s} : {value}")


def main():
    print("FULL HISTORICAL EXPERIMENT-SPECIFICATION AUDIT")
    print("=============================================")
    print()
    print(
        "This report extracts evidence only. Conflicting notebook values are "
        "deliberately left unresolved."
    )

    for notebook in NOTEBOOKS:
        print_notebook_evidence(notebook)

    print_current_reproducible_config()


if __name__ == "__main__":
    main()
