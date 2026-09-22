"""One-time, archive-only packaging of the conservative review's scalar evidence.

Run in a repository that has the accepted local result archives. This writes
only the compact public evidence bundle, never the source results.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path


SOURCES = {
    "historical_per_seed.csv": "results/final_reproduction/resolved_accuracy_v1/per_seed_metrics.csv",
    "historical_summary.json": "results/final_reproduction/resolved_accuracy_v1/summary.json",
    "baseline_per_seed.csv": "results/final_ex8_publication_bundle/summaries/baseline/per_seed_split_metrics.csv",
    "baseline_summary.json": "results/final_ex8_publication_bundle/summaries/baseline/summary.json",
    "N_per_seed.csv": "results/final_ex8_publication_bundle/summaries/N/per_seed_metrics.csv",
    "h_per_seed.csv": "results/final_ex8_publication_bundle/summaries/h/per_seed_metrics.csv",
    "capacity_per_seed.csv": "results/final_ex8_publication_bundle/summaries/capacity/per_seed_metrics.csv",
    "sensitivity_spd.csv": "results/final_ex8_publication_bundle/figures/arff_raw_spd.csv",
    "lag_ridge_paired.csv": "results/ex8_h_lambda_drift_v1/paired_results.csv",
    "oracle_cpu_per_seed.csv": "results/controlled_study_2026/float64_v2/oracle_component_swap/evaluation_v2/per_seed.csv",
    "oracle_floor_per_seed.csv": "results/controlled_study_2026/float64_v2/oracle_component_swap/evaluation_v2/floor_sensitivity_per_seed.csv",
    "oracle_projection_per_seed.csv": "results/controlled_study_2026/float64_v2/oracle_component_swap/evaluation_v2/projection_per_seed.csv",
    "parameter_counts.csv": "results/final_ex8_publication_bundle/parameter_counts.csv",
    "fixed_basis_ridge_paired.csv": "results/arff_fixed_basis_ridge_diagnostic_v1/paired.csv",
    "adaptive_ridge_paired.csv": "results/arff_B_ridge_intervention_v1/paired_results.csv",
    "resampling_paired.csv": "results/arff_resampling_isolation_v1/paired_results.csv",
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--refresh-public-validation", action="store_true", help="Update only the portable reconstruction record")
    args = parser.parse_args()
    repo = args.repo.resolve()
    review = repo / "results/conservative_manuscript_revision_v1"
    out = review / "archive_inputs"
    if not args.refresh_public_validation:
        out.mkdir(exist_ok=False)
    manifest = {}
    for name, relative in (() if args.refresh_public_validation else SOURCES.items()):
        source = repo / relative
        target = out / name
        if source.suffix == ".csv":
            target.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
        else:
            shutil.copyfile(source, target)
        manifest[name] = {"origin": relative, "origin_sha256": digest(source), "sha256": digest(target), "copy_conversion": "CRLF to LF" if source.suffix == ".csv" else "none"}
    # The source CSV includes local artifact paths. Retain all numerical fields,
    # converting only those paths to paths relative to the repository root.
    if not args.refresh_public_validation:
        source = repo / "results/ex8_h_lambda_drift_v1/per_run.csv"
        target = out / "lag_ridge_per_run.csv"
        with source.open(newline="") as inp, target.open("x", newline="") as dst:
            reader = csv.DictReader(inp)
            writer = csv.DictWriter(dst, fieldnames=reader.fieldnames, lineterminator="\n")
            writer.writeheader()
            for row in reader:
                artifact = row["artifact"]
                assert "/results/" in artifact
                row["artifact"] = "results/" + artifact.split("/results/", 1)[1]
                writer.writerow(row)
        manifest[target.name] = {"origin": str(source.relative_to(repo)), "origin_sha256": digest(source), "sha256": digest(target), "path_conversion": "absolute artifact column to repository-relative; numerical fields unchanged"}
        for repeat in range(3):
            for arm in ("M", "MR"):
                base = f"results/arff_resampling_isolation_v1/jobs/{repeat}_{arm}"
                name = f"mechanism_{repeat}_{arm}.csv"
                source = repo / base / "mechanisms.csv"
                (out / name).write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
                manifest[name] = {"origin": str(source.relative_to(repo)), "origin_sha256": digest(source), "sha256": digest(out / name), "copy_conversion": "CRLF to LF"}
        selections = []
        for repeat in range(3):
            for arm in ("M", "MR"):
                relative = f"results/arff_resampling_isolation_v1/jobs/{repeat}_{arm}/COMPLETE.json"
                record = json.loads((repo / relative).read_text())
                selections.append({"repeat": repeat, "arm": arm, "selected_iteration": int(record["selected_iteration"])})
        selected = out / "mechanism_selections.csv"
        with selected.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["repeat", "arm", "selected_iteration"], lineterminator="\n")
            writer.writeheader()
            writer.writerows(selections)
        manifest[selected.name] = {"origin": "six archived resampling-isolation COMPLETE.json files", "sha256": digest(selected)}
    complete = json.loads((review / "evaluation/COMPLETE.json").read_text())
    provenance = complete["provenance"]
    relative_hashes = {}
    for key in ("source_hashes", "checkpoint_hashes", "dataset_hashes"):
        relative_hashes[key] = {str(Path(path).resolve().relative_to(repo)): digest_value for path, digest_value in provenance[key].items()}
    public = {"status": complete["status"], "seed": 0, "validation": complete["validation"], "training": False, "new_observations": False, "grid_data_sha256": {
        name: digest(review / "evaluation" / name) for name in ("ex1_seed0_coefficients.npz", "ex8_seed0_coefficients.npz")
    }, **relative_hashes}
    (review / "evaluation/RECONSTRUCTION_CHECKS_PUBLIC.json").write_text(json.dumps(public, indent=2) + "\n")
    if not args.refresh_public_validation:
        (out / "MANIFEST.json").write_text(json.dumps({"status": "archive-only public evidence", "files": manifest}, indent=2) + "\n")


if __name__ == "__main__":
    main()
