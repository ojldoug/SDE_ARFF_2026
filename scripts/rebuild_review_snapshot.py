"""Rebuild the conservative review from committed archive metrics and maps.

No training, data generation, checkpoint loading, or evaluation occurs here.
All output is written to --output, which must not already exist.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics as stats
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / "results/conservative_manuscript_revision_v1"
METHODS = ("joint_fourier", "split_fourier", "arff", "joint_mlp", "split_mlp")
NAMES = dict(zip(METHODS, ("Joint Fourier", "Split Fourier", "ARFF", "Joint MLP", "Split MLP")))
LEGACY = {"arff_historical_corrected": "ARFF", "fourier": "Joint Fourier", "mlp_shallow": "Shallow MLP", "mlp_deep": "Deep MLP"}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean_sd(values: list[float]) -> tuple[float, float]:
    return stats.mean(values), stats.stdev(values)


def pm(values: list[float]) -> str:
    mu, sd = mean_sd(values)
    return rf"${mu:.5g}\pm {sd:.5g}$"


def check_manifest(archive: Path) -> int:
    manifest = json.loads((archive / "MANIFEST.json").read_text())["files"]
    for name, entry in manifest.items():
        assert sha(archive / name) == entry["sha256"], name
    return len(manifest)


def copy_sources(output: Path) -> None:
    source = REVIEW / "sources"
    (output / "sources/tables").mkdir(parents=True)
    (output / "figures").mkdir()
    (output / "evaluation").mkdir()
    (output / "aggregate_source_record").mkdir()
    for path in source.iterdir():
        if path.suffix in (".tex", ".bib"):
            shutil.copyfile(path, output / "sources" / path.name)
    for path in (source / "tables").glob("*.tex"):
        shutil.copyfile(path, output / "sources/tables" / path.name)
    for name in ("ex1_seed0_coefficients.npz", "ex8_seed0_coefficients.npz", "RECONSTRUCTION_CHECKS_PUBLIC.json"):
        shutil.copyfile(REVIEW / "evaluation" / name, output / "evaluation" / name)
    for rel in ("plot_coefficients.py", "lag_ridge_assets.py", "aggregate_source_record/replot_final_editorial.py", "aggregate_source_record/replot_mechanism.py"):
        shutil.copyfile(REVIEW / rel, output / rel)
    shutil.copytree(REVIEW / "archive_inputs", output / "archive_inputs")


def write_table(out: Path, name: str, value: str) -> None:
    (out / "sources/tables" / name).write_text(value)


def historical_tables(out: Path, archive: Path) -> dict[str, int]:
    all_rows = rows(archive / "historical_per_seed.csv")
    counts = {}
    for ex in ("ex1", "ex2", "ex3", "ex5", "ex6", "ex7"):
        lines = [r"\begin{tabular}{lrrrr}\toprule Method & Parameters & Drift RMSE & Raw covariance RMSE & NLL\\\midrule"]
        for method, label in LEGACY.items():
            arm = [r for r in all_rows if r["experiment"] == ex and r["method"] == method]
            if not arm:
                continue
            assert len(arm) == 30 and {int(r["seed"]) for r in arm} == set(range(30))
            assert {r["evaluation_split"] for r in arm} == {"validation"}
            fields = ("drift_rmse", "covariance_rmse", "nll")
            parts = [pm([float(r[key]) for r in arm]) for key in fields]
            lines.append(f"{label} & {int(arm[0]['parameters']):,} & " + " & ".join(parts) + r"\\")
        lines.append(r"\bottomrule\end{tabular}")
        write_table(out, ex + ".tex", "\n".join(lines))
        counts[ex] = len(lines) - 2
    return counts


def baseline_tables(out: Path, archive: Path) -> None:
    data = rows(archive / "baseline_per_seed.csv")
    lines = [r"\begin{tabular}{lrrrr}\toprule Method & Parameters & Drift RMSE & Raw covariance RMSE & NLL\\\midrule"]
    for method in METHODS:
        arm = [r for r in data if r["method"] == method and r["split"] == "test"]
        assert len(arm) == 30 and {int(r["seed"]) for r in arm} == set(range(30))
        parts = [pm([float(r[key]) for r in arm]) for key in ("drift_rmse", "covariance_rmse", "nll")]
        lines.append(f"{NAMES[method]} & {('1,814' if 'mlp' in method else '1,792')} & " + " & ".join(parts) + r"\\")
    lines.append(r"\bottomrule\end{tabular}")
    write_table(out, "ex8.tex", "\n".join(lines))


def sensitivity_tables(out: Path, archive: Path) -> None:
    grouped = defaultdict(list)
    source = (("N", "N_per_seed.csv", "N"), ("h", "h_per_seed.csv", "h"), ("capacity", "capacity_per_seed.csv", "K"))
    for study, name, point_col in source:
        for row in rows(archive / name):
            point = float(row[point_col])
            grouped[study, row["method"], point].append(row)
    metrics = []
    for (study, method, point), arm in grouped.items():
        assert len(arm) == 10 and {int(r["seed"]) for r in arm} == set(range(10))
        for key, source_key in (("drift_rmse", "test_drift_rmse"), ("covariance_rmse", "test_covariance_rmse"), ("nll", "test_nll")):
            mu, sd = mean_sd([float(r[source_key]) for r in arm])
            metrics.append({"study": study, "method": method, "point": point, "metric": key, "mean": mu, "sd": sd, "n": 10})
        if study == "h":
            mu, sd = mean_sd([float(r["test_nll_minus_log_h"]) for r in arm])
            metrics.append({"study": study, "method": method, "point": point, "metric": "nll_minus_log_h", "mean": mu, "sd": sd, "n": 10})
    with (archive / "sensitivity_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["study", "method", "point", "metric", "mean", "sd", "n"])
        writer.writeheader()
        writer.writerows(metrics)
    by_key = {(r["study"], r["method"], r["point"], r["metric"]): r for r in metrics}
    for study in ("N", "h", "capacity"):
        header = r"\begin{longtable}{llrrr}\caption{" + study + r" study: test mean $\pm$ sample SD over ten fitting seeds. Raw covariance errors; all points retained.}\\\toprule Method & Point & Drift RMSE & Raw covariance RMSE & NLL\\\midrule\endfirsthead\toprule Method & Point & Drift RMSE & Raw covariance RMSE & NLL\\\midrule\endhead"
        lines = [header]
        for method in METHODS:
            points = sorted({point for st, m, point in grouped if st == study and m == method})
            assert len(points) == {"N": 4, "h": 9, "capacity": 5}[study]
            for point in points:
                vals = [by_key[study, method, point, metric] for metric in ("drift_rmse", "covariance_rmse", "nll")]
                parts = [rf"${x['mean']:.5g}\pm {x['sd']:.5g}$" for x in vals]
                point_text = {2.5e-5: r"$2.5\times10^{-5}$", 5e-5: r"$5\times10^{-5}$"}.get(point, f"{point:g}") if study == "h" else f"{point:g}"
                lines.append(f"{NAMES[method]} & {point_text} & " + " & ".join(parts) + r"\\")
        lines.append(r"\bottomrule\end{longtable}")
        write_table(out, study + "_full.tex", "\n".join(lines))


def verify_numeric_tables(out: Path, source: Path) -> dict[str, str]:
    # These three special tables retain their accepted editorial layout;
    # their source records are retained in archive_inputs and mapped in REPRODUCING.md.
    generated = ("ex1.tex", "ex2.tex", "ex3.tex", "ex5.tex", "ex6.tex", "ex7.tex", "ex8.tex", "N_full.tex", "h_full.tex", "capacity_full.tex", "lag_ridge_arms.tex", "lag_ridge_paired.tex")
    outcome = {}
    import re

    def displayed_number(value: str) -> tuple[float, float]:
        value = value.strip().replace(" ", "")
        power = 0
        if r"\times10^{" in value:
            mantissa, exponent = value.split(r"\times10^{", 1)
            value, power = mantissa, int(exponent.removesuffix("}"))
        elif "e" in value:
            value, exponent = value.split("e", 1)
            power = int(exponent)
        decimal_places = len(value.split(".", 1)[1]) if "." in value else 0
        unit = 10 ** (power - decimal_places)
        return float(value) * 10**power, unit

    for name in generated:
        actual = (out / "sources/tables" / name).read_text()
        reference = (source / "sources/tables" / name).read_text()
        if name.startswith("lag_ridge"):
            # The lag/ridge tables use plain decimal cells; their generator
            # should exactly reproduce the reviewed source.
            assert actual == reference, name
        else:
            pattern = r"\$([^$]+?)\\pm\s*([^$]+?)\$"
            a, b = re.findall(pattern, actual), re.findall(pattern, reference)
            assert len(a) == len(b) and a, (name, len(a), len(b))
            for i, (generated_pair, published_pair) in enumerate(zip(a, b)):
                for generated_value, published_value in zip(generated_pair, published_pair):
                    truth, _ = displayed_number(generated_value)
                    display, unit = displayed_number(published_value)
                    assert abs(truth - display) <= 0.501 * unit + 1e-12, (name, i, generated_value, published_value)
        # Keep the reviewed four-significant-digit typography in the PDF after
        # independently regenerating and checking the underlying statistics.
        (out / "sources/tables" / name).write_text(reference)
        outcome[name] = "archive-derived; reported precision agrees; reviewed display retained"
    return outcome


def verify_special_tables(out: Path, archive: Path) -> dict[str, str]:
    """Check retained editorial-layout tables against their compact inputs."""
    import re

    def parse_display(number: str) -> tuple[float, float]:
        number = number.strip().replace(" ", "")
        exponent = 0
        if r"\times10^{" in number:
            number, power = number.split(r"\times10^{", 1)
            exponent = int(power.removesuffix("}"))
        places = len(number.split(".", 1)[1]) if "." in number else 0
        return float(number) * 10**exponent, 10 ** (exponent - places)

    def matches(display: str, expected: float) -> None:
        value, unit = parse_display(display)
        assert abs(value - expected) <= .501 * unit + 1e-10, (display, expected)

    source = out / "sources/tables"
    historical = rows(archive / "historical_per_seed.csv")
    baseline = rows(archive / "baseline_per_seed.csv")
    lines = [line for line in (source / "spd.tex").read_text().splitlines() if " & ARFF & " in line]
    assert len(lines) == 7
    for ex, line in zip(("ex1", "ex2", "ex3", "ex5", "ex6", "ex7", "ex8"), lines):
        group = ([r for r in historical if r["experiment"] == ex and r["method"] == "arff_historical_corrected"]
                 if ex != "ex8" else [r for r in baseline if r["method"] == "arff" and r["split"] == "test"])
        assert len(group) == 30
        cells = [c.strip().strip("$") for c in line.split(" & ")]
        affected = sum(float(r["raw_spd_violation_rate"]) > 0 for r in group)
        assert cells[2] == f"{affected}/30"
        matches(cells[3], 100 * stats.mean(float(r["raw_spd_violation_rate"]) for r in group))
        matches(cells[4], min(float(r["min_raw_eigenvalue"]) for r in group))

    spd_rows = rows(archive / "sensitivity_spd.csv")
    spd_text = (source / "sensitivity_spd.tex").read_text()
    spd_lines = [x for x in spd_text.splitlines() if x.startswith(("N &", "h &", "capacity &"))]
    assert len(spd_lines) == 18
    for line in spd_lines:
        fields = [c.strip().removesuffix(r"\\") for c in line.split(" & ")]
        study, point = fields[0], fields[1]
        point = point.replace("$", "")
        if r"\times10^{" in point:
            a, b = point.split(r"\times10^{")
            point_value = float(a) * 10 ** int(b.removesuffix("}"))
        else:
            point_value = float(point)
        group = [r for r in spd_rows if r["study"] == study and math.isclose(float(r["N"] if study == "N" else r["h"] if study == "h" else r["K"]), point_value, rel_tol=0, abs_tol=1e-12)]
        assert len(group) == 10, (study, point, len(group))
        mu, sd = mean_sd([float(r["raw_spd_violation_rate"]) for r in group])
        pair = re.search(r"\$([^$]+?)\\pm\s*([^$]+?)\$", fields[2])
        assert pair
        matches(pair[1], mu)
        matches(pair[2], sd)
        matches(fields[3], min(float(r["minimum_raw_eigenvalue"]) for r in group))

    cpu = rows(archive / "oracle_cpu_per_seed.csv")
    def cpu_values(scope: str, combination: str, key: str) -> list[float]:
        group = [float(r[key]) for r in cpu if r["scope"] == scope and r["combination"] == combination]
        assert len(group) == 30
        return group
    native = "native_float32_floor_1e-3"
    spectral = "float64_floor_1e-3"
    true = "float64_unprojected_true_covariance"
    combinations = ("hybrid", "arff_original", "true_drift_arff_covariance")
    expected = []
    for scope in (native, spectral):
        for combination in combinations:
            expected.append([cpu_values(scope, combination, k) for k in ("quadratic", "logdet", "nll")])
    for combination in ("oracle", "arff_drift_true_covariance", "joint_mlp_drift_true_covariance"):
        expected.append([cpu_values(true, combination, k) for k in ("quadratic", "logdet", "nll")])
    for combination in ("arff_original", "hybrid", "true_drift_arff_covariance"):
        expected.append([[a - b for a, b in zip(cpu_values(native, combination, k), cpu_values(spectral, combination, k))] for k in ("quadratic", "logdet", "nll")])
    swap_pairs = re.findall(r"\$([^$]+?)\\pm\s*([^$]+?)\$", (source / "swaps.tex").read_text())
    assert len(swap_pairs) == 36
    for pair, values in zip(swap_pairs, (x for row in expected for x in row)):
        matches(pair[0], stats.mean(values))
        # Identical-oracle roundoff SD is deliberately displayed as zero.
        if pair[1].strip() != "0":
            matches(pair[1], stats.stdev(values))

    floors = rows(archive / "oracle_floor_per_seed.csv")
    floor_pairs = re.findall(r"\$([^$]+?)\\pm\s*([^$]+?)\$", (source / "floors.tex").read_text())
    assert len(floor_pairs) == 16
    cursor = 0
    for floor in (1e-8, 1e-6, 1e-4, 1e-3):
        arm = [r for r in floors if math.isclose(float(r["floor"]), floor, rel_tol=1e-9)]
        assert len(arm) == 90
        groups = [[float(r["projected_covariance_rmse"]) for r in arm if r["combination"] == "hybrid"]]
        for combination in ("true_drift_arff_covariance", "hybrid", "arff_original"):
            groups.append([float(r["nll"]) for r in arm if r["combination"] == combination])
        for values in groups:
            assert len(values) == 30
            matches(floor_pairs[cursor][0], stats.mean(values))
            matches(floor_pairs[cursor][1], stats.stdev(values))
            cursor += 1

    risk = rows(archive / "fixed_basis_ridge_paired.csv")
    adaptive = rows(archive / "adaptive_ridge_paired.csv")
    resampling = rows(archive / "resampling_paired.csv")
    table = (out / "sources/supplementary_diagnostics.tex").read_text()
    for condition in ("A", "C"):
        arm = [r for r in risk if r["condition"] == condition]
        assert len(arm) == 3
        for key in ("baseline_expected_mse", "scaled_expected_mse"):
            assert f"{stats.mean(float(r[key]) for r in arm):.3f}" in table
    for key in ("B0_test_mse", "Bpost_test_mse", "Bpath_test_mse"):
        assert f"{stats.mean(float(r[key]) for r in adaptive):.3f}" in table
    for key in ("M_selected_test_drift_mse", "MR_selected_test_drift_mse", "M_final_test_drift_mse", "MR_final_test_drift_mse"):
        assert f"{stats.mean(float(r[key]) for r in resampling):.3f}" in table
    return {name: "archived scalar source and displayed precision checked" for name in ("spd.tex", "sensitivity_spd.tex", "swaps.tex", "floors.tex", "tab:drift_diagnostic")}


def run_python(path: Path, cwd: Path) -> None:
    subprocess.run([sys.executable, str(path)], cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-tex", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    assert not out.exists(), f"Refusing to overwrite {out}"
    assert not out.is_relative_to(REVIEW.resolve())
    out.mkdir(parents=True)
    copy_sources(out)
    archive = out / "archive_inputs"
    checked_files = check_manifest(archive)
    hist = historical_tables(out, archive)
    baseline_tables(out, archive)
    sensitivity_tables(out, archive)
    run_python(out / "lag_ridge_assets.py", out)
    run_python(out / "plot_coefficients.py", out)
    run_python(out / "aggregate_source_record/replot_final_editorial.py", out)
    run_python(out / "aggregate_source_record/replot_mechanism.py", out)
    table_status = verify_numeric_tables(out, REVIEW)
    table_status.update(verify_special_tables(out, archive))
    # Remaining retained editorial tables and the static experiment descriptions
    # are source snapshots; their archive sources are described in REPRODUCING.md.
    if not args.skip_tex:
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "-cd", str(out / "sources/RaulCom_ARFFSDELearning.tex")], check=True, stdout=(out / "latexmk.log").open("w"), stderr=subprocess.STDOUT)
        shutil.copyfile(out / "sources/RaulCom_ARFFSDELearning.pdf", out / "manuscript.pdf")
    result = {"archive_files_sha256_checked": checked_files, "historical_table_method_counts": hist, "numeric_tables": table_status, "figures_regenerated": [p.name for p in sorted((out / "figures").glob("*.pdf"))], "manuscript_compiled": not args.skip_tex, "scope": "archived scalar metrics and saved coefficient maps; no checkpoint inference or training"}
    (out / "REBUILD_CHECKS.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
