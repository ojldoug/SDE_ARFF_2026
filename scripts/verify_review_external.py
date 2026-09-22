"""Check availability and hashes of local-only review datasets/checkpoints.

The compact archive-only manuscript rebuild does not require these files.
This check is for repeating saved-checkpoint inference or numerical fitting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "results/conservative_manuscript_revision_v1/evaluation/RECONSTRUCTION_CHECKS_PUBLIC.json"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path, help="Optional JSON report path; never overwrites")
    args = parser.parse_args()
    record = json.loads(RECORD.read_text())
    checks = []
    for kind in ("dataset_hashes", "checkpoint_hashes"):
        for relative, expected in record[kind].items():
            prefix, suffix = relative.split("/", 1)
            assert prefix in ("data", "results"), relative
            root = args.data_root if prefix == "data" else args.results_root
            path = root / suffix
            if path.is_file():
                actual = sha(path)
                state = "verified" if actual == expected else "hash_mismatch"
            else:
                actual, state = None, "missing"
            checks.append({"kind": kind, "path": relative, "state": state, "expected_sha256": expected, "actual_sha256": actual})
    summary = {"verified": sum(x["state"] == "verified" for x in checks), "missing": sum(x["state"] == "missing" for x in checks), "hash_mismatch": sum(x["state"] == "hash_mismatch" for x in checks), "checks": checks}
    if args.output:
        assert not args.output.exists(), args.output
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    if summary["missing"] or summary["hash_mismatch"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
