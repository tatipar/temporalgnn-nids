#!/usr/bin/env python3
"""Audit a corrected NF-v3 CSV and create deterministic manual-review samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils.nfv3_relabel import Columns, relabel_chunk  # noqa: E402


REQUIRED_OUTPUT_COLUMNS = {
    "source_row_id", "source_file", "binary_target", "label_corrected_detail",
    "correction_rule", "attempted_category",
}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected-csv", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--forward-bytes-column", default=None)
    parser.add_argument("--counterfactual-forward-bytes-column", action="append", default=[],
                        help="Alternative column to inspect without changing the corrected CSV. Repeat if needed.")
    parser.add_argument("--sample-per-rule", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunksize <= 0 or args.sample_per_rule <= 0:
        raise ValueError("--chunksize and --sample-per-rule must be positive")
    for path in (args.corrected_csv, args.manifest):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    header = set(pd.read_csv(args.corrected_csv, nrows=0).columns)
    missing = REQUIRED_OUTPUT_COLUMNS - header
    if missing:
        raise ValueError(f"Corrected CSV lacks audit columns: {sorted(missing)}")

    expected_row_id = 0
    total_rows = 0
    binary_counts: Counter[str] = Counter()
    detail_counts: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    transition_counts: Counter[str] = Counter()
    source_files: Counter[str] = Counter()
    counterfactuals = {
        column: {"attempted_category_0_rows": 0, "binary_target_changes_vs_stored": 0}
        for column in args.counterfactual_forward_bytes_column
        if column != args.forward_bytes_column
    }
    samples: dict[str, list[pd.DataFrame]] = {}

    for chunk_index, chunk in enumerate(pd.read_csv(args.corrected_csv, chunksize=args.chunksize, low_memory=False)):
        row_ids = pd.to_numeric(chunk["source_row_id"], errors="raise").to_numpy()
        expected = np.arange(expected_row_id, expected_row_id + len(chunk))
        if not np.array_equal(row_ids, expected):
            raise AssertionError("source_row_id is not unique, contiguous, and ordered")
        expected_row_id += len(chunk)
        total_rows += len(chunk)

        targets = pd.to_numeric(chunk["binary_target"], errors="coerce")
        if targets.isna().any() or not targets.isin([0, 1]).all():
            raise AssertionError("binary_target contains values other than 0 or 1")

        # Replay the versioned rules from the preserved raw columns and compare
        # every derived label field. This catches partial writes or wrong options.
        replay = relabel_chunk(chunk, columns=Columns(), forward_bytes_column=args.forward_bytes_column)
        for column in ["binary_target", "label_corrected_detail", "correction_rule", "attempted_category"]:
            replay_values = replay[column].reset_index(drop=True).astype(str)
            stored_values = chunk[column].reset_index(drop=True).astype(str)
            if not replay_values.equals(stored_values):
                raise AssertionError(f"Stored {column} does not match replayed correction rules")

        for column, summary in counterfactuals.items():
            alternative = relabel_chunk(chunk, columns=Columns(), forward_bytes_column=column)
            summary["attempted_category_0_rows"] += int(
                alternative["correction_rule"].eq("attempted_category_0_to_benign").sum()
            )
            summary["binary_target_changes_vs_stored"] += int(
                alternative["binary_target"].ne(targets).sum()
            )

        binary_counts.update(targets.astype(int).astype(str))
        detail_counts.update(chunk["label_corrected_detail"].astype(str))
        rule_counts.update(chunk["correction_rule"].astype(str))
        original_binary = (
            chunk["Attack"].fillna("Benign").astype(str).str.strip().str.casefold().ne("benign").astype(int)
        )
        transition_counts.update(
            f"{before}->{after}"
            for before, after in zip(original_binary, targets.astype(int))
        )
        source_files.update(chunk["source_file"].astype(str))

        for rule, rule_rows in chunk.groupby("correction_rule", sort=False):
            samples.setdefault(rule, []).append(
                rule_rows.sample(n=min(len(rule_rows), args.sample_per_rule), random_state=42 + chunk_index)
            )

    output_info = manifest.get("output", {})
    checks = {
        "row_count_matches_manifest": total_rows == output_info.get("rows"),
        "sha256_matches_manifest": sha256_file(args.corrected_csv) == output_info.get("sha256"),
        "binary_counts_match_manifest": dict(sorted(binary_counts.items())) == manifest["counts"].get("binary_after"),
        "rule_counts_match_manifest": dict(sorted(rule_counts.items())) == manifest["counts"].get("correction_rules"),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Manifest audit failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_frames = []
    for rule, frames in sorted(samples.items()):
        pool = pd.concat(frames, ignore_index=True)
        sample_frames.append(pool.sample(n=min(len(pool), args.sample_per_rule), random_state=42))
    samples_path = args.output_dir / f"{args.corrected_csv.stem}.manual_review_samples.csv"
    pd.concat(sample_frames, ignore_index=True).to_csv(samples_path, index=False)

    audit = {
        "status": "passed",
        "corrected_csv": str(args.corrected_csv),
        "manifest": str(args.manifest),
        "checks": checks,
        "rows": total_rows,
        "binary_counts": dict(sorted(binary_counts.items())),
        "detail_counts": dict(sorted(detail_counts.items())),
        "rule_counts": dict(sorted(rule_counts.items())),
        "binary_transitions": dict(sorted(transition_counts.items())),
        "changed_binary_labels": transition_counts["0->1"] + transition_counts["1->0"],
        "changed_binary_label_pct": (
            100 * (transition_counts["0->1"] + transition_counts["1->0"]) / total_rows
        ),
        "source_file_counts": dict(sorted(source_files.items())),
        "manual_review_samples": str(samples_path),
        "category_0_detection": manifest.get("category_0_detection"),
        "category_0_counterfactuals": counterfactuals,
    }
    audit_path = args.output_dir / f"{args.corrected_csv.stem}.audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
