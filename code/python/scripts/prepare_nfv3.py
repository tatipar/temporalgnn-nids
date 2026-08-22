#!/usr/bin/env python3
"""Create a corrected, manifest-backed NF-v3 CSV from one or more raw CSVs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils.nfv3_relabel import Columns, RULE_VERSION, relabel_chunk  # noqa: E402


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", action="append", required=True, type=Path,
                        help="Raw NF-v3 CSV. Repeat for multiple files.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--output-name", default="nfv3_corrected.csv")
    parser.add_argument("--chunksize", type=int, default=500_000)
    parser.add_argument("--attack-column", default="Attack")
    parser.add_argument("--forward-bytes-column", default=None,
                        help="Validated NF-v3 equivalent of CICFlowMeter Total Length of Fwd Packets.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunksize <= 0:
        raise ValueError("--chunksize must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / args.output_name
    manifest_path = args.output_dir / f"{output_csv.stem}.manifest.json"
    if (output_csv.exists() or manifest_path.exists()) and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output_csv}. Use --overwrite only for a deliberate rerun.")

    columns = Columns(attack=args.attack_column)
    input_info = []
    for input_csv in args.input_csv:
        if not input_csv.is_file():
            raise FileNotFoundError(input_csv)
        input_info.append({"path": str(input_csv), "sha256": sha256_file(input_csv)})

    before_labels: Counter[str] = Counter()
    after_labels: Counter[str] = Counter()
    rule_counts: Counter[str] = Counter()
    binary_before: Counter[str] = Counter()
    binary_after: Counter[str] = Counter()
    total_rows = 0
    wrote_header = False

    # The generated source_row_id is a stable provenance key for this exact
    # ordered input list. It is intentionally created before filtering/graphs.
    with output_csv.open("w", encoding="utf-8", newline="") as output_handle:
        for input_csv in args.input_csv:
            for chunk in pd.read_csv(input_csv, chunksize=args.chunksize, low_memory=False):
                original_attack = chunk[args.attack_column].fillna("Benign").astype(str).str.strip()
                before_labels.update(original_attack)
                binary_before.update((original_attack.str.casefold() != "benign").astype(int).astype(str))

                corrected = relabel_chunk(
                    chunk, columns=columns, forward_bytes_column=args.forward_bytes_column,
                )
                corrected.insert(0, "source_row_id", range(total_rows, total_rows + len(corrected)))
                corrected.insert(1, "source_file", input_csv.name)
                total_rows += len(corrected)

                after_labels.update(corrected["label_corrected_detail"].astype(str))
                rule_counts.update(corrected["correction_rule"].astype(str))
                binary_after.update(corrected["binary_target"].astype(int).astype(str))
                corrected.to_csv(output_handle, index=False, header=not wrote_header)
                wrote_header = True

    manifest = {
        "dataset": "NF-CSE-CIC-IDS2018-v3",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "correction_rule_version": RULE_VERSION,
        "correction_source": "docs/improved_cse_cic_ids2018_documentation_infiltration.md",
        "attempted_policy": "Attempted Category 0 and 4 are benign; never a separate ML label.",
        "category_0_detection": (
            f"enabled using {args.forward_bytes_column}" if args.forward_bytes_column
            else "not applied: no validated NF-v3 equivalent of Total Length of Fwd Packets was supplied"
        ),
        "binary_target_definition": "1 = every non-Benign attack except historical Infilteration rebuilt by confirmed rules; 0 = Benign and all Attempted flows.",
        "columns_excluded_from_edge_attr": [
            "source_row_id", "source_file", columns.time_ms, columns.source_ip,
            columns.destination_ip, columns.attack, "label_corrected_detail",
            "binary_target", "correction_rule", "attempted_category",
        ],
        "inputs": input_info,
        "output": {"path": str(output_csv), "sha256": sha256_file(output_csv), "rows": total_rows},
        "counts": {
            "original_attack_labels": dict(sorted(before_labels.items())),
            "corrected_detail_labels": dict(sorted(after_labels.items())),
            "correction_rules": dict(sorted(rule_counts.items())),
            "binary_before": dict(sorted(binary_before.items())),
            "binary_after": dict(sorted(binary_after.items())),
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_csv": str(output_csv), "manifest": str(manifest_path), "rows": total_rows,
                      "rules": dict(sorted(rule_counts.items()))}, indent=2))


if __name__ == "__main__":
    main()
