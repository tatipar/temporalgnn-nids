#!/usr/bin/env python3
"""Generate the deterministic Phase-4A calibration manifest from train graphs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON_ROOT = SCRIPT_DIR.parent
REPOSITORY_ROOT = PYTHON_ROOT.parents[1]
sys.path.insert(0, str(PYTHON_ROOT))

from utils.calibration import (  # noqa: E402
    REQUIRED_PROFILES,
    calibrate_graph_collection,
    git_code_revision,
    write_calibration_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--profile",
        action="append",
        choices=REQUIRED_PROFILES,
        default=[],
        help="Repeat for both profiles; defaults to the complete required pair.",
    )
    parser.add_argument(
        "--corrected-manifest",
        type=Path,
        default=None,
        help="Override only when the manifest path recorded by the graph build moved.",
    )
    parser.add_argument(
        "--code-revision",
        default=None,
        help="Explicit revision for non-Git execution; defaults to the current Git revision.",
    )
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace a different existing manifest; identical reruns need no flag.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    revision = args.code_revision or git_code_revision(REPOSITORY_ROOT)
    payload = calibrate_graph_collection(
        args.graph_root,
        profiles=tuple(args.profile) or REQUIRED_PROFILES,
        corrected_manifest_path=args.corrected_manifest,
        code_revision=revision,
        verify_checksums=args.verify_checksums,
    )
    status = write_calibration_manifest(payload, args.output, overwrite=args.overwrite)
    serialized = args.output.read_bytes()
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "status": status,
                "calibration_manifest_sha256": hashlib.sha256(serialized).hexdigest(),
                "counts": payload["counts"],
                "candidates": payload["candidates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
