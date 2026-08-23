#!/usr/bin/env python3
"""Build auditable Day-1 and Day-2 NF-v3 temporal graph collections.

Run this script from the thin Colab notebook. Start with ``--max-windows`` for
a smoke build; omit it only after reviewing the generated audit and manifests.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import joblib
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from utils.graph_construction import (  # noqa: E402
    DAY1, DAY2, DaySpec, IpIdMap, atomic_json_dump, atomic_torch_save,
    audit_graph_file, build_graph, fit_scalers, prepare_chunk, required_columns,
    sha256_file, split_cutoffs, split_for_time,
)
from utils.graph_schema import get_feature_profile  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", action="append", type=Path, required=True,
                        help="Corrected CSV input. Repeat only when corrected data is split across files.")
    parser.add_argument("--corrected-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="New versioned graph directory, for example .../graphs/infiltration_v1_w30.")
    parser.add_argument("--profile", action="append", default=[], choices=("nfv3_extended", "portable_core"),
                        help="Feature profile to build. Repeat; defaults to both profiles.")
    parser.add_argument("--day1-source-file", default=DAY1.source_file)
    parser.add_argument("--day2-source-file", default=DAY2.source_file)
    parser.add_argument("--chunksize", type=int, default=250_000)
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Stop after this many windows per day for a smoke build.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously interrupted build with the same output root and configuration.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete the exact output root before beginning a new build.")
    return parser.parse_args()


def make_days(args: argparse.Namespace) -> tuple[DaySpec, DaySpec]:
    return (
        DaySpec(DAY1.name, args.day1_source_file, DAY1.split_names, DAY1.train_ratio, DAY1.validation_ratio),
        DaySpec(DAY2.name, args.day2_source_file, DAY2.split_names),
    )


def iter_complete_windows(input_csvs: list[Path], day: DaySpec, usecols: list[str], chunksize: int):
    """Yield complete chronological decision-time windows without chunk splits."""
    buffer = pd.DataFrame()
    last_yielded: int | None = None
    for path in input_csvs:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            chunk = chunk.loc[chunk["source_file"].eq(day.source_file)]
            if chunk.empty:
                continue
            prepared, excluded = prepare_chunk(chunk)
            if prepared.empty:
                continue
            prepared = prepared.sort_values(["decision_time_ms", "source_row_id"], kind="stable")
            combined = pd.concat((buffer, prepared), ignore_index=True)
            last_time = int(combined["decision_time_ms"].max())
            complete = combined.loc[combined["decision_time_ms"] < last_time]
            buffer = combined.loc[combined["decision_time_ms"] == last_time].copy()
            for decision_time, group in complete.groupby("decision_time_ms", sort=True):
                decision_time = int(decision_time)
                if last_yielded is not None and decision_time <= last_yielded:
                    raise ValueError("Input rows are not chronologically ordered by flow completion time.")
                last_yielded = decision_time
                yield decision_time, group.reset_index(drop=True), excluded
    if not buffer.empty:
        for decision_time, group in buffer.groupby("decision_time_ms", sort=True):
            decision_time = int(decision_time)
            if last_yielded is not None and decision_time <= last_yielded:
                raise ValueError("Input rows are not chronologically ordered by flow completion time.")
            yield decision_time, group.reset_index(drop=True), {"input_rows": 0, "invalid_endpoint_rows": 0, "invalid_time_or_duration_rows": 0}


def load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"days": {}, "completed": False}
    return json.loads(path.read_text(encoding="utf-8"))


def save_profile_artifacts(root: Path, profiles, scalers) -> None:
    for profile in profiles:
        profile_root = root / profile.name
        profile_root.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(profile.to_dict() | {"sha256": profile.sha256()}, profile_root / "feature_schema.json")
        joblib.dump(scalers[profile.name], profile_root / "scaler.joblib")


def audit_output(root: Path, profiles, days: tuple[DaySpec, DaySpec]) -> dict[str, object]:
    """Audit every graph and verify timestamps per profile/split."""
    result: dict[str, object] = {"profiles": {}}
    for profile in profiles:
        profile_summary: dict[str, object] = {"splits": {}}
        for day in days:
            map_path = root / "mappings" / f"{day.name}_ip_to_id.json"
            mapping = IpIdMap.from_file(map_path)
            for split in day.split_names:
                graph_dir = root / profile.name / split
                if not graph_dir.exists():
                    continue
                files = sorted(graph_dir.glob("graph_*.pt"))
                previous_timestamp = None
                totals = Counter()
                for graph_path in files:
                    provenance_path = root / "provenance" / day.name / f"{graph_path.stem}.csv"
                    audit = audit_graph_file(graph_path, profile, mapping, provenance_path)
                    timestamp = int(graph_path.stem.split("_")[1])
                    if previous_timestamp is not None and timestamp <= previous_timestamp:
                        raise AssertionError(f"Graph timestamps are not strictly increasing in {graph_dir}.")
                    previous_timestamp = timestamp
                    totals.update(audit)
                profile_summary["splits"][split] = dict(totals)
        result["profiles"][profile.name] = profile_summary
    return result


def main() -> None:
    args = parse_args()
    if args.chunksize <= 0 or (args.max_windows is not None and args.max_windows <= 0):
        raise ValueError("chunksize and max-windows must be positive.")
    input_csvs = [path.resolve() for path in args.input_csv]
    for path in [*input_csvs, args.corrected_manifest]:
        if not path.is_file():
            raise FileNotFoundError(path)

    profiles = tuple(get_feature_profile(name) for name in (args.profile or ["nfv3_extended", "portable_core"]))
    days = make_days(args)
    root = args.output_root.resolve()
    state_path = root / "build_state.json"
    if args.overwrite:
        if root.exists():
            shutil.rmtree(root)
    elif root.exists() and not args.resume:
        raise FileExistsError(f"{root} already exists. Use a new graph version, --resume, or deliberate --overwrite.")
    root.mkdir(parents=True, exist_ok=True)

    state = load_state(state_path)
    if state.get("completed"):
        raise RuntimeError("This graph build is already complete. Use a new versioned output root.")
    usecols = required_columns(profiles)
    day1_cutoffs = split_cutoffs(input_csvs, days[0], args.chunksize, usecols)

    scaler_paths = {profile.name: root / profile.name / "scaler.joblib" for profile in profiles}
    if args.resume and all(path.is_file() for path in scaler_paths.values()):
        scalers = {profile.name: joblib.load(scaler_paths[profile.name]) for profile in profiles}
    else:
        scalers = fit_scalers(input_csvs, days[0], profiles, day1_cutoffs, args.chunksize)
        save_profile_artifacts(root, profiles, scalers)

    build_metadata = {
        "input_csvs": [{"path": str(path), "sha256": sha256_file(path)} for path in input_csvs],
        "corrected_manifest": {"path": str(args.corrected_manifest), "sha256": sha256_file(args.corrected_manifest)},
        "profiles": {profile.name: profile.sha256() for profile in profiles},
        "window_ms": 30_000,
        "window_policy": "flow_end_in_half_open_window; decision_time_is_window_close",
        "day1_cutoffs": day1_cutoffs,
    }
    atomic_json_dump(build_metadata, root / "build_configuration.json")

    for day in days:
        day_state = state.setdefault("days", {}).setdefault(day.name, {})
        map_path = root / "mappings" / f"{day.name}_ip_to_id.json"
        mapping = IpIdMap.from_file(map_path) if args.resume and map_path.exists() else IpIdMap()
        last_completed = day_state.get("last_completed_decision_time_ms") if args.resume else None
        processed = 0
        exclusions = Counter(day_state.get("exclusions", {}))
        for decision_time, group, counts in iter_complete_windows(input_csvs, day, usecols, args.chunksize):
            exclusions.update(counts)
            if last_completed is not None and decision_time <= int(last_completed):
                continue
            split = split_for_time(day, decision_time, day1_cutoffs)
            graph_name = f"graph_{decision_time:013d}.pt"
            provenance_name = f"graph_{decision_time:013d}.csv"
            provenance = None
            for profile in profiles:
                graph, current_provenance = build_graph(group, profile, scalers[profile.name], mapping)
                atomic_torch_save(graph, root / profile.name / split / graph_name)
                provenance = current_provenance
            provenance_path = root / "provenance" / day.name / provenance_name
            provenance_path.parent.mkdir(parents=True, exist_ok=True)
            provenance.to_csv(provenance_path, index=False)
            atomic_json_dump(mapping.payload(day.name), map_path)
            day_state["last_completed_decision_time_ms"] = decision_time
            day_state["exclusions"] = dict(exclusions)
            state["completed"] = False
            atomic_json_dump(state, state_path)
            processed += 1
            if args.max_windows is not None and processed >= args.max_windows:
                print(f"Stopped after {processed} windows for {day.name} as requested.")
                break

    is_partial = args.max_windows is not None
    audit = audit_output(root, profiles, days)
    audit["status"] = "partial" if is_partial else "passed"
    atomic_json_dump(audit, root / "graph_audit.json")
    state["completed"] = not is_partial
    atomic_json_dump(state, state_path)
    if not is_partial:
        atomic_json_dump(build_metadata | {"audit": audit, "status": "passed"}, root / "graph_manifest.json")
    print(json.dumps({"output_root": str(root), "audit": str(root / "graph_audit.json"), "status": audit["status"]}, indent=2))


if __name__ == "__main__":
    main()
