#!/usr/bin/env python3
"""Build auditable Day-1 and Day-2 NF-v3 temporal graph collections.

Run this script from the thin Colab notebook. Start with ``--max-windows`` for
a smoke build; omit it only after reviewing the generated audit and manifests.
"""

from __future__ import annotations

import argparse
import hashlib
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
    DAY1, DAY2, END_TIME_COLUMN, DaySpec, IpIdMap, atomic_json_dump, atomic_torch_save,
    audit_graph_file, audit_provenance_file, build_graph, feature_preflight_audit, fit_scalers,
    prepare_chunk, required_columns, sha256_file, split_cutoffs, split_for_time,
)
from utils.graph_schema import get_feature_profile  # noqa: E402


DEFAULT_CHECKPOINT_EVERY_WINDOWS = 200


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
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY_WINDOWS,
                        help="Persist the day mapping and build state every N newly built windows (default: 200).")
    parser.add_argument("--max-windows", type=int, default=None,
                        help="Stop after this many windows per day for a smoke build.")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Write the full input feature audit and exit without producing graphs.")
    parser.add_argument("--resume", action="store_true",
                        help="Resume a previously interrupted build with the same output root and configuration.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Delete the exact output root before beginning a new build.")
    return parser.parse_args()


def make_days(args: argparse.Namespace) -> tuple[DaySpec, DaySpec]:
    return (
        DaySpec(
            name=DAY1.name,
            source_file=args.day1_source_file,
            split_names=DAY1.split_names,
            split_policy=DAY1.split_policy,
            train_end_ms=DAY1.train_end_ms,
            val_end_ms=DAY1.val_end_ms,
        ),
        DaySpec(
            name=DAY2.name,
            source_file=args.day2_source_file,
            split_names=DAY2.split_names,
            split_policy=DAY2.split_policy,
        ),
    )


def iter_complete_windows(input_csvs: list[Path], day: DaySpec, usecols: list[str], chunksize: int):
    """Yield complete chronological decision-time windows without chunk splits.

    Input rows must be ordered by flow start within each day. A window is safe
    to emit once its decision time is not later than the greatest flow start
    observed so far: every unseen flow starts at or after that watermark and
    therefore must finish strictly after it.
    """
    buffer = pd.DataFrame()
    last_yielded: int | None = None
    last_seen_start: float | None = None
    for path in input_csvs:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            chunk = chunk.loc[chunk["source_file"].eq(day.source_file)]
            if chunk.empty:
                continue
            prepared, _ = prepare_chunk(chunk)
            if prepared.empty:
                continue

            starts = prepared["flow_start_ms"]
            if not starts.is_monotonic_increasing:
                raise ValueError(
                    f"Rows for {day.source_file} are not ordered by FLOW_START_MILLISECONDS."
                )
            chunk_min_start = float(starts.iloc[0])
            if last_seen_start is not None and chunk_min_start < last_seen_start:
                raise ValueError(
                    f"Rows for {day.source_file} are not ordered by FLOW_START_MILLISECONDS."
                )
            last_seen_start = float(starts.iloc[-1])

            combined = pd.concat((buffer, prepared), ignore_index=True)
            complete_mask = combined["decision_time_ms"] <= last_seen_start
            complete = combined.loc[complete_mask]
            buffer = combined.loc[~complete_mask].copy()
            for decision_time, group in complete.groupby("decision_time_ms", sort=True):
                decision_time = int(decision_time)
                if last_yielded is not None and decision_time <= last_yielded:
                    raise ValueError("Input rows are not chronologically ordered by flow completion time.")
                last_yielded = decision_time
                yield decision_time, group.reset_index(drop=True)
    if not buffer.empty:
        for decision_time, group in buffer.groupby("decision_time_ms", sort=True):
            decision_time = int(decision_time)
            if last_yielded is not None and decision_time <= last_yielded:
                raise ValueError("Input rows are not chronologically ordered by flow completion time.")
            yield decision_time, group.reset_index(drop=True)


def load_state(path: Path) -> dict[str, object]:
    if not path.exists():
        return {"days": {}, "completed": False}
    return json.loads(path.read_text(encoding="utf-8"))


def register_window_endpoints(mapping: IpIdMap, group: pd.DataFrame) -> None:
    """Replay the exact append-only ID assignment performed by ``build_graph``."""
    for ip in group["source_ip"]:
        mapping.id_for(ip)
    for ip in group["destination_ip"]:
        mapping.id_for(ip)


def save_day_checkpoint(
    mapping: IpIdMap,
    map_path: Path,
    day: DaySpec,
    state: dict[str, object],
    state_path: Path,
) -> None:
    """Persist a recoverable mapping/state pair, publishing state last.

    Resume always reconstructs the mapping chronologically from the input, so
    a crash after the map write but before the state write is harmless.
    """
    atomic_json_dump(mapping.payload(day.name), map_path)
    atomic_json_dump(state, state_path)


def save_profile_artifacts(root: Path, profiles, scalers) -> None:
    for profile in profiles:
        profile_root = root / profile.name
        profile_root.mkdir(parents=True, exist_ok=True)
        atomic_json_dump(profile.to_dict() | {"sha256": profile.sha256()}, profile_root / "feature_schema.json")
        joblib.dump(scalers[profile.name], profile_root / "scaler.joblib")


def collection_checksum_summary(files: dict[str, dict[str, int | str]]) -> dict[str, int | str]:
    """Hash an ordered path/file-hash index into one deterministic collection digest."""
    digest = hashlib.sha256()
    total_bytes = 0
    for relative_path, record in sorted(files.items()):
        total_bytes += int(record["bytes"])
        line = json.dumps(
            [relative_path, record["sha256"], int(record["bytes"])],
            separators=(",", ":"),
        )
        digest.update(line.encode("utf-8") + b"\n")
    return {"sha256": digest.hexdigest(), "files": len(files), "bytes": total_bytes}


def file_artifact(root: Path, path: Path) -> dict[str, int | str]:
    """Describe one exact artifact using a root-relative path and SHA-256."""
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build_artifact_summary(root: Path, profiles, days, checksums_path: Path, checksums: dict) -> dict[str, object]:
    """Build the final reproducibility summary for immutable graph artifacts."""
    return {
        "collection_digest_contract": (
            "sha256 of newline-delimited compact JSON arrays "
            "[root_relative_path,file_sha256,byte_size] sorted by path"
        ),
        "checksum_index": file_artifact(root, checksums_path),
        "feature_schemas": {
            profile.name: file_artifact(root, root / profile.name / "feature_schema.json")
            for profile in profiles
        },
        "scalers": {
            profile.name: file_artifact(root, root / profile.name / "scaler.joblib")
            for profile in profiles
        },
        "mappings": {
            day.name: file_artifact(root, root / "mappings" / f"{day.name}_ip_to_id.json")
            for day in days
        },
        "graph_collections": {
            profile.name: collection_checksum_summary(checksums["graphs"][profile.name])
            for profile in profiles
        },
        "provenance_collection": collection_checksum_summary(checksums["provenance"]),
    }


def audit_output(
    root: Path,
    profiles,
    days: tuple[DaySpec, DaySpec],
    preflight: dict[str, object],
    *,
    partial: bool,
    artifact_checksums: dict[str, object] | None = None,
) -> dict[str, object]:
    """Audit aligned profile graphs while reading each provenance table once."""
    profiles = tuple(profiles)
    if artifact_checksums is None:
        artifact_checksums = {
            "algorithm": "sha256",
            "graphs": {profile.name: {} for profile in profiles},
            "provenance": {},
        }
    result: dict[str, object] = {
        "profiles": {
            profile.name: {"splits": {}, "conservation_by_day": {}}
            for profile in profiles
        },
        "input_accounting": {
            key: preflight[key]
            for key in (
                "input_rows", "positive_rows", "retained_rows",
                "retained_positive_rows", "excluded_rows",
                "excluded_positive_rows", "by_source_file",
            )
        },
    }
    seen_flow_ids: set[str] = set()
    expected_graph_paths = {profile.name: set() for profile in profiles}
    expected_provenance_paths: set[Path] = set()
    for day in days:
        map_path = root / "mappings" / f"{day.name}_ip_to_id.json"
        mapping = IpIdMap.from_file(map_path)
        day_totals = {profile.name: Counter() for profile in profiles}
        previous_split_max_timestamp: int | None = None

        for split in day.split_names:
            graph_dirs = {
                profile.name: root / profile.name / split
                for profile in profiles
            }
            files_by_profile = {
                profile.name: sorted(graph_dirs[profile.name].glob("graph_*.pt"))
                if graph_dirs[profile.name].exists() else []
                for profile in profiles
            }
            if not any(graph_dirs[profile.name].exists() for profile in profiles):
                continue

            reference_names = [path.name for path in files_by_profile[profiles[0].name]]
            for profile in profiles[1:]:
                profile_names = [path.name for path in files_by_profile[profile.name]]
                if profile_names != reference_names:
                    raise AssertionError(
                        f"Profile graph windows disagree for {day.name}/{split}: "
                        f"{profiles[0].name} versus {profile.name}."
                    )

            timestamps = [int(Path(name).stem.split("_")[1]) for name in reference_names]
            if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
                raise AssertionError(f"Graph timestamps are not strictly increasing for {day.name}/{split}.")
            if timestamps:
                if previous_split_max_timestamp is not None and timestamps[0] <= previous_split_max_timestamp:
                    raise AssertionError(f"Graph splits are not chronologically ordered for {day.name}.")
                previous_split_max_timestamp = timestamps[-1]

            split_totals = {profile.name: Counter() for profile in profiles}
            for graph_name, timestamp in zip(reference_names, timestamps):
                provenance_path = root / "provenance" / day.name / f"{Path(graph_name).stem}.csv"
                expected_provenance_paths.add(provenance_path)
                provenance, flow_ids, provenance_artifact = audit_provenance_file(
                    provenance_path, mapping, split, timestamp,
                )
                provenance_relative = provenance_path.relative_to(root).as_posix()
                artifact_checksums["provenance"][provenance_relative] = provenance_artifact
                duplicates = seen_flow_ids.intersection(flow_ids)
                if duplicates:
                    duplicate = next(iter(duplicates))
                    raise AssertionError(f"Flow ID {duplicate} occurs in more than one graph.")
                seen_flow_ids.update(flow_ids)

                for profile in profiles:
                    graph_path = graph_dirs[profile.name] / graph_name
                    expected_graph_paths[profile.name].add(graph_path)
                    audit, _, graph_artifact = audit_graph_file(
                        graph_path, profile, mapping, provenance_path, split,
                        provenance=provenance,
                    )
                    graph_relative = graph_path.relative_to(root).as_posix()
                    artifact_checksums["graphs"][profile.name][graph_relative] = graph_artifact
                    split_totals[profile.name].update(audit)

            for profile in profiles:
                result["profiles"][profile.name]["splits"][split] = dict(split_totals[profile.name])
                day_totals[profile.name].update(split_totals[profile.name])

        expected = preflight["by_source_file"][day.source_file]
        for profile in profiles:
            totals = day_totals[profile.name]
            conservation = {
                "source_file": day.source_file,
                "expected_edges": int(expected["retained_rows"]),
                "observed_edges": int(totals["edges"]),
                "expected_positive_edges": int(expected["retained_positive_rows"]),
                "observed_positive_edges": int(totals["positive_edges"]),
            }
            matches = (
                conservation["observed_edges"] == conservation["expected_edges"]
                and conservation["observed_positive_edges"] == conservation["expected_positive_edges"]
            )
            conservation["status"] = "partial" if partial else ("passed" if matches else "failed")
            if not partial and not matches:
                raise AssertionError(
                    f"Flow conservation failed for {profile.name}/{day.name}: {conservation}"
                )
            result["profiles"][profile.name]["conservation_by_day"][day.name] = conservation

    actual_provenance_paths = set((root / "provenance").rglob("graph_*.csv"))
    if actual_provenance_paths != expected_provenance_paths:
        raise AssertionError("Provenance files do not exactly match the audited graph windows.")
    for profile in profiles:
        actual_graph_paths = set((root / profile.name).rglob("graph_*.pt"))
        if actual_graph_paths != expected_graph_paths[profile.name]:
            raise AssertionError(f"Unexpected or unaudited graph files found for {profile.name}.")
    return result


def main() -> None:
    args = parse_args()
    if (
        args.chunksize <= 0
        or args.checkpoint_every <= 0
        or (args.max_windows is not None and args.max_windows <= 0)
    ):
        raise ValueError("chunksize, checkpoint-every, and max-windows must be positive.")
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

    if args.preflight_only:
        preflight = feature_preflight_audit(input_csvs, profiles, args.chunksize)
        atomic_json_dump(preflight, root / "feature_preflight.json")
        print(json.dumps({"output_root": str(root), "preflight": str(root / "feature_preflight.json"), "status": preflight["status"]}, indent=2))
        if preflight["status"] != "passed":
            raise RuntimeError("Feature preflight failed. Resolve data-quality failures before graph construction.")
        return

    preflight = feature_preflight_audit(input_csvs, profiles, args.chunksize)
    atomic_json_dump(preflight, root / "feature_preflight.json")
    if preflight["status"] != "passed":
        raise RuntimeError("Feature preflight failed. Resolve data-quality failures before graph construction.")

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
        "flow_end_column": END_TIME_COLUMN,
        "day1_cutoffs": day1_cutoffs,
        "checkpoint_every_windows": args.checkpoint_every,
    }
    atomic_json_dump(build_metadata, root / "build_configuration.json")

    for day in days:
        day_state = state.setdefault("days", {}).setdefault(day.name, {})
        map_path = root / "mappings" / f"{day.name}_ip_to_id.json"
        # Rebuild from the chronological stream even on resume. Loading a map
        # checkpoint and then skipping earlier windows can silently change IDs
        # when the map and state were persisted at different instants.
        mapping = IpIdMap()
        last_completed = day_state.get("last_completed_decision_time_ms") if args.resume else None
        processed = 0
        since_checkpoint = 0
        for decision_time, group in iter_complete_windows(input_csvs, day, usecols, args.chunksize):
            if last_completed is not None and decision_time <= int(last_completed):
                register_window_endpoints(mapping, group)
                continue
            split = split_for_time(day, decision_time, day1_cutoffs)
            graph_name = f"graph_{decision_time:013d}.pt"
            provenance_name = f"graph_{decision_time:013d}.csv"
            provenance = None
            for profile in profiles:
                graph, current_provenance = build_graph(
                    group, profile, scalers[profile.name], mapping, split,
                )
                atomic_torch_save(graph, root / profile.name / split / graph_name)
                provenance = current_provenance
            provenance_path = root / "provenance" / day.name / provenance_name
            provenance_path.parent.mkdir(parents=True, exist_ok=True)
            provenance.to_csv(provenance_path, index=False)
            day_state["last_completed_decision_time_ms"] = decision_time
            state["completed"] = False
            processed += 1
            since_checkpoint += 1
            if since_checkpoint >= args.checkpoint_every:
                save_day_checkpoint(mapping, map_path, day, state, state_path)
                since_checkpoint = 0
            if args.max_windows is not None and processed >= args.max_windows:
                print(f"Stopped after {processed} windows for {day.name} as requested.")
                break
        # Flush at every day boundary and after an intentional smoke stop,
        # including when resume only replayed already completed windows.
        save_day_checkpoint(mapping, map_path, day, state, state_path)

    is_partial = args.max_windows is not None
    artifact_checksums = {
        "algorithm": "sha256",
        "graphs": {profile.name: {} for profile in profiles},
        "provenance": {},
    }
    audit = audit_output(
        root, profiles, days, preflight, partial=is_partial,
        artifact_checksums=artifact_checksums,
    )
    audit["status"] = "partial" if is_partial else "passed"
    checksums_path = root / "artifact_checksums.json"
    atomic_json_dump(artifact_checksums, checksums_path)
    audit["artifacts"] = build_artifact_summary(
        root, profiles, days, checksums_path, artifact_checksums,
    )
    atomic_json_dump(audit, root / "graph_audit.json")
    state["completed"] = not is_partial
    atomic_json_dump(state, state_path)
    if not is_partial:
        atomic_json_dump(
            build_metadata | {
                "input_accounting": audit["input_accounting"],
                "artifacts": audit["artifacts"],
                "audit": audit,
                "status": "passed",
            },
            root / "graph_manifest.json",
        )
    print(json.dumps({"output_root": str(root), "audit": str(root / "graph_audit.json"), "status": audit["status"]}, indent=2))


if __name__ == "__main__":
    main()
