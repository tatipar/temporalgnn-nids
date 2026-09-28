"""Streaming structural audit for the cAPTure graph pilot.

This module does not fit a model and does not access held-out scenarios. It
constructs each five-second development snapshot in memory, records compact
window metrics, and persists only hashed endpoint/pair aggregates needed for
shortcut diagnostics.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time
from typing import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .capture_data import selected_scenarios, sha256_file, write_json
from .capture_feature_profile import load_prepared_full_dev
from .capture_xgb_p import NANOSECONDS_PER_SECOND, window_coordinates


REPORT_VERSION = 1
AUDIT_COLUMNS = (
    "source_row_id",
    "packet_timestamp_ns",
    "src_endpoint",
    "dst_endpoint",
    "src_node_role",
    "dst_node_role",
    "binary_label",
    "attack_step",
    "sequence_id",
)
SUMMARY_METRICS = (
    "nodes",
    "edges",
    "unique_directed_pairs",
    "unique_undirected_pairs",
    "weak_components",
    "largest_weak_component_fraction",
    "simple_directed_density",
    "mean_multiplicity",
    "max_multiplicity",
    "parallel_edge_fraction",
    "branching_node_fraction",
    "adjacent_node_jaccard",
    "adjacent_pair_jaccard",
    "previous_nonempty_node_jaccard",
    "previous_nonempty_pair_jaccard",
)


def load_structural_audit_config(path: Path) -> dict:
    """Load the frozen Stage-1 audit contract and reject implicit defaults."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if (
        config.get("audit_version") != 1
        or config.get("scope") != "development_only"
        or config.get("stage") != "structural_audit_without_training"
    ):
        raise ValueError("Unsupported graph structural-audit configuration.")
    windows = config.get("windows", {})
    required_windows = {
        "duration_seconds": 5,
        "type": "fixed_non_overlapping",
        "interval": "left_closed_right_open",
        "origin_rule": "first_packet_timestamp_per_scenario",
        "decision_time": "window_end",
        "empty_windows": "count_for_exposure_but_do_not_materialize",
    }
    for name, value in required_windows.items():
        if windows.get(name) != value:
            raise ValueError(f"Structural-audit window contract changed: {name}")
    graph = config.get("graph", {})
    required_graph = {
        "type": "directed_temporal_multigraph",
        "node_key": "normalized_ethernet_mac_address",
        "edge_unit": "packet",
        "preserve_parallel_edges": True,
        "preserve_self_loops": True,
        "weak_components": True,
        "simple_density_excludes_self_loops": True,
        "endpoint_values_in_outputs": "sha256_only",
        "materialize_training_graphs": False,
    }
    for name, value in required_graph.items():
        if graph.get(name) != value:
            raise ValueError(f"Structural-audit graph contract changed: {name}")
    percentiles = config.get("percentiles")
    if percentiles != [0, 1, 5, 25, 50, 75, 95, 99, 100]:
        raise ValueError("The structural-audit percentile grid changed.")
    indicators = config.get("protocol_indicators")
    if not indicators or len(indicators) != len(set(indicators)):
        raise ValueError("Protocol indicators must be a non-empty unique list.")
    if set(config.get("topology_metrics", [])) != set(SUMMARY_METRICS):
        raise ValueError("The configured topology metrics differ from the code contract.")
    resources = config.get("resource_accounting", {})
    for name in (
        "model_edge_feature_count",
        "model_edge_feature_dtype_bytes",
        "topology_index_dtype_bytes",
        "target_dtype_bytes",
        "row_id_dtype_bytes",
    ):
        if not isinstance(resources.get(name), int) or resources[name] <= 0:
            raise ValueError(f"Invalid resource-accounting field: {name}")
    return config


def _stable_hash(kind: str, *values: str) -> str:
    payload = json.dumps([kind, *values], ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _rss_bytes() -> int | None:
    """Return current Linux RSS without adding a runtime dependency."""
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _safe_fraction(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _jaccard(left: set, right: set) -> float:
    union = left | right
    return _safe_fraction(len(left & right), len(union)) if union else 1.0


class _DisjointSet:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: str) -> str:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: str, right: str) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.size[left_root] < self.size[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        self.size[left_root] += self.size[right_root]

    def component_sizes(self) -> list[int]:
        sizes = Counter(self.find(value) for value in self.parent)
        return list(sizes.values())


def _iter_window_frames(
    path: Path,
    columns: list[str],
    *,
    origin_ns: int,
    width_seconds: int,
    batch_size: int,
    batch_callback=None,
):
    """Yield complete chronological window frames across Parquet batch cuts."""
    parquet = pq.ParquetFile(path)
    expected_row_id = 0
    previous_timestamp = None
    active_index = None
    active_parts: list[pd.DataFrame] = []
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        row_ids = frame["source_row_id"].to_numpy(dtype=np.int64)
        expected = np.arange(expected_row_id, expected_row_id + len(frame), dtype=np.int64)
        if not np.array_equal(row_ids, expected):
            raise ValueError("Prepared source_row_id order changed during graph audit.")
        expected_row_id += len(frame)
        timestamps = frame["packet_timestamp_ns"].to_numpy(dtype=np.int64)
        if np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError("Packet timestamps are not ordered inside a Parquet batch.")
        if previous_timestamp is not None and int(timestamps[0]) < previous_timestamp:
            raise ValueError("Packet timestamps decreased across Parquet batches.")
        previous_timestamp = int(timestamps[-1])
        if batch_callback is not None:
            batch_callback(frame)
        indexes, _ = window_coordinates(timestamps, origin_ns, width_seconds)
        frame["_window_index"] = indexes
        for raw_index, group in frame.groupby("_window_index", sort=False, observed=True):
            index = int(raw_index)
            part = group.drop(columns="_window_index")
            if active_index is None:
                active_index = index
            if index < active_index:
                raise ValueError("Window indexes are not chronological.")
            if index != active_index:
                yield active_index, pd.concat(active_parts, ignore_index=True)
                active_index, active_parts = index, []
            active_parts.append(part)
    if active_index is not None:
        yield active_index, pd.concat(active_parts, ignore_index=True)


def _update_shortcut_counts(
    frame: pd.DataFrame,
    pair_counts: dict,
    endpoint_counts: dict,
    endpoint_hash_cache: dict[str, str],
    protocol_counts: dict,
    role_counts: dict,
    protocol_indicators: list[str],
) -> None:
    labels = frame["binary_label"].to_numpy(dtype=np.int8)
    grouped_pairs = frame.groupby(
        ["src_endpoint", "dst_endpoint", "binary_label"],
        observed=True,
        dropna=False,
    ).size()
    for (raw_source, raw_destination, raw_label), raw_count in grouped_pairs.items():
        source, destination = str(raw_source), str(raw_destination)
        label, count = int(raw_label), int(raw_count)
        if source not in endpoint_hash_cache:
            endpoint_hash_cache[source] = _stable_hash("endpoint", source)
        if destination not in endpoint_hash_cache:
            endpoint_hash_cache[destination] = _stable_hash("endpoint", destination)
        src_hash = endpoint_hash_cache[source]
        dst_hash = endpoint_hash_cache[destination]
        pair_hash = _stable_hash("directed_pair", source, destination)
        endpoint_counts[src_hash][label] += count
        endpoint_counts[dst_hash][label] += count
        item = pair_counts.get(pair_hash)
        if item is None:
            item = {
                "key_type": "directed_pair",
                "key_hash": pair_hash,
                "src_hash": src_hash,
                "dst_hash": dst_hash,
                "normal_packets": 0,
                "attack_packets": 0,
            }
            pair_counts[pair_hash] = item
        item["attack_packets" if label else "normal_packets"] += count
    for name in protocol_indicators:
        present = pd.to_numeric(frame[name], errors="coerce").fillna(0).to_numpy() == 1
        protocol_counts[name]["present_normal"] += int(np.count_nonzero(present & (labels == 0)))
        protocol_counts[name]["present_attack"] += int(np.count_nonzero(present & (labels == 1)))
    for direction in ("src", "dst"):
        column = f"{direction}_node_role"
        grouped_roles = frame.groupby(
            [column, "binary_label"], observed=True, dropna=False
        ).size()
        for (raw_role, raw_label), raw_count in grouped_roles.items():
            role_counts[f"{direction}:{raw_role}"][int(raw_label)] += int(raw_count)


def _window_record(
    frame: pd.DataFrame,
    *,
    scenario: str,
    benign_source: str,
    window_index: int,
    origin_ns: int,
    width_seconds: int,
    protocol_indicators: list[str],
    previous_nodes: set[str] | None,
    previous_pairs: set[tuple[str, str]] | None,
    previous_index: int | None,
    resources: dict,
) -> tuple[dict, set[str], set[tuple[str, str]]]:
    started = time.perf_counter()
    sources = frame["src_endpoint"].astype(str).tolist()
    destinations = frame["dst_endpoint"].astype(str).tolist()
    pairs = list(zip(sources, destinations))
    pair_counter = Counter(pairs)
    unique_pairs = set(pair_counter)
    nodes = set(sources) | set(destinations)
    undirected_pairs = {tuple(sorted(pair)) for pair in unique_pairs}
    self_loops = sum(count for pair, count in pair_counter.items() if pair[0] == pair[1])
    nonself_unique_pairs = sum(1 for source, destination in unique_pairs if source != destination)

    components = _DisjointSet(nodes)
    neighbors = {node: set() for node in nodes}
    for source, destination in unique_pairs:
        components.union(source, destination)
        if source != destination:
            neighbors[source].add(destination)
            neighbors[destination].add(source)
    component_sizes = components.component_sizes()
    branching_nodes = sum(len(peers) >= 2 for peers in neighbors.values())

    labels = frame["binary_label"].to_numpy(dtype=np.int8)
    normal = int(np.count_nonzero(labels == 0))
    attack = int(np.count_nonzero(labels == 1))
    if normal and attack:
        window_type = "mixed"
    elif attack:
        window_type = "attack_only"
    else:
        window_type = "benign_only"
    attack_steps = sorted(
        str(value) for value in frame.loc[frame["binary_label"].eq(1), "attack_step"].dropna().unique()
    )
    iterations = sorted(
        str(value) for value in frame.loc[frame["binary_label"].eq(1), "sequence_id"].dropna().unique()
    )

    width_ns = width_seconds * NANOSECONDS_PER_SECOND
    start_ns = origin_ns + window_index * width_ns
    end_ns = start_ns + width_ns
    edge_count = len(frame)
    node_count = len(nodes)
    unique_pair_count = len(unique_pairs)
    adjacent = previous_index is not None and window_index == previous_index + 1
    endpoint_hashes = sorted(_stable_hash("endpoint", value) for value in nodes)
    pair_hashes = sorted(_stable_hash("directed_pair", *pair) for pair in unique_pairs)
    topology_bytes = (
        2 * edge_count * resources["topology_index_dtype_bytes"]
        + edge_count * resources["target_dtype_bytes"]
        + edge_count * resources["row_id_dtype_bytes"]
        + node_count * resources["topology_index_dtype_bytes"]
    )
    model_bytes = topology_bytes + (
        edge_count
        * resources["model_edge_feature_count"]
        * resources["model_edge_feature_dtype_bytes"]
    )
    record = {
        "scenario": scenario,
        "benign_source": benign_source,
        "window_index": int(window_index),
        "window_start_ns": int(start_ns),
        "window_end_ns": int(end_ns),
        "decision_time_ns": int(end_ns),
        "first_packet_timestamp_ns": int(frame["packet_timestamp_ns"].iloc[0]),
        "last_packet_timestamp_ns": int(frame["packet_timestamp_ns"].iloc[-1]),
        "first_source_row_id": int(frame["source_row_id"].iloc[0]),
        "last_source_row_id": int(frame["source_row_id"].iloc[-1]),
        "nodes": node_count,
        "edges": edge_count,
        "unique_directed_pairs": unique_pair_count,
        "unique_undirected_pairs": len(undirected_pairs),
        "self_loop_edges": int(self_loops),
        "weak_components": len(component_sizes),
        "largest_weak_component_nodes": max(component_sizes),
        "largest_weak_component_fraction": _safe_fraction(max(component_sizes), node_count),
        "simple_directed_density": _safe_fraction(
            nonself_unique_pairs, node_count * (node_count - 1)
        ),
        "mean_multiplicity": _safe_fraction(edge_count, unique_pair_count),
        "max_multiplicity": max(pair_counter.values()),
        "parallel_edge_fraction": _safe_fraction(edge_count - unique_pair_count, edge_count),
        "branching_nodes": int(branching_nodes),
        "branching_node_fraction": _safe_fraction(branching_nodes, node_count),
        "gap_empty_windows": (
            int(window_index - previous_index - 1) if previous_index is not None else 0
        ),
        "adjacent_node_jaccard": (
            _jaccard(nodes, previous_nodes) if adjacent and previous_nodes is not None else np.nan
        ),
        "adjacent_pair_jaccard": (
            _jaccard(unique_pairs, previous_pairs)
            if adjacent and previous_pairs is not None
            else np.nan
        ),
        "previous_nonempty_node_jaccard": (
            _jaccard(nodes, previous_nodes) if previous_nodes is not None else np.nan
        ),
        "previous_nonempty_pair_jaccard": (
            _jaccard(unique_pairs, previous_pairs) if previous_pairs is not None else np.nan
        ),
        "normal_packets": normal,
        "attack_packets": attack,
        "window_type": window_type,
        "attack_steps_json": json.dumps(attack_steps, separators=(",", ":")),
        "attack_step_count": len(attack_steps),
        "iteration_count": len(iterations),
        "iterations_sha256": _stable_hash("iterations", *iterations),
        "endpoint_set_sha256": _stable_hash("endpoint_set", *endpoint_hashes),
        "topology_sha256": _stable_hash("topology", *pair_hashes),
        "estimated_topology_tensor_bytes": int(topology_bytes),
        "estimated_model_tensor_bytes": int(model_bytes),
    }
    for name in protocol_indicators:
        record[f"{name}_packets"] = int(
            (pd.to_numeric(frame[name], errors="coerce").fillna(0) == 1).sum()
        )
    record["construction_seconds"] = float(time.perf_counter() - started)
    return record, nodes, unique_pairs


def _write_scenario_artifacts(
    *,
    scenario: str,
    benign_source: str,
    packet_path: Path,
    prepared_report: dict,
    output_dir: Path,
    config: dict,
    config_sha256: str,
    batch_size: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    protocol_indicators = list(config["protocol_indicators"])
    columns = list(dict.fromkeys([*AUDIT_COLUMNS, *protocol_indicators]))
    available = set(pq.read_schema(packet_path).names)
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"Prepared packet artifact is missing audit columns: {missing}")
    origin_ns = int(prepared_report["scenario_origin_timestamp_ns"])
    expected_rows = int(prepared_report["counts"]["packets"])
    width_seconds = int(config["windows"]["duration_seconds"])
    resources = config["resource_accounting"]
    pair_counts: dict[str, dict] = {}
    endpoint_counts = defaultdict(lambda: [0, 0])
    endpoint_hash_cache: dict[str, str] = {}
    protocol_counts = {
        name: {"present_normal": 0, "present_attack": 0}
        for name in protocol_indicators
    }
    role_counts = defaultdict(lambda: [0, 0])
    window_records = []
    previous_nodes = previous_pairs = None
    previous_index = None
    observed_rows = 0
    peak_rss = _rss_bytes()
    started = time.perf_counter()

    def update_shortcut_batch(frame: pd.DataFrame) -> None:
        _update_shortcut_counts(
            frame,
            pair_counts,
            endpoint_counts,
            endpoint_hash_cache,
            protocol_counts,
            role_counts,
            protocol_indicators,
        )

    for window_index, frame in _iter_window_frames(
        packet_path,
        columns,
        origin_ns=origin_ns,
        width_seconds=width_seconds,
        batch_size=batch_size,
        batch_callback=update_shortcut_batch,
    ):
        if set(frame["binary_label"].dropna().unique()) - {0, 1}:
            raise ValueError(f"Invalid labels in {scenario}.")
        record, nodes, pairs = _window_record(
            frame,
            scenario=scenario,
            benign_source=benign_source,
            window_index=window_index,
            origin_ns=origin_ns,
            width_seconds=width_seconds,
            protocol_indicators=protocol_indicators,
            previous_nodes=previous_nodes,
            previous_pairs=previous_pairs,
            previous_index=previous_index,
            resources=resources,
        )
        window_records.append(record)
        observed_rows += len(frame)
        previous_nodes, previous_pairs, previous_index = nodes, pairs, window_index
        rss = _rss_bytes()
        if rss is not None:
            peak_rss = rss if peak_rss is None else max(peak_rss, rss)
        if len(window_records) % 5000 == 0:
            print(
                f"{scenario}: {len(window_records):,} nonempty windows, "
                f"{observed_rows:,} packets",
                flush=True,
            )

    if observed_rows != expected_rows or not window_records:
        raise ValueError(
            f"Scenario row conservation failed for {scenario}: "
            f"observed={observed_rows}, expected={expected_rows}."
        )
    metrics = pd.DataFrame.from_records(window_records)
    if not metrics["window_index"].is_monotonic_increasing or metrics["window_index"].duplicated().any():
        raise ValueError(f"Window indexes are invalid for {scenario}.")
    metrics_path = output_dir / "window_metrics.parquet"
    metrics.to_parquet(metrics_path, index=False, compression="zstd")

    shortcut_rows = list(pair_counts.values())
    shortcut_rows.extend(
        {
            "key_type": "endpoint",
            "key_hash": key,
            "src_hash": None,
            "dst_hash": None,
            "normal_packets": int(counts[0]),
            "attack_packets": int(counts[1]),
        }
        for key, counts in endpoint_counts.items()
    )
    shortcut_path = output_dir / "shortcut_keys.parquet"
    pd.DataFrame.from_records(shortcut_rows).to_parquet(
        shortcut_path, index=False, compression="zstd"
    )
    totals = {
        "normal_packets": int(metrics["normal_packets"].sum()),
        "attack_packets": int(metrics["attack_packets"].sum()),
    }
    protocol_report = {
        "scenario": scenario,
        "totals": totals,
        "protocol_indicators": protocol_counts,
        "node_roles": {
            key: {"normal_packets": int(value[0]), "attack_packets": int(value[1])}
            for key, value in sorted(role_counts.items())
        },
    }
    write_json(output_dir / "protocol_counts.json", protocol_report)
    final_index = int(metrics["window_index"].max())
    total_windows = final_index + 1
    wall_seconds = time.perf_counter() - started
    scenario_report = {
        "report_version": REPORT_VERSION,
        "scenario": scenario,
        "benign_source": benign_source,
        "source_artifact_sha256": prepared_report["output_sha256"],
        "structural_config_sha256": config_sha256,
        "origin_ns": origin_ns,
        "width_seconds": width_seconds,
        "rows": observed_rows,
        "normal_packets": totals["normal_packets"],
        "attack_packets": totals["attack_packets"],
        "nonempty_windows": len(metrics),
        "total_wall_clock_windows": total_windows,
        "empty_windows": total_windows - len(metrics),
        "last_window_index": final_index,
        "wall_seconds": float(wall_seconds),
        "construction_seconds": float(metrics["construction_seconds"].sum()),
        "packets_per_wall_second": _safe_fraction(observed_rows, wall_seconds),
        "peak_sampled_rss_bytes": peak_rss,
        "window_metrics_bytes": metrics_path.stat().st_size,
        "shortcut_keys_bytes": shortcut_path.stat().st_size,
        "estimated_topology_tensor_bytes": int(
            metrics["estimated_topology_tensor_bytes"].sum()
        ),
        "estimated_model_tensor_bytes": int(metrics["estimated_model_tensor_bytes"].sum()),
        "window_type_counts": {
            str(key): int(value) for key, value in metrics["window_type"].value_counts().items()
        },
        "distinct_endpoint_set_hashes": int(metrics["endpoint_set_sha256"].nunique()),
        "distinct_topology_hashes": int(metrics["topology_sha256"].nunique()),
        "most_common_topology_fraction": float(
            metrics["topology_sha256"].value_counts(normalize=True).iloc[0]
        ),
    }
    write_json(output_dir / "scenario_report.json", scenario_report)
    artifact_names = (
        "window_metrics.parquet",
        "shortcut_keys.parquet",
        "protocol_counts.json",
        "scenario_report.json",
    )
    checksums = {name: sha256_file(output_dir / name) for name in artifact_names}
    write_json(output_dir / "artifact_checksums.json", checksums)
    write_json(
        output_dir / "run_status.json",
        {
            "complete": True,
            "scenario": scenario,
            "artifact_checksums_sha256": sha256_file(output_dir / "artifact_checksums.json"),
        },
    )
    return scenario_report


def _validate_scenario_artifacts(
    output_dir: Path,
    scenario: str,
    source_sha256: str,
    config_sha256: str,
) -> dict:
    required = (
        "window_metrics.parquet",
        "shortcut_keys.parquet",
        "protocol_counts.json",
        "scenario_report.json",
        "artifact_checksums.json",
        "run_status.json",
    )
    if not all((output_dir / name).is_file() for name in required):
        raise FileNotFoundError(f"Incomplete existing structural audit for {scenario}.")
    status = json.loads((output_dir / "run_status.json").read_text(encoding="utf-8"))
    checksums = json.loads((output_dir / "artifact_checksums.json").read_text(encoding="utf-8"))
    report = json.loads((output_dir / "scenario_report.json").read_text(encoding="utf-8"))
    if (
        status.get("complete") is not True
        or status.get("scenario") != scenario
        or status.get("artifact_checksums_sha256")
        != sha256_file(output_dir / "artifact_checksums.json")
    ):
        raise ValueError(f"Existing structural-audit status is invalid for {scenario}.")
    for name, expected in checksums.items():
        if sha256_file(output_dir / name) != expected:
            raise ValueError(f"Structural-audit checksum mismatch: {scenario}/{name}")
    if (
        report.get("scenario") != scenario
        or report.get("source_artifact_sha256") != source_sha256
        or report.get("structural_config_sha256") != config_sha256
    ):
        raise ValueError(f"Existing structural audit has different provenance: {scenario}.")
    return report


def _confusion(normal: int, attack: int, predict_positive: bool) -> dict:
    if predict_positive:
        return {"tp": int(attack), "fp": int(normal), "tn": 0, "fn": 0}
    return {"tp": 0, "fp": 0, "tn": int(normal), "fn": int(attack)}


def _merge_confusions(items: Iterable[dict]) -> dict:
    result = {name: 0 for name in ("tp", "fp", "tn", "fn")}
    for item in items:
        for name in result:
            result[name] += int(item[name])
    result.update(
        {
            "precision": _safe_fraction(result["tp"], result["tp"] + result["fp"]),
            "recall": _safe_fraction(result["tp"], result["tp"] + result["fn"]),
            "false_positive_rate": _safe_fraction(
                result["fp"], result["fp"] + result["tn"]
            ),
        }
    )
    return result


def _key_sets(frames: list[pd.DataFrame], key_type: str) -> tuple[set[str], set[str]]:
    selected = pd.concat(
        [frame.loc[frame["key_type"].eq(key_type)] for frame in frames],
        ignore_index=True,
    )
    grouped = selected.groupby("key_hash", as_index=False)[
        ["normal_packets", "attack_packets"]
    ].sum()
    seen_attack = set(grouped.loc[grouped["attack_packets"].gt(0), "key_hash"])
    attack_only = set(
        grouped.loc[
            grouped["attack_packets"].gt(0) & grouped["normal_packets"].eq(0), "key_hash"
        ]
    )
    return seen_attack, attack_only


def _evaluate_key_rules(
    validation: pd.DataFrame,
    endpoint_sets: tuple[set[str], set[str]],
    pair_sets: tuple[set[str], set[str]],
) -> dict:
    pairs = validation.loc[validation["key_type"].eq("directed_pair")]
    rules = {
        "endpoint_seen_with_attack_in_fold_train": (
            pairs["src_hash"].isin(endpoint_sets[0]) | pairs["dst_hash"].isin(endpoint_sets[0])
        ),
        "endpoint_attack_only_in_fold_train": (
            pairs["src_hash"].isin(endpoint_sets[1]) | pairs["dst_hash"].isin(endpoint_sets[1])
        ),
        "directed_pair_seen_with_attack_in_fold_train": pairs["key_hash"].isin(pair_sets[0]),
        "directed_pair_attack_only_in_fold_train": pairs["key_hash"].isin(pair_sets[1]),
    }
    output = {}
    for name, predicted in rules.items():
        items = [
            _confusion(int(row.normal_packets), int(row.attack_packets), bool(flag))
            for row, flag in zip(pairs.itertuples(index=False), predicted)
        ]
        output[name] = _merge_confusions(items)
    return output


def _protocol_rule_report(
    manifest: dict,
    scenarios: list[str],
    protocol_reports: dict[str, dict],
    protocol_indicators: list[str],
) -> dict:
    result = {"scenario_diagnostics": {}, "fold_trained_rules": {}}
    for scenario in scenarios:
        report = protocol_reports[scenario]
        normal = int(report["totals"]["normal_packets"])
        attack = int(report["totals"]["attack_packets"])
        diagnostics = {}
        for name in protocol_indicators:
            counts = report["protocol_indicators"][name]
            present_normal = int(counts["present_normal"])
            present_attack = int(counts["present_attack"])
            tpr = _safe_fraction(present_attack, attack)
            tnr = _safe_fraction(normal - present_normal, normal)
            auc = (tpr + tnr) / 2
            diagnostics[name] = {
                "present_normal": present_normal,
                "present_attack": present_attack,
                "attack_prevalence_when_present": _safe_fraction(
                    present_attack, present_attack + present_normal
                ),
                "auc_present_means_attack": auc,
                "best_direction_auc": max(auc, 1 - auc),
                "best_direction": "present" if auc >= 0.5 else "absent",
            }
        result["scenario_diagnostics"][scenario] = diagnostics
    if set(scenarios) != set(selected_scenarios(manifest, "FULL_DEV")):
        return result
    for fold, split in manifest["validation"]["folds"].items():
        fold_result = {}
        for name in protocol_indicators:
            train_present_normal = sum(
                protocol_reports[scenario]["protocol_indicators"][name]["present_normal"]
                for scenario in split["train"]
            )
            train_present_attack = sum(
                protocol_reports[scenario]["protocol_indicators"][name]["present_attack"]
                for scenario in split["train"]
            )
            train_normal = sum(
                protocol_reports[scenario]["totals"]["normal_packets"]
                for scenario in split["train"]
            )
            train_attack = sum(
                protocol_reports[scenario]["totals"]["attack_packets"]
                for scenario in split["train"]
            )
            present_prevalence = _safe_fraction(
                train_present_attack, train_present_attack + train_present_normal
            )
            absent_prevalence = _safe_fraction(
                train_attack - train_present_attack,
                train_attack + train_normal - train_present_attack - train_present_normal,
            )
            positive_when_present = present_prevalence >= absent_prevalence
            scenario_results = {}
            for scenario in split["validate"]:
                report = protocol_reports[scenario]
                counts = report["protocol_indicators"][name]
                present = _confusion(
                    int(counts["present_normal"]),
                    int(counts["present_attack"]),
                    positive_when_present,
                )
                absent = _confusion(
                    int(report["totals"]["normal_packets"] - counts["present_normal"]),
                    int(report["totals"]["attack_packets"] - counts["present_attack"]),
                    not positive_when_present,
                )
                scenario_results[scenario] = _merge_confusions([present, absent])
            fold_result[name] = {
                "positive_when": "present" if positive_when_present else "absent",
                "training_present_attack_prevalence": present_prevalence,
                "training_absent_attack_prevalence": absent_prevalence,
                "validation_scenarios": scenario_results,
                "validation_pooled_diagnostic": _merge_confusions(scenario_results.values()),
            }
        result["fold_trained_rules"][fold] = fold_result
    return result


def _shortcut_report(
    manifest: dict,
    scenarios: list[str],
    scenario_dirs: dict[str, Path],
    protocol_indicators: list[str],
) -> dict:
    key_frames = {
        scenario: pd.read_parquet(directory / "shortcut_keys.parquet")
        for scenario, directory in scenario_dirs.items()
    }
    protocol_reports = {
        scenario: json.loads((directory / "protocol_counts.json").read_text(encoding="utf-8"))
        for scenario, directory in scenario_dirs.items()
    }
    result = {
        "endpoint_values_persisted": False,
        "hash_algorithm": "sha256",
        "within_scenario": {},
        "fold_memorisation_rules": {},
        "protocol_rules": _protocol_rule_report(
            manifest, scenarios, protocol_reports, protocol_indicators
        ),
    }
    for scenario, frame in key_frames.items():
        endpoints = frame.loc[frame["key_type"].eq("endpoint")]
        pairs = frame.loc[frame["key_type"].eq("directed_pair")]
        result["within_scenario"][scenario] = {
            "distinct_endpoints": int(len(endpoints)),
            "endpoints_seen_with_attack": int(endpoints["attack_packets"].gt(0).sum()),
            "endpoints_attack_only": int(
                (endpoints["attack_packets"].gt(0) & endpoints["normal_packets"].eq(0)).sum()
            ),
            "distinct_directed_pairs": int(len(pairs)),
            "directed_pairs_seen_with_attack": int(pairs["attack_packets"].gt(0).sum()),
            "directed_pairs_attack_only": int(
                (pairs["attack_packets"].gt(0) & pairs["normal_packets"].eq(0)).sum()
            ),
        }
    if set(scenarios) != set(selected_scenarios(manifest, "FULL_DEV")):
        return result
    for fold, split in manifest["validation"]["folds"].items():
        train_frames = [key_frames[name] for name in split["train"]]
        endpoint_sets = _key_sets(train_frames, "endpoint")
        pair_sets = _key_sets(train_frames, "directed_pair")
        per_scenario = {
            scenario: _evaluate_key_rules(
                key_frames[scenario], endpoint_sets, pair_sets
            )
            for scenario in split["validate"]
        }
        rule_names = next(iter(per_scenario.values())).keys()
        result["fold_memorisation_rules"][fold] = {
            "train_scenarios": split["train"],
            "validation_scenarios": per_scenario,
            "pooled_validation_diagnostic": {
                rule: _merge_confusions(
                    per_scenario[scenario][rule] for scenario in split["validate"]
                )
                for rule in rule_names
            },
            "training_key_counts": {
                "endpoints_seen_with_attack": len(endpoint_sets[0]),
                "endpoints_attack_only": len(endpoint_sets[1]),
                "pairs_seen_with_attack": len(pair_sets[0]),
                "pairs_attack_only": len(pair_sets[1]),
            },
        }
    return result


def _write_combined_window_metrics(
    scenario_dirs: dict[str, Path], output_path: Path
) -> None:
    writer = None
    try:
        for directory in scenario_dirs.values():
            table = pq.read_table(directory / "window_metrics.parquet")
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            elif table.schema != writer.schema:
                raise ValueError("Scenario window-metric schemas differ.")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("No scenario metrics were available to combine.")


def _summary_tables(metrics: pd.DataFrame, percentiles: list[int]):
    scenario_rows = []
    percentile_rows = []
    for scenario, frame in metrics.groupby("scenario", sort=False):
        type_counts = frame["window_type"].value_counts()
        topology_frequency = frame["topology_sha256"].value_counts(normalize=True)
        scenario_rows.append(
            {
                "scenario": scenario,
                "nonempty_windows": len(frame),
                "benign_only_windows": int(type_counts.get("benign_only", 0)),
                "mixed_windows": int(type_counts.get("mixed", 0)),
                "attack_only_windows": int(type_counts.get("attack_only", 0)),
                "mixed_window_fraction": _safe_fraction(
                    int(type_counts.get("mixed", 0)), len(frame)
                ),
                "distinct_endpoint_sets": int(frame["endpoint_set_sha256"].nunique()),
                "distinct_topologies": int(frame["topology_sha256"].nunique()),
                "most_common_topology_fraction": float(topology_frequency.iloc[0]),
                "median_nodes": float(frame["nodes"].median()),
                "median_edges": float(frame["edges"].median()),
                "median_adjacent_node_jaccard": float(
                    frame["adjacent_node_jaccard"].median()
                ),
                "median_adjacent_pair_jaccard": float(
                    frame["adjacent_pair_jaccard"].median()
                ),
                "median_largest_component_fraction": float(
                    frame["largest_weak_component_fraction"].median()
                ),
            }
        )
        for metric in SUMMARY_METRICS:
            values = frame[metric].dropna().to_numpy(dtype=np.float64)
            for percentile in percentiles:
                percentile_rows.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "percentile": percentile,
                        "value": float(np.percentile(values, percentile)) if len(values) else np.nan,
                        "observations": int(len(values)),
                    }
                )
    return pd.DataFrame(scenario_rows), pd.DataFrame(percentile_rows)


def _step_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    topology_fields = (
        "nodes",
        "edges",
        "weak_components",
        "largest_weak_component_fraction",
        "simple_directed_density",
        "mean_multiplicity",
        "branching_node_fraction",
    )
    for item in metrics.itertuples(index=False):
        steps = json.loads(item.attack_steps_json)
        if not steps:
            steps = ["__BENIGN__"]
        for step in steps:
            row = {
                "scenario": item.scenario,
                "step": step,
                "window_type": item.window_type,
            }
            row.update({name: getattr(item, name) for name in topology_fields})
            rows.append(row)
    exploded = pd.DataFrame(rows)
    aggregations = {name: ["mean", "median"] for name in topology_fields}
    summary = exploded.groupby(
        ["scenario", "step", "window_type"], dropna=False
    ).agg(aggregations)
    summary.columns = [f"{name}_{statistic}" for name, statistic in summary.columns]
    summary.insert(0, "windows", exploded.groupby(
        ["scenario", "step", "window_type"], dropna=False
    ).size())
    return summary.reset_index()


def _review_markdown(mode: str, scenario_summary: pd.DataFrame, reports: dict) -> str:
    rows = [
        "# cAPTure structural-audit review",
        "",
        f"Mode: `{mode}`.",
        "",
        "This file is a descriptive template. It does not automatically approve the gate.",
        "",
        "## Main signals",
        "",
        "| Scenario | Nonempty windows | Median nodes | Median edges | Adjacent-node Jaccard | Dominant-topology fraction | Mixed-window fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in scenario_summary.itertuples(index=False):
        rows.append(
            f"| `{item.scenario}` | {item.nonempty_windows:,} | {item.median_nodes:.2f} | "
            f"{item.median_edges:.2f} | {item.median_adjacent_node_jaccard:.4f} | "
            f"{item.most_common_topology_fraction:.4f} | {item.mixed_window_fraction:.4f} |"
        )
    rows.extend(
        [
            "",
            "## Resources",
            "",
            "| Scenario | Total time (s) | Packets/s | Sampled peak RSS (GiB) | Estimated model tensors (GiB) |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for scenario, report in reports.items():
        peak = report["peak_sampled_rss_bytes"]
        peak_gib = peak / 1024**3 if peak is not None else float("nan")
        rows.append(
            f"| `{scenario}` | {report['wall_seconds']:.2f} | "
            f"{report['packets_per_wall_second']:.2f} | {peak_gib:.3f} | "
            f"{report['estimated_model_tensor_bytes'] / 1024**3:.3f} |"
        )
    rows.extend(
        [
            "",
            "## Manual review questions",
            "",
            "- [ ] Do nodes and pairs change materially between adjacent windows?",
            "- [ ] Do components contain shared neighborhoods, or do isolated pairs dominate?",
            "- [ ] Do step-level differences survive scenario and protocol breakdowns?",
            "- [ ] Do hashed endpoint/pair or protocol rules explain much of the attack traffic?",
            "- [ ] Does construction fit the time, RAM, and storage budgets?",
            "- [ ] Is the one-seed model comparison approved?",
            "",
            "Do not access Test1 or Test2 to resolve this review.",
            "",
        ]
    )
    return "\n".join(rows)


def run_capture_graph_structural_audit(
    *,
    manifest_path: Path,
    packet_schema_path: Path,
    structural_config_path: Path,
    prepared_run_dir: Path,
    output_dir: Path,
    local_work_root: Path,
    mode: str,
    batch_size: int = 250_000,
) -> dict:
    """Run or resume the Stage-1 structural audit on SMOKE or FULL_DEV."""
    if mode not in {"SMOKE", "FULL_DEV"} or batch_size <= 0:
        raise ValueError("mode must be SMOKE or FULL_DEV and batch_size must be positive.")
    manifest_path = Path(manifest_path)
    packet_schema_path = Path(packet_schema_path)
    structural_config_path = Path(structural_config_path)
    prepared_run_dir = Path(prepared_run_dir)
    output_dir = Path(output_dir)
    local_work_root = Path(local_work_root)
    config = load_structural_audit_config(structural_config_path)
    manifest, _, prepared_reports, packet_paths = load_prepared_full_dev(
        prepared_run_dir, manifest_path, packet_schema_path
    )
    if manifest["windows"]["selected_duration_seconds"] != config["windows"]["duration_seconds"]:
        raise ValueError("Manifest and structural-audit window widths differ.")
    if manifest["windows"]["origin_rule"] != config["windows"]["origin_rule"]:
        raise ValueError("Manifest and structural-audit origins differ.")
    scenarios = selected_scenarios(manifest, mode)
    config_sha256 = sha256_file(structural_config_path)
    code_sha256 = sha256_file(Path(__file__))
    run_config = {
        "report_version": REPORT_VERSION,
        "mode": mode,
        "scenarios": scenarios,
        "batch_size": int(batch_size),
        "manifest_sha256": sha256_file(manifest_path),
        "packet_schema_sha256": sha256_file(packet_schema_path),
        "structural_config_sha256": config_sha256,
        "prepared_run_config_sha256": sha256_file(prepared_run_dir / "run_config.json"),
        "prepared_packet_sha256": {
            scenario: prepared_reports[scenario]["output_sha256"] for scenario in scenarios
        },
        "code_sha256": code_sha256,
    }
    if output_dir.exists():
        config_path = output_dir / "run_config.json"
        if not config_path.is_file() or json.loads(config_path.read_text(encoding="utf-8")) != run_config:
            raise FileExistsError(
                "The output directory exists with a different or incomplete run contract. "
                "Use a new run ID."
            )
    else:
        output_dir.mkdir(parents=True)
        write_json(output_dir / "run_config.json", run_config)
        for path in (manifest_path, packet_schema_path, structural_config_path):
            shutil.copyfile(path, output_dir / path.name)
    local_work_root.mkdir(parents=True, exist_ok=True)

    reports = {}
    scenario_dirs = {}
    for scenario in scenarios:
        durable_dir = output_dir / "scenarios" / scenario
        scenario_dirs[scenario] = durable_dir
        source_hash = prepared_reports[scenario]["output_sha256"]
        if durable_dir.exists():
            print(f"Validating completed scenario {scenario}...", flush=True)
            reports[scenario] = _validate_scenario_artifacts(
                durable_dir, scenario, source_hash, config_sha256
            )
            continue
        source_path = packet_paths[scenario]
        required_bytes = source_path.stat().st_size + 1024**3
        if shutil.disk_usage(local_work_root).free < required_bytes:
            raise OSError(
                f"Insufficient local space to stage {scenario}: need source size plus 1 GiB."
            )
        print(f"Staging and auditing {scenario}...", flush=True)
        with tempfile.TemporaryDirectory(
            dir=local_work_root, prefix=f"graph_audit_{scenario}_"
        ) as temporary:
            temporary_path = Path(temporary)
            local_source = temporary_path / source_path.name
            shutil.copyfile(source_path, local_source)
            if sha256_file(local_source) != source_hash:
                raise IOError(f"Staged prepared artifact checksum mismatch: {scenario}")
            local_output = temporary_path / "output"
            report = _write_scenario_artifacts(
                scenario=scenario,
                benign_source=manifest["scenarios"][scenario]["benign_source"],
                packet_path=local_source,
                prepared_report=prepared_reports[scenario],
                output_dir=local_output,
                config=config,
                config_sha256=config_sha256,
                batch_size=batch_size,
            )
            durable_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(local_output, durable_dir)
            reports[scenario] = _validate_scenario_artifacts(
                durable_dir, scenario, source_hash, config_sha256
            )
            if reports[scenario] != report:
                raise IOError(f"Persisted scenario report changed during copy: {scenario}")
        print(
            f"Completed {scenario}: {reports[scenario]['nonempty_windows']:,} windows, "
            f"{reports[scenario]['rows']:,} packets",
            flush=True,
        )

    with tempfile.TemporaryDirectory(
        dir=local_work_root, prefix="graph_audit_summary_"
    ) as temporary:
        stage = Path(temporary)
        combined_path = stage / "window_metrics.parquet"
        _write_combined_window_metrics(scenario_dirs, combined_path)
        metrics = pd.read_parquet(combined_path)
        scenario_summary, percentile_summary = _summary_tables(
            metrics, list(config["percentiles"])
        )
        step_summary = _step_summary(metrics)
        scenario_summary.to_csv(stage / "scenario_summary.csv", index=False)
        percentile_summary.to_csv(stage / "percentile_summary.csv", index=False)
        step_summary.to_csv(stage / "step_topology_summary.csv", index=False)
        shortcut = _shortcut_report(
            manifest, scenarios, scenario_dirs, list(config["protocol_indicators"])
        )
        write_json(stage / "shortcut_audit.json", shortcut)
        construction = {
            "report_version": REPORT_VERSION,
            "mode": mode,
            "scenarios": reports,
            "total_packets": sum(report["rows"] for report in reports.values()),
            "total_nonempty_windows": sum(
                report["nonempty_windows"] for report in reports.values()
            ),
            "total_empty_windows": sum(report["empty_windows"] for report in reports.values()),
            "row_conservation_passed": all(
                report["rows"] == prepared_reports[name]["counts"]["packets"]
                for name, report in reports.items()
            ),
        }
        resources = {
            "scenarios": {
                name: {
                    key: report[key]
                    for key in (
                        "wall_seconds",
                        "construction_seconds",
                        "packets_per_wall_second",
                        "peak_sampled_rss_bytes",
                        "window_metrics_bytes",
                        "shortcut_keys_bytes",
                        "estimated_topology_tensor_bytes",
                        "estimated_model_tensor_bytes",
                    )
                }
                for name, report in reports.items()
            },
            "totals": {
                "wall_seconds": sum(report["wall_seconds"] for report in reports.values()),
                "construction_seconds": sum(
                    report["construction_seconds"] for report in reports.values()
                ),
                "estimated_topology_tensor_bytes": sum(
                    report["estimated_topology_tensor_bytes"] for report in reports.values()
                ),
                "estimated_model_tensor_bytes": sum(
                    report["estimated_model_tensor_bytes"] for report in reports.values()
                ),
            },
        }
        repository = manifest_path.resolve().parent.parent
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
        tree = subprocess.run(
            ["git", "status", "--short"],
            cwd=repository,
            capture_output=True,
            text=True,
            check=False,
        )
        provenance = {
            **run_config,
            "git_commit": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
            "working_tree_status": tree.stdout.strip() if tree.returncode == 0 else "unavailable",
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "pyarrow": pa.__version__,
            },
            "held_out_scenarios_accessed": False,
            "training_performed": False,
        }
        write_json(stage / "construction_summary.json", construction)
        write_json(stage / "resource_summary.json", resources)
        write_json(stage / "provenance.json", provenance)
        (stage / "review.md").write_text(
            _review_markdown(mode, scenario_summary, reports), encoding="utf-8"
        )
        final_names = (
            "window_metrics.parquet",
            "scenario_summary.csv",
            "percentile_summary.csv",
            "step_topology_summary.csv",
            "shortcut_audit.json",
            "construction_summary.json",
            "resource_summary.json",
            "provenance.json",
            "review.md",
        )
        for name in final_names:
            shutil.copyfile(stage / name, output_dir / name)
    final_checksums = {
        name: sha256_file(output_dir / name)
        for name in (
            "window_metrics.parquet",
            "scenario_summary.csv",
            "percentile_summary.csv",
            "step_topology_summary.csv",
            "shortcut_audit.json",
            "construction_summary.json",
            "resource_summary.json",
            "provenance.json",
            "review.md",
        )
    }
    write_json(output_dir / "artifact_checksums.json", final_checksums)
    write_json(
        output_dir / "run_status.json",
        {
            "complete": True,
            "mode": mode,
            "artifact_checksums_sha256": sha256_file(output_dir / "artifact_checksums.json"),
        },
    )
    print(f"Structural audit complete: {output_dir}", flush=True)
    return {
        "output_dir": str(output_dir),
        "mode": mode,
        "scenario_summary": scenario_summary.to_dict(orient="records"),
        "construction_summary": construction,
        "resource_summary": resources,
    }
