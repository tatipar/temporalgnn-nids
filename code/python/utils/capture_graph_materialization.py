"""Materialize fold-specific, model-ready cAPTure graph shards.

The materializer is development-only. It stores many graph windows in each
compressed NumPy shard to avoid tens of thousands of small Drive files. Graph
topology and labels are identical across folds; edge features are transformed
with the current fold's already audited training-only preprocessor.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from .capture_data import selected_scenarios, sha256_file, write_json
from .capture_feature_profile import (
    load_prepared_full_dev,
    load_preprocessing_schema,
    preprocessing_schema_sha256,
)
from .capture_xgb_p import (
    NANOSECONDS_PER_SECOND,
    _load_fold_preprocessor,
    window_coordinates,
)


REPORT_VERSION = 1
METADATA_COLUMNS = (
    "source_row_id",
    "packet_timestamp_ns",
    "src_endpoint",
    "dst_endpoint",
    "binary_label",
)
SHARD_ARRAYS = (
    "edge_index",
    "edge_attr",
    "y",
    "source_row_id",
    "global_node_ids",
    "edge_ptr",
    "node_ptr",
    "window_index",
    "window_start_ns",
    "window_end_ns",
    "decision_time_ns",
)


def load_graph_materialization_config(path: Path) -> dict:
    """Load and strictly validate the frozen graph-shard contract."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if (
        config.get("materialization_version") != 1
        or config.get("scope") != "development_only"
        or config.get("stage") != "model_ready_graph_materialization"
    ):
        raise ValueError("Unsupported graph-materialization configuration.")
    required_windows = {
        "duration_seconds": 5,
        "type": "fixed_non_overlapping",
        "interval": "left_closed_right_open",
        "origin_rule": "first_packet_timestamp_per_scenario",
        "decision_time": "window_end",
        "empty_windows": "do_not_materialize_preserve_index_gaps",
    }
    for name, expected in required_windows.items():
        if config.get("windows", {}).get(name) != expected:
            raise ValueError(f"Graph-materialization window contract changed: {name}")
    required_graph = {
        "type": "directed_temporal_multigraph",
        "node_key": "normalized_ethernet_mac_address",
        "edge_unit": "packet",
        "preserve_parallel_edges": True,
        "preserve_self_loops": True,
        "persist_node_features_x": False,
        "local_node_order": "ascending_scenario_global_node_id",
        "global_node_id_scope": "scenario",
        "global_node_id_assignment": "deterministic_first_packet_appearance",
        "raw_endpoint_identity_in_artifacts": False,
    }
    for name, expected in required_graph.items():
        if config.get("graph", {}).get(name) != expected:
            raise ValueError(f"Graph-materialization graph contract changed: {name}")
    features = config.get("features", {})
    if (
        features.get("view") != "fold_preprocessed_packet_features"
        or features.get("dimension") != 103
        or features.get("dtype") != "float32"
        or features.get("fit_scope") != "current_fold_training_scenarios_only"
        or features.get("materialize_all_development_scenarios_for_each_fold") is not True
    ):
        raise ValueError("Unsupported graph edge-feature contract.")
    if config.get("targets") != {
        "unit": "edge",
        "field": "binary_label",
        "dtype": "uint8",
    }:
        raise ValueError("Unsupported graph target contract.")
    storage = config.get("storage", {})
    required_storage = {
        "format": "compressed_numpy_npz_shards",
        "compression": "zip_deflate",
        "edge_index_dtype": "int32",
        "global_node_id_dtype": "int32",
        "pointer_dtype": "int64",
        "source_row_id_dtype": "int64",
        "window_time_dtype": "int64",
        "write_one_file_per_window": False,
    }
    for name, expected in required_storage.items():
        if storage.get(name) != expected:
            raise ValueError(f"Graph-materialization storage contract changed: {name}")
    for name in ("maximum_edges_per_shard", "maximum_graphs_per_shard"):
        if not isinstance(storage.get(name), int) or storage[name] <= 0:
            raise ValueError(f"Invalid graph-shard limit: {name}")
    required_alignment = {
        "preserve_packet_order_within_scenario": True,
        "persist_source_row_id_per_edge": True,
        "require_one_edge_per_prepared_packet": True,
        "require_identical_topology_across_folds": True,
        "require_identical_labels_across_folds": True,
        "require_identical_node_mapping_across_folds": True,
    }
    if config.get("alignment") != required_alignment:
        raise ValueError("The graph-alignment contract changed.")
    return config


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _update_array_digest(digest, array: np.ndarray) -> None:
    contiguous = np.ascontiguousarray(array)
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
    digest.update(contiguous.tobytes())


def _rss_bytes() -> int | None:
    try:
        status = Path("/proc/self/status").read_text(encoding="utf-8")
        for line in status.splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return None
    return None


def _iter_transformed_windows(
    path: Path,
    preprocessor,
    *,
    origin_ns: int,
    width_seconds: int,
    batch_size: int,
):
    """Yield complete window metadata and transformed features across batch cuts."""
    columns = list(dict.fromkeys([*preprocessor.required_columns, *METADATA_COLUMNS]))
    available = set(pq.read_schema(path).names)
    missing = sorted(set(columns) - available)
    if missing:
        raise ValueError(f"Prepared packet artifact is missing graph columns: {missing}")
    parquet = pq.ParquetFile(path)
    expected_row_id = 0
    previous_timestamp = None
    active_index = None
    active_frames: list[pd.DataFrame] = []
    active_features: list[np.ndarray] = []
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        if frame.empty:
            continue
        row_ids = frame["source_row_id"].to_numpy(dtype=np.int64)
        expected = np.arange(expected_row_id, expected_row_id + len(frame), dtype=np.int64)
        if not np.array_equal(row_ids, expected):
            raise ValueError("Prepared packet order changed during graph materialization.")
        expected_row_id += len(frame)
        timestamps = frame["packet_timestamp_ns"].to_numpy(dtype=np.int64)
        if np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError("Packet timestamps are not ordered inside a source batch.")
        if previous_timestamp is not None and int(timestamps[0]) < previous_timestamp:
            raise ValueError("Packet timestamps decreased across source batches.")
        previous_timestamp = int(timestamps[-1])
        features = preprocessor.transform(frame).to_numpy(dtype=np.float32, copy=False)
        if features.shape != (len(frame), len(preprocessor.feature_names)):
            raise ValueError("Fold-preprocessed edge features have an unexpected shape.")
        if not np.isfinite(features).all():
            raise ValueError("Fold-preprocessed edge features contain non-finite values.")
        indexes, _ = window_coordinates(timestamps, origin_ns, width_seconds)
        boundaries = np.flatnonzero(indexes[1:] != indexes[:-1]) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [len(frame)]))
        for start, end in zip(starts, ends):
            index = int(indexes[start])
            if active_index is None:
                active_index = index
            if index < active_index:
                raise ValueError("Window indexes are not chronological.")
            if index != active_index:
                yield (
                    active_index,
                    pd.concat(active_frames, ignore_index=True),
                    np.concatenate(active_features, axis=0),
                )
                active_index, active_frames, active_features = index, [], []
            active_frames.append(
                frame.iloc[start:end].loc[:, list(METADATA_COLUMNS)].copy()
            )
            active_features.append(np.ascontiguousarray(features[start:end]))
    if active_index is not None:
        yield (
            active_index,
            pd.concat(active_frames, ignore_index=True),
            np.concatenate(active_features, axis=0),
        )


class _ShardAccumulator:
    def __init__(
        self,
        output_dir: Path,
        *,
        feature_dim: int,
        maximum_edges: int,
        maximum_graphs: int,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.feature_dim = int(feature_dim)
        self.maximum_edges = int(maximum_edges)
        self.maximum_graphs = int(maximum_graphs)
        self.shard_index = 0
        self.records: list[dict] = []
        self.clear()

    def clear(self) -> None:
        self.edge_indexes: list[np.ndarray] = []
        self.edge_attrs: list[np.ndarray] = []
        self.targets: list[np.ndarray] = []
        self.source_row_ids: list[np.ndarray] = []
        self.global_node_ids: list[np.ndarray] = []
        self.window_indexes: list[int] = []
        self.window_starts: list[int] = []
        self.window_ends: list[int] = []
        self.edge_count = 0
        self.node_count = 0

    @property
    def graph_count(self) -> int:
        return len(self.window_indexes)

    def should_flush_before(self, next_edges: int) -> bool:
        if not self.graph_count:
            return False
        return (
            self.graph_count >= self.maximum_graphs
            or self.edge_count + int(next_edges) > self.maximum_edges
        )

    def add(
        self,
        *,
        edge_index: np.ndarray,
        edge_attr: np.ndarray,
        y: np.ndarray,
        source_row_id: np.ndarray,
        global_node_ids: np.ndarray,
        window_index: int,
        window_start_ns: int,
        window_end_ns: int,
    ) -> None:
        edge_count = int(edge_attr.shape[0])
        node_count = int(global_node_ids.shape[0])
        if (
            edge_index.shape != (2, edge_count)
            or edge_attr.shape != (edge_count, self.feature_dim)
            or y.shape != (edge_count,)
            or source_row_id.shape != (edge_count,)
            or node_count <= 0
            or edge_count <= 0
        ):
            raise ValueError("A graph has inconsistent model-ready tensor shapes.")
        if edge_index.min() < 0 or edge_index.max() >= node_count:
            raise ValueError("A local edge index is outside the graph node range.")
        self.edge_indexes.append(np.ascontiguousarray(edge_index, dtype=np.int32))
        self.edge_attrs.append(np.ascontiguousarray(edge_attr, dtype=np.float32))
        self.targets.append(np.ascontiguousarray(y, dtype=np.uint8))
        self.source_row_ids.append(np.ascontiguousarray(source_row_id, dtype=np.int64))
        self.global_node_ids.append(np.ascontiguousarray(global_node_ids, dtype=np.int32))
        self.window_indexes.append(int(window_index))
        self.window_starts.append(int(window_start_ns))
        self.window_ends.append(int(window_end_ns))
        self.edge_count += edge_count
        self.node_count += node_count

    def flush(self) -> dict | None:
        if not self.graph_count:
            return None
        edge_ptr = np.zeros(self.graph_count + 1, dtype=np.int64)
        edge_ptr[1:] = np.cumsum([array.shape[1] for array in self.edge_indexes])
        node_ptr = np.zeros(self.graph_count + 1, dtype=np.int64)
        node_ptr[1:] = np.cumsum([array.shape[0] for array in self.global_node_ids])
        arrays = {
            "edge_index": np.concatenate(self.edge_indexes, axis=1).astype(np.int32, copy=False),
            "edge_attr": np.concatenate(self.edge_attrs, axis=0).astype(np.float32, copy=False),
            "y": np.concatenate(self.targets).astype(np.uint8, copy=False),
            "source_row_id": np.concatenate(self.source_row_ids).astype(np.int64, copy=False),
            "global_node_ids": np.concatenate(self.global_node_ids).astype(np.int32, copy=False),
            "edge_ptr": edge_ptr,
            "node_ptr": node_ptr,
            "window_index": np.asarray(self.window_indexes, dtype=np.int64),
            "window_start_ns": np.asarray(self.window_starts, dtype=np.int64),
            "window_end_ns": np.asarray(self.window_ends, dtype=np.int64),
            "decision_time_ns": np.asarray(self.window_ends, dtype=np.int64),
        }
        name = f"graph_shard_{self.shard_index:05d}.npz"
        path = self.output_dir / name
        np.savez_compressed(path, **arrays)
        record = {
            "path": name,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "graphs": self.graph_count,
            "edges": self.edge_count,
            "node_appearances": self.node_count,
            "first_window_index": int(self.window_indexes[0]),
            "last_window_index": int(self.window_indexes[-1]),
            "first_source_row_id": int(arrays["source_row_id"][0]),
            "last_source_row_id": int(arrays["source_row_id"][-1]),
            "oversized_single_graph": bool(
                self.graph_count == 1 and self.edge_count > self.maximum_edges
            ),
        }
        self.records.append(record)
        self.shard_index += 1
        self.clear()
        return record


def _validate_shard(path: Path, record: dict, feature_dim: int) -> None:
    if sha256_file(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
        raise ValueError(f"Graph shard checksum or size mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as shard:
        if set(shard.files) != set(SHARD_ARRAYS):
            raise ValueError(f"Graph shard arrays differ from the schema: {path.name}")
        graphs = len(shard["window_index"])
        edges = len(shard["y"])
        nodes = len(shard["global_node_ids"])
        if (
            graphs != record["graphs"]
            or edges != record["edges"]
            or nodes != record["node_appearances"]
            or shard["edge_index"].shape != (2, edges)
            or shard["edge_attr"].shape != (edges, feature_dim)
            or shard["source_row_id"].shape != (edges,)
            or shard["edge_ptr"].shape != (graphs + 1,)
            or shard["node_ptr"].shape != (graphs + 1,)
        ):
            raise ValueError(f"Graph shard shape mismatch: {path.name}")
        if (
            shard["edge_index"].dtype != np.int32
            or shard["edge_attr"].dtype != np.float32
            or shard["y"].dtype != np.uint8
            or shard["global_node_ids"].dtype != np.int32
        ):
            raise ValueError(f"Graph shard dtype mismatch: {path.name}")
        if not np.array_equal(shard["decision_time_ns"], shard["window_end_ns"]):
            raise ValueError(f"Graph shard decision times differ from window closes: {path.name}")
        if (
            shard["edge_ptr"][0] != 0
            or shard["edge_ptr"][-1] != edges
            or shard["node_ptr"][0] != 0
            or shard["node_ptr"][-1] != nodes
        ):
            raise ValueError(f"Graph shard pointers do not span their arrays: {path.name}")
        if not np.all(shard["edge_ptr"][1:] > shard["edge_ptr"][:-1]):
            raise ValueError(f"Graph shard contains an empty graph: {path.name}")
        if not np.all(shard["node_ptr"][1:] > shard["node_ptr"][:-1]):
            raise ValueError(f"Graph shard contains a graph without nodes: {path.name}")


def _materialize_scenario_fold(
    *,
    scenario: str,
    fold: str,
    packet_path: Path,
    prepared_report: dict,
    preprocessor,
    preprocessor_sha256: str,
    preprocessing_contract_sha256: str,
    output_dir: Path,
    config: dict,
    config_sha256: str,
    batch_size: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=False)
    width_seconds = int(config["windows"]["duration_seconds"])
    width_ns = width_seconds * NANOSECONDS_PER_SECOND
    origin_ns = int(prepared_report["scenario_origin_timestamp_ns"])
    expected_rows = int(prepared_report["counts"]["packets"])
    storage = config["storage"]
    accumulator = _ShardAccumulator(
        output_dir,
        feature_dim=len(preprocessor.feature_names),
        maximum_edges=int(storage["maximum_edges_per_shard"]),
        maximum_graphs=int(storage["maximum_graphs_per_shard"]),
    )
    endpoint_to_global: dict[str, int] = {}
    endpoint_first_row: dict[str, int] = {}
    topology_digest = hashlib.sha256()
    target_digest = hashlib.sha256()
    feature_digest = hashlib.sha256()
    row_digest = hashlib.sha256()
    graph_count = edge_count = attack_edges = 0
    node_appearances = 0
    previous_window_index = None
    peak_rss = _rss_bytes()
    started = time.perf_counter()

    for window_index, frame, edge_attr in _iter_transformed_windows(
        packet_path,
        preprocessor,
        origin_ns=origin_ns,
        width_seconds=width_seconds,
        batch_size=batch_size,
    ):
        if previous_window_index is not None and window_index <= previous_window_index:
            raise ValueError("Materialized graph windows are not strictly chronological.")
        previous_window_index = window_index
        raw_labels = frame["binary_label"].to_numpy()
        if not np.isin(raw_labels, [0, 1]).all():
            raise ValueError(f"Invalid edge targets in {scenario}.")
        labels = raw_labels.astype(np.uint8, copy=False)
        if frame[["src_endpoint", "dst_endpoint"]].isna().any().any():
            raise ValueError(f"Null graph endpoints remain in {scenario}.")
        source_rows = frame["source_row_id"].to_numpy(dtype=np.int64)
        sources = frame["src_endpoint"].astype(str).to_numpy()
        destinations = frame["dst_endpoint"].astype(str).to_numpy()
        source_global = np.empty(len(frame), dtype=np.int32)
        destination_global = np.empty(len(frame), dtype=np.int32)
        for index, (source, destination, source_row_id) in enumerate(
            zip(sources, destinations, source_rows)
        ):
            if source not in endpoint_to_global:
                endpoint_to_global[source] = len(endpoint_to_global)
                endpoint_first_row[source] = int(source_row_id)
            if destination not in endpoint_to_global:
                endpoint_to_global[destination] = len(endpoint_to_global)
                endpoint_first_row[destination] = int(source_row_id)
            source_global[index] = endpoint_to_global[source]
            destination_global[index] = endpoint_to_global[destination]
        global_node_ids = np.unique(
            np.concatenate([source_global, destination_global])
        ).astype(np.int32, copy=False)
        local_sources = np.searchsorted(global_node_ids, source_global).astype(np.int32)
        local_destinations = np.searchsorted(global_node_ids, destination_global).astype(np.int32)
        edge_index = np.stack([local_sources, local_destinations])
        window_start_ns = origin_ns + int(window_index) * width_ns
        window_end_ns = window_start_ns + width_ns

        if accumulator.should_flush_before(len(frame)):
            accumulator.flush()
        accumulator.add(
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=labels,
            source_row_id=source_rows,
            global_node_ids=global_node_ids,
            window_index=window_index,
            window_start_ns=window_start_ns,
            window_end_ns=window_end_ns,
        )
        topology_digest.update(np.asarray([window_index], dtype=np.int64).tobytes())
        _update_array_digest(topology_digest, edge_index)
        _update_array_digest(topology_digest, global_node_ids)
        _update_array_digest(target_digest, labels)
        _update_array_digest(feature_digest, edge_attr)
        _update_array_digest(row_digest, source_rows)
        graph_count += 1
        edge_count += len(frame)
        attack_edges += int(labels.sum())
        node_appearances += len(global_node_ids)
        rss = _rss_bytes()
        if rss is not None:
            peak_rss = rss if peak_rss is None else max(peak_rss, rss)
        if graph_count % 2500 == 0:
            print(
                f"fold {fold}/{scenario}: {graph_count:,} graphs, "
                f"{edge_count:,} edges, {len(accumulator.records):,} completed shards",
                flush=True,
            )
    accumulator.flush()
    if edge_count != expected_rows or not graph_count:
        raise ValueError(
            f"Packet-edge conservation failed for fold {fold}/{scenario}: "
            f"edges={edge_count}, expected={expected_rows}."
        )
    if attack_edges != int(prepared_report["counts"]["attack_packets"]):
        raise ValueError(f"Attack-edge conservation failed for fold {fold}/{scenario}.")
    mapping_contract_rows = [
        {
            "global_node_id": int(global_id),
            "endpoint": endpoint,
            "first_seen_source_row_id": int(endpoint_first_row[endpoint]),
        }
        for endpoint, global_id in sorted(endpoint_to_global.items(), key=lambda item: item[1])
    ]
    mapping_rows = [
        {
            "global_node_id": row["global_node_id"],
            "first_seen_source_row_id": row["first_seen_source_row_id"],
        }
        for row in mapping_contract_rows
    ]
    mapping_path = output_dir / "node_mapping.parquet"
    pd.DataFrame(mapping_rows).to_parquet(mapping_path, index=False, compression="zstd")
    mapping_sha256 = sha256_file(mapping_path)
    mapping_contract_sha256 = _canonical_hash(mapping_contract_rows)
    elapsed = time.perf_counter() - started
    report = {
        "report_version": REPORT_VERSION,
        "scenario": scenario,
        "fold": fold,
        "source_artifact_sha256": prepared_report["output_sha256"],
        "materialization_config_sha256": config_sha256,
        "preprocessor_sha256": preprocessor_sha256,
        "preprocessing_contract_sha256": preprocessing_contract_sha256,
        "feature_names": list(preprocessor.feature_names),
        "feature_names_sha256": _canonical_hash(list(preprocessor.feature_names)),
        "feature_dim": len(preprocessor.feature_names),
        "origin_ns": origin_ns,
        "window_width_seconds": width_seconds,
        "graphs": graph_count,
        "edges": edge_count,
        "normal_edges": edge_count - attack_edges,
        "attack_edges": attack_edges,
        "distinct_global_nodes": len(endpoint_to_global),
        "node_appearances": node_appearances,
        "first_window_index": int(accumulator.records[0]["first_window_index"]),
        "last_window_index": int(accumulator.records[-1]["last_window_index"]),
        "empty_window_gaps_preserved": True,
        "topology_sha256": topology_digest.hexdigest(),
        "targets_sha256": target_digest.hexdigest(),
        "features_sha256": feature_digest.hexdigest(),
        "source_row_ids_sha256": row_digest.hexdigest(),
        "node_mapping_artifact": mapping_path.name,
        "node_mapping_sha256": mapping_sha256,
        "node_mapping_contract_sha256": mapping_contract_sha256,
        "shards": accumulator.records,
        "shard_count": len(accumulator.records),
        "oversized_single_graph_shards": sum(
            bool(record["oversized_single_graph"]) for record in accumulator.records
        ),
        "compressed_shard_bytes": sum(record["bytes"] for record in accumulator.records),
        "wall_seconds": float(elapsed),
        "edges_per_second": edge_count / elapsed,
        "peak_sampled_rss_bytes": peak_rss,
        "raw_endpoint_identifiers_persisted": False,
        "node_features_x_persisted": False,
    }
    write_json(output_dir / "scenario_fold_report.json", report)
    checksums = {
        "scenario_fold_report.json": sha256_file(output_dir / "scenario_fold_report.json"),
        "node_mapping.parquet": mapping_sha256,
        **{record["path"]: record["sha256"] for record in accumulator.records},
    }
    write_json(output_dir / "artifact_checksums.json", checksums)
    write_json(
        output_dir / "run_status.json",
        {
            "complete": True,
            "scenario": scenario,
            "fold": fold,
            "artifact_checksums_sha256": sha256_file(output_dir / "artifact_checksums.json"),
        },
    )
    return report


def _validate_scenario_fold(
    directory: Path,
    *,
    scenario: str,
    fold: str,
    source_sha256: str,
    config_sha256: str,
    preprocessor_sha256: str,
) -> dict:
    required = (
        "scenario_fold_report.json",
        "node_mapping.parquet",
        "artifact_checksums.json",
        "run_status.json",
    )
    if not all((directory / name).is_file() for name in required):
        raise FileNotFoundError(f"Incomplete graph materialization: fold {fold}/{scenario}")
    report = json.loads((directory / "scenario_fold_report.json").read_text(encoding="utf-8"))
    status = json.loads((directory / "run_status.json").read_text(encoding="utf-8"))
    checksums = json.loads((directory / "artifact_checksums.json").read_text(encoding="utf-8"))
    if (
        status.get("complete") is not True
        or status.get("scenario") != scenario
        or status.get("fold") != fold
        or status.get("artifact_checksums_sha256")
        != sha256_file(directory / "artifact_checksums.json")
    ):
        raise ValueError(f"Invalid graph-materialization status: fold {fold}/{scenario}")
    if (
        report.get("scenario") != scenario
        or report.get("fold") != fold
        or report.get("source_artifact_sha256") != source_sha256
        or report.get("materialization_config_sha256") != config_sha256
        or report.get("preprocessor_sha256") != preprocessor_sha256
    ):
        raise ValueError(f"Graph-materialization provenance changed: fold {fold}/{scenario}")
    for name, expected in checksums.items():
        path = directory / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Graph-materialization checksum mismatch: {fold}/{scenario}/{name}")
    for record in report["shards"]:
        _validate_shard(directory / record["path"], record, int(report["feature_dim"]))
    return report


def _cross_fold_invariants(reports: dict[str, dict[str, dict]], scenarios: list[str]) -> dict:
    if set(reports) != {"A", "B"}:
        raise ValueError("Graph materialization requires exactly folds A and B.")
    comparisons = {}
    for scenario in scenarios:
        left, right = reports["A"][scenario], reports["B"][scenario]
        invariant_fields = (
            "graphs",
            "edges",
            "normal_edges",
            "attack_edges",
            "distinct_global_nodes",
            "node_appearances",
            "first_window_index",
            "last_window_index",
            "topology_sha256",
            "targets_sha256",
            "source_row_ids_sha256",
            "node_mapping_contract_sha256",
            "feature_names_sha256",
        )
        differences = {
            name: {"A": left[name], "B": right[name]}
            for name in invariant_fields
            if left[name] != right[name]
        }
        if differences:
            raise ValueError(
                f"Fold-specific graph artifacts changed topology or labels for {scenario}: "
                f"{sorted(differences)}"
            )
        comparisons[scenario] = {
            "topology_and_labels_identical": True,
            "feature_tensors_differ": left["features_sha256"] != right["features_sha256"],
            "graphs": left["graphs"],
            "edges": left["edges"],
            "nodes": left["distinct_global_nodes"],
        }
    return comparisons


def run_capture_graph_materialization(
    *,
    manifest_path: Path,
    packet_schema_path: Path,
    preprocessing_schema_path: Path,
    materialization_config_path: Path,
    prepared_run_dir: Path,
    preprocessing_audit_dir: Path,
    stage1_decision_path: Path,
    output_dir: Path,
    local_work_root: Path,
    mode: str,
    batch_size: int = 100_000,
) -> dict:
    """Run or resume model-ready graph materialization for both development folds."""
    if mode not in {"SMOKE", "FULL_DEV"} or batch_size <= 0:
        raise ValueError("mode must be SMOKE or FULL_DEV and batch_size must be positive.")
    manifest_path = Path(manifest_path)
    packet_schema_path = Path(packet_schema_path)
    preprocessing_schema_path = Path(preprocessing_schema_path)
    materialization_config_path = Path(materialization_config_path)
    prepared_run_dir = Path(prepared_run_dir)
    preprocessing_audit_dir = Path(preprocessing_audit_dir)
    stage1_decision_path = Path(stage1_decision_path)
    output_dir = Path(output_dir)
    local_work_root = Path(local_work_root)
    if not stage1_decision_path.is_file():
        raise FileNotFoundError("The Stage-1 decision document is required.")
    decision_text = stage1_decision_path.read_text(encoding="utf-8")
    if "Status: **PASS_WITH_LIMITATIONS**." not in decision_text.splitlines():
        raise ValueError("Stage 1 has not been approved as PASS_WITH_LIMITATIONS.")
    config = load_graph_materialization_config(materialization_config_path)
    manifest, packet_schema, prepared_reports, packet_paths = load_prepared_full_dev(
        prepared_run_dir, manifest_path, packet_schema_path
    )
    preprocessing_schema = load_preprocessing_schema(
        preprocessing_schema_path, packet_schema
    )
    manifest_window_contract = {
        "selected_duration_seconds": config["windows"]["duration_seconds"],
        "type": config["windows"]["type"],
        "interval": config["windows"]["interval"],
        "origin_rule": config["windows"]["origin_rule"],
        "decision_time": config["windows"]["decision_time"],
        "empty_window_policy": config["windows"]["empty_windows"],
    }
    for name, expected in manifest_window_contract.items():
        if manifest["windows"].get(name) != expected:
            raise ValueError(f"Manifest and materialization window contracts differ: {name}")
    manifest_graph_contract = {
        "type": config["graph"]["type"],
        "edge_unit": config["graph"]["edge_unit"],
        "preserve_parallel_packet_edges": config["graph"]["preserve_parallel_edges"],
        "edge_target": config["targets"]["field"],
        "timestamp": config["windows"]["decision_time"],
        "node_ids_stable_within_scenario": True,
        "packet_edge_score_correspondence": "one_to_one",
        "endpoint_key": config["graph"]["node_key"],
        "missing_endpoint_policy": "error",
        "non_ip_packet_policy": "preserve_including_arp",
    }
    for name, expected in manifest_graph_contract.items():
        if manifest["graphs"].get(name) != expected:
            raise ValueError(f"Manifest and materialization graph contracts differ: {name}")
    scenarios = selected_scenarios(manifest, mode)
    all_development = selected_scenarios(manifest, "FULL_DEV")
    config_sha256 = sha256_file(materialization_config_path)
    preprocessors = {}
    preprocessor_hashes = {}
    for fold, split in manifest["validation"]["folds"].items():
        preprocessor, preprocessor_hash = _load_fold_preprocessor(
            preprocessing_audit_dir,
            preprocessing_schema,
            fold,
            split["train"],
            prepared_run_dir,
        )
        preprocessors[fold] = preprocessor
        preprocessor_hashes[fold] = preprocessor_hash
    run_config = {
        "report_version": REPORT_VERSION,
        "mode": mode,
        "scenarios": scenarios,
        "all_development_scenarios": all_development,
        "batch_size": int(batch_size),
        "manifest_sha256": sha256_file(manifest_path),
        "packet_schema_sha256": sha256_file(packet_schema_path),
        "preprocessing_schema_sha256": sha256_file(preprocessing_schema_path),
        "preprocessing_contract_sha256": preprocessing_schema_sha256(preprocessing_schema),
        "materialization_config_sha256": config_sha256,
        "prepared_run_config_sha256": sha256_file(prepared_run_dir / "run_config.json"),
        "preprocessing_audit_sha256": sha256_file(
            preprocessing_audit_dir / "capture_preprocessing_audit.json"
        ),
        "stage1_decision_sha256": sha256_file(stage1_decision_path),
        "code_sha256": sha256_file(Path(__file__)),
        "preprocessor_sha256": preprocessor_hashes,
        "prepared_packet_sha256": {
            scenario: prepared_reports[scenario]["output_sha256"] for scenario in scenarios
        },
    }
    if output_dir.exists():
        existing = output_dir / "run_config.json"
        if not existing.is_file() or json.loads(existing.read_text(encoding="utf-8")) != run_config:
            raise FileExistsError(
                "The graph output directory has a different or incomplete contract. "
                "Use a new run ID."
            )
    else:
        output_dir.mkdir(parents=True)
        write_json(output_dir / "run_config.json", run_config)
        for path in (
            manifest_path,
            packet_schema_path,
            preprocessing_schema_path,
            materialization_config_path,
            stage1_decision_path,
        ):
            shutil.copyfile(path, output_dir / path.name)
    local_work_root.mkdir(parents=True, exist_ok=True)
    reports: dict[str, dict[str, dict]] = {"A": {}, "B": {}}
    for fold in ("A", "B"):
        for scenario in scenarios:
            durable_dir = output_dir / f"fold_{fold}" / scenario
            source_hash = prepared_reports[scenario]["output_sha256"]
            if durable_dir.exists():
                print(f"Validating completed fold {fold}/{scenario}...", flush=True)
                reports[fold][scenario] = _validate_scenario_fold(
                    durable_dir,
                    scenario=scenario,
                    fold=fold,
                    source_sha256=source_hash,
                    config_sha256=config_sha256,
                    preprocessor_sha256=preprocessor_hashes[fold],
                )
                continue
            source_path = packet_paths[scenario]
            feature_dim = len(preprocessors[fold].feature_names)
            expected_rows = int(prepared_reports[scenario]["counts"]["packets"])
            estimated_uncompressed_output = expected_rows * (
                feature_dim * np.dtype(np.float32).itemsize
                + 2 * np.dtype(np.int32).itemsize
                + np.dtype(np.uint8).itemsize
                + np.dtype(np.int64).itemsize
            )
            required_bytes = (
                source_path.stat().st_size
                + estimated_uncompressed_output
                + 2 * 1024**3
            )
            if shutil.disk_usage(local_work_root).free < required_bytes:
                raise OSError(
                    f"Insufficient local space for fold {fold}/{scenario}: "
                    "need the staged source, an uncompressed-output estimate, "
                    "and a 2-GiB working reserve."
                )
            print(f"Staging fold {fold}/{scenario}...", flush=True)
            with tempfile.TemporaryDirectory(
                dir=local_work_root, prefix=f"graph_materialize_{fold}_{scenario}_"
            ) as temporary:
                temporary_path = Path(temporary)
                local_source = temporary_path / source_path.name
                shutil.copyfile(source_path, local_source)
                if sha256_file(local_source) != source_hash:
                    raise IOError(f"Staged prepared checksum mismatch: {scenario}")
                local_output = temporary_path / "output"
                report = _materialize_scenario_fold(
                    scenario=scenario,
                    fold=fold,
                    packet_path=local_source,
                    prepared_report=prepared_reports[scenario],
                    preprocessor=preprocessors[fold],
                    preprocessor_sha256=preprocessor_hashes[fold],
                    preprocessing_contract_sha256=preprocessing_schema_sha256(
                        preprocessing_schema
                    ),
                    output_dir=local_output,
                    config=config,
                    config_sha256=config_sha256,
                    batch_size=batch_size,
                )
                for shard in report["shards"]:
                    _validate_shard(
                        local_output / shard["path"], shard, int(report["feature_dim"])
                    )
                durable_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(local_output, durable_dir)
                reports[fold][scenario] = _validate_scenario_fold(
                    durable_dir,
                    scenario=scenario,
                    fold=fold,
                    source_sha256=source_hash,
                    config_sha256=config_sha256,
                    preprocessor_sha256=preprocessor_hashes[fold],
                )
            print(
                f"Completed fold {fold}/{scenario}: "
                f"{reports[fold][scenario]['graphs']:,} graphs, "
                f"{reports[fold][scenario]['edges']:,} edges, "
                f"{reports[fold][scenario]['shard_count']:,} shards",
                flush=True,
            )
    comparisons = _cross_fold_invariants(reports, scenarios)
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
    manifest_report = {
        "report_version": REPORT_VERSION,
        "status": "passed",
        "mode": mode,
        "scenarios": scenarios,
        "folds": {
            fold: {
                "training_scenarios": manifest["validation"]["folds"][fold]["train"],
                "validation_scenarios": manifest["validation"]["folds"][fold]["validate"],
                "preprocessor_sha256": preprocessor_hashes[fold],
                "feature_names": list(preprocessors[fold].feature_names),
                "feature_dim": len(preprocessors[fold].feature_names),
                "scenario_reports": {
                    scenario: f"fold_{fold}/{scenario}/scenario_fold_report.json"
                    for scenario in scenarios
                },
            }
            for fold in ("A", "B")
        },
        "cross_fold_invariants": comparisons,
        "totals": {
            "unique_development_packets": sum(
                int(prepared_reports[scenario]["counts"]["packets"])
                for scenario in scenarios
            ),
            "materialized_edges_across_folds": sum(
                reports[fold][scenario]["edges"]
                for fold in ("A", "B")
                for scenario in scenarios
            ),
            "graphs_across_folds": sum(
                reports[fold][scenario]["graphs"]
                for fold in ("A", "B")
                for scenario in scenarios
            ),
            "compressed_shard_bytes": sum(
                reports[fold][scenario]["compressed_shard_bytes"]
                for fold in ("A", "B")
                for scenario in scenarios
            ),
            "shards": sum(
                reports[fold][scenario]["shard_count"]
                for fold in ("A", "B")
                for scenario in scenarios
            ),
        },
        "provenance": {
            **run_config,
            "git_commit": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
            "working_tree_status": tree.stdout.strip() if tree.returncode == 0 else "unavailable",
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
            "held_out_scenarios_accessed": False,
            "training_performed": False,
        },
    }
    write_json(output_dir / "graph_materialization_manifest.json", manifest_report)
    write_json(
        output_dir / "run_status.json",
        {
            "complete": True,
            "mode": mode,
            "manifest_sha256": sha256_file(
                output_dir / "graph_materialization_manifest.json"
            ),
        },
    )
    print(f"Graph materialization complete: {output_dir}", flush=True)
    return manifest_report
