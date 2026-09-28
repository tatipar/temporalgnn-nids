"""Load and audit fold-specific cAPTure graph-materialization shards."""

from __future__ import annotations

from bisect import bisect_right
import json
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import yaml

from .capture_data import sha256_file, write_json
from .capture_graph_materialization import SHARD_ARRAYS


REPORT_VERSION = 2
NANOSECONDS_PER_MILLISECOND = 1_000_000
VALID_FOLDS = ("A", "B")
VALID_PARTITIONS = ("train", "validation")


def _load_json(path: Path, label: str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected {label} to contain a JSON object: {path}")
    return value


def load_capture_graph_input_contract(path: Path) -> dict:
    """Load and strictly validate the frozen Stage-2 graph-input contract."""
    contract = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("The Stage-2 graph-input contract must be a mapping.")
    if (
        contract.get("input_contract_version") != 2
        or contract.get("scope") != "development_only"
        or contract.get("stage") != "stage2_graph_input_validation"
    ):
        raise ValueError("Unsupported Stage-2 graph-input contract.")
    if contract.get("materialization") != {
        "required_mode": "FULL_DEV",
        "required_status": "passed",
        "required_feature_dimension": 103,
        "window_duration_seconds": 5,
        "decision_time": "window_end",
        "held_out_scenarios_accessed": False,
        "training_performed": False,
    }:
        raise ValueError("The required materialization contract changed.")
    required_folds = {
        "A": {
            "train": ["train_empty_conn", "train_qos_mid"],
            "validation": [
                "train_dollar_char",
                "train_slash_char",
                "train_sub_exf",
            ],
        },
        "B": {
            "train": [
                "train_dollar_char",
                "train_slash_char",
                "train_sub_exf",
            ],
            "validation": ["train_empty_conn", "train_qos_mid"],
        },
    }
    if contract.get("folds") != required_folds:
        raise ValueError("The Stage-2 fold assignments changed.")
    required_loader = {
        "graph_order": "chronological_within_scenario",
        "shuffle": False,
        "graph_batch_size": 1,
        "scenario_boundary_resets_temporal_state": True,
        "empty_windows": "preserve_through_window_index_and_timestamp_gaps",
        "timestamp_conversion": "floor_nanoseconds_to_integer_milliseconds",
        "edge_index_dtype": "torch_int64",
        "edge_attr_dtype": "torch_float32",
        "target_dtype": "torch_float32",
        "global_node_id_dtype": "torch_int64",
        "source_row_id_dtype": "torch_int64",
        "persist_node_features_x": False,
        "copy_drive_artifacts_to_local_before_repeated_access": True,
        "verify_all_checksums_after_staging": True,
    }
    if contract.get("loader") != required_loader:
        raise ValueError("The Stage-2 graph-loader contract changed.")
    required_audit = {
        "sample_graph_positions": ["first", "middle", "last"],
        "full_scan_required_before_training": True,
        "require_contiguous_source_rows": True,
        "require_one_oof_role_per_scenario": True,
        "allow_training": False,
        "allow_held_out_access": False,
    }
    if contract.get("audit") != required_audit:
        raise ValueError("The Stage-2 graph-input audit contract changed.")
    required_identity_audit = {
        "canonical_graph_fold": "A",
        "reconstruction": "per_edge_source_row_join",
        "prepared_endpoint_columns": ["src_endpoint", "dst_endpoint"],
        "require_exact_prepared_artifact_hash": True,
        "require_bijective_id_endpoint_mapping": True,
        "require_all_graph_nodes_resolved": True,
        "require_first_seen_rows_match": True,
        "require_reconstructed_contract_hash_match_both_folds": True,
        "persist_lookup_outside_model_inputs": True,
        "model_loader_access": False,
    }
    if contract.get("identity_audit") != required_identity_audit:
        raise ValueError("The Stage-2 node-identity audit contract changed.")
    return contract


class CaptureGraphCollection:
    """Validate one immutable FULL_DEV materialization and expose its fold roles."""

    def __init__(
        self,
        root: str | Path,
        contract_path: str | Path,
        *,
        expected_run_id: str,
        verify_artifact_checksums: bool = False,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.contract_path = Path(contract_path).expanduser().resolve()
        self.expected_run_id = str(expected_run_id)
        self.contract = load_capture_graph_input_contract(self.contract_path)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Graph materialization is missing: {self.root}")
        if self.root.name != self.expected_run_id:
            raise ValueError(
                f"Expected materialization run {self.expected_run_id}, found {self.root.name}."
            )

        self.manifest_path = self.root / "graph_materialization_manifest.json"
        self.run_config_path = self.root / "run_config.json"
        self.run_status_path = self.root / "run_status.json"
        self.manifest = _load_json(self.manifest_path, "graph materialization manifest")
        self.run_config = _load_json(self.run_config_path, "graph materialization run config")
        self.run_status = _load_json(self.run_status_path, "graph materialization status")
        self._validate_collection()
        self.artifact_checksums_verified = False
        if verify_artifact_checksums:
            self.verify_all_artifact_checksums()

    @property
    def scenarios(self) -> list[str]:
        return list(self.manifest["scenarios"])

    @property
    def feature_dim(self) -> int:
        return int(self.contract["materialization"]["required_feature_dimension"])

    @property
    def window_ms(self) -> int:
        return int(self.contract["materialization"]["window_duration_seconds"]) * 1000

    def _validate_collection(self) -> None:
        required = self.contract["materialization"]
        if (
            self.manifest.get("status") != required["required_status"]
            or self.manifest.get("mode") != required["required_mode"]
            or self.run_config.get("mode") != required["required_mode"]
            or self.run_status.get("complete") is not True
            or self.run_status.get("mode") != required["required_mode"]
            or self.run_status.get("manifest_sha256") != sha256_file(self.manifest_path)
        ):
            raise ValueError("The graph materialization is incomplete or has the wrong mode.")
        provenance = self.manifest.get("provenance", {})
        if (
            provenance.get("held_out_scenarios_accessed")
            != required["held_out_scenarios_accessed"]
            or provenance.get("training_performed") != required["training_performed"]
        ):
            raise ValueError("Materialization accessed held-out data or performed training.")
        if self.run_config.get("scenarios") != self.manifest.get("scenarios"):
            raise ValueError("Materialization scenarios differ between config and manifest.")
        for name, value in self.run_config.items():
            if provenance.get(name) != value:
                raise ValueError(
                    f"Materialization run config differs from manifest provenance: {name}"
                )
        if set(self.manifest.get("folds", {})) != set(VALID_FOLDS):
            raise ValueError("Materialization must contain exactly folds A and B.")

        configured_scenarios = []
        validation_roles = []
        for fold in VALID_FOLDS:
            declared = self.contract["folds"][fold]
            fold_manifest = self.manifest["folds"][fold]
            if (
                fold_manifest.get("training_scenarios") != declared["train"]
                or fold_manifest.get("validation_scenarios") != declared["validation"]
                or int(fold_manifest.get("feature_dim", -1)) != self.feature_dim
            ):
                raise ValueError(f"Fold {fold} differs from the Stage-2 input contract.")
            feature_names = fold_manifest.get("feature_names")
            if (
                not isinstance(feature_names, list)
                or len(feature_names) != self.feature_dim
                or len(set(feature_names)) != self.feature_dim
            ):
                raise ValueError(f"Fold {fold} has an invalid feature schema.")
            reports = fold_manifest.get("scenario_reports", {})
            if set(reports) != set(self.manifest["scenarios"]):
                raise ValueError(f"Fold {fold} does not declare every scenario report.")
            configured_scenarios.extend(declared["train"])
            validation_roles.extend(declared["validation"])

        if set(configured_scenarios) != set(self.manifest["scenarios"]):
            raise ValueError("Every development scenario must train in exactly one fold.")
        if (
            len(validation_roles) != len(set(validation_roles))
            or set(validation_roles) != set(self.manifest["scenarios"])
        ):
            raise ValueError("Every development scenario must have exactly one OOF role.")

        archived_hashes = {
            "capture_experiment_v1.yaml": "manifest_sha256",
            "capture_packet_schema_v1.yaml": "packet_schema_sha256",
            "capture_preprocessing_v1.yaml": "preprocessing_schema_sha256",
            "capture_graph_materialization_v1.yaml": "materialization_config_sha256",
            "stage1_decision.md": "stage1_decision_sha256",
        }
        for filename, hash_field in archived_hashes.items():
            path = self.root / filename
            if not path.is_file() or sha256_file(path) != provenance.get(hash_field):
                raise ValueError(f"Archived materialization input changed: {filename}")

    def scenarios_for(self, fold: str, partition: str) -> list[str]:
        if fold not in VALID_FOLDS:
            raise ValueError(f"Unknown fold: {fold}")
        if partition not in VALID_PARTITIONS:
            raise ValueError(f"Unknown partition: {partition}")
        return list(self.contract["folds"][fold][partition])

    def partition_for(self, fold: str, scenario: str) -> str:
        for partition in VALID_PARTITIONS:
            if scenario in self.scenarios_for(fold, partition):
                return partition
        raise ValueError(f"Scenario {scenario} is not assigned to fold {fold}.")

    def scenario_dataset(
        self,
        fold: str,
        scenario: str,
        *,
        expected_partition: str | None = None,
        graph_start: int = 0,
        graph_stop: int | None = None,
        verify_shard_checksums: bool | None = None,
    ) -> "CaptureGraphScenarioDataset":
        partition = self.partition_for(fold, scenario)
        if expected_partition is not None and partition != expected_partition:
            raise ValueError(
                f"Fold {fold}/{scenario} is {partition}, not {expected_partition}."
            )
        if verify_shard_checksums is None:
            verify_shard_checksums = not self.artifact_checksums_verified
        return CaptureGraphScenarioDataset(
            self,
            fold,
            scenario,
            partition=partition,
            graph_start=graph_start,
            graph_stop=graph_stop,
            verify_shard_checksums=verify_shard_checksums,
        )

    def datasets_for(
        self,
        fold: str,
        partition: str,
        *,
        verify_shard_checksums: bool | None = None,
    ) -> list["CaptureGraphScenarioDataset"]:
        return [
            self.scenario_dataset(
                fold,
                scenario,
                expected_partition=partition,
                verify_shard_checksums=verify_shard_checksums,
            )
            for scenario in self.scenarios_for(fold, partition)
        ]

    def verify_all_artifact_checksums(self) -> None:
        """Verify every scenario artifact after copying the collection."""
        for fold in VALID_FOLDS:
            for scenario in self.scenarios:
                directory = self.root / f"fold_{fold}" / scenario
                status = _load_json(directory / "run_status.json", "scenario-fold status")
                checksum_path = directory / "artifact_checksums.json"
                checksums = _load_json(checksum_path, "scenario-fold checksums")
                if (
                    status.get("complete") is not True
                    or status.get("fold") != fold
                    or status.get("scenario") != scenario
                    or status.get("artifact_checksums_sha256") != sha256_file(checksum_path)
                ):
                    raise ValueError(f"Incomplete scenario-fold status: {fold}/{scenario}")
                for name, expected in checksums.items():
                    path = directory / name
                    if not path.is_file() or sha256_file(path) != expected:
                        raise ValueError(
                            f"Artifact checksum mismatch: fold {fold}/{scenario}/{name}"
                        )
        self.artifact_checksums_verified = True


class CaptureGraphScenarioDataset(Dataset):
    """Expose one chronological fold/scenario sequence from compressed shards."""

    def __init__(
        self,
        collection: CaptureGraphCollection,
        fold: str,
        scenario: str,
        *,
        partition: str,
        graph_start: int = 0,
        graph_stop: int | None = None,
        verify_shard_checksums: bool = True,
    ) -> None:
        self.collection = collection
        self.fold = fold
        self.scenario = scenario
        self.partition = partition
        self.verify_shard_checksums = bool(verify_shard_checksums)
        self.directory = collection.root / f"fold_{fold}" / scenario
        report_relative = collection.manifest["folds"][fold]["scenario_reports"][scenario]
        expected_relative = f"fold_{fold}/{scenario}/scenario_fold_report.json"
        if report_relative != expected_relative:
            raise ValueError(f"Unexpected scenario-report path: {report_relative}")
        self.report = _load_json(
            collection.root / report_relative, "scenario-fold report"
        )
        self.checksums = _load_json(
            self.directory / "artifact_checksums.json", "scenario-fold checksums"
        )
        self._validate_report()
        self.shards = list(self.report["shards"])
        self._shard_offsets = np.zeros(len(self.shards) + 1, dtype=np.int64)
        self._shard_offsets[1:] = np.cumsum(
            [int(record["graphs"]) for record in self.shards], dtype=np.int64
        )
        total_graphs = int(self._shard_offsets[-1])
        if total_graphs != int(self.report["graphs"]):
            raise ValueError(f"Shard graph counts do not sum correctly: {fold}/{scenario}")
        if graph_stop is None:
            graph_stop = total_graphs
        if not 0 <= int(graph_start) < int(graph_stop) <= total_graphs:
            raise ValueError(
                f"Invalid graph range [{graph_start}, {graph_stop}) for {total_graphs} graphs."
            )
        self.graph_start = int(graph_start)
        self.graph_stop = int(graph_stop)
        self.edge_dim = int(self.report["feature_dim"])
        self.window_ms = collection.window_ms
        self._cache_index: int | None = None
        self._cache_arrays: dict[str, np.ndarray] | None = None
        self._verified_shards: set[int] = set()

    @property
    def is_complete_sequence(self) -> bool:
        return self.graph_start == 0 and self.graph_stop == int(self.report["graphs"])

    def _validate_report(self) -> None:
        if (
            self.report.get("fold") != self.fold
            or self.report.get("scenario") != self.scenario
            or int(self.report.get("feature_dim", -1)) != self.collection.feature_dim
            or int(self.report.get("window_width_seconds", -1)) * 1000
            != self.collection.window_ms
            or self.report.get("raw_endpoint_identifiers_persisted") is not False
            or self.report.get("node_features_x_persisted") is not False
        ):
            raise ValueError(f"Invalid scenario-fold report: {self.fold}/{self.scenario}")
        if self.report.get("feature_names") != self.collection.manifest["folds"][self.fold][
            "feature_names"
        ]:
            raise ValueError(f"Feature order changed: {self.fold}/{self.scenario}")
        if (
            int(self.report.get("graphs", 0)) <= 0
            or int(self.report.get("edges", 0)) <= 0
            or int(self.report.get("shard_count", 0)) != len(self.report.get("shards", []))
        ):
            raise ValueError(f"Empty or inconsistent scenario report: {self.fold}/{self.scenario}")

    def __len__(self) -> int:
        return self.graph_stop - self.graph_start

    def _load_shard(self, shard_index: int) -> dict[str, np.ndarray]:
        if self._cache_index == shard_index and self._cache_arrays is not None:
            return self._cache_arrays
        record = self.shards[shard_index]
        path = self.directory / record["path"]
        if self.verify_shard_checksums and shard_index not in self._verified_shards:
            expected = self.checksums.get(record["path"])
            if expected != record["sha256"] or sha256_file(path) != expected:
                raise ValueError(f"Shard checksum mismatch: {path}")
            self._verified_shards.add(shard_index)
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) != set(SHARD_ARRAYS):
                raise ValueError(f"Shard arrays differ from the contract: {path}")
            arrays = {name: loaded[name] for name in SHARD_ARRAYS}
        self._validate_shard_arrays(arrays, record, path)
        self._cache_index = shard_index
        self._cache_arrays = arrays
        return arrays

    def _validate_shard_arrays(
        self, arrays: dict[str, np.ndarray], record: dict, path: Path
    ) -> None:
        graphs = int(record["graphs"])
        edges = int(record["edges"])
        nodes = int(record["node_appearances"])
        expected_shapes = {
            "edge_index": (2, edges),
            "edge_attr": (edges, self.edge_dim),
            "y": (edges,),
            "source_row_id": (edges,),
            "global_node_ids": (nodes,),
            "edge_ptr": (graphs + 1,),
            "node_ptr": (graphs + 1,),
            "window_index": (graphs,),
            "window_start_ns": (graphs,),
            "window_end_ns": (graphs,),
            "decision_time_ns": (graphs,),
        }
        for name, shape in expected_shapes.items():
            if arrays[name].shape != shape:
                raise ValueError(f"Unexpected {name} shape in {path}: {arrays[name].shape}")
        expected_dtypes = {
            "edge_index": np.dtype(np.int32),
            "edge_attr": np.dtype(np.float32),
            "y": np.dtype(np.uint8),
            "source_row_id": np.dtype(np.int64),
            "global_node_ids": np.dtype(np.int32),
            "edge_ptr": np.dtype(np.int64),
            "node_ptr": np.dtype(np.int64),
            "window_index": np.dtype(np.int64),
            "window_start_ns": np.dtype(np.int64),
            "window_end_ns": np.dtype(np.int64),
            "decision_time_ns": np.dtype(np.int64),
        }
        for name, dtype in expected_dtypes.items():
            if arrays[name].dtype != dtype:
                raise ValueError(f"Unexpected {name} dtype in {path}: {arrays[name].dtype}")
        if (
            arrays["edge_ptr"][0] != 0
            or arrays["edge_ptr"][-1] != edges
            or arrays["node_ptr"][0] != 0
            or arrays["node_ptr"][-1] != nodes
            or not np.all(arrays["edge_ptr"][1:] > arrays["edge_ptr"][:-1])
            or not np.all(arrays["node_ptr"][1:] > arrays["node_ptr"][:-1])
        ):
            raise ValueError(f"Invalid shard pointers: {path}")
        if not np.array_equal(arrays["decision_time_ns"], arrays["window_end_ns"]):
            raise ValueError(f"Decision times differ from window ends: {path}")
        if not np.all(arrays["window_index"][1:] > arrays["window_index"][:-1]):
            raise ValueError(f"Window indexes are not strictly increasing: {path}")
        if not np.isfinite(arrays["edge_attr"]).all():
            raise ValueError(f"Shard contains non-finite edge features: {path}")

    def _absolute_graph_index(self, index: int) -> int:
        size = len(self)
        if index < 0:
            index += size
        if index < 0 or index >= size:
            raise IndexError(index)
        return self.graph_start + index

    def __getitem__(self, index: int) -> Data:
        absolute = self._absolute_graph_index(int(index))
        shard_index = bisect_right(self._shard_offsets, absolute) - 1
        arrays = self._load_shard(shard_index)
        graph_in_shard = absolute - int(self._shard_offsets[shard_index])
        edge_start = int(arrays["edge_ptr"][graph_in_shard])
        edge_stop = int(arrays["edge_ptr"][graph_in_shard + 1])
        node_start = int(arrays["node_ptr"][graph_in_shard])
        node_stop = int(arrays["node_ptr"][graph_in_shard + 1])

        edge_index_np = arrays["edge_index"][:, edge_start:edge_stop].astype(
            np.int64, copy=True
        )
        edge_attr_np = np.ascontiguousarray(
            arrays["edge_attr"][edge_start:edge_stop]
        )
        targets_np = arrays["y"][edge_start:edge_stop].astype(np.float32, copy=True)
        source_rows_np = np.ascontiguousarray(
            arrays["source_row_id"][edge_start:edge_stop]
        )
        global_nodes_np = arrays["global_node_ids"][node_start:node_stop].astype(
            np.int64, copy=True
        )
        window_index = int(arrays["window_index"][graph_in_shard])
        window_start_ns = int(arrays["window_start_ns"][graph_in_shard])
        window_end_ns = int(arrays["window_end_ns"][graph_in_shard])
        decision_time_ns = int(arrays["decision_time_ns"][graph_in_shard])

        data = Data(
            edge_index=torch.from_numpy(edge_index_np),
            edge_attr=torch.from_numpy(edge_attr_np),
            y=torch.from_numpy(targets_np),
            num_nodes=len(global_nodes_np),
        )
        data.global_node_ids = torch.from_numpy(global_nodes_np)
        data.source_row_id = torch.from_numpy(source_rows_np)
        data.timestamp = decision_time_ns // NANOSECONDS_PER_MILLISECOND
        data.window_start = window_start_ns // NANOSECONDS_PER_MILLISECOND
        data.window_end = window_end_ns // NANOSECONDS_PER_MILLISECOND
        data.decision_time_ns = decision_time_ns
        data.window_start_ns = window_start_ns
        data.window_end_ns = window_end_ns
        data.window_index = window_index
        data.fold = self.fold
        data.scenario = self.scenario
        data.partition = self.partition
        data.feature_names_sha256 = self.report["feature_names_sha256"]
        self._validate_graph(data)
        return data

    def __iter__(self) -> Iterator[Data]:
        for index in range(len(self)):
            yield self[index]

    def _validate_graph(self, data: Data) -> None:
        edge_count = int(data.edge_attr.shape[0])
        node_count = int(data.num_nodes)
        if (
            getattr(data, "x", None) is not None
            or data.edge_index.dtype != torch.long
            or data.edge_index.shape != (2, edge_count)
            or data.edge_attr.dtype != torch.float32
            or data.edge_attr.shape != (edge_count, self.edge_dim)
            or data.y.dtype != torch.float32
            or data.y.shape != (edge_count,)
            or data.source_row_id.dtype != torch.long
            or data.source_row_id.shape != (edge_count,)
            or data.global_node_ids.dtype != torch.long
            or data.global_node_ids.shape != (node_count,)
            or edge_count <= 0
            or node_count <= 0
        ):
            raise ValueError(f"Invalid loaded graph contract: {self.fold}/{self.scenario}")
        if (
            not torch.isfinite(data.edge_attr).all()
            or not torch.all((data.y == 0) | (data.y == 1))
            or int(torch.unique(data.global_node_ids).numel()) != node_count
            or not torch.all(data.global_node_ids[1:] > data.global_node_ids[:-1])
            or int(data.edge_index.min()) < 0
            or int(data.edge_index.max()) >= node_count
        ):
            raise ValueError(f"Invalid loaded graph values: {self.fold}/{self.scenario}")
        if (
            data.window_end_ns != data.decision_time_ns
            or data.window_end_ns - data.window_start_ns != self.window_ms * 1_000_000
            or data.window_end - data.window_start != self.window_ms
            or data.timestamp != data.window_end
        ):
            raise ValueError(f"Invalid loaded graph timing: {self.fold}/{self.scenario}")

    def validate_all(self) -> dict:
        """Load the selected sequence and verify chronological packet correspondence."""
        graph_count = edge_count = attack_edges = node_appearances = 0
        maximum_edges = 0
        first_source_row = last_source_row = None
        first_window_index = last_window_index = None
        previous_source_row = None
        previous_window_index = None
        previous_timestamp = None
        for data in self:
            source_rows = data.source_row_id.numpy()
            expected_start = (
                int(source_rows[0])
                if previous_source_row is None
                else previous_source_row + 1
            )
            expected_rows = np.arange(
                expected_start, expected_start + len(source_rows), dtype=np.int64
            )
            if not np.array_equal(source_rows, expected_rows):
                raise ValueError(
                    f"Source rows are not contiguous: {self.fold}/{self.scenario}"
                )
            if previous_window_index is not None and data.window_index <= previous_window_index:
                raise ValueError(
                    f"Window order decreased: {self.fold}/{self.scenario}"
                )
            if previous_timestamp is not None and data.timestamp <= previous_timestamp:
                raise ValueError(
                    f"Timestamp order decreased: {self.fold}/{self.scenario}"
                )
            if first_source_row is None:
                first_source_row = int(source_rows[0])
                first_window_index = int(data.window_index)
            last_source_row = int(source_rows[-1])
            last_window_index = int(data.window_index)
            previous_source_row = last_source_row
            previous_window_index = last_window_index
            previous_timestamp = int(data.timestamp)
            current_edges = int(data.edge_attr.shape[0])
            graph_count += 1
            edge_count += current_edges
            attack_edges += int(data.y.sum().item())
            node_appearances += int(data.num_nodes)
            maximum_edges = max(maximum_edges, current_edges)

        result = {
            "fold": self.fold,
            "scenario": self.scenario,
            "partition": self.partition,
            "complete_sequence": self.is_complete_sequence,
            "graphs": graph_count,
            "edges": edge_count,
            "attack_edges": attack_edges,
            "node_appearances": node_appearances,
            "maximum_edges_in_graph": maximum_edges,
            "first_source_row_id": first_source_row,
            "last_source_row_id": last_source_row,
            "first_window_index": first_window_index,
            "last_window_index": last_window_index,
            "source_rows_contiguous": True,
            "windows_and_timestamps_strictly_increasing": True,
        }
        if self.is_complete_sequence:
            expected = {
                "graphs": int(self.report["graphs"]),
                "edges": int(self.report["edges"]),
                "attack_edges": int(self.report["attack_edges"]),
                "node_appearances": int(self.report["node_appearances"]),
                "first_source_row_id": 0,
                "last_source_row_id": int(self.report["edges"]) - 1,
                "first_window_index": int(self.report["first_window_index"]),
                "last_window_index": int(self.report["last_window_index"]),
            }
            mismatches = {
                name: {"observed": result[name], "expected": value}
                for name, value in expected.items()
                if result[name] != value
            }
            if mismatches:
                raise ValueError(
                    f"Full loader audit differs from materialization: "
                    f"{self.fold}/{self.scenario}: {sorted(mismatches)}"
                )
        return result


def stage_capture_graph_materialization(
    *,
    source_root: str | Path,
    local_parent: str | Path,
    contract_path: str | Path,
    expected_run_id: str,
    reserve_bytes: int = 2 * 1024**3,
) -> CaptureGraphCollection:
    """Copy one immutable Drive materialization locally and verify every checksum."""
    source_root = Path(source_root).expanduser().resolve()
    local_parent = Path(local_parent).expanduser().resolve()
    contract_path = Path(contract_path).expanduser().resolve()
    target_root = local_parent / expected_run_id
    source_manifest_sha256 = sha256_file(
        source_root / "graph_materialization_manifest.json"
    )
    binding = {
        "expected_run_id": expected_run_id,
        "source_manifest_sha256": source_manifest_sha256,
        "input_contract_sha256": sha256_file(contract_path),
    }
    receipt_name = "local_staging_receipt.json"
    if target_root.exists():
        receipt = _load_json(target_root / receipt_name, "local staging receipt")
        if receipt.get("binding") != binding:
            raise ValueError("The existing local graph copy has a different source binding.")
        collection = CaptureGraphCollection(
            target_root,
            contract_path,
            expected_run_id=expected_run_id,
            verify_artifact_checksums=True,
        )
        if receipt.get("local_manifest_sha256") != sha256_file(collection.manifest_path):
            raise ValueError("The existing local graph manifest changed after staging.")
        print(f"Reusing verified local graph materialization: {target_root}", flush=True)
        return collection

    local_parent.mkdir(parents=True, exist_ok=True)
    source_bytes = sum(
        path.stat().st_size for path in source_root.rglob("*") if path.is_file()
    )
    if shutil.disk_usage(local_parent).free < source_bytes + int(reserve_bytes):
        raise OSError(
            "Insufficient local space for the graph materialization and safety reserve."
        )
    with tempfile.TemporaryDirectory(
        dir=local_parent, prefix="capture_graph_stage_"
    ) as temporary:
        staged_root = Path(temporary) / expected_run_id
        shutil.copytree(source_root, staged_root)
        collection = CaptureGraphCollection(
            staged_root,
            contract_path,
            expected_run_id=expected_run_id,
            verify_artifact_checksums=True,
        )
        write_json(
            staged_root / receipt_name,
            {
                "binding": binding,
                "local_manifest_sha256": sha256_file(collection.manifest_path),
                "all_materialization_checksums_verified": True,
            },
        )
        staged_root.rename(target_root)
    print(f"Staged and verified graph materialization: {target_root}", flush=True)
    collection = CaptureGraphCollection(
        target_root,
        contract_path,
        expected_run_id=expected_run_id,
        verify_artifact_checksums=False,
    )
    collection.artifact_checksums_verified = True
    return collection


def audit_capture_graph_inputs(
    collection: CaptureGraphCollection,
    *,
    mode: str,
) -> dict:
    """Audit sample positions or fully scan every fold/scenario graph sequence."""
    if mode not in {"SAMPLE", "FULL"}:
        raise ValueError("Graph-input audit mode must be SAMPLE or FULL.")
    if not collection.artifact_checksums_verified:
        raise ValueError("Verify every staged materialization checksum before auditing inputs.")
    audit_started = time.perf_counter()
    scenario_results = {}
    for fold in VALID_FOLDS:
        scenario_results[fold] = {}
        for scenario in collection.scenarios:
            dataset = collection.scenario_dataset(
                fold,
                scenario,
                verify_shard_checksums=False,
            )
            scenario_started = time.perf_counter()
            if mode == "FULL":
                result = dataset.validate_all()
            else:
                positions = sorted({0, len(dataset) // 2, len(dataset) - 1})
                samples = []
                for position in positions:
                    data = dataset[position]
                    samples.append(
                        {
                            "graph_position": position,
                            "window_index": int(data.window_index),
                            "edges": int(data.edge_attr.shape[0]),
                            "nodes": int(data.num_nodes),
                            "first_source_row_id": int(data.source_row_id[0]),
                            "last_source_row_id": int(data.source_row_id[-1]),
                            "timestamp_ms": int(data.timestamp),
                        }
                    )
                result = {
                    "fold": fold,
                    "scenario": scenario,
                    "partition": dataset.partition,
                    "complete_sequence": False,
                    "graphs_in_sequence": len(dataset),
                    "sampled_graphs": samples,
                }
            scenario_elapsed = time.perf_counter() - scenario_started
            result["wall_seconds"] = float(scenario_elapsed)
            if mode == "FULL":
                result["graphs_per_second"] = result["graphs"] / scenario_elapsed
                result["edges_per_second"] = result["edges"] / scenario_elapsed
            scenario_results[fold][scenario] = result

    oof_scenarios = {
        scenario: fold
        for fold in VALID_FOLDS
        for scenario in collection.scenarios_for(fold, "validation")
    }
    training_edges = sum(
        int(
            collection.scenario_dataset(
                fold,
                scenario,
                expected_partition="train",
                verify_shard_checksums=False,
            ).report["edges"]
        )
        for fold in VALID_FOLDS
        for scenario in collection.scenarios_for(fold, "train")
    )
    oof_edges = sum(
        int(
            collection.scenario_dataset(
                fold,
                scenario,
                expected_partition="validation",
                verify_shard_checksums=False,
            ).report["edges"]
        )
        for fold in VALID_FOLDS
        for scenario in collection.scenarios_for(fold, "validation")
    )
    unique_packets = int(collection.manifest["totals"]["unique_development_packets"])
    if training_edges != unique_packets or oof_edges != unique_packets:
        raise ValueError("Training or OOF roles do not cover development packets exactly once.")
    return {
        "report_version": REPORT_VERSION,
        "status": "passed",
        "mode": mode,
        "materialization_run_id": collection.expected_run_id,
        "materialization_manifest_sha256": sha256_file(collection.manifest_path),
        "input_contract_sha256": sha256_file(collection.contract_path),
        "loader_code_sha256": sha256_file(Path(__file__)),
        "feature_dim": collection.feature_dim,
        "window_ms": collection.window_ms,
        "folds": {
            fold: {
                "train": collection.scenarios_for(fold, "train"),
                "validation": collection.scenarios_for(fold, "validation"),
            }
            for fold in VALID_FOLDS
        },
        "oof_scenario_fold": oof_scenarios,
        "unique_development_packets": unique_packets,
        "training_edges_across_folds": training_edges,
        "oof_edges_across_folds": oof_edges,
        "scenario_results": scenario_results,
        "wall_seconds": float(time.perf_counter() - audit_started),
        "all_materialization_checksums_verified": (
            collection.artifact_checksums_verified
        ),
        "held_out_scenarios_accessed": False,
        "training_performed": False,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
        },
    }


def save_capture_graph_input_audit(
    output_dir: str | Path,
    report: dict,
    *,
    contract_path: str | Path,
    identity_lookups: dict[str, object],
) -> None:
    """Persist one immutable Stage-2 input audit and diagnostic identity lookups."""
    output_dir = Path(output_dir)
    contract_path = Path(contract_path)
    if output_dir.exists():
        raise FileExistsError(f"Graph-input audit output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    if sha256_file(contract_path) != report.get("input_contract_sha256"):
        raise ValueError("The saved graph-input audit has a different contract hash.")
    identity_report = report.get("identity_audit", {})
    if (
        report.get("mode") != "FULL"
        or identity_report.get("status") != "passed"
        or identity_report.get("raw_identity_lookup_is_model_input") is not False
        or set(identity_report.get("scenarios", {})) != set(identity_lookups)
    ):
        raise ValueError("A passed full identity audit is required before saving.")

    lookup_dir = output_dir / "node_identity_lookup"
    lookup_dir.mkdir()
    lookup_checksums = {}
    required_columns = [
        "scenario",
        "global_node_id",
        "mac_address",
        "first_seen_source_row_id",
        "first_seen_role",
    ]
    for scenario in sorted(identity_lookups):
        table = identity_lookups[scenario]
        if list(table.columns) != required_columns or set(table["scenario"]) != {scenario}:
            raise ValueError(f"Invalid identity lookup table: {scenario}")
        relative = f"node_identity_lookup/{scenario}.parquet"
        path = output_dir / relative
        table.to_parquet(path, index=False, compression="zstd")
        checksum = sha256_file(path)
        lookup_checksums[relative] = checksum
        identity_report["scenarios"][scenario]["lookup_artifact"] = relative
        identity_report["scenarios"][scenario]["lookup_sha256"] = checksum
    lookup_checksum_path = lookup_dir / "artifact_checksums.json"
    write_json(lookup_checksum_path, lookup_checksums)
    identity_report["lookup_artifact_checksums_sha256"] = sha256_file(
        lookup_checksum_path
    )
    shutil.copyfile(contract_path, output_dir / contract_path.name)
    report_path = output_dir / "capture_graph_input_audit.json"
    write_json(report_path, report)
    write_json(
        output_dir / "run_status.json",
        {
            "complete": True,
            "mode": report["mode"],
            "report_sha256": sha256_file(report_path),
            "identity_lookup_checksums_sha256": identity_report[
                "lookup_artifact_checksums_sha256"
            ],
        },
    )


def load_completed_capture_graph_input_audit(
    output_dir: str | Path,
    collection: CaptureGraphCollection,
    *,
    mode: str,
) -> dict:
    """Load a completed immutable audit bound to the current collection and code."""
    output_dir = Path(output_dir)
    report_path = output_dir / "capture_graph_input_audit.json"
    status = _load_json(output_dir / "run_status.json", "graph-input audit status")
    report = _load_json(report_path, "graph-input audit report")
    expected = {
        "mode": mode,
        "materialization_run_id": collection.expected_run_id,
        "materialization_manifest_sha256": sha256_file(collection.manifest_path),
        "input_contract_sha256": sha256_file(collection.contract_path),
        "loader_code_sha256": sha256_file(Path(__file__)),
    }
    from . import capture_graph_identity

    identity_report = report.get("identity_audit", {})
    expected_identity_code_sha256 = sha256_file(Path(capture_graph_identity.__file__))
    if (
        status.get("complete") is not True
        or status.get("mode") != mode
        or status.get("report_sha256") != sha256_file(report_path)
        or report.get("status") != "passed"
        or any(report.get(name) != value for name, value in expected.items())
        or identity_report.get("status") != "passed"
        or identity_report.get("identity_audit_code_sha256")
        != expected_identity_code_sha256
        or identity_report.get("raw_identity_lookup_is_model_input") is not False
        or identity_report.get("cross_fold_mapping_contracts_match") is not True
        or set(identity_report.get("scenarios", {})) != set(collection.scenarios)
    ):
        raise ValueError("The completed graph-input audit has a different binding.")
    archived_contract = output_dir / collection.contract_path.name
    if (
        not archived_contract.is_file()
        or sha256_file(archived_contract) != expected["input_contract_sha256"]
    ):
        raise ValueError("The archived graph-input contract changed.")
    lookup_checksum_path = output_dir / "node_identity_lookup" / "artifact_checksums.json"
    lookup_checksums = _load_json(
        lookup_checksum_path, "node-identity lookup checksums"
    )
    if (
        sha256_file(lookup_checksum_path)
        != identity_report.get("lookup_artifact_checksums_sha256")
        or status.get("identity_lookup_checksums_sha256")
        != identity_report.get("lookup_artifact_checksums_sha256")
    ):
        raise ValueError("The node-identity lookup checksum manifest changed.")
    for scenario, scenario_report in identity_report["scenarios"].items():
        relative = scenario_report.get("lookup_artifact")
        expected_checksum = scenario_report.get("lookup_sha256")
        path = output_dir / str(relative)
        if (
            scenario_report.get("status") != "passed"
            or scenario_report.get("id_to_mac_conflicts") != 0
            or scenario_report.get("mac_to_id_conflicts") != 0
            or scenario_report.get("unresolved_global_nodes") != 0
            or scenario_report.get("reconstructed_contract_matches_both_folds") is not True
            or lookup_checksums.get(relative) != expected_checksum
            or not path.is_file()
            or sha256_file(path) != expected_checksum
        ):
            raise ValueError(f"Invalid completed node-identity audit: {scenario}")
    return report
