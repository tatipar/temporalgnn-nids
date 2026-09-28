"""Audit and freeze the cAPTure Stage-2 inner-checkpoint split."""

from __future__ import annotations

import json
import math
from pathlib import Path
import platform
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch_geometric.data import Data
import yaml

from .capture_data import sha256_file, write_json
from .capture_feature_profile import load_prepared_full_dev
from .capture_graph_dataset import (
    CaptureGraphCollection,
    load_completed_capture_graph_input_audit,
)
from .models import (
    EdgeGRU_Baseline_NoX,
    SimpleMLP,
    StaticGNN_Identity,
    ST_GNN_Identity,
)
from .training import forward_graph, validate_temporal_configuration


REPORT_VERSION = 1
WINDOW_SECONDS = 5
WINDOW_NANOSECONDS = WINDOW_SECONDS * 1_000_000_000
VALID_FOLDS = ("A", "B")
EXPECTED_VARIANTS = [
    "edge_mlp",
    "edge_gru",
    "static_gnn",
    "st_gnn",
    "st_gnn_without_gat",
    "st_gnn_without_direct_edge_attr",
]


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected {label} to contain a JSON object: {path}")
    return value


def load_training_preflight_config(path: str | Path) -> dict:
    """Load and strictly validate the frozen training-preflight contract."""
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The training-preflight configuration must be a mapping.")
    if (
        config.get("preflight_version") != 1
        or config.get("scope") != "development_only"
        or config.get("stage") != "stage2_training_preflight"
    ):
        raise ValueError("Unsupported training-preflight configuration.")
    bindings = config.get("bindings", {})
    required_binding_names = {
        "prepared_run_id",
        "graph_materialization_run_id",
        "graph_input_audit_run_id",
        "graph_input_contract",
        "require_completed_identity_audit",
    }
    if set(bindings) != required_binding_names or (
        bindings.get("require_completed_identity_audit") is not True
    ):
        raise ValueError("The training-preflight input bindings changed.")

    split = config.get("inner_split", {})
    if split.get("basis") != "scenario_wall_clock_window_index":
        raise ValueError("The inner split must use wall-clock window indexes.")
    if split.get("training_condition") != "window_index_less_than_boundary":
        raise ValueError("The inner-training boundary condition changed.")
    if split.get("validation_condition") != "window_index_greater_than_or_equal_to_boundary":
        raise ValueError("The inner-validation boundary condition changed.")
    fractions = split.get("candidate_validation_tail_fractions")
    if fractions != [0.20, 0.25, 0.30]:
        raise ValueError("The declared inner-validation candidates changed.")
    if (
        split.get("choose_first_candidate_valid_for_every_development_scenario")
        is not True
        or split.get("never_split_iteration_key") != ["attack_step", "sequence_id"]
    ):
        raise ValueError("The global split-selection policy changed.")
    shift = split.get("maximum_boundary_shift_fraction")
    if not isinstance(shift, float) or not 0 <= shift <= 0.05:
        raise ValueError("maximum_boundary_shift_fraction must be in [0, 0.05].")
    minimums = split.get("minimums", {})
    expected_minimums = {
        "inner_train_nonempty_windows": 500,
        "inner_validation_nonempty_windows": 500,
        "inner_train_benign_only_windows": 100,
        "inner_validation_benign_only_windows": 100,
        "inner_train_attack_containing_windows": 100,
        "inner_validation_attack_containing_windows": 100,
        "inner_train_complete_attack_iterations": 10,
        "inner_validation_complete_attack_iterations": 10,
        "inner_train_distinct_attack_steps": 2,
        "inner_validation_distinct_attack_steps": 2,
        "require_both_packet_labels_in_each_partition": True,
    }
    if minimums != expected_minimums:
        raise ValueError("The training-preflight admissibility minimums changed.")

    checkpointing = config.get("checkpointing", {})
    required_checkpointing = {
        "metric": "unweighted_mean_scenario_average_precision",
        "inner_validation_only": True,
        "outer_validation_forbidden": True,
        "thresholds_selected_during_checkpointing": False,
        "scenario_order": "declared_fold_order",
        "graph_order": "chronological",
        "shuffle": False,
        "reset_temporal_state_at_each_scenario_boundary": True,
        "inner_validation_starts_with_empty_temporal_state": True,
    }
    if checkpointing != required_checkpointing:
        raise ValueError("The checkpointing preflight contract changed.")

    model_contract = config.get("model_contract", {})
    if (
        model_contract.get("edge_feature_dimension") != 103
        or model_contract.get("graph_batch_size") != 1
        or model_contract.get("node_features_x_allowed") is not False
        or model_contract.get("one_logit_per_edge") is not True
        or model_contract.get("variants") != EXPECTED_VARIANTS
        or model_contract.get("synthetic_forward_only") is not True
        or model_contract.get("optimization_allowed") is not False
    ):
        raise ValueError("The synthetic model-interface contract changed.")
    required_prohibitions = {
        "held_out_scenarios_accessed": False,
        "model_training_performed": False,
        "model_scores_used_to_choose_split": False,
        "raw_endpoint_identity_is_model_input": False,
    }
    if config.get("prohibitions") != required_prohibitions:
        raise ValueError("The training-preflight prohibitions changed.")
    return config


def _profile_prepared_scenario(
    packet_path: Path,
    prepared_report: dict,
    *,
    batch_size: int,
) -> dict:
    origin_ns = int(prepared_report["scenario_origin_timestamp_ns"])
    last_timestamp_ns = int(prepared_report["last_packet_timestamp_ns"])
    final_window_index = (last_timestamp_ns - origin_ns) // WINDOW_NANOSECONDS
    total_windows = int(final_window_index) + 1
    packet_counts = np.zeros(total_windows, dtype=np.int64)
    attack_counts = np.zeros(total_windows, dtype=np.int64)
    iteration_spans: dict[tuple[str, str], list[int]] = {}
    expected_row = 0
    previous_timestamp = None
    parquet = pq.ParquetFile(packet_path)
    columns = [
        "source_row_id",
        "packet_timestamp_ns",
        "binary_label",
        "attack_step",
        "sequence_id",
    ]
    for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
        frame = batch.to_pandas()
        row_ids = frame["source_row_id"].to_numpy(dtype=np.int64)
        expected = np.arange(expected_row, expected_row + len(frame), dtype=np.int64)
        if not np.array_equal(row_ids, expected):
            raise ValueError("Prepared source rows changed during split preflight.")
        expected_row += len(frame)
        timestamps = frame["packet_timestamp_ns"].to_numpy(dtype=np.int64)
        if (
            np.any(timestamps[1:] < timestamps[:-1])
            or (previous_timestamp is not None and int(timestamps[0]) < previous_timestamp)
        ):
            raise ValueError("Prepared timestamps are not chronological.")
        previous_timestamp = int(timestamps[-1])
        window_indexes = ((timestamps - origin_ns) // WINDOW_NANOSECONDS).astype(
            np.int64
        )
        labels = frame["binary_label"].to_numpy(dtype=np.int8)
        if not np.isin(labels, [0, 1]).all():
            raise ValueError("Prepared binary labels changed during split preflight.")
        np.add.at(packet_counts, window_indexes, 1)
        np.add.at(attack_counts, window_indexes, labels)

        attack_rows = frame.loc[labels == 1, ["attack_step", "sequence_id"]].copy()
        if attack_rows.isna().any().any():
            raise ValueError("Attack rows have incomplete iteration metadata.")
        attack_rows["window_index"] = window_indexes[labels == 1]
        grouped = (
            attack_rows.groupby(["attack_step", "sequence_id"], observed=True)[
                "window_index"
            ]
            .agg(["min", "max", "size"])
            .reset_index()
        )
        for item in grouped.itertuples(index=False):
            key = (str(item.attack_step), str(item.sequence_id))
            observed = [int(item.min), int(item.max), int(item.size)]
            previous = iteration_spans.get(key)
            if previous is None:
                iteration_spans[key] = observed
            else:
                previous[0] = min(previous[0], observed[0])
                previous[1] = max(previous[1], observed[1])
                previous[2] += observed[2]

    expected_packets = int(prepared_report["counts"]["packets"])
    expected_attacks = int(prepared_report["counts"]["attack_packets"])
    if (
        expected_row != expected_packets
        or int(packet_counts.sum()) != expected_packets
        or int(attack_counts.sum()) != expected_attacks
        or not iteration_spans
    ):
        raise ValueError("Prepared split profiling failed packet conservation.")
    return {
        "origin_ns": origin_ns,
        "total_windows": total_windows,
        "packet_counts": packet_counts,
        "attack_counts": attack_counts,
        "iteration_spans": iteration_spans,
    }


def _nearest_iteration_safe_boundary(
    profile: dict,
    *,
    tail_fraction: float,
    maximum_shift_fraction: float,
) -> tuple[int | None, int, int]:
    total_windows = int(profile["total_windows"])
    target = int(round(total_windows * (1.0 - tail_fraction)))
    maximum_shift = max(1, int(math.ceil(total_windows * maximum_shift_fraction)))
    lower = max(1, target - maximum_shift)
    upper = min(total_windows - 1, target + maximum_shift)
    candidates = sorted(range(lower, upper + 1), key=lambda value: (abs(value - target), -value))
    spans = profile["iteration_spans"].values()
    for boundary in candidates:
        if not any(start < boundary <= stop for start, stop, _ in spans):
            return boundary, target, maximum_shift
    return None, target, maximum_shift


def _partition_counts(profile: dict, boundary: int) -> dict:
    packets = profile["packet_counts"]
    attacks = profile["attack_counts"]
    nonempty = packets > 0
    attack_windows = attacks > 0
    benign_only = nonempty & ~attack_windows
    train_slice = slice(0, boundary)
    validation_slice = slice(boundary, len(packets))
    train_iterations = []
    validation_iterations = []
    split_iterations = []
    for (step, sequence), (start, stop, attack_packets) in profile[
        "iteration_spans"
    ].items():
        item = {
            "attack_step": step,
            "sequence_id": sequence,
            "first_window_index": start,
            "last_window_index": stop,
            "attack_packets": attack_packets,
        }
        if stop < boundary:
            train_iterations.append(item)
        elif start >= boundary:
            validation_iterations.append(item)
        else:
            split_iterations.append(item)

    def summarize(partition_slice: slice, iterations: list[dict]) -> dict:
        partition_packets = int(packets[partition_slice].sum())
        partition_attacks = int(attacks[partition_slice].sum())
        return {
            "wall_clock_windows": int(partition_slice.stop - partition_slice.start),
            "nonempty_windows": int(nonempty[partition_slice].sum()),
            "benign_only_windows": int(benign_only[partition_slice].sum()),
            "attack_containing_windows": int(attack_windows[partition_slice].sum()),
            "packets": partition_packets,
            "normal_packets": partition_packets - partition_attacks,
            "attack_packets": partition_attacks,
            "complete_attack_iterations": len(iterations),
            "distinct_attack_steps": len(
                {item["attack_step"] for item in iterations}
            ),
        }

    return {
        "inner_train": summarize(train_slice, train_iterations),
        "inner_validation": summarize(validation_slice, validation_iterations),
        "split_iterations": split_iterations,
    }


def _admissibility_reasons(counts: dict, minimums: dict) -> list[str]:
    reasons = []
    for partition in ("inner_train", "inner_validation"):
        values = counts[partition]
        for metric in (
            "nonempty_windows",
            "benign_only_windows",
            "attack_containing_windows",
            "complete_attack_iterations",
            "distinct_attack_steps",
        ):
            minimum = int(minimums[f"{partition}_{metric}"])
            if int(values[metric]) < minimum:
                reasons.append(f"{partition}.{metric}={values[metric]}<{minimum}")
        if minimums["require_both_packet_labels_in_each_partition"] and (
            values["normal_packets"] <= 0 or values["attack_packets"] <= 0
        ):
            reasons.append(f"{partition}.both_packet_labels=false")
    if counts["split_iterations"]:
        reasons.append(f"split_iterations={len(counts['split_iterations'])}")
    return reasons


def _graph_window_alignment(dataset, boundary: int) -> dict:
    window_chunks = []
    edge_chunks = []
    for record in dataset.shards:
        path = dataset.directory / record["path"]
        with np.load(path, allow_pickle=False) as shard:
            windows = shard["window_index"]
            edge_ptr = shard["edge_ptr"]
        window_chunks.append(windows.astype(np.int64, copy=False))
        edge_chunks.append(np.diff(edge_ptr).astype(np.int64, copy=False))
    windows = np.concatenate(window_chunks)
    edges = np.concatenate(edge_chunks)
    if not np.all(windows[1:] > windows[:-1]):
        raise ValueError(f"Graph windows are not chronological: {dataset.scenario}")
    position = int(np.searchsorted(windows, boundary, side="left"))
    return {
        "graph_position_boundary": position,
        "inner_train_graphs": position,
        "inner_validation_graphs": int(len(windows) - position),
        "inner_train_edges": int(edges[:position].sum()),
        "inner_validation_edges": int(edges[position:].sum()),
        "first_inner_validation_window_index": int(windows[position]),
    }


def _synthetic_model_contract(edge_dim: int) -> dict:
    torch.manual_seed(42)
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 2, 0]], dtype=torch.long)
    edge_attr = torch.arange(4 * edge_dim, dtype=torch.float32).reshape(4, edge_dim)
    edge_attr = edge_attr / float(4 * edge_dim)
    data = Data(edge_index=edge_index, edge_attr=edge_attr, num_nodes=3)
    data.global_node_ids = torch.tensor([7, 11, 19], dtype=torch.long)
    data.timestamp = 5_000
    original_edge_index = data.edge_index.clone()
    original_edge_attr = data.edge_attr.clone()
    original_global_ids = data.global_node_ids.clone()
    factories = {
        "edge_mlp": lambda: SimpleMLP(edge_dim=edge_dim, hidden_dim=32, dropout=0.0),
        "edge_gru": lambda: EdgeGRU_Baseline_NoX(
            edge_dim=edge_dim,
            hidden_dim=32,
            dropout=0.0,
            memory_policy="carry_no_decay",
            time_scale_ms=5_000,
        ),
        "static_gnn": lambda: StaticGNN_Identity(
            node_dim=16,
            edge_dim=edge_dim,
            hidden_dim=32,
            dropout=0.0,
            identity_mode="current",
            window_ms=5_000,
        ),
        "st_gnn": lambda: ST_GNN_Identity(
            node_dim=16,
            edge_dim=edge_dim,
            hidden_dim=32,
            dropout=0.0,
            identity_mode="current",
            use_memory=True,
            use_topology=True,
            use_direct_edge_attr=True,
            memory_policy="carry_no_decay",
            time_scale_ms=5_000,
            window_ms=5_000,
        ),
        "st_gnn_without_gat": lambda: ST_GNN_Identity(
            node_dim=16,
            edge_dim=edge_dim,
            hidden_dim=32,
            dropout=0.0,
            identity_mode="current",
            use_memory=True,
            use_topology=False,
            use_direct_edge_attr=True,
            memory_policy="carry_no_decay",
            time_scale_ms=5_000,
            window_ms=5_000,
        ),
        "st_gnn_without_direct_edge_attr": lambda: ST_GNN_Identity(
            node_dim=16,
            edge_dim=edge_dim,
            hidden_dim=32,
            dropout=0.0,
            identity_mode="current",
            use_memory=True,
            use_topology=True,
            use_direct_edge_attr=False,
            memory_policy="carry_no_decay",
            time_scale_ms=5_000,
            window_ms=5_000,
        ),
    }
    results = {}
    for name in EXPECTED_VARIANTS:
        model = factories[name]()
        model.eval()
        temporal = bool(model.temporal)
        validate_temporal_configuration(
            model,
            temporal=temporal,
            temporal_memory_policy=model.temporal_memory_policy,
        )
        with torch.no_grad():
            logits = forward_graph(model, data)
        if logits.shape != (data.edge_attr.shape[0], 1) or not torch.isfinite(logits).all():
            raise ValueError(f"Synthetic one-logit-per-edge contract failed: {name}")
        state_created = bool(getattr(model, "node_memory", {})) if temporal else False
        if temporal:
            if not state_created:
                raise ValueError(f"Synthetic temporal state was not created: {name}")
            model.reset_memory()
            if bool(getattr(model, "node_memory", {})):
                raise ValueError(f"Synthetic temporal reset failed: {name}")
        if (
            not torch.equal(data.edge_index, original_edge_index)
            or not torch.equal(data.edge_attr, original_edge_attr)
            or not torch.equal(data.global_node_ids, original_global_ids)
        ):
            raise ValueError(f"A model mutated shared graph inputs: {name}")
        results[name] = {
            "status": "passed",
            "temporal": temporal,
            "temporal_state_created": state_created,
            "temporal_reset_verified": temporal,
            "logits": int(logits.shape[0]),
            "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        }
    return {
        "status": "passed",
        "synthetic_edges": int(data.edge_attr.shape[0]),
        "edge_feature_dimension": edge_dim,
        "shared_inputs_unchanged": True,
        "optimization_performed": False,
        "variants": results,
    }


def run_capture_graph_training_preflight(
    *,
    collection: CaptureGraphCollection,
    prepared_run_dir: str | Path,
    graph_input_audit_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    local_work_root: str | Path,
    batch_size: int = 250_000,
) -> dict:
    """Resolve and persist the inner split without fitting or scoring a model."""
    started = time.perf_counter()
    prepared_run_dir = Path(prepared_run_dir).expanduser().resolve()
    graph_input_audit_dir = Path(graph_input_audit_dir).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    local_work_root = Path(local_work_root).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Training-preflight output already exists: {output_dir}")
    if batch_size <= 0:
        raise ValueError("Training-preflight batch size must be positive.")
    config = load_training_preflight_config(config_path)
    bindings = config["bindings"]
    if (
        collection.expected_run_id != bindings["graph_materialization_run_id"]
        or collection.contract_path.name != bindings["graph_input_contract"]
        or prepared_run_dir.name != bindings["prepared_run_id"]
        or graph_input_audit_dir.name != bindings["graph_input_audit_run_id"]
    ):
        raise ValueError("Training-preflight inputs differ from their frozen run IDs.")
    input_audit = load_completed_capture_graph_input_audit(
        graph_input_audit_dir,
        collection,
        mode="FULL",
    )
    if input_audit.get("identity_audit", {}).get("status") != "passed":
        raise ValueError("A completed node-identity audit is required.")

    manifest_path = collection.root / "capture_experiment_v1.yaml"
    packet_schema_path = collection.root / "capture_packet_schema_v1.yaml"
    _, _, prepared_reports, prepared_paths = load_prepared_full_dev(
        prepared_run_dir,
        manifest_path,
        packet_schema_path,
    )
    provenance = collection.manifest["provenance"]
    observed_packet_hashes = {
        scenario: prepared_reports[scenario]["output_sha256"]
        for scenario in collection.scenarios
    }
    if observed_packet_hashes != provenance.get("prepared_packet_sha256"):
        raise ValueError("Prepared packet hashes differ from materialization provenance.")

    local_work_root.mkdir(parents=True, exist_ok=True)
    profiles = {}
    for scenario in collection.scenarios:
        source = prepared_paths[scenario]
        with tempfile.TemporaryDirectory(
            dir=local_work_root,
            prefix=f"{scenario}_preflight_",
        ) as temporary:
            local_path = Path(temporary) / source.name
            required_bytes = source.stat().st_size + 512 * 1024**2
            if shutil.disk_usage(local_work_root).free < required_bytes:
                raise OSError("Insufficient local space for preflight staging.")
            shutil.copyfile(source, local_path)
            if sha256_file(local_path) != prepared_reports[scenario]["output_sha256"]:
                raise IOError(f"Staged prepared checksum mismatch: {scenario}")
            profiles[scenario] = _profile_prepared_scenario(
                local_path,
                prepared_reports[scenario],
                batch_size=batch_size,
            )
        print(f"Profiled checkpoint candidates for {scenario}.", flush=True)

    split_config = config["inner_split"]
    minimums = split_config["minimums"]
    candidate_rows = []
    candidate_results: dict[float, dict[str, dict]] = {}
    for tail_fraction in split_config["candidate_validation_tail_fractions"]:
        scenario_results = {}
        for scenario in collection.scenarios:
            profile = profiles[scenario]
            boundary, target, maximum_shift = _nearest_iteration_safe_boundary(
                profile,
                tail_fraction=float(tail_fraction),
                maximum_shift_fraction=float(
                    split_config["maximum_boundary_shift_fraction"]
                ),
            )
            if boundary is None:
                result = {
                    "scenario": scenario,
                    "candidate_validation_tail_fraction": float(tail_fraction),
                    "target_boundary_window_index": target,
                    "maximum_boundary_shift_windows": maximum_shift,
                    "boundary_window_index": None,
                    "realized_validation_tail_fraction": None,
                    "admissible": False,
                    "rejection_reasons": ["no_iteration_safe_boundary_within_shift"],
                }
            else:
                counts = _partition_counts(profile, boundary)
                reasons = _admissibility_reasons(counts, minimums)
                result = {
                    "scenario": scenario,
                    "candidate_validation_tail_fraction": float(tail_fraction),
                    "target_boundary_window_index": target,
                    "maximum_boundary_shift_windows": maximum_shift,
                    "boundary_window_index": boundary,
                    "boundary_shift_windows": boundary - target,
                    "realized_validation_tail_fraction": (
                        profile["total_windows"] - boundary
                    )
                    / profile["total_windows"],
                    "admissible": not reasons,
                    "rejection_reasons": reasons,
                    **counts,
                }
            scenario_results[scenario] = result
            candidate_rows.append(result)
        candidate_results[float(tail_fraction)] = scenario_results

    selected_fraction = next(
        (
            float(fraction)
            for fraction in split_config["candidate_validation_tail_fractions"]
            if all(
                result["admissible"]
                for result in candidate_results[float(fraction)].values()
            )
        ),
        None,
    )
    selected_splits = {}
    split_summary_rows = []
    fold_totals = {
        fold: {
            "inner_train_edges": 0,
            "inner_validation_edges": 0,
            "inner_train_graphs": 0,
            "inner_validation_graphs": 0,
        }
        for fold in VALID_FOLDS
    }
    if selected_fraction is not None:
        for scenario, result in candidate_results[selected_fraction].items():
            training_folds = [
                fold
                for fold in VALID_FOLDS
                if collection.partition_for(fold, scenario) == "train"
            ]
            if len(training_folds) != 1:
                raise ValueError(f"Scenario does not train in exactly one fold: {scenario}")
            fold = training_folds[0]
            dataset = collection.scenario_dataset(
                fold,
                scenario,
                expected_partition="train",
                verify_shard_checksums=False,
            )
            alignment = _graph_window_alignment(
                dataset,
                int(result["boundary_window_index"]),
            )
            if (
                alignment["inner_train_edges"] != result["inner_train"]["packets"]
                or alignment["inner_validation_edges"]
                != result["inner_validation"]["packets"]
            ):
                raise ValueError(f"Prepared and graph split counts differ: {scenario}")
            selected = {
                "scenario": scenario,
                "fold": fold,
                "boundary_window_index": int(result["boundary_window_index"]),
                "target_validation_tail_fraction": selected_fraction,
                "realized_validation_tail_fraction": float(
                    result["realized_validation_tail_fraction"]
                ),
                "inner_train": result["inner_train"],
                "inner_validation": result["inner_validation"],
                **alignment,
            }
            selected_splits[scenario] = selected
            split_summary_rows.append(selected)
            for metric in fold_totals[fold]:
                fold_totals[fold][metric] += int(alignment[metric])

    model_contract_report = _synthetic_model_contract(
        int(config["model_contract"]["edge_feature_dimension"])
    )
    largest_graph_edges = max(
        int(result["maximum_edges_in_graph"])
        for fold_results in input_audit["scenario_results"].values()
        for result in fold_results.values()
    )
    resource_estimate = {
        "fold_totals": fold_totals,
        "largest_graph_edges": largest_graph_edges,
        "largest_graph_edge_attr_bytes": (
            largest_graph_edges
            * int(config["model_contract"]["edge_feature_dimension"])
            * np.dtype(np.float32).itemsize
        ),
        "compressed_graph_collection_bytes": int(
            collection.manifest["totals"]["compressed_shard_bytes"]
        ),
        "estimate_scope": "input_tensors_only_not_model_vram",
    }
    status = "review_required" if selected_fraction is not None else "blocked"
    report = {
        "report_version": REPORT_VERSION,
        "status": status,
        "materialization_run_id": collection.expected_run_id,
        "materialization_manifest_sha256": sha256_file(collection.manifest_path),
        "graph_input_audit_run_id": graph_input_audit_dir.name,
        "graph_input_audit_report_sha256": sha256_file(
            graph_input_audit_dir / "capture_graph_input_audit.json"
        ),
        "prepared_run_id": prepared_run_dir.name,
        "prepared_run_config_sha256": sha256_file(
            prepared_run_dir / "run_config.json"
        ),
        "preflight_config_sha256": sha256_file(config_path),
        "preflight_code_sha256": sha256_file(Path(__file__)),
        "selected_validation_tail_fraction": selected_fraction,
        "selected_splits": selected_splits,
        "fold_totals": fold_totals,
        "candidate_fractions": split_config[
            "candidate_validation_tail_fractions"
        ],
        "candidate_is_valid_for_every_scenario": {
            str(fraction): all(
                item["admissible"]
                for item in candidate_results[float(fraction)].values()
            )
            for fraction in split_config["candidate_validation_tail_fractions"]
        },
        "checkpointing": config["checkpointing"],
        "model_contract": model_contract_report,
        "held_out_scenarios_accessed": False,
        "model_training_performed": False,
        "model_scores_used_to_choose_split": False,
        "raw_endpoint_identity_is_model_input": False,
        "wall_seconds": float(time.perf_counter() - started),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
        },
    }

    output_dir.mkdir(parents=True)
    shutil.copyfile(config_path, output_dir / config_path.name)
    candidate_table = pd.json_normalize(candidate_rows, sep=".")
    candidate_table["rejection_reasons"] = candidate_table[
        "rejection_reasons"
    ].map(lambda values: json.dumps(values, separators=(",", ":")))
    candidate_table.to_csv(output_dir / "split_candidates.csv", index=False)
    pd.json_normalize(split_summary_rows, sep=".").to_csv(
        output_dir / "split_summary.csv",
        index=False,
    )
    write_json(output_dir / "inner_split_manifest.json", {
        "status": status,
        "selected_validation_tail_fraction": selected_fraction,
        "splits": selected_splits,
    })
    write_json(output_dir / "model_contract_report.json", model_contract_report)
    write_json(output_dir / "resource_estimate.json", resource_estimate)
    resolved_config = dict(config)
    resolved_config["resolved_inner_split"] = {
        "status": status,
        "selected_validation_tail_fraction": selected_fraction,
        "splits": selected_splits,
    }
    (output_dir / "resolved_training_config.yaml").write_text(
        yaml.safe_dump(resolved_config, sort_keys=False),
        encoding="utf-8",
    )
    write_json(output_dir / "training_preflight_report.json", report)
    artifact_names = [
        config_path.name,
        "split_candidates.csv",
        "split_summary.csv",
        "inner_split_manifest.json",
        "model_contract_report.json",
        "resource_estimate.json",
        "resolved_training_config.yaml",
        "training_preflight_report.json",
    ]
    checksums = {name: sha256_file(output_dir / name) for name in artifact_names}
    write_json(output_dir / "artifact_checksums.json", checksums)
    write_json(output_dir / "run_status.json", {
        "complete": True,
        "status": status,
        "artifact_checksums_sha256": sha256_file(
            output_dir / "artifact_checksums.json"
        ),
        "training_preflight_report_sha256": checksums[
            "training_preflight_report.json"
        ],
    })
    return report


def load_completed_training_preflight(
    output_dir: str | Path,
    *,
    collection: CaptureGraphCollection,
    config_path: str | Path,
) -> dict:
    """Load one immutable completed preflight and verify every artifact hash."""
    output_dir = Path(output_dir).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    status = _load_json(output_dir / "run_status.json", "training-preflight status")
    checksums = _load_json(
        output_dir / "artifact_checksums.json",
        "training-preflight artifact checksums",
    )
    report = _load_json(
        output_dir / "training_preflight_report.json",
        "training-preflight report",
    )
    config = load_training_preflight_config(config_path)
    bindings = config["bindings"]
    if (
        status.get("complete") is not True
        or status.get("artifact_checksums_sha256")
        != sha256_file(output_dir / "artifact_checksums.json")
        or report.get("materialization_run_id") != collection.expected_run_id
        or report.get("materialization_manifest_sha256")
        != sha256_file(collection.manifest_path)
        or report.get("preflight_config_sha256") != sha256_file(config_path)
        or report.get("preflight_code_sha256") != sha256_file(Path(__file__))
        or report.get("graph_input_audit_run_id")
        != bindings["graph_input_audit_run_id"]
        or report.get("prepared_run_id") != bindings["prepared_run_id"]
    ):
        raise ValueError("The completed training preflight has a different binding.")
    for name, expected in checksums.items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Training-preflight artifact changed: {name}")
    return report


def save_training_preflight_decision(
    output_dir: str | Path,
    *,
    approved: bool,
    review_note: str,
) -> dict:
    """Record the manual decision that authorizes or rejects runner implementation."""
    output_dir = Path(output_dir).expanduser().resolve()
    decision_path = output_dir / "training_preflight_decision.json"
    if decision_path.exists():
        raise FileExistsError(f"Training-preflight decision already exists: {decision_path}")
    if not isinstance(approved, bool) or not review_note.strip():
        raise ValueError("The decision requires a boolean and a non-empty review note.")
    report_path = output_dir / "training_preflight_report.json"
    report = _load_json(report_path, "training-preflight report")
    if approved and report.get("status") != "review_required":
        raise ValueError("A blocked training preflight cannot be approved.")
    decision = {
        "approved": approved,
        "review_note": review_note.strip(),
        "training_preflight_report_sha256": sha256_file(report_path),
        "selected_validation_tail_fraction": report.get(
            "selected_validation_tail_fraction"
        ),
        "training_authorized": False,
        "next_action": (
            "freeze_the_full_training_configuration_and_implement_the_runner"
            if approved
            else "revise_the_preflight_contract_before_training"
        ),
    }
    write_json(decision_path, decision)
    return decision
