"""Audit and freeze bounded chronological validation blocks for cAPTure Stage 2."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import shutil
import tempfile
import time

import numpy as np
import pandas as pd
import torch
import yaml

from .capture_data import sha256_file, write_json
from .capture_feature_profile import load_prepared_full_dev
from .capture_graph_dataset import (
    CaptureGraphCollection,
    load_completed_capture_graph_input_audit,
)
from .capture_graph_training_preflight import (
    EXPECTED_VARIANTS,
    _admissibility_reasons,
    _profile_prepared_scenario,
    _synthetic_model_contract,
)


REPORT_VERSION = 2
VALID_FOLDS = ("A", "B")
CANDIDATE_BLOCK_FRACTIONS = [0.20, 0.25]
BASE_PREFLIGHT_MODULE = Path(__file__).with_name("capture_graph_training_preflight.py")
EMPTY_SPLIT_SUMMARY_COLUMNS = [
    "scenario",
    "fold",
    "validation_start_window_index",
    "validation_end_window_index_exclusive",
    "selected_validation_block_fraction",
    "realized_validation_block_fraction",
]


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected {label} to contain a JSON object: {path}")
    return value


def load_training_preflight_v2_config(path: str | Path) -> dict:
    """Load and strictly validate the frozen block-selection contract."""
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The training-preflight configuration must be a mapping.")
    if (
        config.get("preflight_version") != 2
        or config.get("scope") != "development_only"
        or config.get("stage") != "stage2_training_preflight"
    ):
        raise ValueError("Unsupported training-preflight v2 configuration.")

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
    required_split_values = {
        "basis": "latest_admissible_chronological_block_per_scenario",
        "training_condition": "window_index_less_than_validation_start",
        "validation_condition": (
            "validation_start_less_than_or_equal_to_window_index_less_than_validation_end"
        ),
        "post_validation_condition": (
            "window_index_greater_than_or_equal_to_validation_end"
        ),
        "candidate_validation_block_fractions": CANDIDATE_BLOCK_FRACTIONS,
        "choose_smallest_fraction_then_latest_admissible_block": True,
        "never_split_iteration_key": ["attack_step", "sequence_id"],
        "post_validation_data_used_during_epoch_selection": False,
    }
    for key, expected in required_split_values.items():
        if split.get(key) != expected:
            raise ValueError(f"The inner-split contract changed: {key}")
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
    if split.get("minimums") != expected_minimums:
        raise ValueError("The training-preflight admissibility minimums changed.")

    expected_checkpointing = {
        "selection_unit": "model_and_fold",
        "metric": "unweighted_mean_scenario_average_precision",
        "maximum_epochs": 60,
        "minimum_epochs": 10,
        "patience_epochs": 10,
        "minimum_absolute_improvement": 0.0001,
        "restore_best_epoch": True,
        "preprocessing_fit_scope": "inner_train_prefixes_only",
        "training_weight_fit_scope": "inner_train_prefixes_only",
        "inner_validation_only": True,
        "outer_validation_forbidden": True,
        "thresholds_selected_during_checkpointing": False,
        "scenario_order": "declared_fold_order",
        "graph_order": "chronological",
        "shuffle": False,
        "reset_temporal_state_at_each_scenario_boundary": True,
        "reset_temporal_state_at_each_epoch": True,
        "inner_validation_starts_with_empty_temporal_state": True,
    }
    if config.get("checkpointing") != expected_checkpointing:
        raise ValueError("The checkpointing v2 contract changed.")

    expected_refit = {
        "selection_output": "best_epoch_count",
        "one_epoch_count_per_model_and_fold": True,
        "reinitialize_model": True,
        "reuse_declared_seed": True,
        "use_complete_fold_training_scenarios": True,
        "preprocessing_fit_scope": "complete_fold_training_scenarios",
        "training_weight_fit_scope": "complete_fold_training_scenarios",
        "include_inner_validation_data": True,
        "include_post_validation_data": True,
        "epochs": "selected_best_epoch_count",
        "early_stopping": False,
        "checkpoint_selection": False,
        "outer_validation_access_during_refit": False,
        "evaluate_outer_validation_after_refit": True,
    }
    if config.get("final_refit") != expected_refit:
        raise ValueError("The final-refit contract changed.")

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
    expected_prohibitions = {
        "held_out_scenarios_accessed": False,
        "model_training_performed": False,
        "model_scores_used_to_choose_split": False,
        "raw_endpoint_identity_is_model_input": False,
    }
    if config.get("prohibitions") != expected_prohibitions:
        raise ValueError("The training-preflight prohibitions changed.")
    return config


def _boundary_splits_iteration(profile: dict, boundary: int) -> bool:
    return any(
        start < boundary <= stop
        for start, stop, _ in profile["iteration_spans"].values()
    )


def _block_partition_counts(profile: dict, start: int, end: int) -> dict:
    packets = profile["packet_counts"]
    attacks = profile["attack_counts"]
    total_windows = int(profile["total_windows"])
    if not 0 < start < end <= total_windows:
        raise ValueError("Invalid chronological validation block.")

    nonempty = packets > 0
    attack_windows = attacks > 0
    benign_only = nonempty & ~attack_windows
    train_iterations = []
    validation_iterations = []
    post_validation_iterations = []
    split_iterations = []
    for (step, sequence), (first, last, attack_packets) in profile[
        "iteration_spans"
    ].items():
        item = {
            "attack_step": step,
            "sequence_id": sequence,
            "first_window_index": first,
            "last_window_index": last,
            "attack_packets": attack_packets,
        }
        if last < start:
            train_iterations.append(item)
        elif first >= start and last < end:
            validation_iterations.append(item)
        elif first >= end:
            post_validation_iterations.append(item)
        else:
            split_iterations.append(item)

    def summarize(left: int, right: int, iterations: list[dict]) -> dict:
        partition_slice = slice(left, right)
        partition_packets = int(packets[partition_slice].sum())
        partition_attacks = int(attacks[partition_slice].sum())
        return {
            "wall_clock_windows": right - left,
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
        "inner_train": summarize(0, start, train_iterations),
        "inner_validation": summarize(start, end, validation_iterations),
        "post_validation": summarize(
            end,
            total_windows,
            post_validation_iterations,
        ),
        "split_iterations": split_iterations,
    }


def _latest_admissible_block(
    profile: dict,
    *,
    block_fraction: float,
    minimums: dict,
) -> dict:
    total_windows = int(profile["total_windows"])
    block_windows = max(1, int(round(total_windows * block_fraction)))
    boundaries_examined = 0
    iteration_safe_blocks = 0
    latest_safe_result = None
    for end in range(total_windows, block_windows, -1):
        start = end - block_windows
        boundaries_examined += 1
        if (
            _boundary_splits_iteration(profile, start)
            or _boundary_splits_iteration(profile, end)
        ):
            continue
        iteration_safe_blocks += 1
        counts = _block_partition_counts(profile, start, end)
        reasons = _admissibility_reasons(counts, minimums)
        candidate = {
            "candidate_validation_block_fraction": float(block_fraction),
            "target_validation_block_windows": block_windows,
            "validation_start_window_index": start,
            "validation_end_window_index_exclusive": end,
            "realized_validation_block_fraction": block_windows / total_windows,
            "post_validation_wall_clock_windows": total_windows - end,
            "admissible": not reasons,
            "rejection_reasons": reasons,
            **counts,
        }
        if latest_safe_result is None:
            latest_safe_result = candidate
        if not reasons:
            candidate["boundaries_examined"] = boundaries_examined
            candidate["iteration_safe_blocks_examined"] = iteration_safe_blocks
            return candidate

    if latest_safe_result is None:
        return {
            "candidate_validation_block_fraction": float(block_fraction),
            "target_validation_block_windows": block_windows,
            "validation_start_window_index": None,
            "validation_end_window_index_exclusive": None,
            "realized_validation_block_fraction": None,
            "post_validation_wall_clock_windows": None,
            "admissible": False,
            "rejection_reasons": ["no_iteration_safe_block_found"],
            "boundaries_examined": boundaries_examined,
            "iteration_safe_blocks_examined": iteration_safe_blocks,
        }
    latest_safe_result["rejection_reasons"] = [
        "no_admissible_block_found",
        *latest_safe_result["rejection_reasons"],
    ]
    latest_safe_result["boundaries_examined"] = boundaries_examined
    latest_safe_result["iteration_safe_blocks_examined"] = iteration_safe_blocks
    return latest_safe_result


def _graph_block_alignment(dataset, start: int, end: int) -> dict:
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
    start_position = int(np.searchsorted(windows, start, side="left"))
    end_position = int(np.searchsorted(windows, end, side="left"))
    result = {
        "inner_train_graphs": start_position,
        "inner_validation_graphs": end_position - start_position,
        "post_validation_graphs": int(len(windows) - end_position),
        "inner_train_edges": int(edges[:start_position].sum()),
        "inner_validation_edges": int(edges[start_position:end_position].sum()),
        "post_validation_edges": int(edges[end_position:].sum()),
        "complete_refit_graphs": int(len(windows)),
        "complete_refit_edges": int(edges.sum()),
    }
    if result["inner_validation_graphs"] <= 0:
        raise ValueError(f"Validation block has no graphs: {dataset.scenario}")
    return result


def run_capture_graph_training_preflight_v2(
    *,
    collection: CaptureGraphCollection,
    prepared_run_dir: str | Path,
    graph_input_audit_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
    local_work_root: str | Path,
    batch_size: int = 250_000,
) -> dict:
    """Resolve bounded chronological blocks without fitting or scoring a model."""
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

    config = load_training_preflight_v2_config(config_path)
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
    observed_packet_hashes = {
        scenario: prepared_reports[scenario]["output_sha256"]
        for scenario in collection.scenarios
    }
    if observed_packet_hashes != collection.manifest["provenance"].get(
        "prepared_packet_sha256"
    ):
        raise ValueError("Prepared packet hashes differ from materialization provenance.")

    local_work_root.mkdir(parents=True, exist_ok=True)
    profiles = {}
    for scenario in collection.scenarios:
        source = prepared_paths[scenario]
        with tempfile.TemporaryDirectory(
            dir=local_work_root,
            prefix=f"{scenario}_preflight_v2_",
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
        print(f"Profiled validation blocks for {scenario}.", flush=True)

    split_config = config["inner_split"]
    minimums = split_config["minimums"]
    candidate_rows = []
    candidate_results: dict[str, dict[float, dict]] = {}
    selected_candidates = {}
    for scenario in collection.scenarios:
        scenario_results = {}
        for block_fraction in split_config["candidate_validation_block_fractions"]:
            result = _latest_admissible_block(
                profiles[scenario],
                block_fraction=float(block_fraction),
                minimums=minimums,
            )
            result = {"scenario": scenario, **result}
            scenario_results[float(block_fraction)] = result
            candidate_rows.append(result)
        candidate_results[scenario] = scenario_results
        selected = next(
            (
                scenario_results[float(fraction)]
                for fraction in split_config["candidate_validation_block_fractions"]
                if scenario_results[float(fraction)]["admissible"]
            ),
            None,
        )
        if selected is not None:
            selected_candidates[scenario] = selected

    selected_splits = {}
    split_summary_rows = []
    fold_totals = {
        fold: {
            "selection_inner_train_edges": 0,
            "selection_inner_validation_edges": 0,
            "selection_excluded_post_validation_edges": 0,
            "complete_refit_edges": 0,
            "complete_refit_graphs": 0,
        }
        for fold in VALID_FOLDS
    }
    if len(selected_candidates) == len(collection.scenarios):
        for scenario, result in selected_candidates.items():
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
            alignment = _graph_block_alignment(
                dataset,
                int(result["validation_start_window_index"]),
                int(result["validation_end_window_index_exclusive"]),
            )
            expected_partition_edges = {
                "inner_train_edges": result["inner_train"]["packets"],
                "inner_validation_edges": result["inner_validation"]["packets"],
                "post_validation_edges": result["post_validation"]["packets"],
            }
            if any(
                alignment[name] != expected
                for name, expected in expected_partition_edges.items()
            ):
                raise ValueError(f"Prepared and graph block counts differ: {scenario}")
            selected = {
                "scenario": scenario,
                "fold": fold,
                "validation_start_window_index": int(
                    result["validation_start_window_index"]
                ),
                "validation_end_window_index_exclusive": int(
                    result["validation_end_window_index_exclusive"]
                ),
                "selected_validation_block_fraction": float(
                    result["candidate_validation_block_fraction"]
                ),
                "realized_validation_block_fraction": float(
                    result["realized_validation_block_fraction"]
                ),
                "inner_train": result["inner_train"],
                "inner_validation": result["inner_validation"],
                "post_validation": result["post_validation"],
                **alignment,
            }
            selected_splits[scenario] = selected
            split_summary_rows.append(selected)
            fold_totals[fold]["selection_inner_train_edges"] += alignment[
                "inner_train_edges"
            ]
            fold_totals[fold]["selection_inner_validation_edges"] += alignment[
                "inner_validation_edges"
            ]
            fold_totals[fold]["selection_excluded_post_validation_edges"] += alignment[
                "post_validation_edges"
            ]
            fold_totals[fold]["complete_refit_edges"] += alignment[
                "complete_refit_edges"
            ]
            fold_totals[fold]["complete_refit_graphs"] += alignment[
                "complete_refit_graphs"
            ]

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
    status = (
        "review_required"
        if len(selected_splits) == len(collection.scenarios)
        else "blocked"
    )
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
        "preflight_v2_code_sha256": sha256_file(Path(__file__)),
        "base_preflight_code_sha256": sha256_file(BASE_PREFLIGHT_MODULE),
        "selected_validation_block_fraction_by_scenario": {
            scenario: split["selected_validation_block_fraction"]
            for scenario, split in selected_splits.items()
        },
        "selected_splits": selected_splits,
        "fold_totals": fold_totals,
        "checkpointing": config["checkpointing"],
        "final_refit": config["final_refit"],
        "model_contract": model_contract_report,
        "held_out_scenarios_accessed": False,
        "model_training_performed": False,
        "model_scores_used_to_choose_split": False,
        "raw_endpoint_identity_is_model_input": False,
        "selection_preprocessing_fitted": False,
        "selection_features_materialized": False,
        "final_refit_performed": False,
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
    split_summary_table = pd.json_normalize(split_summary_rows, sep=".")
    if split_summary_table.empty and not len(split_summary_table.columns):
        split_summary_table = pd.DataFrame(columns=EMPTY_SPLIT_SUMMARY_COLUMNS)
    split_summary_table.to_csv(output_dir / "split_summary.csv", index=False)
    write_json(output_dir / "inner_split_manifest.json", {
        "status": status,
        "selection_rule": (
            "smallest_declared_fraction_then_latest_admissible_block_per_scenario"
        ),
        "splits": selected_splits,
    })
    write_json(output_dir / "model_contract_report.json", model_contract_report)
    write_json(output_dir / "resource_estimate.json", resource_estimate)
    resolved_config = dict(config)
    resolved_config["resolved_inner_split"] = {
        "status": status,
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


def load_completed_training_preflight_v2(
    output_dir: str | Path,
    *,
    collection: CaptureGraphCollection,
    config_path: str | Path,
) -> dict:
    """Load a completed v2 preflight and verify its immutable artifacts."""
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
    config = load_training_preflight_v2_config(config_path)
    bindings = config["bindings"]
    if (
        status.get("complete") is not True
        or status.get("artifact_checksums_sha256")
        != sha256_file(output_dir / "artifact_checksums.json")
        or report.get("report_version") != REPORT_VERSION
        or report.get("materialization_run_id") != collection.expected_run_id
        or report.get("materialization_manifest_sha256")
        != sha256_file(collection.manifest_path)
        or report.get("preflight_config_sha256") != sha256_file(config_path)
        or report.get("preflight_v2_code_sha256") != sha256_file(Path(__file__))
        or report.get("base_preflight_code_sha256")
        != sha256_file(BASE_PREFLIGHT_MODULE)
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


def save_training_preflight_v2_decision(
    output_dir: str | Path,
    *,
    approved: bool,
    review_note: str,
) -> dict:
    """Record the decision that permits or rejects training-runner implementation."""
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
        "selected_validation_block_fraction_by_scenario": report.get(
            "selected_validation_block_fraction_by_scenario"
        ),
        "checkpointing": report.get("checkpointing"),
        "final_refit": report.get("final_refit"),
        "training_authorized": False,
        "next_action": (
            "freeze_the_full_training_configuration_and_implement_the_runner"
            if approved
            else "revise_the_preflight_contract_before_training"
        ),
    }
    write_json(decision_path, decision)
    return decision
