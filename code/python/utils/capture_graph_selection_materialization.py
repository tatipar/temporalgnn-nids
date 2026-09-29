"""Build inner-training-only feature views for cAPTure graph epoch selection."""

from __future__ import annotations

import json
from pathlib import Path
import platform
import shutil
import tempfile
import time
from typing import Iterator

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from .capture_data import sha256_file, write_json
from .capture_feature_profile import (
    load_prepared_full_dev,
    load_preprocessing_schema,
    preprocessing_schema_sha256,
)
from .capture_graph_dataset import CaptureGraphCollection
from .capture_graph_materialization import (
    _materialize_scenario_fold,
    _validate_scenario_fold,
    _validate_shard,
)
from .capture_graph_training_preflight_v2 import (
    load_completed_training_preflight_v2,
)
from .capture_preprocess import CaptureFoldPreprocessor


REPORT_VERSION = 1
VALID_FOLDS = ("A", "B")
MODULE_PATH = Path(__file__)
GRAPH_MATERIALIZATION_MODULE = MODULE_PATH.with_name(
    "capture_graph_materialization.py"
)
PREPROCESSING_MODULE = MODULE_PATH.with_name("capture_preprocess.py")


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected {label} to contain a JSON object: {path}")
    return value


def load_selection_materialization_config(path: str | Path) -> dict:
    """Load and strictly validate the frozen selection-feature contract."""
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The selection-materialization configuration must be a mapping.")
    if (
        config.get("selection_materialization_version") != 1
        or config.get("scope") != "development_only"
        or config.get("stage") != "stage2_selection_feature_materialization"
    ):
        raise ValueError("Unsupported selection-materialization configuration.")
    bindings = config.get("bindings", {})
    required_bindings = {
        "prepared_run_id",
        "graph_materialization_run_id",
        "graph_input_audit_run_id",
        "training_preflight_run_id",
        "training_preflight_report_sha256",
        "graph_input_contract",
        "preprocessing_contract",
    }
    if set(bindings) != required_bindings:
        raise ValueError("The selection-materialization bindings changed.")
    if len(str(bindings["training_preflight_report_sha256"])) != 64:
        raise ValueError("The training-preflight report hash is invalid.")
    if config.get("preprocessing") != {
        "fit_scope": "inner_train_prefixes_only",
        "one_preprocessor_per_fold": True,
        "feature_dimension": 103,
        "validation_block_used_for_fit": False,
        "post_validation_suffix_used_for_fit": False,
        "fit_batch_size": 100000,
    }:
        raise ValueError("The inner-preprocessing contract changed.")
    if config.get("materialization") != {
        "scenarios": "fold_training_only",
        "include_complete_scenario_for_alignment": True,
        "training_runner_must_slice_using_preflight_boundaries": True,
        "outer_validation_scenarios_materialized": False,
        "held_out_scenarios_accessed": False,
        "training_performed": False,
    }:
        raise ValueError("The selection materialization scope changed.")
    required_windows = {
        "duration_seconds": 5,
        "type": "fixed_non_overlapping",
        "interval": "left_closed_right_open",
        "origin_rule": "first_packet_timestamp_per_scenario",
        "decision_time": "window_end",
        "empty_windows": "do_not_materialize_preserve_index_gaps",
    }
    if config.get("windows") != required_windows:
        raise ValueError("The selection window contract changed.")
    required_storage = {
        "format": "compressed_numpy_npz_shards",
        "compression": "zip_deflate",
        "maximum_edges_per_shard": 250000,
        "maximum_graphs_per_shard": 256,
        "edge_index_dtype": "int32",
        "global_node_id_dtype": "int32",
        "pointer_dtype": "int64",
        "source_row_id_dtype": "int64",
        "window_time_dtype": "int64",
        "write_one_file_per_window": False,
    }
    if config.get("storage") != required_storage:
        raise ValueError("The selection storage contract changed.")
    if config.get("required_invariants") != {
        "topology_matches_full_fold_materialization": True,
        "targets_match_full_fold_materialization": True,
        "source_row_ids_match_full_fold_materialization": True,
        "node_mapping_matches_full_fold_materialization": True,
        "graph_and_edge_counts_match_prepared_packets": True,
        "transformed_values_are_finite": True,
        "raw_endpoint_identity_in_model_artifacts": False,
    }:
        raise ValueError("The selection-materialization invariants changed.")
    return config


def _prefix_training_frames(
    *,
    scenarios: list[str],
    packet_paths: dict[str, Path],
    prepared_reports: dict[str, dict],
    selected_splits: dict[str, dict],
    required_columns: list[str],
    local_work_root: Path,
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    columns = list(dict.fromkeys([*required_columns, "source_row_id"]))
    for scenario in scenarios:
        required_rows = int(selected_splits[scenario]["inner_train_edges"])
        if required_rows != int(selected_splits[scenario]["inner_train"]["packets"]):
            raise ValueError(f"Inner-training packet counts disagree: {scenario}")
        source = packet_paths[scenario]
        with tempfile.TemporaryDirectory(
            dir=local_work_root,
            prefix=f"selection_fit_{scenario}_",
        ) as temporary:
            if shutil.disk_usage(local_work_root).free < (
                source.stat().st_size + 512 * 1024**2
            ):
                raise OSError(
                    f"Insufficient local space to fit from {scenario}."
                )
            local_path = Path(temporary) / source.name
            shutil.copyfile(source, local_path)
            if sha256_file(local_path) != prepared_reports[scenario]["output_sha256"]:
                raise IOError(f"Staged prepared checksum mismatch: {scenario}")
            remaining = required_rows
            expected_row = 0
            parquet = pq.ParquetFile(local_path)
            for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
                if remaining == 0:
                    break
                frame = batch.to_pandas()
                take = min(remaining, len(frame))
                frame = frame.iloc[:take].copy()
                observed_rows = frame["source_row_id"].to_numpy()
                expected_rows = range(expected_row, expected_row + take)
                if observed_rows.tolist() != list(expected_rows):
                    raise ValueError(f"Prepared prefix rows changed: {scenario}")
                expected_row += take
                remaining -= take
                yield frame.loc[:, required_columns]
            if remaining != 0 or expected_row != required_rows:
                raise ValueError(f"Prepared inner-training prefix is incomplete: {scenario}")


def _load_approved_preflight(
    *,
    preflight_dir: Path,
    preflight_config_path: Path,
    collection: CaptureGraphCollection,
    selection_config: dict,
) -> tuple[dict, dict]:
    bindings = selection_config["bindings"]
    if preflight_dir.name != bindings["training_preflight_run_id"]:
        raise ValueError("The training-preflight run ID changed.")
    report_path = preflight_dir / "training_preflight_report.json"
    if sha256_file(report_path) != bindings["training_preflight_report_sha256"]:
        raise ValueError("The training-preflight report hash changed.")
    report = load_completed_training_preflight_v2(
        preflight_dir,
        collection=collection,
        config_path=preflight_config_path,
    )
    decision = _load_json(
        preflight_dir / "training_preflight_decision.json",
        "training-preflight decision",
    )
    if (
        report.get("status") != "review_required"
        or decision.get("approved") is not True
        or decision.get("training_preflight_report_sha256")
        != bindings["training_preflight_report_sha256"]
        or decision.get("next_action")
        != "freeze_the_full_training_configuration_and_implement_the_runner"
        or decision.get("training_authorized") is not False
    ):
        raise ValueError("The training preflight is not approved for implementation.")
    if set(report.get("selected_splits", {})) != set(collection.scenarios):
        raise ValueError("The approved preflight does not contain every scenario split.")
    return report, decision


def _preprocessor_artifact(
    preprocessor: CaptureFoldPreprocessor,
    *,
    fold: str,
    scenarios: list[str],
    selected_splits: dict[str, dict],
    preflight_report_sha256: str,
) -> dict:
    artifact = preprocessor.to_dict(fold=fold, training_scenarios=scenarios)
    artifact.update({
        "fit_scope": "inner_train_prefixes_only",
        "training_preflight_report_sha256": preflight_report_sha256,
        "inner_train_boundaries": {
            scenario: {
                "validation_start_window_index": int(
                    selected_splits[scenario]["validation_start_window_index"]
                ),
                "training_rows": int(selected_splits[scenario]["inner_train_edges"]),
            }
            for scenario in scenarios
        },
        "validation_block_used_for_fit": False,
        "post_validation_suffix_used_for_fit": False,
    })
    return artifact


def _load_or_fit_preprocessor(
    *,
    fold: str,
    scenarios: list[str],
    schema: dict,
    packet_paths: dict[str, Path],
    prepared_reports: dict[str, dict],
    selected_splits: dict[str, dict],
    output_dir: Path,
    local_work_root: Path,
    batch_size: int,
    preflight_report_sha256: str,
) -> tuple[CaptureFoldPreprocessor, str, dict]:
    path = output_dir / f"fold_{fold}_inner_preprocessor.json"
    if path.exists():
        artifact = _load_json(path, f"fold {fold} inner preprocessor")
        expected = _preprocessor_artifact(
            CaptureFoldPreprocessor.from_dict(schema, artifact),
            fold=fold,
            scenarios=scenarios,
            selected_splits=selected_splits,
            preflight_report_sha256=preflight_report_sha256,
        )
        if artifact != expected:
            raise ValueError(f"The existing fold {fold} inner preprocessor changed.")
        return CaptureFoldPreprocessor.from_dict(schema, artifact), sha256_file(path), artifact

    preprocessor = CaptureFoldPreprocessor(schema)
    frames = _prefix_training_frames(
        scenarios=scenarios,
        packet_paths=packet_paths,
        prepared_reports=prepared_reports,
        selected_splits=selected_splits,
        required_columns=preprocessor.required_columns,
        local_work_root=local_work_root,
        batch_size=batch_size,
    )
    preprocessor.fit(frames)
    expected_rows = sum(
        int(selected_splits[scenario]["inner_train_edges"])
        for scenario in scenarios
    )
    if preprocessor.training_rows != expected_rows:
        raise ValueError(f"Fold {fold} inner preprocessor row count changed.")
    artifact = _preprocessor_artifact(
        preprocessor,
        fold=fold,
        scenarios=scenarios,
        selected_splits=selected_splits,
        preflight_report_sha256=preflight_report_sha256,
    )
    write_json(path, artifact)
    return preprocessor, sha256_file(path), artifact


def run_capture_graph_selection_materialization(
    *,
    collection: CaptureGraphCollection,
    manifest_path: str | Path,
    packet_schema_path: str | Path,
    preprocessing_schema_path: str | Path,
    selection_config_path: str | Path,
    preflight_config_path: str | Path,
    prepared_run_dir: str | Path,
    graph_input_audit_dir: str | Path,
    preflight_dir: str | Path,
    output_dir: str | Path,
    local_work_root: str | Path,
) -> dict:
    """Fit inner preprocessors and materialize aligned training-scenario graphs."""
    started = time.perf_counter()
    manifest_path = Path(manifest_path).expanduser().resolve()
    packet_schema_path = Path(packet_schema_path).expanduser().resolve()
    preprocessing_schema_path = Path(preprocessing_schema_path).expanduser().resolve()
    selection_config_path = Path(selection_config_path).expanduser().resolve()
    preflight_config_path = Path(preflight_config_path).expanduser().resolve()
    prepared_run_dir = Path(prepared_run_dir).expanduser().resolve()
    graph_input_audit_dir = Path(graph_input_audit_dir).expanduser().resolve()
    preflight_dir = Path(preflight_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    local_work_root = Path(local_work_root).expanduser().resolve()
    config = load_selection_materialization_config(selection_config_path)
    bindings = config["bindings"]
    if (
        collection.expected_run_id != bindings["graph_materialization_run_id"]
        or collection.contract_path.name != bindings["graph_input_contract"]
        or prepared_run_dir.name != bindings["prepared_run_id"]
        or graph_input_audit_dir.name != bindings["graph_input_audit_run_id"]
        or preprocessing_schema_path.name != bindings["preprocessing_contract"]
    ):
        raise ValueError("Selection-materialization inputs differ from their bindings.")
    input_audit_report = graph_input_audit_dir / "capture_graph_input_audit.json"
    if not input_audit_report.is_file():
        raise FileNotFoundError("The completed graph-input audit is required.")
    preflight, decision = _load_approved_preflight(
        preflight_dir=preflight_dir,
        preflight_config_path=preflight_config_path,
        collection=collection,
        selection_config=config,
    )
    if sha256_file(input_audit_report) != preflight.get(
        "graph_input_audit_report_sha256"
    ):
        raise ValueError("The graph-input audit differs from the approved preflight.")
    manifest, packet_schema, prepared_reports, packet_paths = load_prepared_full_dev(
        prepared_run_dir,
        manifest_path,
        packet_schema_path,
    )
    schema = load_preprocessing_schema(preprocessing_schema_path, packet_schema)
    selected_splits = preflight["selected_splits"]
    batch_size = int(config["preprocessing"]["fit_batch_size"])
    local_work_root.mkdir(parents=True, exist_ok=True)

    run_config = {
        "report_version": REPORT_VERSION,
        "prepared_run_id": prepared_run_dir.name,
        "graph_materialization_run_id": collection.expected_run_id,
        "graph_input_audit_run_id": graph_input_audit_dir.name,
        "training_preflight_run_id": preflight_dir.name,
        "training_preflight_report_sha256": sha256_file(
            preflight_dir / "training_preflight_report.json"
        ),
        "training_preflight_decision_sha256": sha256_file(
            preflight_dir / "training_preflight_decision.json"
        ),
        "selection_config_sha256": sha256_file(selection_config_path),
        "preflight_config_sha256": sha256_file(preflight_config_path),
        "manifest_sha256": sha256_file(manifest_path),
        "packet_schema_sha256": sha256_file(packet_schema_path),
        "preprocessing_schema_sha256": sha256_file(preprocessing_schema_path),
        "preprocessing_contract_sha256": preprocessing_schema_sha256(schema),
        "graph_materialization_manifest_sha256": sha256_file(
            collection.manifest_path
        ),
        "graph_input_audit_report_sha256": sha256_file(input_audit_report),
        "code_sha256": sha256_file(MODULE_PATH),
        "graph_materialization_code_sha256": sha256_file(
            GRAPH_MATERIALIZATION_MODULE
        ),
        "preprocessing_code_sha256": sha256_file(PREPROCESSING_MODULE),
        "fit_batch_size": batch_size,
        "folds": {
            fold: collection.scenarios_for(fold, "train") for fold in VALID_FOLDS
        },
        "selected_splits": selected_splits,
    }
    if output_dir.exists():
        existing = _load_json(output_dir / "run_config.json", "run configuration")
        if existing != run_config:
            raise FileExistsError(
                "The selection-materialization output has a different contract. "
                "Use a new run ID."
            )
    else:
        output_dir.mkdir(parents=True)
        write_json(output_dir / "run_config.json", run_config)
        for path in (
            selection_config_path,
            preflight_config_path,
            preprocessing_schema_path,
        ):
            shutil.copyfile(path, output_dir / path.name)

    fold_reports = {}
    config_sha256 = sha256_file(selection_config_path)
    preprocessing_contract_sha256 = preprocessing_schema_sha256(schema)
    for fold in VALID_FOLDS:
        scenarios = collection.scenarios_for(fold, "train")
        preprocessor, preprocessor_sha256, artifact = _load_or_fit_preprocessor(
            fold=fold,
            scenarios=scenarios,
            schema=schema,
            packet_paths=packet_paths,
            prepared_reports=prepared_reports,
            selected_splits=selected_splits,
            output_dir=output_dir,
            local_work_root=local_work_root,
            batch_size=batch_size,
            preflight_report_sha256=bindings["training_preflight_report_sha256"],
        )
        scenario_reports = {}
        invariants = {}
        for scenario in scenarios:
            durable_dir = output_dir / f"fold_{fold}" / scenario
            source_sha256 = prepared_reports[scenario]["output_sha256"]
            if durable_dir.exists():
                report = _validate_scenario_fold(
                    durable_dir,
                    scenario=scenario,
                    fold=fold,
                    source_sha256=source_sha256,
                    config_sha256=config_sha256,
                    preprocessor_sha256=preprocessor_sha256,
                )
            else:
                source = packet_paths[scenario]
                expected_rows = int(prepared_reports[scenario]["counts"]["packets"])
                estimated_uncompressed_output = expected_rows * (
                    len(preprocessor.feature_names) * np.dtype(np.float32).itemsize
                    + 2 * np.dtype(np.int32).itemsize
                    + np.dtype(np.uint8).itemsize
                    + np.dtype(np.int64).itemsize
                )
                required_bytes = (
                    source.stat().st_size
                    + estimated_uncompressed_output
                    + 2 * 1024**3
                )
                if shutil.disk_usage(local_work_root).free < required_bytes:
                    raise OSError(
                        f"Insufficient local space for fold {fold}/{scenario}."
                    )
                with tempfile.TemporaryDirectory(
                    dir=local_work_root,
                    prefix=f"selection_materialize_{fold}_{scenario}_",
                ) as temporary:
                    temporary_path = Path(temporary)
                    local_source = temporary_path / source.name
                    shutil.copyfile(source, local_source)
                    if sha256_file(local_source) != source_sha256:
                        raise IOError(f"Staged prepared checksum mismatch: {scenario}")
                    local_output = temporary_path / "output"
                    report = _materialize_scenario_fold(
                        scenario=scenario,
                        fold=fold,
                        packet_path=local_source,
                        prepared_report=prepared_reports[scenario],
                        preprocessor=preprocessor,
                        preprocessor_sha256=preprocessor_sha256,
                        preprocessing_contract_sha256=preprocessing_contract_sha256,
                        output_dir=local_output,
                        config=config,
                        config_sha256=config_sha256,
                        batch_size=batch_size,
                    )
                    for shard in report["shards"]:
                        _validate_shard(
                            local_output / shard["path"],
                            shard,
                            int(report["feature_dim"]),
                        )
                    durable_dir.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copytree(local_output, durable_dir)
                report = _validate_scenario_fold(
                    durable_dir,
                    scenario=scenario,
                    fold=fold,
                    source_sha256=source_sha256,
                    config_sha256=config_sha256,
                    preprocessor_sha256=preprocessor_sha256,
                )

            base_report = collection.scenario_dataset(
                fold,
                scenario,
                expected_partition="train",
                verify_shard_checksums=False,
            ).report
            invariant = {
                "graphs_match": report["graphs"] == base_report["graphs"],
                "edges_match": report["edges"] == base_report["edges"],
                "topology_matches": (
                    report["topology_sha256"] == base_report["topology_sha256"]
                ),
                "targets_match": (
                    report["targets_sha256"] == base_report["targets_sha256"]
                ),
                "source_row_ids_match": (
                    report["source_row_ids_sha256"]
                    == base_report["source_row_ids_sha256"]
                ),
                "node_mapping_matches": (
                    report["node_mapping_contract_sha256"]
                    == base_report["node_mapping_contract_sha256"]
                ),
                "selection_features_differ_from_complete_fold_features": (
                    report["features_sha256"] != base_report["features_sha256"]
                ),
            }
            required_invariant_names = (
                "graphs_match",
                "edges_match",
                "topology_matches",
                "targets_match",
                "source_row_ids_match",
                "node_mapping_matches",
            )
            failed = [
                name for name in required_invariant_names if not invariant[name]
            ]
            if failed:
                raise ValueError(
                    f"Selection graph invariants failed for fold {fold}/{scenario}: {failed}"
                )
            scenario_reports[scenario] = report
            invariants[scenario] = invariant
            print(
                f"Completed selection features for fold {fold}/{scenario}: "
                f"{report['graphs']:,} graphs, {report['edges']:,} edges",
                flush=True,
            )
        fold_reports[fold] = {
            "training_scenarios": scenarios,
            "preprocessor_artifact": f"fold_{fold}_inner_preprocessor.json",
            "preprocessor_sha256": preprocessor_sha256,
            "preprocessor_training_rows": int(artifact["training_rows"]),
            "active_feature_count": int(artifact["active_feature_count"]),
            "masked_feature_count": int(artifact["masked_feature_count"]),
            "scenario_reports": {
                scenario: f"fold_{fold}/{scenario}/scenario_fold_report.json"
                for scenario in scenarios
            },
            "invariants": invariants,
        }

    total_edges = sum(
        int(
            _load_json(
                output_dir / relative,
                "selection scenario report",
            )["edges"]
        )
        for fold_report in fold_reports.values()
        for relative in fold_report["scenario_reports"].values()
    )
    expected_edges = int(collection.manifest["totals"]["unique_development_packets"])
    if total_edges != expected_edges:
        raise ValueError("Selection materialization does not cover development packets once.")
    scenario_report_values = [
        _load_json(output_dir / relative, "selection scenario report")
        for fold_report in fold_reports.values()
        for relative in fold_report["scenario_reports"].values()
    ]
    manifest_report = {
        "report_version": REPORT_VERSION,
        "status": "passed",
        "selection_materialization_run_id": output_dir.name,
        "run_config_sha256": sha256_file(output_dir / "run_config.json"),
        "training_preflight_run_id": preflight_dir.name,
        "training_preflight_report_sha256": bindings[
            "training_preflight_report_sha256"
        ],
        "training_preflight_decision_sha256": sha256_file(
            preflight_dir / "training_preflight_decision.json"
        ),
        "folds": fold_reports,
        "total_materialized_edges": total_edges,
        "expected_unique_development_packets": expected_edges,
        "total_graphs": sum(int(item["graphs"]) for item in scenario_report_values),
        "total_shards": sum(int(item["shard_count"]) for item in scenario_report_values),
        "compressed_shard_bytes": sum(
            int(item["compressed_shard_bytes"]) for item in scenario_report_values
        ),
        "selection_preprocessing_fitted": True,
        "selection_features_materialized": True,
        "model_training_performed": False,
        "outer_validation_scenarios_materialized": False,
        "outer_validation_scenarios_used_for_fit": False,
        "complete_fold_reference_checksums_verified": bool(
            collection.artifact_checksums_verified
        ),
        "held_out_scenarios_accessed": False,
        "wall_seconds": float(time.perf_counter() - started),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
        },
    }
    write_json(output_dir / "selection_materialization_manifest.json", manifest_report)
    root_artifacts = [
        "run_config.json",
        selection_config_path.name,
        preflight_config_path.name,
        preprocessing_schema_path.name,
        "fold_A_inner_preprocessor.json",
        "fold_B_inner_preprocessor.json",
        "selection_materialization_manifest.json",
    ]
    checksums = {name: sha256_file(output_dir / name) for name in root_artifacts}
    write_json(output_dir / "artifact_checksums.json", checksums)
    write_json(output_dir / "run_status.json", {
        "complete": True,
        "status": "passed",
        "manifest_sha256": checksums["selection_materialization_manifest.json"],
        "artifact_checksums_sha256": sha256_file(
            output_dir / "artifact_checksums.json"
        ),
        "model_training_performed": False,
        "held_out_scenarios_accessed": False,
    })
    return manifest_report


def load_completed_selection_materialization(
    output_dir: str | Path,
    *,
    collection: CaptureGraphCollection,
    config_path: str | Path,
) -> dict:
    """Validate and load one completed immutable selection materialization."""
    output_dir = Path(output_dir).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    config = load_selection_materialization_config(config_path)
    status = _load_json(output_dir / "run_status.json", "selection status")
    manifest = _load_json(
        output_dir / "selection_materialization_manifest.json",
        "selection manifest",
    )
    checksums = _load_json(
        output_dir / "artifact_checksums.json",
        "selection root checksums",
    )
    run_config = _load_json(output_dir / "run_config.json", "selection run config")
    if (
        status.get("complete") is not True
        or status.get("status") != "passed"
        or status.get("model_training_performed") is not False
        or status.get("held_out_scenarios_accessed") is not False
        or manifest.get("status") != "passed"
        or manifest.get("model_training_performed") is not False
        or manifest.get("outer_validation_scenarios_materialized") is not False
        or manifest.get("outer_validation_scenarios_used_for_fit") is not False
        or manifest.get("complete_fold_reference_checksums_verified") is not True
        or manifest.get("held_out_scenarios_accessed") is not False
        or manifest.get("selection_materialization_run_id") != output_dir.name
        or manifest.get("total_materialized_edges")
        != int(collection.manifest["totals"]["unique_development_packets"])
        or run_config.get("graph_materialization_run_id")
        != collection.expected_run_id
        or run_config.get("selection_config_sha256") != sha256_file(config_path)
        or run_config.get("training_preflight_run_id")
        != config["bindings"]["training_preflight_run_id"]
        or run_config.get("code_sha256") != sha256_file(MODULE_PATH)
        or run_config.get("graph_materialization_code_sha256")
        != sha256_file(GRAPH_MATERIALIZATION_MODULE)
        or run_config.get("preprocessing_code_sha256")
        != sha256_file(PREPROCESSING_MODULE)
        or status.get("manifest_sha256")
        != sha256_file(output_dir / "selection_materialization_manifest.json")
        or status.get("artifact_checksums_sha256")
        != sha256_file(output_dir / "artifact_checksums.json")
    ):
        raise ValueError("The completed selection materialization has a different binding.")
    for name, expected in checksums.items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"Selection root artifact changed: {name}")
    for fold in VALID_FOLDS:
        fold_report = manifest["folds"][fold]
        for scenario in fold_report["training_scenarios"]:
            invariants = fold_report["invariants"][scenario]
            required_invariants = (
                "graphs_match",
                "edges_match",
                "topology_matches",
                "targets_match",
                "source_row_ids_match",
                "node_mapping_matches",
            )
            if not all(invariants.get(name) is True for name in required_invariants):
                raise ValueError(
                    f"Stored selection invariants failed: fold {fold}/{scenario}"
                )
            directory = output_dir / f"fold_{fold}" / scenario
            _validate_scenario_fold(
                directory,
                scenario=scenario,
                fold=fold,
                source_sha256=collection.scenario_dataset(
                    fold,
                    scenario,
                    expected_partition="train",
                    verify_shard_checksums=False,
                ).report["source_artifact_sha256"],
                config_sha256=sha256_file(config_path),
                preprocessor_sha256=fold_report["preprocessor_sha256"],
            )
    return manifest


def save_selection_materialization_decision(
    output_dir: str | Path,
    *,
    approved: bool,
    review_note: str,
) -> dict:
    """Record whether selection tensors may be consumed by the training runner."""
    output_dir = Path(output_dir).expanduser().resolve()
    decision_path = output_dir / "selection_materialization_decision.json"
    if decision_path.exists():
        raise FileExistsError(f"Selection decision already exists: {decision_path}")
    if not isinstance(approved, bool) or not review_note.strip():
        raise ValueError("The decision requires a boolean and a non-empty review note.")
    manifest_path = output_dir / "selection_materialization_manifest.json"
    manifest = _load_json(manifest_path, "selection manifest")
    if approved and manifest.get("status") != "passed":
        raise ValueError("A failed selection materialization cannot be approved.")
    decision = {
        "approved": approved,
        "review_note": review_note.strip(),
        "selection_materialization_manifest_sha256": sha256_file(manifest_path),
        "model_training_authorized": False,
        "next_action": (
            "bind_the_training_runner_to_the_approved_selection_materialization"
            if approved
            else "revise_selection_feature_materialization_before_training"
        ),
    }
    write_json(decision_path, decision)
    return decision
