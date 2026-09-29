"""Freeze and audit the cAPTure one-seed graph-training runner binding."""

from __future__ import annotations

import json
import math
from pathlib import Path
import platform
import shutil
import tempfile
import time

import yaml

from .capture_data import sha256_file, write_json
from .capture_graph_dataset import CaptureGraphCollection
from .capture_graph_selection_materialization import (
    load_completed_selection_materialization,
)
from .capture_graph_training_preflight_v2 import (
    load_completed_training_preflight_v2,
)


REPORT_VERSION = 1
VALID_FOLDS = ("A", "B")
EXPECTED_VARIANTS = (
    "edge_mlp",
    "edge_gru",
    "static_gnn",
    "st_gnn",
    "st_gnn_without_gat",
    "st_gnn_without_direct_edge_attr",
)
MODULE_PATH = Path(__file__)


def _load_json(path: str | Path, label: str) -> dict:
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


def load_graph_training_config(path: str | Path) -> dict:
    """Load and strictly validate the frozen one-seed training contract."""
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The graph-training configuration must be a mapping.")
    if (
        config.get("training_contract_version") != 1
        or config.get("scope") != "development_only"
        or config.get("stage") != "stage2_one_seed_graph_training"
    ):
        raise ValueError("Unsupported graph-training configuration.")

    bindings = config.get("bindings", {})
    required_binding_names = {
        "prepared_run_id",
        "graph_materialization_run_id",
        "graph_input_audit_run_id",
        "training_preflight_run_id",
        "training_preflight_report_sha256",
        "selection_materialization_run_id",
        "selection_materialization_manifest_sha256",
        "graph_input_contract",
        "training_preflight_contract",
        "selection_materialization_contract",
    }
    if set(bindings) != required_binding_names:
        raise ValueError("The graph-training artifact bindings changed.")
    for name in (
        "training_preflight_report_sha256",
        "selection_materialization_manifest_sha256",
    ):
        if len(str(bindings[name])) != 64:
            raise ValueError(f"Invalid SHA-256 binding: {name}")

    required_folds = {
        "A": {
            "train": ["train_empty_conn", "train_qos_mid"],
            "outer_validation": [
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
            "outer_validation": ["train_empty_conn", "train_qos_mid"],
        },
    }
    if config.get("folds") != required_folds:
        raise ValueError("The graph-training fold assignments changed.")

    models = config.get("models", {})
    if tuple(models.get("order", ())) != EXPECTED_VARIANTS:
        raise ValueError("The model order or minimum matrix changed.")
    if models.get("common") != {
        "edge_dim": 103,
        "hidden_dim": 64,
        "node_dim": 16,
        "dropout": 0.2,
        "output_bias_init": None,
        "identity_mode": "current",
        "window_ms": 5000,
    }:
        raise ValueError("The common model architecture changed.")
    if models.get("temporal_memory") != {
        "policy": "exponential_decay",
        "time_scale_ms": 5000,
        "decay_half_life_windows": 20.0,
        "max_gap_ms": None,
    }:
        raise ValueError("The temporal-memory contract changed.")
    variants = models.get("variants", {})
    if set(variants) != set(EXPECTED_VARIANTS):
        raise ValueError("The declared model variants changed.")
    expected_classes = {
        "edge_mlp": "SimpleMLP",
        "edge_gru": "EdgeGRU_Baseline_NoX",
        "static_gnn": "StaticGNN_Identity",
        "st_gnn": "ST_GNN_Identity",
        "st_gnn_without_gat": "ST_GNN_Identity",
        "st_gnn_without_direct_edge_attr": "ST_GNN_Identity",
    }
    expected_temporal = {
        "edge_mlp": False,
        "edge_gru": True,
        "static_gnn": False,
        "st_gnn": True,
        "st_gnn_without_gat": True,
        "st_gnn_without_direct_edge_attr": True,
    }
    for name in EXPECTED_VARIANTS:
        variant = variants[name]
        if (
            variant.get("class") != expected_classes[name]
            or variant.get("temporal") is not expected_temporal[name]
            or variant.get("temporal_memory_policy")
            != ("exponential_decay" if expected_temporal[name] else "none")
            or set(variant)
            != {
                "class",
                "temporal",
                "temporal_memory_policy",
                "use_topology",
                "use_memory",
                "use_direct_edge_attr",
            }
        ):
            raise ValueError(f"The model variant contract changed: {name}")

    if config.get("optimization") != {
        "seed": 42,
        "device": "cuda",
        "optimizer": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.00001,
        "maximum_gradient_norm": None,
        "graph_batch_size": 1,
        "truncated_backpropagation_windows": 10,
        "shuffle": False,
        "loss": (
            "per_edge_binary_cross_entropy_times_fold_local_scenario_class_weight"
        ),
        "loss_normalization": "sum_weighted_loss_divided_by_sum_edge_weights",
        "class_weight_policy": (
            "equal_total_weight_per_scenario_and_binary_class_cell"
        ),
        "class_weight_normalization": (
            "mean_inner_training_or_refit_weight_one"
        ),
        "validation_weighting": "none",
    }:
        raise ValueError("The optimization contract changed.")
    if config.get("checkpoint_selection") != {
        "unit": "model_and_fold",
        "metric": "unweighted_mean_scenario_average_precision",
        "maximum_epochs": 60,
        "minimum_epochs": 10,
        "patience_epochs": 10,
        "minimum_absolute_improvement": 0.0001,
        "restore_best_epoch": True,
        "threshold_selection": "none",
        "preprocessing_scope": "approved_inner_training_prefixes",
        "training_weight_scope": "approved_inner_training_prefixes",
        "validation_scope": "approved_inner_validation_blocks",
        "post_validation_suffix_used": False,
        "outer_validation_used": False,
    }:
        raise ValueError("The checkpoint-selection contract changed.")
    if config.get("final_refit") != {
        "duration_source": "selected_best_epoch_count_per_model_and_fold",
        "reinitialize_model": True,
        "reuse_seed": True,
        "preprocessing_scope": "complete_fold_training_scenarios",
        "training_weight_scope": "complete_fold_training_scenarios",
        "include_inner_validation_and_post_validation": True,
        "early_stopping": False,
        "checkpoint_selection": False,
        "outer_validation_access_during_refit": False,
        "evaluate_outer_validation_after_refit": True,
    }:
        raise ValueError("The final-refit contract changed.")
    if config.get("sequence") != {
        "scenario_order": "declared_fold_order",
        "graph_order": "chronological",
        "reset_temporal_state_at_scenario_boundary": True,
        "reset_temporal_state_at_epoch_boundary": True,
        "reset_temporal_state_before_each_evaluation_scenario": True,
        "inner_validation_starts_with_empty_temporal_state": True,
        "outer_validation_starts_with_empty_temporal_state": True,
    }:
        raise ValueError("The chronological sequence contract changed.")
    authorization = config.get("authorization", {})
    if authorization != {
        "runner_binding_report_required": True,
        "manual_training_authorization_required": True,
        "model_training_authorized_by_this_file": False,
        "held_out_scenarios_accessed": False,
    }:
        raise ValueError("The training authorization boundary changed.")

    pilot_decision = config.get("pilot_decision", {})
    positive_fields = (
        "practical_average_precision_margin",
        "practical_timely_iteration_coverage_margin",
        "primary_false_alert_windows_per_hour",
        "maximum_peak_host_ram_gib",
        "maximum_peak_device_memory_gib",
        "maximum_inference_seconds_per_million_packets",
        "maximum_shifted_origin_absolute_coverage_drop",
    )
    if any(
        not math.isfinite(float(pilot_decision.get(name, -1)))
        or float(pilot_decision[name]) <= 0
        for name in positive_fields
    ):
        raise ValueError("The pilot decision thresholds must be positive and finite.")
    if (
        pilot_decision.get("topology_claim_requires_direct_ablation_support")
        is not True
        or pilot_decision.get("one_seed_results_are_exploratory") is not True
    ):
        raise ValueError("The pilot interpretation contract changed.")
    return config


def _validate_selection_decision(
    selection_dir: Path,
    *,
    expected_manifest_sha256: str,
) -> dict:
    decision = _load_json(
        selection_dir / "selection_materialization_decision.json",
        "selection-materialization decision",
    )
    if (
        decision.get("approved") is not True
        or decision.get("selection_materialization_manifest_sha256")
        != expected_manifest_sha256
        or decision.get("model_training_authorized") is not False
        or decision.get("next_action")
        != "bind_the_training_runner_to_the_approved_selection_materialization"
    ):
        raise ValueError("Selection materialization is not approved for runner binding.")
    return decision


def stage_capture_selection_materialization(
    *,
    source_root: str | Path,
    local_parent: str | Path,
    collection: CaptureGraphCollection,
    config_path: str | Path,
    expected_run_id: str,
    expected_manifest_sha256: str,
    reserve_bytes: int = 2 * 1024**3,
) -> tuple[Path, dict]:
    """Copy the approved selection tensors locally and verify every checksum."""
    source_root = Path(source_root).expanduser().resolve()
    local_parent = Path(local_parent).expanduser().resolve()
    config_path = Path(config_path).expanduser().resolve()
    target_root = local_parent / expected_run_id
    source_manifest_path = source_root / "selection_materialization_manifest.json"
    source_decision_path = source_root / "selection_materialization_decision.json"
    if source_root.name != expected_run_id:
        raise ValueError("The selection-materialization run ID changed.")
    if sha256_file(source_manifest_path) != expected_manifest_sha256:
        raise ValueError("The selection-materialization manifest hash changed.")
    source_decision_sha256 = sha256_file(source_decision_path)
    _validate_selection_decision(
        source_root,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    binding = {
        "expected_run_id": expected_run_id,
        "source_manifest_sha256": expected_manifest_sha256,
        "source_decision_sha256": source_decision_sha256,
        "selection_contract_sha256": sha256_file(config_path),
    }
    receipt_name = "local_training_staging_receipt.json"

    if target_root.exists():
        receipt = _load_json(target_root / receipt_name, "selection staging receipt")
        if receipt.get("binding") != binding:
            raise ValueError("The existing local selection copy has a different binding.")
        manifest = load_completed_selection_materialization(
            target_root,
            collection=collection,
            config_path=config_path,
        )
        _validate_selection_decision(
            target_root,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        print(f"Reusing verified local selection materialization: {target_root}", flush=True)
        return target_root, manifest

    local_parent.mkdir(parents=True, exist_ok=True)
    source_bytes = sum(
        path.stat().st_size for path in source_root.rglob("*") if path.is_file()
    )
    if shutil.disk_usage(local_parent).free < source_bytes + int(reserve_bytes):
        raise OSError(
            "Insufficient local space for selection tensors and the safety reserve."
        )
    with tempfile.TemporaryDirectory(
        dir=local_parent,
        prefix="capture_selection_stage_",
    ) as temporary:
        staged_root = Path(temporary) / expected_run_id
        shutil.copytree(source_root, staged_root)
        manifest = load_completed_selection_materialization(
            staged_root,
            collection=collection,
            config_path=config_path,
        )
        _validate_selection_decision(
            staged_root,
            expected_manifest_sha256=expected_manifest_sha256,
        )
        write_json(staged_root / receipt_name, {
            "binding": binding,
            "all_selection_checksums_verified": True,
        })
        staged_root.rename(target_root)
    print(f"Staged and verified selection materialization: {target_root}", flush=True)
    return target_root, manifest


def _scenario_class_weights(
    scenarios: list[str],
    counts: dict[str, dict],
) -> dict[str, dict[str, float]]:
    total_edges = sum(int(counts[name]["packets"]) for name in scenarios)
    cell_mass = total_edges / (2 * len(scenarios))
    result = {}
    for scenario in scenarios:
        normal = int(counts[scenario]["normal_packets"])
        attack = int(counts[scenario]["attack_packets"])
        packets = int(counts[scenario]["packets"])
        if normal <= 0 or attack <= 0 or normal + attack != packets:
            raise ValueError(f"Invalid binary class counts for {scenario}.")
        result[scenario] = {
            "normal": cell_mass / normal,
            "attack": cell_mass / attack,
        }
    weighted_mass = sum(
        int(counts[name]["normal_packets"]) * result[name]["normal"]
        + int(counts[name]["attack_packets"]) * result[name]["attack"]
        for name in scenarios
    )
    if not math.isclose(weighted_mass, total_edges, rel_tol=1e-12, abs_tol=1e-6):
        raise ValueError("Scenario/class weights do not have mean one.")
    return result


def _model_specifications(config: dict) -> dict[str, dict]:
    common = config["models"]["common"]
    memory = config["models"]["temporal_memory"]
    result = {}
    for name in EXPECTED_VARIANTS:
        variant = config["models"]["variants"][name]
        parameters = {
            "edge_dim": common["edge_dim"],
            "hidden_dim": common["hidden_dim"],
            "dropout": common["dropout"],
            "output_bias_init": common["output_bias_init"],
        }
        if name in {
            "static_gnn",
            "st_gnn",
            "st_gnn_without_gat",
            "st_gnn_without_direct_edge_attr",
        }:
            parameters.update({
                "node_dim": common["node_dim"],
                "identity_mode": common["identity_mode"],
                "window_ms": common["window_ms"],
            })
        if variant["temporal"]:
            parameters.update({
                "memory_policy": memory["policy"],
                "time_scale_ms": memory["time_scale_ms"],
                "decay_half_life_windows": memory[
                    "decay_half_life_windows"
                ],
                "max_gap_ms": memory["max_gap_ms"],
            })
        if variant["class"] == "ST_GNN_Identity":
            parameters.update({
                "use_memory": variant["use_memory"],
                "use_topology": variant["use_topology"],
                "use_direct_edge_attr": variant["use_direct_edge_attr"],
            })
        result[name] = {
            "class": variant["class"],
            "temporal": variant["temporal"],
            "temporal_memory_policy": variant["temporal_memory_policy"],
            "parameters": parameters,
        }
    return result


def _load_approved_preflight(
    *,
    preflight_dir: Path,
    collection: CaptureGraphCollection,
    preflight_config_path: Path,
    expected_report_sha256: str,
) -> tuple[dict, dict]:
    report_path = preflight_dir / "training_preflight_report.json"
    if sha256_file(report_path) != expected_report_sha256:
        raise ValueError("The approved training-preflight report hash changed.")
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
        decision.get("approved") is not True
        or decision.get("training_preflight_report_sha256")
        != expected_report_sha256
        or decision.get("training_authorized") is not False
    ):
        raise ValueError("The training preflight is not approved for runner binding.")
    return report, decision


def _build_fold_plan(
    *,
    fold: str,
    config: dict,
    collection: CaptureGraphCollection,
    selection_root: Path,
    selection_manifest: dict,
    preflight: dict,
) -> dict:
    scenarios = config["folds"][fold]["train"]
    outer_scenarios = config["folds"][fold]["outer_validation"]
    selected_splits = preflight["selected_splits"]
    inner_counts = {
        scenario: selected_splits[scenario]["inner_train"]
        for scenario in scenarios
    }
    selection_sequences = {}
    complete_counts = {}
    for scenario in scenarios:
        split = selected_splits[scenario]
        relative_report = selection_manifest["folds"][fold]["scenario_reports"][
            scenario
        ]
        selection_report = _load_json(
            selection_root / relative_report,
            f"selection scenario report for fold {fold}/{scenario}",
        )
        complete_graphs = int(split["complete_refit_graphs"])
        complete_edges = int(split["complete_refit_edges"])
        if (
            int(selection_report["graphs"]) != complete_graphs
            or int(selection_report["edges"]) != complete_edges
            or int(selection_report["feature_dim"]) != 103
        ):
            raise ValueError(f"Selection sequence changed: fold {fold}/{scenario}")
        train_stop = int(split["inner_train_graphs"])
        validation_stop = train_stop + int(split["inner_validation_graphs"])
        if validation_stop + int(split["post_validation_graphs"]) != complete_graphs:
            raise ValueError(f"Graph slices do not conserve {scenario}.")
        selection_sequences[scenario] = {
            "scenario_report": relative_report,
            "inner_train_graph_range": [0, train_stop],
            "inner_validation_graph_range": [train_stop, validation_stop],
            "excluded_post_validation_graph_range": [
                validation_stop,
                complete_graphs,
            ],
            "validation_window_index_range": [
                int(split["validation_start_window_index"]),
                int(split["validation_end_window_index_exclusive"]),
            ],
            "inner_train_edges": int(split["inner_train_edges"]),
            "inner_validation_edges": int(split["inner_validation_edges"]),
            "excluded_post_validation_edges": int(split["post_validation_edges"]),
        }
        full_report = collection.scenario_dataset(
            fold,
            scenario,
            expected_partition="train",
            verify_shard_checksums=False,
        ).report
        if (
            int(full_report["graphs"]) != complete_graphs
            or int(full_report["edges"]) != complete_edges
        ):
            raise ValueError(f"Complete-refit sequence changed: fold {fold}/{scenario}")
        normal = int(full_report["edges"]) - int(full_report["attack_edges"])
        complete_counts[scenario] = {
            "packets": int(full_report["edges"]),
            "normal_packets": normal,
            "attack_packets": int(full_report["attack_edges"]),
        }

    outer_sequences = {}
    for scenario in outer_scenarios:
        dataset = collection.scenario_dataset(
            fold,
            scenario,
            expected_partition="validation",
            verify_shard_checksums=False,
        )
        outer_sequences[scenario] = {
            "graphs": int(dataset.report["graphs"]),
            "edges": int(dataset.report["edges"]),
            "graph_range": [0, int(dataset.report["graphs"])],
        }

    return {
        "training_scenarios": scenarios,
        "outer_validation_scenarios": outer_scenarios,
        "selection_sequences": selection_sequences,
        "selection_scenario_class_weights": _scenario_class_weights(
            scenarios,
            inner_counts,
        ),
        "complete_refit_scenario_class_weights": _scenario_class_weights(
            scenarios,
            complete_counts,
        ),
        "complete_refit_graphs": sum(
            int(selected_splits[name]["complete_refit_graphs"])
            for name in scenarios
        ),
        "complete_refit_edges": sum(
            int(complete_counts[name]["packets"]) for name in scenarios
        ),
        "outer_validation_sequences": outer_sequences,
        "outer_validation_edges": sum(
            item["edges"] for item in outer_sequences.values()
        ),
    }


def run_capture_graph_training_binding(
    *,
    collection: CaptureGraphCollection,
    selection_root: str | Path,
    preflight_dir: str | Path,
    training_config_path: str | Path,
    preflight_config_path: str | Path,
    selection_config_path: str | Path,
    output_dir: str | Path,
) -> dict:
    """Create the immutable model/fold job plan without performing optimization."""
    started = time.perf_counter()
    selection_root = Path(selection_root).expanduser().resolve()
    preflight_dir = Path(preflight_dir).expanduser().resolve()
    training_config_path = Path(training_config_path).expanduser().resolve()
    preflight_config_path = Path(preflight_config_path).expanduser().resolve()
    selection_config_path = Path(selection_config_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    config = load_graph_training_config(training_config_path)
    bindings = config["bindings"]
    if (
        collection.expected_run_id != bindings["graph_materialization_run_id"]
        or collection.contract_path.name != bindings["graph_input_contract"]
        or selection_root.name != bindings["selection_materialization_run_id"]
        or preflight_dir.name != bindings["training_preflight_run_id"]
        or preflight_config_path.name != bindings["training_preflight_contract"]
        or selection_config_path.name
        != bindings["selection_materialization_contract"]
    ):
        raise ValueError("The runner inputs differ from the frozen bindings.")
    if not collection.artifact_checksums_verified:
        raise ValueError("The complete-fold graph collection must be checksum verified.")

    selection_manifest_path = (
        selection_root / "selection_materialization_manifest.json"
    )
    if sha256_file(selection_manifest_path) != bindings[
        "selection_materialization_manifest_sha256"
    ]:
        raise ValueError("The selection-materialization manifest hash changed.")
    selection_manifest = load_completed_selection_materialization(
        selection_root,
        collection=collection,
        config_path=selection_config_path,
    )
    selection_decision = _validate_selection_decision(
        selection_root,
        expected_manifest_sha256=bindings[
            "selection_materialization_manifest_sha256"
        ],
    )
    preflight, preflight_decision = _load_approved_preflight(
        preflight_dir=preflight_dir,
        collection=collection,
        preflight_config_path=preflight_config_path,
        expected_report_sha256=bindings["training_preflight_report_sha256"],
    )
    if (
        selection_manifest.get("training_preflight_report_sha256")
        != bindings["training_preflight_report_sha256"]
        or selection_manifest.get("model_training_performed") is not False
        or selection_manifest.get("outer_validation_scenarios_used_for_fit")
        is not False
    ):
        raise ValueError("The selection materialization has a different provenance.")

    fold_plans = {
        fold: _build_fold_plan(
            fold=fold,
            config=config,
            collection=collection,
            selection_root=selection_root,
            selection_manifest=selection_manifest,
            preflight=preflight,
        )
        for fold in VALID_FOLDS
    }
    model_specs = _model_specifications(config)
    jobs = []
    for model in EXPECTED_VARIANTS:
        for fold in VALID_FOLDS:
            jobs.append({
                "job_id": f"{model}__fold_{fold}",
                "model": model,
                "fold": fold,
                "selection_training_scenarios": fold_plans[fold][
                    "training_scenarios"
                ],
                "outer_validation_scenarios": fold_plans[fold][
                    "outer_validation_scenarios"
                ],
                "status": "awaiting_manual_training_authorization",
            })

    plan = {
        "training_contract_sha256": sha256_file(training_config_path),
        "model_order": list(EXPECTED_VARIANTS),
        "fold_order": list(VALID_FOLDS),
        "execution_order": [job["job_id"] for job in jobs],
        "model_specifications": model_specs,
        "optimization": config["optimization"],
        "checkpoint_selection": config["checkpoint_selection"],
        "final_refit": config["final_refit"],
        "sequence": config["sequence"],
        "persistence": config["persistence"],
        "pilot_decision": config["pilot_decision"],
        "fold_plans": fold_plans,
        "jobs": jobs,
    }
    run_config = {
        "report_version": REPORT_VERSION,
        "training_contract_sha256": sha256_file(training_config_path),
        "graph_materialization_run_id": collection.expected_run_id,
        "graph_materialization_manifest_sha256": sha256_file(
            collection.manifest_path
        ),
        "training_preflight_run_id": preflight_dir.name,
        "training_preflight_report_sha256": sha256_file(
            preflight_dir / "training_preflight_report.json"
        ),
        "training_preflight_decision_sha256": sha256_file(
            preflight_dir / "training_preflight_decision.json"
        ),
        "selection_materialization_run_id": selection_root.name,
        "selection_materialization_manifest_sha256": sha256_file(
            selection_manifest_path
        ),
        "selection_materialization_decision_sha256": sha256_file(
            selection_root / "selection_materialization_decision.json"
        ),
        "binding_code_sha256": sha256_file(MODULE_PATH),
    }
    if output_dir.exists():
        existing = _load_json(output_dir / "run_config.json", "binding run config")
        if existing != run_config:
            raise FileExistsError(
                "The binding output has a different contract. Use a new run ID."
            )
        report = _load_json(
            output_dir / "training_runner_binding_report.json",
            "training-runner binding report",
        )
        if report.get("training_performed") is not False:
            raise ValueError("The existing binding report performed training.")
        return report

    output_dir.mkdir(parents=True)
    write_json(output_dir / "run_config.json", run_config)
    shutil.copyfile(training_config_path, output_dir / training_config_path.name)
    write_json(output_dir / "training_job_plan.json", plan)
    report = {
        "report_version": REPORT_VERSION,
        "status": "review_required",
        "runner_binding_run_id": output_dir.name,
        "run_config_sha256": sha256_file(output_dir / "run_config.json"),
        "training_contract_sha256": sha256_file(training_config_path),
        "training_job_plan_sha256": sha256_file(
            output_dir / "training_job_plan.json"
        ),
        "selection_materialization_run_id": selection_root.name,
        "selection_materialization_manifest_sha256": bindings[
            "selection_materialization_manifest_sha256"
        ],
        "selection_materialization_decision_sha256": sha256_file(
            selection_root / "selection_materialization_decision.json"
        ),
        "selection_materialization_approved": selection_decision["approved"],
        "training_preflight_report_sha256": bindings[
            "training_preflight_report_sha256"
        ],
        "training_preflight_decision_sha256": sha256_file(
            preflight_dir / "training_preflight_decision.json"
        ),
        "training_preflight_approved": preflight_decision["approved"],
        "complete_fold_checksums_verified": True,
        "selection_checksums_verified": True,
        "model_variants": list(EXPECTED_VARIANTS),
        "folds": list(VALID_FOLDS),
        "jobs": len(jobs),
        "one_seed": config["optimization"]["seed"],
        "selection_preprocessing_scope": (
            "approved_inner_training_prefixes_only"
        ),
        "final_refit_preprocessing_scope": "complete_fold_training_scenarios",
        "outer_validation_used_for_epoch_selection": False,
        "threshold_selected_during_training": False,
        "model_instantiation_performed": False,
        "model_training_performed": False,
        "training_performed": False,
        "held_out_scenarios_accessed": False,
        "manual_training_authorization_required": True,
        "training_authorized": False,
        "wall_seconds": float(time.perf_counter() - started),
        "versions": {"python": platform.python_version()},
    }
    write_json(output_dir / "training_runner_binding_report.json", report)
    artifact_names = [
        "run_config.json",
        training_config_path.name,
        "training_job_plan.json",
        "training_runner_binding_report.json",
    ]
    checksums = {name: sha256_file(output_dir / name) for name in artifact_names}
    write_json(output_dir / "artifact_checksums.json", checksums)
    write_json(output_dir / "run_status.json", {
        "complete": True,
        "status": "review_required",
        "binding_report_sha256": checksums[
            "training_runner_binding_report.json"
        ],
        "artifact_checksums_sha256": sha256_file(
            output_dir / "artifact_checksums.json"
        ),
        "training_performed": False,
        "held_out_scenarios_accessed": False,
    })
    return report


def save_graph_training_authorization(
    output_dir: str | Path,
    *,
    approved: bool,
    review_note: str,
) -> dict:
    """Bind a manual optimization decision to one immutable runner plan."""
    output_dir = Path(output_dir).expanduser().resolve()
    decision_path = output_dir / "graph_training_authorization.json"
    if decision_path.exists():
        raise FileExistsError(f"Training authorization already exists: {decision_path}")
    if not isinstance(approved, bool) or not review_note.strip():
        raise ValueError("Authorization requires a boolean and a non-empty review note.")
    report_path = output_dir / "training_runner_binding_report.json"
    plan_path = output_dir / "training_job_plan.json"
    report = _load_json(report_path, "training-runner binding report")
    if approved and report.get("status") != "review_required":
        raise ValueError("Only a review-required binding report can be approved.")
    decision = {
        "approved": approved,
        "review_note": review_note.strip(),
        "training_runner_binding_report_sha256": sha256_file(report_path),
        "training_job_plan_sha256": sha256_file(plan_path),
        "selection_materialization_manifest_sha256": report[
            "selection_materialization_manifest_sha256"
        ],
        "model_training_authorized": approved,
        "held_out_scenarios_authorized": False,
        "next_action": (
            "implement_and_run_one_resumable_model_fold_job"
            if approved
            else "revise_the_training_contract_before_optimization"
        ),
    }
    write_json(decision_path, decision)
    return decision
