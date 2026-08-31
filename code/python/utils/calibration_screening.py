"""Validation and configuration helpers for Phase-4B calibration screening."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .calibration import (
    REQUIRED_PROFILES,
    candidate_weight_bias_pairs,
    canonical_json_bytes,
    output_bias,
    sha256_file,
)


SCREENING_SEED = 42
SCREENING_PROFILE = "nfv3_extended"
SCREENING_STAGES = frozenset({"mlp", "stgnn"})
DEFAULT_AP_EQUIVALENCE_MARGIN = 0.005


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {label} to contain a JSON object: {path}")
    return payload


def load_calibration_manifest(path: str | Path) -> tuple[dict[str, Any], str]:
    """Load and fully validate one frozen Phase-4A manifest."""
    manifest_path = Path(path).expanduser().resolve()
    payload = _load_json_object(manifest_path, "calibration manifest")
    validate_calibration_manifest(payload)
    return payload, sha256_file(manifest_path)


def validate_calibration_manifest(payload: Mapping[str, Any]) -> None:
    """Reject altered, incomplete, or internally inconsistent calibration data."""
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported calibration manifest format_version.")
    if payload.get("artifact_type") != "nfv3_training_class_calibration":
        raise ValueError("Unexpected calibration artifact_type.")
    try:
        scope = payload["scope"]
        sources = payload["source_artifacts"]
        counts = payload["counts"]
        alignment = payload["profile_alignment"]
        candidates = payload["candidates"]
    except KeyError as error:
        raise ValueError(f"Calibration manifest is missing {error.args[0]!r}.") from error

    if scope.get("split") != "train" or scope.get("label_access_policy") != "training_split_only":
        raise ValueError("Calibration manifest must be restricted to the training split.")
    if set(scope.get("profiles", [])) != set(REQUIRED_PROFILES):
        raise ValueError("Calibration manifest does not cover both required profiles.")
    for key in (
        "graph_manifest_sha256",
        "feature_schemas",
        "corrected_manifest_sha256",
        "corrected_data_sha256",
        "correction_rule_version",
    ):
        if not sources.get(key):
            raise ValueError(f"Calibration source_artifacts is missing {key!r}.")

    try:
        negative = counts["negative_flows"]
        positive = counts["positive_flows"]
        total = counts["total_flows"]
        prevalence = float(counts["positive_prevalence"])
        ratio = float(counts["class_ratio_negative_to_positive"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Calibration class counts are incomplete or invalid.") from error
    if (
        isinstance(negative, bool)
        or isinstance(positive, bool)
        or not isinstance(negative, int)
        or not isinstance(positive, int)
        or negative <= 0
        or positive <= 0
        or total != negative + positive
    ):
        raise ValueError("Calibration class counts are inconsistent.")
    if not math.isclose(prevalence, positive / total, rel_tol=1e-15, abs_tol=1e-15):
        raise ValueError("Calibration prevalence does not match the class counts.")
    if not math.isclose(ratio, negative / positive, rel_tol=1e-15, abs_tol=1e-15):
        raise ValueError("Calibration class ratio does not match the class counts.")

    if alignment.get("status") != "passed" or alignment.get("split") != "train":
        raise ValueError("Calibration profile alignment did not pass on train.")
    aligned_profiles = alignment.get("profiles", {})
    if set(aligned_profiles) != set(REQUIRED_PROFILES):
        raise ValueError("Calibration alignment does not cover both required profiles.")
    target_hashes = set()
    for profile in REQUIRED_PROFILES:
        record = aligned_profiles[profile]
        if record.get("flows") != total or not isinstance(record.get("graphs"), int):
            raise ValueError(f"Calibration alignment counts are invalid for {profile!r}.")
        target_hash = record.get("target_sha256")
        if not isinstance(target_hash, str) or not target_hash:
            raise ValueError(f"Calibration target hash is missing for {profile!r}.")
        target_hashes.add(target_hash)
    if len(target_hashes) != 1:
        raise ValueError("Calibration feature profiles do not have equal target hashes.")

    expected = candidate_weight_bias_pairs(negative, positive)
    if not isinstance(candidates, list) or len(candidates) != len(expected):
        raise ValueError("Calibration candidates do not match the predeclared grid.")
    seen_ids: set[str] = set()
    for actual, frozen in zip(candidates, expected):
        candidate_id = actual.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in seen_ids:
            raise ValueError("Calibration candidate IDs must be unique strings.")
        seen_ids.add(candidate_id)
        if candidate_id != frozen["candidate_id"] or actual.get("anchors") != frozen["anchors"]:
            raise ValueError("Calibration candidate order or anchors were altered.")
        try:
            weight = float(actual["pos_weight"])
            bias = float(actual["output_bias_init"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Calibration candidate {candidate_id!r} is invalid.") from error
        if not math.isclose(weight, frozen["pos_weight"], rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Calibration weight mismatch for {candidate_id!r}.")
        expected_bias = output_bias(weight, negative, positive)
        if not math.isclose(bias, expected_bias, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError(f"Calibration output bias mismatch for {candidate_id!r}.")


def validate_calibration_sources(
    calibration: Mapping[str, Any],
    graph_manifest: Mapping[str, Any],
    *,
    actual_graph_manifest_sha256: str,
    profile: str = SCREENING_PROFILE,
) -> None:
    """Match the frozen calibration to the exact graph artifacts being screened."""
    if profile != SCREENING_PROFILE:
        raise ValueError(f"Phase 4B calibration screening requires {SCREENING_PROFILE!r}.")
    sources = calibration["source_artifacts"]
    if sources["graph_manifest_sha256"] != actual_graph_manifest_sha256:
        raise ValueError("Calibration and screening graph-manifest hashes differ.")
    if graph_manifest.get("status") != "passed":
        raise ValueError("Phase 4B requires a graph manifest with status='passed'.")
    try:
        graph_schema_hash = graph_manifest["profiles"][profile]
        graph_schema_file_hash = graph_manifest["artifacts"]["feature_schemas"][profile][
            "sha256"
        ]
        graph_collection_hash = graph_manifest["artifacts"]["graph_collections"][profile][
            "sha256"
        ]
        corrected_manifest_hash = graph_manifest["corrected_manifest"]["sha256"]
        calibrated_schema = sources["feature_schemas"][profile]
    except (KeyError, TypeError) as error:
        raise ValueError("Graph/calibration manifests lack required Phase-4B hashes.") from error
    expected_pairs = (
        (calibrated_schema["schema_definition_sha256"], graph_schema_hash, "schema definition"),
        (calibrated_schema["schema_file_sha256"], graph_schema_file_hash, "schema file"),
        (calibrated_schema["graph_collection_sha256"], graph_collection_hash, "graph collection"),
        (sources["corrected_manifest_sha256"], corrected_manifest_hash, "corrected manifest"),
    )
    for calibration_hash, graph_hash, label in expected_pairs:
        if calibration_hash != graph_hash:
            raise ValueError(f"Calibration {label} hash does not match the graph manifest.")


def calibration_candidates(payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Return validated candidates keyed by their stable IDs."""
    validate_calibration_manifest(payload)
    return {item["candidate_id"]: dict(item) for item in payload["candidates"]}


def select_stage_candidates(
    payload: Mapping[str, Any], stage: str, requested_ids: Sequence[str] = ()
) -> list[dict[str, Any]]:
    """Select all MLP candidates or an explicit frozen ST-GNN shortlist."""
    if stage not in SCREENING_STAGES:
        raise ValueError(f"Unknown screening stage {stage!r}.")
    candidates = calibration_candidates(payload)
    requested = list(requested_ids)
    if len(requested) != len(set(requested)):
        raise ValueError("Screening candidate IDs must not be repeated.")
    unknown = sorted(set(requested) - set(candidates))
    if unknown:
        raise ValueError(f"Unknown calibration candidate IDs: {unknown}.")
    if stage == "mlp":
        if requested and set(requested) != set(candidates):
            raise ValueError("The MLP stage must screen every frozen calibration candidate.")
        selected_ids = list(candidates)
    else:
        if not requested:
            raise ValueError("The ST-GNN stage requires an explicit candidate shortlist.")
        selected_ids = requested
    return [candidates[candidate_id] for candidate_id in selected_ids]


def build_screening_model_config(
    *,
    stage: str,
    candidate: Mapping[str, Any],
    edge_dim: int,
    window_ms: int,
    correction_rule_version: str,
    calibration_manifest_sha256: str,
    calibration_code_revision: str,
    epochs: int = 60,
    hidden_dim: int = 64,
    node_dim: int = 32,
    dropout: float = 0.2,
    learning_rate: float = 1e-3,
    batch_steps: int = 10,
    patience: int = 10,
    min_delta: float = 1e-4,
) -> dict[str, Any]:
    """Build one explicit Phase-2/Phase-3 compatible screening configuration."""
    if stage not in SCREENING_STAGES:
        raise ValueError(f"Unknown screening stage {stage!r}.")
    if any(value <= 0 for value in (edge_dim, window_ms, epochs, hidden_dim, batch_steps, patience)):
        raise ValueError("Screening dimensions, epochs, batch steps, and patience must be positive.")
    weight = float(candidate["pos_weight"])
    bias = float(candidate["output_bias_init"])
    candidate_id = str(candidate["candidate_id"])
    common = {
        "model_name": f"phase4b_{stage}_{candidate_id}",
        "variant": f"calibration_screening_{candidate_id}",
        "selection_metric": "average_precision",
        "threshold": {"strategy": "max_f1"},
        "data_params": {
            "label_correction_version": correction_rule_version,
            "feature_profile": SCREENING_PROFILE,
            "calibration_manifest_sha256": calibration_manifest_sha256,
            "calibration_code_revision": calibration_code_revision,
            "calibration_candidate_id": candidate_id,
            "screening_label_policy": "train_for_optimization; validation_for_selection_only",
        },
        "extra_params": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "pos_weight": weight,
            "batch_steps": batch_steps,
            "patience": patience,
            "min_delta": min_delta,
            "max_grad_norm": None,
            "screening_phase": "4B",
            "screening_stage": stage,
        },
    }
    if stage == "mlp":
        return common | {
            "type": "edge_baseline",
            "model_params": {
                "edge_dim": edge_dim,
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "output_bias_init": bias,
            },
            "temporal": False,
            "temporal_memory_policy": "none",
        }
    return common | {
        "type": "spatiotemporal_gnn",
        "model_params": {
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "output_bias_init": bias,
            "identity_mode": "current",
            "use_memory": True,
            "use_topology": True,
            "use_direct_edge_attr": True,
            "memory_policy": "exponential_decay",
            "time_scale_ms": window_ms,
            "window_ms": window_ms,
            "decay_half_life_windows": 20.0,
        },
        "temporal": True,
        "temporal_memory_policy": "exponential_decay",
    }


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_screening_plan(
    *,
    stage: str,
    calibration_manifest_sha256: str,
    graph_manifest_sha256: str,
    code_revision: str,
    configurations: Sequence[Mapping[str, Any]],
    device: str,
    num_workers: int,
    ap_equivalence_margin: float = DEFAULT_AP_EQUIVALENCE_MARGIN,
) -> dict[str, Any]:
    """Freeze one stage before launching any expensive training run."""
    if not math.isfinite(ap_equivalence_margin) or not 0 <= ap_equivalence_margin <= 1:
        raise ValueError("AP-equivalence margin must be finite and in [0, 1].")
    return {
        "format_version": 1,
        "artifact_type": "phase4b_calibration_screening_plan",
        "stage": stage,
        "seed": SCREENING_SEED,
        "feature_profile": SCREENING_PROFILE,
        "selection_policy": {
            "checkpoint_metric": "validation_average_precision",
            "checkpoint_mode": "max",
            "threshold_split": "validation",
            "threshold_strategy": "max_f1",
            "fixed_0_5_comparison_prohibited": True,
            "test_splits_accessed": False,
            "practical_ap_equivalence_margin_absolute": ap_equivalence_margin,
            "shortlist_rule": (
                "include every candidate with validation AP >= best validation AP "
                "minus the frozen absolute margin"
            ),
            "preference_inside_margin": "smaller pos_weight",
        },
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "graph_manifest_sha256": graph_manifest_sha256,
        "code_revision": code_revision,
        "execution": {
            "device": device,
            "batch_size_graphs": 1,
            "shuffle": False,
            "num_workers": num_workers,
        },
        "configurations": [
            {
                "candidate_id": config["data_params"]["calibration_candidate_id"],
                "configuration_sha256": canonical_sha256(config),
                "configuration": dict(config),
            }
            for config in configurations
        ],
    }


def write_json_artifact(
    payload: Mapping[str, Any], path: str | Path, *, overwrite: bool = False
) -> str:
    """Atomically write canonical JSON, accepting an identical existing file."""
    output = Path(path)
    serialized = canonical_json_bytes(payload)
    if output.exists():
        if output.read_bytes() == serialized:
            return "unchanged"
        if not overwrite:
            raise FileExistsError(f"Existing artifact differs from the frozen payload: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(serialized)
    os.replace(temporary, output)
    return "written"


def result_from_run_record(
    record: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any],
    stage: str,
    calibration_manifest_sha256: str,
    screening_plan_sha256: str,
    run_record_path: str,
    run_record_sha256: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Extract the auditable validation result from one completed run record."""
    try:
        configuration = record["configuration"]
        metrics = record["metrics"]
        data_params = configuration["data_params"]
        extra_params = configuration["extra_params"]
    except (KeyError, TypeError) as error:
        raise ValueError("Completed screening run record is incomplete.") from error
    candidate_id = candidate["candidate_id"]
    expected = {
        "calibration_candidate_id": candidate_id,
        "calibration_manifest_sha256": calibration_manifest_sha256,
    }
    if any(data_params.get(key) != value for key, value in expected.items()):
        raise ValueError("Run record does not match the frozen calibration candidate.")
    if extra_params.get("screening_stage") != stage or int(extra_params.get("seed", -1)) != SCREENING_SEED:
        raise ValueError("Run record does not match the Phase-4B stage or seed.")
    if configuration.get("selection_metric") != "average_precision":
        raise ValueError("Run record did not select its checkpoint by validation AP.")
    if configuration.get("threshold") != {"strategy": "max_f1"}:
        raise ValueError("Run record did not use the frozen validation-threshold strategy.")
    try:
        best_ap = float(metrics["best_validation_ap"])
        auc_pr = float(metrics["AUC-PR"])
        threshold = float(metrics["optimal_threshold"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Run record lacks required validation results.") from error
    if not math.isclose(best_ap, auc_pr, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Run-record AP disagrees with the restored best checkpoint AP.")
    return {
        "candidate_id": candidate_id,
        "stage": stage,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "screening_plan_sha256": screening_plan_sha256,
        "pos_weight": float(candidate["pos_weight"]),
        "output_bias_init": float(candidate["output_bias_init"]),
        "seed": SCREENING_SEED,
        "best_validation_ap": best_ap,
        "best_epoch": int(metrics["best_epoch"]),
        "stopped_epoch": int(metrics["stopped_epoch"]),
        "selected_validation_threshold": threshold,
        "validation_metrics_at_selected_threshold": {
            key: metrics[key]
            for key in ("Precision", "Recall", "F1", "F2", "FPR", "TP", "FP", "TN", "FN")
        },
        "run_id": record["run_id"],
        "configuration_sha256": extra_params["configuration_sha256"],
        "code_revision": extra_params["code_version"],
        "run_record": {"path": run_record_path, "sha256": run_record_sha256},
        "checkpoint": {"path": checkpoint_path, "sha256": checkpoint_sha256},
    }


def build_screening_summary(
    *,
    stage: str,
    calibration_manifest_sha256: str,
    plan_sha256: str,
    expected_candidate_ids: Sequence[str],
    results: Sequence[Mapping[str, Any]],
    ap_equivalence_margin: float = DEFAULT_AP_EQUIVALENCE_MARGIN,
) -> dict[str, Any]:
    """Build the validation ranking and apply the predeclared shortlist rule."""
    if not math.isfinite(ap_equivalence_margin) or not 0 <= ap_equivalence_margin <= 1:
        raise ValueError("AP-equivalence margin must be finite and in [0, 1].")
    by_id: dict[str, dict[str, Any]] = {}
    for result in results:
        candidate_id = str(result["candidate_id"])
        if candidate_id in by_id:
            raise ValueError(f"Duplicate completed result for {candidate_id!r}.")
        if result.get("stage") != stage:
            raise ValueError(f"Completed result {candidate_id!r} belongs to another stage.")
        if result.get("calibration_manifest_sha256") != calibration_manifest_sha256:
            raise ValueError(f"Completed result {candidate_id!r} uses another calibration manifest.")
        if result.get("screening_plan_sha256") != plan_sha256:
            raise ValueError(f"Completed result {candidate_id!r} uses another screening plan.")
        by_id[candidate_id] = dict(result)
    unknown = sorted(set(by_id) - set(expected_candidate_ids))
    if unknown:
        raise ValueError(f"Summary contains unplanned candidates: {unknown}.")
    ranking = sorted(
        by_id.values(),
        key=lambda item: (-float(item["best_validation_ap"]), float(item["pos_weight"])),
    )
    missing = [candidate_id for candidate_id in expected_candidate_ids if candidate_id not in by_id]
    best_ap = float(ranking[0]["best_validation_ap"]) if ranking else None
    shortlist = (
        [
            item["candidate_id"]
            for item in ranking
            if best_ap - float(item["best_validation_ap"])
            <= ap_equivalence_margin + 1e-15
        ]
        if best_ap is not None
        else []
    )
    preferred = (
        min(
            (by_id[candidate_id] for candidate_id in shortlist),
            key=lambda item: float(item["pos_weight"]),
        )["candidate_id"]
        if shortlist and not missing
        else None
    )
    return {
        "format_version": 1,
        "artifact_type": "phase4b_calibration_screening_summary",
        "stage": stage,
        "status": "complete" if not missing else "partial",
        "seed": SCREENING_SEED,
        "feature_profile": SCREENING_PROFILE,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "screening_plan_sha256": plan_sha256,
        "expected_candidate_ids": list(expected_candidate_ids),
        "missing_candidate_ids": missing,
        "ranking_policy": {
            "primary": "descending validation average precision",
            "display_tie_break": "ascending pos_weight",
            "practical_ap_equivalence_margin_absolute": ap_equivalence_margin,
            "shortlist": "validation AP >= best validation AP minus the frozen margin",
            "preference_inside_margin": "smaller pos_weight",
            "test_splits_accessed": False,
        },
        "candidates_within_ap_margin": shortlist if not missing else [],
        "recommended_stgnn_shortlist_ids": (
            shortlist if stage == "mlp" and not missing else []
        ),
        "preferred_smaller_weight_candidate_id": preferred,
        "ranking": ranking,
    }
