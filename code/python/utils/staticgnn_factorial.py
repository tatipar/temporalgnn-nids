"""Pure helpers for the Phase-4B StaticGNN calibration factorial."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .calibration_screening import (
    DEFAULT_AP_EQUIVALENCE_MARGIN,
    SCREENING_PROFILE,
    canonical_sha256,
)


FACTORIAL_STAGE = "staticgnn_factorial"
FACTORIAL_SEED = 42
DEFAULT_LEARNING_RATES = (1e-3, 5e-3)
DEFAULT_HIDDEN_DIMS = (32, 64)
DEFAULT_FINALIST_COUNT = 2


def _float_token(value: float) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def factorial_configuration_id(
    candidate_id: str, learning_rate: float, hidden_dim: int
) -> str:
    """Return a stable, filesystem-safe identifier for one factorial cell."""
    if not candidate_id:
        raise ValueError("candidate_id must be non-empty.")
    if not math.isfinite(float(learning_rate)) or float(learning_rate) <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
        raise ValueError("hidden_dim must be a positive integer.")
    return f"{candidate_id}_lr{_float_token(learning_rate)}_h{hidden_dim}"


def build_staticgnn_factorial_configuration(
    *,
    candidate: Mapping[str, Any],
    learning_rate: float,
    hidden_dim: int,
    edge_dim: int,
    node_dim: int,
    window_ms: int,
    correction_rule_version: str,
    calibration_manifest_sha256: str,
    calibration_code_revision: str,
    epochs: int = 100,
    dropout: float = 0.2,
    batch_steps: int = 10,
    patience: int = 10,
    min_delta: float = 1e-4,
) -> dict[str, Any]:
    """Build one fully explicit StaticGNN factorial configuration."""
    positive_integers = (edge_dim, node_dim, window_ms, epochs, hidden_dim, batch_steps, patience)
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in positive_integers):
        raise ValueError("Factorial dimensions and training counts must be positive integers.")
    learning_rate = float(learning_rate)
    dropout = float(dropout)
    min_delta = float(min_delta)
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite.")
    if not math.isfinite(dropout) or not 0 <= dropout < 1:
        raise ValueError("dropout must be finite and in [0, 1).")
    if not math.isfinite(min_delta) or min_delta < 0:
        raise ValueError("min_delta must be finite and non-negative.")

    try:
        candidate_id = str(candidate["candidate_id"])
        pos_weight = float(candidate["pos_weight"])
        output_bias_init = float(candidate["output_bias_init"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Calibration candidate is incomplete or invalid.") from error
    if not candidate_id or not all(
        math.isfinite(value) for value in (pos_weight, output_bias_init)
    ) or pos_weight <= 0:
        raise ValueError("Calibration candidate values are invalid.")

    configuration_id = factorial_configuration_id(
        candidate_id, learning_rate, hidden_dim
    )
    return {
        "model_name": f"phase4b_{configuration_id}",
        "type": "spatial_gnn",
        "variant": f"calibration_factorial_{configuration_id}",
        "model_params": {
            "node_dim": node_dim,
            "edge_dim": edge_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "output_bias_init": output_bias_init,
            "identity_mode": "current",
            "window_ms": window_ms,
        },
        "temporal": False,
        "temporal_memory_policy": "none",
        "selection_metric": "average_precision",
        "threshold": {"strategy": "max_f1"},
        "data_params": {
            "label_correction_version": correction_rule_version,
            "feature_profile": SCREENING_PROFILE,
            "calibration_manifest_sha256": calibration_manifest_sha256,
            "calibration_code_revision": calibration_code_revision,
            "calibration_candidate_id": candidate_id,
            "factorial_configuration_id": configuration_id,
            "screening_label_policy": (
                "train_for_optimization; validation_for_selection_only"
            ),
        },
        "extra_params": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "pos_weight": pos_weight,
            "batch_steps": batch_steps,
            "patience": patience,
            "min_delta": min_delta,
            "max_grad_norm": None,
            "screening_phase": "4B",
            "screening_stage": FACTORIAL_STAGE,
        },
    }


def build_staticgnn_factorial_configurations(
    *,
    candidates: Sequence[Mapping[str, Any]],
    learning_rates: Sequence[float] = DEFAULT_LEARNING_RATES,
    hidden_dims: Sequence[int] = DEFAULT_HIDDEN_DIMS,
    **common: Any,
) -> list[dict[str, Any]]:
    """Expand candidates, learning rates, and widths in predeclared order."""
    if not candidates:
        raise ValueError("The factorial requires at least one calibration candidate.")
    if not learning_rates or len(set(float(value) for value in learning_rates)) != len(learning_rates):
        raise ValueError("Factorial learning rates must be non-empty and unique.")
    if not hidden_dims or len(set(hidden_dims)) != len(hidden_dims):
        raise ValueError("Factorial hidden dimensions must be non-empty and unique.")
    configurations = [
        build_staticgnn_factorial_configuration(
            candidate=candidate,
            learning_rate=learning_rate,
            hidden_dim=hidden_dim,
            **common,
        )
        for candidate in candidates
        for learning_rate in learning_rates
        for hidden_dim in hidden_dims
    ]
    identifiers = [
        config["data_params"]["factorial_configuration_id"]
        for config in configurations
    ]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Factorial configuration identifiers are not unique.")
    return configurations


def factorial_run_configuration_matches(
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    seed: int = FACTORIAL_SEED,
) -> bool:
    """Match frozen factorial fields while permitting runtime metadata additions."""
    try:
        if int(actual["extra_params"]["seed"]) != seed:
            return False
        for key in (
            "type",
            "variant",
            "model_params",
            "temporal",
            "temporal_memory_policy",
            "selection_metric",
            "threshold",
        ):
            if actual.get(key) != expected.get(key):
                return False
        for section in ("data_params", "extra_params"):
            if any(
                actual[section].get(key) != value
                for key, value in expected[section].items()
            ):
                return False
        return True
    except (KeyError, TypeError, ValueError):
        return False


def build_staticgnn_factorial_plan(
    *,
    calibration_manifest_sha256: str,
    graph_manifest_sha256: str,
    code_revision: str,
    configurations: Sequence[Mapping[str, Any]],
    learning_rates: Sequence[float],
    hidden_dims: Sequence[int],
    device: str,
    num_workers: int,
    finalist_count: int = DEFAULT_FINALIST_COUNT,
    ap_equivalence_margin: float = DEFAULT_AP_EQUIVALENCE_MARGIN,
) -> dict[str, Any]:
    """Freeze the complete factorial and its validation-only selection policy."""
    if not configurations:
        raise ValueError("The factorial plan requires configurations.")
    if finalist_count <= 0 or finalist_count > len(configurations):
        raise ValueError("finalist_count must fit inside the factorial.")
    if not math.isfinite(ap_equivalence_margin) or not 0 <= ap_equivalence_margin <= 1:
        raise ValueError("AP-equivalence margin must be finite and in [0, 1].")
    learning_rates = tuple(float(value) for value in learning_rates)
    hidden_dims = tuple(int(value) for value in hidden_dims)
    if (
        not learning_rates
        or len(set(learning_rates)) != len(learning_rates)
        or any(not math.isfinite(value) or value <= 0 for value in learning_rates)
    ):
        raise ValueError("Plan learning rates must be positive, finite, and unique.")
    if (
        not hidden_dims
        or len(set(hidden_dims)) != len(hidden_dims)
        or any(value <= 0 for value in hidden_dims)
    ):
        raise ValueError("Plan hidden dimensions must be positive and unique.")
    candidate_ids = {
        str(config["data_params"]["calibration_candidate_id"])
        for config in configurations
    }
    expected_size = 5 * len(learning_rates) * len(hidden_dims)
    if len(configurations) != expected_size:
        raise ValueError(
            "The StaticGNN factorial must contain all five calibration candidates "
            "crossed with every learning rate and hidden dimension."
        )
    if len(candidate_ids) != 5:
        raise ValueError("The StaticGNN factorial requires five distinct candidates.")
    actual_cells = {
        (
            str(config["data_params"]["calibration_candidate_id"]),
            float(config["extra_params"]["learning_rate"]),
            int(config["model_params"]["hidden_dim"]),
        )
        for config in configurations
    }
    expected_cells = {
        (candidate_id, learning_rate, hidden_dim)
        for candidate_id in candidate_ids
        for learning_rate in learning_rates
        for hidden_dim in hidden_dims
    }
    if actual_cells != expected_cells:
        raise ValueError("The StaticGNN factorial does not match the declared grid.")
    reference_model = configurations[0]["model_params"]
    reference_training = configurations[0]["extra_params"]
    fixed_model_keys = ("node_dim", "edge_dim", "dropout", "identity_mode", "window_ms")
    fixed_training_keys = (
        "epochs",
        "batch_steps",
        "patience",
        "min_delta",
        "max_grad_norm",
    )
    if any(
        any(
            config["model_params"].get(key) != reference_model.get(key)
            for key in fixed_model_keys
        )
        or any(
            config["extra_params"].get(key) != reference_training.get(key)
            for key in fixed_training_keys
        )
        for config in configurations
    ):
        raise ValueError("Non-factorial model and training parameters must be fixed.")
    candidate_pairs: dict[str, set[tuple[float, float]]] = {}
    for config in configurations:
        candidate_id = str(config["data_params"]["calibration_candidate_id"])
        candidate_pairs.setdefault(candidate_id, set()).add(
            (
                float(config["extra_params"]["pos_weight"]),
                float(config["model_params"]["output_bias_init"]),
            )
        )
    if any(len(pairs) != 1 for pairs in candidate_pairs.values()):
        raise ValueError("Each calibration candidate must retain one weight/bias pair.")
    configuration_ids = [
        str(config["data_params"]["factorial_configuration_id"])
        for config in configurations
    ]
    if len(configuration_ids) != len(set(configuration_ids)):
        raise ValueError("Factorial plan contains duplicate configuration IDs.")
    return {
        "format_version": 1,
        "artifact_type": "phase4b_staticgnn_factorial_plan",
        "stage": FACTORIAL_STAGE,
        "seed": FACTORIAL_SEED,
        "feature_profile": SCREENING_PROFILE,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "graph_manifest_sha256": graph_manifest_sha256,
        "code_revision": code_revision,
        "grid": {
            "calibration_candidate_count": 5,
            "learning_rates": list(learning_rates),
            "hidden_dims": list(hidden_dims),
            "configuration_count": len(configurations),
        },
        "fixed_parameters": {
            "node_dim": int(reference_model["node_dim"]),
            "edge_dim": int(reference_model["edge_dim"]),
            "dropout": float(reference_model["dropout"]),
            "identity_mode": str(reference_model["identity_mode"]),
            "window_ms": int(reference_model["window_ms"]),
            "maximum_epochs": int(reference_training["epochs"]),
            "batch_steps": int(reference_training["batch_steps"]),
            "patience": int(reference_training["patience"]),
            "min_delta": float(reference_training["min_delta"]),
            "max_grad_norm": reference_training["max_grad_norm"],
        },
        "selection_policy": {
            "checkpoint_metric": "validation_average_precision",
            "checkpoint_mode": "max",
            "initial_ranking": "descending best validation average precision at seed 42",
            "finalist_count": finalist_count,
            "finalist_confirmation_seeds": [42, 123, 777],
            "confirmation_primary": "highest mean validation average precision",
            "practical_ap_equivalence_margin_absolute": ap_equivalence_margin,
            "preference_inside_margin": (
                "smaller pos_weight, then smaller learning_rate, then smaller hidden_dim"
            ),
            "threshold_split": "validation",
            "threshold_strategy": "max_f1",
            "fixed_0_5_comparison_prohibited": True,
            "test_splits_accessed": False,
        },
        "execution": {
            "device": device,
            "batch_size_graphs": 1,
            "shuffle": False,
            "num_workers": num_workers,
            "progress_output": "every checkpoint improvement and every tenth epoch",
        },
        "configurations": [
            {
                "configuration_id": config["data_params"]["factorial_configuration_id"],
                "configuration_sha256": canonical_sha256(config),
                "configuration": dict(config),
            }
            for config in configurations
        ],
    }


def factorial_result_from_run_record(
    record: Mapping[str, Any],
    *,
    expected_configuration: Mapping[str, Any],
    calibration_manifest_sha256: str,
    factorial_plan_sha256: str,
    run_record_path: str,
    run_record_sha256: str,
    checkpoint_path: str,
    checkpoint_sha256: str,
) -> dict[str, Any]:
    """Extract one auditable validation-only factorial result."""
    try:
        configuration = record["configuration"]
        metrics = record["metrics"]
        data_params = configuration["data_params"]
        model_params = configuration["model_params"]
        extra_params = configuration["extra_params"]
        expected_data = expected_configuration["data_params"]
    except (KeyError, TypeError) as error:
        raise ValueError("Completed factorial run record is incomplete.") from error
    if not factorial_run_configuration_matches(configuration, expected_configuration):
        raise ValueError("Run record does not match the frozen factorial configuration.")
    configuration_id = expected_data["factorial_configuration_id"]
    if data_params.get("factorial_configuration_id") != configuration_id:
        raise ValueError("Run record belongs to another factorial configuration.")
    if data_params.get("calibration_manifest_sha256") != calibration_manifest_sha256:
        raise ValueError("Run record uses another calibration manifest.")
    if int(extra_params.get("seed", -1)) != FACTORIAL_SEED:
        raise ValueError("Initial factorial run must use seed 42.")
    if extra_params.get("screening_stage") != FACTORIAL_STAGE:
        raise ValueError("Run record belongs to another screening stage.")
    if configuration.get("selection_metric") != "average_precision":
        raise ValueError("Factorial checkpoint was not selected by validation AP.")
    if configuration.get("threshold") != {"strategy": "max_f1"}:
        raise ValueError("Factorial threshold was not selected by validation max_f1.")
    try:
        best_ap = float(metrics["best_validation_ap"])
        restored_ap = float(metrics["AUC-PR"])
        threshold = float(metrics["optimal_threshold"])
        total_seconds = float(metrics["time_total_sec"])
        training_seconds = float(metrics["time_train_sec"])
        validation_seconds = float(metrics["time_eval_sec"])
        final_validation_seconds = float(metrics["time_final_eval_sec"])
        threshold_seconds = float(metrics["time_threshold_sec"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("Factorial run lacks required validation or timing results.") from error
    if not math.isclose(best_ap, restored_ap, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Best and restored-checkpoint validation AP disagree.")
    if any(
        not math.isfinite(value) or value < 0
        for value in (
            total_seconds,
            training_seconds,
            validation_seconds,
            final_validation_seconds,
            threshold_seconds,
        )
    ):
        raise ValueError("Factorial timing results are invalid.")
    return {
        "configuration_id": configuration_id,
        "candidate_id": data_params["calibration_candidate_id"],
        "stage": FACTORIAL_STAGE,
        "seed": FACTORIAL_SEED,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "factorial_plan_sha256": factorial_plan_sha256,
        "pos_weight": float(extra_params["pos_weight"]),
        "output_bias_init": float(model_params["output_bias_init"]),
        "learning_rate": float(extra_params["learning_rate"]),
        "hidden_dim": int(model_params["hidden_dim"]),
        "best_validation_ap": best_ap,
        "best_epoch": int(metrics["best_epoch"]),
        "stopped_epoch": int(metrics["stopped_epoch"]),
        "selected_validation_threshold": threshold,
        "threshold_selection": str(metrics["threshold_selection"]),
        "validation_metrics_at_selected_threshold": {
            key: metrics[key]
            for key in ("Precision", "Recall", "F1", "F2", "FPR", "TP", "FP", "TN", "FN")
        },
        "timing_seconds": {
            "total": total_seconds,
            "training": training_seconds,
            "epoch_validation": validation_seconds,
            "final_validation": final_validation_seconds,
            "threshold_selection": threshold_seconds,
        },
        "run_id": record["run_id"],
        "configuration_sha256": extra_params["configuration_sha256"],
        "code_revision": extra_params["code_version"],
        "run_record": {"path": run_record_path, "sha256": run_record_sha256},
        "checkpoint": {"path": checkpoint_path, "sha256": checkpoint_sha256},
    }


def build_staticgnn_factorial_summary(
    *,
    calibration_manifest_sha256: str,
    plan_sha256: str,
    expected_configuration_ids: Sequence[str],
    results: Sequence[Mapping[str, Any]],
    finalist_count: int = DEFAULT_FINALIST_COUNT,
) -> dict[str, Any]:
    """Rank completed factorial cells and identify the two seed-42 finalists."""
    by_id: dict[str, dict[str, Any]] = {}
    for result in results:
        configuration_id = str(result["configuration_id"])
        if configuration_id in by_id:
            raise ValueError(f"Duplicate factorial result {configuration_id!r}.")
        if result.get("stage") != FACTORIAL_STAGE:
            raise ValueError(f"Result {configuration_id!r} belongs to another stage.")
        if result.get("calibration_manifest_sha256") != calibration_manifest_sha256:
            raise ValueError(f"Result {configuration_id!r} uses another calibration manifest.")
        if result.get("factorial_plan_sha256") != plan_sha256:
            raise ValueError(f"Result {configuration_id!r} uses another factorial plan.")
        by_id[configuration_id] = dict(result)
    unknown = sorted(set(by_id) - set(expected_configuration_ids))
    if unknown:
        raise ValueError(f"Summary contains unplanned configurations: {unknown}.")
    missing = [identifier for identifier in expected_configuration_ids if identifier not in by_id]
    ranking = sorted(
        by_id.values(),
        key=lambda item: (
            -float(item["best_validation_ap"]),
            float(item["pos_weight"]),
            float(item["learning_rate"]),
            int(item["hidden_dim"]),
        ),
    )
    complete = not missing
    finalists = [
        item["configuration_id"] for item in ranking[:finalist_count]
    ] if complete else []
    timing_keys = (
        "total",
        "training",
        "epoch_validation",
        "final_validation",
        "threshold_selection",
    )
    aggregate_timing = {
        key: sum(float(item["timing_seconds"][key]) for item in ranking)
        for key in timing_keys
    }
    return {
        "format_version": 1,
        "artifact_type": "phase4b_staticgnn_factorial_summary",
        "stage": FACTORIAL_STAGE,
        "status": "complete" if complete else "partial",
        "seed": FACTORIAL_SEED,
        "feature_profile": SCREENING_PROFILE,
        "calibration_manifest_sha256": calibration_manifest_sha256,
        "factorial_plan_sha256": plan_sha256,
        "expected_configuration_ids": list(expected_configuration_ids),
        "missing_configuration_ids": missing,
        "ranking_policy": {
            "primary": "descending best validation average precision at seed 42",
            "tie_breaks": (
                "ascending pos_weight, learning_rate, then hidden_dim for display only"
            ),
            "finalist_count": finalist_count,
            "test_splits_accessed": False,
        },
        "recommended_finalist_configuration_ids": finalists,
        "aggregate_completed_run_timing_seconds": aggregate_timing,
        "ranking": ranking,
    }
