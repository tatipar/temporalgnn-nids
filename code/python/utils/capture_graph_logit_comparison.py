"""Corrected raw-logit comparison for the cAPTure graph-model pilot."""

from __future__ import annotations

from collections import defaultdict
import json
import math
from pathlib import Path
import subprocess
from statistics import mean

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml

from . import capture_early_warning, capture_graph_comparison, capture_oof_operational
from .capture_data import load_manifest, sha256_file, write_json
from .capture_early_warning import validate_early_warning_audit
from .capture_graph_comparison import (
    COMPARISONS,
    DISPLAY_NAMES,
    MODEL_NAMES,
    PRIMARY_BUDGET,
    _build_tables as _build_graph_tables,
    _fold_macro,
    _load_terminal_policy,
    _operational_metrics,
)
from .capture_graph_oof_rescoring import (
    validate_capture_graph_oof_rescoring,
)
from .capture_oof_operational import (
    evaluate_scenario,
    operational_budgets,
    validate_operational_run,
)


REPORT_VERSION = 1
MODULE_PATH = Path(__file__)
XGB_MODELS = ("xgb_p", "current_window", "history", "full")
XGB_DISPLAY_NAMES = {
    "xgb_p": "XGB-P",
    "current_window": "XGB current window",
    "history": "XGB history",
    "full": "XGB full",
}
COMMON_ALIGNMENT_COLUMNS = (
    "packet_id",
    "source_row_id",
    "window_index",
    "window_end_ns",
    "packet_timestamp_ns",
    "binary_label",
    "attack_step",
    "sequence_id",
)


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


def load_graph_logit_comparison_config(path: str | Path) -> dict:
    """Load and strictly validate the corrected comparison contract."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The graph logit-comparison contract must be a mapping.")
    if (
        config.get("comparison_contract_version") != 1
        or config.get("scope") != "development_only"
        or config.get("stage") != "stage2_corrected_four_model_comparison"
        or tuple(config.get("graph_models", [])) != tuple(MODEL_NAMES)
        or tuple(config.get("xgb_benchmarks", [])) != XGB_MODELS
    ):
        raise ValueError("Unsupported graph logit-comparison contract.")
    expected_scoring = {
        "graph_ranking_field": "raw_logit",
        "graph_probability_diagnostic_field": "score_float64",
        "xgb_ranking_field": "score",
        "packet_threshold_rule": "score_greater_than_or_equal_to_threshold",
        "window_alert_rule": (
            "maximum_packet_score_greater_than_or_equal_to_threshold"
        ),
        "threshold_tie_rule": "nextafter_above_tied_benign_score",
        "threshold_selection_scope": "development_oof_only",
        "primary_false_alert_budget": "one_per_hour",
        "sensitivity_budgets": ["one_per_12_hours", "one_per_5_minutes"],
    }
    if config.get("scoring") != expected_scoring:
        raise ValueError("The corrected scoring contract changed.")
    if config.get("aggregation") != {
        "primary": "equal_scenario_mean_within_fold_then_equal_fold_mean",
        "threshold_constraint": "maximum_fold_mean_false_alert_windows_per_hour",
        "flat_scenario_macro_is_diagnostic_only": True,
    }:
        raise ValueError("The corrected aggregation contract changed.")
    if config.get("interpretation") != {
        "xgb_is_external_development_benchmark_not_architecture_ablation": True,
        "one_seed_results_are_exploratory": True,
        "topology_claim_authorized": False,
    }:
        raise ValueError("The comparison interpretation contract changed.")
    if config.get("prohibitions") != {
        "model_training": True,
        "checkpoint_loading": True,
        "threshold_selection_from_held_out_data": True,
        "held_out_scenario_access": True,
        "source_artifact_overwrite": True,
    }:
        raise ValueError("The comparison prohibitions changed.")
    return config


def _validate_bound_paths(
    *,
    config: dict,
    rescoring_root: Path,
    source_comparison_dir: Path,
    saturation_audit_dir: Path,
    xgb_operational_dir: Path,
    xgb_early_warning_dir: Path,
    xgb_run_dirs: dict[str, Path],
) -> None:
    bindings = config["bindings"]
    observed = {
        "graph_rescoring_run_id": rescoring_root.name,
        "source_graph_comparison_run_id": source_comparison_dir.name,
        "score_saturation_audit_run_id": saturation_audit_dir.name,
        "xgb_operational_run_id": xgb_operational_dir.name,
        "xgb_early_warning_run_id": xgb_early_warning_dir.name,
        "xgb_p_run_id": xgb_run_dirs["xgb_p"].name,
        "xgb_p_t_run_id": xgb_run_dirs["full"].name,
        "xgb_p_t_ablation_run_id": xgb_run_dirs["history"].parent.name,
    }
    for name, value in observed.items():
        if bindings.get(name) != value:
            raise ValueError(f"The bound source changed: {name}")
    if (
        xgb_run_dirs["current_window"].parent != xgb_run_dirs["history"].parent
        or xgb_run_dirs["current_window"].name != "current_window"
        or xgb_run_dirs["history"].name != "history"
    ):
        raise ValueError("The XGB ablation directories changed.")


def _graph_paths(rescoring_root: Path, manifest: dict) -> dict[str, dict[str, Path]]:
    paths = {model_name: {} for model_name in MODEL_NAMES}
    for model_name in MODEL_NAMES:
        for scenario, item in manifest["scenario_outputs"][model_name].items():
            path = rescoring_root / item["path"]
            if (
                item.get("operational_ranking_field") != "raw_logit"
                or sha256_file(path) != item.get("sha256")
                or pq.ParquetFile(path).metadata.num_rows != int(item["rows"])
            ):
                raise ValueError(f"The rescored OOF artifact changed: {model_name}/{scenario}")
            schema = pq.ParquetFile(path).schema_arrow
            for field in ("raw_logit", "score_float64", *COMMON_ALIGNMENT_COLUMNS):
                if schema.get_field_index(field) < 0:
                    raise ValueError(f"The rescored OOF schema is missing {field}.")
            paths[model_name][scenario] = path
    return paths


def _xgb_paths(
    manifest: dict,
    xgb_run_dirs: dict[str, Path],
) -> tuple[dict, dict[str, dict[str, Path]]]:
    reports = capture_oof_operational._validated_runs(manifest, xgb_run_dirs)
    paths = {model_name: {} for model_name in XGB_MODELS}
    for model_name in XGB_MODELS:
        for fold, split in manifest["validation"]["folds"].items():
            report = reports[model_name][fold]
            for scenario in split["validate"]:
                path = capture_oof_operational._scenario_oof_path(
                    xgb_run_dirs[model_name], fold, report, scenario
                )
                item = report["validation"][scenario]
                if (
                    sha256_file(path) != item["oof_sha256"]
                    or pq.ParquetFile(path).metadata.num_rows != int(item["rows"])
                ):
                    raise ValueError(f"The XGB OOF artifact changed: {model_name}/{scenario}")
                paths[model_name][scenario] = path
    return reports, paths


def _validate_cross_family_alignment(
    *, graph_paths: dict[str, dict[str, Path]], xgb_paths: dict[str, dict[str, Path]]
) -> dict[str, int]:
    """Require exact canonical packet identities between graph and XGB OOF outputs."""
    counts = {}
    fields = ", ".join(COMMON_ALIGNMENT_COLUMNS)
    mismatches = " OR ".join(
        f"g.{field} IS DISTINCT FROM x.{field}"
        for field in COMMON_ALIGNMENT_COLUMNS
        if field != "source_row_id"
    )
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 2")
        connection.execute("SET memory_limit = '4GB'")
        for scenario, graph_path in graph_paths[MODEL_NAMES[0]].items():
            xgb_path = xgb_paths["xgb_p"][scenario]
            graph_rows = pq.ParquetFile(graph_path).metadata.num_rows
            xgb_rows = pq.ParquetFile(xgb_path).metadata.num_rows
            if graph_rows != xgb_rows:
                raise ValueError(f"Graph/XGB OOF row counts differ for {scenario}.")
            for path, family in ((graph_path, "graph"), (xgb_path, "XGB")):
                count, distinct_rows, minimum, maximum = connection.execute(
                    """
                    SELECT count(*), count(DISTINCT source_row_id),
                           min(source_row_id), max(source_row_id)
                    FROM read_parquet(?)
                    """,
                    [str(path)],
                ).fetchone()
                if (
                    int(count) != graph_rows
                    or int(distinct_rows) != graph_rows
                    or int(minimum) != 0
                    or int(maximum) != graph_rows - 1
                ):
                    raise ValueError(
                        f"The {family} source-row sequence is not canonical: {scenario}"
                    )
            query = f"""
                WITH graph_rows AS (
                    SELECT {fields}
                    FROM read_parquet(?)
                ),
                xgb_rows AS (
                    SELECT {fields}
                    FROM read_parquet(?)
                )
                SELECT count(*)
                FROM graph_rows AS g
                FULL OUTER JOIN xgb_rows AS x USING (source_row_id)
                WHERE g.source_row_id IS NULL OR x.source_row_id IS NULL OR {mismatches}
            """
            different = int(
                connection.execute(query, [str(graph_path), str(xgb_path)]).fetchone()[0]
            )
            if different:
                raise ValueError(
                    f"Graph/XGB evaluation keys differ for {scenario}: {different}"
                )
            counts[scenario] = int(graph_rows)
    finally:
        connection.close()
    return counts


def _ranking_metrics(path: Path, score_field: str) -> dict:
    table = pq.read_table(path, columns=["binary_label", score_field])
    labels = table.column("binary_label").to_numpy(zero_copy_only=False)
    scores = table.column(score_field).to_numpy(zero_copy_only=False).astype(
        np.float64, copy=False
    )
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError(f"Invalid ranking values in {path}.")
    return {
        "packets": int(len(labels)),
        "normal_packets": int(np.count_nonzero(labels == 0)),
        "attack_packets": int(np.count_nonzero(labels == 1)),
        "average_precision": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def _summarize_scenario(
    path: Path,
    *,
    score_field: str,
    expected_rows: int,
    expected_attack_packets: int,
) -> dict:
    if pq.ParquetFile(path).metadata.num_rows != expected_rows:
        raise ValueError("OOF row count differs from the sealed manifest.")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 2")
        connection.execute("SET memory_limit = '4GB'")
        window_rows = connection.execute(f"""
            SELECT window_index, count(*) AS packets,
                   sum(binary_label) AS attack_packets,
                   max(CAST({score_field} AS DOUBLE)) AS max_score,
                   min(window_end_ns) AS first_end_ns,
                   max(window_end_ns) AS last_end_ns
            FROM read_parquet(?)
            GROUP BY window_index ORDER BY window_index
        """, [str(path)]).fetchall()
        iteration_rows = connection.execute(f"""
            SELECT attack_step, sequence_id, window_index,
                   min(packet_timestamp_ns) AS first_packet_ns,
                   max(packet_timestamp_ns) AS last_packet_ns,
                   min(window_end_ns) AS first_end_ns,
                   max(window_end_ns) AS last_end_ns,
                   max(CAST({score_field} AS DOUBLE)) AS max_malicious_score,
                   count(*) AS attack_packets
            FROM read_parquet(?)
            WHERE binary_label = 1
            GROUP BY attack_step, sequence_id, window_index
            ORDER BY attack_step, sequence_id, window_index
        """, [str(path)]).fetchall()
    finally:
        connection.close()
    if not window_rows or int(window_rows[0][0]) != 0:
        raise ValueError("OOF windows must begin at scenario window zero.")
    if (
        sum(int(row[1]) for row in window_rows) != expected_rows
        or sum(int(row[2]) for row in window_rows) != expected_attack_packets
    ):
        raise ValueError("OOF window aggregation did not conserve packet counts.")
    if any(row[4] != row[5] or not np.isfinite(row[3]) for row in window_rows):
        raise ValueError("OOF decision times or scores are invalid.")
    attack_windows = sum(int(row[2]) > 0 for row in window_rows)
    total_windows = int(window_rows[-1][0]) + 1
    benign_windows = total_windows - attack_windows
    exposure_hours = benign_windows * 5 / 3600
    if exposure_hours <= 0:
        raise ValueError("A scenario has no benign wall-clock exposure.")
    negative_scores = np.sort(np.asarray(
        [float(row[3]) for row in window_rows if int(row[2]) == 0],
        dtype=np.float64,
    ))
    iterations = {}
    for step, sequence, _, first, last, first_end, last_end, score, packets in iteration_rows:
        if (
            step is None
            or sequence is None
            or first_end != last_end
            or not np.isfinite(score)
        ):
            raise ValueError("Attack-step iteration metadata or scores are invalid.")
        key = (str(step), str(sequence))
        item = iterations.setdefault(key, {
            "attack_step": str(step),
            "sequence_id": str(sequence),
            "first_packet_ns": first,
            "last_packet_ns": last,
            "window_scores": [],
            "attack_packets": 0,
        })
        item["first_packet_ns"] = min(item["first_packet_ns"], first)
        item["last_packet_ns"] = max(item["last_packet_ns"], last)
        item["window_scores"].append((int(first_end), float(score)))
        item["attack_packets"] += int(packets)
    if (
        sum(item["attack_packets"] for item in iterations.values())
        != expected_attack_packets
        or not iterations
    ):
        raise ValueError("Attack-step aggregation did not conserve packet counts.")
    return {
        "negative_scores": negative_scores,
        "exposure_hours": exposure_hours,
        "total_wall_clock_windows": total_windows,
        "attack_windows": attack_windows,
        "benign_windows": benign_windows,
        "iterations": list(iterations.values()),
    }


def _false_alert_rate(summary: dict, threshold: float) -> tuple[int, float]:
    scores = summary["negative_scores"]
    alerts = len(scores) - int(np.searchsorted(scores, threshold, side="left"))
    return alerts, alerts / summary["exposure_hours"]


def _select_threshold(
    summaries: dict[str, dict], folds: dict[str, dict], budget_per_hour: float
) -> dict:
    if not np.isfinite(budget_per_hour) or budget_per_hour <= 0:
        raise ValueError("False-alert budget must be positive and finite.")
    negative_scores = np.concatenate([
        item["negative_scores"] for item in summaries.values()
    ])
    unique = np.unique(negative_scores)
    lower = np.nextafter(unique[0], -np.inf)
    candidates = np.concatenate((
        np.asarray([lower], dtype=np.float64),
        np.nextafter(unique, np.inf),
    ))

    def fold_rates(threshold: float) -> dict[str, float]:
        return {
            fold: float(np.mean([
                _false_alert_rate(summaries[scenario], threshold)[1]
                for scenario in split["validate"]
            ]))
            for fold, split in folds.items()
        }

    left, right = 0, len(candidates) - 1
    while left < right:
        middle = (left + right) // 2
        if max(fold_rates(float(candidates[middle])).values()) <= budget_per_hour:
            right = middle
        else:
            left = middle + 1
    threshold = float(candidates[left])
    rates = fold_rates(threshold)
    if max(rates.values()) > budget_per_hour:
        raise AssertionError("No threshold satisfies the false-alert budget.")
    return {
        "threshold": threshold,
        "fold_false_alert_windows_per_hour": rates,
        "worst_fold_false_alert_windows_per_hour": max(rates.values()),
    }


def _packet_counts_by_threshold(
    path: Path,
    *,
    score_field: str,
    thresholds: dict[str, float],
    expected_rows: int,
    batch_size: int,
) -> dict:
    counts = {
        name: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
        for name in thresholds
    }
    rows = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["binary_label", score_field],
    ):
        labels = batch.column(0).to_numpy(zero_copy_only=False)
        scores = batch.column(1).to_numpy(zero_copy_only=False).astype(
            np.float64, copy=False
        )
        if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
            raise ValueError("OOF packets contain invalid labels or scores.")
        rows += len(labels)
        for name, threshold in thresholds.items():
            flagged = scores >= threshold
            item = counts[name]
            item["tp"] += int(np.count_nonzero(flagged & (labels == 1)))
            item["fp"] += int(np.count_nonzero(flagged & (labels == 0)))
            item["tn"] += int(np.count_nonzero(~flagged & (labels == 0)))
            item["fn"] += int(np.count_nonzero(~flagged & (labels == 1)))
    if rows != expected_rows:
        raise ValueError("OOF packet count changed during threshold evaluation.")
    return counts


def _sigmoid_float64(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _evaluate_graph_models(
    *,
    paths: dict[str, dict[str, Path]],
    rescoring_root: Path,
    folds: dict[str, dict],
    budgets: list[tuple[str, float]],
    terminal_steps: dict[str, list[str]],
    batch_size: int,
) -> dict:
    result = {}
    scenarios = {
        scenario for split in folds.values() for scenario in split["validate"]
    }
    reference_attack_counts = None
    for model_name in MODEL_NAMES:
        print(f"Evaluating raw-logit OOF predictions for {model_name}...", flush=True)
        ranking_scenarios = {}
        summaries = {}
        expected = {}
        for fold, split in folds.items():
            job_report = _load_json(
                rescoring_root / f"{model_name}__fold_{fold}" / "job_report.json",
                f"{model_name}/fold {fold} rescoring report",
            )
            for scenario in split["validate"]:
                ranking = _ranking_metrics(paths[model_name][scenario], "raw_logit")
                recorded = job_report["scenarios"][scenario]
                if (
                    ranking["packets"] != int(recorded["edges"])
                    or not np.isclose(
                        ranking["average_precision"],
                        recorded["average_precision_raw_logit"],
                        rtol=0,
                        atol=1e-12,
                    )
                ):
                    raise ValueError(f"Rescored ranking metrics changed: {model_name}/{scenario}")
                ranking_scenarios[scenario] = {"fold": fold, **ranking}
                expected[scenario] = {
                    "rows": ranking["packets"],
                    "attack_packets": ranking["attack_packets"],
                }
                summaries[scenario] = _summarize_scenario(
                    paths[model_name][scenario],
                    score_field="raw_logit",
                    expected_rows=ranking["packets"],
                    expected_attack_packets=ranking["attack_packets"],
                )
        attack_counts = {
            scenario: item["attack_packets"]
            for scenario, item in ranking_scenarios.items()
        }
        if reference_attack_counts is None:
            reference_attack_counts = attack_counts
        elif attack_counts != reference_attack_counts:
            raise ValueError(f"Attack counts differ for {model_name}.")
        ranking_aggregate = _fold_macro(
            ranking_scenarios, folds, ("average_precision", "roc_auc")
        )
        thresholds = {}
        for name, budget in budgets:
            threshold = _select_threshold(summaries, folds, budget)
            threshold["target_false_alert_windows_per_hour"] = budget
            threshold["threshold_score_field"] = "raw_logit"
            threshold["threshold_probability_float64_diagnostic"] = _sigmoid_float64(
                threshold["threshold"]
            )
            thresholds[name] = threshold
        packet_counts = {
            scenario: _packet_counts_by_threshold(
                paths[model_name][scenario],
                score_field="raw_logit",
                thresholds={
                    name: item["threshold"] for name, item in thresholds.items()
                },
                expected_rows=expected[scenario]["rows"],
                batch_size=batch_size,
            )
            for scenario in scenarios
        }
        budget_results = {}
        for budget_name, _ in budgets:
            threshold = thresholds[budget_name]["threshold"]
            raw_scenario_metrics = {}
            raw_iteration_rows = []
            for fold, split in folds.items():
                for scenario in split["validate"]:
                    scenario_metrics, iterations = evaluate_scenario(
                        summaries[scenario],
                        packet_counts[scenario][budget_name],
                        threshold,
                        scenario,
                        fold,
                    )
                    raw_scenario_metrics[scenario] = scenario_metrics
                    raw_iteration_rows.extend(iterations)
            budget_results[budget_name] = _operational_metrics(
                raw_scenario_metrics=raw_scenario_metrics,
                raw_iteration_rows=raw_iteration_rows,
                folds=folds,
                terminal_steps=terminal_steps,
            )
        result[model_name] = {
            "ranking": {
                "scenario_metrics": ranking_scenarios,
                **ranking_aggregate,
            },
            "thresholds": thresholds,
            "budgets": budget_results,
        }
    return result


def _evaluate_xgb_benchmarks(
    *,
    paths: dict[str, dict[str, Path]],
    operational: dict,
    early_warning: dict,
    folds: dict[str, dict],
    terminal_steps: dict[str, list[str]],
) -> dict:
    result = {}
    for model_name in XGB_MODELS:
        ranking_scenarios = {}
        for fold, split in folds.items():
            for scenario in split["validate"]:
                ranking_scenarios[scenario] = {
                    "fold": fold,
                    **_ranking_metrics(paths[model_name][scenario], "score"),
                }
        ranking_aggregate = _fold_macro(
            ranking_scenarios, folds, ("average_precision", "roc_auc")
        )
        budget_results = {}
        for budget_name in operational["budget_order"]:
            source = operational["models"][model_name]["budgets"][budget_name]
            corrected = _operational_metrics(
                raw_scenario_metrics=source["scenario_metrics"],
                raw_iteration_rows=source["iteration_rows"],
                folds=folds,
                terminal_steps=terminal_steps,
            )
            audited = early_warning["models"][model_name]["budgets"][budget_name]
            for field in (
                "sequence_detection_rate",
                "timely_iteration_rate",
                "early_before_terminal_action",
            ):
                if not np.isclose(
                    corrected["hierarchical_macro"][field],
                    audited["hierarchical_macro"][
                        "score_positive_iteration_rate"
                        if field == "sequence_detection_rate"
                        else field
                    ],
                    rtol=0,
                    atol=1e-12,
                ):
                    raise ValueError(
                        f"XGB early-warning metric changed: {model_name}/{budget_name}/{field}"
                    )
            budget_results[budget_name] = corrected
        result[model_name] = {
            "ranking": {
                "scenario_metrics": ranking_scenarios,
                **ranking_aggregate,
            },
            "thresholds": operational["models"][model_name]["thresholds"],
            "budgets": budget_results,
        }
    return result


def _summary_rows(models: dict, family: str, display_names: dict) -> tuple[list, list, list]:
    ranking_rows = []
    primary_rows = []
    sensitivity_rows = []
    for model_name, model in models.items():
        ranking_rows.append({
            "family": family,
            "model": model_name,
            "model_display": display_names[model_name],
            **model["ranking"]["hierarchical_macro"],
        })
        primary = model["budgets"][PRIMARY_BUDGET]
        threshold = model["thresholds"][PRIMARY_BUDGET]
        primary_rows.append({
            "family": family,
            "model": model_name,
            "model_display": display_names[model_name],
            "score_field": "raw_logit" if family == "graph" else "score",
            "threshold": threshold["threshold"],
            "worst_fold_false_alert_windows_per_hour": threshold[
                "worst_fold_false_alert_windows_per_hour"
            ],
            **primary["hierarchical_macro"],
            **primary["pooled_lead_time_diagnostics"],
        })
        for budget_name, budget in model["budgets"].items():
            threshold = model["thresholds"][budget_name]
            sensitivity_rows.append({
                "family": family,
                "model": model_name,
                "model_display": display_names[model_name],
                "budget": budget_name,
                "target_false_alert_windows_per_hour": threshold[
                    "target_false_alert_windows_per_hour"
                ],
                "threshold": threshold["threshold"],
                "worst_fold_false_alert_windows_per_hour": threshold[
                    "worst_fold_false_alert_windows_per_hour"
                ],
                **budget["hierarchical_macro"],
                **budget["pooled_lead_time_diagnostics"],
            })
    return ranking_rows, primary_rows, sensitivity_rows


def _correction_impact_rows(graph_models: dict, source_comparison: dict) -> list[dict]:
    old_ranking = {
        row["model"]: row for row in source_comparison["tables"]["ranking_summary"]
    }
    old_primary = {
        row["model"]: row
        for row in source_comparison["tables"]["primary_operational_summary"]
    }
    rows = []
    for model_name in MODEL_NAMES:
        ranking = graph_models[model_name]["ranking"]["hierarchical_macro"]
        operational = graph_models[model_name]["budgets"][PRIMARY_BUDGET][
            "hierarchical_macro"
        ]
        threshold = graph_models[model_name]["thresholds"][PRIMARY_BUDGET]
        rows.append({
            "model": model_name,
            "model_display": DISPLAY_NAMES[model_name],
            "old_float32_average_precision": old_ranking[model_name][
                "average_precision"
            ],
            "raw_logit_average_precision": ranking["average_precision"],
            "average_precision_delta": (
                ranking["average_precision"]
                - old_ranking[model_name]["average_precision"]
            ),
            "old_float32_packet_f1": old_primary[model_name]["packet_f1"],
            "raw_logit_packet_f1": operational["packet_f1"],
            "old_float32_timely_iteration_rate": old_primary[model_name][
                "timely_iteration_rate"
            ],
            "raw_logit_timely_iteration_rate": operational[
                "timely_iteration_rate"
            ],
            "old_float32_worst_fold_false_alert_windows_per_hour": old_primary[
                model_name
            ]["worst_fold_false_alert_windows_per_hour"],
            "raw_logit_worst_fold_false_alert_windows_per_hour": threshold[
                "worst_fold_false_alert_windows_per_hour"
            ],
        })
    return rows


def _benchmark_delta_rows(graph_models: dict, xgb_models: dict) -> list[dict]:
    rows = []
    metrics = (
        ("average_precision", "ranking"),
        ("roc_auc", "ranking"),
        ("packet_recall", "operational"),
        ("packet_f1", "operational"),
        ("packet_f2", "operational"),
        ("sequence_detection_rate", "operational"),
        ("timely_iteration_rate", "operational"),
        ("early_before_terminal_action", "operational"),
    )
    for graph_name, graph in graph_models.items():
        for reference_name in ("xgb_p", "history"):
            reference = xgb_models[reference_name]
            for metric, source in metrics:
                if source == "ranking":
                    graph_summary = graph["ranking"]["hierarchical_macro"]
                    reference_summary = reference["ranking"]["hierarchical_macro"]
                    graph_scenarios = graph["ranking"]["scenario_metrics"]
                    reference_scenarios = reference["ranking"]["scenario_metrics"]
                else:
                    graph_budget = graph["budgets"][PRIMARY_BUDGET]
                    reference_budget = reference["budgets"][PRIMARY_BUDGET]
                    graph_summary = graph_budget["hierarchical_macro"]
                    reference_summary = reference_budget["hierarchical_macro"]
                    graph_scenarios = graph_budget["scenario_metrics"]
                    reference_scenarios = reference_budget["scenario_metrics"]
                deltas = [
                    float(graph_scenarios[scenario][metric])
                    - float(reference_scenarios[scenario][metric])
                    for scenario in graph_scenarios
                ]
                rows.append({
                    "graph_model": graph_name,
                    "graph_model_display": DISPLAY_NAMES[graph_name],
                    "xgb_reference": reference_name,
                    "xgb_reference_display": XGB_DISPLAY_NAMES[reference_name],
                    "metric": metric,
                    "graph_hierarchical_macro": graph_summary[metric],
                    "xgb_hierarchical_macro": reference_summary[metric],
                    "graph_minus_xgb_delta": (
                        graph_summary[metric] - reference_summary[metric]
                    ),
                    "scenarios_with_positive_delta": sum(value > 0 for value in deltas),
                    "scenarios_with_zero_delta": sum(value == 0 for value in deltas),
                    "scenarios_with_negative_delta": sum(value < 0 for value in deltas),
                    "causal_architecture_contrast": False,
                })
    return rows


def _corrected_resource_rows(
    source_comparison: dict, rescoring_manifest: dict
) -> list[dict]:
    source = {
        row["model"]: row
        for row in source_comparison["tables"]["resource_summary"]
    }
    result = []
    for model_name in MODEL_NAMES:
        jobs = [
            item for item in rescoring_manifest["jobs"].values()
            if item["model"] == model_name
        ]
        packets = sum(int(item["total_edges"]) for item in jobs)
        inference = sum(float(item["pure_inference_seconds"]) for item in jobs)
        result.append({
            **source[model_name],
            "source_training_and_original_job_wall_seconds": source[model_name][
                "total_wall_seconds"
            ],
            "corrected_oof_inference_seconds": inference,
            "corrected_oof_seconds_per_million_packets": (
                inference * 1_000_000 / packets
            ),
        })
    return result


def _markdown_table(rows: list[dict], columns: list[tuple[str, str]], digits: int = 4) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.{digits}f}")
            elif value is None:
                values.append("NA")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_review(output_dir: Path, report: dict) -> None:
    tables = report["tables"]
    graph_contrasts = [
        row for row in tables["pairwise_comparisons"]
        if row["metric"] in {"average_precision", "timely_iteration_rate"}
    ]
    text = "\n".join([
        "# cAPTure corrected graph-logit development comparison",
        "",
        "Status: **exploratory corrected comparison complete**.",
        "",
        "This report replaces saturated float32 graph probabilities with raw logits for "
        "ranking and operational thresholding. It uses only checksum-verified seed-42 "
        "development OOF predictions. No checkpoint is loaded and no model is trained.",
        "",
        "## Corrected ranking metrics and XGB benchmarks",
        "",
        _markdown_table(tables["combined_ranking_summary"], [
            ("family", "Family"),
            ("model_display", "Model"),
            ("average_precision", "Hierarchical AP"),
            ("roc_auc", "Hierarchical ROC-AUC"),
        ], 6),
        "",
        "## Primary operational budget",
        "",
        "Each model retains its own development-OOF threshold under the frozen "
        "one-false-alert-window-per-hour budget. XGB is an external development "
        "benchmark, not an architecture-matched ablation.",
        "",
        _markdown_table(tables["combined_primary_operational_summary"], [
            ("family", "Family"),
            ("model_display", "Model"),
            ("worst_fold_false_alert_windows_per_hour", "Worst-fold FA/h"),
            ("packet_precision", "Precision"),
            ("packet_recall", "Recall"),
            ("packet_f1", "F1"),
            ("packet_f2", "F2"),
            ("sequence_detection_rate", "Iteration coverage"),
            ("timely_iteration_rate", "Timely coverage"),
            ("early_before_terminal_action", "Preterminal coverage"),
        ], 4),
        "",
        "## Effect of replacing stored float32 probabilities",
        "",
        _markdown_table(tables["graph_score_correction_impact"], [
            ("model_display", "Model"),
            ("old_float32_average_precision", "Old AP"),
            ("raw_logit_average_precision", "Logit AP"),
            ("old_float32_packet_f1", "Old F1"),
            ("raw_logit_packet_f1", "Logit F1"),
            ("old_float32_timely_iteration_rate", "Old timely"),
            ("raw_logit_timely_iteration_rate", "Logit timely"),
        ], 4),
        "",
        "## Predeclared graph contrasts",
        "",
        _markdown_table(graph_contrasts, [
            ("comparison", "Comparison"),
            ("metric", "Metric"),
            ("hierarchical_macro_delta", "Delta"),
            ("scenarios_with_positive_delta", "Positive scenarios"),
            ("scenarios_with_negative_delta", "Negative scenarios"),
        ], 6),
        "",
        "## Interpretation boundaries",
        "",
        "- Graph operational results in this report supersede the saturated operational "
        "rows in the original four-model comparison.",
        "- XGB and graph models share packets, folds, windows, labels, alert budgets, and "
        "evaluation rules, but they do not share an architecture or identical feature "
        "representation. Their differences are benchmarking evidence, not causal ablations.",
        "- `StaticGNN - Edge MLP` is the current message-passing signal without recurrent memory.",
        "- `ST-GNN - EdgeGRU` remains only a preliminary temporal message-passing contrast.",
        "- The matched ST-GNN control without GAT message passing, the no-direct-edge-attribute "
        "control, and shifted-origin sensitivity remain pending.",
        "- One seed and reused development OOF calibration make every result exploratory.",
        "",
        "Test1, Test2, `train_pub_exf`, and `train_user_prop` were not accessed.",
        "",
    ])
    (output_dir / "review.md").write_text(text, encoding="utf-8")


def run_graph_logit_comparison(
    *,
    comparison_config_path: str | Path,
    rescoring_config_path: str | Path,
    rescoring_root: str | Path,
    source_comparison_dir: str | Path,
    saturation_audit_dir: str | Path,
    manifest_path: str | Path,
    training_config_path: str | Path,
    terminal_policy_path: str | Path,
    xgb_operational_dir: str | Path,
    xgb_early_warning_dir: str | Path,
    xgb_run_dirs: dict[str, Path],
    output_dir: str | Path,
    batch_size: int = 250_000,
) -> dict:
    """Run the corrected graph comparison and immutable XGB benchmark."""
    comparison_config_path = Path(comparison_config_path).expanduser().resolve()
    rescoring_config_path = Path(rescoring_config_path).expanduser().resolve()
    rescoring_root = Path(rescoring_root).expanduser().resolve()
    source_comparison_dir = Path(source_comparison_dir).expanduser().resolve()
    saturation_audit_dir = Path(saturation_audit_dir).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    training_config_path = Path(training_config_path).expanduser().resolve()
    terminal_policy_path = Path(terminal_policy_path).expanduser().resolve()
    xgb_operational_dir = Path(xgb_operational_dir).expanduser().resolve()
    xgb_early_warning_dir = Path(xgb_early_warning_dir).expanduser().resolve()
    xgb_run_dirs = {
        name: Path(path).expanduser().resolve() for name, path in xgb_run_dirs.items()
    }
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("The corrected comparison output directory must be new.")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    config = load_graph_logit_comparison_config(comparison_config_path)
    if rescoring_config_path.name != config["bindings"]["graph_rescoring_contract"]:
        raise ValueError("The bound rescoring contract changed.")
    _validate_bound_paths(
        config=config,
        rescoring_root=rescoring_root,
        source_comparison_dir=source_comparison_dir,
        saturation_audit_dir=saturation_audit_dir,
        xgb_operational_dir=xgb_operational_dir,
        xgb_early_warning_dir=xgb_early_warning_dir,
        xgb_run_dirs=xgb_run_dirs,
    )
    rescoring_manifest = validate_capture_graph_oof_rescoring(
        config_path=rescoring_config_path,
        comparison_dir=source_comparison_dir,
        saturation_audit_dir=saturation_audit_dir,
        output_root=rescoring_root,
    )
    source_comparison = _load_json(
        source_comparison_dir / "comparison_report.json",
        "source graph comparison",
    )
    manifest = load_manifest(manifest_path)
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    folds = manifest["validation"]["folds"]
    scenarios = {
        scenario for split in folds.values() for scenario in split["validate"]
    }
    policy = _load_terminal_policy(terminal_policy_path, scenarios)
    budgets = operational_budgets(manifest)
    if [name for name, _ in budgets] != [
        PRIMARY_BUDGET,
        *config["scoring"]["sensitivity_budgets"],
    ]:
        raise ValueError("The false-alert budget order changed.")
    xgb_operational = validate_operational_run(
        xgb_operational_dir, manifest_path, xgb_run_dirs
    )
    xgb_early_warning = validate_early_warning_audit(
        operational_dir=xgb_operational_dir,
        manifest_path=manifest_path,
        policy_path=terminal_policy_path,
        output_dir=xgb_early_warning_dir,
    )
    graph_paths = _graph_paths(rescoring_root, rescoring_manifest)
    _, xgb_paths = _xgb_paths(manifest, xgb_run_dirs)
    aligned_counts = _validate_cross_family_alignment(
        graph_paths=graph_paths, xgb_paths=xgb_paths
    )
    graph_models = _evaluate_graph_models(
        paths=graph_paths,
        rescoring_root=rescoring_root,
        folds=folds,
        budgets=budgets,
        terminal_steps=policy["terminal_action_steps"],
        batch_size=batch_size,
    )
    xgb_models = _evaluate_xgb_benchmarks(
        paths=xgb_paths,
        operational=xgb_operational,
        early_warning=xgb_early_warning,
        folds=folds,
        terminal_steps=policy["terminal_action_steps"],
    )
    graph_result = {
        "budget_order": [name for name, _ in budgets],
        "models": graph_models,
    }
    graph_tables = _build_graph_tables(graph_result, training_config)
    graph_ranking, graph_primary, graph_sensitivity = _summary_rows(
        graph_models, "graph", DISPLAY_NAMES
    )
    xgb_ranking, xgb_primary, xgb_sensitivity = _summary_rows(
        xgb_models, "xgb", XGB_DISPLAY_NAMES
    )
    tables = {
        **graph_tables,
        "combined_ranking_summary": graph_ranking + xgb_ranking,
        "combined_primary_operational_summary": graph_primary + xgb_primary,
        "combined_budget_sensitivity_summary": (
            graph_sensitivity + xgb_sensitivity
        ),
        "graph_score_correction_impact": _correction_impact_rows(
            graph_models, source_comparison
        ),
        "graph_vs_xgb_benchmark_deltas": _benchmark_delta_rows(
            graph_models, xgb_models
        ),
        "corrected_graph_resource_summary": _corrected_resource_rows(
            source_comparison, rescoring_manifest
        ),
    }
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=manifest_path.parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    worktree = subprocess.run(
        ["git", "status", "--short"],
        cwd=manifest_path.parent.parent,
        capture_output=True,
        text=True,
        check=False,
    )
    report = {
        "report_version": REPORT_VERSION,
        "status": "exploratory_graph_logit_comparison_complete",
        "run_id": output_dir.name,
        "models_compared": list(MODEL_NAMES),
        "xgb_benchmarks": list(XGB_MODELS),
        "operational_graph_score_field": "raw_logit",
        "graph_probability_field_is_diagnostic_only": True,
        "thresholds_selected_from": "development_oof_only",
        "threshold_evaluation_reuses_calibration_oof": True,
        "xgb_is_external_development_benchmark_not_architecture_ablation": True,
        "one_seed_exploratory": True,
        "topology_claim_authorized": False,
        "model_training_performed": False,
        "checkpoint_loading_performed": False,
        "held_out_scenarios_accessed": False,
        "comparison_contract_sha256": sha256_file(comparison_config_path),
        "rescoring_manifest_sha256": sha256_file(
            rescoring_root / "rescoring_manifest.json"
        ),
        "source_graph_comparison_report_sha256": sha256_file(
            source_comparison_dir / "comparison_report.json"
        ),
        "saturation_audit_report_sha256": sha256_file(
            saturation_audit_dir / "score_saturation_audit.json"
        ),
        "xgb_operational_report_sha256": sha256_file(
            xgb_operational_dir / "operational_report.json"
        ),
        "xgb_early_warning_report_sha256": sha256_file(
            xgb_early_warning_dir / "early_warning_report.json"
        ),
        "manifest_sha256": sha256_file(manifest_path),
        "training_config_sha256": sha256_file(training_config_path),
        "terminal_policy_sha256": sha256_file(terminal_policy_path),
        "comparison_code_sha256": sha256_file(MODULE_PATH),
        "dependency_code_sha256": {
            "capture_graph_oof_rescoring.py": sha256_file(
                Path(validate_capture_graph_oof_rescoring.__code__.co_filename)
            ),
            "capture_graph_comparison.py": sha256_file(
                Path(capture_graph_comparison.__file__)
            ),
            "capture_oof_operational.py": sha256_file(
                Path(capture_oof_operational.__file__)
            ),
            "capture_early_warning.py": sha256_file(
                Path(capture_early_warning.__file__)
            ),
        },
        "graph_input_oof_sha256": {
            model_name: {
                scenario: rescoring_manifest["scenario_outputs"][model_name][scenario][
                    "sha256"
                ]
                for scenario in scenarios
            }
            for model_name in MODEL_NAMES
        },
        "xgb_input_oof_sha256": xgb_operational["input_oof_sha256"],
        "aligned_graph_xgb_packet_counts": aligned_counts,
        "budget_order": [name for name, _ in budgets],
        "budgets_per_hour": {name: value for name, value in budgets},
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
        "git_worktree_status": worktree.stdout if worktree.returncode == 0 else "unavailable",
        "graph_models": graph_models,
        "xgb_models": xgb_models,
        "tables": tables,
    }
    output_dir.mkdir(parents=True)
    write_json(output_dir / "comparison_report.json", report)
    for name, rows in tables.items():
        pd.DataFrame(rows).to_csv(output_dir / f"{name}.csv", index=False)
    _write_review(output_dir, report)
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.name != "run_status.json"
    }
    write_json(output_dir / "run_status.json", {
        "complete": True,
        "report_sha256": sha256_file(output_dir / "comparison_report.json"),
        "artifact_sha256": artifact_hashes,
        "model_training_performed": False,
        "checkpoint_loading_performed": False,
        "held_out_scenarios_accessed": False,
    })
    return report


def validate_graph_logit_comparison(
    *,
    comparison_config_path: str | Path,
    rescoring_config_path: str | Path,
    rescoring_root: str | Path,
    source_comparison_dir: str | Path,
    saturation_audit_dir: str | Path,
    manifest_path: str | Path,
    training_config_path: str | Path,
    terminal_policy_path: str | Path,
    xgb_operational_dir: str | Path,
    xgb_early_warning_dir: str | Path,
    xgb_run_dirs: dict[str, Path],
    output_dir: str | Path,
) -> dict:
    """Validate a completed corrected comparison and all bound reports."""
    comparison_config_path = Path(comparison_config_path).expanduser().resolve()
    rescoring_config_path = Path(rescoring_config_path).expanduser().resolve()
    rescoring_root = Path(rescoring_root).expanduser().resolve()
    source_comparison_dir = Path(source_comparison_dir).expanduser().resolve()
    saturation_audit_dir = Path(saturation_audit_dir).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    training_config_path = Path(training_config_path).expanduser().resolve()
    terminal_policy_path = Path(terminal_policy_path).expanduser().resolve()
    xgb_operational_dir = Path(xgb_operational_dir).expanduser().resolve()
    xgb_early_warning_dir = Path(xgb_early_warning_dir).expanduser().resolve()
    xgb_run_dirs = {
        name: Path(path).expanduser().resolve() for name, path in xgb_run_dirs.items()
    }
    output_dir = Path(output_dir).expanduser().resolve()
    config = load_graph_logit_comparison_config(comparison_config_path)
    _validate_bound_paths(
        config=config,
        rescoring_root=rescoring_root,
        source_comparison_dir=source_comparison_dir,
        saturation_audit_dir=saturation_audit_dir,
        xgb_operational_dir=xgb_operational_dir,
        xgb_early_warning_dir=xgb_early_warning_dir,
        xgb_run_dirs=xgb_run_dirs,
    )
    validate_capture_graph_oof_rescoring(
        config_path=rescoring_config_path,
        comparison_dir=source_comparison_dir,
        saturation_audit_dir=saturation_audit_dir,
        output_root=rescoring_root,
    )
    validate_operational_run(xgb_operational_dir, manifest_path, xgb_run_dirs)
    validate_early_warning_audit(
        operational_dir=xgb_operational_dir,
        manifest_path=manifest_path,
        policy_path=terminal_policy_path,
        output_dir=xgb_early_warning_dir,
    )
    report_path = output_dir / "comparison_report.json"
    status = _load_json(output_dir / "run_status.json", "comparison status")
    report = _load_json(report_path, "corrected comparison report")
    expected_dependencies = {
        "capture_graph_oof_rescoring.py": sha256_file(
            Path(validate_capture_graph_oof_rescoring.__code__.co_filename)
        ),
        "capture_graph_comparison.py": sha256_file(
            Path(capture_graph_comparison.__file__)
        ),
        "capture_oof_operational.py": sha256_file(
            Path(capture_oof_operational.__file__)
        ),
        "capture_early_warning.py": sha256_file(
            Path(capture_early_warning.__file__)
        ),
    }
    if (
        status.get("complete") is not True
        or status.get("report_sha256") != sha256_file(report_path)
        or report.get("report_version") != REPORT_VERSION
        or report.get("status") != "exploratory_graph_logit_comparison_complete"
        or report.get("models_compared") != list(MODEL_NAMES)
        or report.get("xgb_benchmarks") != list(XGB_MODELS)
        or report.get("operational_graph_score_field") != "raw_logit"
        or report.get("model_training_performed") is not False
        or report.get("checkpoint_loading_performed") is not False
        or report.get("held_out_scenarios_accessed") is not False
        or report.get("comparison_contract_sha256")
        != sha256_file(comparison_config_path)
        or report.get("rescoring_manifest_sha256")
        != sha256_file(rescoring_root / "rescoring_manifest.json")
        or report.get("source_graph_comparison_report_sha256")
        != sha256_file(source_comparison_dir / "comparison_report.json")
        or report.get("saturation_audit_report_sha256")
        != sha256_file(saturation_audit_dir / "score_saturation_audit.json")
        or report.get("xgb_operational_report_sha256")
        != sha256_file(xgb_operational_dir / "operational_report.json")
        or report.get("xgb_early_warning_report_sha256")
        != sha256_file(xgb_early_warning_dir / "early_warning_report.json")
        or report.get("manifest_sha256") != sha256_file(manifest_path)
        or report.get("training_config_sha256") != sha256_file(training_config_path)
        or report.get("terminal_policy_sha256") != sha256_file(terminal_policy_path)
        or report.get("comparison_code_sha256") != sha256_file(MODULE_PATH)
        or report.get("dependency_code_sha256") != expected_dependencies
    ):
        raise ValueError("The corrected graph-logit comparison has unexpected provenance.")
    for name, expected in status["artifact_sha256"].items():
        if sha256_file(output_dir / name) != expected:
            raise ValueError(f"Corrected comparison artifact changed: {name}")
    return report
