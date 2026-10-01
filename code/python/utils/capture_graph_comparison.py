"""Development-only comparison of the first four cAPTure graph models."""

from __future__ import annotations

from collections import defaultdict
from itertools import zip_longest
import json
from pathlib import Path
import subprocess
from statistics import mean, median

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml

from .capture_data import load_manifest, sha256_file, write_json
from .capture_oof_operational import (
    _packet_counts_by_threshold,
    evaluate_scenario,
    operational_budgets,
    select_threshold,
    summarize_scenario_oof,
)


REPORT_VERSION = 1
MODEL_NAMES = ("edge_mlp", "edge_gru", "static_gnn", "st_gnn")
DISPLAY_NAMES = {
    "edge_mlp": "Edge MLP",
    "edge_gru": "EdgeGRU",
    "static_gnn": "StaticGNN",
    "st_gnn": "ST-GNN",
}
PRIMARY_BUDGET = "one_per_hour"
NANOSECONDS_PER_SECOND = 1_000_000_000
OOF_ALIGNMENT_COLUMNS = (
    "packet_id",
    "source_row_id",
    "window_index",
    "window_start_ns",
    "window_end_ns",
    "decision_time_ns",
    "packet_timestamp_ns",
    "binary_label",
    "attack_step",
    "sequence_id",
)
COMPARISONS = (
    {
        "name": "temporal_memory_without_gat",
        "candidate": "edge_gru",
        "control": "edge_mlp",
        "interpretation": "Incremental per-endpoint temporal memory without GAT message passing.",
    },
    {
        "name": "current_message_passing_without_memory",
        "candidate": "static_gnn",
        "control": "edge_mlp",
        "interpretation": "Current endpoint aggregation and GAT message passing without recurrent memory.",
    },
    {
        "name": "temporal_memory_with_gat",
        "candidate": "st_gnn",
        "control": "static_gnn",
        "interpretation": "Incremental temporal memory when GAT message passing is present.",
    },
    {
        "name": "gat_signal_in_temporal_models",
        "candidate": "st_gnn",
        "control": "edge_gru",
        "interpretation": (
            "Preliminary GAT/message-passing signal in temporal models; residual "
            "architecture differences prevent a causal attribution."
        ),
    },
)


def _load_json(path: Path, label: str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected {label} to contain a JSON object: {path}")
    return value


def _canonical_sha256(value: dict) -> str:
    import hashlib

    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_terminal_policy(path: Path, scenarios: set[str]) -> dict:
    policy = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {
        "audit_version": 1,
        "scope": "development_oof_only",
        "step_timely_rule": "first_correct_window_alert_strictly_before_last_malicious_packet",
        "chain_onset_rule": "first_malicious_packet_in_scenario",
        "terminal_onset_rule": "first_malicious_packet_of_any_declared_terminal_action_step",
        "chain_early_rule": "first_correct_window_alert_strictly_before_terminal_onset",
        "first_correct_alert_rule": "earliest_window_end_with_a_score_positive_malicious_packet",
    }
    if not isinstance(policy, dict) or set(policy) != {*expected, "terminal_action_steps"}:
        raise ValueError("The early-warning policy has unexpected fields.")
    for name, value in expected.items():
        if policy[name] != value:
            raise ValueError(f"The early-warning policy changed: {name}")
    if set(policy["terminal_action_steps"]) != scenarios:
        raise ValueError("Terminal-action steps do not cover the development scenarios.")
    return policy


def _validate_training_inputs(
    *,
    training_root: Path,
    binding_dir: Path,
    training_config_path: Path,
    manifest_path: Path,
) -> tuple[dict, dict, dict[str, dict[str, dict]]]:
    training_root = Path(training_root)
    binding_dir = Path(binding_dir)
    training_config_path = Path(training_config_path)
    manifest = load_manifest(manifest_path)
    training_config = yaml.safe_load(training_config_path.read_text(encoding="utf-8"))
    if training_config.get("scope") != "development_only":
        raise ValueError("The graph-training configuration is not development-only.")
    authorization = _load_json(
        binding_dir / "graph_training_authorization.json", "training authorization"
    )
    if (
        authorization.get("model_training_authorized") is not True
        or authorization.get("held_out_scenarios_authorized") is not False
    ):
        raise ValueError("The graph-training authorization has an unexpected scope.")
    manifest_folds = manifest["validation"]["folds"]
    training_folds = training_config["folds"]
    if set(manifest_folds) != set(training_folds):
        raise ValueError("Training and evaluation fold names differ.")
    for fold in manifest_folds:
        if (
            manifest_folds[fold]["train"] != training_folds[fold]["train"]
            or manifest_folds[fold]["validate"]
            != training_folds[fold]["outer_validation"]
        ):
            raise ValueError(f"Training and evaluation assignments differ for fold {fold}.")

    jobs: dict[str, dict[str, dict]] = {name: {} for name in MODEL_NAMES}
    config_sha256 = sha256_file(training_config_path)
    binding_report_sha256 = sha256_file(
        binding_dir / "training_runner_binding_report.json"
    )
    for model_name in MODEL_NAMES:
        for fold, split in training_folds.items():
            job_id = f"{model_name}__fold_{fold}"
            job_dir = training_root / job_id
            completion = _load_json(job_dir / "completion.json", f"{job_id} completion")
            metrics = _load_json(job_dir / "metrics.json", f"{job_id} metrics")
            resources = _load_json(
                job_dir / "resource_usage.json", f"{job_id} resource usage"
            )
            contract = _load_json(job_dir / "job_contract.json", f"{job_id} contract")
            if (
                completion.get("status") != "complete"
                or completion.get("model") != model_name
                or completion.get("fold") != fold
                or completion.get("held_out_scenarios_accessed") is not False
                or completion.get("threshold_selected") is not False
                or metrics.get("held_out_scenarios_accessed") is not False
                or metrics.get("threshold_selected") is not False
            ):
                raise ValueError(f"The completed job has unexpected scope: {job_id}")
            if (
                contract.get("job_id") != job_id
                or contract.get("model") != model_name
                or contract.get("fold") != fold
                or contract.get("binding_run_id") != binding_dir.name
                or contract.get("training_contract_sha256") != config_sha256
                or contract.get("binding_report_sha256") != binding_report_sha256
                or _canonical_sha256(contract) != completion.get("job_contract_sha256")
                or metrics.get("job_contract_sha256")
                != completion.get("job_contract_sha256")
            ):
                raise ValueError(f"The job contract changed: {job_id}")
            expected_scenarios = set(split["outer_validation"])
            scenario_metrics = metrics.get("outer_oof", {}).get("scenarios", {})
            if set(scenario_metrics) != expected_scenarios:
                raise ValueError(f"OOF scenarios differ for {job_id}.")
            observed_artifact_hashes = {
                relative: sha256_file(job_dir / relative)
                for relative in completion["artifact_checksums"]
            }
            for relative, expected_hash in completion["artifact_checksums"].items():
                if observed_artifact_hashes[relative] != expected_hash:
                    raise ValueError(f"Completed artifact changed: {job_id}/{relative}")
            oof_hashes = {}
            for scenario, item in scenario_metrics.items():
                relative = item["predictions"]
                if item["predictions_sha256"] != observed_artifact_hashes[relative]:
                    raise ValueError(f"OOF predictions changed: {job_id}/{scenario}")
                oof_hashes[scenario] = observed_artifact_hashes[relative]
            jobs[model_name][fold] = {
                "job_id": job_id,
                "job_dir": job_dir,
                "completion": completion,
                "metrics": metrics,
                "resources": resources,
                "contract": contract,
                "oof_hashes": oof_hashes,
            }
    return manifest, training_config, jobs


def _scenario_paths(
    jobs: dict[str, dict[str, dict]], folds: dict[str, dict]
) -> dict[str, dict[str, Path]]:
    result: dict[str, dict[str, Path]] = {name: {} for name in MODEL_NAMES}
    for model_name in MODEL_NAMES:
        for fold, split in folds.items():
            job = jobs[model_name][fold]
            scenario_metrics = job["metrics"]["outer_oof"]["scenarios"]
            for scenario in split["validate"]:
                result[model_name][scenario] = (
                    job["job_dir"] / scenario_metrics[scenario]["predictions"]
                )
    return result


def _validate_oof_alignment(paths: dict[str, dict[str, Path]]) -> dict[str, int]:
    """Require exact evaluation keys and labels for all four model outputs."""
    counts = {}
    for scenario in paths[MODEL_NAMES[0]]:
        parquet_files = {
            model: pq.ParquetFile(paths[model][scenario]) for model in MODEL_NAMES
        }
        rows = {model: item.metadata.num_rows for model, item in parquet_files.items()}
        if len(set(rows.values())) != 1:
            raise ValueError(f"OOF row counts differ for {scenario}: {rows}")
        iterators = [
            parquet_files[model].iter_batches(
                batch_size=100_000, columns=list(OOF_ALIGNMENT_COLUMNS)
            )
            for model in MODEL_NAMES
        ]
        for batches in zip_longest(*iterators):
            if any(batch is None for batch in batches):
                raise ValueError(f"OOF batch counts differ for {scenario}.")
            reference = batches[0]
            for model_name, candidate in zip(MODEL_NAMES[1:], batches[1:]):
                if candidate.num_rows != reference.num_rows:
                    raise ValueError(
                        f"OOF batch sizes differ for {model_name}/{scenario}."
                    )
                for column in OOF_ALIGNMENT_COLUMNS:
                    reference_index = reference.schema.get_field_index(column)
                    candidate_index = candidate.schema.get_field_index(column)
                    if (
                        reference_index < 0
                        or candidate_index < 0
                        or not reference.column(reference_index).equals(
                            candidate.column(candidate_index)
                        )
                    ):
                        raise ValueError(
                            f"OOF field {column} differs for {model_name}/{scenario}."
                        )
        counts[scenario] = int(next(iter(rows.values())))
    return counts


def _ranking_metrics(path: Path, expected_average_precision: float) -> dict:
    table = pq.read_table(path, columns=["binary_label", "score"])
    labels = table.column("binary_label").to_numpy(zero_copy_only=False)
    scores = table.column("score").to_numpy(zero_copy_only=False).astype(
        np.float64, copy=False
    )
    if not np.isin(labels, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError(f"Invalid labels or scores in {path}.")
    average_precision = float(average_precision_score(labels, scores))
    if not np.isclose(average_precision, expected_average_precision, rtol=0, atol=1e-12):
        raise ValueError(f"Stored average precision differs from OOF scores: {path}")
    return {
        "packets": int(len(labels)),
        "normal_packets": int(np.count_nonzero(labels == 0)),
        "attack_packets": int(np.count_nonzero(labels == 1)),
        "average_precision": average_precision,
        "roc_auc": float(roc_auc_score(labels, scores)),
    }


def _fold_macro(
    scenario_metrics: dict[str, dict], folds: dict[str, dict], fields: tuple[str, ...]
) -> dict:
    def optional_mean(values) -> float | None:
        values = list(values)
        if any(value is None for value in values):
            return None
        return float(mean(float(value) for value in values))

    fold_means = {
        fold: {
            field: optional_mean(
                scenario_metrics[scenario][field] for scenario in split["validate"]
            )
            for field in fields
        }
        for fold, split in folds.items()
    }
    return {
        "fold_means": fold_means,
        "hierarchical_macro": {
            field: optional_mean(fold_means[fold][field] for fold in folds)
            for field in fields
        },
        "flat_scenario_macro_diagnostic": {
            field: optional_mean(item[field] for item in scenario_metrics.values())
            for field in fields
        },
    }


def _classify_iteration(item: dict) -> dict:
    first_packet = int(item["first_malicious_packet_ns"])
    last_packet = int(item["last_malicious_packet_ns"])
    raw_alert = item["first_detecting_window_end_ns"]
    first_alert = None if raw_alert is None else int(raw_alert)
    timely = first_alert is not None and first_alert < last_packet
    return {
        **item,
        "first_detecting_window_end_ns": first_alert,
        "timely_before_last_malicious_packet": timely,
        "late_at_or_after_last_malicious_packet": (
            first_alert is not None and not timely
        ),
        "lead_seconds_before_last_malicious_packet": (
            (last_packet - first_alert) / NANOSECONDS_PER_SECOND if timely else None
        ),
    }


def _iteration_summary(items: list[dict]) -> dict:
    if not items:
        raise ValueError("An attack-step group has no iterations.")
    detected = [item for item in items if item["detected"]]
    timely = [item for item in items if item["timely_before_last_malicious_packet"]]
    return {
        "iterations": len(items),
        "detected_iterations": len(detected),
        "missed_iterations": len(items) - len(detected),
        "timely_iterations": len(timely),
        "late_positive_iterations": len(detected) - len(timely),
        "sequence_detection_rate": len(detected) / len(items),
        "timely_iteration_rate": len(timely) / len(items),
        "median_timely_lead_seconds": (
            float(median(item["lead_seconds_before_last_malicious_packet"] for item in timely))
            if timely
            else None
        ),
    }


def _chain_summary(
    items: list[dict], terminal_steps: list[str], scenario: str, fold: str
) -> dict:
    terminal = [item for item in items if item["attack_step"] in terminal_steps]
    if not terminal:
        raise ValueError(f"No declared terminal action appears in {scenario}.")
    chain_onset = min(int(item["first_malicious_packet_ns"]) for item in items)
    terminal_onset = min(int(item["first_malicious_packet_ns"]) for item in terminal)
    detected_alerts = [
        int(item["first_detecting_window_end_ns"])
        for item in items
        if item["first_detecting_window_end_ns"] is not None
    ]
    first_alert = min(detected_alerts) if detected_alerts else None
    early = first_alert is not None and first_alert < terminal_onset
    return {
        "scenario": scenario,
        "fold": fold,
        "declared_terminal_action_steps": terminal_steps,
        "chain_first_malicious_packet_ns": chain_onset,
        "terminal_action_first_malicious_packet_ns": terminal_onset,
        "preterminal_opportunity_seconds": (
            terminal_onset - chain_onset
        ) / NANOSECONDS_PER_SECOND,
        "first_correct_alert_ns": first_alert,
        "score_positive_chain": first_alert is not None,
        "early_before_terminal_action": early,
        "seconds_before_terminal_action": (
            (terminal_onset - first_alert) / NANOSECONDS_PER_SECOND if early else None
        ),
    }


def _operational_metrics(
    *,
    raw_scenario_metrics: dict[str, dict],
    raw_iteration_rows: list[dict],
    folds: dict[str, dict],
    terminal_steps: dict[str, list[str]],
) -> dict:
    scenario_folds = {
        scenario: fold
        for fold, split in folds.items()
        for scenario in split["validate"]
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for raw in raw_iteration_rows:
        grouped[raw["scenario"]].append(_classify_iteration(raw))
    if set(grouped) != set(scenario_folds):
        raise ValueError("Operational results omit a development scenario.")

    scenario_metrics = {}
    step_metrics = {}
    chain_metrics = {}
    all_iterations = []
    for scenario, fold in scenario_folds.items():
        items = grouped[scenario]
        iteration = _iteration_summary(items)
        raw = dict(raw_scenario_metrics[scenario])
        tp = int(raw["packet_true_positives"])
        fp = int(raw["packet_false_positives"])
        fn = int(raw["packet_false_negatives"])
        f1_denominator = 2 * tp + fp + fn
        f2_denominator = 5 * tp + fp + 4 * fn
        raw["packet_f1"] = 2 * tp / f1_denominator if f1_denominator else 0.0
        raw["packet_f2"] = 5 * tp / f2_denominator if f2_denominator else 0.0
        if (
            iteration["iterations"] != raw["attack_step_iterations"]
            or iteration["detected_iterations"] != raw["detected_iterations"]
        ):
            raise ValueError(f"Iteration counts differ for {scenario}.")
        chain = _chain_summary(items, terminal_steps[scenario], scenario, fold)
        chain_metrics[scenario] = chain
        scenario_metrics[scenario] = {
            **raw,
            **iteration,
            "early_before_terminal_action": chain["early_before_terminal_action"],
            "seconds_before_terminal_action": chain["seconds_before_terminal_action"],
        }
        by_step: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            by_step[item["attack_step"]].append(item)
            all_iterations.append(item)
        for step, step_items in by_step.items():
            step_metrics[f"{scenario}::{step}"] = {
                "scenario": scenario,
                "fold": fold,
                "attack_step": step,
                **_iteration_summary(step_items),
            }

    fields = (
        "false_alert_windows_per_hour",
        "packet_recall",
        "packet_precision",
        "packet_false_positive_rate",
        "packet_f1",
        "packet_f2",
        "sequence_detection_rate",
        "timely_iteration_rate",
        "early_before_terminal_action",
    )
    aggregate = _fold_macro(scenario_metrics, folds, fields)
    timely = [
        item["lead_seconds_before_last_malicious_packet"]
        for item in all_iterations
        if item["timely_before_last_malicious_packet"]
    ]
    early_leads = [
        item["seconds_before_terminal_action"]
        for item in chain_metrics.values()
        if item["early_before_terminal_action"]
    ]
    aggregate["pooled_lead_time_diagnostics"] = {
        "timely_iterations_with_lead": len(timely),
        "median_seconds_before_iteration_end": float(median(timely)) if timely else None,
        "early_chains_with_lead": len(early_leads),
        "median_seconds_before_terminal_action": (
            float(median(early_leads)) if early_leads else None
        ),
    }
    return {
        "scenario_metrics": scenario_metrics,
        "step_metrics": step_metrics,
        "chain_metrics": chain_metrics,
        "iteration_rows": all_iterations,
        **aggregate,
    }


def _resource_tables(jobs: dict[str, dict[str, dict]]) -> tuple[list[dict], list[dict]]:
    job_rows = []
    for model_name in MODEL_NAMES:
        for fold, job in jobs[model_name].items():
            completion = job["completion"]
            metrics = job["metrics"]
            resources = job["resources"]
            outer = metrics["outer_oof"]
            job_rows.append({
                "model": model_name,
                "model_display": DISPLAY_NAMES[model_name],
                "fold": fold,
                "best_epoch_count": int(completion["best_epoch_count"]),
                "parameters": int(completion["parameter_count"]),
                "checkpoint_mib": int(completion["final_checkpoint_bytes"]) / 1024**2,
                "wall_seconds": float(resources["wall_seconds"]),
                "peak_host_rss_gib": float(resources["peak_host_rss_gib"]),
                "peak_cuda_allocated_gib": resources.get("peak_cuda_allocated_gib"),
                "peak_cuda_reserved_gib": resources.get("peak_cuda_reserved_gib"),
                "oof_packets": int(outer["total_edges"]),
                "pure_inference_seconds": float(outer["pure_inference_seconds"]),
                "seconds_per_million_packets": (
                    float(outer["pure_inference_seconds"])
                    * 1_000_000
                    / int(outer["total_edges"])
                ),
            })
    model_rows = []
    for model_name in MODEL_NAMES:
        rows = [row for row in job_rows if row["model"] == model_name]
        packets = sum(row["oof_packets"] for row in rows)
        inference = sum(row["pure_inference_seconds"] for row in rows)
        model_rows.append({
            "model": model_name,
            "model_display": DISPLAY_NAMES[model_name],
            "fold_jobs": len(rows),
            "best_epoch_count_fold_A": next(
                row["best_epoch_count"] for row in rows if row["fold"] == "A"
            ),
            "best_epoch_count_fold_B": next(
                row["best_epoch_count"] for row in rows if row["fold"] == "B"
            ),
            "parameters": max(row["parameters"] for row in rows),
            "total_wall_seconds": sum(row["wall_seconds"] for row in rows),
            "maximum_peak_host_rss_gib": max(row["peak_host_rss_gib"] for row in rows),
            "maximum_peak_cuda_allocated_gib": max(
                float(row["peak_cuda_allocated_gib"] or 0) for row in rows
            ),
            "oof_packets": packets,
            "pure_inference_seconds": inference,
            "seconds_per_million_packets": inference * 1_000_000 / packets,
        })
    return job_rows, model_rows


def _build_tables(result: dict, training_config: dict) -> dict[str, list[dict]]:
    ranking_scenarios = []
    ranking_summary = []
    operational_scenarios = []
    operational_summary = []
    sensitivity_scenarios = []
    sensitivity_summary = []
    step_rows = []
    chain_rows = []
    for model_name in MODEL_NAMES:
        model = result["models"][model_name]
        for scenario, item in model["ranking"]["scenario_metrics"].items():
            ranking_scenarios.append({
                "model": model_name,
                "model_display": DISPLAY_NAMES[model_name],
                "scenario": scenario,
                **item,
            })
        ranking_summary.append({
            "model": model_name,
            "model_display": DISPLAY_NAMES[model_name],
            **model["ranking"]["hierarchical_macro"],
            "flat_average_precision_diagnostic": model["ranking"][
                "flat_scenario_macro_diagnostic"
            ]["average_precision"],
            "flat_roc_auc_diagnostic": model["ranking"][
                "flat_scenario_macro_diagnostic"
            ]["roc_auc"],
        })
        primary = model["budgets"][PRIMARY_BUDGET]
        threshold = model["thresholds"][PRIMARY_BUDGET]
        operational_summary.append({
            "model": model_name,
            "model_display": DISPLAY_NAMES[model_name],
            "threshold": threshold["threshold"],
            "worst_fold_false_alert_windows_per_hour": threshold[
                "worst_fold_false_alert_windows_per_hour"
            ],
            **primary["hierarchical_macro"],
            **primary["pooled_lead_time_diagnostics"],
        })
        for scenario, item in primary["scenario_metrics"].items():
            operational_scenarios.append({
                "model": model_name,
                "model_display": DISPLAY_NAMES[model_name],
                "scenario": scenario,
                **item,
            })
        for budget_name in result["budget_order"]:
            budget_result = model["budgets"][budget_name]
            budget_threshold = model["thresholds"][budget_name]
            sensitivity_summary.append({
                "model": model_name,
                "model_display": DISPLAY_NAMES[model_name],
                "budget": budget_name,
                "target_false_alert_windows_per_hour": budget_threshold[
                    "target_false_alert_windows_per_hour"
                ],
                "threshold": budget_threshold["threshold"],
                "worst_fold_false_alert_windows_per_hour": budget_threshold[
                    "worst_fold_false_alert_windows_per_hour"
                ],
                **budget_result["hierarchical_macro"],
                **budget_result["pooled_lead_time_diagnostics"],
            })
            for scenario, item in budget_result["scenario_metrics"].items():
                sensitivity_scenarios.append({
                    "model": model_name,
                    "model_display": DISPLAY_NAMES[model_name],
                    "budget": budget_name,
                    "target_false_alert_windows_per_hour": budget_threshold[
                        "target_false_alert_windows_per_hour"
                    ],
                    "scenario": scenario,
                    **item,
                })
        for item in primary["step_metrics"].values():
            step_rows.append({
                "model": model_name,
                "model_display": DISPLAY_NAMES[model_name],
                **item,
            })
        for item in primary["chain_metrics"].values():
            chain_rows.append({
                "model": model_name,
                "model_display": DISPLAY_NAMES[model_name],
                **item,
            })

    comparison_rows = []
    ranking_by_model = {row["model"]: row for row in ranking_summary}
    operational_by_model = {row["model"]: row for row in operational_summary}
    ranking_scenario_map = {
        (row["model"], row["scenario"]): row for row in ranking_scenarios
    }
    operational_scenario_map = {
        (row["model"], row["scenario"]): row for row in operational_scenarios
    }
    comparison_metrics = (
        ("average_precision", ranking_by_model, ranking_scenario_map),
        ("roc_auc", ranking_by_model, ranking_scenario_map),
        ("packet_recall", operational_by_model, operational_scenario_map),
        ("packet_f1", operational_by_model, operational_scenario_map),
        ("packet_f2", operational_by_model, operational_scenario_map),
        ("sequence_detection_rate", operational_by_model, operational_scenario_map),
        ("timely_iteration_rate", operational_by_model, operational_scenario_map),
        ("early_before_terminal_action", operational_by_model, operational_scenario_map),
    )
    scenarios = sorted({row["scenario"] for row in ranking_scenarios})
    for comparison in COMPARISONS:
        candidate = comparison["candidate"]
        control = comparison["control"]
        for metric, summary_map, scenario_map in comparison_metrics:
            deltas = [
                float(scenario_map[(candidate, scenario)][metric])
                - float(scenario_map[(control, scenario)][metric])
                for scenario in scenarios
            ]
            comparison_rows.append({
                "comparison": comparison["name"],
                "candidate": candidate,
                "control": control,
                "metric": metric,
                "candidate_hierarchical_macro": summary_map[candidate][metric],
                "control_hierarchical_macro": summary_map[control][metric],
                "hierarchical_macro_delta": (
                    summary_map[candidate][metric] - summary_map[control][metric]
                ),
                "scenarios_with_positive_delta": sum(value > 0 for value in deltas),
                "scenarios_with_zero_delta": sum(value == 0 for value in deltas),
                "scenarios_with_negative_delta": sum(value < 0 for value in deltas),
                "interpretation": comparison["interpretation"],
            })

    practical_ap_margin = training_config["pilot_decision"][
        "practical_average_precision_margin"
    ]
    practical_timely_margin = training_config["pilot_decision"][
        "practical_timely_iteration_coverage_margin"
    ]
    for row in comparison_rows:
        if row["metric"] == "average_precision":
            row["frozen_practical_margin"] = practical_ap_margin
        elif row["metric"] == "timely_iteration_rate":
            row["frozen_practical_margin"] = practical_timely_margin
        else:
            row["frozen_practical_margin"] = None
    return {
        "ranking_summary": ranking_summary,
        "ranking_by_scenario": ranking_scenarios,
        "primary_operational_summary": operational_summary,
        "primary_operational_by_scenario": operational_scenarios,
        "budget_sensitivity_summary": sensitivity_summary,
        "budget_sensitivity_by_scenario": sensitivity_scenarios,
        "primary_step_metrics": step_rows,
        "primary_chain_metrics": chain_rows,
        "pairwise_comparisons": comparison_rows,
    }


def _markdown_table(rows: list[dict], columns: list[tuple[str, str]], digits: int = 4) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, separator]
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


def _write_review(output_dir: Path, result: dict) -> None:
    ranking = result["tables"]["ranking_summary"]
    operational = result["tables"]["primary_operational_summary"]
    sensitivity = result["tables"]["budget_sensitivity_summary"]
    comparison = [
        row
        for row in result["tables"]["pairwise_comparisons"]
        if row["metric"] in {"average_precision", "timely_iteration_rate"}
    ]
    resources = result["tables"]["resource_summary"]
    text = "\n".join([
        "# cAPTure four-model development comparison",
        "",
        "Status: **exploratory partial comparison complete**.",
        "",
        "This report uses only checksum-verified development OOF predictions from seed 42. "
        "It does not access held-out scenarios and does not support a confirmatory claim.",
        "",
        "## Ranking metrics",
        "",
        _markdown_table(ranking, [
            ("model_display", "Model"),
            ("average_precision", "Hierarchical AP"),
            ("roc_auc", "Hierarchical ROC-AUC"),
        ], 6),
        "",
        "## Primary operational budget",
        "",
        "Each model receives its own development-OOF threshold under the frozen budget of "
        "one false-alert window per hour. Calibration and reporting use the same OOF "
        "predictions, so these values are screening estimates.",
        "",
        _markdown_table(operational, [
            ("model_display", "Model"),
            ("worst_fold_false_alert_windows_per_hour", "Worst-fold FA/h"),
            ("packet_f1", "Packet F1"),
            ("packet_f2", "Packet F2"),
            ("sequence_detection_rate", "Iteration coverage"),
            ("timely_iteration_rate", "Timely coverage"),
            ("early_before_terminal_action", "Preterminal scenario coverage"),
        ], 4),
        "",
        "## False-alert-budget sensitivity",
        "",
        "The one-per-hour budget remains primary. The one-per-12-hours and "
        "one-per-five-minutes budgets were frozen before this comparison and are "
        "reported only as sensitivity analyses; they do not replace the primary budget.",
        "",
        _markdown_table(sensitivity, [
            ("model_display", "Model"),
            ("budget", "Budget"),
            ("worst_fold_false_alert_windows_per_hour", "Worst-fold FA/h"),
            ("packet_recall", "Packet recall"),
            ("packet_f1", "Packet F1"),
            ("packet_f2", "Packet F2"),
            ("timely_iteration_rate", "Timely coverage"),
        ], 4),
        "",
        "## Predeclared contrasts",
        "",
        _markdown_table(comparison, [
            ("comparison", "Comparison"),
            ("metric", "Metric"),
            ("hierarchical_macro_delta", "Delta"),
            ("scenarios_with_positive_delta", "Positive scenarios"),
            ("scenarios_with_negative_delta", "Negative scenarios"),
        ], 6),
        "",
        "## Resource summary",
        "",
        _markdown_table(resources, [
            ("model_display", "Model"),
            ("total_wall_seconds", "Total job wall seconds"),
            ("seconds_per_million_packets", "Inference s/M packets"),
            ("maximum_peak_host_rss_gib", "Peak host GiB"),
            ("maximum_peak_cuda_allocated_gib", "Peak CUDA GiB"),
        ], 3),
        "",
        "## Interpretation boundary",
        "",
        "- `StaticGNN - Edge MLP` is the current aggregation/message-passing signal without recurrent memory.",
        "- `ST-GNN - EdgeGRU` is the temporal comparison, but it retains residual architecture differences.",
        "- `ST-GNN without GAT message passing` is still pending. It retains endpoint aggregation and per-node memory; it is not a model without all relational structure.",
        "- The no-direct-edge-attribute ablation and shifted-window-origin sensitivity are also pending.",
        "- A topology-centered proposal cannot be approved from this partial comparison alone.",
        "",
        "Test1, Test2, `train_pub_exf`, and `train_user_prop` were not accessed.",
        "",
    ])
    (output_dir / "review.md").write_text(text, encoding="utf-8")


def run_graph_four_model_comparison(
    *,
    training_root: str | Path,
    binding_dir: str | Path,
    training_config_path: str | Path,
    manifest_path: str | Path,
    terminal_policy_path: str | Path,
    output_dir: str | Path,
    batch_size: int = 250_000,
) -> dict:
    """Evaluate the four completed seed-42 graph models without retraining."""
    training_root = Path(training_root).expanduser().resolve()
    binding_dir = Path(binding_dir).expanduser().resolve()
    training_config_path = Path(training_config_path).expanduser().resolve()
    manifest_path = Path(manifest_path).expanduser().resolve()
    terminal_policy_path = Path(terminal_policy_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("The comparison output directory must be new.")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    manifest, training_config, jobs = _validate_training_inputs(
        training_root=training_root,
        binding_dir=binding_dir,
        training_config_path=training_config_path,
        manifest_path=manifest_path,
    )
    folds = manifest["validation"]["folds"]
    scenarios = {
        scenario for split in folds.values() for scenario in split["validate"]
    }
    policy = _load_terminal_policy(terminal_policy_path, scenarios)
    budgets = operational_budgets(manifest)
    paths = _scenario_paths(jobs, folds)
    aligned_counts = _validate_oof_alignment(paths)
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
    result = {
        "report_version": REPORT_VERSION,
        "status": "exploratory_four_model_development_comparison_complete",
        "run_id": output_dir.name,
        "models_compared": list(MODEL_NAMES),
        "models_pending": [
            "st_gnn_without_gat",
            "st_gnn_without_direct_edge_attr",
        ],
        "gat_ablation_display_name": "ST-GNN without GAT message passing",
        "gat_ablation_retains_endpoint_aggregation": True,
        "gat_ablation_retains_per_node_memory": True,
        "topology_claim_authorized": False,
        "one_seed_exploratory": True,
        "thresholds_selected_from": "development_oof_only",
        "threshold_evaluation_reuses_calibration_oof": True,
        "held_out_scenarios_accessed": False,
        "model_training_performed": False,
        "manifest_sha256": sha256_file(manifest_path),
        "training_config_sha256": sha256_file(training_config_path),
        "terminal_policy_sha256": sha256_file(terminal_policy_path),
        "comparison_code_sha256": sha256_file(Path(__file__)),
        "operational_helper_code_sha256": sha256_file(
            Path(__file__).with_name("capture_oof_operational.py")
        ),
        "binding_report_sha256": sha256_file(
            binding_dir / "training_runner_binding_report.json"
        ),
        "source_completion_sha256": {
            job["job_id"]: sha256_file(job["job_dir"] / "completion.json")
            for model_jobs in jobs.values()
            for job in model_jobs.values()
        },
        "input_oof_sha256": {
            model_name: {
                scenario: jobs[model_name][fold]["oof_hashes"][scenario]
                for fold, split in folds.items()
                for scenario in split["validate"]
            }
            for model_name in MODEL_NAMES
        },
        "aligned_oof_packet_counts": aligned_counts,
        "budget_order": [name for name, _ in budgets],
        "budgets_per_hour": {name: value for name, value in budgets},
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
        "git_worktree_status": worktree.stdout if worktree.returncode == 0 else "unavailable",
        "models": {},
    }

    reference_attack_counts = None
    for model_name in MODEL_NAMES:
        print(f"Evaluating immutable OOF predictions for {model_name}...", flush=True)
        ranking_scenarios = {}
        summaries = {}
        expected_items = {}
        for fold, split in folds.items():
            job = jobs[model_name][fold]
            scenario_items = job["metrics"]["outer_oof"]["scenarios"]
            for scenario in split["validate"]:
                item = scenario_items[scenario]
                ranking = _ranking_metrics(paths[model_name][scenario], item["average_precision"])
                ranking_scenarios[scenario] = {"fold": fold, **ranking}
                expected = {
                    "rows": ranking["packets"],
                    "attack_packets": ranking["attack_packets"],
                }
                expected_items[scenario] = expected
                summaries[scenario] = summarize_scenario_oof(
                    paths[model_name][scenario], expected
                )
        attack_counts = {
            scenario: item["attack_packets"] for scenario, item in ranking_scenarios.items()
        }
        if reference_attack_counts is None:
            reference_attack_counts = attack_counts
        elif attack_counts != reference_attack_counts:
            raise ValueError(f"Attack counts differ for {model_name}.")
        ranking_aggregate = _fold_macro(
            ranking_scenarios, folds, ("average_precision", "roc_auc")
        )
        thresholds = {
            name: {
                "target_false_alert_windows_per_hour": budget,
                **select_threshold(summaries, folds, budget),
            }
            for name, budget in budgets
        }
        packet_counts = {
            scenario: _packet_counts_by_threshold(
                paths[model_name][scenario],
                {name: item["threshold"] for name, item in thresholds.items()},
                expected_items[scenario],
                batch_size,
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
                terminal_steps=policy["terminal_action_steps"],
            )
        result["models"][model_name] = {
            "ranking": {
                "scenario_metrics": ranking_scenarios,
                **ranking_aggregate,
            },
            "thresholds": thresholds,
            "budgets": budget_results,
        }

    job_resources, resource_summary = _resource_tables(jobs)
    result["tables"] = _build_tables(result, training_config)
    result["tables"]["resource_by_job"] = job_resources
    result["tables"]["resource_summary"] = resource_summary

    output_dir.mkdir(parents=True)
    write_json(output_dir / "comparison_report.json", result)
    for table_name, rows in result["tables"].items():
        pd.DataFrame(rows).to_csv(output_dir / f"{table_name}.csv", index=False)
    _write_review(output_dir, result)
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
        "held_out_scenarios_accessed": False,
    })
    return result


def validate_graph_four_model_comparison(
    *,
    output_dir: str | Path,
    training_root: str | Path,
    binding_dir: str | Path,
    training_config_path: str | Path,
    manifest_path: str | Path,
    terminal_policy_path: str | Path,
) -> dict:
    """Validate an immutable completed comparison and its source completions."""
    output_dir = Path(output_dir).expanduser().resolve()
    report_path = output_dir / "comparison_report.json"
    status = _load_json(output_dir / "run_status.json", "comparison run status")
    report = _load_json(report_path, "comparison report")
    if (
        status.get("complete") is not True
        or status.get("report_sha256") != sha256_file(report_path)
        or report.get("report_version") != REPORT_VERSION
        or report.get("status")
        != "exploratory_four_model_development_comparison_complete"
        or report.get("models_compared") != list(MODEL_NAMES)
        or report.get("held_out_scenarios_accessed") is not False
        or report.get("model_training_performed") is not False
        or report.get("comparison_code_sha256") != sha256_file(Path(__file__))
        or report.get("operational_helper_code_sha256")
        != sha256_file(Path(__file__).with_name("capture_oof_operational.py"))
        or report.get("manifest_sha256") != sha256_file(Path(manifest_path))
        or report.get("training_config_sha256")
        != sha256_file(Path(training_config_path))
        or report.get("terminal_policy_sha256")
        != sha256_file(Path(terminal_policy_path))
    ):
        raise ValueError("The four-model comparison has unexpected provenance.")
    for name, expected in status["artifact_sha256"].items():
        if sha256_file(output_dir / name) != expected:
            raise ValueError(f"Comparison artifact changed: {name}")
    observed_completions = {
        job_id: sha256_file(Path(training_root) / job_id / "completion.json")
        for job_id in report["source_completion_sha256"]
    }
    if observed_completions != report["source_completion_sha256"]:
        raise ValueError("A source graph-training completion changed.")
    if report.get("binding_report_sha256") != sha256_file(
        Path(binding_dir) / "training_runner_binding_report.json"
    ):
        raise ValueError("The source training binding changed.")
    return report
