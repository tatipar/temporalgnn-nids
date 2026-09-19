"""Thresholded, window-aligned development OOF evaluation for cAPTure."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import subprocess

import duckdb
import numpy as np
import pyarrow.parquet as pq

from .capture_data import load_manifest, sha256_file, write_json
from .capture_xgb_p import validate_xgb_p_fold_run
from .capture_xgb_p_t import validate_xgb_p_t_fold_run


MODEL_NAMES = ("xgb_p", "current_window", "history", "full")
REPORT_VERSION = 1
WINDOW_SECONDS = 5
NANOSECONDS_PER_SECOND = 1_000_000_000


def operational_budgets(manifest: dict) -> list[tuple[str, float]]:
    """Read the frozen budget and alert definitions without notebook overrides."""
    policy = manifest["evaluation"]
    expected = {
        "threshold_selection_data": "development_out_of_fold_predictions_only",
        "threshold_selection_procedure": "smallest_score_threshold_whose_worst_fold_mean_false_alert_windows_per_hour_meets_budget",
        "window_alert_aggregation_rule": "maximum_packet_score_in_window_greater_than_or_equal_to_threshold",
        "false_alert_exposure_definition": "all_five_second_wall_clock_windows_from_scenario_origin_through_last_packet_excluding_windows_with_any_attack_packet_including_empty_windows",
        "false_alert_rate_aggregation": "mean_scenario_rates_within_each_fold_then_maximum_fold_rate_for_threshold_selection",
        "threshold_tie_rule": "score_greater_than_or_equal_to_threshold_with_nextafter_above_tied_negative_score",
        "sequence_detection_rule": "any_malicious_packet_score_above_or_equal_to_threshold_with_alert_at_window_end",
        "missed_sequence_latency": "last_malicious_packet_timestamp_minus_first_malicious_packet_timestamp_report_with_miss_flag",
    }
    for name, value in expected.items():
        if policy.get(name) != value:
            raise ValueError(f"The operational OOF policy changed: {name}")
    if manifest["windows"]["selected_duration_seconds"] != WINDOW_SECONDS:
        raise ValueError("Operational OOF evaluation requires five-second windows.")
    declared = policy["false_alert_budgets"]
    budgets = [("one_per_hour", float(declared["primary_per_hour"])),
               ("one_per_12_hours", float(declared["sensitivity_per_hour"][0])),
               ("one_per_5_minutes", float(declared["sensitivity_per_hour"][1]))]
    if not np.allclose([value for _, value in budgets], [1.0, 1 / 12, 12.0], rtol=0, atol=1e-12):
        raise ValueError("The predeclared false-alert budgets changed.")
    return budgets


def _fold_report(run_dir: Path, model_name: str, fold: str) -> dict:
    directory = Path(run_dir) / "depth5_primary" / f"fold_{fold}"
    if model_name == "xgb_p":
        report = validate_xgb_p_fold_run(directory, fold, "depth5_primary")
        if report.get("model_family") != "xgb_p":
            raise ValueError("The baseline run has an unexpected model family.")
        return report
    variant = "full" if model_name == "full" else model_name
    return validate_xgb_p_t_fold_run(directory, fold, variant)


def _validated_runs(manifest: dict, run_dirs: dict[str, Path]) -> dict:
    if set(run_dirs) != set(MODEL_NAMES):
        raise ValueError(f"Exactly these model runs are required: {MODEL_NAMES}")
    reports = {}
    for model_name in MODEL_NAMES:
        reports[model_name] = {}
        for fold, split in manifest["validation"]["folds"].items():
            report = _fold_report(run_dirs[model_name], model_name, fold)
            if (report["training_scenarios"] != split["train"]
                    or report["validation_scenarios"] != split["validate"]
                    or report["window_width_seconds"] != WINDOW_SECONDS
                    or report["decision_time"] != "window_end"):
                raise ValueError(f"Fold or decision-time provenance differs for {model_name}/{fold}.")
            reports[model_name][fold] = report
    for fold in manifest["validation"]["folds"]:
        baseline_hashes = reports["xgb_p"][fold]["prepared_packet_sha256"]
        context_hash = reports["full"][fold]["context_report_sha256"]
        for model_name in MODEL_NAMES[1:]:
            report = reports[model_name][fold]
            if (report["prepared_packet_sha256"] != baseline_hashes
                    or report["context_report_sha256"] != context_hash):
                raise ValueError(f"Packet or context provenance differs for {model_name}/{fold}.")
        for scenario in manifest["validation"]["folds"][fold]["validate"]:
            reference = reports["xgb_p"][fold]["validation"][scenario]
            for model_name in MODEL_NAMES[1:]:
                item = reports[model_name][fold]["validation"][scenario]
                if (item["rows"], item["normal_packets"], item["attack_packets"]) != (
                        reference["rows"], reference["normal_packets"], reference["attack_packets"]):
                    raise ValueError(f"OOF packet counts differ for {model_name}/{scenario}.")
    return reports


def _scenario_oof_path(run_dir: Path, fold: str, report: dict, scenario: str) -> Path:
    return (Path(run_dir) / "depth5_primary" / f"fold_{fold}"
            / report["validation"][scenario]["oof_artifact"])


def summarize_scenario_oof(path: Path, expected: dict) -> dict:
    """Aggregate packet scores into benign windows and attack-step iterations."""
    path = Path(path)
    if pq.ParquetFile(path).metadata.num_rows != expected["rows"]:
        raise ValueError("OOF Parquet row count differs from its fold report.")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 2")
        connection.execute("SET memory_limit = '4GB'")
        window_rows = connection.execute("""
            SELECT window_index, count(*) AS packets,
                   sum(binary_label) AS attack_packets,
                   max(CAST(score AS DOUBLE)) AS max_score,
                   min(window_end_ns) AS first_end_ns,
                   max(window_end_ns) AS last_end_ns
            FROM read_parquet(?)
            GROUP BY window_index ORDER BY window_index
        """, [str(path)]).fetchall()
        iteration_rows = connection.execute("""
            SELECT attack_step, sequence_id, window_index,
                   min(packet_timestamp_ns) AS first_packet_ns,
                   max(packet_timestamp_ns) AS last_packet_ns,
                   min(window_end_ns) AS first_end_ns,
                   max(window_end_ns) AS last_end_ns,
                   max(CAST(score AS DOUBLE)) AS max_malicious_score,
                   count(*) AS attack_packets
            FROM read_parquet(?)
            WHERE binary_label = 1
            GROUP BY attack_step, sequence_id, window_index
            ORDER BY attack_step, sequence_id, window_index
        """, [str(path)]).fetchall()
    finally:
        connection.close()
    if not window_rows or window_rows[0][0] != 0:
        raise ValueError("OOF windows must begin at scenario window zero.")
    if (sum(row[1] for row in window_rows) != expected["rows"]
            or sum(row[2] for row in window_rows) != expected["attack_packets"]):
        raise ValueError("OOF window aggregation did not conserve packet counts.")
    if any(row[4] != row[5] or not 0 <= row[3] <= 1 for row in window_rows):
        raise ValueError("OOF window decision times or scores are invalid.")
    attack_windows = sum(row[2] > 0 for row in window_rows)
    total_windows = int(window_rows[-1][0]) + 1
    benign_windows = total_windows - attack_windows
    exposure_hours = benign_windows * WINDOW_SECONDS / 3600
    if exposure_hours <= 0:
        raise ValueError("A scenario has no benign wall-clock exposure.")
    negative_scores = np.sort(np.array(
        [row[3] for row in window_rows if row[2] == 0], dtype=np.float64))
    iterations = {}
    for step, sequence, _, first, last, first_end, last_end, score, packets in iteration_rows:
        if step is None or sequence is None or first_end != last_end or not 0 <= score <= 1:
            raise ValueError("Attack-step iteration metadata or scores are invalid.")
        key = (str(step), str(sequence))
        item = iterations.setdefault(key, {
            "attack_step": str(step), "sequence_id": str(sequence),
            "first_packet_ns": first, "last_packet_ns": last,
            "window_scores": [], "attack_packets": 0,
        })
        item["first_packet_ns"] = min(item["first_packet_ns"], first)
        item["last_packet_ns"] = max(item["last_packet_ns"], last)
        item["window_scores"].append((int(first_end), float(score)))
        item["attack_packets"] += int(packets)
    if (sum(item["attack_packets"] for item in iterations.values())
            != expected["attack_packets"] or not iterations):
        raise ValueError("Attack-step iteration aggregation did not conserve packets.")
    return {
        "negative_scores": negative_scores,
        "exposure_hours": exposure_hours,
        "total_wall_clock_windows": total_windows,
        "attack_windows": attack_windows,
        "benign_windows": benign_windows,
        "iterations": list(iterations.values()),
    }


def false_alert_rate(summary: dict, threshold: float) -> tuple[int, float]:
    scores = summary["negative_scores"]
    alerts = len(scores) - int(np.searchsorted(scores, threshold, side="left"))
    return alerts, alerts / summary["exposure_hours"]


def select_threshold(summaries: dict[str, dict], folds: dict[str, dict],
                     budget_per_hour: float) -> dict:
    """Select the most permissive threshold satisfying both fold means."""
    if not np.isfinite(budget_per_hour) or budget_per_hour <= 0:
        raise ValueError("False-alert budget must be positive and finite.")
    negative_scores = np.concatenate([
        summary["negative_scores"] for summary in summaries.values()])
    candidates = np.concatenate((
        np.array([0.0]),
        np.nextafter(np.unique(negative_scores), np.inf),
    ))

    def fold_rates(threshold: float) -> dict[str, float]:
        return {
            fold: float(np.mean([
                false_alert_rate(summaries[scenario], threshold)[1]
                for scenario in split["validate"]]))
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
        raise AssertionError("No threshold satisfies the declared false-alert budget.")
    return {"threshold": threshold, "fold_false_alert_windows_per_hour": rates,
            "worst_fold_false_alert_windows_per_hour": max(rates.values())}


def _packet_counts_by_threshold(path: Path, thresholds: dict[str, float],
                                expected: dict, batch_size: int) -> dict:
    counts = {name: {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
              for name in thresholds}
    rows = 0
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size,
                                      columns=["binary_label", "score"]):
        labels = batch.column(0).to_numpy(zero_copy_only=False)
        scores = batch.column(1).to_numpy(zero_copy_only=False)
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
    if rows != expected["rows"]:
        raise ValueError("OOF packet count changed during threshold evaluation.")
    return counts


def evaluate_scenario(summary: dict, packet_counts: dict,
                      threshold: float, scenario: str, fold: str) -> tuple[dict, list[dict]]:
    """Compute operational metrics while retaining every missed iteration."""
    alerts, alert_rate = false_alert_rate(summary, threshold)
    iterations = []
    for item in summary["iterations"]:
        eligible = [end for end, score in item["window_scores"] if score >= threshold]
        first_packet = int(item["first_packet_ns"])
        last_packet = int(item["last_packet_ns"])
        first_alert = min(eligible) if eligible else None
        if first_alert is not None and first_alert < first_packet:
            raise ValueError("A detection precedes its attack-step iteration.")
        iterations.append({
            "scenario": scenario, "fold": fold,
            "attack_step": item["attack_step"],
            "sequence_id": item["sequence_id"],
            "attack_packets": item["attack_packets"],
            "first_malicious_packet_ns": first_packet,
            "last_malicious_packet_ns": last_packet,
            "first_detecting_window_end_ns": first_alert,
            "detected": first_alert is not None,
            "latency_seconds": (
                (first_alert - first_packet) / NANOSECONDS_PER_SECOND
                if first_alert is not None else None),
            "miss_capped_latency_seconds": (
                (first_alert - first_packet) / NANOSECONDS_PER_SECOND
                if first_alert is not None else
                (last_packet - first_packet) / NANOSECONDS_PER_SECOND),
        })
    detected = [item for item in iterations if item["detected"]]
    tp, fp = packet_counts["tp"], packet_counts["fp"]
    tn, fn = packet_counts["tn"], packet_counts["fn"]
    metrics = {
        "scenario": scenario, "fold": fold,
        "threshold": threshold,
        "benign_exposure_hours": summary["exposure_hours"],
        "benign_wall_clock_windows": summary["benign_windows"],
        "attack_windows": summary["attack_windows"],
        "false_alert_windows": alerts,
        "false_alert_windows_per_hour": alert_rate,
        "packet_true_positives": tp,
        "packet_false_positives": fp,
        "packet_true_negatives": tn,
        "packet_false_negatives": fn,
        "packet_recall": tp / (tp + fn),
        "packet_precision": tp / (tp + fp) if tp + fp else None,
        "packet_false_positive_rate": fp / (fp + tn),
        "attack_step_iterations": len(iterations),
        "detected_iterations": len(detected),
        "sequence_detection_rate": len(detected) / len(iterations),
        "mean_miss_capped_latency_seconds": float(np.mean([
            item["miss_capped_latency_seconds"] for item in iterations])),
        "mean_detected_latency_seconds_diagnostic": (
            float(np.mean([item["latency_seconds"] for item in detected]))
            if detected else None),
    }
    return metrics, iterations


def _macro_metrics(scenarios: dict[str, dict], folds: dict[str, dict]) -> dict:
    fields = ("false_alert_windows_per_hour", "packet_recall",
              "packet_false_positive_rate", "sequence_detection_rate",
              "mean_miss_capped_latency_seconds")
    fold_means = {
        fold: {name: float(np.mean([
            scenarios[scenario][name] for scenario in split["validate"]]))
               for name in fields}
        for fold, split in folds.items()
    }
    return {
        "fold_means": fold_means,
        "hierarchical_macro": {name: float(np.mean([
            fold_means[fold][name] for fold in folds])) for name in fields},
    }


def run_operational_oof_evaluation(*, manifest_path: Path,
                                   run_dirs: dict[str, Path], output_dir: Path,
                                   batch_size: int = 250_000) -> dict:
    """Evaluate four immutable OOF runs under one predeclared alert policy."""
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    if batch_size <= 0 or output_dir.exists():
        raise ValueError("Batch size must be positive and the output run must be new.")
    manifest = load_manifest(manifest_path)
    budgets = operational_budgets(manifest)
    reports = _validated_runs(manifest, run_dirs)
    folds = manifest["validation"]["folds"]
    repository_root = manifest_path.resolve().parent.parent
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repository_root,
                              capture_output=True, text=True, check=False)
    worktree = subprocess.run(["git", "status", "--short"], cwd=repository_root,
                              capture_output=True, text=True, check=False)
    result = {
        "report_version": REPORT_VERSION,
        "status": "development_oof_operational_evaluation_complete",
        "manifest_sha256": sha256_file(manifest_path),
        "evaluator_code_sha256": sha256_file(Path(__file__)),
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
        "git_worktree_status": worktree.stdout if worktree.returncode == 0 else "unavailable",
        "budget_order": [name for name, _ in budgets],
        "budgets_per_hour": {name: rate for name, rate in budgets},
        "input_runs": {name: str(path) for name, path in run_dirs.items()},
        "input_oof_sha256": {
            model_name: {
                scenario: item["oof_sha256"]
                for fold_report in reports[model_name].values()
                for scenario, item in fold_report["validation"].items()}
            for model_name in MODEL_NAMES},
        "models": {},
        "thresholds_selected_from": "development_oof_only",
        "test_data_accessed": False,
    }
    for model_name in MODEL_NAMES:
        print(f"Aggregating {model_name} OOF windows and iterations...", flush=True)
        summaries = {}
        paths = {}
        for fold, split in folds.items():
            report = reports[model_name][fold]
            for scenario in split["validate"]:
                item = report["validation"][scenario]
                path = _scenario_oof_path(run_dirs[model_name], fold, report, scenario)
                summaries[scenario] = summarize_scenario_oof(path, item)
                paths[scenario] = path
        thresholds = {
            name: {"target_false_alert_windows_per_hour": budget,
                   **select_threshold(summaries, folds, budget)}
            for name, budget in budgets}
        packet_counts = {
            scenario: _packet_counts_by_threshold(
                paths[scenario],
                {name: item["threshold"] for name, item in thresholds.items()},
                reports[model_name][fold]["validation"][scenario], batch_size)
            for fold, split in folds.items()
            for scenario in split["validate"]
        }
        per_budget = {}
        for budget_name, _ in budgets:
            threshold = thresholds[budget_name]["threshold"]
            scenario_metrics = {}
            iteration_rows = []
            step_groups = defaultdict(list)
            for fold, split in folds.items():
                for scenario in split["validate"]:
                    metrics, iterations = evaluate_scenario(
                        summaries[scenario], packet_counts[scenario][budget_name],
                        threshold, scenario, fold)
                    scenario_metrics[scenario] = metrics
                    iteration_rows.extend(iterations)
                    for item in iterations:
                        step_groups[(scenario, item["attack_step"])].append(item)
            step_metrics = {
                f"{scenario}::{step}": {
                    "scenario": scenario, "attack_step": step,
                    "iterations": len(items),
                    "detected_iterations": sum(item["detected"] for item in items),
                    "sequence_detection_rate": (
                        sum(item["detected"] for item in items) / len(items)),
                    "mean_miss_capped_latency_seconds": float(np.mean([
                        item["miss_capped_latency_seconds"] for item in items])),
                }
                for (scenario, step), items in step_groups.items()
            }
            per_budget[budget_name] = {
                "scenario_metrics": scenario_metrics,
                "step_metrics": step_metrics,
                "iteration_rows": iteration_rows,
                **_macro_metrics(scenario_metrics, folds),
            }
        result["models"][model_name] = {
            "thresholds": thresholds, "budgets": per_budget,
        }
    output_dir.mkdir(parents=True)
    write_json(output_dir / "operational_report.json", result)
    write_json(output_dir / "run_status.json", {
        "complete": True,
        "report_sha256": sha256_file(output_dir / "operational_report.json")})
    return result


def validate_operational_run(output_dir: Path, manifest_path: Path,
                             run_dirs: dict[str, Path]) -> dict:
    """Verify a completed report belongs to the current protocol and runs."""
    output_dir = Path(output_dir)
    report_path = output_dir / "operational_report.json"
    status_path = output_dir / "run_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("complete") is not True or status.get("report_sha256") != sha256_file(report_path):
        raise ValueError("The operational OOF report is incomplete or changed.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("manifest_sha256") != sha256_file(manifest_path)
            or report.get("input_runs") != {name: str(path) for name, path in run_dirs.items()}
            or report.get("evaluator_code_sha256") != sha256_file(Path(__file__))):
        raise ValueError("The operational report belongs to a different protocol or input run.")
    reports = _validated_runs(load_manifest(manifest_path), run_dirs)
    observed_hashes = {
        model_name: {
            scenario: item["oof_sha256"]
            for fold_report in reports[model_name].values()
            for scenario, item in fold_report["validation"].items()}
        for model_name in MODEL_NAMES}
    if report.get("input_oof_sha256") != observed_hashes:
        raise ValueError("An input OOF artifact changed after evaluation.")
    return report
