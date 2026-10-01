"""Read-only score-tail and saturation audit for cAPTure graph OOF outputs."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .capture_data import sha256_file, write_json
from .capture_graph_comparison import DISPLAY_NAMES, MODEL_NAMES, PRIMARY_BUDGET


REPORT_VERSION = 1
EXPECTED_COMPARISON_STATUS = "exploratory_four_model_development_comparison_complete"
QUANTILE_NAMES = (
    "score_q50",
    "score_q90",
    "score_q99",
    "score_q999",
    "score_q9999",
    "score_q99999",
)


def _load_json(path: Path, label: str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected {label} to contain a JSON object: {path}")
    return value


def _unpack_quantiles(values) -> dict:
    if values is None or len(values) != len(QUANTILE_NAMES):
        raise ValueError("The score-tail query returned invalid quantiles.")
    return {
        name: float(value) for name, value in zip(QUANTILE_NAMES, values)
    }


def _load_comparison(comparison_dir: Path) -> dict:
    comparison_dir = Path(comparison_dir)
    report_path = comparison_dir / "comparison_report.json"
    status = _load_json(comparison_dir / "run_status.json", "comparison status")
    report = _load_json(report_path, "comparison report")
    if (
        status.get("complete") is not True
        or status.get("report_sha256") != sha256_file(report_path)
        or report.get("report_version") != 1
        or report.get("status") != EXPECTED_COMPARISON_STATUS
        or report.get("run_id") != comparison_dir.name
        or report.get("models_compared") != list(MODEL_NAMES)
        or report.get("held_out_scenarios_accessed") is not False
        or report.get("model_training_performed") is not False
    ):
        raise ValueError("The source four-model comparison has unexpected provenance.")
    for name, expected in status.get("artifact_sha256", {}).items():
        if sha256_file(comparison_dir / name) != expected:
            raise ValueError(f"Source comparison artifact changed: {name}")
    return report


def _validated_oof_paths(
    *, comparison: dict, training_root: Path
) -> tuple[dict[str, dict[str, Path]], dict[str, str]]:
    training_root = Path(training_root)
    paths: dict[str, dict[str, Path]] = {model: {} for model in MODEL_NAMES}
    score_types = {}
    for model_name in MODEL_NAMES:
        for fold in ("A", "B"):
            job_id = f"{model_name}__fold_{fold}"
            job_dir = training_root / job_id
            completion_path = job_dir / "completion.json"
            completion = _load_json(completion_path, f"{job_id} completion")
            metrics_path = job_dir / "metrics.json"
            metrics = _load_json(metrics_path, f"{job_id} metrics")
            if (
                completion.get("status") != "complete"
                or completion.get("model") != model_name
                or completion.get("fold") != fold
                or completion.get("held_out_scenarios_accessed") is not False
                or sha256_file(completion_path)
                != comparison["source_completion_sha256"].get(job_id)
                or sha256_file(metrics_path)
                != completion["artifact_checksums"].get("metrics.json")
            ):
                raise ValueError(f"The source training job changed: {job_id}")
            for scenario, item in metrics["outer_oof"]["scenarios"].items():
                path = job_dir / item["predictions"]
                expected_hash = comparison["input_oof_sha256"][model_name].get(
                    scenario
                )
                observed_hash = sha256_file(path)
                if (
                    observed_hash != expected_hash
                    or observed_hash != item["predictions_sha256"]
                    or observed_hash
                    != completion["artifact_checksums"].get(item["predictions"])
                ):
                    raise ValueError(
                        f"The source OOF predictions changed: {model_name}/{scenario}"
                    )
                parquet = pq.ParquetFile(path)
                score_type = str(parquet.schema_arrow.field("score").type)
                if model_name in score_types and score_types[model_name] != score_type:
                    raise ValueError(f"Score storage type changed within {model_name}.")
                score_types[model_name] = score_type
                paths[model_name][scenario] = path
        if set(paths[model_name]) != set(comparison["aligned_oof_packet_counts"]):
            raise ValueError(f"OOF scenario coverage changed for {model_name}.")
    return paths, score_types


def _packet_tail_rows(
    connection: duckdb.DuckDBPyConnection,
    *,
    path: Path,
    model_name: str,
    fold: str,
    scenario: str,
) -> list[dict]:
    rows = connection.execute(
        """
        WITH packets AS (
            SELECT binary_label, CAST(score AS DOUBLE) AS score
            FROM read_parquet(?)
        ),
        annotated AS (
            SELECT *, max(score) OVER (PARTITION BY binary_label) AS class_maximum
            FROM packets
        )
        SELECT
            binary_label,
            count(*) AS packets,
            count(DISTINCT score) AS distinct_scores,
            min(score) AS minimum_score,
            approx_quantile(
                score, [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999]
            ) AS score_quantiles,
            max(score) AS maximum_score,
            count_if(score = 0.0) AS exact_zero_scores,
            count_if(score = 1.0) AS exact_one_scores,
            count_if(score = class_maximum) AS maximum_tie_scores
        FROM annotated
        GROUP BY binary_label
        ORDER BY binary_label
        """,
        [str(path)],
    ).fetchall()
    result = []
    for (
        binary_label,
        packets,
        distinct_scores,
        minimum_score,
        quantiles,
        maximum_score,
        exact_zero_scores,
        exact_one_scores,
        maximum_tie_scores,
    ) in rows:
        result.append({
            "model": model_name,
            "model_display": DISPLAY_NAMES[model_name],
            "fold": fold,
            "scenario": scenario,
            "packet_class": "attack" if int(binary_label) == 1 else "benign",
            "packets": int(packets),
            "distinct_scores": int(distinct_scores),
            "distinct_score_fraction": int(distinct_scores) / int(packets),
            "minimum_score": float(minimum_score),
            **_unpack_quantiles(quantiles),
            "maximum_score": float(maximum_score),
            "exact_zero_scores": int(exact_zero_scores),
            "exact_zero_score_fraction": int(exact_zero_scores) / int(packets),
            "exact_one_scores": int(exact_one_scores),
            "exact_one_score_fraction": int(exact_one_scores) / int(packets),
            "maximum_tie_scores": int(maximum_tie_scores),
            "maximum_tie_fraction": int(maximum_tie_scores) / int(packets),
        })
    if {row["packet_class"] for row in result} != {"benign", "attack"}:
        raise ValueError(f"Both packet classes are required for {model_name}/{scenario}.")
    return result


def _window_tail_rows(
    connection: duckdb.DuckDBPyConnection,
    *,
    path: Path,
    model_name: str,
    fold: str,
    scenario: str,
    scenario_metrics: dict,
) -> list[dict]:
    rows = connection.execute(
        """
        WITH window_scores AS (
            SELECT
                window_index,
                max(binary_label) AS contains_attack,
                max(CAST(score AS DOUBLE)) AS maximum_score
            FROM read_parquet(?)
            GROUP BY window_index
        ),
        annotated AS (
            SELECT *, max(maximum_score) OVER (
                PARTITION BY contains_attack
            ) AS class_maximum
            FROM window_scores
        )
        SELECT
            contains_attack,
            count(*) AS observed_windows,
            count(DISTINCT maximum_score) AS distinct_scores,
            min(maximum_score) AS minimum_score,
            approx_quantile(
                maximum_score, [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999]
            ) AS score_quantiles,
            max(maximum_score) AS maximum_score,
            count_if(maximum_score = 0.0) AS exact_zero_scores,
            count_if(maximum_score = 1.0) AS exact_one_scores,
            count_if(maximum_score = class_maximum) AS maximum_tie_scores
        FROM annotated
        GROUP BY contains_attack
        ORDER BY contains_attack
        """,
        [str(path)],
    ).fetchall()
    result = []
    for (
        contains_attack,
        observed_windows,
        distinct_scores,
        minimum_score,
        quantiles,
        maximum_score,
        exact_zero_scores,
        exact_one_scores,
        maximum_tie_scores,
    ) in rows:
        window_class = "attack_containing" if int(contains_attack) == 1 else "benign"
        total_wall_clock_windows = (
            int(scenario_metrics["attack_windows"])
            if window_class == "attack_containing"
            else int(scenario_metrics["benign_wall_clock_windows"])
        )
        empty_windows = total_wall_clock_windows - int(observed_windows)
        if empty_windows < 0:
            raise ValueError(f"Observed windows exceed exposure for {model_name}/{scenario}.")
        result.append({
            "model": model_name,
            "model_display": DISPLAY_NAMES[model_name],
            "fold": fold,
            "scenario": scenario,
            "window_class": window_class,
            "wall_clock_windows": total_wall_clock_windows,
            "observed_windows": int(observed_windows),
            "empty_windows": empty_windows,
            "distinct_window_maximum_scores": int(distinct_scores),
            "minimum_window_maximum_score": float(minimum_score),
            **{
                name.replace("score_", "window_maximum_score_"): value
                for name, value in _unpack_quantiles(quantiles).items()
            },
            "maximum_window_maximum_score": float(maximum_score),
            "exact_zero_window_maxima": int(exact_zero_scores),
            "exact_one_window_maxima": int(exact_one_scores),
            "exact_one_observed_window_fraction": (
                int(exact_one_scores) / int(observed_windows)
            ),
            "maximum_tie_windows": int(maximum_tie_scores),
            "maximum_tie_observed_window_fraction": (
                int(maximum_tie_scores) / int(observed_windows)
            ),
        })
    if {row["window_class"] for row in result} != {"benign", "attack_containing"}:
        raise ValueError(f"Both window classes are required for {model_name}/{scenario}.")
    return result


def _budget_diagnostics(
    *, comparison: dict, packet_rows: list[dict], window_rows: list[dict]
) -> list[dict]:
    folds = {
        row["scenario"]: row["fold"]
        for row in window_rows
        if row["window_class"] == "benign"
    }
    result = []
    for model_name in MODEL_NAMES:
        model_packet_rows = [row for row in packet_rows if row["model"] == model_name]
        benign_windows = [
            row for row in window_rows
            if row["model"] == model_name and row["window_class"] == "benign"
        ]
        attack_windows = [
            row for row in window_rows
            if row["model"] == model_name
            and row["window_class"] == "attack_containing"
        ]
        attack_packets = [
            row for row in model_packet_rows if row["packet_class"] == "attack"
        ]
        total_attack_packets = sum(row["packets"] for row in attack_packets)
        exact_one_attack_packets = sum(row["exact_one_scores"] for row in attack_packets)
        exact_one_benign_windows = sum(
            row["exact_one_window_maxima"] for row in benign_windows
        )
        exact_one_attack_windows = sum(
            row["exact_one_window_maxima"] for row in attack_windows
        )
        maximum_benign_window_score = max(
            row["maximum_window_maximum_score"] for row in benign_windows
        )
        scenario_exact_one_rates = {
            row["scenario"]: (
                row["exact_one_window_maxima"]
                / (row["wall_clock_windows"] * 5 / 3600)
            )
            for row in benign_windows
        }
        fold_exact_one_rates = {
            fold: float(np.mean([
                rate for scenario, rate in scenario_exact_one_rates.items()
                if folds[scenario] == fold
            ]))
            for fold in sorted(set(folds.values()))
        }
        scenario_top_score_block_rates = {
            row["scenario"]: (
                row["maximum_tie_windows"]
                / (row["wall_clock_windows"] * 5 / 3600)
                if row["maximum_window_maximum_score"]
                == maximum_benign_window_score
                else 0.0
            )
            for row in benign_windows
        }
        fold_top_score_block_rates = {
            fold: float(np.mean([
                rate for scenario, rate in scenario_top_score_block_rates.items()
                if folds[scenario] == fold
            ]))
            for fold in sorted(set(folds.values()))
        }
        minimum_positive_worst_fold_rate = max(
            fold_top_score_block_rates.values()
        )
        exact_one_worst_fold_rate = max(fold_exact_one_rates.values())
        maximum_attack_packet_score = max(
            row["maximum_score"] for row in attack_packets
        )
        for budget_name in comparison["budget_order"]:
            selected = comparison["models"][model_name]["thresholds"][budget_name]
            threshold = float(selected["threshold"])
            target = float(selected["target_false_alert_windows_per_hour"])
            threshold_above_one = threshold > 1.0
            shared_exact_one_tie = (
                exact_one_benign_windows > 0 and exact_one_attack_packets > 0
            )
            top_tie_exceeds_budget = minimum_positive_worst_fold_rate > target
            if threshold_above_one and shared_exact_one_tie and top_tie_exceeds_budget:
                diagnosis = "saturated_top_tie_forces_zero_alert_threshold"
            elif threshold_above_one:
                diagnosis = "threshold_above_one_requires_review"
            elif shared_exact_one_tie:
                diagnosis = "shared_exact_one_tie_within_selected_budget"
            else:
                diagnosis = "no_exact_one_operational_blocker_detected"
            result.append({
                "model": model_name,
                "model_display": DISPLAY_NAMES[model_name],
                "budget": budget_name,
                "target_false_alert_windows_per_hour": target,
                "selected_threshold": threshold,
                "selected_threshold_above_one": threshold_above_one,
                "reported_worst_fold_false_alert_windows_per_hour": float(
                    selected["worst_fold_false_alert_windows_per_hour"]
                ),
                "maximum_benign_window_score": maximum_benign_window_score,
                "maximum_attack_packet_score": maximum_attack_packet_score,
                "exact_one_benign_windows": exact_one_benign_windows,
                "exact_one_attack_windows": exact_one_attack_windows,
                "exact_one_attack_packets": exact_one_attack_packets,
                "exact_one_attack_packet_fraction": (
                    exact_one_attack_packets / total_attack_packets
                ),
                "exact_one_block_worst_fold_false_alert_windows_per_hour": (
                    exact_one_worst_fold_rate
                ),
                "minimum_positive_worst_fold_false_alert_windows_per_hour": (
                    minimum_positive_worst_fold_rate
                ),
                "top_score_block_within_budget": (
                    minimum_positive_worst_fold_rate <= target
                ),
                "shared_exact_one_benign_attack_tie": shared_exact_one_tie,
                "diagnosis": diagnosis,
            })
    return result


def _model_summary(budget_rows: list[dict], score_types: dict[str, str]) -> list[dict]:
    result = []
    for model_name in MODEL_NAMES:
        primary = next(
            row for row in budget_rows
            if row["model"] == model_name and row["budget"] == PRIMARY_BUDGET
        )
        any_degenerate = any(
            row["diagnosis"] == "saturated_top_tie_forces_zero_alert_threshold"
            for row in budget_rows
            if row["model"] == model_name
        )
        result.append({
            "model": model_name,
            "model_display": DISPLAY_NAMES[model_name],
            "stored_score_type": score_types[model_name],
            "primary_threshold": primary["selected_threshold"],
            "primary_threshold_above_one": primary["selected_threshold_above_one"],
            "primary_worst_fold_false_alert_windows_per_hour": primary[
                "reported_worst_fold_false_alert_windows_per_hour"
            ],
            "minimum_positive_worst_fold_false_alert_windows_per_hour": primary[
                "minimum_positive_worst_fold_false_alert_windows_per_hour"
            ],
            "exact_one_benign_windows": primary["exact_one_benign_windows"],
            "exact_one_attack_packets": primary["exact_one_attack_packets"],
            "exact_one_attack_packet_fraction": primary[
                "exact_one_attack_packet_fraction"
            ],
            "primary_diagnosis": primary["diagnosis"],
            "any_budget_has_degenerate_saturation": any_degenerate,
            "oof_rescoring_from_checkpoint_recommended": any_degenerate,
        })
    return result


def _markdown_table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    lines = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            if isinstance(value, float):
                values.append(f"{value:.6g}")
            elif value is None:
                values.append("NA")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_review(output_dir: Path, report: dict) -> None:
    summaries = report["tables"]["model_summary"]
    saturated = [
        row["model_display"] for row in summaries
        if row["oof_rescoring_from_checkpoint_recommended"]
    ]
    if saturated:
        conclusion = (
            "Score saturation is confirmed for " + ", ".join(saturated) + 
            ". Their thresholded operational metrics must not be interpreted as "
            "model failures. Re-run OOF inference from the immutable final checkpoints "
            "and preserve raw logits or float64 sigmoid scores."
        )
    else:
        conclusion = (
            "No model meets the frozen signature for a shared exact-one score tie that "
            "forces a threshold above one. Investigate the threshold implementation "
            "before rescoring."
        )
    text = "\n".join([
        "# cAPTure graph score-saturation audit",
        "",
        f"Status: **{report['status']}**.",
        "",
        conclusion,
        "",
        "## Primary-budget diagnosis",
        "",
        _markdown_table(summaries, [
            ("model_display", "Model"),
            ("stored_score_type", "Stored score"),
            ("primary_threshold", "Threshold"),
            ("primary_threshold_above_one", "Threshold > 1"),
            ("exact_one_benign_windows", "Benign windows at 1"),
            ("exact_one_attack_packet_fraction", "Attack packets at 1"),
            ("minimum_positive_worst_fold_false_alert_windows_per_hour", "Minimum positive worst-fold FA/h"),
            ("primary_diagnosis", "Diagnosis"),
        ]),
        "",
        "## Interpretation",
        "",
        "A threshold above one is a valid consequence of the frozen `score >= threshold` "
        "and `nextafter` tie policy when the benign top-score block is too large for the "
        "budget. It is not evidence that the threshold search malfunctioned. When attack "
        "scores share the same stored float32 value of one, probability artifacts no "
        "longer contain enough resolution to separate that tie.",
        "",
        "Ranking metrics remain valid for the stored score ordering, but operational "
        "metrics from a degenerate threshold are not suitable for comparing detection "
        "utility. Existing OOF files and the four-model comparison remain immutable "
        "evidence; any corrected evaluation must use new versioned artifacts.",
        "",
        "## Next action",
        "",
        "If rescoring is recommended, load each saved final checkpoint and repeat only "
        "outer-fold inference. Store raw logits and a float64 sigmoid derived before "
        "serialization, verify identical packet keys, and run a versioned operational "
        "comparison. Do not retrain, refit, or access held-out scenarios.",
        "",
        "Test1, Test2, `train_pub_exf`, and `train_user_prop` were not accessed.",
        "",
    ])
    (output_dir / "review.md").write_text(text, encoding="utf-8")


def run_graph_score_saturation_audit(
    *,
    comparison_dir: str | Path,
    training_root: str | Path,
    output_dir: str | Path,
) -> dict:
    """Audit immutable graph OOF score tails without training or threshold changes."""
    comparison_dir = Path(comparison_dir).expanduser().resolve()
    training_root = Path(training_root).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("The saturation-audit output directory must be new.")
    comparison = _load_comparison(comparison_dir)
    paths, score_types = _validated_oof_paths(
        comparison=comparison, training_root=training_root
    )
    scenario_folds = {
        row["scenario"]: row["fold"]
        for row in comparison["tables"]["ranking_by_scenario"]
        if row["model"] == MODEL_NAMES[0]
    }
    if set(scenario_folds) != set(comparison["aligned_oof_packet_counts"]):
        raise ValueError("Scenario-to-fold assignments are incomplete.")

    packet_rows = []
    window_rows = []
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 2")
        connection.execute("SET memory_limit = '4GB'")
        for model_name in MODEL_NAMES:
            print(f"Auditing score tails for {model_name}...", flush=True)
            for scenario, path in paths[model_name].items():
                fold = scenario_folds[scenario]
                primary_metrics = comparison["models"][model_name]["budgets"][
                    PRIMARY_BUDGET
                ]["scenario_metrics"][scenario]
                packet_rows.extend(_packet_tail_rows(
                    connection,
                    path=path,
                    model_name=model_name,
                    fold=fold,
                    scenario=scenario,
                ))
                window_rows.extend(_window_tail_rows(
                    connection,
                    path=path,
                    model_name=model_name,
                    fold=fold,
                    scenario=scenario,
                    scenario_metrics=primary_metrics,
                ))
    finally:
        connection.close()

    budget_rows = _budget_diagnostics(
        comparison=comparison,
        packet_rows=packet_rows,
        window_rows=window_rows,
    )
    summaries = _model_summary(budget_rows, score_types)
    saturated_models = [
        row["model"] for row in summaries
        if row["oof_rescoring_from_checkpoint_recommended"]
    ]
    status = (
        "score_saturation_confirmed"
        if saturated_models
        else "no_degenerate_score_saturation_detected"
    )
    repository_root = Path(__file__).resolve().parents[3]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root,
        capture_output=True, text=True, check=False,
    )
    worktree = subprocess.run(
        ["git", "status", "--short"], cwd=repository_root,
        capture_output=True, text=True, check=False,
    )
    report = {
        "report_version": REPORT_VERSION,
        "status": status,
        "run_id": output_dir.name,
        "source_comparison_run_id": comparison_dir.name,
        "source_comparison_report_sha256": sha256_file(
            comparison_dir / "comparison_report.json"
        ),
        "source_input_oof_sha256": comparison["input_oof_sha256"],
        "audit_code_sha256": sha256_file(Path(__file__)),
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
        "git_worktree_status": worktree.stdout if worktree.returncode == 0 else "unavailable",
        "score_storage_types": score_types,
        "threshold_policy_changed": False,
        "model_training_performed": False,
        "held_out_scenarios_accessed": False,
        "saturated_models": saturated_models,
        "oof_rescoring_recommended": bool(saturated_models),
        "rescoring_contract": {
            "reuse_immutable_final_checkpoints": True,
            "repeat_outer_oof_inference_only": True,
            "preserve_raw_logits": True,
            "preserve_float64_sigmoid_scores": True,
            "require_identical_packet_keys": True,
            "retraining_forbidden": True,
            "threshold_refitting_outside_development_oof_forbidden": True,
            "held_out_scenario_access_forbidden": True,
        },
        "tables": {
            "model_summary": summaries,
            "budget_diagnostics": budget_rows,
            "packet_score_tails": packet_rows,
            "window_maximum_score_tails": window_rows,
        },
    }
    output_dir.mkdir(parents=True)
    write_json(output_dir / "score_saturation_audit.json", report)
    for name, rows in report["tables"].items():
        pd.DataFrame(rows).to_csv(output_dir / f"{name}.csv", index=False)
    _write_review(output_dir, report)
    artifact_hashes = {
        path.name: sha256_file(path)
        for path in sorted(output_dir.iterdir())
        if path.name != "run_status.json"
    }
    write_json(output_dir / "run_status.json", {
        "complete": True,
        "report_sha256": sha256_file(output_dir / "score_saturation_audit.json"),
        "artifact_sha256": artifact_hashes,
        "model_training_performed": False,
        "held_out_scenarios_accessed": False,
    })
    return report


def validate_graph_score_saturation_audit(
    *, comparison_dir: str | Path, output_dir: str | Path
) -> dict:
    """Validate a completed saturation audit against its immutable comparison."""
    comparison_dir = Path(comparison_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    report_path = output_dir / "score_saturation_audit.json"
    status = _load_json(output_dir / "run_status.json", "saturation-audit status")
    report = _load_json(report_path, "saturation-audit report")
    if (
        status.get("complete") is not True
        or status.get("report_sha256") != sha256_file(report_path)
        or report.get("report_version") != REPORT_VERSION
        or report.get("run_id") != output_dir.name
        or report.get("source_comparison_run_id") != comparison_dir.name
        or report.get("source_comparison_report_sha256")
        != sha256_file(comparison_dir / "comparison_report.json")
        or report.get("audit_code_sha256") != sha256_file(Path(__file__))
        or report.get("threshold_policy_changed") is not False
        or report.get("model_training_performed") is not False
        or report.get("held_out_scenarios_accessed") is not False
    ):
        raise ValueError("The score-saturation audit has unexpected provenance.")
    for name, expected in status["artifact_sha256"].items():
        if sha256_file(output_dir / name) != expected:
            raise ValueError(f"Saturation-audit artifact changed: {name}")
    comparison = _load_comparison(comparison_dir)
    if report.get("source_input_oof_sha256") != comparison["input_oof_sha256"]:
        raise ValueError("The saturation audit references different OOF predictions.")
    return report
