"""Audit step and chain warning times from a completed development OOF report."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

import yaml

from .capture_data import load_manifest, sha256_file, write_json
from . import capture_oof_operational


REPORT_VERSION = 1
NANOSECONDS_PER_SECOND = 1_000_000_000
POLICY_RULES = {
    "audit_version": 1,
    "scope": "development_oof_only",
    "step_timely_rule": "first_correct_window_alert_strictly_before_last_malicious_packet",
    "chain_onset_rule": "first_malicious_packet_in_scenario",
    "terminal_onset_rule": "first_malicious_packet_of_any_declared_terminal_action_step",
    "chain_early_rule": "first_correct_window_alert_strictly_before_terminal_onset",
    "first_correct_alert_rule": "earliest_window_end_with_a_score_positive_malicious_packet",
}


def _load_policy(path: Path, manifest: dict) -> dict:
    policy = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(policy, dict) or set(policy) != {*POLICY_RULES, "terminal_action_steps"}:
        raise ValueError("The early-warning audit policy has unexpected fields.")
    for name, expected in POLICY_RULES.items():
        if policy[name] != expected:
            raise ValueError(f"The early-warning audit policy changed: {name}")
    scenarios = {
        scenario for split in manifest["validation"]["folds"].values()
        for scenario in split["validate"]
    }
    terminal_steps = policy["terminal_action_steps"]
    if not isinstance(terminal_steps, dict) or set(terminal_steps) != scenarios:
        raise ValueError("Terminal actions must cover exactly the development OOF scenarios.")
    for scenario, steps in terminal_steps.items():
        if (not isinstance(steps, list) or not steps
                or any(not isinstance(step, str) or not step for step in steps)
                or len(set(steps)) != len(steps)):
            raise ValueError(f"Invalid terminal action steps for {scenario}.")
    return policy


def _load_operational_report(operational_dir: Path, manifest_path: Path) -> dict:
    """Verify the immutable report without reopening its large OOF Parquet files."""
    operational_dir = Path(operational_dir)
    report_path = operational_dir / "operational_report.json"
    status_path = operational_dir / "run_status.json"
    if not report_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("A completed operational OOF report is required.")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("complete") is not True or status.get("report_sha256") != sha256_file(report_path):
        raise ValueError("The operational OOF report is incomplete or changed.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (report.get("report_version") != capture_oof_operational.REPORT_VERSION
            or report.get("status") != "development_oof_operational_evaluation_complete"
            or report.get("manifest_sha256") != sha256_file(manifest_path)
            or report.get("evaluator_code_sha256") != sha256_file(
                Path(capture_oof_operational.__file__))
            or report.get("thresholds_selected_from") != "development_oof_only"
            or report.get("test_data_accessed") is not False
            or set(report.get("models", {})) != set(capture_oof_operational.MODEL_NAMES)
            or set(report.get("input_oof_sha256", {}))
            != set(capture_oof_operational.MODEL_NAMES)):
        raise ValueError("The operational OOF report has unexpected provenance.")
    return report


def _classified_iteration(item: dict, *, scenario: str, fold: str) -> dict:
    if item.get("scenario") != scenario or item.get("fold") != fold:
        raise ValueError("An iteration belongs to a different scenario or fold.")
    first_packet = int(item["first_malicious_packet_ns"])
    last_packet = int(item["last_malicious_packet_ns"])
    raw_alert = item["first_detecting_window_end_ns"]
    first_alert = None if raw_alert is None else int(raw_alert)
    if (last_packet < first_packet
            or first_alert is not None and first_alert < first_packet
            or item["detected"] is not (first_alert is not None)):
        raise ValueError("An iteration has inconsistent timestamps or detection status.")
    timely = first_alert is not None and first_alert < last_packet
    late = first_alert is not None and not timely
    return {
        "scenario": scenario,
        "fold": fold,
        "attack_step": str(item["attack_step"]),
        "sequence_id": str(item["sequence_id"]),
        "attack_packets": int(item["attack_packets"]),
        "first_malicious_packet_ns": first_packet,
        "last_malicious_packet_ns": last_packet,
        "duration_seconds": (last_packet - first_packet) / NANOSECONDS_PER_SECOND,
        "first_correct_alert_ns": first_alert,
        "score_positive": first_alert is not None,
        "timely_before_last_packet": timely,
        "late_at_or_after_last_packet": late,
        "seconds_from_last_packet_to_alert": (
            (first_alert - last_packet) / NANOSECONDS_PER_SECOND
            if first_alert is not None else None),
    }


def _iteration_metrics(items: list[dict]) -> dict:
    if not items:
        raise ValueError("A scenario or attack step has no iterations.")
    total = len(items)
    positive = sum(item["score_positive"] for item in items)
    timely = sum(item["timely_before_last_packet"] for item in items)
    late = sum(item["late_at_or_after_last_packet"] for item in items)
    return {
        "iterations": total,
        "score_positive_iterations": positive,
        "timely_iterations": timely,
        "late_positive_iterations": late,
        "no_score_positive_iterations": total - positive,
        "score_positive_iteration_rate": positive / total,
        "timely_iteration_rate": timely / total,
        "late_positive_iteration_rate": late / total,
        "late_share_among_score_positive": late / positive if positive else None,
    }


def _chain_metrics(items: list[dict], terminal_steps: list[str],
                   scenario: str, fold: str) -> dict:
    terminal = [item for item in items if item["attack_step"] in terminal_steps]
    if not terminal:
        raise ValueError(f"No declared terminal action appears in {scenario}.")
    first_terminal = min(terminal, key=lambda item: item["first_malicious_packet_ns"])
    onset = min(item["first_malicious_packet_ns"] for item in items)
    terminal_onset = first_terminal["first_malicious_packet_ns"]
    score_positive = [item for item in items if item["score_positive"]]
    first_alert = min(
        (item["first_correct_alert_ns"] for item in score_positive),
        default=None,
    )
    first_alert_steps = sorted({
        item["attack_step"] for item in score_positive
        if item["first_correct_alert_ns"] == first_alert
    })
    early = first_alert is not None and first_alert < terminal_onset
    if terminal_onset < onset:
        raise ValueError("The terminal action starts before the attack chain.")
    return {
        "scenario": scenario,
        "fold": fold,
        "declared_terminal_action_steps": terminal_steps,
        "first_terminal_action_step": first_terminal["attack_step"],
        "chain_first_malicious_packet_ns": onset,
        "terminal_action_first_malicious_packet_ns": terminal_onset,
        "preterminal_opportunity_seconds": (
            terminal_onset - onset) / NANOSECONDS_PER_SECOND),
        "first_correct_alert_ns": first_alert,
        "first_correct_alert_steps": first_alert_steps,
        "score_positive_chain": first_alert is not None,
        "early_before_terminal_action": early,
        "seconds_before_terminal_action": (
            (terminal_onset - first_alert) / NANOSECONDS_PER_SECOND
            if early else None),
        "seconds_at_or_after_terminal_action": (
            (first_alert - terminal_onset) / NANOSECONDS_PER_SECOND
            if first_alert is not None and not early else None),
    }


def _hierarchical_macro(scenarios: dict[str, dict], folds: dict[str, dict]) -> dict:
    fields = ("score_positive_iteration_rate", "timely_iteration_rate",
              "late_positive_iteration_rate", "early_before_terminal_action")
    fold_means = {
        fold: {field: mean(float(scenarios[scenario][field])
                           for scenario in split["validate"])
               for field in fields}
        for fold, split in folds.items()
    }
    return {
        "fold_means": fold_means,
        "hierarchical_macro": {
            field: mean(fold_means[fold][field] for fold in folds)
            for field in fields},
    }


def _audit_budget(budget_report: dict, folds: dict[str, dict],
                  terminal_steps: dict[str, list[str]]) -> dict:
    scenario_folds = {
        scenario: fold for fold, split in folds.items()
        for scenario in split["validate"]
    }
    grouped: dict[str, list[dict]] = defaultdict(list)
    for item in budget_report["iteration_rows"]:
        scenario = item["scenario"]
        if scenario not in scenario_folds:
            raise ValueError(f"Unexpected OOF scenario: {scenario}")
        grouped[scenario].append(_classified_iteration(
            item, scenario=scenario, fold=scenario_folds[scenario]))
    if set(grouped) != set(scenario_folds):
        raise ValueError("The operational report omits a development scenario.")
    scenario_metrics = {}
    step_metrics = {}
    chain_metrics = {}
    rows = []
    for scenario, fold in scenario_folds.items():
        items = grouped[scenario]
        original = budget_report["scenario_metrics"][scenario]
        summary = _iteration_metrics(items)
        if (summary["iterations"] != original["attack_step_iterations"]
                or summary["score_positive_iterations"] != original["detected_iterations"]):
            raise ValueError(f"Iteration counts differ from the operational report for {scenario}.")
        chain = _chain_metrics(items, terminal_steps[scenario], scenario, fold)
        chain_metrics[scenario] = chain
        scenario_metrics[scenario] = {
            "scenario": scenario, "fold": fold,
            **summary,
            "early_before_terminal_action": chain["early_before_terminal_action"],
            "preterminal_opportunity_seconds": chain["preterminal_opportunity_seconds"],
        }
        steps: dict[str, list[dict]] = defaultdict(list)
        for item in items:
            steps[item["attack_step"]].append(item)
            rows.append(item)
        for step, step_items in steps.items():
            step_metrics[f"{scenario}::{step}"] = {
                "scenario": scenario, "fold": fold, "attack_step": step,
                **_iteration_metrics(step_items),
            }
    return {
        "scenario_metrics": scenario_metrics,
        "step_metrics": step_metrics,
        "chain_metrics": chain_metrics,
        "iteration_rows": rows,
        **_hierarchical_macro(scenario_metrics, folds),
    }


def run_early_warning_audit(*, operational_dir: Path, manifest_path: Path,
                            policy_path: Path, output_dir: Path) -> dict:
    """Derive warning-time metrics without retraining or loading packet Parquet."""
    operational_dir = Path(operational_dir)
    manifest_path = Path(manifest_path)
    policy_path = Path(policy_path)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError("The early-warning audit output directory must be new.")
    manifest = load_manifest(manifest_path)
    policy = _load_policy(policy_path, manifest)
    operational = _load_operational_report(operational_dir, manifest_path)
    folds = manifest["validation"]["folds"]
    expected_budgets = set(operational["budget_order"])
    result = {
        "report_version": REPORT_VERSION,
        "status": "development_oof_early_warning_audit_complete",
        "run_id": output_dir.name,
        "manifest_sha256": sha256_file(manifest_path),
        "policy_sha256": sha256_file(policy_path),
        "audit_code_sha256": sha256_file(Path(__file__)),
        "operational_report_sha256": sha256_file(
            operational_dir / "operational_report.json"),
        "operational_run_id": operational_dir.name,
        "source_input_oof_sha256": operational["input_oof_sha256"],
        "source_thresholds_selected_from": operational["thresholds_selected_from"],
        "budget_order": operational["budget_order"],
        "policy": policy,
        "test_data_accessed": False,
        "models": {},
    }
    for model_name in capture_oof_operational.MODEL_NAMES:
        source_model = operational["models"][model_name]
        if (set(source_model["budgets"]) != expected_budgets
                or set(source_model["thresholds"]) != expected_budgets):
            raise ValueError(f"Operational budgets differ for {model_name}.")
        result["models"][model_name] = {
            "thresholds": source_model["thresholds"],
            "budgets": {
                budget: _audit_budget(source_model["budgets"][budget], folds,
                                      policy["terminal_action_steps"])
                for budget in operational["budget_order"]
            },
        }
    reference_chains = result["models"]["xgb_p"]["budgets"][
        operational["budget_order"][0]]["chain_metrics"]
    for model_name in capture_oof_operational.MODEL_NAMES:
        for budget in operational["budget_order"]:
            chains = result["models"][model_name]["budgets"][budget]["chain_metrics"]
            for scenario, reference in reference_chains.items():
                current = chains[scenario]
                for field in ("chain_first_malicious_packet_ns",
                              "terminal_action_first_malicious_packet_ns",
                              "first_terminal_action_step"):
                    if current[field] != reference[field]:
                        raise ValueError(
                            f"Chain milestones differ for {model_name}/{budget}/{scenario}.")
    output_dir.mkdir(parents=True)
    write_json(output_dir / "early_warning_report.json", result)
    write_json(output_dir / "run_status.json", {
        "complete": True,
        "report_sha256": sha256_file(output_dir / "early_warning_report.json"),
    })
    return result


def validate_early_warning_audit(*, operational_dir: Path, manifest_path: Path,
                                 policy_path: Path, output_dir: Path) -> dict:
    """Verify a completed audit against its source report and frozen rules."""
    output_dir = Path(output_dir)
    report_path = output_dir / "early_warning_report.json"
    status_path = output_dir / "run_status.json"
    if not report_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("A completed early-warning audit is required.")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("complete") is not True or status.get("report_sha256") != sha256_file(report_path):
        raise ValueError("The early-warning audit report is incomplete or changed.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    operational = _load_operational_report(operational_dir, manifest_path)
    if (report.get("report_version") != REPORT_VERSION
            or report.get("status") != "development_oof_early_warning_audit_complete"
            or report.get("run_id") != output_dir.name
            or report.get("manifest_sha256") != sha256_file(manifest_path)
            or report.get("policy_sha256") != sha256_file(policy_path)
            or report.get("audit_code_sha256") != sha256_file(Path(__file__))
            or report.get("operational_report_sha256") != sha256_file(
                Path(operational_dir) / "operational_report.json")
            or report.get("source_input_oof_sha256") != operational["input_oof_sha256"]
            or report.get("test_data_accessed") is not False):
        raise ValueError("The early-warning audit belongs to a different protocol or input run.")
    return report
