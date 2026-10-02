"""Development-only score-tail and threshold-scope audit for graph OOF logits."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import yaml

from .capture_data import sha256_file, write_json
from .capture_graph_comparison import DISPLAY_NAMES, MODEL_NAMES
from .capture_graph_logit_comparison import (
    _packet_counts_by_threshold,
    _select_threshold,
)


REPORT_VERSION = 1
QUANTILES = (0.5, 0.9, 0.99, 0.999, 0.9999)
QUANTILE_LABELS = ("q50", "q90", "q99", "q999", "q9999")
PROTOCOL_FLAGS = ("is_arp", "is_ipv4", "is_ipv6", "is_tcp", "is_udp", "is_mqtt", "is_ssh")


def _load_json(path: Path, description: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {description}: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid {description}: {path}")
    return value


def load_logit_tail_audit_config(path: str | Path) -> dict:
    """Validate the frozen diagnostic scope before reading OOF data."""
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("audit_contract_version") != 1:
        raise ValueError("Unsupported score-tail audit contract.")
    if config.get("scope") != "development_only" or config.get("stage") != "post_comparison_score_tail_diagnostic":
        raise ValueError("The audit scope changed.")
    if tuple(config.get("models", [])) != MODEL_NAMES or config.get("primary_model") != "st_gnn":
        raise ValueError("The audited graph models changed.")
    if config.get("primary_budget") != "one_per_hour" or config.get("score_field") != "raw_logit":
        raise ValueError("The score or alert budget changed.")
    if config.get("threshold_rule") != "smallest_threshold_satisfying_one_false_alert_window_per_hour":
        raise ValueError("The threshold rule changed.")
    if config.get("threshold_scopes") != [
        "frozen_global", "fold_local_diagnostic", "scenario_local_diagnostic"
    ]:
        raise ValueError("The threshold scopes changed.")
    if config.get("quantiles") != list(QUANTILES):
        raise ValueError("The quantile grid changed.")
    if not isinstance(config.get("top_benign_windows_per_scenario"), int) or not 1 <= config["top_benign_windows_per_scenario"] <= 50:
        raise ValueError("Invalid top-window count.")
    if config.get("interpretation") != {
        "local_thresholds_are_in_sample_diagnostics": True,
        "local_thresholds_do_not_replace_frozen_global_threshold": True,
        "no_model_selection_from_this_audit": True,
    } or config.get("prohibitions") != {
        "model_training": True,
        "checkpoint_loading": True,
        "held_out_scenario_access": True,
        "source_artifact_overwrite": True,
    }:
        raise ValueError("The audit interpretation or prohibitions changed.")
    return config


def _validate_sources(config: dict, comparison_dir: Path, rescoring_dir: Path) -> tuple[dict, dict, dict]:
    bindings = config["bindings"]
    if comparison_dir.name != bindings["corrected_comparison_run_id"] or rescoring_dir.name != bindings["rescoring_run_id"]:
        raise ValueError("The bound comparison or rescoring run changed.")
    comparison_path = comparison_dir / "comparison_report.json"
    comparison_status = _load_json(comparison_dir / "run_status.json", "comparison status")
    comparison = _load_json(comparison_path, "corrected comparison")
    rescoring_path = rescoring_dir / "rescoring_manifest.json"
    rescoring_status = _load_json(rescoring_dir / "run_status.json", "rescoring status")
    rescoring = _load_json(rescoring_path, "rescoring manifest")
    if (comparison_status.get("complete") is not True
            or comparison_status.get("report_sha256") != sha256_file(comparison_path)
            or comparison.get("status") != "exploratory_graph_logit_comparison_complete"
            or comparison.get("operational_graph_score_field") != "raw_logit"
            or comparison.get("rescoring_manifest_sha256") != sha256_file(rescoring_path)
            or comparison.get("models_compared") != list(MODEL_NAMES)
            or rescoring_status.get("complete") is not True
            or rescoring_status.get("manifest_sha256") != sha256_file(rescoring_path)
            or rescoring.get("status") != "complete"
            or rescoring.get("models") != list(MODEL_NAMES)
            or comparison.get("held_out_scenarios_accessed") is not False
            or rescoring.get("held_out_scenarios_accessed") is not False):
        raise ValueError("The bound source reports are incomplete or changed.")
    for filename, expected in comparison_status.get("artifact_sha256", {}).items():
        if sha256_file(comparison_dir / filename) != expected:
            raise ValueError(f"Corrected comparison artifact changed: {filename}")
    return comparison, rescoring, {
        "comparison_report_sha256": sha256_file(comparison_path),
        "rescoring_manifest_sha256": sha256_file(rescoring_path),
    }


def _paths(rescoring_dir: Path, rescoring: dict, folds: dict) -> dict[str, dict[str, Path]]:
    expected = {scenario: fold for fold, split in folds.items() for scenario in split["validate"]}
    if len(expected) != sum(len(split["validate"]) for split in folds.values()):
        raise ValueError("A development scenario is assigned to multiple folds.")
    result = {}
    for model in MODEL_NAMES:
        outputs = rescoring["scenario_outputs"][model]
        if set(outputs) != set(expected):
            raise ValueError(f"OOF scenario coverage changed for {model}.")
        result[model] = {}
        for scenario, item in outputs.items():
            path = (rescoring_dir / item["path"]).resolve()
            if (not path.is_relative_to(rescoring_dir.resolve())
                    or item["fold"] != expected[scenario]
                    or item["operational_ranking_field"] != "raw_logit"
                    or sha256_file(path) != item["sha256"]
                    or pq.ParquetFile(path).metadata.num_rows != int(item["rows"])):
                raise ValueError(f"OOF artifact changed: {model}/{scenario}")
            schema = pq.read_schema(path)
            required = {"source_row_id", "window_index", "binary_label", "raw_logit"}
            if not required.issubset(schema.names):
                raise ValueError(f"OOF schema changed: {model}/{scenario}")
            result[model][scenario] = path
    return result


def _packet_tails(connection: duckdb.DuckDBPyConnection, path: Path, model: str, fold: str, scenario: str) -> list[dict]:
    rows = connection.execute("""
        SELECT binary_label, count(*), min(raw_logit),
               approx_quantile(CAST(raw_logit AS DOUBLE), [0.5, 0.9, 0.99, 0.999, 0.9999]),
               max(raw_logit)
        FROM read_parquet(?) GROUP BY binary_label ORDER BY binary_label
    """, [str(path)]).fetchall()
    if len(rows) != 2 or [int(row[0]) for row in rows] != [0, 1]:
        raise ValueError(f"Both packet classes are required: {model}/{scenario}")
    return [{
        "model": model, "fold": fold, "scenario": scenario,
        "class": "attack" if int(label) else "benign", "packets": int(count),
        "minimum_raw_logit": float(minimum),
        **{name: float(value) for name, value in zip(QUANTILE_LABELS, quantiles)},
        "maximum_raw_logit": float(maximum),
    } for label, count, minimum, quantiles, maximum in rows]


def _window_data(connection: duckdb.DuckDBPyConnection, path: Path, model: str, fold: str, scenario: str, top_count: int) -> tuple[list[dict], dict, list[dict]]:
    rows = connection.execute("""
        SELECT window_index, min(window_end_ns), max(window_end_ns), count(*),
               sum(binary_label), max(CAST(raw_logit AS DOUBLE)),
               max(CAST(raw_logit AS DOUBLE)) FILTER (WHERE binary_label = 1)
        FROM read_parquet(?) GROUP BY window_index ORDER BY window_index
    """, [str(path)]).fetchall()
    if not rows or int(rows[0][0]) != 0:
        raise ValueError(f"Missing initial window: {model}/{scenario}")
    windows = []
    for index, first_end, last_end, packets, attacks, score, attack_score in rows:
        if first_end != last_end or not np.isfinite(score):
            raise ValueError(f"Invalid window time or logit: {model}/{scenario}/{index}")
        windows.append({
            "window_index": int(index), "window_end_ns": int(first_end),
            "packets": int(packets), "attack_packets": int(attacks),
            "maximum_raw_logit": float(score),
            "maximum_attack_raw_logit": None if attack_score is None else float(attack_score),
        })
    total_windows = windows[-1]["window_index"] + 1
    benign = [item for item in windows if item["attack_packets"] == 0]
    attack = [item for item in windows if item["attack_packets"] > 0]
    exposure_hours = (total_windows - len(attack)) * 5 / 3600
    if not benign or not attack or exposure_hours <= 0:
        raise ValueError(f"Invalid benign or attack exposure: {model}/{scenario}")
    negative_scores = np.sort(np.asarray([item["maximum_raw_logit"] for item in benign], dtype=np.float64))
    tails = []
    for window_class, subset in (("benign", benign), ("attack_containing", attack)):
        scores = np.asarray([item["maximum_raw_logit"] for item in subset], dtype=np.float64)
        tails.append({
            "model": model, "fold": fold, "scenario": scenario,
            "class": window_class, "observed_windows": len(subset),
            "empty_windows": total_windows - len(windows) if window_class == "benign" else 0,
            "minimum_raw_logit": float(np.min(scores)),
            **{name: float(value) for name, value in zip(QUANTILE_LABELS, np.quantile(scores, QUANTILES))},
            "maximum_raw_logit": float(np.max(scores)),
        })
    summary = {
        "negative_scores": negative_scores,
        "attack_window_scores": np.asarray([
            item["maximum_attack_raw_logit"] for item in attack
        ], dtype=np.float64),
        "exposure_hours": exposure_hours,
        "total_wall_clock_windows": total_windows,
        "attack_windows": len(attack),
        "benign_windows": total_windows - len(attack),
    }
    top = sorted(benign, key=lambda item: (-item["maximum_raw_logit"], item["window_index"]))[:top_count]
    return tails, summary, [
        {"model": model, "fold": fold, "scenario": scenario,
         "rank_within_scenario": rank, **item}
        for rank, item in enumerate(top, start=1)
    ]


def _diagnose_top_windows(connection: duckdb.DuckDBPyConnection, prepared_path: Path, oof_path: Path, top_rows: list[dict]) -> list[dict]:
    """Join only selected benign windows back to canonical packet metadata."""
    selected = [int(row["window_index"]) for row in top_rows]
    if not selected:
        return []
    schema = set(pq.read_schema(prepared_path).names)
    required = {"source_row_id", "src_endpoint", "dst_endpoint", *PROTOCOL_FLAGS}
    if not required.issubset(schema):
        raise ValueError(f"Prepared metadata is missing diagnostic columns: {sorted(required - schema)}")
    placeholders = ", ".join("?" for _ in selected)
    joined = f"""
        WITH selected AS (
            SELECT source_row_id, window_index, binary_label, raw_logit
            FROM read_parquet(?) WHERE window_index IN ({placeholders})
        ), joined AS (
            SELECT s.source_row_id, s.window_index, s.binary_label, s.raw_logit,
                   p.src_endpoint, p.dst_endpoint,
                   {', '.join('p.' + field for field in PROTOCOL_FLAGS)}
            FROM selected s INNER JOIN read_parquet(?) p USING (source_row_id)
        )
    """
    parameters = [str(oof_path), *selected, str(prepared_path)]
    protocol_sql = ", ".join(f"sum(CAST({field} AS INTEGER)) AS {field}_packets" for field in PROTOCOL_FLAGS)
    aggregates = connection.execute(joined + f"""
        SELECT window_index, count(*), sum(binary_label), {protocol_sql},
               count(DISTINCT (src_endpoint, dst_endpoint))
        FROM joined GROUP BY window_index
    """, parameters).fetchall()
    node_rows = connection.execute(joined + """
        SELECT window_index, count(DISTINCT endpoint) FROM (
            SELECT window_index, src_endpoint AS endpoint FROM joined
            UNION ALL SELECT window_index, dst_endpoint AS endpoint FROM joined
        ) GROUP BY window_index
    """, parameters).fetchall()
    pair_rows = connection.execute(joined + """
        , pairs AS (
            SELECT window_index, src_endpoint, dst_endpoint, count(*) AS multiplicity
            FROM joined GROUP BY window_index, src_endpoint, dst_endpoint
        ), ranked AS (
            SELECT *, row_number() OVER (
                PARTITION BY window_index
                ORDER BY multiplicity DESC, src_endpoint, dst_endpoint
            ) AS pair_rank FROM pairs
        )
        SELECT window_index, src_endpoint, dst_endpoint, multiplicity
        FROM ranked WHERE pair_rank = 1
    """, parameters).fetchall()
    top_packet_rows = connection.execute(joined + """
        SELECT window_index, source_row_id, raw_logit, src_endpoint, dst_endpoint,
               is_arp, is_ipv4, is_ipv6, is_tcp, is_udp, is_mqtt, is_ssh
        FROM joined
        QUALIFY row_number() OVER (
            PARTITION BY window_index ORDER BY raw_logit DESC, source_row_id
        ) = 1
    """, parameters).fetchall()
    by_node = {int(index): int(nodes) for index, nodes in node_rows}
    by_pair = {int(index): (str(src), str(dst), int(count)) for index, src, dst, count in pair_rows}
    by_top_packet = {int(row[0]): row for row in top_packet_rows}
    by_aggregate = {int(row[0]): row for row in aggregates}
    if (set(by_aggregate) != set(selected) or set(by_node) != set(selected)
            or set(by_pair) != set(selected) or set(by_top_packet) != set(selected)):
        raise ValueError("Top-window packet join did not cover every selected window.")
    result = []
    for item in top_rows:
        index = item["window_index"]
        row = by_aggregate[index]
        if int(row[1]) != item["packets"] or int(row[2]) != 0:
            raise ValueError(f"Top-window packet join changed benign counts: {index}")
        source, destination, multiplicity = by_pair[index]
        triggering = by_top_packet[index]
        if float(triggering[2]) != item["maximum_raw_logit"]:
            raise ValueError(f"Top-window triggering logit changed: {index}")
        result.append({
            **item, "distinct_nodes": by_node[index],
            "distinct_directed_pairs": int(row[-1]),
            **{f"{field}_packets": int(row[3 + position]) for position, field in enumerate(PROTOCOL_FLAGS)},
            "most_frequent_pair_multiplicity": multiplicity,
            "most_frequent_pair_source_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "most_frequent_pair_destination_sha256": hashlib.sha256(destination.encode("utf-8")).hexdigest(),
            "triggering_source_row_id": int(triggering[1]),
            "triggering_source_sha256": hashlib.sha256(str(triggering[3]).encode("utf-8")).hexdigest(),
            "triggering_destination_sha256": hashlib.sha256(str(triggering[4]).encode("utf-8")).hexdigest(),
            **{f"triggering_{field}": int(triggering[5 + position]) for position, field in enumerate(PROTOCOL_FLAGS)},
        })
    return result


def _markdown_table(rows: list[dict], columns: tuple[tuple[str, str], ...], digits: int = 4) -> str:
    lines = ["| " + " | ".join(label for _, label in columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key)
            values.append("NA" if value is None else f"{value:.{digits}f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_review(output_dir: Path, report: dict) -> None:
    comparison = [row for row in report["threshold_scope_metrics"] if row["model"] == "st_gnn"]
    global_rows = [row for row in comparison if row["scope"] == "frozen_global"]
    summary = [row for row in report["fold_scale_summary"] if row["model"] == "st_gnn"]
    text = "\n".join([
        "# cAPTure graph logit tail audit", "",
        "Status: exploratory diagnostic complete. The frozen global operational threshold remains primary.", "",
        "## ST-GNN fold score scales", "",
        _markdown_table(summary, (("fold", "Fold"), ("frozen_global_threshold", "Global threshold"),
                                  ("fold_local_threshold", "Fold diagnostic threshold"),
                                  ("maximum_attack_packet_logit", "Maximum attack logit"),
                                  ("global_fold_mean_packet_recall", "Global mean recall"),
                                  ("local_fold_mean_packet_recall", "Local mean recall"),
                                  ("local_fold_mean_false_alert_windows_per_hour", "Local mean FA/h"))), "",
        "## ST-GNN at the frozen global threshold", "",
        _markdown_table(global_rows, (("scenario", "Scenario"), ("fold", "Fold"),
                                      ("threshold", "Threshold"), ("false_alert_windows_per_hour", "FA/h"),
                                      ("packet_recall", "Packet recall"),
                                      ("attack_window_recall", "Attack-window recall"))), "",
        "## Diagnostic threshold scopes", "",
        "`fold_local_diagnostic` and `scenario_local_diagnostic` select and evaluate a threshold on the same OOF predictions. "
        "They are optimistic diagnostic counterfactuals, not deployable calibration results. "
        "Compare the three scopes in `threshold_scope_metrics.csv`; only `frozen_global` reproduces the primary result.", "",
        "## Benign upper tail", "",
        "`top_benign_windows.csv` contains the ten highest-scoring benign windows per scenario and model. "
        "`st_gnn_top_window_metadata.csv` joins ST-GNN's selected windows to prepared packets and reports "
        "protocol counts, node and pair counts, the triggering packet's protocol flags, "
        "and hashes of its endpoints and the most frequent directed pair. "
        "Raw MAC addresses are not written.", "",
        "## Interpretation", "",
        "Check whether fold or scenario diagnostic thresholds recover recall while satisfying their local budget. "
        "Also check whether the benign upper tail overlaps attack scores within each fold. "
        "A local recovery suggests score-scale mismatch; persistent low recall suggests within-scenario tail overlap. "
        "These mechanisms can coexist. The audit does not establish a cause in the model internals.", "",
        "No training, checkpoint loading, source overwrite, or held-out scenario access occurred.", "",
    ])
    (output_dir / "review.md").write_text(text, encoding="utf-8")


def run_logit_tail_audit(*, config_path: str | Path, comparison_dir: str | Path,
                         rescoring_dir: str | Path, prepared_dir: str | Path,
                         output_dir: str | Path, batch_size: int = 250_000) -> dict:
    """Build a versioned audit from immutable development OOF and prepared packets."""
    config_path = Path(config_path).expanduser().resolve()
    comparison_dir = Path(comparison_dir).expanduser().resolve()
    rescoring_dir = Path(rescoring_dir).expanduser().resolve()
    prepared_dir = Path(prepared_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError("The audit output directory must be new.")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    config = load_logit_tail_audit_config(config_path)
    if prepared_dir.name != config["bindings"]["prepared_run_id"]:
        raise ValueError("The prepared run changed.")
    comparison, rescoring, sources = _validate_sources(config, comparison_dir, rescoring_dir)
    folds = rescoring["folds"]
    normalized_folds = {fold: {"validate": item["outer_validation"]} for fold, item in folds.items()}
    paths = _paths(rescoring_dir, rescoring, normalized_folds)
    prepared_paths = {}
    prepared_hashes = {}
    for scenario in paths["st_gnn"]:
        packet_path = prepared_dir / scenario / "packets.capture_packet_v1.parquet"
        checksums = _load_json(prepared_dir / scenario / "artifact_checksums.json", "prepared checksums")
        if checksums.get(packet_path.name) != sha256_file(packet_path):
            raise ValueError(f"Prepared packet checksum changed: {scenario}")
        prepared_paths[scenario] = packet_path
        prepared_hashes[scenario] = checksums[packet_path.name]
    packet_tails, window_tails, top_windows, local_rows, fold_rows = [], [], [], [], []
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 2")
        connection.execute("SET memory_limit = '4GB'")
        for model in MODEL_NAMES:
            print(f"Auditing score tails for {DISPLAY_NAMES[model]}...", flush=True)
            summaries = {}
            for fold, split in normalized_folds.items():
                for scenario in split["validate"]:
                    path = paths[model][scenario]
                    packet_tails.extend(_packet_tails(connection, path, model, fold, scenario))
                    tail, summary, top = _window_data(
                        connection, path, model, fold, scenario,
                        config["top_benign_windows_per_scenario"],
                    )
                    window_tails.extend(tail)
                    summaries[scenario] = summary
                    top_windows.extend(top)
            primary = comparison["graph_models"][model]["thresholds"]["one_per_hour"]
            frozen = float(primary["threshold"])
            recomputed = _select_threshold(summaries, normalized_folds, 1.0)["threshold"]
            if frozen != recomputed:
                raise ValueError(f"The frozen global threshold cannot be reproduced for {model}.")
            for item in top_windows:
                if item["model"] == model:
                    item["above_frozen_global_threshold"] = item["maximum_raw_logit"] >= frozen
            fold_thresholds = {
                fold: _select_threshold(summaries, {fold: split}, 1.0)["threshold"]
                for fold, split in normalized_folds.items()
            }
            scenario_thresholds = {
                scenario: _select_threshold(summaries, {fold: {"validate": [scenario]}}, 1.0)["threshold"]
                for fold, split in normalized_folds.items() for scenario in split["validate"]
            }
            for fold, split in normalized_folds.items():
                attacks = [row for row in packet_tails if row["model"] == model and row["fold"] == fold and row["class"] == "attack"]
                benign_window_scores = np.concatenate([
                    summaries[scenario]["negative_scores"] for scenario in split["validate"]
                ])
                attack_window_scores = np.concatenate([
                    summaries[scenario]["attack_window_scores"] for scenario in split["validate"]
                ])
                fold_rows.append({
                    "model": model, "fold": fold, "frozen_global_threshold": frozen,
                    "fold_local_threshold": fold_thresholds[fold],
                    "maximum_attack_packet_logit": max(row["maximum_raw_logit"] for row in attacks),
                    "benign_window_logit_q99": float(np.quantile(benign_window_scores, 0.99)),
                    "benign_window_logit_q999": float(np.quantile(benign_window_scores, 0.999)),
                    "maximum_benign_window_logit": float(np.max(benign_window_scores)),
                    "attack_window_logit_q99": float(np.quantile(attack_window_scores, 0.99)),
                    "global_attack_packets_above_threshold": None,
                })
                for scenario in split["validate"]:
                    thresholds = {
                        "frozen_global": frozen,
                        "fold_local_diagnostic": fold_thresholds[fold],
                        "scenario_local_diagnostic": scenario_thresholds[scenario],
                    }
                    path = paths[model][scenario]
                    counts = _packet_counts_by_threshold(
                        path, score_field="raw_logit", thresholds=thresholds,
                        expected_rows=int(rescoring["scenario_outputs"][model][scenario]["rows"]),
                        batch_size=batch_size,
                    )
                    summary = summaries[scenario]
                    attack_scores = summary["attack_window_scores"]
                    for scope, threshold in thresholds.items():
                        packet_counts = counts[scope]
                        false_alert_windows = int(np.count_nonzero(
                            summary["negative_scores"] >= threshold
                        ))
                        tp, fp = packet_counts["tp"], packet_counts["fp"]
                        attack_packets = tp + packet_counts["fn"]
                        false_alert_rate = false_alert_windows / summary["exposure_hours"]
                        packet_recall = tp / attack_packets
                        if scope == "frozen_global":
                            reference = comparison["graph_models"][model]["budgets"]["one_per_hour"]["scenario_metrics"][scenario]
                            if (not np.isclose(false_alert_rate, reference["false_alert_windows_per_hour"], rtol=0, atol=1e-12)
                                    or not np.isclose(packet_recall, reference["packet_recall"], rtol=0, atol=1e-12)):
                                raise ValueError(f"The frozen operational result changed: {model}/{scenario}")
                        local_rows.append({
                            "model": model, "fold": fold, "scenario": scenario,
                            "scope": scope, "threshold": threshold,
                            "false_alert_windows": false_alert_windows,
                            "false_alert_windows_per_hour": false_alert_rate,
                            "packet_true_positives": tp,
                            "packet_false_positives": fp,
                            "packet_recall": packet_recall,
                            "packet_precision": tp / (tp + fp) if tp + fp else None,
                            "attack_window_recall": float(np.mean(attack_scores >= threshold)),
                            "attack_windows": len(attack_scores),
                            "maximum_attack_window_logit": float(np.max(attack_scores)),
                            "in_sample_diagnostic": scope != "frozen_global",
                        })
            for row in fold_rows:
                if row["model"] == model:
                    scope_rows = [
                        item for item in local_rows
                        if item["model"] == model and item["fold"] == row["fold"]
                    ]
                    global_rows = [item for item in scope_rows if item["scope"] == "frozen_global"]
                    local_fold_rows = [item for item in scope_rows if item["scope"] == "fold_local_diagnostic"]
                    row["global_attack_packets_above_threshold"] = sum(
                        item["packet_true_positives"] for item in global_rows
                    )
                    row["global_fold_mean_false_alert_windows_per_hour"] = float(np.mean([
                        item["false_alert_windows_per_hour"] for item in global_rows
                    ]))
                    row["local_fold_mean_false_alert_windows_per_hour"] = float(np.mean([
                        item["false_alert_windows_per_hour"] for item in local_fold_rows
                    ]))
                    row["global_fold_mean_packet_recall"] = float(np.mean([
                        item["packet_recall"] for item in global_rows
                    ]))
                    row["local_fold_mean_packet_recall"] = float(np.mean([
                        item["packet_recall"] for item in local_fold_rows
                    ]))
                    if row["local_fold_mean_false_alert_windows_per_hour"] > 1.0 + 1e-12:
                        raise ValueError(f"The fold diagnostic exceeded its false-alert budget: {model}/{row['fold']}")
        st_top = [row for row in top_windows if row["model"] == "st_gnn"]
        top_metadata = []
        for scenario in paths["st_gnn"]:
            selected = [row for row in st_top if row["scenario"] == scenario]
            top_metadata.extend(_diagnose_top_windows(
                connection, prepared_paths[scenario], paths["st_gnn"][scenario], selected
            ))
    finally:
        connection.close()
    report = {
        "report_version": REPORT_VERSION,
        "status": "exploratory_logit_tail_audit_complete",
        "run_id": output_dir.name,
        "audit_contract_sha256": sha256_file(config_path),
        **sources,
        "prepared_packet_sha256": prepared_hashes,
        "score_field": "raw_logit",
        "primary_budget_false_alert_windows_per_hour": 1.0,
        "local_thresholds_are_in_sample_diagnostics": True,
        "global_threshold_unchanged": True,
        "model_training_performed": False,
        "checkpoint_loading_performed": False,
        "held_out_scenarios_accessed": False,
        "packet_logit_tails": packet_tails,
        "window_logit_tails": window_tails,
        "threshold_scope_metrics": local_rows,
        "fold_scale_summary": fold_rows,
        "top_benign_windows": top_windows,
        "st_gnn_top_window_metadata": top_metadata,
    }
    output_dir.mkdir(parents=True)
    write_json(output_dir / "audit_report.json", report)
    for name in ("packet_logit_tails", "window_logit_tails", "threshold_scope_metrics",
                 "fold_scale_summary", "top_benign_windows", "st_gnn_top_window_metadata"):
        pd.DataFrame(report[name]).to_csv(output_dir / f"{name}.csv", index=False)
    _write_review(output_dir, report)
    write_json(output_dir / "run_status.json", {
        "complete": True,
        "report_sha256": sha256_file(output_dir / "audit_report.json"),
        "artifact_sha256": {
            path.name: sha256_file(path) for path in sorted(output_dir.iterdir())
            if path.name != "run_status.json"
        },
        "model_training_performed": False,
        "held_out_scenarios_accessed": False,
    })
    return report


def validate_logit_tail_audit(*, config_path: str | Path, comparison_dir: str | Path,
                              rescoring_dir: str | Path, prepared_dir: str | Path,
                              output_dir: str | Path) -> dict:
    """Check an existing report and its immutable source bindings."""
    config_path = Path(config_path).expanduser().resolve()
    comparison_dir = Path(comparison_dir).expanduser().resolve()
    rescoring_dir = Path(rescoring_dir).expanduser().resolve()
    prepared_dir = Path(prepared_dir).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    config = load_logit_tail_audit_config(config_path)
    comparison, rescoring, sources = _validate_sources(config, comparison_dir, rescoring_dir)
    paths = _paths(rescoring_dir, rescoring, {
        fold: {"validate": item["outer_validation"]} for fold, item in rescoring["folds"].items()
    })
    report_path = output_dir / "audit_report.json"
    report = _load_json(report_path, "score-tail report")
    status = _load_json(output_dir / "run_status.json", "score-tail status")
    if (status.get("complete") is not True
            or status.get("report_sha256") != sha256_file(report_path)
            or report.get("status") != "exploratory_logit_tail_audit_complete"
            or report.get("run_id") != output_dir.name
            or report.get("audit_contract_sha256") != sha256_file(config_path)
            or report.get("comparison_report_sha256") != sources["comparison_report_sha256"]
            or report.get("rescoring_manifest_sha256") != sources["rescoring_manifest_sha256"]
            or report.get("model_training_performed") is not False
            or report.get("held_out_scenarios_accessed") is not False):
        raise ValueError("The score-tail audit is incomplete or changed.")
    for scenario, expected in report["prepared_packet_sha256"].items():
        if (prepared_dir.name != config["bindings"]["prepared_run_id"]
                or sha256_file(prepared_dir / scenario / "packets.capture_packet_v1.parquet") != expected):
            raise ValueError(f"Prepared packet changed: {scenario}")
    for name, expected in status["artifact_sha256"].items():
        if sha256_file(output_dir / name) != expected:
            raise ValueError(f"Audit artifact changed: {name}")
    return report
