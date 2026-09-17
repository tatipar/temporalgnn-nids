"""Post-audit cAPTure diagnostics used to close Gate 0 decisions."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import platform
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .capture_data import load_manifest, selected_scenarios, sha256_file, write_json


ANNOTATION_COLUMNS = {
    "label", "phase_idx", "phase_name", "phase_number", "step_number", "sequence_id",
}
ABSOLUTE_TIME_COLUMNS = {
    "timestamp", "layers_frame_frame.time", "layers_frame_frame.time_epoch",
}
RECORD_IDENTIFIER_COLUMNS = {
    "layers_frame_frame.number", "layers_frame_frame.section_number",
}
ENDPOINT_IDENTITY_COLUMNS = {
    "layers_eth_eth.src", "layers_eth_eth.dst",
    "layers_eth_eth.dst_tree_eth.addr",
    "layers_eth_eth.dst_tree_eth.addr.oui",
    "layers_eth_eth.dst_tree_eth.addr_resolved",
    "layers_eth_eth.dst_tree_eth.dst.oui",
    "layers_eth_eth.dst_tree_eth.dst_resolved",
    "layers_eth_eth.src_tree_eth.addr.oui",
    "layers_eth_eth.src_tree_eth.src.oui",
    "layers_eth_eth.src_tree_eth.src_resolved",
    "layers_ip_ip.addr", "layers_ip_ip.host", "layers_ip_ip.src_host",
    "layers_ipv6_ipv6.addr", "layers_ipv6_ipv6.host",
}


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _records(connection: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    cursor = connection.execute(sql)
    names = [item[0] for item in cursor.description]
    return [dict(zip(names, row)) for row in cursor.fetchall()]


def load_full_dev_artifacts(run_dir: Path, manifest_path: Path) -> tuple[dict, dict, dict[str, Path]]:
    """Validate one completed FULL_DEV run and return reports and packet artifacts."""
    run_dir, manifest_path = Path(run_dir), Path(manifest_path)
    manifest = load_manifest(manifest_path)
    status_path, config_path = run_dir / "run_status.json", run_dir / "run_config.json"
    if not status_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("The run is missing run_status.json or run_config.json.")
    status, config = json.loads(status_path.read_text()), json.loads(config_path.read_text())
    if status.get("complete") is not True or config.get("mode") != "FULL_DEV":
        raise ValueError("Decision analysis requires one completed FULL_DEV run.")
    if config.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("The FULL_DEV run does not match the current manifest.")
    expected = selected_scenarios(manifest, "FULL_DEV")
    if set(status.get("scenarios", {})) != set(expected):
        raise ValueError("The FULL_DEV scenario set is incomplete.")
    reports, packets = {}, {}
    for scenario in expected:
        directory = run_dir / scenario
        report_path = directory / "audit_report.json"
        parquet_path = directory / "packets.audit.parquet"
        checksum_path = directory / "artifact_checksums.json"
        if not all(path.is_file() for path in (report_path, parquet_path, checksum_path)):
            raise FileNotFoundError(f"Missing persisted artifacts for {scenario}.")
        checksums = json.loads(checksum_path.read_text())
        if checksums.get("audit_report.json") != sha256_file(report_path):
            raise ValueError(f"Audit report checksum mismatch: {scenario}")
        report = json.loads(report_path.read_text())
        if (report.get("scenario") != scenario or report.get("mode") != "FULL_DEV"
                or report.get("manifest_sha256") != config["manifest_sha256"]):
            raise ValueError(f"Audit report binding mismatch: {scenario}")
        if report.get("blockers"):
            raise ValueError(f"Unresolved automatic blockers in {scenario}: {report['blockers']}")
        declared_parquet_hash = checksums.get("packets.audit.parquet")
        if not declared_parquet_hash or declared_parquet_hash != report.get("audit_parquet_sha256"):
            raise ValueError(f"Audit Parquet checksum binding mismatch: {scenario}")
        if sha256_file(parquet_path) != declared_parquet_hash:
            raise ValueError(f"Audit Parquet checksum mismatch: {scenario}")
        reports[scenario], packets[scenario] = report, parquet_path
    return manifest, reports, packets


def _raw_columns(path: Path) -> set[str]:
    return {name for name in pq.read_schema(path).names if name.startswith("raw::")}


def ordered_benign_signature(path: Path, columns: Iterable[str], *, memory_limit: str = "2GB",
                             threads: int = 2, batch_size: int = 100_000) -> dict:
    """Hash an ordered stream of DuckDB row hashes for exact-background comparison.

    Row hashes use every common raw column. SHA-256 then binds their ordered stream.
    The method is a high-confidence equality diagnostic, not a cryptographic proof
    over the original CSV byte representation.
    """
    columns = sorted(columns)
    if not columns:
        raise ValueError("At least one raw column is required for a benign signature.")
    digest = hashlib.sha256()
    digest.update(json.dumps(columns, separators=(",", ":")).encode())
    expressions = ", ".join(_identifier(column) for column in columns)
    path_literal = _literal(str(Path(path)))
    sql = f"""SELECT row_hash FROM (
        SELECT packet_timestamp_ns, hash({expressions}) AS row_hash
        FROM read_parquet({path_literal}) WHERE binary_label=0
        ) ORDER BY packet_timestamp_ns, row_hash"""
    connection = duckdb.connect()
    count = 0
    try:
        connection.execute(f"SET memory_limit={_literal(memory_limit)}")
        connection.execute(f"SET threads={int(threads)}")
        reader = connection.execute(sql).fetch_record_batch(rows_per_batch=batch_size)
        for batch in reader:
            values = batch.column(0).to_numpy(zero_copy_only=False)
            digest.update(values.astype("<u8", copy=False).tobytes())
            count += len(values)
    finally:
        connection.close()
    return {"benign_packets": count, "sha256_of_ordered_duckdb_row_hashes": digest.hexdigest(),
            "common_raw_columns": len(columns)}


def compare_benign_backgrounds(manifest: dict, packets: dict[str, Path], *,
                               memory_limit: str = "2GB", threads: int = 2) -> list[dict]:
    groups: dict[str, list[str]] = defaultdict(list)
    for scenario in selected_scenarios(manifest, "FULL_DEV"):
        groups[manifest["scenarios"][scenario]["benign_source"]].append(scenario)
    results = []
    for benign_source, scenarios in groups.items():
        common = set.intersection(*(_raw_columns(packets[scenario]) for scenario in scenarios))
        signatures = {scenario: ordered_benign_signature(
            packets[scenario], common, memory_limit=memory_limit, threads=threads,
        ) for scenario in scenarios}
        keys = {(item["benign_packets"], item["sha256_of_ordered_duckdb_row_hashes"])
                for item in signatures.values()}
        results.append({"benign_source": benign_source, "scenarios": scenarios,
                        "common_raw_columns": len(common), "signatures": signatures,
                        "all_benign_packets_identical": len(keys) == 1})
    return results


def audit_topology(path: Path, scenario: str, *, memory_limit: str = "2GB",
                   threads: int = 2) -> dict:
    required = {
        "raw::layers_eth_eth.src", "raw::layers_eth_eth.dst",
        "raw::layers_ip_ip.version", "raw::layers_ipv6_ipv6.version",
        "raw::layers_frame_frame.protocols",
    }
    missing = required - _raw_columns(path)
    if missing:
        raise ValueError(f"Missing topology columns in {scenario}: {sorted(missing)}")
    src, dst = _identifier("raw::layers_eth_eth.src"), _identifier("raw::layers_eth_eth.dst")
    ipv4, ipv6 = _identifier("raw::layers_ip_ip.version"), _identifier("raw::layers_ipv6_ipv6.version")
    protocols = _identifier("raw::layers_frame_frame.protocols")
    valid = r"^[0-9a-f]{2}(:[0-9a-f]{2}){5}$"
    path_literal = _literal(str(Path(path)))
    sql = f"""WITH base AS (
        SELECT lower(trim({src})) AS src, lower(trim({dst})) AS dst,
            nullif(trim({ipv4}), '') IS NOT NULL AS has_ipv4,
            nullif(trim({ipv6}), '') IS NOT NULL AS has_ipv6,
            lower(coalesce({protocols}, '')) AS protocols
        FROM read_parquet({path_literal})
    ), typed AS (
        SELECT *, regexp_full_match(src, {_literal(valid)}) AS valid_src,
            regexp_full_match(dst, {_literal(valid)}) AS valid_dst,
            dst='ff:ff:ff:ff:ff:ff' AS broadcast_dst,
            regexp_full_match(dst, {_literal(valid)})
                AND substr(dst,2,1) IN ('1','3','5','7','9','b','d','f') AS group_dst
        FROM base
    ) SELECT count(*) AS packets,
        count(*) FILTER (WHERE NOT valid_src OR NOT valid_dst) AS invalid_mac_packets,
        count(*) FILTER (WHERE src=dst) AS self_loop_packets,
        count(*) FILTER (WHERE broadcast_dst) AS broadcast_destination_packets,
        count(*) FILTER (WHERE group_dst AND NOT broadcast_dst) AS multicast_destination_packets,
        count(*) FILTER (WHERE valid_dst AND NOT group_dst) AS unicast_destination_packets,
        count(*) FILTER (WHERE has_ipv4 AND NOT has_ipv6) AS ipv4_packets,
        count(*) FILTER (WHERE has_ipv6 AND NOT has_ipv4) AS ipv6_packets,
        count(*) FILTER (WHERE NOT has_ipv4 AND NOT has_ipv6) AS non_ip_packets,
        count(*) FILTER (WHERE has_ipv4 AND has_ipv6) AS conflicting_ip_version_packets,
        count(*) FILTER (WHERE protocols LIKE '%arp%') AS arp_protocol_packets,
        count(DISTINCT src) AS unique_sources, count(DISTINCT dst) AS unique_destinations
        FROM typed"""
    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit={_literal(memory_limit)}")
        connection.execute(f"SET threads={int(threads)}")
        counts = _records(connection, sql)[0]
        special = _records(connection, f"""WITH base AS (
            SELECT lower(trim({dst})) AS dst FROM read_parquet({path_literal})
        ) SELECT dst AS destination, count(*) AS packets FROM base
        WHERE dst='ff:ff:ff:ff:ff:ff' OR (
            regexp_full_match(dst, {_literal(valid)})
            AND substr(dst,2,1) IN ('1','3','5','7','9','b','d','f'))
        GROUP BY 1 ORDER BY 2 DESC, 1 LIMIT 20""")
    finally:
        connection.close()
    return {"scenario": scenario, **counts, "top_group_destinations": special}


def _suggested_role(column: str) -> str:
    lower = column.lower()
    if column in ANNOTATION_COLUMNS:
        return "evaluation_metadata"
    if column in ABSOLUTE_TIME_COLUMNS:
        return "ordering_metadata"
    if column in RECORD_IDENTIFIER_COLUMNS:
        return "record_identifier"
    if column in ENDPOINT_IDENTITY_COLUMNS:
        return "topology_identity"
    if ("payload" in lower or "mqtt.msg" in lower
            or "encryption_algorithms" in lower or "mac_algorithms" in lower
            or "compression_algorithms" in lower):
        return "content_or_protocol_fingerprint"
    return "candidate_packet_feature"


def build_feature_inventory(reports: dict[str, dict]) -> list[dict]:
    scenarios = sorted(reports)
    all_columns = sorted(set().union(*(report["column_profiles"] for report in reports.values())))
    duplicate_occurrences: dict[tuple[str, ...], int] = defaultdict(int)
    for report in reports.values():
        for group in report["duplicate_column_groups_sha256"]:
            duplicate_occurrences[tuple(sorted(group))] += 1
    duplicate_member = {}
    for group, occurrences in duplicate_occurrences.items():
        keeper = min(group, key=lambda value: (len(value), value))
        for column in group:
            duplicate_member[column] = {"group": list(group), "keeper": keeper,
                                        "scenarios_identical": occurrences}
    inventory = []
    for column in all_columns:
        profiles = {scenario: reports[scenario]["column_profiles"].get(column)
                    for scenario in scenarios}
        present = {key: value for key, value in profiles.items() if value is not None}
        missing = sum(item["missing"] for item in present.values())
        rows = sum(reports[scenario]["counts"]["packets"] for scenario in present)
        nonmissing = rows - missing
        numeric = sum(item["numeric_values"] for item in present.values())
        role = _suggested_role(column)
        duplicate = duplicate_member.get(column)
        if role == "evaluation_metadata":
            action = "exclude_evaluation_metadata"
        elif role == "ordering_metadata":
            action = "exclude_absolute_time_feature"
        elif role == "record_identifier":
            action = "exclude_record_identifier"
        elif role == "topology_identity":
            action = "topology_only_exclude_model_feature"
        elif role == "content_or_protocol_fingerprint":
            action = "exclude_primary_review_for_secondary_ablation"
        elif len(present) != len(scenarios):
            action = "exclude_schema_inconsistent"
        elif (duplicate and duplicate["scenarios_identical"] == len(scenarios)
              and column != duplicate["keeper"]):
            action = "exclude_exact_duplicate"
        elif duplicate and column != duplicate["keeper"]:
            action = "review_scenario_specific_duplicate"
        elif all(item["constant_nonmissing"] for item in present.values()):
            action = "fit_fold_variance_filter"
        elif numeric == nonmissing:
            action = "candidate_numeric"
        else:
            action = "candidate_categorical_or_mixed"
        inventory.append({
            "column": column, "suggested_role": role, "suggested_action": action,
            "present_scenarios": sorted(present), "missing_scenarios": sorted(set(scenarios) - set(present)),
            "missing_values": missing, "nonmissing_values": nonmissing,
            "numeric_values": numeric, "numeric_when_present": numeric == nonmissing,
            "constant_nonmissing_in_each_present_scenario": all(
                item["constant_nonmissing"] for item in present.values()),
            "duplicate_group": duplicate,
        })
    return inventory


def run_gate0_decision_audit(*, manifest_path: Path, run_dir: Path,
                             memory_limit: str = "2GB", threads: int = 2,
                             overwrite: bool = False) -> dict:
    """Run post-audit diagnostics and persist one decision-audit JSON artifact."""
    run_dir = Path(run_dir)
    output_path = run_dir / "gate0_decision_audit.json"
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Preserve the existing decision audit: {output_path}")
    manifest, reports, packets = load_full_dev_artifacts(run_dir, manifest_path)
    benign = compare_benign_backgrounds(
        manifest, packets, memory_limit=memory_limit, threads=threads,
    )
    topology = [audit_topology(packets[scenario], scenario, memory_limit=memory_limit,
                               threads=threads) for scenario in selected_scenarios(manifest, "FULL_DEV")]
    inventory = build_feature_inventory(reports)
    blockers = []
    for item in benign:
        if not item["all_benign_packets_identical"]:
            blockers.append(f"benign_signature_mismatch:{item['benign_source']}")
    for item in topology:
        if item["invalid_mac_packets"]:
            blockers.append(f"invalid_mac_endpoint:{item['scenario']}")
        if item["conflicting_ip_version_packets"]:
            blockers.append(f"conflicting_ip_version:{item['scenario']}")
    action_counts: dict[str, int] = defaultdict(int)
    for item in inventory:
        action_counts[item["suggested_action"]] += 1
    result = {
        "report_version": 1, "status": "blocked" if blockers else "review_required",
        "blockers": blockers, "manifest_sha256": sha256_file(manifest_path),
        "full_dev_run": str(run_dir),
        "benign_background_comparison": benign,
        "topology": topology,
        "feature_inventory": inventory,
        "feature_action_counts": dict(sorted(action_counts.items())),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "duckdb": duckdb.__version__,
        },
        "module_sha256": sha256_file(Path(__file__)),
        "method_notes": {
            "benign_signature": "SHA-256 over ordered DuckDB 64-bit hashes of every common raw column",
            "feature_actions": "diagnostic suggestions requiring manual review; no schema is frozen",
            "endpoint_policy": "Ethernet MAC remains provisional until topology counts are reviewed",
        },
    }
    write_json(output_path, result)
    return result
