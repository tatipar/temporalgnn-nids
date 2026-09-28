"""Recover and audit cAPTure graph-node identities from immutable provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .capture_data import sha256_file
from .capture_feature_profile import load_prepared_full_dev

if TYPE_CHECKING:
    from .capture_graph_dataset import CaptureGraphCollection


REPORT_VERSION = 1
CANONICAL_FOLD = "A"
ENDPOINT_COLUMNS = ("src_endpoint", "dst_endpoint")


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _graph_endpoint_ids(dataset) -> tuple[np.ndarray, np.ndarray]:
    edge_count = int(dataset.report["edges"])
    source_ids = np.empty(edge_count, dtype=np.int32)
    destination_ids = np.empty(edge_count, dtype=np.int32)
    cursor = 0
    for record in dataset.shards:
        path = dataset.directory / record["path"]
        with np.load(path, allow_pickle=False) as shard:
            edge_index = shard["edge_index"]
            source_rows = shard["source_row_id"]
            global_node_ids = shard["global_node_ids"]
            edge_ptr = shard["edge_ptr"]
            node_ptr = shard["node_ptr"]
        expected = np.arange(cursor, cursor + len(source_rows), dtype=np.int64)
        if not np.array_equal(source_rows, expected):
            raise ValueError(
                f"Identity recovery found non-contiguous source rows: "
                f"{dataset.fold}/{dataset.scenario}"
            )
        for graph_index in range(len(edge_ptr) - 1):
            edge_start = int(edge_ptr[graph_index])
            edge_stop = int(edge_ptr[graph_index + 1])
            node_start = int(node_ptr[graph_index])
            node_stop = int(node_ptr[graph_index + 1])
            graph_edges = edge_index[:, edge_start:edge_stop]
            graph_nodes = global_node_ids[node_start:node_stop]
            output_start = cursor + edge_start
            output_stop = cursor + edge_stop
            source_ids[output_start:output_stop] = graph_nodes[graph_edges[0]]
            destination_ids[output_start:output_stop] = graph_nodes[graph_edges[1]]
        cursor += len(source_rows)
    if cursor != edge_count:
        raise ValueError(f"Identity recovery edge count changed: {dataset.scenario}")
    return source_ids, destination_ids


def _register_batch(
    *,
    row_ids: np.ndarray,
    source_ids: np.ndarray,
    destination_ids: np.ndarray,
    sources: np.ndarray,
    destinations: np.ndarray,
    id_to_endpoint: dict[int, str],
    endpoint_to_id: dict[str, int],
    first_seen: dict[int, tuple[int, str]],
) -> None:
    count = len(row_ids)
    global_ids = np.column_stack((source_ids, destination_ids)).reshape(count * 2)
    endpoints = np.column_stack((sources, destinations)).reshape(count * 2)
    repeated_rows = np.repeat(row_ids, 2)
    roles = np.tile(np.asarray(["source", "destination"], dtype=object), count)
    pairs = pd.DataFrame(
        {
            "global_node_id": global_ids,
            "mac_address": endpoints,
            "source_row_id": repeated_rows,
            "first_seen_role": roles,
        }
    ).drop_duplicates(["global_node_id", "mac_address"], keep="first")

    for row in pairs.itertuples(index=False):
        global_id = int(row.global_node_id)
        endpoint = str(row.mac_address)
        previous_endpoint = id_to_endpoint.get(global_id)
        if previous_endpoint is not None and previous_endpoint != endpoint:
            raise ValueError(
                f"Global node ID {global_id} maps to both {previous_endpoint} and "
                f"{endpoint}."
            )
        previous_id = endpoint_to_id.get(endpoint)
        if previous_id is not None and previous_id != global_id:
            raise ValueError(
                f"Endpoint {endpoint} maps to both {previous_id} and {global_id}."
            )
        id_to_endpoint[global_id] = endpoint
        endpoint_to_id[endpoint] = global_id
        if global_id not in first_seen:
            first_seen[global_id] = (int(row.source_row_id), str(row.first_seen_role))


def _recover_scenario(
    collection: "CaptureGraphCollection",
    scenario: str,
    prepared_path: Path,
    prepared_report: dict,
    *,
    batch_size: int,
) -> tuple[dict, pd.DataFrame]:
    dataset = collection.scenario_dataset(
        CANONICAL_FOLD,
        scenario,
        verify_shard_checksums=False,
    )
    started = time.perf_counter()
    source_ids, destination_ids = _graph_endpoint_ids(dataset)
    expected_edges = int(dataset.report["edges"])
    if expected_edges != int(prepared_report["counts"]["packets"]):
        raise ValueError(f"Prepared and graph edge counts differ: {scenario}")

    id_to_endpoint: dict[int, str] = {}
    endpoint_to_id: dict[str, int] = {}
    first_seen: dict[int, tuple[int, str]] = {}
    cursor = 0
    parquet = pq.ParquetFile(prepared_path)
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=["source_row_id", *ENDPOINT_COLUMNS],
    ):
        frame = batch.to_pandas()
        count = len(frame)
        row_ids = frame["source_row_id"].to_numpy(dtype=np.int64)
        expected_rows = np.arange(cursor, cursor + count, dtype=np.int64)
        if not np.array_equal(row_ids, expected_rows):
            raise ValueError(f"Prepared source rows changed: {scenario}")
        if frame[list(ENDPOINT_COLUMNS)].isna().any().any():
            raise ValueError(f"Prepared endpoints contain null values: {scenario}")
        stop = cursor + count
        _register_batch(
            row_ids=row_ids,
            source_ids=source_ids[cursor:stop],
            destination_ids=destination_ids[cursor:stop],
            sources=frame["src_endpoint"].astype(str).to_numpy(),
            destinations=frame["dst_endpoint"].astype(str).to_numpy(),
            id_to_endpoint=id_to_endpoint,
            endpoint_to_id=endpoint_to_id,
            first_seen=first_seen,
        )
        cursor = stop
    if cursor != expected_edges:
        raise ValueError(f"Prepared packet count changed during identity recovery: {scenario}")

    expected_nodes = int(dataset.report["distinct_global_nodes"])
    expected_ids = set(range(expected_nodes))
    if set(id_to_endpoint) != expected_ids or len(endpoint_to_id) != expected_nodes:
        raise ValueError(f"Not every graph node resolved to one endpoint: {scenario}")

    lookup_rows = [
        {
            "scenario": scenario,
            "global_node_id": global_id,
            "mac_address": id_to_endpoint[global_id],
            "first_seen_source_row_id": first_seen[global_id][0],
            "first_seen_role": first_seen[global_id][1],
        }
        for global_id in range(expected_nodes)
    ]
    lookup = pd.DataFrame(lookup_rows)
    contract_rows = [
        {
            "global_node_id": int(row["global_node_id"]),
            "endpoint": str(row["mac_address"]),
            "first_seen_source_row_id": int(row["first_seen_source_row_id"]),
        }
        for row in lookup_rows
    ]
    recovered_hash = _canonical_hash(contract_rows)

    fold_hashes = {}
    for fold in ("A", "B"):
        fold_dataset = collection.scenario_dataset(
            fold,
            scenario,
            verify_shard_checksums=False,
        )
        report = fold_dataset.report
        if report.get("source_artifact_sha256") != prepared_report["output_sha256"]:
            raise ValueError(f"Prepared-artifact binding changed: {fold}/{scenario}")
        fold_hash = report.get("node_mapping_contract_sha256")
        if fold_hash != recovered_hash:
            raise ValueError(
                f"Recovered node mapping differs from fold {fold}: {scenario}"
            )
        mapping_path = fold_dataset.directory / "node_mapping.parquet"
        stored = pd.read_parquet(mapping_path).sort_values("global_node_id")
        expected_mapping = lookup.loc[
            :, ["global_node_id", "first_seen_source_row_id"]
        ]
        if (
            list(stored.columns) != list(expected_mapping.columns)
            or not stored.reset_index(drop=True).equals(expected_mapping)
        ):
            raise ValueError(f"Stored node mapping differs from recovery: {fold}/{scenario}")
        fold_hashes[fold] = fold_hash

    result = {
        "status": "passed",
        "prepared_packet_sha256": prepared_report["output_sha256"],
        "canonical_graph_fold": CANONICAL_FOLD,
        "edges_joined": expected_edges,
        "global_nodes_resolved": expected_nodes,
        "unique_mac_addresses": len(endpoint_to_id),
        "id_to_mac_conflicts": 0,
        "mac_to_id_conflicts": 0,
        "unresolved_global_nodes": 0,
        "source_rows_contiguous": True,
        "first_seen_rows_match_stored_mapping": True,
        "reconstructed_mapping_contract_sha256": recovered_hash,
        "fold_mapping_contract_sha256": fold_hashes,
        "reconstructed_contract_matches_both_folds": True,
        "wall_seconds": float(time.perf_counter() - started),
    }
    return result, lookup


def audit_capture_graph_identities(
    collection: "CaptureGraphCollection",
    *,
    prepared_run_dir: str | Path,
    local_work_root: str | Path,
    batch_size: int = 250_000,
) -> dict:
    """Recover scenario node identities and prove their exact graph correspondence."""
    if batch_size <= 0:
        raise ValueError("Identity-audit batch size must be positive.")
    if not collection.artifact_checksums_verified:
        raise ValueError("Graph checksums must be verified before identity recovery.")
    contract = collection.contract.get("identity_audit")
    if not contract or contract.get("canonical_graph_fold") != CANONICAL_FOLD:
        raise ValueError("The identity-audit contract is missing or unsupported.")

    prepared_run_dir = Path(prepared_run_dir).expanduser().resolve()
    local_work_root = Path(local_work_root).expanduser().resolve()
    local_work_root.mkdir(parents=True, exist_ok=True)
    manifest_path = collection.root / "capture_experiment_v1.yaml"
    packet_schema_path = collection.root / "capture_packet_schema_v1.yaml"
    _, _, prepared_reports, prepared_paths = load_prepared_full_dev(
        prepared_run_dir,
        manifest_path,
        packet_schema_path,
    )
    provenance = collection.manifest["provenance"]
    if (
        sha256_file(prepared_run_dir / "run_config.json")
        != provenance.get("prepared_run_config_sha256")
    ):
        raise ValueError("The prepared run differs from graph-materialization provenance.")
    expected_packet_hashes = provenance.get("prepared_packet_sha256")
    observed_packet_hashes = {
        scenario: prepared_reports[scenario]["output_sha256"]
        for scenario in collection.scenarios
    }
    if expected_packet_hashes != observed_packet_hashes:
        raise ValueError("Prepared packet hashes differ from materialization provenance.")

    started = time.perf_counter()
    scenario_reports = {}
    lookups = {}
    for scenario in collection.scenarios:
        source_path = prepared_paths[scenario]
        with tempfile.TemporaryDirectory(
            dir=local_work_root,
            prefix=f"{scenario}_identity_",
        ) as temporary:
            local_path = Path(temporary) / source_path.name
            required_bytes = source_path.stat().st_size + 512 * 1024**2
            if shutil.disk_usage(local_work_root).free < required_bytes:
                raise OSError("Insufficient local space for prepared identity-audit staging.")
            shutil.copyfile(source_path, local_path)
            if sha256_file(local_path) != prepared_reports[scenario]["output_sha256"]:
                raise IOError(f"Staged prepared checksum mismatch: {scenario}")
            scenario_report, lookup = _recover_scenario(
                collection,
                scenario,
                local_path,
                prepared_reports[scenario],
                batch_size=batch_size,
            )
        scenario_reports[scenario] = scenario_report
        lookups[scenario] = lookup
        print(
            f"Recovered {scenario}: {len(lookup):,} node identities from "
            f"{scenario_report['edges_joined']:,} edges",
            flush=True,
        )

    return {
        "report": {
            "report_version": REPORT_VERSION,
            "status": "passed",
            "method": "per_edge_source_row_join",
            "prepared_run_id": prepared_run_dir.name,
            "prepared_run_config_sha256": sha256_file(
                prepared_run_dir / "run_config.json"
            ),
            "identity_audit_code_sha256": sha256_file(Path(__file__)),
            "canonical_graph_fold": CANONICAL_FOLD,
            "raw_identity_lookup_is_model_input": False,
            "cross_fold_mapping_contracts_match": True,
            "scenarios": scenario_reports,
            "held_out_scenarios_accessed": False,
            "training_performed": False,
            "wall_seconds": float(time.perf_counter() - started),
        },
        "lookups": lookups,
    }
