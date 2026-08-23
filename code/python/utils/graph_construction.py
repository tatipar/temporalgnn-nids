"""Auditable NF-v3 temporal graph construction utilities.

The builder assigns a completed flow to exactly one 30-second graph window: the
window containing the flow end. It intentionally has no dependency on labels
for feature construction, split selection, or IP-to-ID mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import ipaddress
import json
import os
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

from .graph_schema import (
    FeatureProfile, PORT_CATEGORY_COLUMNS, PROTOCOL_CATEGORY_COLUMNS,
    get_feature_profile, destination_port_one_hot, protocol_one_hot,
    validate_numeric_frame,
)


WINDOW_MS = 30_000
TIME_COLUMN = "FLOW_START_MILLISECONDS"
DURATION_COLUMN = "FLOW_DURATION_MILLISECONDS"
SOURCE_IP_COLUMN = "IPV4_SRC_ADDR"
DESTINATION_IP_COLUMN = "IPV4_DST_ADDR"
DESTINATION_PORT_COLUMN = "L4_DST_PORT"
PROTOCOL_COLUMN = "PROTOCOL"
TARGET_COLUMN = "binary_target"
SOURCE_FILE_COLUMN = "source_file"
SOURCE_ROW_ID_COLUMN = "source_row_id"


@dataclass(frozen=True)
class DaySpec:
    """A chronological stream and its split policy."""

    name: str
    source_file: str
    split_names: tuple[str, ...]
    train_ratio: float | None = None
    validation_ratio: float | None = None

    @property
    def is_day1(self) -> bool:
        return self.train_ratio is not None and self.validation_ratio is not None


DAY1 = DaySpec("day1", "cicids2018v3_wed2802.csv", ("train", "val", "test1"), 0.70, 0.15)
DAY2 = DaySpec("day2", "cicids2018v3_thu0103.csv", ("test2",))


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 hash without loading a file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json_dump(payload: object, path: Path) -> None:
    """Write JSON atomically so an interrupted Drive session cannot publish it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_torch_save(data: Data, path: Path) -> None:
    """Write one graph atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, temporary)
    os.replace(temporary, path)


class IpIdMap:
    """Append-only, day-scoped mapping for temporal-memory keys."""

    def __init__(self, ip_to_id: dict[str, int] | None = None) -> None:
        self.ip_to_id = {str(ip): int(node_id) for ip, node_id in (ip_to_id or {}).items()}
        self.id_to_ip = {node_id: ip for ip, node_id in self.ip_to_id.items()}
        if len(self.id_to_ip) != len(self.ip_to_id):
            raise ValueError("IP-to-ID mapping is not one-to-one.")

    def id_for(self, ip: str) -> int:
        if ip not in self.ip_to_id:
            node_id = len(self.ip_to_id)
            self.ip_to_id[ip] = node_id
            self.id_to_ip[node_id] = ip
        return self.ip_to_id[ip]

    def payload(self, day_name: str) -> dict[str, object]:
        return {
            "day": day_name,
            "creation_policy": "append_only_first_valid_chronological_appearance",
            "ip_normalization": "ipaddress.ip_address canonical IPv4 string; missing, invalid, IPv6, and 0.0.0.0 excluded",
            "entries": len(self.ip_to_id),
            "ip_to_id": dict(sorted(self.ip_to_id.items(), key=lambda item: item[1])),
            "id_to_ip": {str(node_id): ip for node_id, ip in sorted(self.id_to_ip.items())},
        }

    @classmethod
    def from_file(cls, path: Path) -> "IpIdMap":
        return cls(json.loads(path.read_text(encoding="utf-8"))["ip_to_id"])


def canonical_ipv4(value: object) -> str | None:
    """Return a canonical usable IPv4 address, otherwise ``None``."""
    if pd.isna(value):
        return None
    try:
        address = ipaddress.ip_address(str(value).strip())
    except ValueError:
        return None
    if address.version != 4 or str(address) == "0.0.0.0":
        return None
    return str(address)


def required_columns(profiles: Iterable[FeatureProfile]) -> list[str]:
    """Return the complete input schema required by a build request."""
    columns = {
        TIME_COLUMN, DURATION_COLUMN, SOURCE_IP_COLUMN, DESTINATION_IP_COLUMN,
        DESTINATION_PORT_COLUMN, PROTOCOL_COLUMN, TARGET_COLUMN,
        SOURCE_FILE_COLUMN, SOURCE_ROW_ID_COLUMN,
    }
    for profile in profiles:
        columns.update(profile.numeric_columns)
    return sorted(columns)


def prepare_chunk(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate rows and derive flow-end and decision-time columns."""
    counts = {"input_rows": len(frame), "invalid_endpoint_rows": 0, "invalid_time_or_duration_rows": 0}
    result = frame.copy()
    result["source_ip"] = result[SOURCE_IP_COLUMN].map(canonical_ipv4)
    result["destination_ip"] = result[DESTINATION_IP_COLUMN].map(canonical_ipv4)
    endpoint_ok = result["source_ip"].notna() & result["destination_ip"].notna()
    counts["invalid_endpoint_rows"] = int((~endpoint_ok).sum())
    result = result.loc[endpoint_ok].copy()

    start = pd.to_numeric(result[TIME_COLUMN], errors="coerce")
    duration = pd.to_numeric(result[DURATION_COLUMN], errors="coerce")
    time_ok = np.isfinite(start) & np.isfinite(duration) & (duration >= 0)
    counts["invalid_time_or_duration_rows"] = int((~time_ok).sum())
    result = result.loc[time_ok].copy()
    result["flow_start_ms"] = start.loc[time_ok].to_numpy(dtype=np.float64)
    result["flow_end_ms"] = result["flow_start_ms"] + duration.loc[time_ok].to_numpy(dtype=np.float64)
    result["window_start_ms"] = (np.floor(result["flow_end_ms"] / WINDOW_MS) * WINDOW_MS).astype(np.int64)
    result["decision_time_ms"] = result["window_start_ms"] + WINDOW_MS
    return result, counts


def split_cutoffs(input_csvs: Iterable[Path], day: DaySpec, chunksize: int, usecols: list[str]) -> dict[str, int | None]:
    """Compute Day-1 decision-time cutoffs after filtering invalid source rows."""
    if not day.is_day1:
        return {"train_end_ms": None, "val_end_ms": None}
    minimum: int | None = None
    maximum: int | None = None
    for path in input_csvs:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            chunk = chunk.loc[chunk[SOURCE_FILE_COLUMN].eq(day.source_file)]
            if chunk.empty:
                continue
            prepared, _ = prepare_chunk(chunk)
            if prepared.empty:
                continue
            values = prepared["decision_time_ms"]
            current_min, current_max = int(values.min()), int(values.max())
            minimum = current_min if minimum is None else min(minimum, current_min)
            maximum = current_max if maximum is None else max(maximum, current_max)
    if minimum is None or maximum is None or maximum <= minimum:
        raise ValueError(f"Could not derive valid Day-1 decision-time bounds for {day.source_file}.")
    duration = maximum - minimum
    raw_train_end = minimum + int(duration * float(day.train_ratio))
    raw_val_end = minimum + int(duration * float(day.train_ratio + day.validation_ratio))
    train_end = ((raw_train_end + WINDOW_MS - 1) // WINDOW_MS) * WINDOW_MS
    val_end = ((raw_val_end + WINDOW_MS - 1) // WINDOW_MS) * WINDOW_MS
    if not minimum < train_end < val_end <= maximum + WINDOW_MS:
        raise AssertionError("Window-aligned Day-1 split cutoffs are invalid.")
    return {
        "train_end_ms": train_end,
        "val_end_ms": val_end,
        "raw_train_end_ms": raw_train_end,
        "raw_val_end_ms": raw_val_end,
    }


def split_for_time(day: DaySpec, decision_time_ms: int, cutoffs: dict[str, int | None]) -> str:
    """Assign one graph window to its declared chronological split."""
    if not day.is_day1:
        return "test2"
    if decision_time_ms < int(cutoffs["train_end_ms"]):
        return "train"
    if decision_time_ms < int(cutoffs["val_end_ms"]):
        return "val"
    return "test1"


def fit_scalers(input_csvs: Iterable[Path], day: DaySpec, profiles: Iterable[FeatureProfile], cutoffs: dict[str, int | None], chunksize: int) -> dict[str, StandardScaler]:
    """Fit one scaler per profile using only Day-1 train flows."""
    if not day.is_day1:
        raise ValueError("Only Day 1 has a training split for scaler fitting.")
    profiles = tuple(profiles)
    usecols = required_columns(profiles)
    scalers = {profile.name: StandardScaler() for profile in profiles}
    fitted_rows = 0
    for path in input_csvs:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            chunk = chunk.loc[chunk[SOURCE_FILE_COLUMN].eq(day.source_file)]
            if chunk.empty:
                continue
            prepared, _ = prepare_chunk(chunk)
            train = prepared.loc[prepared["decision_time_ms"] < int(cutoffs["train_end_ms"])]
            if train.empty:
                continue
            for profile in profiles:
                scalers[profile.name].partial_fit(np.log1p(validate_numeric_frame(train, profile)))
            fitted_rows += len(train)
    if fitted_rows == 0:
        raise ValueError("No valid Day-1 training flows were available to fit scalers.")
    return scalers


def encode_edge_attributes(frame: pd.DataFrame, profile: FeatureProfile, scaler: StandardScaler) -> np.ndarray:
    """Return scaled numerical and fixed categorical edge features in schema order."""
    numerical = scaler.transform(np.log1p(validate_numeric_frame(frame, profile))).astype(np.float32)
    ports = destination_port_one_hot(frame[DESTINATION_PORT_COLUMN])
    protocols = protocol_one_hot(frame[PROTOCOL_COLUMN])
    edge_attr = np.concatenate((numerical, ports, protocols), axis=1)
    if edge_attr.shape[1] != profile.dimension:
        raise AssertionError(f"Unexpected {profile.name} feature dimension {edge_attr.shape[1]}.")
    if not np.isfinite(edge_attr).all():
        raise ValueError(f"{profile.name} produced NaN or infinite edge features.")
    return edge_attr


def _counter_payload(counter: Counter, limit: int | None = None) -> dict[str, int]:
    """Convert a counter to a stable JSON-ready representation."""
    items = sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))
    if limit is not None:
        items = items[:limit]
    return {str(key): int(value) for key, value in items}


def feature_preflight_audit(input_csvs: Iterable[Path], profiles: Iterable[FeatureProfile], chunksize: int) -> dict[str, object]:
    """Scan corrected inputs for feature validity and taxonomy coverage.

    This audit intentionally does not use labels. It is run before graph
    construction so data-quality failures are reported without partially
    producing graph files.
    """
    profiles = tuple(profiles)
    usecols = required_columns(profiles)
    summary: dict[str, object] = {
        "input_rows": 0,
        "invalid_source_endpoint_rows": 0,
        "invalid_destination_endpoint_rows": 0,
        "invalid_port_rows": 0,
        "invalid_protocol_rows": 0,
        "invalid_numeric_rows_by_profile": Counter(),
        "port_category_counts": Counter(),
        "protocol_category_counts": Counter(),
        "port_zero_by_protocol": Counter(),
        "other_privileged_top_ports": Counter(),
        "other_high_top_ports": Counter(),
        "valid_protocol_number_counts": Counter(),
    }

    for path in input_csvs:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            summary["input_rows"] += len(chunk)
            summary["invalid_source_endpoint_rows"] += int(chunk[SOURCE_IP_COLUMN].map(canonical_ipv4).isna().sum())
            summary["invalid_destination_endpoint_rows"] += int(chunk[DESTINATION_IP_COLUMN].map(canonical_ipv4).isna().sum())

            port = pd.to_numeric(chunk[DESTINATION_PORT_COLUMN], errors="coerce")
            port_valid = port.notna() & np.isfinite(port) & (np.floor(port) == port) & port.between(0, 65535)
            summary["invalid_port_rows"] += int((~port_valid).sum())
            if port_valid.any():
                encoded_ports = destination_port_one_hot(port.loc[port_valid])
                port_categories = encoded_ports.argmax(axis=1)
                valid_ports = port.loc[port_valid].astype(np.int64).to_numpy()
                summary["port_category_counts"].update(
                    PORT_CATEGORY_COLUMNS[index] for index in port_categories
                )
                summary["other_privileged_top_ports"].update(
                    int(value) for value, index in zip(valid_ports, port_categories) if index == 5
                )
                summary["other_high_top_ports"].update(
                    int(value) for value, index in zip(valid_ports, port_categories) if index == 6
                )

            protocol = pd.to_numeric(chunk[PROTOCOL_COLUMN], errors="coerce")
            protocol_valid = protocol.notna() & np.isfinite(protocol) & (np.floor(protocol) == protocol) & protocol.between(0, 255)
            summary["invalid_protocol_rows"] += int((~protocol_valid).sum())
            protocol_labels = pd.Series("invalid", index=chunk.index, dtype="object")
            if protocol_valid.any():
                valid_protocol = protocol.loc[protocol_valid].astype(np.int64)
                encoded_protocols = protocol_one_hot(valid_protocol)
                summary["protocol_category_counts"].update(
                    PROTOCOL_CATEGORY_COLUMNS[index] for index in encoded_protocols.argmax(axis=1)
                )
                summary["valid_protocol_number_counts"].update(valid_protocol.tolist())
                protocol_labels.loc[protocol_valid] = valid_protocol.astype(str)
            zero_ports = port_valid & port.eq(0)
            summary["port_zero_by_protocol"].update(protocol_labels.loc[zero_ports].tolist())

            for profile in profiles:
                numeric = chunk.loc[:, profile.numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
                valid_numeric_rows = np.isfinite(numeric).all(axis=1) & (numeric >= 0).all(axis=1)
                summary["invalid_numeric_rows_by_profile"][profile.name] += int((~valid_numeric_rows).sum())

    failed = (
        int(summary["invalid_port_rows"]) > 0
        or int(summary["invalid_protocol_rows"]) > 0
        or any(value > 0 for value in summary["invalid_numeric_rows_by_profile"].values())
    )
    return {
        "status": "failed" if failed else "passed",
        "input_rows": int(summary["input_rows"]),
        "invalid_source_endpoint_rows": int(summary["invalid_source_endpoint_rows"]),
        "invalid_destination_endpoint_rows": int(summary["invalid_destination_endpoint_rows"]),
        "invalid_port_rows": int(summary["invalid_port_rows"]),
        "invalid_protocol_rows": int(summary["invalid_protocol_rows"]),
        "invalid_numeric_rows_by_profile": _counter_payload(summary["invalid_numeric_rows_by_profile"]),
        "port_category_counts": _counter_payload(summary["port_category_counts"]),
        "protocol_category_counts": _counter_payload(summary["protocol_category_counts"]),
        "port_zero_by_protocol": _counter_payload(summary["port_zero_by_protocol"]),
        "other_privileged_top_ports": _counter_payload(summary["other_privileged_top_ports"], limit=50),
        "other_high_top_ports": _counter_payload(summary["other_high_top_ports"], limit=50),
        "valid_protocol_number_counts": _counter_payload(summary["valid_protocol_number_counts"]),
    }


def build_graph(frame: pd.DataFrame, profile: FeatureProfile, scaler: StandardScaler, ip_map: IpIdMap) -> tuple[Data, pd.DataFrame]:
    """Build one graph and matching per-edge provenance from one complete window."""
    global_source = [ip_map.id_for(ip) for ip in frame["source_ip"]]
    global_destination = [ip_map.id_for(ip) for ip in frame["destination_ip"]]
    global_nodes = sorted(set(global_source) | set(global_destination))
    local = {global_id: index for index, global_id in enumerate(global_nodes)}
    edge_index = torch.tensor(
        [[local[node] for node in global_source], [local[node] for node in global_destination]], dtype=torch.long
    )
    targets = pd.to_numeric(frame[TARGET_COLUMN], errors="raise").to_numpy(dtype=np.float32)
    if not np.isin(targets, (0.0, 1.0)).all():
        raise ValueError("binary_target must contain only 0 or 1.")
    decision_time = int(frame["decision_time_ms"].iloc[0])
    window_start = int(frame["window_start_ms"].iloc[0])
    data = Data(
        edge_index=edge_index,
        edge_attr=torch.from_numpy(encode_edge_attributes(frame, profile, scaler)),
        y=torch.from_numpy(targets),
        num_nodes=len(global_nodes),
    )
    data.global_node_ids = torch.tensor(global_nodes, dtype=torch.long)
    data.timestamp = decision_time
    data.window_start = window_start
    data.window_end = decision_time
    data.feature_profile = profile.name
    data.schema_hash = profile.sha256()

    provenance = pd.DataFrame({
        "flow_id": frame[SOURCE_FILE_COLUMN].astype(str) + ":" + frame[SOURCE_ROW_ID_COLUMN].astype(str),
        "source_file": frame[SOURCE_FILE_COLUMN].astype(str),
        "source_row_id": frame[SOURCE_ROW_ID_COLUMN],
        "source_ip": frame["source_ip"],
        "destination_ip": frame["destination_ip"],
        "source_global_id": global_source,
        "destination_global_id": global_destination,
        "edge_position": np.arange(len(frame), dtype=np.int64),
        "flow_start_ms": frame["flow_start_ms"],
        "flow_end_ms": frame["flow_end_ms"],
        "decision_time_ms": decision_time,
        "window_start_ms": window_start,
        "window_end_ms": decision_time,
        "binary_target": targets.astype(np.int8),
    })
    return data, provenance


def audit_graph_file(graph_path: Path, profile: FeatureProfile, mapping: IpIdMap, provenance_path: Path) -> dict[str, int]:
    """Audit one serialized graph against its schema, mapping, and provenance."""
    data = torch.load(graph_path, weights_only=False)
    edges = int(data.edge_index.shape[1])
    if data.edge_attr.shape != (edges, profile.dimension) or data.y.shape[0] != edges:
        raise AssertionError(f"Edge dimensions disagree in {graph_path}.")
    if data.global_node_ids.numel() != data.num_nodes:
        raise AssertionError(f"Node mapping dimensions disagree in {graph_path}.")
    if not torch.isfinite(data.edge_attr).all() or not torch.isfinite(data.y).all():
        raise AssertionError(f"Non-finite graph tensor found in {graph_path}.")
    if data.feature_profile != profile.name or data.schema_hash != profile.sha256():
        raise AssertionError(f"Profile schema metadata disagrees in {graph_path}.")
    if not all(int(node_id) in mapping.id_to_ip for node_id in data.global_node_ids.tolist()):
        raise AssertionError(f"A graph ID cannot be decoded by its declared map in {graph_path}.")
    provenance = pd.read_csv(provenance_path)
    if len(provenance) != edges or not provenance["flow_id"].is_unique:
        raise AssertionError(f"Provenance does not match graph edges in {provenance_path}.")
    if not np.all(provenance["flow_end_ms"] < provenance["decision_time_ms"]):
        raise AssertionError(f"Flow end is not strictly before decision time in {provenance_path}.")
    if not np.all(provenance["window_end_ms"] == provenance["decision_time_ms"]):
        raise AssertionError(f"Window close disagrees with decision time in {provenance_path}.")
    sample = provenance.head(min(10, len(provenance)))
    for row in sample.itertuples(index=False):
        if mapping.id_to_ip[int(row.source_global_id)] != row.source_ip or mapping.id_to_ip[int(row.destination_global_id)] != row.destination_ip:
            raise AssertionError(f"Sampled edge endpoint decoding failed in {provenance_path}.")
    return {"graphs": 1, "edges": edges, "positive_edges": int(data.y.sum().item())}
