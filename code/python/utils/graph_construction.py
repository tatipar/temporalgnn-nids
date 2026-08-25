"""Auditable NF-v3 temporal graph construction utilities.

The builder assigns a completed flow to exactly one 30-second graph window: the
window containing the flow end. It intentionally has no dependency on labels
for feature construction, split selection, or IP-to-ID mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
import hashlib
import io
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
    tcp_flags_multi_hot, validate_numeric_frame,
)


WINDOW_MS = 30_000
TIME_COLUMN = "FLOW_START_MILLISECONDS"
END_TIME_COLUMN = "FLOW_END_MILLISECONDS"
DURATION_COLUMN = "FLOW_DURATION_MILLISECONDS"
SOURCE_IP_COLUMN = "IPV4_SRC_ADDR"
DESTINATION_IP_COLUMN = "IPV4_DST_ADDR"
DESTINATION_PORT_COLUMN = "L4_DST_PORT"
PROTOCOL_COLUMN = "PROTOCOL"
TCP_FLAGS_COLUMN = "TCP_FLAGS"
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
            "ip_normalization": "ipaddress.ip_address canonical IPv4 or IPv6 string; missing, non-parseable, and unspecified addresses (0.0.0.0 or ::) excluded",
            "entries": len(self.ip_to_id),
            "ip_to_id": dict(sorted(self.ip_to_id.items(), key=lambda item: item[1])),
            "id_to_ip": {str(node_id): ip for node_id, ip in sorted(self.id_to_ip.items())},
        }

    @classmethod
    def from_file(cls, path: Path) -> "IpIdMap":
        return cls(json.loads(path.read_text(encoding="utf-8"))["ip_to_id"])


def canonical_ip(value: object) -> str | None:
    """Return a canonical usable IPv4 or IPv6 address, otherwise ``None``."""
    if pd.isna(value):
        return None
    try:
        address = ipaddress.ip_address(str(value).strip())
    except ValueError:
        return None
    if address.is_unspecified:
        return None
    return str(address)


def endpoint_invalid_reason(value: object) -> str | None:
    """Return a stable exclusion reason for an unusable endpoint value."""
    if pd.isna(value) or not str(value).strip():
        return "missing"
    try:
        address = ipaddress.ip_address(str(value).strip())
    except ValueError:
        return "non_parseable"
    if address.is_unspecified:
        if address.version == 4:
            return "zero_ipv4"
        return "unspecified_ipv6"
    return None


def required_columns(profiles: Iterable[FeatureProfile]) -> list[str]:
    """Return the complete input schema required by a build request."""
    columns = {
        TIME_COLUMN, END_TIME_COLUMN, DURATION_COLUMN, SOURCE_IP_COLUMN, DESTINATION_IP_COLUMN,
        DESTINATION_PORT_COLUMN, PROTOCOL_COLUMN, TARGET_COLUMN,
        SOURCE_FILE_COLUMN, SOURCE_ROW_ID_COLUMN,
    }
    for profile in profiles:
        columns.update(profile.numeric_columns)
        if profile.tcp_flag_columns:
            columns.add(TCP_FLAGS_COLUMN)
    return sorted(columns)


def prepare_chunk(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Validate rows and derive flow-end and decision-time columns."""
    counts = {"input_rows": len(frame), "invalid_endpoint_rows": 0, "invalid_time_or_duration_rows": 0}
    result = frame.copy()
    result["source_ip"] = result[SOURCE_IP_COLUMN].map(canonical_ip)
    result["destination_ip"] = result[DESTINATION_IP_COLUMN].map(canonical_ip)
    endpoint_ok = result["source_ip"].notna() & result["destination_ip"].notna()
    counts["invalid_endpoint_rows"] = int((~endpoint_ok).sum())
    result = result.loc[endpoint_ok].copy()

    start = pd.to_numeric(result[TIME_COLUMN], errors="coerce")
    end = pd.to_numeric(result[END_TIME_COLUMN], errors="coerce")
    duration = pd.to_numeric(result[DURATION_COLUMN], errors="coerce")
    time_ok = (
        np.isfinite(start)
        & np.isfinite(end)
        & np.isfinite(duration)
        & (duration >= 0)
        & (end >= start)
    )
    counts["invalid_time_or_duration_rows"] = int((~time_ok).sum())
    result = result.loc[time_ok].copy()
    result["flow_start_ms"] = start.loc[time_ok].to_numpy(dtype=np.float64)
    result["flow_end_ms"] = end.loc[time_ok].to_numpy(dtype=np.float64)
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
    components = [numerical, ports, protocols]
    if profile.tcp_flag_columns:
        components.append(tcp_flags_multi_hot(frame[TCP_FLAGS_COLUMN]))
    edge_attr = np.concatenate(components, axis=1)
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

    Labels are used only to prove row/class conservation and never to choose a
    feature, split, threshold, or model. The audit runs before graph
    construction so data-quality failures are reported without partially
    producing graph files.
    """
    profiles = tuple(profiles)
    usecols = required_columns(profiles)
    summary: dict[str, object] = {
        "input_rows": 0,
        "positive_rows": 0,
        "retained_rows": 0,
        "retained_positive_rows": 0,
        "excluded_rows": 0,
        "excluded_positive_rows": 0,
        "invalid_any_endpoint_rows": 0,
        "invalid_any_endpoint_positive_rows": 0,
        "invalid_source_endpoint_rows": 0,
        "invalid_destination_endpoint_rows": 0,
        "invalid_source_endpoint_reasons": Counter(),
        "invalid_destination_endpoint_reasons": Counter(),
        "invalid_port_rows": 0,
        "invalid_protocol_rows": 0,
        "invalid_tcp_flags_rows": 0,
        "non_tcp_nonzero_tcp_flags_rows": 0,
        "invalid_binary_target_rows": 0,
        "invalid_time_or_duration_rows": 0,
        "invalid_time_or_duration_positive_rows": 0,
        "flow_end_difference_gt_1ms_rows": 0,
        "flow_end_max_absolute_difference_ms": 0.0,
        "invalid_numeric_rows_by_profile": Counter(),
        "port_category_counts": Counter(),
        "protocol_category_counts": Counter(),
        "port_zero_by_protocol": Counter(),
        "other_privileged_top_ports": Counter(),
        "other_high_top_ports": Counter(),
        "valid_protocol_number_counts": Counter(),
        "by_source_file": {},
    }

    for path in input_csvs:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            summary["input_rows"] += len(chunk)
            source_reasons = chunk[SOURCE_IP_COLUMN].map(endpoint_invalid_reason)
            destination_reasons = chunk[DESTINATION_IP_COLUMN].map(endpoint_invalid_reason)
            source_invalid = source_reasons.notna()
            destination_invalid = destination_reasons.notna()
            endpoint_invalid = source_invalid | destination_invalid

            target = pd.to_numeric(chunk[TARGET_COLUMN], errors="coerce")
            target_valid = target.notna() & np.isfinite(target) & target.isin((0, 1))
            positive = target_valid & target.eq(1)

            start = pd.to_numeric(chunk[TIME_COLUMN], errors="coerce")
            end = pd.to_numeric(chunk[END_TIME_COLUMN], errors="coerce")
            duration = pd.to_numeric(chunk[DURATION_COLUMN], errors="coerce")
            time_valid = (
                start.notna() & np.isfinite(start)
                & end.notna() & np.isfinite(end)
                & duration.notna() & np.isfinite(duration)
                & duration.ge(0) & end.ge(start)
            )
            invalid_time_after_endpoint = ~endpoint_invalid & ~time_valid
            retained = ~endpoint_invalid & time_valid & target_valid
            excluded = ~retained

            summary["positive_rows"] += int(positive.sum())
            summary["retained_rows"] += int(retained.sum())
            summary["retained_positive_rows"] += int((retained & positive).sum())
            summary["excluded_rows"] += int(excluded.sum())
            summary["excluded_positive_rows"] += int((excluded & positive).sum())
            summary["invalid_source_endpoint_rows"] += int(source_invalid.sum())
            summary["invalid_destination_endpoint_rows"] += int(destination_invalid.sum())
            summary["invalid_any_endpoint_rows"] += int(endpoint_invalid.sum())
            summary["invalid_any_endpoint_positive_rows"] += int((endpoint_invalid & positive).sum())
            summary["invalid_source_endpoint_reasons"].update(source_reasons.loc[source_invalid].tolist())
            summary["invalid_destination_endpoint_reasons"].update(destination_reasons.loc[destination_invalid].tolist())
            summary["invalid_binary_target_rows"] += int((~target_valid).sum())
            summary["invalid_time_or_duration_rows"] += int(invalid_time_after_endpoint.sum())
            summary["invalid_time_or_duration_positive_rows"] += int((invalid_time_after_endpoint & positive).sum())

            valid_time = time_valid
            if valid_time.any():
                end_difference = end.loc[valid_time] - (start.loc[valid_time] + duration.loc[valid_time])
                absolute_difference = end_difference.abs()
                summary["flow_end_difference_gt_1ms_rows"] += int(absolute_difference.gt(1).sum())
                summary["flow_end_max_absolute_difference_ms"] = max(
                    float(summary["flow_end_max_absolute_difference_ms"]),
                    float(absolute_difference.max()),
                )

            for source_file, indices in chunk.groupby(SOURCE_FILE_COLUMN, sort=False).groups.items():
                file_summary = summary["by_source_file"].setdefault(str(source_file), Counter())
                file_summary["input_rows"] += len(indices)
                file_summary["positive_rows"] += int(positive.loc[indices].sum())
                file_summary["retained_rows"] += int(retained.loc[indices].sum())
                file_summary["retained_positive_rows"] += int((retained & positive).loc[indices].sum())
                file_summary["excluded_rows"] += int(excluded.loc[indices].sum())
                file_summary["excluded_positive_rows"] += int((excluded & positive).loc[indices].sum())
                file_summary["invalid_endpoint_rows"] += int(endpoint_invalid.loc[indices].sum())
                file_summary["invalid_endpoint_positive_rows"] += int((endpoint_invalid & positive).loc[indices].sum())
                file_summary["invalid_time_or_duration_rows"] += int(invalid_time_after_endpoint.loc[indices].sum())
                file_summary["invalid_time_or_duration_positive_rows"] += int((invalid_time_after_endpoint & positive).loc[indices].sum())
                file_summary["invalid_binary_target_rows"] += int((~target_valid).loc[indices].sum())

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

            if any(profile.tcp_flag_columns for profile in profiles):
                tcp_flags = pd.to_numeric(chunk[TCP_FLAGS_COLUMN], errors="coerce")
                tcp_flags_valid = (
                    tcp_flags.notna() & np.isfinite(tcp_flags)
                    & (np.floor(tcp_flags) == tcp_flags) & tcp_flags.between(0, 255)
                )
                summary["invalid_tcp_flags_rows"] += int((~tcp_flags_valid).sum())
                summary["non_tcp_nonzero_tcp_flags_rows"] += int(
                    (tcp_flags_valid & tcp_flags.ne(0) & ~protocol.eq(6)).sum()
                )

            for profile in profiles:
                numeric = chunk.loc[:, profile.numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
                valid_numeric_rows = np.isfinite(numeric).all(axis=1) & (numeric >= 0).all(axis=1)
                summary["invalid_numeric_rows_by_profile"][profile.name] += int((~valid_numeric_rows).sum())

    failed = (
        int(summary["invalid_port_rows"]) > 0
        or int(summary["invalid_protocol_rows"]) > 0
        or int(summary["invalid_tcp_flags_rows"]) > 0
        or int(summary["non_tcp_nonzero_tcp_flags_rows"]) > 0
        or int(summary["invalid_binary_target_rows"]) > 0
        or int(summary["invalid_time_or_duration_rows"]) > 0
        or int(summary["flow_end_difference_gt_1ms_rows"]) > 0
        or any(value > 0 for value in summary["invalid_numeric_rows_by_profile"].values())
    )
    return {
        "status": "failed" if failed else "passed",
        "input_rows": int(summary["input_rows"]),
        "positive_rows": int(summary["positive_rows"]),
        "retained_rows": int(summary["retained_rows"]),
        "retained_positive_rows": int(summary["retained_positive_rows"]),
        "excluded_rows": int(summary["excluded_rows"]),
        "excluded_positive_rows": int(summary["excluded_positive_rows"]),
        "invalid_any_endpoint_rows": int(summary["invalid_any_endpoint_rows"]),
        "invalid_any_endpoint_positive_rows": int(summary["invalid_any_endpoint_positive_rows"]),
        "invalid_source_endpoint_rows": int(summary["invalid_source_endpoint_rows"]),
        "invalid_destination_endpoint_rows": int(summary["invalid_destination_endpoint_rows"]),
        "invalid_source_endpoint_reasons": _counter_payload(summary["invalid_source_endpoint_reasons"]),
        "invalid_destination_endpoint_reasons": _counter_payload(summary["invalid_destination_endpoint_reasons"]),
        "invalid_port_rows": int(summary["invalid_port_rows"]),
        "invalid_protocol_rows": int(summary["invalid_protocol_rows"]),
        "invalid_tcp_flags_rows": int(summary["invalid_tcp_flags_rows"]),
        "non_tcp_nonzero_tcp_flags_rows": int(summary["non_tcp_nonzero_tcp_flags_rows"]),
        "invalid_binary_target_rows": int(summary["invalid_binary_target_rows"]),
        "invalid_time_or_duration_rows": int(summary["invalid_time_or_duration_rows"]),
        "invalid_time_or_duration_positive_rows": int(summary["invalid_time_or_duration_positive_rows"]),
        "flow_end_difference_gt_1ms_rows": int(summary["flow_end_difference_gt_1ms_rows"]),
        "flow_end_max_absolute_difference_ms": float(summary["flow_end_max_absolute_difference_ms"]),
        "invalid_numeric_rows_by_profile": _counter_payload(summary["invalid_numeric_rows_by_profile"]),
        "port_category_counts": _counter_payload(summary["port_category_counts"]),
        "protocol_category_counts": _counter_payload(summary["protocol_category_counts"]),
        "port_zero_by_protocol": _counter_payload(summary["port_zero_by_protocol"]),
        "other_privileged_top_ports": _counter_payload(summary["other_privileged_top_ports"], limit=50),
        "other_high_top_ports": _counter_payload(summary["other_high_top_ports"], limit=50),
        "valid_protocol_number_counts": _counter_payload(summary["valid_protocol_number_counts"]),
        "by_source_file": {
            source_file: {key: int(value) for key, value in sorted(counts.items())}
            for source_file, counts in sorted(summary["by_source_file"].items())
        },
    }


def build_graph(
    frame: pd.DataFrame,
    profile: FeatureProfile,
    scaler: StandardScaler,
    ip_map: IpIdMap,
    split: str,
) -> tuple[Data, pd.DataFrame]:
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
        "window_wait_ms": decision_time - frame["flow_end_ms"],
        "split": split,
        "binary_target": targets.astype(np.int8),
    })
    return data, provenance


def audit_provenance_file(
    provenance_path: Path,
    mapping: IpIdMap,
    expected_split: str,
    expected_timestamp: int,
) -> tuple[pd.DataFrame, list[str], dict[str, int | str]]:
    """Read and audit one profile-independent per-window provenance table."""
    serialized = provenance_path.read_bytes()
    provenance = pd.read_csv(io.BytesIO(serialized))
    required_provenance = {
        "flow_id", "source_file", "source_row_id", "source_ip", "destination_ip",
        "source_global_id", "destination_global_id",
        "edge_position", "flow_start_ms", "flow_end_ms", "decision_time_ms",
        "window_start_ms", "window_end_ms", "window_wait_ms", "split", "binary_target",
    }
    if not required_provenance.issubset(provenance.columns):
        raise AssertionError(f"Required provenance columns are missing in {provenance_path}.")
    if not provenance["flow_id"].is_unique:
        raise AssertionError(f"Flow IDs are not unique in {provenance_path}.")
    if not np.array_equal(provenance["edge_position"].to_numpy(), np.arange(len(provenance))):
        raise AssertionError(f"Provenance edge positions are invalid in {provenance_path}.")
    if not provenance["split"].eq(expected_split).all():
        raise AssertionError(f"Provenance split disagrees in {provenance_path}.")
    if not provenance["decision_time_ms"].eq(expected_timestamp).all():
        raise AssertionError(f"Provenance decision time disagrees in {provenance_path}.")
    expected_flow_ids = (
        provenance["source_file"].astype(str)
        + ":"
        + provenance["source_row_id"].astype(str)
    )
    if not provenance["flow_id"].astype(str).equals(expected_flow_ids):
        raise AssertionError(f"Flow IDs disagree with their stable provenance keys in {provenance_path}.")
    contained = (
        provenance["window_start_ms"].le(provenance["flow_end_ms"])
        & provenance["flow_end_ms"].lt(provenance["window_end_ms"])
    )
    if not contained.all():
        raise AssertionError(f"Flow end is outside its half-open window in {provenance_path}.")
    if not np.all(provenance["window_end_ms"] == provenance["decision_time_ms"]):
        raise AssertionError(f"Window close disagrees with decision time in {provenance_path}.")
    expected_wait = provenance["decision_time_ms"] - provenance["flow_end_ms"]
    if not np.allclose(provenance["window_wait_ms"], expected_wait, rtol=0, atol=1e-6):
        raise AssertionError(f"Window wait disagrees in {provenance_path}.")
    sample = provenance.head(min(10, len(provenance)))
    for row in sample.itertuples(index=False):
        if mapping.id_to_ip[int(row.source_global_id)] != row.source_ip or mapping.id_to_ip[int(row.destination_global_id)] != row.destination_ip:
            raise AssertionError(f"Sampled edge endpoint decoding failed in {provenance_path}.")
    return (
        provenance,
        provenance["flow_id"].astype(str).tolist(),
        {"sha256": hashlib.sha256(serialized).hexdigest(), "bytes": len(serialized)},
    )


def audit_graph_file(
    graph_path: Path,
    profile: FeatureProfile,
    mapping: IpIdMap,
    provenance_path: Path,
    expected_split: str,
    provenance: pd.DataFrame | None = None,
) -> tuple[dict[str, int], list[str], dict[str, int | str]]:
    """Audit one serialized graph against its schema, mapping, and provenance."""
    graph_timestamp = int(graph_path.stem.split("_")[1])
    if provenance is None:
        provenance, flow_ids, _ = audit_provenance_file(
            provenance_path, mapping, expected_split, graph_timestamp,
        )
    else:
        flow_ids = provenance["flow_id"].astype(str).tolist()

    serialized = graph_path.read_bytes()
    data = torch.load(io.BytesIO(serialized), weights_only=False)
    edges = int(data.edge_index.shape[1])
    if len(provenance) != edges:
        raise AssertionError(f"Provenance does not match graph edges in {provenance_path}.")
    if data.edge_attr.shape != (edges, profile.dimension) or data.y.shape[0] != edges:
        raise AssertionError(f"Edge dimensions disagree in {graph_path}.")
    if data.global_node_ids.numel() != data.num_nodes:
        raise AssertionError(f"Node mapping dimensions disagree in {graph_path}.")
    if not torch.isfinite(data.edge_attr).all() or not torch.isfinite(data.y).all():
        raise AssertionError(f"Non-finite graph tensor found in {graph_path}.")
    if data.feature_profile != profile.name or data.schema_hash != profile.sha256():
        raise AssertionError(f"Profile schema metadata disagrees in {graph_path}.")
    if int(data.timestamp) != graph_timestamp or int(data.window_end) != graph_timestamp:
        raise AssertionError(f"Graph timestamp metadata disagrees in {graph_path}.")
    if not all(int(node_id) in mapping.id_to_ip for node_id in data.global_node_ids.tolist()):
        raise AssertionError(f"A graph ID cannot be decoded by its declared map in {graph_path}.")
    if (
        not provenance["window_start_ms"].eq(int(data.window_start)).all()
        or not provenance["window_end_ms"].eq(int(data.window_end)).all()
    ):
        raise AssertionError(f"Provenance window metadata disagrees in {provenance_path}.")
    if not np.array_equal(
        data.y.detach().cpu().numpy().astype(np.int8),
        provenance["binary_target"].to_numpy(dtype=np.int8),
    ):
        raise AssertionError(f"Graph targets disagree with provenance in {provenance_path}.")
    return (
        {"graphs": 1, "edges": edges, "positive_edges": int(data.y.sum().item())},
        flow_ids,
        {"sha256": hashlib.sha256(serialized).hexdigest(), "bytes": len(serialized)},
    )
