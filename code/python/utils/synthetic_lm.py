"""Synthetic lateral-movement sensitivity experiment for NF-CSE-CIC-IDS2018.

This module creates *aggregate-flow* counterfactuals.  It does not generate
packets and it must not be used as evidence that remote authentication or
execution really succeeded.  The intended use is a controlled sensitivity
test of already-trained binary flow classifiers.

The implementation deliberately preserves the existing graph/model contract:

* 30-second windows;
* 16 constant node features;
* 7 destination-port roles + 5 protocol roles + 20 numeric features;
* the scaler fitted on Wednesday 28 February;
* the recovered Day-2 IP/global-node-ID mapping.

Only sparse overlay graph files are written.  The original CSV, graph files,
checkpoints, and thresholds are never modified.
"""

from __future__ import annotations

import copy
import glob
import hashlib
import ipaddress
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


TIME_WINDOW_SECONDS = 30
DEFAULT_SEED = 20260804
DEFAULT_PIVOT_IP = "172.31.69.13"
DEFAULT_ATTACKER_IP = "13.58.225.34"
DEFAULT_INTERNAL_NETWORK = "172.31.0.0/16"

NUMERIC_FEATURES: Tuple[str, ...] = (
    "IN_BYTES",
    "OUT_BYTES",
    "IN_PKTS",
    "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN",
    "DURATION_OUT",
    "SRC_TO_DST_IAT_AVG",
    "DST_TO_SRC_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_STDDEV",
    "MIN_IP_PKT_LEN",
    "MAX_IP_PKT_LEN",
    "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_PKTS",
    "TCP_WIN_MAX_IN",
    "TCP_WIN_MAX_OUT",
    "TCP_FLAGS",
    "MIN_TTL",
    "MAX_TTL",
)

RAW_REQUIRED_COLUMNS: Tuple[str, ...] = (
    "FLOW_START_TIME",
    "IPV4_SRC_ADDR",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "PROTOCOL",
    "Attack",
) + NUMERIC_FEATURES

STAGE_CONTROL = 0
STAGE_AUTHENTICATION = 1
STAGE_LATERAL_MOVEMENT = 2
STAGE_CONFIRMATION = 3


class SyntheticLMError(RuntimeError):
    """Raised when an invariant needed for a defensible experiment fails."""


@dataclass(frozen=True)
class ProtocolSpec:
    name: str
    discovery_port: int
    lateral_ports: Tuple[int, ...]
    technique: str
    technique_name: str


PROTOCOL_SPECS: Mapping[str, ProtocolSpec] = {
    "SMB_RPC": ProtocolSpec(
        name="SMB_RPC",
        discovery_port=445,
        lateral_ports=(445, 135, 49152),
        technique="T1021.002/T1021.003",
        technique_name="SMB/Windows Admin Shares and DCOM/RPC",
    ),
    "RDP": ProtocolSpec(
        name="RDP",
        discovery_port=3389,
        lateral_ports=(3389,),
        technique="T1021.001",
        technique_name="Remote Desktop Protocol",
    ),
    "SSH": ProtocolSpec(
        name="SSH",
        discovery_port=22,
        lateral_ports=(22,),
        technique="T1021.004",
        technique_name="SSH",
    ),
}


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    protocol: str
    pivot_ip: str
    target_ip: str
    discovery_time: pd.Timestamp
    lm_time: pd.Timestamp
    horizon_minutes: int
    access_path: str
    first_window: int
    last_window: int
    scenario_type: str = "attack"
    linked_attack_scenario: str = ""


def _torch_load(path: os.PathLike | str, map_location: str | torch.device = "cpu"):
    """Load PyTorch objects across versions with and without weights_only."""

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_ip_map(path: os.PathLike | str) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Load and validate the recovered one-to-one IP/global-ID map."""

    with open(path, "rb") as handle:
        ip_to_id = pickle.load(handle)
    ip_to_id = {str(ip): int(node_id) for ip, node_id in ip_to_id.items()}
    if len(ip_to_id) != len(set(ip_to_id.values())):
        raise SyntheticLMError(f"IP map is not one-to-one: {path}")
    return ip_to_id, {node_id: ip for ip, node_id in ip_to_id.items()}


def audit_ip_map_usage(
    graph_root: os.PathLike | str,
    ip_to_id: Mapping[str, int],
    required_ips: Sequence[str],
    split: str = "test2",
) -> dict:
    """Check that required recovered IDs are genuinely present in Day-2 graphs."""

    missing_ips = sorted(set(map(str, required_ips)) - set(ip_to_id))
    if missing_ips:
        raise SyntheticLMError(f"Required IPs are missing from recovered map: {missing_ips}")

    required_ids = {int(ip_to_id[ip]) for ip in required_ips}
    seen_ids: set[int] = set()
    files = sorted(Path(graph_root, split).glob("graph_*.pt"))
    if not files:
        raise SyntheticLMError(f"No base graphs found under {Path(graph_root, split)}")

    for path in files:
        graph = _torch_load(path)
        seen_ids.update(int(value) for value in graph.global_node_ids.tolist())
        if required_ids <= seen_ids:
            break

    absent_ids = sorted(required_ids - seen_ids)
    if absent_ids:
        raise SyntheticLMError(f"Recovered IDs not observed in Day-2 graphs: {absent_ids}")

    return {
        "graph_files": len(files),
        "required_ips": len(required_ips),
        "required_ids_observed": len(required_ids),
        "map_entries": len(ip_to_id),
    }


def get_protocol_vector(protocol: int) -> List[int]:
    """Return the exact five-dimensional encoding used by graph creation."""

    protocol = int(protocol)
    if protocol == 6:
        return [1, 0, 0, 0, 0]
    if protocol == 17:
        return [0, 1, 0, 0, 0]
    if protocol in (1, 58):
        return [0, 0, 1, 0, 0]
    if protocol == 2:
        return [0, 0, 0, 1, 0]
    return [0, 0, 0, 0, 1]


def port_category_index(port: int) -> int:
    """Return the exact seven-role destination-port index used by the models."""

    port = int(port)
    if port in (80, 443, 8080, 8443, 81, 3128, 8545):
        return 0
    if port in (22, 222, 2222, 23, 2323, 3389, 3390, 3394, 5900, 5901, 5555, 21, 2131):
        return 1
    if port in (445, 135, 137, 138, 139):
        return 2
    if port in (53, 5355, 67, 547, 123, 1900, 5060):
        return 3
    if port in (1433, 3306, 5432, 6379, 27017):
        return 4
    if port < 1024:
        return 5
    return 6


def get_port_role_vector(port: int) -> List[int]:
    vector = [0] * 7
    vector[port_category_index(port)] = 1
    return vector


def window_id_for_time(
    timestamp: pd.Timestamp | str,
    global_start: pd.Timestamp | str,
    window_seconds: int = TIME_WINDOW_SECONDS,
) -> int:
    timestamp = pd.Timestamp(timestamp)
    global_start = pd.Timestamp(global_start)
    return int((timestamp - global_start) // pd.Timedelta(seconds=window_seconds))


def window_start_for_id(
    window_id: int,
    global_start: pd.Timestamp | str,
    window_seconds: int = TIME_WINDOW_SECONDS,
) -> pd.Timestamp:
    return pd.Timestamp(global_start) + pd.Timedelta(seconds=int(window_id) * window_seconds)


def operational_availability(
    flow_start: pd.Timestamp | str,
    duration_ms: float,
    global_start: pd.Timestamp | str,
    window_seconds: int = TIME_WINDOW_SECONDS,
) -> pd.Timestamp:
    """Earliest time the current offline graph/complete-flow features exist."""

    flow_start = pd.Timestamp(flow_start)
    win_id = window_id_for_time(flow_start, global_start, window_seconds)
    window_end = window_start_for_id(win_id, global_start, window_seconds) + pd.Timedelta(
        seconds=window_seconds
    )
    flow_end = flow_start + pd.Timedelta(milliseconds=float(duration_ms))
    return max(window_end, flow_end)


def _is_internal_series(series: pd.Series, network: str) -> pd.Series:
    # NF-CSE Day 2 uses 172.31.0.0/16.  The fast prefix path avoids applying
    # ipaddress to millions of rows; other networks retain correct semantics.
    parsed = ipaddress.ip_network(network)
    if parsed == ipaddress.ip_network("172.31.0.0/16"):
        return series.astype(str).str.startswith("172.31.")
    return series.astype(str).map(
        lambda value: ipaddress.ip_address(value) in parsed if value != "0.0.0.0" else False
    )


def load_relevant_day2_flows(
    csv_path: os.PathLike | str,
    pivot_ip: str = DEFAULT_PIVOT_IP,
    attacker_ip: str = DEFAULT_ATTACKER_IP,
    internal_network: str = DEFAULT_INTERNAL_NETWORK,
    chunksize: int = 250_000,
) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Load only rows needed for target selection and empirical donor pools.

    This is intentionally a filtered, chunked reader for Colab.  It retains
    campaign endpoints, internal TCP remote-service/admin/high-port traffic,
    and traffic involving the documented external attacker.
    """

    csv_path = Path(csv_path)
    header = set(pd.read_csv(csv_path, nrows=0).columns)
    missing = sorted(set(RAW_REQUIRED_COLUMNS) - header)
    if missing:
        raise SyntheticLMError(f"Day-2 CSV is missing required columns: {missing}")

    optional = [name for name in ("L4_SRC_PORT", "FLOW_END_TIME") if name in header]
    usecols = list(RAW_REQUIRED_COLUMNS) + optional
    retained: List[pd.DataFrame] = []
    global_start: Optional[pd.Timestamp] = None
    global_end: Optional[pd.Timestamp] = None
    row_offset = 0
    service_ports = {22, 135, 139, 445, 3389}

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        chunk["_raw_row_id"] = np.arange(row_offset, row_offset + len(chunk), dtype=np.int64)
        row_offset += len(chunk)
        chunk["FLOW_START_TIME"] = pd.to_datetime(chunk["FLOW_START_TIME"], errors="raise")

        chunk_min = chunk["FLOW_START_TIME"].min()
        chunk_max = chunk["FLOW_START_TIME"].max()
        global_start = chunk_min if global_start is None else min(global_start, chunk_min)
        global_end = chunk_max if global_end is None else max(global_end, chunk_max)

        src = chunk["IPV4_SRC_ADDR"].astype(str)
        dst = chunk["IPV4_DST_ADDR"].astype(str)
        valid = src.ne("0.0.0.0") & dst.ne("0.0.0.0")
        internal_pair = _is_internal_series(src, internal_network) & _is_internal_series(
            dst, internal_network
        )
        tcp = pd.to_numeric(chunk["PROTOCOL"], errors="coerce").eq(6)
        ports = pd.to_numeric(chunk["L4_DST_PORT"], errors="coerce").fillna(-1).astype(int)
        service_or_high = ports.isin(service_ports) | ports.ge(1024)
        campaign_endpoint = src.eq(pivot_ip) | dst.eq(pivot_ip)
        attacker_endpoint = src.eq(attacker_ip) | dst.eq(attacker_ip)
        keep = valid & (campaign_endpoint | attacker_endpoint | (internal_pair & tcp & service_or_high))

        if keep.any():
            retained.append(chunk.loc[keep].copy())

    if global_start is None or global_end is None:
        raise SyntheticLMError(f"No rows found in {csv_path}")
    if not retained:
        raise SyntheticLMError("No Day-2 rows matched the campaign/donor filter")

    result = pd.concat(retained, ignore_index=True)
    result["L4_DST_PORT"] = pd.to_numeric(result["L4_DST_PORT"], errors="raise").astype(int)
    result["PROTOCOL"] = pd.to_numeric(result["PROTOCOL"], errors="raise").astype(int)
    result["_src_internal"] = _is_internal_series(result["IPV4_SRC_ADDR"], internal_network)
    result["_dst_internal"] = _is_internal_series(result["IPV4_DST_ADDR"], internal_network)
    result["_port_category"] = result["L4_DST_PORT"].map(port_category_index).astype(int)
    return result, pd.Timestamp(global_start), pd.Timestamp(global_end)


def select_targets(
    relevant_flows: pd.DataFrame,
    ip_to_id: Mapping[str, int],
    pivot_ip: str = DEFAULT_PIVOT_IP,
    targets_per_protocol: int = 3,
    internal_network: str = DEFAULT_INTERNAL_NETWORK,
    excluded_ips: Sequence[str] = ("172.31.0.2",),
) -> pd.DataFrame:
    """Select deterministic targets with exact-port discovery evidence."""

    endpoint_counts = pd.concat(
        [
            relevant_flows["IPV4_SRC_ADDR"].astype(str),
            relevant_flows["IPV4_DST_ADDR"].astype(str),
        ],
        ignore_index=True,
    ).value_counts()
    rows: List[dict] = []
    network = ipaddress.ip_network(internal_network)

    for protocol_name, spec in PROTOCOL_SPECS.items():
        probes = relevant_flows[
            relevant_flows["IPV4_SRC_ADDR"].astype(str).eq(pivot_ip)
            & relevant_flows["Attack"].astype(str).eq("Infilteration")
            & relevant_flows["PROTOCOL"].eq(6)
            & relevant_flows["L4_DST_PORT"].eq(spec.discovery_port)
        ].copy()
        probes["IPV4_DST_ADDR"] = probes["IPV4_DST_ADDR"].astype(str)
        probes = probes[
            probes["IPV4_DST_ADDR"].map(
                lambda value: ipaddress.ip_address(value) in network
                and value != pivot_ip
                and value not in set(excluded_ips)
                and value in ip_to_id
            )
        ]

        grouped = (
            probes.groupby("IPV4_DST_ADDR", as_index=False)
            .agg(
                discovery_time=("FLOW_START_TIME", "min"),
                discovery_flows=("FLOW_START_TIME", "size"),
            )
        )
        grouped["background_flows"] = grouped["IPV4_DST_ADDR"].map(endpoint_counts).fillna(0).astype(int)
        grouped = grouped.sort_values(
            ["discovery_flows", "background_flows", "IPV4_DST_ADDR"],
            ascending=[False, False, True],
        )

        if len(grouped) < targets_per_protocol:
            raise SyntheticLMError(
                f"{protocol_name} has only {len(grouped)} exact-port discovered targets; "
                f"{targets_per_protocol} are required"
            )

        for rank, (_, target) in enumerate(grouped.head(targets_per_protocol).iterrows(), start=1):
            rows.append(
                {
                    "Protocol": protocol_name,
                    "Target_Rank": rank,
                    "Target_IP": str(target["IPV4_DST_ADDR"]),
                    "Target_Global_ID": int(ip_to_id[str(target["IPV4_DST_ADDR"])]),
                    "Discovery_Port": spec.discovery_port,
                    "Discovery_Time": pd.Timestamp(target["discovery_time"]),
                    "Discovery_Flows": int(target["discovery_flows"]),
                    "Background_Flows": int(target["background_flows"]),
                }
            )

    return pd.DataFrame(rows)


def build_scenario_matrix(
    selected_targets: pd.DataFrame,
    global_start: pd.Timestamp | str,
    global_end: pd.Timestamp | str,
    pivot_ip: str = DEFAULT_PIVOT_IP,
    horizons: Sequence[int] = (5, 15, 30),
    access_paths: Sequence[str] = ("valid_account", "authentication_attempts"),
) -> pd.DataFrame:
    """Build the selected 3×3×3×2 = 54 deterministic attack scenarios."""

    global_start = pd.Timestamp(global_start)
    global_end = pd.Timestamp(global_end)
    scenarios: List[ScenarioSpec] = []

    for _, target in selected_targets.sort_values(["Protocol", "Target_Rank"]).iterrows():
        discovery_time = pd.Timestamp(target["Discovery_Time"])
        for horizon in horizons:
            lm_time = discovery_time + pd.Timedelta(minutes=int(horizon))
            if lm_time + pd.Timedelta(minutes=2) > global_end:
                raise SyntheticLMError(
                    f"Scenario for {target['Target_IP']} at {lm_time} exceeds Day-2 capture"
                )
            for access_path in access_paths:
                first_time = (
                    lm_time - pd.Timedelta(seconds=120)
                    if access_path == "authentication_attempts"
                    else lm_time
                )
                scenario_id = (
                    f"{target['Protocol'].lower()}_target{int(target['Target_Rank'])}_"
                    f"h{int(horizon):02d}_{access_path}"
                )
                scenarios.append(
                    ScenarioSpec(
                        scenario_id=scenario_id,
                        protocol=str(target["Protocol"]),
                        pivot_ip=pivot_ip,
                        target_ip=str(target["Target_IP"]),
                        discovery_time=discovery_time,
                        lm_time=lm_time,
                        horizon_minutes=int(horizon),
                        access_path=access_path,
                        first_window=window_id_for_time(first_time, global_start),
                        last_window=window_id_for_time(
                            lm_time + pd.Timedelta(seconds=90), global_start
                        ),
                    )
                )

    frame = pd.DataFrame([asdict(scenario) for scenario in scenarios])
    expected = len(PROTOCOL_SPECS) * 3 * len(horizons) * len(access_paths)
    if len(frame) != expected:
        raise SyntheticLMError(f"Expected {expected} attack scenarios, generated {len(frame)}")
    return frame


def donor_support_report(
    relevant_flows: pd.DataFrame,
    min_exact: int = 20,
    min_category: int = 50,
) -> pd.DataFrame:
    """Report and enforce empirical support before any flow is generated."""

    internal_tcp = relevant_flows[
        relevant_flows["_src_internal"]
        & relevant_flows["_dst_internal"]
        & relevant_flows["PROTOCOL"].eq(6)
    ]
    rows: List[dict] = []
    for name, spec in PROTOCOL_SPECS.items():
        port = spec.discovery_port
        category = port_category_index(port)
        exact = int(internal_tcp["L4_DST_PORT"].eq(port).sum())
        same_category = int(internal_tcp["_port_category"].eq(category).sum())
        supported = exact >= min_exact or same_category >= min_category
        rows.append(
            {
                "Protocol": name,
                "Exact_Port": port,
                "Exact_Donors": exact,
                "Same_Category_Donors": same_category,
                "Support_Mode": "exact" if exact >= min_exact else "category",
                "Supported": supported,
            }
        )

    report = pd.DataFrame(rows)
    unsupported = report.loc[~report["Supported"], "Protocol"].tolist()
    if unsupported:
        raise SyntheticLMError(f"Insufficient donor support for protocols: {unsupported}")
    return report


def _stable_index(key: str, size: int, seed: int = DEFAULT_SEED) -> int:
    if size <= 0:
        raise SyntheticLMError(f"Cannot select a donor from an empty pool: {key}")
    digest = hashlib.sha256(f"{seed}:{key}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % size


def _candidate_donors(
    relevant_flows: pd.DataFrame,
    port: int,
    role: str,
    min_exact: int = 20,
) -> pd.DataFrame:
    pool = relevant_flows[
        relevant_flows["_src_internal"]
        & relevant_flows["_dst_internal"]
        & relevant_flows["PROTOCOL"].eq(6)
    ].copy()
    exact = pool[pool["L4_DST_PORT"].eq(int(port))]
    if len(exact) >= min_exact:
        pool = exact
    else:
        pool = pool[pool["_port_category"].eq(port_category_index(port))]

    duration = pd.to_numeric(pool["FLOW_DURATION_MILLISECONDS"], errors="coerce")
    in_pkts = pd.to_numeric(pool["IN_PKTS"], errors="coerce")
    out_pkts = pd.to_numeric(pool["OUT_PKTS"], errors="coerce")
    total_bytes = pd.to_numeric(pool["IN_BYTES"], errors="coerce") + pd.to_numeric(
        pool["OUT_BYTES"], errors="coerce"
    )

    if role == "authentication":
        role_pool = pool[duration.le(20_000)]
        if len(role_pool) >= 8:
            q25 = total_bytes.loc[role_pool.index].quantile(0.25)
            small = role_pool[total_bytes.loc[role_pool.index].le(q25)]
            pool = small if len(small) >= 8 else role_pool
    elif role == "session":
        bidirectional = pool[in_pkts.gt(0) & out_pkts.gt(0)]
        if not bidirectional.empty:
            median_duration = duration.loc[bidirectional.index].median()
            sustained = bidirectional[duration.loc[bidirectional.index].ge(median_duration)]
            pool = sustained if not sustained.empty else bidirectional
    elif role == "bulk":
        bidirectional = pool[in_pkts.gt(0) & out_pkts.gt(0)]
        if not bidirectional.empty:
            q75 = total_bytes.loc[bidirectional.index].quantile(0.75)
            bulk = bidirectional[total_bytes.loc[bidirectional.index].ge(q75)]
            pool = bulk if not bulk.empty else bidirectional

    if pool.empty:
        raise SyntheticLMError(f"No donor rows for port={port}, role={role}")
    return pool.sort_values("_raw_row_id")


def _select_donor(
    relevant_flows: pd.DataFrame,
    port: int,
    role: str,
    key: str,
    seed: int,
) -> pd.Series:
    pool = _candidate_donors(relevant_flows, port=port, role=role)
    return pool.iloc[_stable_index(key, len(pool), seed)]


def _select_confirmation_donor(
    relevant_flows: pd.DataFrame,
    key: str,
    seed: int,
) -> pd.Series:
    pool = relevant_flows[
        relevant_flows["PROTOCOL"].eq(6)
        & (relevant_flows["_src_internal"] ^ relevant_flows["_dst_internal"])
        & pd.to_numeric(relevant_flows["IN_PKTS"], errors="coerce").gt(0)
        & pd.to_numeric(relevant_flows["OUT_PKTS"], errors="coerce").gt(0)
    ].sort_values("_raw_row_id")
    if pool.empty:
        raise SyntheticLMError("No bidirectional internal/external TCP donor for confirmation")
    return pool.iloc[_stable_index(key, len(pool), seed)]


def _event_from_donor(
    donor: pd.Series,
    scenario: pd.Series,
    event_number: int,
    start_time: pd.Timestamp,
    source_ip: str,
    dest_ip: str,
    destination_port: int,
    stage: str,
    stage_code: int,
    technique: str,
    technique_name: str,
    role: str,
    global_start: pd.Timestamp,
    attack_label: str = "Infilteration",
) -> dict:
    row = {feature: donor[feature] for feature in NUMERIC_FEATURES}
    duration_ms = float(row["FLOW_DURATION_MILLISECONDS"])
    available = operational_availability(start_time, duration_ms, global_start)
    event_id = f"{scenario['scenario_id']}:{event_number:03d}"
    row.update(
        {
            "Synthetic_Event_ID": event_id,
            "Scenario_ID": str(scenario["scenario_id"]),
            "Scenario_Type": str(scenario.get("scenario_type", "attack")),
            "Protocol_Mechanism": str(scenario["protocol"]),
            "Access_Path": str(scenario["access_path"]),
            "Horizon_Minutes": int(scenario["horizon_minutes"]),
            "Pivot_IP": str(scenario["pivot_ip"]),
            "Target_IP": str(scenario["target_ip"]),
            "Discovery_Time": pd.Timestamp(scenario["discovery_time"]),
            "LM_Time": pd.Timestamp(scenario["lm_time"]),
            "FLOW_START_TIME": pd.Timestamp(start_time),
            "FLOW_END_TIME": pd.Timestamp(start_time) + pd.Timedelta(milliseconds=duration_ms),
            "Operational_Available_Time": available,
            "IPV4_SRC_ADDR": str(source_ip),
            "IPV4_DST_ADDR": str(dest_ip),
            "L4_DST_PORT": int(destination_port),
            "PROTOCOL": 6,
            "Attack": attack_label,
            "Synthetic_Stage": stage,
            "Synthetic_Stage_ID": int(stage_code),
            "ATTACK_Technique": technique,
            "ATTACK_Technique_Name": technique_name,
            "Event_Role": role,
            "Assumed_Success": stage_code in (STAGE_LATERAL_MOVEMENT, STAGE_CONFIRMATION),
            "Donor_Row_ID": int(donor["_raw_row_id"]),
            "Donor_Original_Port": int(donor["L4_DST_PORT"]),
            "Donor_Port_Category": int(donor["_port_category"]),
            "window_id": window_id_for_time(start_time, global_start),
        }
    )
    return row


def create_synthetic_flows(
    relevant_flows: pd.DataFrame,
    scenarios: pd.DataFrame,
    global_start: pd.Timestamp | str,
    attacker_ip: str = DEFAULT_ATTACKER_IP,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Create empirical-donor aggregate flows for every attack scenario."""

    global_start = pd.Timestamp(global_start)
    events: List[dict] = []

    for _, scenario in scenarios.sort_values("scenario_id").iterrows():
        protocol = PROTOCOL_SPECS[str(scenario["protocol"])]
        event_number = 0

        if scenario["access_path"] == "authentication_attempts":
            # The last attempt starts 45 seconds before LM.  With the enforced
            # <=20-second donor duration, both the flow and its 30-second graph
            # window are complete strictly before the LM onset.
            offsets = np.linspace(-120, -45, 8)
            for attempt_number, seconds in enumerate(offsets, start=1):
                donor = _select_donor(
                    relevant_flows,
                    protocol.discovery_port,
                    "authentication",
                    f"{scenario['scenario_id']}:auth:{attempt_number}",
                    seed,
                )
                events.append(
                    _event_from_donor(
                        donor=donor,
                        scenario=scenario,
                        event_number=event_number,
                        start_time=pd.Timestamp(scenario["lm_time"]) + pd.Timedelta(seconds=float(seconds)),
                        source_ip=str(scenario["pivot_ip"]),
                        dest_ip=str(scenario["target_ip"]),
                        destination_port=protocol.discovery_port,
                        stage="Authentication attempts",
                        stage_code=STAGE_AUTHENTICATION,
                        technique="T1110 (candidate)",
                        technique_name="Brute Force / repeated authentication attempts",
                        role="precursor",
                        global_start=global_start,
                    )
                )
                event_number += 1

        for port_offset, destination_port in enumerate(protocol.lateral_ports):
            role = "bulk" if protocol.name == "SMB_RPC" and destination_port == 445 else "session"
            donor = _select_donor(
                relevant_flows,
                destination_port,
                role,
                f"{scenario['scenario_id']}:lm:{destination_port}",
                seed,
            )
            events.append(
                _event_from_donor(
                    donor=donor,
                    scenario=scenario,
                    event_number=event_number,
                    start_time=pd.Timestamp(scenario["lm_time"]) + pd.Timedelta(seconds=10 * port_offset),
                    source_ip=str(scenario["pivot_ip"]),
                    dest_ip=str(scenario["target_ip"]),
                    destination_port=int(destination_port),
                    stage="Lateral Movement",
                    stage_code=STAGE_LATERAL_MOVEMENT,
                    technique=protocol.technique,
                    technique_name=protocol.technique_name,
                    role="synthetic_lm",
                    global_start=global_start,
                )
            )
            event_number += 1

        donor = _select_confirmation_donor(
            relevant_flows, f"{scenario['scenario_id']}:confirmation", seed
        )
        confirmation_port = int(donor["L4_DST_PORT"])
        events.append(
            _event_from_donor(
                donor=donor,
                scenario=scenario,
                event_number=event_number,
                start_time=pd.Timestamp(scenario["lm_time"]) + pd.Timedelta(seconds=60),
                source_ip=str(scenario["target_ip"]),
                dest_ip=attacker_ip,
                destination_port=confirmation_port,
                stage="Post-compromise confirmation",
                stage_code=STAGE_CONFIRMATION,
                technique="Synthetic confirmation only",
                technique_name="Target-originated communication after assumed movement",
                role="confirmation",
                global_start=global_start,
            )
        )

    return pd.DataFrame(events).sort_values(
        ["Scenario_ID", "FLOW_START_TIME", "Synthetic_Event_ID"]
    ).reset_index(drop=True)


def _select_control_sources(
    relevant_flows: pd.DataFrame,
    ip_to_id: Mapping[str, int],
    forbidden_ips: Iterable[str],
    count: int,
) -> List[str]:
    forbidden = set(map(str, forbidden_ips))
    internal = relevant_flows[
        relevant_flows["_src_internal"] & relevant_flows["_dst_internal"]
    ]
    candidates = pd.concat(
        [internal["IPV4_SRC_ADDR"].astype(str), internal["IPV4_DST_ADDR"].astype(str)]
    ).value_counts()
    candidates = [ip for ip in candidates.index if ip in ip_to_id and ip not in forbidden]
    if len(candidates) < count:
        raise SyntheticLMError(f"Need {count} unrelated control sources; found {len(candidates)}")
    return candidates[:count]


def create_matched_controls(
    synthetic_flows: pd.DataFrame,
    scenarios: pd.DataFrame,
    relevant_flows: pd.DataFrame,
    ip_to_id: Mapping[str, int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Create nine endpoint-permuted benign controls at the 15-minute horizon."""

    reference = scenarios[
        scenarios["horizon_minutes"].eq(15)
        & scenarios["access_path"].eq("valid_account")
    ].sort_values(["protocol", "target_ip"])
    if len(reference) != 9:
        raise SyntheticLMError(f"Expected nine matched-control references, found {len(reference)}")

    forbidden = set(scenarios["pivot_ip"]) | set(scenarios["target_ip"]) | {DEFAULT_ATTACKER_IP}
    control_sources = _select_control_sources(
        relevant_flows, ip_to_id, forbidden_ips=forbidden, count=len(reference)
    )

    control_scenarios: List[dict] = []
    control_flows: List[pd.DataFrame] = []
    for control_index, ((_, attack_scenario), control_source) in enumerate(
        zip(reference.iterrows(), control_sources), start=1
    ):
        linked_id = str(attack_scenario["scenario_id"])
        control_id = f"control_{control_index:02d}_{linked_id}"
        scenario_row = attack_scenario.to_dict()
        scenario_row.update(
            {
                "scenario_id": control_id,
                "scenario_type": "control",
                "linked_attack_scenario": linked_id,
                "pivot_ip": control_source,
            }
        )
        control_scenarios.append(scenario_row)

        copied = synthetic_flows[synthetic_flows["Scenario_ID"].eq(linked_id)].copy()
        copied["Scenario_ID"] = control_id
        copied["Scenario_Type"] = "control"
        copied["Pivot_IP"] = control_source
        copied.loc[
            copied["IPV4_SRC_ADDR"].eq(str(attack_scenario["pivot_ip"])),
            "IPV4_SRC_ADDR",
        ] = control_source
        copied["Attack"] = "Benign"
        copied["Synthetic_Stage"] = "Benign remote-service control"
        copied["Synthetic_Stage_ID"] = STAGE_CONTROL
        copied["ATTACK_Technique"] = ""
        copied["ATTACK_Technique_Name"] = "Matched endpoint-permutation control"
        copied["Assumed_Success"] = False
        copied["Synthetic_Event_ID"] = [
            f"{control_id}:{index:03d}" for index in range(len(copied))
        ]
        control_flows.append(copied)

    controls = pd.concat(control_flows, ignore_index=True)
    control_manifest = pd.DataFrame(control_scenarios)
    return controls, control_manifest


def validate_synthetic_flows(
    flows: pd.DataFrame,
    scenarios: pd.DataFrame,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """Validate numeric, temporal, identity, and scenario-count invariants."""

    issues: List[dict] = []

    def add(code: str, detail: str, scenario_id: str = ""):
        issues.append({"Scenario_ID": scenario_id, "Code": code, "Detail": detail})

    required = set(RAW_REQUIRED_COLUMNS) | {
        "Scenario_ID",
        "Synthetic_Event_ID",
        "Synthetic_Stage_ID",
        "Operational_Available_Time",
        "LM_Time",
        "window_id",
    }
    missing = sorted(required - set(flows.columns))
    if missing:
        add("missing_columns", str(missing))
    else:
        numeric = flows.loc[:, NUMERIC_FEATURES].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            add("non_finite_numeric", "At least one numeric donor feature is NaN or infinite")
        if (numeric < 0).any().any():
            add("negative_numeric", "At least one log1p input is negative")
        if (numeric["MIN_IP_PKT_LEN"] > numeric["MAX_IP_PKT_LEN"]).any():
            add("packet_length_order", "MIN_IP_PKT_LEN exceeds MAX_IP_PKT_LEN")
        if (numeric["MIN_TTL"] > numeric["MAX_TTL"]).any():
            add("ttl_order", "MIN_TTL exceeds MAX_TTL")
        if (numeric[["MIN_TTL", "MAX_TTL"]] > 255).any().any():
            add("ttl_range", "TTL exceeds 255")

        if flows["Synthetic_Event_ID"].duplicated().any():
            add("duplicate_event_id", "Synthetic event IDs are not unique")

        attack_scenarios = scenarios[scenarios["scenario_type"].eq("attack")]
        for _, scenario in attack_scenarios.iterrows():
            scenario_id = str(scenario["scenario_id"])
            part = flows[flows["Scenario_ID"].eq(scenario_id)]
            if part.empty:
                add("missing_scenario_flows", "No synthetic flows generated", scenario_id)
                continue
            lm = part[part["Synthetic_Stage_ID"].eq(STAGE_LATERAL_MOVEMENT)]
            confirmation = part[part["Synthetic_Stage_ID"].eq(STAGE_CONFIRMATION)]
            auth = part[part["Synthetic_Stage_ID"].eq(STAGE_AUTHENTICATION)]
            if lm.empty:
                add("missing_lm", "No lateral-movement event", scenario_id)
            if len(confirmation) != 1:
                add("confirmation_count", f"Expected one confirmation, found {len(confirmation)}", scenario_id)
            if scenario["access_path"] == "authentication_attempts" and len(auth) != 8:
                add("authentication_count", f"Expected eight attempts, found {len(auth)}", scenario_id)
            if scenario["access_path"] == "valid_account" and not auth.empty:
                add("unexpected_authentication", "Direct path contains authentication attempts", scenario_id)
            if not auth.empty and (
                pd.to_datetime(auth["Operational_Available_Time"]) >= pd.Timestamp(scenario["lm_time"])
            ).any():
                add("late_auth_features", "Authentication features complete at/after LM", scenario_id)
            if not lm.empty and pd.to_datetime(lm["FLOW_START_TIME"]).min() != pd.Timestamp(
                scenario["lm_time"]
            ):
                add("lm_onset_mismatch", "First LM flow does not equal declared onset", scenario_id)

    report = pd.DataFrame(issues, columns=["Scenario_ID", "Code", "Detail"])
    if raise_on_error and not report.empty:
        preview = report.head(20).to_dict("records")
        raise SyntheticLMError(f"Synthetic-flow validation failed: {preview}")
    return report


def build_edge_attr(flows: pd.DataFrame, scaler) -> torch.Tensor:
    """Build the exact 32-dimensional edge vector expected by checkpoints."""

    if flows.empty:
        return torch.empty((0, 32), dtype=torch.float32)
    numeric = flows.loc[:, NUMERIC_FEATURES].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise SyntheticLMError("Numeric features must be finite and nonnegative before log1p")
    scaled = scaler.transform(np.log1p(numeric)).astype(np.float32)
    ports = np.asarray([get_port_role_vector(port) for port in flows["L4_DST_PORT"]], dtype=np.float32)
    protocols = np.asarray(
        [get_protocol_vector(proto) for proto in flows["PROTOCOL"]], dtype=np.float32
    )
    edge_attr = np.concatenate([ports, protocols, scaled], axis=1)
    if edge_attr.shape[1] != 32 or not np.isfinite(edge_attr).all():
        raise SyntheticLMError(f"Invalid transformed edge attributes: shape={edge_attr.shape}")
    return torch.tensor(edge_attr, dtype=torch.float32)


def _metadata_list(graph, name: str, length: int, default):
    existing = getattr(graph, name, None)
    if existing is None:
        return [default for _ in range(length)]
    values = list(existing)
    if len(values) != length:
        raise SyntheticLMError(f"Existing graph metadata {name} has length {len(values)}, expected {length}")
    return values


def append_synthetic_edges(
    base_graph,
    flows: pd.DataFrame,
    ip_to_id: Mapping[str, int],
    scaler,
):
    """Return a copy of a graph with synthetic rows appended after base edges."""

    graph = copy.deepcopy(base_graph)
    old_edges = int(graph.edge_index.shape[1])
    if flows.empty:
        return graph

    missing_ips = sorted(
        (set(flows["IPV4_SRC_ADDR"].astype(str)) | set(flows["IPV4_DST_ADDR"].astype(str)))
        - set(ip_to_id)
    )
    if missing_ips:
        raise SyntheticLMError(f"Synthetic endpoints missing from recovered map: {missing_ips}")

    global_ids = [int(value) for value in graph.global_node_ids.tolist()]
    global_to_local = {global_id: local_id for local_id, global_id in enumerate(global_ids)}
    node_dim = int(graph.x.shape[1]) if graph.x.ndim == 2 and graph.x.shape[1] else 16

    def local_id_for_ip(ip: str) -> int:
        global_id = int(ip_to_id[str(ip)])
        if global_id not in global_to_local:
            global_to_local[global_id] = len(global_ids)
            global_ids.append(global_id)
            new_node = torch.ones((1, node_dim), dtype=graph.x.dtype, device=graph.x.device)
            graph.x = torch.cat([graph.x, new_node], dim=0)
        return global_to_local[global_id]

    source_local = [local_id_for_ip(ip) for ip in flows["IPV4_SRC_ADDR"].astype(str)]
    dest_local = [local_id_for_ip(ip) for ip in flows["IPV4_DST_ADDR"].astype(str)]
    new_edge_index = torch.tensor(
        [source_local, dest_local], dtype=torch.long, device=graph.edge_index.device
    )
    new_edge_attr = build_edge_attr(flows, scaler).to(graph.edge_attr.device)
    new_y = torch.tensor(
        flows["Attack"].astype(str).eq("Infilteration").astype(float).to_numpy(),
        dtype=graph.y.dtype,
        device=graph.y.device,
    )

    graph.edge_index = torch.cat([graph.edge_index, new_edge_index], dim=1)
    graph.edge_attr = torch.cat([graph.edge_attr, new_edge_attr], dim=0)
    graph.y = torch.cat([graph.y, new_y], dim=0)
    graph.global_node_ids = torch.tensor(
        global_ids, dtype=torch.long, device=graph.global_node_ids.device
    )

    graph.synthetic_mask = torch.cat(
        [
            torch.zeros(old_edges, dtype=torch.bool, device=graph.y.device),
            torch.ones(len(flows), dtype=torch.bool, device=graph.y.device),
        ]
    )
    graph.synthetic_stage_id = torch.cat(
        [
            torch.full((old_edges,), -1, dtype=torch.long, device=graph.y.device),
            torch.tensor(
                flows["Synthetic_Stage_ID"].to_numpy(dtype=np.int64),
                dtype=torch.long,
                device=graph.y.device,
            ),
        ]
    )
    graph.synthetic_start_time_ms = torch.cat(
        [
            torch.full((old_edges,), -1, dtype=torch.long, device=graph.y.device),
            torch.tensor(
                pd.to_datetime(flows["FLOW_START_TIME"]).astype("int64").to_numpy() // 1_000_000,
                dtype=torch.long,
                device=graph.y.device,
            ),
        ]
    )
    graph.synthetic_available_time_ms = torch.cat(
        [
            torch.full((old_edges,), -1, dtype=torch.long, device=graph.y.device),
            torch.tensor(
                pd.to_datetime(flows["Operational_Available_Time"]).astype("int64").to_numpy()
                // 1_000_000,
                dtype=torch.long,
                device=graph.y.device,
            ),
        ]
    )
    graph.synthetic_event_ids = _metadata_list(graph, "synthetic_event_ids", old_edges, "") + list(
        flows["Synthetic_Event_ID"].astype(str)
    )
    graph.synthetic_scenario_ids = _metadata_list(
        graph, "synthetic_scenario_ids", old_edges, ""
    ) + list(flows["Scenario_ID"].astype(str))
    graph.synthetic_stage_names = _metadata_list(
        graph, "synthetic_stage_names", old_edges, ""
    ) + list(flows["Synthetic_Stage"].astype(str))
    graph.synthetic_techniques = _metadata_list(
        graph, "synthetic_techniques", old_edges, ""
    ) + list(flows["ATTACK_Technique"].astype(str))
    graph.synthetic_roles = _metadata_list(graph, "synthetic_roles", old_edges, "") + list(
        flows["Event_Role"].astype(str)
    )
    graph.is_empty = False
    return graph


def graph_file_for_window(
    graph_root: os.PathLike | str, window_id: int, split: str = "test2"
) -> Path:
    return Path(graph_root, split, f"graph_{int(window_id):06d}.pt")


def write_sparse_overlays(
    base_graph_root: os.PathLike | str,
    overlay_root: os.PathLike | str,
    flows: pd.DataFrame,
    scenario_manifest: pd.DataFrame,
    ip_to_id: Mapping[str, int],
    scaler,
    split: str = "test2",
) -> pd.DataFrame:
    """Write only modified graph windows plus CSV provenance manifests."""

    overlay_root = Path(overlay_root)
    overlay_root.mkdir(parents=True, exist_ok=True)
    written: List[dict] = []

    scenario_manifest.to_csv(overlay_root / "scenario_manifest.csv", index=False)
    flows.to_csv(overlay_root / "synthetic_flows.csv", index=False)

    for scenario_id, scenario_flows in flows.groupby("Scenario_ID", sort=True):
        scenario_dir = overlay_root / str(scenario_id) / split
        scenario_dir.mkdir(parents=True, exist_ok=True)
        expected_names = {
            f"graph_{int(window_id):06d}.pt" for window_id in scenario_flows["window_id"].unique()
        }
        existing_names = {path.name for path in scenario_dir.glob("graph_*.pt")}
        stale_names = sorted(existing_names - expected_names)
        if stale_names:
            raise SyntheticLMError(
                f"Stale overlay windows in {scenario_dir}: {stale_names}. "
                "Use a new output directory or remove only that scenario overlay."
            )
        for window_id, window_flows in scenario_flows.groupby("window_id", sort=True):
            base_path = graph_file_for_window(base_graph_root, int(window_id), split)
            if not base_path.exists():
                raise SyntheticLMError(f"Base graph does not exist: {base_path}")
            base_graph = _torch_load(base_path)
            base_edges = int(base_graph.edge_index.shape[1])
            overlay_graph = append_synthetic_edges(base_graph, window_flows, ip_to_id, scaler)
            assert_overlay_preserves_base(base_graph, overlay_graph)
            output_path = scenario_dir / base_path.name
            torch.save(overlay_graph, output_path)
            written.append(
                {
                    "Scenario_ID": scenario_id,
                    "Window_ID": int(window_id),
                    "Base_Edges": base_edges,
                    "Synthetic_Edges": len(window_flows),
                    "Overlay_Edges": int(overlay_graph.edge_index.shape[1]),
                    "Overlay_Path": str(output_path),
                }
            )

    write_report = pd.DataFrame(written)
    write_report.to_csv(overlay_root / "overlay_write_report.csv", index=False)
    return write_report


def validate_overlay_layout(
    overlay_root: os.PathLike | str,
    flows: pd.DataFrame,
    split: str = "test2",
) -> pd.DataFrame:
    """Require exactly the expected sparse files for every scenario."""

    overlay_root = Path(overlay_root)
    rows: List[dict] = []
    errors: List[str] = []
    for scenario_id, part in flows.groupby("Scenario_ID", sort=True):
        directory = overlay_root / str(scenario_id) / split
        expected = {f"graph_{int(value):06d}.pt" for value in part["window_id"].unique()}
        actual = {path.name for path in directory.glob("graph_*.pt")} if directory.exists() else set()
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        rows.append(
            {
                "Scenario_ID": scenario_id,
                "Expected_Windows": len(expected),
                "Actual_Windows": len(actual),
                "Missing_Windows": len(missing),
                "Unexpected_Windows": len(unexpected),
            }
        )
        if missing or unexpected:
            errors.append(
                f"{scenario_id}: missing={missing}, unexpected={unexpected}"
            )
    if errors:
        raise SyntheticLMError("Invalid sparse overlay layout: " + "; ".join(errors[:10]))
    return pd.DataFrame(rows)


class SyntheticOverlayDataset(Dataset):
    """Chronological Day-2 dataset that substitutes sparse overlay windows."""

    def __init__(
        self,
        base_root: os.PathLike | str,
        overlay_root: Optional[os.PathLike | str] = None,
        split: str = "test2",
    ):
        self.base_dir = Path(base_root, split)
        self.overlay_dir = Path(overlay_root, split) if overlay_root is not None else None
        self.files = sorted(
            self.base_dir.glob("graph_*.pt"), key=lambda path: int(path.stem.split("_")[1])
        )
        if not self.files:
            raise SyntheticLMError(f"No graphs found under {self.base_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def path_at(self, index: int) -> Path:
        base_path = self.files[index]
        if self.overlay_dir is not None:
            overlay_path = self.overlay_dir / base_path.name
            if overlay_path.exists():
                return overlay_path
        return base_path

    def window_id_at(self, index: int) -> int:
        return int(self.files[index].stem.split("_")[1])

    def index_for_window(self, window_id: int) -> int:
        # Graph generation writes every window, so filename order and window ID
        # normally coincide.  Keep a checked fallback for robustness.
        window_id = int(window_id)
        if 0 <= window_id < len(self.files) and self.window_id_at(window_id) == window_id:
            return window_id
        for index, path in enumerate(self.files):
            if int(path.stem.split("_")[1]) == window_id:
                return index
        raise KeyError(f"Window {window_id} is not present in {self.base_dir}")

    def __getitem__(self, index: int):
        return _torch_load(self.path_at(index))


def snapshot_node_memory(model) -> Dict[int, torch.Tensor]:
    return {
        int(node_id): value.detach().cpu().clone()
        for node_id, value in getattr(model, "node_memory", {}).items()
    }


def restore_node_memory(model, snapshot: Mapping[int, torch.Tensor], device) -> None:
    model.node_memory = {
        int(node_id): value.detach().to(device).clone() for node_id, value in snapshot.items()
    }


def _forward_model(model, graph, temporal: bool):
    if temporal:
        return model(graph.x, graph.edge_index, graph.edge_attr, graph.global_node_ids)
    return model(graph.x, graph.edge_index, graph.edge_attr)


def _edge_endpoint_ips(graph, id_to_ip: Mapping[int, str]) -> Tuple[List[str], List[str]]:
    src_local = graph.edge_index[0].detach().cpu().numpy()
    dst_local = graph.edge_index[1].detach().cpu().numpy()
    global_ids = graph.global_node_ids.detach().cpu().numpy()
    source = [id_to_ip[int(global_ids[index])] for index in src_local]
    dest = [id_to_ip[int(global_ids[index])] for index in dst_local]
    return source, dest


def _prediction_rows(
    model_name: str,
    graph,
    window_id: int,
    probabilities: np.ndarray,
    threshold: float,
    id_to_ip: Mapping[int, str],
    keep_mask: np.ndarray,
) -> List[dict]:
    source, dest = _edge_endpoint_ips(graph, id_to_ip)
    y_true = graph.y.detach().cpu().numpy().astype(int)
    synthetic_mask = getattr(
        graph, "synthetic_mask", torch.zeros(len(y_true), dtype=torch.bool, device=graph.y.device)
    ).detach().cpu().numpy().astype(bool)
    stage_ids = getattr(
        graph, "synthetic_stage_id", torch.full((len(y_true),), -1, dtype=torch.long, device=graph.y.device)
    ).detach().cpu().numpy().astype(int)
    start_ms = getattr(
        graph,
        "synthetic_start_time_ms",
        torch.full((len(y_true),), -1, dtype=torch.long, device=graph.y.device),
    ).detach().cpu().numpy()
    available_ms = getattr(
        graph,
        "synthetic_available_time_ms",
        torch.full((len(y_true),), -1, dtype=torch.long, device=graph.y.device),
    ).detach().cpu().numpy()
    event_ids = _metadata_list(graph, "synthetic_event_ids", len(y_true), "")
    scenario_ids = _metadata_list(graph, "synthetic_scenario_ids", len(y_true), "")
    stage_names = _metadata_list(graph, "synthetic_stage_names", len(y_true), "")
    techniques = _metadata_list(graph, "synthetic_techniques", len(y_true), "")
    roles = _metadata_list(graph, "synthetic_roles", len(y_true), "")
    window_start = pd.Timestamp(graph.timestamp)

    rows: List[dict] = []
    for edge_index in np.flatnonzero(keep_mask):
        synthetic = bool(synthetic_mask[edge_index])
        flow_start = (
            pd.to_datetime(int(start_ms[edge_index]), unit="ms") if synthetic else window_start
        )
        available = (
            pd.to_datetime(int(available_ms[edge_index]), unit="ms")
            if synthetic
            else window_start + pd.Timedelta(seconds=TIME_WINDOW_SECONDS)
        )
        rows.append(
            {
                "Model": model_name,
                "Event_ID": f"{window_id}:{edge_index}",
                "Graph_Window_Idx": int(window_id),
                "Flow_Index_In_Window": int(edge_index),
                "Window_Start": window_start,
                "Flow_Start": flow_start,
                "Operational_Available_Time": available,
                "Availability_Basis": "complete_flow_and_window" if synthetic else "window_end_pending_raw_merge",
                "Source_IP": source[edge_index],
                "Dest_IP": dest[edge_index],
                "y_real": int(y_true[edge_index]),
                "Probability": float(probabilities[edge_index]),
                "y_pred": int(probabilities[edge_index] >= threshold),
                "Validation_Threshold": float(threshold),
                "Is_Synthetic": synthetic,
                "Synthetic_Event_ID": event_ids[edge_index],
                "Scenario_ID": scenario_ids[edge_index],
                "Synthetic_Stage_ID": int(stage_ids[edge_index]),
                "Synthetic_Stage": stage_names[edge_index],
                "ATTACK_Technique": techniques[edge_index],
                "Synthetic_Role": roles[edge_index],
            }
        )
    return rows


def collect_base_context_and_snapshots(
    model,
    model_name: str,
    threshold: float,
    dataset: SyntheticOverlayDataset,
    id_to_ip: Mapping[int, str],
    pivot_ip: str,
    snapshot_windows: Iterable[int],
    device: str | torch.device,
    temporal: bool,
) -> Tuple[pd.DataFrame, Dict[int, Dict[int, torch.Tensor]]]:
    """Run the base sequence once, collecting pivot events and memory snapshots."""

    device = torch.device(device)
    wanted_snapshots = set(map(int, snapshot_windows))
    snapshots: Dict[int, Dict[int, torch.Tensor]] = {}
    rows: List[dict] = []
    model.eval()
    if temporal and hasattr(model, "reset_memory"):
        model.reset_memory()

    with torch.no_grad():
        for index in range(len(dataset)):
            window_id = dataset.window_id_at(index)
            if temporal and window_id in wanted_snapshots:
                snapshots[window_id] = snapshot_node_memory(model)
            graph = dataset[index].to(device)
            if graph.x.shape[0] == 0:
                continue
            logits = _forward_model(model, graph, temporal).view(-1)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            source, dest = _edge_endpoint_ips(graph, id_to_ip)
            endpoint_mask = np.asarray(
                [src == pivot_ip or dst == pivot_ip for src, dst in zip(source, dest)], dtype=bool
            )
            y_true = graph.y.detach().cpu().numpy().astype(int)
            relevant = endpoint_mask & ((probabilities >= threshold) | (y_true == 1))
            rows.extend(
                _prediction_rows(
                    model_name,
                    graph,
                    window_id,
                    probabilities,
                    threshold,
                    id_to_ip,
                    relevant,
                )
            )

    missing = wanted_snapshots - set(snapshots) if temporal else set()
    if missing:
        raise SyntheticLMError(f"Missing temporal snapshots for windows: {sorted(missing)}")
    return pd.DataFrame(rows), snapshots


def collect_scenario_predictions(
    model,
    model_name: str,
    threshold: float,
    dataset: SyntheticOverlayDataset,
    id_to_ip: Mapping[int, str],
    scenario: pd.Series,
    device: str | torch.device,
    temporal: bool,
    memory_snapshot: Optional[Mapping[int, torch.Tensor]] = None,
) -> pd.DataFrame:
    """Evaluate only the short modified interval for one scenario."""

    device = torch.device(device)
    if temporal:
        if memory_snapshot is None:
            raise SyntheticLMError("Temporal scenario evaluation requires a base-memory snapshot")
        restore_node_memory(model, memory_snapshot, device)
    model.eval()
    rows: List[dict] = []
    focus_ips = {str(scenario["pivot_ip"]), str(scenario["target_ip"])}

    start_index = dataset.index_for_window(int(scenario["first_window"]))
    end_index = dataset.index_for_window(int(scenario["last_window"]))
    with torch.no_grad():
        for index in range(start_index, end_index + 1):
            graph = dataset[index].to(device)
            if graph.x.shape[0] == 0:
                continue
            logits = _forward_model(model, graph, temporal).view(-1)
            probabilities = torch.sigmoid(logits).detach().cpu().numpy()
            source, dest = _edge_endpoint_ips(graph, id_to_ip)
            synthetic_mask = getattr(
                graph,
                "synthetic_mask",
                torch.zeros(len(probabilities), dtype=torch.bool, device=graph.y.device),
            ).detach().cpu().numpy().astype(bool)
            focus_mask = np.asarray(
                [src in focus_ips or dst in focus_ips for src, dst in zip(source, dest)], dtype=bool
            )
            y_true = graph.y.detach().cpu().numpy().astype(int)
            keep = synthetic_mask | (focus_mask & ((probabilities >= threshold) | (y_true == 1)))
            rows.extend(
                _prediction_rows(
                    model_name,
                    graph,
                    dataset.window_id_at(index),
                    probabilities,
                    threshold,
                    id_to_ip,
                    keep,
                )
            )
    return pd.DataFrame(rows)


def evaluate_model_scenarios(
    model,
    model_name: str,
    threshold: float,
    base_graph_root: os.PathLike | str,
    overlay_root: os.PathLike | str,
    scenario_manifest: pd.DataFrame,
    id_to_ip: Mapping[int, str],
    pivot_ip: str,
    device: str | torch.device,
    temporal: bool,
    split: str = "test2",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate one fixed champion efficiently across all attack/control overlays."""

    base_dataset = SyntheticOverlayDataset(base_graph_root, split=split)
    snapshot_windows = sorted(set(scenario_manifest["first_window"].astype(int)))
    base_events, snapshots = collect_base_context_and_snapshots(
        model=model,
        model_name=model_name,
        threshold=threshold,
        dataset=base_dataset,
        id_to_ip=id_to_ip,
        pivot_ip=pivot_ip,
        snapshot_windows=snapshot_windows,
        device=device,
        temporal=temporal,
    )

    scenario_results: List[pd.DataFrame] = []
    for _, scenario in scenario_manifest.sort_values("scenario_id").iterrows():
        scenario_dataset = SyntheticOverlayDataset(
            base_graph_root,
            overlay_root=Path(overlay_root, str(scenario["scenario_id"])),
            split=split,
        )
        frame = collect_scenario_predictions(
            model=model,
            model_name=model_name,
            threshold=threshold,
            dataset=scenario_dataset,
            id_to_ip=id_to_ip,
            scenario=scenario,
            device=device,
            temporal=temporal,
            memory_snapshot=snapshots.get(int(scenario["first_window"])),
        )
        scenario_results.append(frame)

    combined = pd.concat(scenario_results, ignore_index=True) if scenario_results else pd.DataFrame()
    return base_events, combined


def build_raw_timing_index(
    csv_path: os.PathLike | str,
    focus_ips: Sequence[str],
    global_start: pd.Timestamp | str,
    chunksize: int = 500_000,
) -> pd.DataFrame:
    """Recover exact edge indexes and complete-flow availability for focus IPs.

    Edge indexes require counting every valid row in a window, but only focus-IP
    rows are retained in the returned table.  Chunk carry-over preserves counts
    for a window split across adjacent CSV chunks.
    """

    usecols = [
        "FLOW_START_TIME",
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
        "FLOW_DURATION_MILLISECONDS",
        "Attack",
    ]
    global_start = pd.Timestamp(global_start)
    focus = set(map(str, focus_ips))
    output: List[pd.DataFrame] = []
    next_index_by_window: MutableMapping[int, int] = {}

    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize):
        chunk["FLOW_START_TIME"] = pd.to_datetime(chunk["FLOW_START_TIME"], errors="raise")
        chunk = chunk[
            chunk["IPV4_SRC_ADDR"].astype(str).ne("0.0.0.0")
            & chunk["IPV4_DST_ADDR"].astype(str).ne("0.0.0.0")
        ].copy()
        chunk["Graph_Window_Idx"] = (
            (chunk["FLOW_START_TIME"] - global_start)
            // pd.Timedelta(seconds=TIME_WINDOW_SECONDS)
        ).astype(int)

        edge_indexes = np.empty(len(chunk), dtype=np.int64)
        for window_id, positions in chunk.groupby("Graph_Window_Idx", sort=False).indices.items():
            start = next_index_by_window.get(int(window_id), 0)
            positions = np.asarray(positions)
            edge_indexes[positions] = np.arange(start, start + len(positions), dtype=np.int64)
            next_index_by_window[int(window_id)] = start + len(positions)
        chunk["Flow_Index_In_Window"] = edge_indexes

        keep = chunk["IPV4_SRC_ADDR"].astype(str).isin(focus) | chunk[
            "IPV4_DST_ADDR"
        ].astype(str).isin(focus)
        part = chunk.loc[keep].copy()
        if not part.empty:
            part["Event_ID"] = (
                part["Graph_Window_Idx"].astype(str)
                + ":"
                + part["Flow_Index_In_Window"].astype(str)
            )
            part["Operational_Available_Time_Raw"] = [
                operational_availability(start, duration, global_start)
                for start, duration in zip(
                    part["FLOW_START_TIME"], part["FLOW_DURATION_MILLISECONDS"]
                )
            ]
            output.append(part)

    if not output:
        return pd.DataFrame()
    return pd.concat(output, ignore_index=True)


def enrich_base_availability(
    base_events: pd.DataFrame, raw_timing_index: pd.DataFrame
) -> pd.DataFrame:
    if base_events.empty or raw_timing_index.empty:
        return base_events.copy()
    timing = raw_timing_index[
        ["Event_ID", "FLOW_START_TIME", "Operational_Available_Time_Raw", "Attack"]
    ].drop_duplicates("Event_ID")
    merged = base_events.merge(timing, on="Event_ID", how="left", validate="many_to_one")
    found = merged["Operational_Available_Time_Raw"].notna()
    merged.loc[found, "Flow_Start"] = merged.loc[found, "FLOW_START_TIME"]
    merged.loc[found, "Operational_Available_Time"] = merged.loc[
        found, "Operational_Available_Time_Raw"
    ]
    merged.loc[found, "Availability_Basis"] = "complete_flow_and_window"
    return merged.drop(
        columns=["FLOW_START_TIME", "Operational_Available_Time_Raw", "Attack"],
        errors="ignore",
    )


def attach_corrected_campaign_ground_truth(
    base_events: pd.DataFrame,
    corrected_output_dir: os.PathLike | str,
    require_all_models: bool = True,
) -> pd.DataFrame:
    """Attach corrected notebook campaign attribution to base graph events.

    The corrected analysis exports one ``campaign_test2_*_corrected_v2.csv``
    per champion.  Reusing those Event_ID/Model assignments prevents a generic
    attack-labelled pivot flow from being silently promoted to a documented
    campaign precursor.
    """

    corrected_output_dir = Path(corrected_output_dir)
    files = sorted(corrected_output_dir.glob("campaign_test2_*_corrected_v2.csv"))
    if not files:
        raise SyntheticLMError(
            f"No corrected campaign outputs found under {corrected_output_dir}"
        )

    assignments: List[pd.DataFrame] = []
    for path in files:
        header = set(pd.read_csv(path, nrows=0).columns)
        required = {"Event_ID", "Model", "Campaign_GT"}
        if not required <= header:
            continue
        frame = pd.read_csv(path, usecols=list(required))
        frame["Campaign_GT"] = frame["Campaign_GT"].map(
            lambda value: value
            if isinstance(value, (bool, np.bool_))
            else str(value).strip().lower() == "true"
        )
        assignments.append(frame)

    if not assignments:
        raise SyntheticLMError("Corrected files do not contain Event_ID/Model/Campaign_GT")
    campaign = pd.concat(assignments, ignore_index=True).drop_duplicates(["Model", "Event_ID"])
    result = base_events.merge(campaign, on=["Model", "Event_ID"], how="left", validate="many_to_one")
    result["Campaign_GT"] = result["Campaign_GT"].fillna(False).astype(bool)

    missing_models = sorted(set(base_events["Model"]) - set(campaign["Model"]))
    if require_all_models and missing_models:
        raise SyntheticLMError(
            f"Corrected campaign attribution is missing models: {missing_models}"
        )
    return result


def summarize_experiment(
    scenario_predictions: pd.DataFrame,
    base_events: pd.DataFrame,
    scenario_manifest: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Produce scenario-level attack results and matched-control results."""

    attack_manifest = scenario_manifest[scenario_manifest["scenario_type"].eq("attack")]
    control_manifest = scenario_manifest[scenario_manifest["scenario_type"].eq("control")]
    attack_rows: List[dict] = []
    control_rows: List[dict] = []

    for model_name in sorted(scenario_predictions["Model"].unique()):
        model_base = base_events[base_events["Model"].eq(model_name)]
        model_predictions = scenario_predictions[scenario_predictions["Model"].eq(model_name)]

        for _, scenario in attack_manifest.iterrows():
            part = model_predictions[model_predictions["Scenario_ID"].eq(scenario["scenario_id"])]
            lm = part[part["Synthetic_Stage_ID"].eq(STAGE_LATERAL_MOVEMENT)]
            auth = part[
                part["Synthetic_Stage_ID"].eq(STAGE_AUTHENTICATION)
                & part["y_real"].eq(1)
                & part["y_pred"].eq(1)
                & (pd.to_datetime(part["Operational_Available_Time"]) < pd.Timestamp(scenario["lm_time"]))
            ]
            campaign_mask = (
                model_base["Campaign_GT"].astype(bool)
                if "Campaign_GT" in model_base.columns
                else model_base["y_real"].eq(1)
            )
            original = model_base[
                campaign_mask
                & model_base["y_pred"].eq(1)
                & (pd.to_datetime(model_base["Operational_Available_Time"]) < pd.Timestamp(scenario["lm_time"]))
            ]
            precursor_times = pd.concat(
                [
                    pd.to_datetime(original["Operational_Available_Time"]),
                    pd.to_datetime(auth["Operational_Available_Time"]),
                ],
                ignore_index=True,
            ).dropna()
            first_precursor = precursor_times.min() if not precursor_times.empty else pd.NaT
            lead = (
                (pd.Timestamp(scenario["lm_time"]) - first_precursor).total_seconds() / 60
                if pd.notna(first_precursor)
                else np.nan
            )
            detected_lm = lm[lm["y_pred"].eq(1)]
            first_lm_alert = (
                pd.to_datetime(detected_lm["Operational_Available_Time"]).min()
                if not detected_lm.empty
                else pd.NaT
            )
            lm_detection_delay = (
                (first_lm_alert - pd.Timestamp(scenario["lm_time"])).total_seconds() / 60
                if pd.notna(first_lm_alert)
                else np.nan
            )
            attack_rows.append(
                {
                    "Model": model_name,
                    "Scenario_ID": scenario["scenario_id"],
                    "Protocol": scenario["protocol"],
                    "Target_IP": scenario["target_ip"],
                    "Horizon_Minutes": int(scenario["horizon_minutes"]),
                    "Access_Path": scenario["access_path"],
                    "Synthetic_LM_Flows": len(lm),
                    "Synthetic_LM_Detected_Flows": int(lm["y_pred"].sum()),
                    "Synthetic_LM_Detected": bool(lm["y_pred"].any()) if not lm.empty else False,
                    "First_Operational_LM_Alert": first_lm_alert,
                    "LM_Detection_Delay_Minutes": lm_detection_delay,
                    "Original_Precursor_Detected": not original.empty,
                    "Original_Precursor_Basis": (
                        "corrected_documented_campaign"
                        if "Campaign_GT" in model_base.columns
                        else "binary_attack_label_only"
                    ),
                    "Synthetic_Auth_Precursor_Detected": not auth.empty,
                    "Precursor_Warning": pd.notna(first_precursor),
                    "First_Operational_Precursor": first_precursor,
                    "Operational_Lead_Minutes": lead,
                    "Lead_At_Least_5_Minutes": bool(pd.notna(lead) and lead >= 5),
                }
            )

        for _, control in control_manifest.iterrows():
            part = model_predictions[
                model_predictions["Scenario_ID"].eq(control["scenario_id"])
                & model_predictions["Is_Synthetic"]
                & model_predictions["Synthetic_Role"].eq("synthetic_lm")
            ]
            control_rows.append(
                {
                    "Model": model_name,
                    "Scenario_ID": control["scenario_id"],
                    "Linked_Attack_Scenario": control["linked_attack_scenario"],
                    "Protocol": control["protocol"],
                    "Target_IP": control["target_ip"],
                    "Synthetic_Control_Flows": len(part),
                    "Detected_Control_Flows": int(part["y_pred"].sum()),
                    "Control_Positive_Rate": float(part["y_pred"].mean()) if not part.empty else np.nan,
                }
            )

    return pd.DataFrame(attack_rows), pd.DataFrame(control_rows)


def apply_diagnostic_gate(
    attack_summary: pd.DataFrame,
    control_summary: pd.DataFrame,
    lm_coverage_threshold: float = 0.70,
    warning_coverage_threshold: float = 0.70,
    lead_threshold_minutes: float = 5.0,
    control_advantage_threshold: float = 0.10,
) -> pd.DataFrame:
    """Apply the predeclared robust-diagnostic continuation gate per model."""

    rows: List[dict] = []
    for model_name, part in attack_summary.groupby("Model"):
        controls = control_summary[control_summary["Model"].eq(model_name)]
        linked_reference_ids = set(controls["Linked_Attack_Scenario"])
        linked = part[part["Scenario_ID"].isin(linked_reference_ids)]
        linked_rate = (
            linked["Synthetic_LM_Detected_Flows"].sum() / linked["Synthetic_LM_Flows"].sum()
            if linked["Synthetic_LM_Flows"].sum()
            else np.nan
        )
        control_rate = (
            controls["Detected_Control_Flows"].sum() / controls["Synthetic_Control_Flows"].sum()
            if controls["Synthetic_Control_Flows"].sum()
            else np.nan
        )
        advantage = linked_rate - control_rate if pd.notna(linked_rate) and pd.notna(control_rate) else np.nan

        lm_coverage = float(part["Synthetic_LM_Detected"].mean())
        warning_coverage = float(part["Precursor_Warning"].mean())
        median_lead = float(part["Operational_Lead_Minutes"].median())
        protocol_target_coverage = (
            part[part["Precursor_Warning"]]
            .groupby("Protocol")["Target_IP"]
            .nunique()
        )
        protocols_with_two_targets = int((protocol_target_coverage >= 2).sum())
        graph_or_temporal = model_name != "Simple MLP"
        passes = (
            lm_coverage >= lm_coverage_threshold
            and warning_coverage >= warning_coverage_threshold
            and median_lead >= lead_threshold_minutes
            and protocols_with_two_targets >= 2
            and graph_or_temporal
            and pd.notna(advantage)
            and advantage >= control_advantage_threshold
        )
        rows.append(
            {
                "Model": model_name,
                "LM_Coverage": lm_coverage,
                "Warning_Coverage": warning_coverage,
                "Median_Operational_Lead_Minutes": median_lead,
                "Protocols_With_At_Least_2_Targets": protocols_with_two_targets,
                "Linked_LM_Flow_Positive_Rate": linked_rate,
                "Control_Flow_Positive_Rate": control_rate,
                "Linked_Control_Advantage": advantage,
                "Passes_Robust_Diagnostic_Gate": bool(passes),
            }
        )
    return pd.DataFrame(rows)


def assert_overlay_preserves_base(base_graph, overlay_graph) -> None:
    """Assert that overlay creation only appended nodes/edges and metadata."""

    base_edges = int(base_graph.edge_index.shape[1])
    base_nodes = int(base_graph.x.shape[0])
    checks = {
        "x": torch.equal(base_graph.x, overlay_graph.x[:base_nodes]),
        "edge_index": torch.equal(base_graph.edge_index, overlay_graph.edge_index[:, :base_edges]),
        "edge_attr": torch.equal(base_graph.edge_attr, overlay_graph.edge_attr[:base_edges]),
        "y": torch.equal(base_graph.y, overlay_graph.y[:base_edges]),
        "global_node_ids": torch.equal(
            base_graph.global_node_ids, overlay_graph.global_node_ids[:base_nodes]
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SyntheticLMError(f"Overlay changed original graph tensors: {failed}")
