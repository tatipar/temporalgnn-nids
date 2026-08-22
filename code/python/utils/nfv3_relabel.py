"""Versioned correction rules for NF-CSE-CIC-IDS2018-v3 Infiltration.

The rules are transcribed from Liu et al.'s corrected CSE-CIC-IDS2018
documentation, stored in ``docs/improved_cse_cic_ids2018_documentation_infiltration.md``.
They operate on Unix epoch milliseconds so that the implementation is not
affected by Colab's local timezone.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


RULE_VERSION = "cse-cic-ids2018-infiltration-v1"

DROPBOX_2802 = {
    "162.125.3.1", "162.125.3.5", "162.125.3.6", "162.125.248.1",
    "162.125.18.133",
}
DROPBOX_ATTEMPTED_2802 = {
    "104.16.100.29", "104.16.99.29", "52.84.128.3", "52.85.101.236",
    "52.85.131.81", "52.85.95.206",
}
DROPBOX_0103 = {"162.125.3.1", "162.125.3.6", "162.125.248.1", "162.125.18.133"}
DROPBOX_ATTEMPTED_0103 = {"104.16.100.29", "13.32.168.125", "52.85.112.72"}

NMAP_DESTINATIONS_2802 = {
    "172.31.69.1", "172.31.69.4", "172.31.69.5", "172.31.69.6",
    "172.31.69.7", "172.31.69.8", "172.31.69.9", "172.31.69.10",
    "172.31.69.11", "172.31.69.12", "172.31.69.13", "172.31.69.14",
    "172.31.69.15", "172.31.69.16", "172.31.69.17", "172.31.69.18",
    "172.31.69.19", "172.31.69.20", "172.31.69.21", "172.31.69.22",
    "172.31.69.23",
}
NMAP_DESTINATIONS_0103 = {
    "172.31.69.1", "172.31.69.4", "172.31.69.5", "172.31.69.6",
    "172.31.69.7", "172.31.69.8", "172.31.69.9", "172.31.69.10",
    "172.31.69.11", "172.31.69.12", "172.31.69.14", "172.31.69.15",
    "172.31.69.16", "172.31.69.17", "172.31.69.18", "172.31.69.19",
    "172.31.69.20", "172.31.69.21", "172.31.69.22", "172.31.69.23",
    "172.31.69.24",
}


@dataclass(frozen=True)
class Columns:
    time_ms: str = "FLOW_START_MILLISECONDS"
    source_ip: str = "IPV4_SRC_ADDR"
    destination_ip: str = "IPV4_DST_ADDR"
    source_port: str = "L4_SRC_PORT"
    attack: str = "Attack"


def _between(values: pd.Series, intervals: Iterable[tuple[float, float]]) -> pd.Series:
    """Return an inclusive mask for one or more Unix-second intervals."""
    seconds = pd.to_numeric(values, errors="coerce") / 1000.0
    result = pd.Series(False, index=values.index)
    for start, end in intervals:
        result |= seconds.between(start, end, inclusive="both")
    return result


def validate_columns(frame: pd.DataFrame, columns: Columns) -> None:
    required = [columns.time_ms, columns.source_ip, columns.destination_ip,
                columns.source_port, columns.attack]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"NF-v3 CSV lacks required columns: {missing}")


def relabel_chunk(
    frame: pd.DataFrame,
    *,
    columns: Columns = Columns(),
    benign_value: str = "Benign",
    infiltration_value: str = "Infilteration",
    forward_bytes_column: str | None = None,
) -> pd.DataFrame:
    """Apply the Infiltration correction rules to one NF-v3 CSV chunk.

    ``binary_target`` represents the repository's binary task: every original
    non-benign attack remains positive except the old Infilteration label, which
    is rebuilt from documented rules. Attempted flows are benign, following the
    source authors' recommendation. Category 0 can only be detected if the
    caller provides a validated NF-v3 equivalent of CICFlowMeter's
    ``Total Length of Fwd Packets``.
    """
    validate_columns(frame, columns)
    if forward_bytes_column is not None and forward_bytes_column not in frame.columns:
        raise ValueError(f"Configured forward-bytes column is absent: {forward_bytes_column}")

    result = frame.copy()
    attack = result[columns.attack].fillna(benign_value).astype(str).str.strip()
    is_benign = attack.str.casefold().eq(benign_value.casefold())
    is_old_infiltration = attack.str.casefold().eq(infiltration_value.casefold())

    # Preserve other attacks; rebuild only the corrupted historical label.
    result["binary_target"] = (~is_benign & ~is_old_infiltration).astype("int8")
    result["label_corrected_detail"] = attack
    result["correction_rule"] = "unchanged"
    result["attempted_category"] = -1

    src = result[columns.source_ip].astype(str).str.strip()
    dst = result[columns.destination_ip].astype(str).str.strip()
    src_port = pd.to_numeric(result[columns.source_port], errors="coerce")
    time = result[columns.time_ms]

    def apply(mask: pd.Series, detail: str, rule: str, target: int, attempted: int = -1) -> None:
        result.loc[mask, "binary_target"] = target
        result.loc[mask, "label_corrected_detail"] = detail
        result.loc[mask, "correction_rule"] = rule
        result.loc[mask, "attempted_category"] = attempted

    # Clear the historically corrupted Infilteration label before applying its
    # documented replacement rules. This also corrects the documented ARP FPs.
    apply(is_old_infiltration, benign_value, "old_infilteration_to_benign", 0)

    dropbox_2802_time = [(1519828404, 1519829172), (1519839771, 1519839824)]
    dropbox_0103_time = [(1519912390, 1519912760), (1519913032, 1519918454)]
    communication_2802_time = [(1519829140, 1519834135), (1519839839, 1519843200)]
    communication_0103_time = [(1519912674, 1519912745), (1519913075, 1519928245), (1519928295, 1519933041)]

    confirmed_dropbox_2802 = (src == "172.31.69.24") & dst.isin(DROPBOX_2802) & _between(time, dropbox_2802_time)
    confirmed_dropbox_0103 = (src == "172.31.69.13") & dst.isin(DROPBOX_0103) & _between(time, dropbox_0103_time)
    apply(confirmed_dropbox_2802 | confirmed_dropbox_0103, "Infiltration - Dropbox Download", "confirmed_dropbox_download", 1)

    confirmed_comm_2802 = (src == "172.31.69.24") & (dst == "13.58.225.34") & _between(time, communication_2802_time)
    confirmed_comm_0103 = (src == "172.31.69.13") & (dst == "13.58.225.34") & _between(time, communication_0103_time)
    apply(confirmed_comm_2802 | confirmed_comm_0103, "Infiltration - Communication Victim Attacker", "confirmed_victim_attacker_communication", 1)

    nmap_2802 = (src == "172.31.69.24") & dst.isin(NMAP_DESTINATIONS_2802) & src_port.ne(68) & _between(time, [(1519829182, 1519843140.746247)])
    nmap_0103 = (src == "172.31.69.13") & dst.isin(NMAP_DESTINATIONS_0103) & src_port.ne(68) & _between(time, [(1519913388.354333, 1519933092.182726)])
    apply(nmap_2802 | nmap_0103, "Infiltration - NMAP Portscan", "confirmed_nmap_portscan", 1)

    attempted_artifact_2802 = (src == "172.31.69.24") & dst.isin(DROPBOX_ATTEMPTED_2802) & _between(time, dropbox_2802_time)
    attempted_artifact_0103 = (src == "172.31.69.13") & dst.isin(DROPBOX_ATTEMPTED_0103) & _between(time, dropbox_0103_time)
    apply(attempted_artifact_2802 | attempted_artifact_0103, "Benign", "attempted_category_4_to_benign", 0, attempted=4)

    if forward_bytes_column is not None:
        fwd_bytes = pd.to_numeric(result[forward_bytes_column], errors="coerce")
        attempted_zero = fwd_bytes.eq(0) & (confirmed_dropbox_2802 | confirmed_dropbox_0103 | confirmed_comm_2802 | confirmed_comm_0103)
        apply(attempted_zero, "Benign", "attempted_category_0_to_benign", 0, attempted=0)

    return result
