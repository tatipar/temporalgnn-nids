"""Feature schemas and deterministic encoders for NF-v3 graph construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd


NUMERIC_EXTENDED_COLUMNS = (
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT",
    "SRC_TO_DST_IAT_AVG", "DST_TO_SRC_IAT_AVG",
    "SRC_TO_DST_IAT_STDDEV", "DST_TO_SRC_IAT_STDDEV",
    "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_PKTS",
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "TCP_FLAGS", "MIN_TTL", "MAX_TTL",
)

NUMERIC_PORTABLE_CORE_COLUMNS = (
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
)

PORT_CATEGORY_COLUMNS = (
    "dst_port_web", "dst_port_admin_remote", "dst_port_windows_smb",
    "dst_port_dns_infrastructure", "dst_port_database",
    "dst_port_other_privileged", "dst_port_other_high",
)

PROTOCOL_CATEGORY_COLUMNS = (
    "protocol_tcp", "protocol_udp", "protocol_icmp", "protocol_igmp",
    "protocol_other",
)


@dataclass(frozen=True)
class FeatureProfile:
    """Immutable ordered feature contract for one graph collection."""

    name: str
    numeric_columns: tuple[str, ...]
    port_category_columns: tuple[str, ...] = PORT_CATEGORY_COLUMNS
    protocol_category_columns: tuple[str, ...] = PROTOCOL_CATEGORY_COLUMNS
    numeric_transform: str = "log1p_then_standard_scaler"

    @property
    def edge_attr_columns(self) -> tuple[str, ...]:
        return self.numeric_columns + self.port_category_columns + self.protocol_category_columns

    @property
    def dimension(self) -> int:
        return len(self.edge_attr_columns)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["numeric_columns"] = list(self.numeric_columns)
        payload["port_category_columns"] = list(self.port_category_columns)
        payload["protocol_category_columns"] = list(self.protocol_category_columns)
        payload["edge_attr_columns"] = list(self.edge_attr_columns)
        payload["dimension"] = self.dimension
        payload["port_encoding"] = "seven_fixed_destination_port_roles"
        payload["protocol_encoding"] = "five_fixed_protocol_roles"
        return payload

    def sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


NFV3_EXTENDED = FeatureProfile(
    name="nfv3_extended",
    numeric_columns=NUMERIC_EXTENDED_COLUMNS,
)

PORTABLE_CORE = FeatureProfile(
    name="portable_core",
    numeric_columns=NUMERIC_PORTABLE_CORE_COLUMNS,
)

FEATURE_PROFILES = {
    NFV3_EXTENDED.name: NFV3_EXTENDED,
    PORTABLE_CORE.name: PORTABLE_CORE,
}


def get_feature_profile(name: str) -> FeatureProfile:
    """Return a registered profile or raise a clear configuration error."""
    try:
        return FEATURE_PROFILES[name]
    except KeyError as error:
        available = ", ".join(sorted(FEATURE_PROFILES))
        raise ValueError(f"Unknown feature profile {name!r}. Available profiles: {available}.") from error


def protocol_one_hot(values: pd.Series) -> np.ndarray:
    """Encode protocol identifiers as TCP, UDP, ICMP, IGMP, or other."""
    protocol = pd.to_numeric(values, errors="coerce")
    encoded = np.zeros((len(protocol), len(PROTOCOL_CATEGORY_COLUMNS)), dtype=np.float32)
    encoded[:, 4] = 1.0
    encoded[protocol.eq(6).to_numpy(), :] = (1, 0, 0, 0, 0)
    encoded[protocol.eq(17).to_numpy(), :] = (0, 1, 0, 0, 0)
    encoded[protocol.isin((1, 58)).to_numpy(), :] = (0, 0, 1, 0, 0)
    encoded[protocol.eq(2).to_numpy(), :] = (0, 0, 0, 1, 0)
    return encoded


def destination_port_one_hot(values: pd.Series) -> np.ndarray:
    """Encode destination ports using the fixed seven-role NF-v3 taxonomy."""
    port = pd.to_numeric(values, errors="coerce")
    if port.isna().any():
        raise ValueError("L4_DST_PORT contains missing or non-numeric values.")
    if (port < 0).any() or (port > 65535).any():
        raise ValueError("L4_DST_PORT contains values outside the valid range 0..65535.")

    encoded = np.zeros((len(port), len(PORT_CATEGORY_COLUMNS)), dtype=np.float32)
    web = port.isin((80, 443, 8080, 8443, 81, 3128, 8545))
    admin_remote = port.isin((22, 222, 2222, 23, 2323, 3389, 3390, 3394, 5900, 5901, 5555, 21, 2131))
    windows_smb = port.isin((445, 135, 137, 138, 139))
    dns_infrastructure = port.isin((53, 5355, 67, 547, 123, 1900, 5060))
    database = port.isin((1433, 3306, 5432, 6379, 27017))
    other_privileged = (port < 1024) & ~(web | admin_remote | windows_smb | dns_infrastructure | database)
    other_high = ~(web | admin_remote | windows_smb | dns_infrastructure | database | other_privileged)

    for index, mask in enumerate((web, admin_remote, windows_smb, dns_infrastructure, database, other_privileged, other_high)):
        encoded[mask.to_numpy(), index] = 1.0
    return encoded


def validate_numeric_frame(frame: pd.DataFrame, profile: FeatureProfile) -> np.ndarray:
    """Validate non-negative finite values before applying log1p and scaling."""
    missing = sorted(set(profile.numeric_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing numeric feature columns for {profile.name}: {missing}")
    numeric = frame.loc[:, profile.numeric_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{profile.name} contains NaN or infinite numeric values.")
    if (numeric < 0).any():
        raise ValueError(f"{profile.name} contains negative values, which are invalid for log1p.")
    return numeric
