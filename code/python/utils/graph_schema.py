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
    "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT", "MIN_TTL", "MAX_TTL",
)

NUMERIC_PORTABLE_CORE_COLUMNS = (
    "IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS",
    "FLOW_DURATION_MILLISECONDS",
)

PORT_CATEGORY_COLUMNS = (
    "dst_port_web_http_proxy", "dst_port_admin_remote", "dst_port_windows_smb_rpc",
    "dst_port_infrastructure", "dst_port_database", "dst_port_other_privileged",
    "dst_port_other_high", "dst_port_not_applicable_or_zero",
)

PROTOCOL_CATEGORY_COLUMNS = (
    "protocol_tcp", "protocol_udp", "protocol_icmp", "protocol_igmp",
    "protocol_other",
)

TCP_FLAG_COLUMNS = (
    "tcp_flag_fin", "tcp_flag_syn", "tcp_flag_rst", "tcp_flag_psh",
    "tcp_flag_ack", "tcp_flag_urg", "tcp_flag_ece", "tcp_flag_cwr",
)

TCP_FLAG_MASKS = {
    "tcp_flag_fin": 0x01,
    "tcp_flag_syn": 0x02,
    "tcp_flag_rst": 0x04,
    "tcp_flag_psh": 0x08,
    "tcp_flag_ack": 0x10,
    "tcp_flag_urg": 0x20,
    "tcp_flag_ece": 0x40,
    "tcp_flag_cwr": 0x80,
}

WEB_HTTP_PROXY_PORTS = (80, 81, 443, 3128, 8000, 8008, 8080, 8081, 8082, 8084, 8088, 8090, 8181, 8443, 8545, 8888)
ADMIN_REMOTE_PORTS = (22, 23, 222, 1723, 2222, 2323, 3389, 3390, 3394, 5555, 5900, 5901, 5902, 5903, 5985, 5986, 8022)
WINDOWS_SMB_RPC_PORTS = (135, 137, 138, 139, 445)
INFRASTRUCTURE_PORTS = (53, 67, 68, 88, 123, 161, 389, 464, 500, 514, 546, 547, 636, 1900, 5060, 5353, 5355)
DATABASE_PORTS = (1433, 1434, 1521, 3306, 5432, 6379, 11211, 27017, 9200)

PORT_ROLE_PORTS = {
    "dst_port_web_http_proxy": WEB_HTTP_PROXY_PORTS,
    "dst_port_admin_remote": ADMIN_REMOTE_PORTS,
    "dst_port_windows_smb_rpc": WINDOWS_SMB_RPC_PORTS,
    "dst_port_infrastructure": INFRASTRUCTURE_PORTS,
    "dst_port_database": DATABASE_PORTS,
}

PROTOCOL_ROLE_VALUES = {
    "protocol_tcp": (6,),
    "protocol_udp": (17,),
    "protocol_icmp": (1, 58),
    "protocol_igmp": (2,),
    "protocol_other": "all other valid IANA protocol numbers in 0..255",
}


@dataclass(frozen=True)
class FeatureProfile:
    """Immutable ordered feature contract for one graph collection."""

    name: str
    numeric_columns: tuple[str, ...]
    port_category_columns: tuple[str, ...] = PORT_CATEGORY_COLUMNS
    protocol_category_columns: tuple[str, ...] = PROTOCOL_CATEGORY_COLUMNS
    tcp_flag_columns: tuple[str, ...] = ()
    numeric_transform: str = "log1p_then_standard_scaler"

    @property
    def edge_attr_columns(self) -> tuple[str, ...]:
        return (
            self.numeric_columns
            + self.port_category_columns
            + self.protocol_category_columns
            + self.tcp_flag_columns
        )

    @property
    def dimension(self) -> int:
        return len(self.edge_attr_columns)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["numeric_columns"] = list(self.numeric_columns)
        payload["port_category_columns"] = list(self.port_category_columns)
        payload["protocol_category_columns"] = list(self.protocol_category_columns)
        if self.tcp_flag_columns:
            payload["tcp_flag_columns"] = list(self.tcp_flag_columns)
        else:
            payload.pop("tcp_flag_columns")
        payload["edge_attr_columns"] = list(self.edge_attr_columns)
        payload["dimension"] = self.dimension
        payload["port_encoding"] = {
            "named_roles": {name: list(ports) for name, ports in PORT_ROLE_PORTS.items()},
            "dst_port_other_privileged": "integer ports 1..1023 outside named roles",
            "dst_port_other_high": "integer ports 1024..65535 outside named roles",
            "dst_port_not_applicable_or_zero": "port 0",
        }
        payload["protocol_encoding"] = PROTOCOL_ROLE_VALUES
        if self.tcp_flag_columns:
            payload["tcp_flag_encoding"] = {
                "source_column": "TCP_FLAGS",
                "representation": "multi_hot_bitmask_no_scaling",
                "bit_masks": {name: TCP_FLAG_MASKS[name] for name in self.tcp_flag_columns},
            }
        return payload

    def sha256(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


NFV3_EXTENDED = FeatureProfile(
    name="nfv3_extended",
    numeric_columns=NUMERIC_EXTENDED_COLUMNS,
    tcp_flag_columns=TCP_FLAG_COLUMNS,
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
    if protocol.isna().any() or not np.isfinite(protocol).all():
        raise ValueError("PROTOCOL contains missing, non-numeric, or non-finite values.")
    if (np.floor(protocol) != protocol).any() or (protocol < 0).any() or (protocol > 255).any():
        raise ValueError("PROTOCOL must contain integer IANA protocol numbers in the range 0..255.")
    encoded = np.zeros((len(protocol), len(PROTOCOL_CATEGORY_COLUMNS)), dtype=np.float32)
    encoded[:, 4] = 1.0
    encoded[protocol.eq(6).to_numpy(), :] = (1, 0, 0, 0, 0)
    encoded[protocol.eq(17).to_numpy(), :] = (0, 1, 0, 0, 0)
    encoded[protocol.isin((1, 58)).to_numpy(), :] = (0, 0, 1, 0, 0)
    encoded[protocol.eq(2).to_numpy(), :] = (0, 0, 0, 1, 0)
    return encoded


def destination_port_one_hot(values: pd.Series) -> np.ndarray:
    """Encode destination ports using the fixed eight-role NF-v3 taxonomy."""
    port = pd.to_numeric(values, errors="coerce")
    if port.isna().any() or not np.isfinite(port).all():
        raise ValueError("L4_DST_PORT contains missing, non-numeric, or non-finite values.")
    if (np.floor(port) != port).any() or (port < 0).any() or (port > 65535).any():
        raise ValueError("L4_DST_PORT must contain integer values in the valid range 0..65535.")

    encoded = np.zeros((len(port), len(PORT_CATEGORY_COLUMNS)), dtype=np.float32)
    web = port.isin(WEB_HTTP_PROXY_PORTS)
    admin_remote = port.isin(ADMIN_REMOTE_PORTS)
    windows_smb = port.isin(WINDOWS_SMB_RPC_PORTS)
    infrastructure = port.isin(INFRASTRUCTURE_PORTS)
    database = port.isin(DATABASE_PORTS)
    named_role = web | admin_remote | windows_smb | infrastructure | database
    other_privileged = (port >= 1) & (port < 1024) & ~named_role
    other_high = (port >= 1024) & ~named_role
    not_applicable_or_zero = port.eq(0)

    for index, mask in enumerate((web, admin_remote, windows_smb, infrastructure, database, other_privileged, other_high, not_applicable_or_zero)):
        encoded[mask.to_numpy(), index] = 1.0
    if not (encoded.sum(axis=1) == 1).all():
        raise AssertionError("Destination-port categories must form an exhaustive one-hot partition.")
    return encoded


def tcp_flags_multi_hot(values: pd.Series) -> np.ndarray:
    """Decode an unsigned 8-bit TCP control-bit mask without scaling it."""
    flags = pd.to_numeric(values, errors="coerce")
    if flags.isna().any() or not np.isfinite(flags).all():
        raise ValueError("TCP_FLAGS contains missing, non-numeric, or non-finite values.")
    if (np.floor(flags) != flags).any() or (flags < 0).any() or (flags > 255).any():
        raise ValueError("TCP_FLAGS must contain integer bitmasks in the range 0..255.")
    masks = np.asarray(tuple(TCP_FLAG_MASKS[name] for name in TCP_FLAG_COLUMNS), dtype=np.uint16)
    integer_flags = flags.to_numpy(dtype=np.uint16)
    return ((integer_flags[:, None] & masks[None, :]) != 0).astype(np.float32)


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
