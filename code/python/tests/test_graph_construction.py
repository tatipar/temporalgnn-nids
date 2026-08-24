"""Unit tests for the NF-v3 graph schema and temporal contract."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import pandas as pd

from utils.graph_construction import IpIdMap, endpoint_invalid_reason, feature_preflight_audit, prepare_chunk
from utils.graph_schema import NFV3_EXTENDED, PORTABLE_CORE, destination_port_one_hot, protocol_one_hot


class GraphSchemaTests(unittest.TestCase):
    def test_profile_dimensions_are_frozen(self) -> None:
        self.assertEqual(NFV3_EXTENDED.dimension, 33)
        self.assertEqual(PORTABLE_CORE.dimension, 18)
        self.assertNotIn("TCP_FLAGS", PORTABLE_CORE.numeric_columns)

    def test_every_port_and_protocol_gets_one_category(self) -> None:
        ports = destination_port_one_hot(pd.Series([0, 8088, 8022, 445, 389, 11211, 25, 49152]))
        protocols = protocol_one_hot(pd.Series([6, 17, 1, 2, 99]))
        self.assertTrue((ports.sum(axis=1) == 1).all())
        self.assertTrue((protocols.sum(axis=1) == 1).all())
        self.assertEqual(ports.shape[1], 8)
        self.assertEqual(ports[0].argmax(), 7)

    def test_port_and_protocol_validation_rejects_invalid_values(self) -> None:
        with self.assertRaises(ValueError):
            destination_port_one_hot(pd.Series([80.5]))
        with self.assertRaises(ValueError):
            protocol_one_hot(pd.Series([17.5]))


class TemporalContractTests(unittest.TestCase):
    def test_long_flow_is_assigned_once_at_its_completion_window(self) -> None:
        frame = pd.DataFrame({
            "FLOW_START_MILLISECONDS": [13 * 3_600_000 + 57 * 60_000],
            "FLOW_END_MILLISECONDS": [14 * 3_600_000 + 3 * 60_000 + 40_000],
            "FLOW_DURATION_MILLISECONDS": [400_000],
            "IPV4_SRC_ADDR": ["172.31.69.24"],
            "IPV4_DST_ADDR": ["13.58.225.34"],
        })
        prepared, counts = prepare_chunk(frame)
        self.assertEqual(counts["invalid_endpoint_rows"], 0)
        self.assertEqual(len(prepared), 1)
        self.assertEqual(int(prepared.iloc[0]["window_start_ms"]), 14 * 3_600_000 + 3 * 60_000 + 30_000)
        self.assertEqual(int(prepared.iloc[0]["decision_time_ms"]), 14 * 3_600_000 + 4 * 60_000)

    def test_boundary_flow_uses_the_following_half_open_window(self) -> None:
        frame = pd.DataFrame({
            "FLOW_START_MILLISECONDS": [14 * 3_600_000 + 3 * 60_000 + 30_000],
            "FLOW_END_MILLISECONDS": [14 * 3_600_000 + 4 * 60_000],
            "FLOW_DURATION_MILLISECONDS": [30_000],
            "IPV4_SRC_ADDR": ["172.31.69.24"],
            "IPV4_DST_ADDR": ["13.58.225.34"],
        })
        prepared, _ = prepare_chunk(frame)
        self.assertEqual(int(prepared.iloc[0]["window_start_ms"]), 14 * 3_600_000 + 4 * 60_000)
        self.assertEqual(int(prepared.iloc[0]["decision_time_ms"]), 14 * 3_600_000 + 4 * 60_000 + 30_000)

    def test_recorded_flow_end_is_authoritative_at_a_window_boundary(self) -> None:
        frame = pd.DataFrame({
            "FLOW_START_MILLISECONDS": [29_998],
            "FLOW_END_MILLISECONDS": [29_999],
            "FLOW_DURATION_MILLISECONDS": [2],
            "IPV4_SRC_ADDR": ["192.0.2.1"],
            "IPV4_DST_ADDR": ["192.0.2.2"],
        })
        prepared, _ = prepare_chunk(frame)
        self.assertEqual(float(prepared.iloc[0]["flow_end_ms"]), 29_999)
        self.assertEqual(int(prepared.iloc[0]["decision_time_ms"]), 30_000)

    def test_map_is_append_only_and_bidirectional(self) -> None:
        mapping = IpIdMap()
        self.assertEqual(mapping.id_for("172.31.69.24"), 0)
        self.assertEqual(mapping.id_for("13.58.225.34"), 1)
        self.assertEqual(mapping.id_for("172.31.69.24"), 0)
        self.assertEqual(mapping.id_to_ip[1], "13.58.225.34")

    def test_endpoint_reasons_are_explicit(self) -> None:
        self.assertEqual(endpoint_invalid_reason("0.0.0.0"), "zero_ipv4")
        self.assertEqual(endpoint_invalid_reason("::"), "unspecified_ipv6")
        self.assertEqual(endpoint_invalid_reason("not-an-ip"), "non_parseable")
        self.assertEqual(endpoint_invalid_reason("2001:db8::1"), None)
        self.assertEqual(endpoint_invalid_reason("172.31.69.24"), None)

    def test_ipv6_endpoints_are_canonicalized_and_retained(self) -> None:
        frame = pd.DataFrame({
            "FLOW_START_MILLISECONDS": [0],
            "FLOW_END_MILLISECONDS": [1],
            "FLOW_DURATION_MILLISECONDS": [1],
            "IPV4_SRC_ADDR": ["2001:0DB8:0:0::1"],
            "IPV4_DST_ADDR": ["2001:db8::2"],
        })
        prepared, counts = prepare_chunk(frame)
        self.assertEqual(counts["invalid_endpoint_rows"], 0)
        self.assertEqual(prepared.iloc[0]["source_ip"], "2001:db8::1")
        self.assertEqual(prepared.iloc[0]["destination_ip"], "2001:db8::2")

    def test_preflight_counts_retained_and_excluded_positives(self) -> None:
        frame = pd.DataFrame({
            "source_file": ["day.csv"] * 4,
            "source_row_id": list(range(4)),
            "FLOW_START_MILLISECONDS": [0, 1, 2, 3],
            "FLOW_END_MILLISECONDS": [0, 1, 1, 3],
            "FLOW_DURATION_MILLISECONDS": [0, 0, 0, 0],
            "IPV4_SRC_ADDR": ["192.0.2.1", "0.0.0.0", "192.0.2.1", "192.0.2.1"],
            "IPV4_DST_ADDR": ["192.0.2.2"] * 4,
            "L4_DST_PORT": [80] * 4,
            "PROTOCOL": [6] * 4,
            "binary_target": [0, 1, 1, 1],
            "IN_BYTES": [1] * 4,
            "OUT_BYTES": [1] * 4,
            "IN_PKTS": [1] * 4,
            "OUT_PKTS": [1] * 4,
        })

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "audit.csv"
            frame.to_csv(path, index=False)
            audit = feature_preflight_audit([path], [PORTABLE_CORE], chunksize=2)

        self.assertEqual(audit["status"], "failed")
        self.assertEqual(audit["input_rows"], 4)
        self.assertEqual(audit["positive_rows"], 3)
        self.assertEqual(audit["retained_rows"], 2)
        self.assertEqual(audit["retained_positive_rows"], 1)
        self.assertEqual(audit["excluded_rows"], 2)
        self.assertEqual(audit["excluded_positive_rows"], 2)
        self.assertEqual(audit["invalid_any_endpoint_positive_rows"], 1)
        self.assertEqual(audit["invalid_time_or_duration_positive_rows"], 1)


if __name__ == "__main__":
    unittest.main()
