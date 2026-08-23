"""Unit tests for the NF-v3 graph schema and temporal contract."""

from __future__ import annotations

import unittest

import pandas as pd

from utils.graph_construction import IpIdMap, prepare_chunk
from utils.graph_schema import NFV3_EXTENDED, PORTABLE_CORE, destination_port_one_hot, protocol_one_hot


class GraphSchemaTests(unittest.TestCase):
    def test_profile_dimensions_are_frozen(self) -> None:
        self.assertEqual(NFV3_EXTENDED.dimension, 33)
        self.assertEqual(PORTABLE_CORE.dimension, 18)
        self.assertNotIn("TCP_FLAGS", PORTABLE_CORE.numeric_columns)

    def test_every_port_and_protocol_gets_one_category(self) -> None:
        ports = destination_port_one_hot(pd.Series([0, 8081, 5985, 445, 5353, 1521, 25, 49152]))
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
            "FLOW_DURATION_MILLISECONDS": [30_000],
            "IPV4_SRC_ADDR": ["172.31.69.24"],
            "IPV4_DST_ADDR": ["13.58.225.34"],
        })
        prepared, _ = prepare_chunk(frame)
        self.assertEqual(int(prepared.iloc[0]["window_start_ms"]), 14 * 3_600_000 + 4 * 60_000)
        self.assertEqual(int(prepared.iloc[0]["decision_time_ms"]), 14 * 3_600_000 + 4 * 60_000 + 30_000)

    def test_map_is_append_only_and_bidirectional(self) -> None:
        mapping = IpIdMap()
        self.assertEqual(mapping.id_for("172.31.69.24"), 0)
        self.assertEqual(mapping.id_for("13.58.225.34"), 1)
        self.assertEqual(mapping.id_for("172.31.69.24"), 0)
        self.assertEqual(mapping.id_to_ip[1], "13.58.225.34")


if __name__ == "__main__":
    unittest.main()
