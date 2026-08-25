"""Unit tests for the NF-v3 graph schema and temporal contract."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.graph_construction import (
    IpIdMap, atomic_torch_save, audit_graph_file, build_graph,
    encode_edge_attributes, endpoint_invalid_reason, feature_preflight_audit,
    prepare_chunk,
)
from utils.graph_schema import (
    NFV3_EXTENDED, PORTABLE_CORE, destination_port_one_hot, protocol_one_hot,
    tcp_flags_multi_hot,
)


class GraphSchemaTests(unittest.TestCase):
    def test_profile_dimensions_are_frozen(self) -> None:
        self.assertEqual(NFV3_EXTENDED.dimension, 40)
        self.assertEqual(PORTABLE_CORE.dimension, 18)
        self.assertNotIn("TCP_FLAGS", PORTABLE_CORE.numeric_columns)
        self.assertNotIn("TCP_FLAGS", NFV3_EXTENDED.numeric_columns)
        self.assertEqual(len(NFV3_EXTENDED.tcp_flag_columns), 8)

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

    def test_tcp_flags_are_decoded_as_independent_bits(self) -> None:
        flags = tcp_flags_multi_hot(pd.Series([0, 18, 24, 31, 32, 255]))
        self.assertEqual(flags.shape, (6, 8))
        self.assertEqual(flags[1].tolist(), [0, 1, 0, 0, 1, 0, 0, 0])
        self.assertEqual(flags[2].tolist(), [0, 0, 0, 1, 1, 0, 0, 0])
        self.assertEqual(int(flags[3].sum()), 5)
        self.assertEqual(int(flags[4].sum()), 1)
        self.assertTrue((flags[5] == 1).all())

    def test_tcp_flag_validation_rejects_non_bitmask_values(self) -> None:
        for invalid in (-1, 1.5, 256, np.nan):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                tcp_flags_multi_hot(pd.Series([invalid]))

    def test_extended_profile_keeps_tcp_flag_bits_unscaled(self) -> None:
        frame = pd.DataFrame({column: [1.0] for column in NFV3_EXTENDED.numeric_columns})
        frame["L4_DST_PORT"] = 443
        frame["PROTOCOL"] = 6
        frame["TCP_FLAGS"] = 18
        numeric = frame.loc[:, NFV3_EXTENDED.numeric_columns].to_numpy(dtype=np.float64)
        scaler = StandardScaler().fit(np.log1p(numeric))

        encoded = encode_edge_attributes(frame, NFV3_EXTENDED, scaler)

        self.assertEqual(encoded.shape, (1, 40))
        self.assertEqual(encoded[0, -8:].tolist(), [0, 1, 0, 0, 1, 0, 0, 0])


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

    def test_graph_provenance_records_split_window_wait_and_stable_flow_id(self) -> None:
        frame = pd.DataFrame({
            "source_file": ["day.csv"],
            "source_row_id": [7],
            "FLOW_START_MILLISECONDS": [29_998],
            "FLOW_END_MILLISECONDS": [29_999],
            "FLOW_DURATION_MILLISECONDS": [1],
            "IPV4_SRC_ADDR": ["192.0.2.1"],
            "IPV4_DST_ADDR": ["192.0.2.2"],
            "L4_DST_PORT": [80],
            "PROTOCOL": [6],
            "binary_target": [1],
            "IN_BYTES": [1],
            "OUT_BYTES": [1],
            "IN_PKTS": [1],
            "OUT_PKTS": [1],
        })
        prepared, _ = prepare_chunk(frame)
        numeric = prepared.loc[:, PORTABLE_CORE.numeric_columns].to_numpy(dtype=np.float64)
        scaler = StandardScaler().fit(np.log1p(numeric))
        mapping = IpIdMap()
        graph, provenance = build_graph(prepared, PORTABLE_CORE, scaler, mapping, "train")

        self.assertEqual(provenance.iloc[0]["flow_id"], "day.csv:7")
        self.assertEqual(provenance.iloc[0]["split"], "train")
        self.assertEqual(float(provenance.iloc[0]["window_wait_ms"]), 1.0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_path = root / "graph_0000000030000.pt"
            provenance_path = root / "graph_0000000030000.csv"
            atomic_torch_save(graph, graph_path)
            provenance.to_csv(provenance_path, index=False)
            counts, flow_ids, artifact = audit_graph_file(
                graph_path, PORTABLE_CORE, mapping, provenance_path, "train",
            )

        self.assertEqual(counts, {"graphs": 1, "edges": 1, "positive_edges": 1})
        self.assertEqual(flow_ids, ["day.csv:7"])
        self.assertEqual(len(artifact["sha256"]), 64)
        self.assertGreater(artifact["bytes"], 0)


if __name__ == "__main__":
    unittest.main()
