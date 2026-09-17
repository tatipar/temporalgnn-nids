"""Synthetic checks for canonical cAPTure packet preparation."""

from copy import deepcopy
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.capture_prepare import (
    METADATA_TYPES, load_packet_schema, ordered_feature_names, prepare_scenario,
    required_source_columns, transform_packet_batch, validate_audit_manifest_compatibility,
)
from utils.capture_data import load_manifest


ROOT = Path(__file__).resolve().parents[3]


class CapturePrepareTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.root = Path(self.workspace.name)
        self.schema = load_packet_schema(ROOT / "configs/capture_packet_schema_v1.yaml")
        self.manifest = load_manifest(ROOT / "configs/capture_experiment_v1.yaml")

    def tearDown(self):
        self.workspace.cleanup()

    def create_audit_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({column: [None, None, None]
                              for column in required_source_columns(self.schema)})
        frame["packet_id"] = ["packet-0", "packet-1", "packet-2"]
        frame["source_row_id"] = [0, 1, 2]
        frame["packet_timestamp"] = pd.to_datetime(
            [1.0, 1.5, 2.1], unit="s", utc=True,
        )
        frame["packet_timestamp_ns"] = [1_000_000_000, 1_500_000_000, 2_100_000_000]
        frame["scenario"] = ["train_empty_conn"] * 3
        frame["attack_chain"] = ["empty_conn"] * 3
        frame["benign_source"] = ["normal_2_3_4"] * 3
        frame["author_split"] = ["train"] * 3
        frame["src_endpoint"] = ["00:00:00:00:00:01"] * 3
        frame["dst_endpoint"] = [
            "00:00:00:00:00:02", "33:33:00:00:00:01", "ff:ff:ff:ff:ff:ff",
        ]
        frame["binary_label"] = [0, 1, 0]
        frame["attack_step"] = ["normal", "scan", "normal"]
        frame["phase"] = [None, "RECONNAISSANCE", None]
        frame["sequence_id"] = [None, "1.1", None]
        frame["raw_row_sha256"] = ["a", "b", "c"]
        frame["raw::layers_frame_frame.len"] = [64, 128, 90]
        frame["raw::layers_eth_eth.type"] = ["0x0800", "0x86dd", "0x0806"]
        frame["raw::layers_ip_ip.version"] = ["4", None, None]
        frame["raw::layers_ipv6_ipv6.version"] = [None, "6", None]
        frame["raw::layers_arp_arp.hw.type"] = [None, None, "1"]
        frame["raw::layers_tcp_tcp.len"] = ["0", "20", None]
        frame["raw::layers_tcp_tcp.srcport"] = ["12345", "443", None]
        frame["raw::layers_tcp_tcp.dstport"] = ["1883", "12345", None]
        frame["raw::layers_tcp_tcp.flags"] = ["0x0012", "0x0018", None]
        frame["raw::layers_tcp_tcp.hdr_len"] = ["20", "20", None]
        frame["raw::layers_tcp_tcp.window_size_value"] = ["1024", "2048", None]
        frame["raw::layers_mqtt_mqtt.hdrflags"] = ["0x10", None, None]
        frame["raw::layers_mqtt_mqtt.hdrflags_tree_mqtt.msgtype"] = ["1", None, None]
        return frame

    def test_transform_preserves_protocols_and_special_nodes(self):
        transformed = transform_packet_batch(
            self.create_audit_frame(), self.schema, origin_ns=1_000_000_000,
        )
        self.assertEqual(transformed["dst_node_role"].tolist(), [
            "unicast", "multicast", "broadcast",
        ])
        self.assertEqual(transformed["is_ipv4"].tolist(), [1.0, 0.0, 0.0])
        self.assertEqual(transformed["is_ipv6"].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(transformed["is_arp"].tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(transformed["destination_is_multicast"].tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(transformed["destination_is_broadcast"].tolist(), [0.0, 0.0, 1.0])
        self.assertEqual(transformed["tcp_flag_syn"].tolist()[:2], [1.0, 0.0])
        self.assertEqual(transformed["tcp_flag_ack"].tolist()[:2], [1.0, 1.0])
        self.assertTrue(pd.isna(transformed.loc[2, "tcp_flag_ack"]))
        self.assertEqual(transformed["ethernet_type"].tolist(), [2048.0, 34525.0, 2054.0])
        self.assertEqual(len(ordered_feature_names(self.schema)), 42)
        self.assertFalse(any(column.startswith("window_") for column in transformed))
        self.assertNotIn("base_sample_weight", transformed)

    def test_invalid_endpoint_is_rejected(self):
        frame = self.create_audit_frame()
        frame.loc[0, "src_endpoint"] = "not-a-mac"
        with self.assertRaisesRegex(ValueError, "invalid MAC"):
            transform_packet_batch(frame, self.schema, origin_ns=1_000_000_000)

    def test_nonempty_numeric_parse_failure_is_rejected(self):
        frame = self.create_audit_frame()
        frame.loc[0, "raw::layers_frame_frame.len"] = "invalid"
        with self.assertRaisesRegex(ValueError, "frame_length"):
            transform_packet_batch(frame, self.schema, origin_ns=1_000_000_000)

    def test_disk_backed_preparation_writes_expected_report(self):
        source = self.root / "packets.audit.parquet"
        self.create_audit_frame().to_parquet(source, index=False)
        report = prepare_scenario(
            source, self.root / "prepared", self.schema, "train_empty_conn", batch_size=2,
        )
        self.assertEqual(report["counts"], {
            "packets": 3, "normal_packets": 2, "attack_packets": 1,
        })
        self.assertEqual(report["scenario_origin_timestamp_ns"], 1_000_000_000)
        self.assertEqual(report["last_packet_timestamp_ns"], 2_100_000_000)
        self.assertAlmostEqual(report["duration_seconds"], 1.1)
        prepared = pd.read_parquet(self.root / "prepared" / self.schema["output_artifact"])
        self.assertEqual(len(prepared), 3)
        expected_columns = [
            *METADATA_TYPES,
            *ordered_feature_names(self.schema),
        ]
        self.assertEqual(prepared.columns.tolist(), expected_columns)
        self.assertEqual(prepared["packet_id"].tolist(), ["packet-0", "packet-1", "packet-2"])

    def test_audit_manifest_compatibility_allows_only_downstream_decision_changes(self):
        archived = deepcopy(self.manifest)
        current = deepcopy(archived)
        current["evaluation"]["primary_decision_time"] = "changed_downstream_decision"
        validate_audit_manifest_compatibility(current, archived)
        current["scenarios"]["train_empty_conn"]["expected_filename"] = "different.csv"
        with self.assertRaisesRegex(ValueError, "audited binding"):
            validate_audit_manifest_compatibility(current, archived)


if __name__ == "__main__":
    unittest.main()
