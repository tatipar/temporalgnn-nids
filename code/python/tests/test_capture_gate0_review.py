"""Synthetic checks for post-audit cAPTure Gate-0 decisions."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.capture_gate0_review import (
    audit_topology, build_feature_inventory, ordered_benign_signature,
)


class CaptureGate0ReviewTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.root = Path(self.workspace.name)

    def tearDown(self):
        self.workspace.cleanup()

    def write_packets(self, name: str, attack_value: str = "attack") -> Path:
        frame = pd.DataFrame({
            "packet_timestamp_ns": pd.Series([1, 2, 3], dtype="Int64"),
            "source_row_id": [0, 1, 2],
            "binary_label": pd.Series([0, 0, 1], dtype="Int8"),
            "raw::timestamp": ["1", "2", "3"],
            "raw::label": ["normal", "normal", attack_value],
            "raw::layers_frame_frame.protocols": ["eth:arp", "eth:ipv6", "eth:data"],
            "raw::layers_eth_eth.src": [
                "00:00:00:00:00:01", "00:00:00:00:00:02", "00:00:00:00:00:03",
            ],
            "raw::layers_eth_eth.dst": [
                "ff:ff:ff:ff:ff:ff", "33:33:00:00:00:01", "00:00:00:00:00:03",
            ],
            "raw::layers_ip_ip.version": ["4", "", ""],
            "raw::layers_ipv6_ipv6.version": ["", "6", ""],
        })
        path = self.root / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        return path

    def test_benign_signature_ignores_different_attack_rows(self):
        left = self.write_packets("left", "attack_a")
        right = self.write_packets("right", "attack_b")
        columns = {"raw::timestamp", "raw::label", "raw::layers_eth_eth.src"}
        self.assertEqual(
            ordered_benign_signature(left, columns),
            ordered_benign_signature(right, columns),
        )
        changed = pd.read_parquet(right)
        changed.loc[0, "raw::layers_eth_eth.src"] = "00:00:00:00:00:99"
        changed.to_parquet(right, index=False)
        self.assertNotEqual(
            ordered_benign_signature(left, columns)["sha256_of_ordered_duckdb_row_hashes"],
            ordered_benign_signature(right, columns)["sha256_of_ordered_duckdb_row_hashes"],
        )

    def test_topology_counts_packet_classes(self):
        result = audit_topology(self.write_packets("topology"), "synthetic")
        self.assertEqual(result["invalid_mac_packets"], 0)
        self.assertEqual(result["broadcast_destination_packets"], 1)
        self.assertEqual(result["multicast_destination_packets"], 1)
        self.assertEqual(result["unicast_destination_packets"], 1)
        self.assertEqual(result["ipv4_packets"], 1)
        self.assertEqual(result["ipv6_packets"], 1)
        self.assertEqual(result["non_ip_packets"], 1)
        self.assertEqual(result["self_loop_packets"], 1)

    def test_feature_inventory_separates_metadata_identity_and_candidates(self):
        base_profiles = {
            "label": {"missing": 0, "numeric_values": 0, "constant_nonmissing": False},
            "layers_eth_eth.src": {"missing": 0, "numeric_values": 0, "constant_nonmissing": False},
            "numeric": {"missing": 0, "numeric_values": 3, "constant_nonmissing": False},
            "payload": {"missing": 1, "numeric_values": 0, "constant_nonmissing": False},
        }
        reports = {
            "a": {"column_profiles": base_profiles, "counts": {"packets": 3},
                  "duplicate_column_groups_sha256": []},
            "b": {"column_profiles": base_profiles, "counts": {"packets": 3},
                  "duplicate_column_groups_sha256": []},
        }
        inventory = {item["column"]: item for item in build_feature_inventory(reports)}
        self.assertEqual(inventory["label"]["suggested_action"], "exclude_evaluation_metadata")
        self.assertEqual(inventory["layers_eth_eth.src"]["suggested_action"],
                         "topology_only_exclude_model_feature")
        self.assertEqual(inventory["numeric"]["suggested_action"], "candidate_numeric")
        self.assertEqual(inventory["payload"]["suggested_action"],
                         "exclude_primary_review_for_secondary_ablation")


if __name__ == "__main__":
    unittest.main()
