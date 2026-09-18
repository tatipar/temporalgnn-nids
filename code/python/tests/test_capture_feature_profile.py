"""Synthetic checks for the cAPTure feature-profile contract."""

from __future__ import annotations

from collections import Counter
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.capture_data import load_manifest
from utils.capture_feature_profile import (
    MISSING_CATEGORY,
    build_fold_profiles,
    encode_tcp_port_roles,
    load_preprocessing_schema,
    profile_scenario,
    tcp_port_role_labels,
    validate_protocol_indicators,
)
from utils.capture_prepare import load_packet_schema, ordered_feature_names


ROOT = Path(__file__).resolve().parents[3]


class CaptureFeatureProfileTests(unittest.TestCase):
    def setUp(self):
        self.packet_schema = load_packet_schema(ROOT / "configs/capture_packet_schema_v1.yaml")
        self.schema = load_preprocessing_schema(
            ROOT / "configs/capture_preprocessing_v1.yaml", self.packet_schema,
        )

    def test_semantic_roles_cover_every_canonical_feature_once(self):
        assigned = [
            name for values in self.schema["feature_roles"].values() for name in values
        ]
        self.assertEqual(len(assigned), 42)
        self.assertEqual(set(assigned), set(ordered_feature_names(self.packet_schema)))
        self.assertEqual(self.schema["primary_model_view"]["feature_count_after_port_encoding_only"], 62)

    def test_fixed_port_taxonomy_is_exhaustive_in_both_directions(self):
        ports = pd.Series([1883, 443, 22, 445, 53, 5432, 25, 49151, 49152, 0, None])
        is_tcp = pd.Series([1] * 10 + [0])
        encoded = encode_tcp_port_roles(ports, is_tcp, self.schema)
        labels = tcp_port_role_labels(ports, is_tcp, self.schema)
        self.assertTrue(encoded.sum(axis=1).eq(1).all())
        self.assertEqual(labels.tolist(), [
            "mqtt_messaging", "web_http_proxy", "admin_remote", "windows_smb_rpc",
            "infrastructure", "database", "other_privileged", "other_registered",
            "other_dynamic", "zero_or_reserved", "not_applicable_to_tcp",
        ])

    def test_port_taxonomy_rejects_applicability_and_range_errors(self):
        invalid_cases = [
            (pd.Series([None]), pd.Series([1])),
            (pd.Series([1883]), pd.Series([0])),
            (pd.Series([65536]), pd.Series([1])),
            (pd.Series([1.5]), pd.Series([1])),
        ]
        for ports, indicator in invalid_cases:
            with self.subTest(ports=ports.tolist()), self.assertRaises(ValueError):
                encode_tcp_port_roles(ports, indicator, self.schema)

    def test_protocol_layers_are_multihot_across_layers(self):
        frame = pd.DataFrame({
            "is_arp": [0, 0, 1], "is_ipv4": [1, 0, 0], "is_ipv6": [0, 1, 0],
            "is_tcp": [1, 1, 0], "is_udp": [0, 0, 0],
            "is_mqtt": [1, 0, 0], "is_ssh": [0, 1, 0], "is_malformed": [0, 0, 0],
        })
        counts = validate_protocol_indicators(frame, self.schema)
        self.assertEqual(counts["is_tcp"], 2)
        self.assertEqual(counts["is_mqtt"], 1)
        invalid = frame.copy()
        invalid.loc[0, "is_udp"] = 1
        with self.assertRaisesRegex(ValueError, "transport-layer"):
            validate_protocol_indicators(invalid, self.schema)

    def test_fold_coverage_does_not_refit_on_validation(self):
        manifest = load_manifest(ROOT / "configs/capture_experiment_v1.yaml")
        scenarios = manifest["gate0"]["modes"]["FULL_DEV"]
        feature_statistics = {
            name: {
                "nonnull": 1, "null": 0, "minimum": 0, "maximum": 0,
            }
            for name in ordered_feature_names(self.packet_schema)
        }
        profiles = {
            scenario: {"packets": 1, "feature_statistics": feature_statistics}
            for scenario in scenarios
        }
        categorical = self.schema["feature_roles"]["categorical_code"]
        counters = {
            scenario: {feature: Counter({"0": 1}) for feature in categorical}
            for scenario in scenarios
        }
        counters["train_dollar_char"]["mqtt_version"] = Counter({"5": 1})
        folds = build_fold_profiles(manifest, profiles, counters, self.schema)
        coverage = folds["A"]["categorical_validation_coverage"]["mqtt_version"]
        self.assertEqual(coverage["validation_unseen_values"], ["5"])
        self.assertEqual(coverage["validation_unseen_rows"], 1)
        self.assertFalse(coverage["encoder_refit_on_validation"])

    def test_synthetic_parquet_profile_conserves_rows(self):
        metadata = {
            "packet_id": ["a", "b", "c"],
            "source_row_id": [0, 1, 2],
            "packet_timestamp": pd.to_datetime([1, 2, 3], unit="s", utc=True),
            "packet_timestamp_ns": [1_000_000_000, 2_000_000_000, 3_000_000_000],
            "scenario": ["train_empty_conn"] * 3,
            "attack_chain": ["empty_conn"] * 3,
            "benign_source": ["normal_2_3_4"] * 3,
            "author_split": ["train"] * 3,
            "src_endpoint": ["00:00:00:00:00:01"] * 3,
            "dst_endpoint": ["00:00:00:00:00:02"] * 3,
            "src_node_role": ["unicast"] * 3,
            "dst_node_role": ["unicast"] * 3,
            "binary_label": [0, 1, 0],
            "attack_step": ["normal", "attack", "normal"],
            "phase": [None, "EXPLOIT", None],
            "sequence_id": [None, "1.1", None],
            "raw_row_sha256": ["x", "y", "z"],
        }
        frame = pd.DataFrame(metadata)
        for name in ordered_feature_names(self.packet_schema):
            frame[name] = 0.0
        frame["is_ipv4"] = [1.0, 1.0, 0.0]
        frame["is_arp"] = [0.0, 0.0, 1.0]
        frame["is_tcp"] = [1.0, 1.0, 0.0]
        frame["is_mqtt"] = [1.0, 0.0, 0.0]
        frame["tcp_source_port"] = [12345.0, 1883.0, None]
        frame["tcp_destination_port"] = [1883.0, 12345.0, None]
        for name in self.schema["feature_roles"]["categorical_code"]:
            frame[name] = [0.0, 1.0, None]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "packets.capture_packet_v1.parquet"
            frame.to_parquet(path, index=False)
            report = {"scenario": "train_empty_conn", "counts": {"packets": 3}}
            profile, counters = profile_scenario(
                path, report, self.packet_schema, self.schema, batch_size=2,
            )
        self.assertEqual(profile["packets"], 3)
        self.assertEqual(sum(profile["port_role_counts"]["source"].values()), 3)
        self.assertEqual(counters["mqtt_version"][MISSING_CATEGORY], 1)


if __name__ == "__main__":
    unittest.main()
