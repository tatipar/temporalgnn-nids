"""Synthetic checks for fold-local cAPTure model preprocessing."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from utils.capture_feature_profile import load_preprocessing_schema
from utils.capture_preprocess import (
    CaptureFoldPreprocessor,
    model_feature_names,
    validate_model_preprocessing_schema,
)
from utils.capture_prepare import load_packet_schema, ordered_feature_names


ROOT = Path(__file__).resolve().parents[3]


class CapturePreprocessTests(unittest.TestCase):
    def setUp(self):
        self.packet_schema = load_packet_schema(ROOT / "configs/capture_packet_schema_v1.yaml")
        self.schema = load_preprocessing_schema(
            ROOT / "configs/capture_preprocessing_v1.yaml", self.packet_schema,
        )

    def frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({
            name: pd.Series([0.0, 0.0, 0.0, 0.0], dtype="float64")
            for name in ordered_feature_names(self.packet_schema)
        })
        frame["is_ipv4"] = [1.0, 1.0, 0.0, 1.0]
        frame["is_arp"] = [0.0, 0.0, 1.0, 0.0]
        frame["is_tcp"] = [1.0, 1.0, 0.0, 1.0]
        frame["is_mqtt"] = [1.0, 0.0, 0.0, 1.0]
        frame["tcp_source_port"] = [12345.0, 22.0, np.nan, 1883.0]
        frame["tcp_destination_port"] = [1883.0, 54321.0, np.nan, 12345.0]
        frame["frame_length"] = [64.0, 128.0, 96.0, 256.0]
        frame["ipv4_fragment_offset"] = [0.0, 1.0, np.nan, 0.0]
        frame["ipv4_length"] = [40.0, 104.0, np.nan, 232.0]
        frame["ipv4_ttl"] = [64.0, 63.0, np.nan, 64.0]
        frame["ssh_padding_length"] = [np.nan, 8.0, np.nan, np.nan]
        frame["tcp_header_length"] = [20.0, 32.0, np.nan, 20.0]
        frame["tcp_payload_length"] = [0.0, 64.0, np.nan, 192.0]
        frame["tcp_window_value"] = [1024.0, 2048.0, np.nan, 4096.0]
        frame["ipv6_fragment_more"] = [np.nan, np.nan, np.nan, np.nan]
        frame["ipv4_dscp"] = [0.0, 8.0, np.nan, 46.0]
        frame["mqtt_message_type"] = [3.0, np.nan, np.nan, 8.0]
        frame["mqtt_qos"] = [1.0, np.nan, np.nan, np.nan]
        frame["mqtt_reserved_flag"] = [2.0, np.nan, np.nan, 2.0]
        frame["mqtt_subscription_qos"] = [np.nan, np.nan, np.nan, 0.0]
        frame["ssh_direction"] = [np.nan, 1.0, np.nan, np.nan]
        frame["ethernet_type"] = [2048.0, 2048.0, 2054.0, 2048.0]
        frame["mqtt_version"] = [4.0, np.nan, np.nan, 4.0]
        frame["mqtt_connack_reason_code"] = [np.nan, np.nan, np.nan, 223.0]
        return frame

    def test_reviewed_contract_declares_exact_fixed_model_view(self):
        names = validate_model_preprocessing_schema(self.schema)
        self.assertEqual(names, model_feature_names(self.schema))
        self.assertEqual(len(names), 103)
        self.assertEqual(len(names), len(set(names)))
        for excluded in self.schema["primary_exclusions"]:
            self.assertNotIn(excluded, names)
        self.assertIn("tcp_source_port_role_mqtt_messaging", names)
        self.assertIn("tcp_destination_port_role_not_applicable_to_tcp", names)
        self.assertIn("mqtt_message_type_15", names)
        self.assertIn("ipv4_dscp_bit_5", names)

    def test_fit_transform_is_finite_ordered_and_fixed_width(self):
        frame = self.frame()
        preprocessor = CaptureFoldPreprocessor(self.schema).fit([frame.iloc[:3]])
        transformed = preprocessor.transform(frame)
        self.assertEqual(transformed.columns.tolist(), model_feature_names(self.schema))
        self.assertEqual(transformed.shape, (4, 103))
        self.assertTrue(np.isfinite(transformed.to_numpy()).all())
        self.assertTrue(transformed.loc[2, "tcp_source_port_role_not_applicable_to_tcp"] == 1.0)
        self.assertTrue(transformed.loc[0, "tcp_destination_port_role_mqtt_messaging"] == 1.0)
        self.assertEqual(transformed.loc[1, "ssh_padding_length_present"], 1.0)

    def test_fragment_offset_uses_bounded_log_without_standardization(self):
        frame = self.frame()
        preprocessor = CaptureFoldPreprocessor(self.schema).fit([frame])
        transformed = preprocessor.transform(frame)
        parameters = preprocessor.numeric_parameters["ipv4_fragment_offset"]
        self.assertEqual(parameters["transform"], "log1p_no_scaling")
        self.assertEqual(parameters["centering_value"], 0.0)
        self.assertEqual(parameters["scaling_divisor"], 1.0)
        self.assertAlmostEqual(
            transformed.loc[1, "ipv4_fragment_offset"], np.log1p(1.0), places=6,
        )

    def test_validation_does_not_change_fit_or_activate_masked_columns(self):
        frame = self.frame()
        train = frame.iloc[:2].copy()
        validation = frame.iloc[2:].copy()
        preprocessor = CaptureFoldPreprocessor(self.schema).fit([train])
        artifact_before = preprocessor.to_dict(fold="A", training_scenarios=["train_a"])
        transformed = preprocessor.transform(validation)
        artifact_after = preprocessor.to_dict(fold="A", training_scenarios=["train_a"])
        self.assertEqual(artifact_before, artifact_after)
        self.assertIn("mqtt_message_type_8", preprocessor.masked_features)
        self.assertTrue(transformed["mqtt_message_type_8"].eq(0.0).all())
        self.assertFalse(artifact_after["validation_was_used_for_fit"])

    def test_serialized_preprocessor_round_trip_is_exact(self):
        frame = self.frame()
        fitted = CaptureFoldPreprocessor(self.schema).fit([frame])
        artifact = fitted.to_dict(fold="A", training_scenarios=["train_a"])
        restored = CaptureFoldPreprocessor.from_dict(self.schema, artifact)
        pd.testing.assert_frame_equal(fitted.transform(frame), restored.transform(frame))

    def test_fixed_domains_reject_invalid_values(self):
        frame = self.frame()
        frame.loc[0, "mqtt_message_type"] = 16.0
        with self.assertRaisesRegex(ValueError, "0..15"):
            CaptureFoldPreprocessor(self.schema).fit([frame])


if __name__ == "__main__":
    unittest.main()
