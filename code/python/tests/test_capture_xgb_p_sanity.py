"""Synthetic checks for cAPTure XGB-P negative and univariate controls."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    import sklearn
except ImportError:
    xgb = None
    sklearn = None

from utils.capture_data import load_manifest, sha256_file
from utils.capture_feature_profile import load_preprocessing_schema
from utils.capture_prepare import load_packet_schema, ordered_feature_names
from utils.capture_preprocess import CaptureFoldPreprocessor
from utils.capture_xgb_p import xgb_p_configuration
from utils.capture_xgb_p_sanity import (
    _sampled_scenario,
    permute_training_labels,
    run_shuffled_label_control,
    run_single_feature_diagnostics,
    single_feature_values,
    summarize_shuffled_label_controls,
)


ROOT = Path(__file__).resolve().parents[3]


class CaptureXgbPSanityTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(ROOT / "configs/capture_experiment_v1.yaml")
        self.packet_schema = load_packet_schema(ROOT / "configs/capture_packet_schema_v1.yaml")
        self.schema = load_preprocessing_schema(
            ROOT / "configs/capture_preprocessing_v1.yaml", self.packet_schema,
        )

    def _frame(self):
        frame = pd.DataFrame({
            name: np.zeros(400, dtype=np.float64)
            for name in ordered_feature_names(self.packet_schema)
        })
        frame["source_row_id"] = np.arange(400)
        frame["binary_label"] = np.arange(400) % 200 >= 100
        frame["is_tcp"] = 1.0
        frame["is_ipv4"] = 1.0
        frame["tcp_source_port"] = 22.0
        frame["tcp_destination_port"] = np.where(frame["binary_label"], 1883.0, 22.0)
        frame["is_mqtt"] = frame["binary_label"].astype(float)
        frame["frame_length"] = np.where(frame["binary_label"], 100.0, 64.0)
        frame["mqtt_qos"] = np.where(frame["binary_label"], 0.0, np.nan)
        return frame

    def test_sample_is_label_independent_and_permutation_preserves_counts(self):
        frame = self._frame()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "packets.parquet"
            frame.to_parquet(path, index=False)
            sampled = _sampled_scenario(path, ["is_mqtt"], 100, 37)
        self.assertEqual(sampled["source_row_id"].tolist(), [0, 100, 200, 300])
        labels = sampled["binary_label"].to_numpy(dtype=np.int8)
        shuffled = permute_training_labels(labels, 142)
        self.assertEqual(np.bincount(shuffled).tolist(), [2, 2])
        self.assertFalse(np.shares_memory(labels, shuffled))

    def test_declared_single_feature_values(self):
        frame = self._frame().iloc[[0, 100]]
        self.assertEqual(
            single_feature_values(frame, "tcp_destination_port_role_mqtt_messaging").tolist(),
            [0.0, 1.0],
        )
        self.assertEqual(single_feature_values(frame, "mqtt_qos_0").tolist(), [0.0, 1.0])
        with self.assertRaisesRegex(ValueError, "Undeclared"):
            single_feature_values(frame, "binary_label")

    def test_context_contract_has_one_current_and_one_historical_block(self):
        context = self.manifest["training"]["temporal_summary_features_and_horizons"]
        self.assertEqual(context["window_width_seconds"], 5)
        self.assertEqual(context["history_window_count"], 6)
        self.assertEqual(context["history_seconds"], 30)
        self.assertEqual(len(context["current_window_features"]), 8)
        self.assertEqual(len(context["history_features"]), 6)
        self.assertEqual(
            len(set(context["current_window_features"] + context["history_features"])), 14,
        )
        self.assertTrue(context["shared_context_inputs_for_xgb_and_graph_models"])
        self.assertEqual(context["endpoint_identity_policy"], "grouping_keys_only_never_model_values")

    @unittest.skipIf(xgb is None or sklearn is None, "XGBoost and scikit-learn are required")
    def test_controls_run_on_frozen_fold_inputs_without_test_data(self):
        frame = self._frame()
        preprocessor = CaptureFoldPreprocessor(self.schema).fit([frame])
        scenarios = self.manifest["gate0"]["modes"]["FULL_DEV"]
        configuration = xgb_p_configuration(self.manifest, "depth5_primary")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            reports = {}
            for scenario in scenarios:
                path = root / f"{scenario}.parquet"
                frame.to_parquet(path, index=False)
                paths[scenario] = path
                reports[scenario] = {
                    "counts": {"packets": 400, "normal_packets": 200, "attack_packets": 200},
                    "output_sha256": sha256_file(path),
                }
            prepared_hashes = {name: reports[name]["output_sha256"] for name in scenarios}
            baselines = {}
            for fold, split in self.manifest["validation"]["folds"].items():
                baseline_dir = root / "baseline" / "depth5_primary" / f"fold_{fold}"
                baseline_dir.mkdir(parents=True)
                (baseline_dir / "fold_report.json").write_text("{}", encoding="utf-8")
                baselines[fold] = {
                    "training_scenarios": split["train"],
                    "validation_scenarios": split["validate"],
                    "prepared_packet_sha256": prepared_hashes,
                    "configuration": configuration,
                    "preprocessor_sha256": "synthetic-preprocessor-hash",
                    "model_sha256": "synthetic-baseline-model-hash",
                    "xgboost_parameters": {
                        "objective": "binary:logistic", "tree_method": "hist",
                        "device": "cpu", "max_depth": 5, "eta": 0.1,
                        "min_child_weight": 20, "lambda": 1.0,
                        "max_bin": 128, "seed": 42, "nthread": 2,
                    },
                    "validation": {
                        name: {"packet_roc_auc": 0.9} for name in split["validate"]
                    },
                }
            with patch(
                "utils.capture_xgb_p_sanity.load_prepared_full_dev",
                return_value=(self.manifest, self.packet_schema, reports, paths),
            ), patch(
                "utils.capture_xgb_p_sanity._load_fold_preprocessor",
                return_value=(preprocessor, "synthetic-preprocessor-hash"),
            ), patch(
                "utils.capture_xgb_p_sanity.validate_xgb_p_fold_run",
                side_effect=lambda directory, fold, configuration_name: baselines[fold],
            ):
                for fold in ("A", "B"):
                    result = run_shuffled_label_control(
                        manifest_path=ROOT / "configs/capture_experiment_v1.yaml",
                        packet_schema_path=ROOT / "configs/capture_packet_schema_v1.yaml",
                        preprocessing_schema_path=ROOT / "configs/capture_preprocessing_v1.yaml",
                        prepared_run_dir=root, preprocessing_audit_dir=root,
                        baseline_run_dir=root / "baseline",
                        output_dir=root / "sanity" / f"fold_{fold}", fold=fold,
                        batch_size=37, nthread=2,
                    )
                    self.assertEqual(result["validation_labels_shuffled"], False)
                    self.assertEqual(set(result["validation"]), set(self.manifest["validation"]["folds"][fold]["validate"]))
                feature_report = run_single_feature_diagnostics(
                    manifest_path=ROOT / "configs/capture_experiment_v1.yaml",
                    packet_schema_path=ROOT / "configs/capture_packet_schema_v1.yaml",
                    prepared_run_dir=root, baseline_run_dir=root / "baseline",
                    output_path=root / "single_features.json", batch_size=37,
                )
            summary = summarize_shuffled_label_controls(root / "sanity")
            self.assertEqual(len(summary["scenario_metrics"]), 5)
            self.assertEqual(len(feature_report["rows"]), 5 * 7)
            self.assertFalse(feature_report["test_data_accessed"])
            self.assertEqual(
                json.loads((root / "single_features.json").read_text(encoding="utf-8"))["features"],
                feature_report["features"],
            )


if __name__ == "__main__":
    unittest.main()
