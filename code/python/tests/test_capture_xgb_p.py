"""Synthetic checks for the cAPTure XGB-P development training protocol."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import sklearn
except ImportError:
    sklearn = None

from utils.capture_data import load_manifest, sha256_file, write_json
from utils.capture_feature_profile import load_preprocessing_schema
from utils.capture_preprocess import CaptureFoldPreprocessor
from utils.capture_prepare import load_packet_schema, ordered_feature_names
from utils.capture_xgb_p import (
    _materialize_training_fold,
    _scenario_metrics,
    _score_validation_scenario,
    run_xgb_p_fold,
    scenario_class_weights,
    summarize_xgb_p_oof,
    validate_xgb_p_fold_run,
    window_coordinates,
    xgb_p_configuration,
)


ROOT = Path(__file__).resolve().parents[3]


class CaptureXgbPTests(unittest.TestCase):
    def setUp(self):
        self.manifest = load_manifest(ROOT / "configs/capture_experiment_v1.yaml")
        packet_schema = load_packet_schema(ROOT / "configs/capture_packet_schema_v1.yaml")
        self.packet_schema = packet_schema
        self.schema = load_preprocessing_schema(
            ROOT / "configs/capture_preprocessing_v1.yaml", packet_schema,
        )

    def _packet_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame({
            name: pd.Series([0.0] * 4, dtype="float64")
            for name in ordered_feature_names(self.packet_schema)
        })
        frame["is_tcp"] = 1.0
        frame["is_ipv4"] = 1.0
        frame["tcp_source_port"] = 1883.0
        frame["tcp_destination_port"] = 22.0
        frame["packet_id"] = [f"p{index}" for index in range(4)]
        frame["source_row_id"] = [0, 1, 2, 3]
        frame["packet_timestamp_ns"] = [100, 101, 5_000_000_100, 5_000_000_101]
        frame["binary_label"] = [0, 0, 1, 1]
        frame["attack_step"] = [None, None, "attack", "attack"]
        frame["phase"] = [None, None, "EXPLOIT", "EXPLOIT"]
        frame["sequence_id"] = [None, None, "1.1", "1.1"]
        return frame

    def test_declared_configurations_keep_all_but_depth_fixed(self):
        primary = xgb_p_configuration(self.manifest, "depth5_primary")
        sensitivity = xgb_p_configuration(self.manifest, "depth10_sensitivity")
        self.assertEqual(primary["max_depth"], 5)
        self.assertEqual(sensitivity["max_depth"], 10)
        self.assertEqual(
            {key: value for key, value in primary.items() if key != "max_depth"},
            {key: value for key, value in sensitivity.items() if key != "max_depth"},
        )
        self.assertEqual(primary["num_boost_round"], 200)
        with self.assertRaisesRegex(ValueError, "Undeclared"):
            xgb_p_configuration(self.manifest, "depth18")

    def test_equal_scenario_class_weight_mass_and_unit_mean(self):
        reports = {
            "one": {"counts": {"packets": 10, "normal_packets": 8, "attack_packets": 2}},
            "two": {"counts": {"packets": 30, "normal_packets": 10, "attack_packets": 20}},
        }
        weights = scenario_class_weights(["one", "two"], reports)
        for scenario in reports:
            counts = reports[scenario]["counts"]
            self.assertAlmostEqual(weights[scenario][0] * counts["normal_packets"], 10)
            self.assertAlmostEqual(weights[scenario][1] * counts["attack_packets"], 10)
        self.assertAlmostEqual(sum(
            weights[scenario][label] * reports[scenario]["counts"][name]
            for scenario in reports
            for label, name in ((0, "normal_packets"), (1, "attack_packets"))
        ), 40)
        reports["two"]["counts"]["attack_packets"] = 0
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            scenario_class_weights(["one", "two"], reports)

    def test_windows_are_half_open_and_origin_relative(self):
        origin = 19_792_000
        timestamps = np.array([
            origin, origin + 5_000_000_000 - 1,
            origin + 5_000_000_000, origin + 12_000_000_000,
        ], dtype=np.int64)
        indexes, ends = window_coordinates(timestamps, origin, 5)
        self.assertEqual(indexes.tolist(), [0, 0, 1, 2])
        self.assertEqual(ends.tolist(), [
            origin + 5_000_000_000, origin + 5_000_000_000,
            origin + 10_000_000_000, origin + 15_000_000_000,
        ])
        with self.assertRaisesRegex(ValueError, "precedes"):
            window_coordinates(np.array([origin - 1]), origin, 5)

    def test_training_matrix_and_oof_batches_preserve_rows_and_nullable_annotations(self):
        frame = self._packet_frame()
        preprocessor = CaptureFoldPreprocessor(self.schema).fit([frame])
        report = {
            "counts": {"packets": 4, "normal_packets": 2, "attack_packets": 2},
            "scenario_origin_timestamp_ns": 100,
        }
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            path = directory / "packets.parquet"
            frame.to_parquet(path, index=False)
            features, labels, weights, observed = _materialize_training_fold(
                directory, ["scenario"], {"scenario": path}, {"scenario": report},
                preprocessor, {"scenario": {0: 1.0, 1: 1.0}}, 2,
            )
            self.assertEqual(features.shape, (4, 103))
            self.assertEqual(labels.tolist(), [0, 0, 1, 1])
            self.assertEqual(weights.tolist(), [1.0] * 4)
            self.assertEqual(observed["scenario"]["rows"], 4)
            if xgb is not None:
                matrix = xgb.QuantileDMatrix(
                    features, label=labels, weight=weights,
                    feature_names=preprocessor.feature_names, max_bin=128, nthread=2,
                )
                booster = xgb.train(
                    {"objective": "binary:logistic", "tree_method": "hist",
                     "max_depth": 2, "min_child_weight": 0, "max_bin": 128,
                     "seed": 42, "nthread": 2},
                    matrix, num_boost_round=2,
                )
                self.assertEqual(booster.num_boosted_rounds(), 2)
                self.assertTrue(np.isfinite(booster.inplace_predict(np.asarray(features))).all())

            class FakeModel:
                offset = 0

                def inplace_predict(self, values):
                    all_scores = np.array([0.1, 0.2, 0.8, 0.9], dtype=np.float32)
                    part = all_scores[self.offset:self.offset + len(values)]
                    self.offset += len(values)
                    return part

            output = directory / "oof.parquet"
            saved = _score_validation_scenario(
                model=FakeModel(), scenario="scenario", packet_path=path,
                report=report, preprocessor=preprocessor, output_path=output,
                width_seconds=5, batch_size=2,
            )
            self.assertEqual(saved["rows"], 4)
            table = pq.read_table(output)
            self.assertEqual(table.column("window_index").to_pylist(), [0, 0, 1, 1])
            self.assertEqual(table.column("sequence_id").to_pylist(), [None, None, "1.1", "1.1"])
            if sklearn is not None:
                self.assertAlmostEqual(_scenario_metrics(output)["packet_roc_auc"], 1.0)

    def test_summary_uses_scenario_then_fold_macro_and_checks_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = "depth5_primary"
            fold_values = {
                "A": {"s3": 0.8, "s4": 0.6, "s5": 0.7},
                "B": {"s1": 0.9, "s2": 0.7},
            }
            for fold, scores in fold_values.items():
                directory = root / config / f"fold_{fold}"
                directory.mkdir(parents=True)
                model = directory / "model.json"
                model.write_text("{}", encoding="utf-8")
                validation = {}
                for scenario, value in scores.items():
                    oof = directory / f"oof_{scenario}.parquet"
                    oof.write_bytes(b"synthetic")
                    validation[scenario] = {
                        "rows": 2, "packet_roc_auc": value,
                        "packet_pr_auc_diagnostic": 0.5,
                        "oof_artifact": oof.name, "oof_sha256": sha256_file(oof),
                    }
                report = {
                    "fold": fold, "configuration_name": config,
                    "configuration": {"max_depth": 5},
                    "manifest_sha256": "shared-manifest",
                    "model_artifact": model.name, "model_sha256": sha256_file(model),
                    "validation": validation,
                    "fold_macro_packet_roc_auc": sum(scores.values()) / len(scores),
                    "fold_macro_packet_pr_auc_diagnostic": 0.5,
                }
                report_path = directory / "fold_report.json"
                write_json(report_path, report)
                write_json(directory / "run_status.json", {
                    "complete": True, "report_sha256": sha256_file(report_path),
                })
            summary = summarize_xgb_p_oof(root, config)
            self.assertAlmostEqual(summary["fold_packet_roc_auc"]["A"], 0.7)
            self.assertAlmostEqual(summary["fold_packet_roc_auc"]["B"], 0.8)
            self.assertAlmostEqual(summary["hierarchical_macro_oof_packet_roc_auc"], 0.75)
            self.assertEqual(set(summary["scenario_metrics"]), set("s1 s2 s3 s4 s5".split()))
            model = root / config / "fold_A" / "model.json"
            model.write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "model changed"):
                validate_xgb_p_fold_run(model.parent, "A", config)

    @unittest.skipIf(xgb is None or sklearn is None, "XGBoost and scikit-learn are required")
    def test_one_fold_runner_writes_verified_model_and_oof_artifacts(self):
        frame = self._packet_frame()
        scenarios = self.manifest["gate0"]["modes"]["FULL_DEV"]
        train_scenarios = self.manifest["validation"]["folds"]["A"]["train"]
        preprocessor = CaptureFoldPreprocessor(self.schema).fit(
            [frame for _ in train_scenarios]
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            reports = {}
            for scenario in scenarios:
                paths[scenario] = root / f"{scenario}.parquet"
                frame.to_parquet(paths[scenario], index=False)
                reports[scenario] = {
                    "counts": {"packets": 4, "normal_packets": 2, "attack_packets": 2},
                    "scenario_origin_timestamp_ns": 100,
                    "output_sha256": sha256_file(paths[scenario]),
                }
            output = root / "output" / "depth5_primary" / "fold_A"
            with patch(
                "utils.capture_xgb_p.load_prepared_full_dev",
                return_value=(self.manifest, self.packet_schema, reports, paths),
            ), patch(
                "utils.capture_xgb_p._load_fold_preprocessor",
                return_value=(preprocessor, "synthetic-preprocessor-hash"),
            ):
                report = run_xgb_p_fold(
                    manifest_path=ROOT / "configs/capture_experiment_v1.yaml",
                    packet_schema_path=ROOT / "configs/capture_packet_schema_v1.yaml",
                    preprocessing_schema_path=ROOT / "configs/capture_preprocessing_v1.yaml",
                    prepared_run_dir=root, preprocessing_audit_dir=root,
                    output_dir=output, local_work_root=root,
                    fold="A", configuration_name="depth5_primary",
                    batch_size=2, nthread=2,
                )
            self.assertEqual(report["training_rows"], 8)
            self.assertEqual(set(report["validation"]), set(self.manifest["validation"]["folds"]["A"]["validate"]))
            self.assertTrue(report["thresholds_selected"] is False)
            self.assertEqual(
                validate_xgb_p_fold_run(output, "A", "depth5_primary"), report,
            )
            self.assertEqual(list(root.glob("capture_xgb_p_*")), [])


if __name__ == "__main__":
    unittest.main()
