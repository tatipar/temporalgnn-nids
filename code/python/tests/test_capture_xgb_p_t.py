"""Synthetic checks for causal cAPTure XGB-P+T context and row alignment."""

import tempfile
import unittest
from copy import deepcopy
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

try:
    import xgboost
except ImportError:
    xgboost = None

from utils.capture_data import load_manifest, sha256_file
from utils.capture_feature_profile import load_preprocessing_schema
from utils.capture_prepare import load_packet_schema, ordered_feature_names
from utils.capture_preprocess import CaptureFoldPreprocessor

from utils.capture_xgb_p_t import (
    CONTEXT_COLUMNS, _paired_batches, build_context_scenario,
    context_for_window, fit_context_scaler, transform_context,
    build_context_run, run_xgb_p_t_fold, validate_xgb_p_t_fold_run,
)


ROOT = Path(__file__).resolve().parents[3]


class CaptureXgbPtTests(unittest.TestCase):
    def _frame(self):
        return pd.DataFrame({
            "source_row_id": [0, 1, 2, 3, 4, 5],
            "packet_timestamp_ns": [0, 1, 5_000_000_000, 6_000_000_000,
                                    40_000_000_000, 40_000_000_001],
            "src_endpoint": ["a", "a", "a", "b", "a", "c"],
            "dst_endpoint": ["b", "c", "b", "a", "b", "a"],
            "is_mqtt": [1, 0, 1, 0, 0, 0],
            "frame_length": [100, 200, 300, 400, 500, 600],
        })

    def test_current_window_history_gap_and_scenario_reset(self):
        frame = self._frame()
        history = deque()
        first = context_for_window(frame.iloc[:2], history, 0)
        second = context_for_window(frame.iloc[2:4], history, 1)
        last = context_for_window(frame.iloc[4:], history, 8)
        self.assertEqual(first["window_packet_count"].tolist(), [2, 2])
        self.assertEqual(first["source_unique_destinations"].tolist(), [2, 2])
        self.assertEqual(second["previous_30s_source_unique_destinations"].tolist(), [2, 0])
        self.assertEqual(second["previous_30s_directed_pair_packet_count"].tolist(), [1, 0])
        self.assertEqual(second["window_mean_frame_length"].tolist(), [350, 350])
        self.assertEqual(last["previous_30s_packet_count"].tolist(), [0, 0])
        reset = context_for_window(frame.iloc[:2], deque(), 0)
        self.assertEqual(reset["previous_30s_packet_count"].tolist(), [0, 0])

    def test_history_counts_distinct_peers_across_the_union(self):
        frame = self._frame()
        history = deque()
        context_for_window(frame.iloc[[0]], history, 0)
        context_for_window(frame.iloc[[0, 1]], history, 1)
        observed = context_for_window(frame.iloc[[0]], history, 2)
        self.assertEqual(observed["previous_30s_packet_count"].tolist(), [3])
        self.assertEqual(observed["previous_30s_directed_pair_packet_count"].tolist(), [2])
        self.assertEqual(observed["previous_30s_source_unique_destinations"].tolist(), [2])

    def test_streaming_artifact_preserves_rows_across_batch_boundaries(self):
        frame = self._frame()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet_path = root / "packets.parquet"
            context_path = root / "context.parquet"
            frame.to_parquet(packet_path, index=False, row_group_size=3)
            report = {"scenario_origin_timestamp_ns": 0,
                      "counts": {"packets": len(frame)}}
            result = build_context_scenario(
                packet_path, context_path, scenario="synthetic",
                report=report, batch_size=3)
            self.assertEqual(result["rows"], 6)
            self.assertEqual(result["nonempty_windows"], 3)
            context = pq.read_table(context_path).to_pandas()
            self.assertEqual(context["source_row_id"].tolist(), list(range(6)))
            self.assertEqual(context["previous_30s_packet_count"].tolist(),
                             [0, 0, 2, 2, 0, 0])
            paired = list(_paired_batches(
                packet_path, context_path, ["source_row_id", "frame_length"], 2))
            self.assertEqual(sum(len(packet) for packet, _ in paired), 6)
            scaler = fit_context_scaler([context_path], batch_size=2)
            self.assertEqual(scaler["training_rows"], 6)
            self.assertEqual(scaler["feature_names"], list(CONTEXT_COLUMNS))
            transformed = transform_context(context, scaler)
            self.assertEqual(transformed.shape, (6, 14))
            self.assertTrue(np.isfinite(transformed).all())

    def test_context_rejects_null_identity_and_unordered_packets(self):
        frame = self._frame()
        frame.loc[0, "src_endpoint"] = None
        with self.assertRaisesRegex(ValueError, "null"):
            context_for_window(frame.iloc[:2], deque(), 0)
        frame = self._frame()
        frame.loc[3, "packet_timestamp_ns"] = -1
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            packet_path = root / "packets.parquet"
            frame.to_parquet(packet_path, index=False)
            with self.assertRaisesRegex(ValueError, "time order"):
                build_context_scenario(packet_path, root / "context.parquet",
                                       scenario="synthetic",
                                       report={"scenario_origin_timestamp_ns": 0,
                                               "counts": {"packets": 6}},
                                       batch_size=2)

    @unittest.skipIf(xgboost is None, "XGBoost is required")
    def test_fold_runner_writes_verified_117_column_oof_artifacts(self):
        manifest = deepcopy(load_manifest(ROOT / "configs/capture_experiment_v1.yaml"))
        manifest["readiness"]["xgb_p_sanity_gate_passed"] = False
        packet_schema = load_packet_schema(ROOT / "configs/capture_packet_schema_v1.yaml")
        preprocessing_schema = load_preprocessing_schema(
            ROOT / "configs/capture_preprocessing_v1.yaml", packet_schema)
        packet_frame = pd.DataFrame({
            name: pd.Series([0.0] * 4, dtype="float64")
            for name in ordered_feature_names(packet_schema)
        })
        packet_frame["is_tcp"] = 1.0
        packet_frame["is_ipv4"] = 1.0
        packet_frame["tcp_source_port"] = 1883.0
        packet_frame["tcp_destination_port"] = 22.0
        packet_frame["packet_id"] = [f"packet_{index}" for index in range(4)]
        packet_frame["source_row_id"] = [0, 1, 2, 3]
        packet_frame["packet_timestamp_ns"] = [100, 101, 5_000_000_100, 5_000_000_101]
        packet_frame["src_endpoint"] = ["a", "a", "b", "b"]
        packet_frame["dst_endpoint"] = ["b", "c", "a", "c"]
        packet_frame["frame_length"] = [100, 200, 300, 400]
        packet_frame["is_mqtt"] = [1, 0, 1, 0]
        packet_frame["binary_label"] = [0, 0, 1, 1]
        packet_frame["attack_step"] = [None, None, "attack", "attack"]
        packet_frame["phase"] = [None, None, "EXPLOIT", "EXPLOIT"]
        packet_frame["sequence_id"] = [None, None, "1.1", "1.1"]
        train_scenarios = manifest["validation"]["folds"]["A"]["train"]
        preprocessor = CaptureFoldPreprocessor(preprocessing_schema).fit(
            [packet_frame for _ in train_scenarios])
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = {}
            reports = {}
            for scenario in manifest["gate0"]["modes"]["FULL_DEV"]:
                paths[scenario] = root / f"{scenario}.parquet"
                packet_frame.to_parquet(paths[scenario], index=False)
                reports[scenario] = {
                    "counts": {"packets": 4, "normal_packets": 2, "attack_packets": 2},
                    "scenario_origin_timestamp_ns": 100,
                    "output_sha256": sha256_file(paths[scenario]),
                }
            prepared = (manifest, packet_schema, reports, paths)
            context_dir = root / "context"
            with patch("utils.capture_xgb_p_t.load_prepared_full_dev", return_value=prepared):
                build_context_run(
                    manifest_path=ROOT / "configs/capture_experiment_v1.yaml",
                    packet_schema_path=ROOT / "configs/capture_packet_schema_v1.yaml",
                    prepared_run_dir=root, output_dir=context_dir, batch_size=2)
                output = root / "output" / "depth5_primary" / "fold_A"
                with self.assertRaisesRegex(ValueError, "sanity gate"):
                    run_xgb_p_t_fold(
                        manifest_path=ROOT / "configs/capture_experiment_v1.yaml",
                        packet_schema_path=ROOT / "configs/capture_packet_schema_v1.yaml",
                        preprocessing_schema_path=ROOT / "configs/capture_preprocessing_v1.yaml",
                        prepared_run_dir=root, preprocessing_audit_dir=root,
                        context_dir=context_dir, output_dir=output,
                        local_work_root=root, fold="A", batch_size=2, nthread=2)
                manifest["readiness"]["xgb_p_sanity_gate_passed"] = True
                with patch("utils.capture_xgb_p_t._load_fold_preprocessor",
                           return_value=(preprocessor, "synthetic-preprocessor-hash")):
                    report = run_xgb_p_t_fold(
                        manifest_path=ROOT / "configs/capture_experiment_v1.yaml",
                        packet_schema_path=ROOT / "configs/capture_packet_schema_v1.yaml",
                        preprocessing_schema_path=ROOT / "configs/capture_preprocessing_v1.yaml",
                        prepared_run_dir=root, preprocessing_audit_dir=root,
                        context_dir=context_dir, output_dir=output,
                        local_work_root=root, fold="A", batch_size=2, nthread=2)
            self.assertEqual(report["feature_count"], 117)
            self.assertEqual(report["training_rows"], 8)
            self.assertEqual(validate_xgb_p_t_fold_run(output, "A"), report)
            self.assertEqual(len(report["validation"]), 3)
            self.assertFalse(report["thresholds_selected"])


if __name__ == "__main__":
    unittest.main()
