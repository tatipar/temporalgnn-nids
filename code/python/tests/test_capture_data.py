"""Small synthetic checks intended to run in the Colab CPU environment."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from utils.capture_data import (
    AuditSchema, audit_scenario, inspect_csv, load_manifest, selected_scenarios,
    sha256_file, validate_smoke_review, write_json,
)


ROOT = Path(__file__).resolve().parents[3]


class CaptureGate0Tests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory()
        self.root = Path(self.workspace.name)
        self.manifest = load_manifest(ROOT / "configs/capture_experiment_v1.yaml")
        self.schema = AuditSchema(
            timestamp="time", timestamp_unit="s", label="label",
            label_mapping={"normal": 0, "attack": 1}, source_endpoint="src",
            destination_endpoint="dst", attack_step="step", phase="phase", sequence_id="seq",
        )

    def tearDown(self):
        self.workspace.cleanup()

    def create_csv(self, times=(0, 1, 1, 5), labels=("normal", "attack", "attack", "normal")):
        frame = pd.DataFrame({"time": times, "label": labels, "src": ["a"] * 4,
            "dst": ["b"] * 4, "step": ["normal", "scan", "scan", "normal"],
            "phase": ["normal", "recon", "recon", "normal"], "seq": ["0", "1", "1", "0"],
            "constant": ["x"] * 4, "constant_copy": ["x"] * 4})
        path = self.root / "scenario.csv"
        frame.to_csv(path, index=False)
        return path

    def test_half_open_windows_duplicates_and_chunk_boundaries(self):
        report = audit_scenario(self.create_csv(), self.root / "audit", self.manifest,
                                "train_empty_conn", self.schema, chunksize=2)
        self.assertEqual(report["counts"]["packets"], 4)
        self.assertEqual(report["counts"]["attack_packets"], 2)
        self.assertEqual(report["counts"]["duplicate_raw_rows_sha256"], 1)
        self.assertFalse(report["blockers"])
        self.assertIn(["constant", "constant_copy"], report["duplicate_column_groups_sha256"])
        windows = pd.read_parquet(self.root / "audit/windows_1s_offset_0s.parquet").set_index("window_id")
        self.assertEqual(windows.loc[1, "packets"], 2)
        self.assertEqual(windows.loc[5, "packets"], 1)
        packets = pd.read_parquet(self.root / "audit/packets.audit.parquet")
        self.assertTrue(packets.packet_id.is_unique)
        self.assertEqual(packets.source_row_id.tolist(), [0, 1, 2, 3])
        iteration = pd.read_parquet(self.root / "audit/attack_iterations.parquet")
        self.assertEqual(iteration.iloc[0].packets, 2)
        self.assertEqual(iteration.iloc[0].duration_seconds, 0)

    def test_bad_labels_timestamps_and_cross_chunk_inversion_block(self):
        report = audit_scenario(self.create_csv(times=(0, 2, 1, "bad"),
            labels=("normal", "attack", "unknown", "normal")), self.root / "audit",
            self.manifest, "train_empty_conn", self.schema, chunksize=2)
        self.assertEqual(report["timestamp_order_inversions"], 1)
        self.assertIn("unmapped_labels", report["blockers"])
        self.assertIn("invalid_timestamps", report["blockers"])
        self.assertEqual(report["counts"]["packets"], 4)

    def test_held_out_access_rejected_before_reading(self):
        for scenario in ["train_pub_exf", "train_user_prop", "test_empty_conn"]:
            with self.assertRaises(ValueError):
                audit_scenario(self.root / "does_not_exist.csv", self.root / scenario,
                               self.manifest, scenario, self.schema)
        self.manifest["gate0"]["modes"]["SMOKE"].append("test_empty_conn")
        with self.assertRaises(ValueError):
            selected_scenarios(self.manifest, "SMOKE")

    def test_duplicate_headers_rejected(self):
        path = self.root / "duplicate.csv"
        path.write_text("src,src\na,b\n")
        with self.assertRaises(ValueError):
            inspect_csv(path)

    def test_smoke_review_binds_reports_and_manifest(self):
        manifest_hash = "example_manifest_hash"
        review = {"approved": True, "manifest_sha256": manifest_hash,
                  "review_notes": "Synthetic fixture only.", "reports": {}}
        for scenario in selected_scenarios(self.manifest, "SMOKE"):
            path = self.root / f"{scenario}.json"
            write_json(path, {"scenario": scenario, "blockers": [],
                             "manifest_sha256": manifest_hash, "mode": "SMOKE"})
            review["reports"][scenario] = {"path": str(path), "sha256": sha256_file(path)}
        path = self.root / "review.json"
        write_json(path, review)
        validate_smoke_review(path, manifest_hash, self.manifest)
        with self.assertRaises(ValueError):
            validate_smoke_review(path, "different_manifest", self.manifest)
        report_path = Path(next(iter(review["reports"].values()))["path"])
        report_path.write_text("{}")
        with self.assertRaises(ValueError):
            validate_smoke_review(path, manifest_hash, self.manifest)


if __name__ == "__main__":
    unittest.main()
