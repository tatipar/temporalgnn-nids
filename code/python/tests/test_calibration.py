import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from utils.calibration import (  # noqa: E402
    build_calibration_manifest,
    candidate_weight_bias_pairs,
    canonical_json_bytes,
    collect_aligned_training_targets,
    output_bias,
    write_calibration_manifest,
)


class FakeGraph:
    def __init__(self, targets):
        self.y = targets


class FakeDataset:
    def __init__(self, profile, windows, *, split="train", timestamps=None):
        self.profile = profile
        self.split = split
        self.graphs = [FakeGraph(targets) for targets in windows]
        self.timestamps = timestamps or [30_000 * (index + 1) for index in range(len(windows))]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, index):
        return self.graphs[index]


def aligned_datasets(windows):
    return {
        "nfv3_extended": FakeDataset("nfv3_extended", windows),
        "portable_core": FakeDataset("portable_core", windows),
    }


def sample_manifest():
    counts, alignment = collect_aligned_training_targets(
        aligned_datasets([[0, 1, 0], [1, 0]])
    )
    return build_calibration_manifest(
        graph_manifest_sha256="a" * 64,
        feature_schemas={
            "nfv3_extended": {
                "schema_definition_sha256": "b" * 64,
                "schema_file_sha256": "c" * 64,
                "graph_collection_sha256": "d" * 64,
                "edge_dim": 40,
            },
            "portable_core": {
                "schema_definition_sha256": "e" * 64,
                "schema_file_sha256": "f" * 64,
                "graph_collection_sha256": "0" * 64,
                "edge_dim": 18,
            },
        },
        corrected_manifest_sha256="1" * 64,
        corrected_data_sha256="2" * 64,
        correction_rule_version="cse-cic-ids2018-infiltration-v1",
        code_revision="3" * 40,
        counts=counts,
        profile_alignment=alignment,
    )


class CalibrationFormulaTests(unittest.TestCase):
    def test_output_bias_matches_both_documented_formulas(self):
        negative = 80
        positive = 20
        weight = 2.0
        prevalence = positive / (negative + positive)

        actual = output_bias(weight, negative, positive)

        self.assertAlmostEqual(actual, math.log(weight * positive / negative))
        self.assertAlmostEqual(
            actual, math.log(weight * prevalence / (1.0 - prevalence))
        )

    def test_candidate_grid_uses_predeclared_order_and_matching_biases(self):
        candidates = candidate_weight_bias_pairs(100, 4)

        self.assertEqual(
            [item["pos_weight"] for item in candidates],
            [1.0, 2.0, 5.0, 12.5, 25.0],
        )
        for item in candidates:
            self.assertAlmostEqual(
                item["output_bias_init"],
                math.log(item["pos_weight"] * 4 / 100),
            )

    def test_candidate_deduplication_retains_all_equivalent_anchors(self):
        candidates = candidate_weight_bias_pairs(16, 4)

        self.assertEqual([item["pos_weight"] for item in candidates], [1.0, 2.0, 4.0])
        self.assertEqual(candidates[1]["anchors"], ["2", "sqrt(R)", "R/2"])
        self.assertEqual(
            [item["candidate_id"] for item in candidates],
            ["calibration_01", "calibration_02", "calibration_03"],
        )

    def test_candidate_deduplication_handles_near_duplicates(self):
        candidates = candidate_weight_bias_pairs(
            4_000_000_000_001,
            1_000_000_000_000,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )

        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[1]["anchors"], ["2", "sqrt(R)", "R/2"])

    def test_empty_classes_are_rejected(self):
        for negative, positive in ((0, 4), (4, 0)):
            with self.subTest(negative=negative, positive=positive):
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    candidate_weight_bias_pairs(negative, positive)


class TargetAlignmentTests(unittest.TestCase):
    def test_counts_train_targets_and_records_equal_digests(self):
        counts, alignment = collect_aligned_training_targets(
            aligned_datasets([[0, 1, 0], [1, 0]])
        )

        self.assertEqual(counts["total_flows"], 5)
        self.assertEqual(counts["negative_flows"], 3)
        self.assertEqual(counts["positive_flows"], 2)
        self.assertAlmostEqual(counts["positive_prevalence"], 0.4)
        self.assertAlmostEqual(counts["class_ratio_negative_to_positive"], 1.5)
        hashes = {
            record["target_sha256"]
            for record in alignment["profiles"].values()
        }
        self.assertEqual(len(hashes), 1)
        self.assertEqual(alignment["status"], "passed")

    def test_invalid_labels_are_rejected(self):
        for invalid in (0.5, 2, -1, float("nan"), float("inf"), "1"):
            with self.subTest(invalid=invalid):
                datasets = aligned_datasets([[0, invalid, 1]])
                with self.assertRaisesRegex(ValueError, "Invalid target"):
                    collect_aligned_training_targets(datasets)

    def test_empty_positive_or_negative_profile_is_rejected(self):
        for windows in ([[0, 0]], [[1, 1]]):
            with self.subTest(windows=windows):
                with self.assertRaisesRegex(ValueError, "non-empty"):
                    collect_aligned_training_targets(aligned_datasets(windows))

    def test_profile_target_disagreement_is_rejected(self):
        datasets = {
            "nfv3_extended": FakeDataset("nfv3_extended", [[0, 1, 0]]),
            "portable_core": FakeDataset("portable_core", [[0, 0, 0]]),
        }

        with self.assertRaisesRegex(ValueError, "disagree on targets"):
            collect_aligned_training_targets(datasets)

    def test_profile_window_disagreement_is_rejected(self):
        datasets = {
            "nfv3_extended": FakeDataset(
                "nfv3_extended", [[0, 1]], timestamps=[30_000]
            ),
            "portable_core": FakeDataset(
                "portable_core", [[0, 1]], timestamps=[60_000]
            ),
        }

        with self.assertRaisesRegex(ValueError, "different train windows"):
            collect_aligned_training_targets(datasets)

    def test_non_training_split_is_rejected(self):
        datasets = aligned_datasets([[0, 1]])
        datasets["portable_core"].split = "val"

        with self.assertRaisesRegex(ValueError, "only the train split"):
            collect_aligned_training_targets(datasets)


class DeterministicManifestTests(unittest.TestCase):
    def test_serialization_is_deterministic_and_has_no_timestamp(self):
        first = sample_manifest()
        second = sample_manifest()

        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertNotIn("created_at", first)
        self.assertEqual(
            hashlib.sha256(canonical_json_bytes(first)).hexdigest(),
            hashlib.sha256(canonical_json_bytes(second)).hexdigest(),
        )

    def test_manifest_contains_required_provenance_formulas_and_pairs(self):
        manifest = sample_manifest()

        self.assertEqual(manifest["scope"]["split"], "train")
        self.assertEqual(
            manifest["source_artifacts"]["correction_rule_version"],
            "cse-cic-ids2018-infiltration-v1",
        )
        self.assertIn("graph_manifest_sha256", manifest["source_artifacts"])
        self.assertEqual(
            set(manifest["source_artifacts"]["feature_schemas"]),
            {"nfv3_extended", "portable_core"},
        )
        self.assertEqual(
            manifest["candidate_grid"]["declaration"],
            ["1", "2", "sqrt(R)", "R/2", "R"],
        )
        self.assertTrue(
            all(
                "pos_weight" in item and "output_bias_init" in item
                for item in manifest["candidates"]
            )
        )

    def test_identical_rerun_is_unchanged_and_difference_requires_overwrite(self):
        manifest = sample_manifest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibration_manifest.json"

            self.assertEqual(write_calibration_manifest(manifest, path), "written")
            self.assertEqual(write_calibration_manifest(manifest, path), "unchanged")
            loaded = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(loaded, manifest)

            changed = dict(manifest)
            changed["code_revision"] = "different"
            with self.assertRaises(FileExistsError):
                write_calibration_manifest(changed, path)
            self.assertEqual(
                write_calibration_manifest(changed, path, overwrite=True), "written"
            )


if __name__ == "__main__":
    unittest.main()
