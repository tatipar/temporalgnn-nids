import copy
from pathlib import Path
import sys
import tempfile
import unittest


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from utils.calibration import build_calibration_manifest  # noqa: E402
from utils.calibration_screening import (  # noqa: E402
    build_screening_model_config,
    build_screening_plan,
    build_screening_summary,
    canonical_sha256,
    calibration_candidates,
    result_from_run_record,
    select_stage_candidates,
    validate_calibration_manifest,
    validate_calibration_sources,
    write_json_artifact,
)
from scripts.screen_calibration import _record_matches  # noqa: E402


GRAPH_HASH = "a" * 64
CALIBRATION_HASH = "b" * 64
SCHEMA_DEFINITION_HASH = "c" * 64
SCHEMA_FILE_HASH = "d" * 64
GRAPH_COLLECTION_HASH = "e" * 64
CORRECTED_MANIFEST_HASH = "f" * 64


def calibration_fixture():
    counts = {
        "total_flows": 104,
        "negative_flows": 100,
        "positive_flows": 4,
        "positive_prevalence": 4 / 104,
        "class_ratio_negative_to_positive": 25.0,
    }
    alignment = {
        "status": "passed",
        "split": "train",
        "target_digest_contract": "test-contract",
        "profiles": {
            "nfv3_extended": {
                "graphs": 2,
                "flows": 104,
                "target_sha256": "1" * 64,
            },
            "portable_core": {
                "graphs": 2,
                "flows": 104,
                "target_sha256": "1" * 64,
            },
        },
    }
    return build_calibration_manifest(
        graph_manifest_sha256=GRAPH_HASH,
        feature_schemas={
            "nfv3_extended": {
                "schema_definition_sha256": SCHEMA_DEFINITION_HASH,
                "schema_file_sha256": SCHEMA_FILE_HASH,
                "graph_collection_sha256": GRAPH_COLLECTION_HASH,
                "edge_dim": 40,
            },
            "portable_core": {
                "schema_definition_sha256": "2" * 64,
                "schema_file_sha256": "3" * 64,
                "graph_collection_sha256": "4" * 64,
                "edge_dim": 18,
            },
        },
        corrected_manifest_sha256=CORRECTED_MANIFEST_HASH,
        corrected_data_sha256="5" * 64,
        correction_rule_version="rules-v1",
        code_revision="6" * 40,
        counts=counts,
        profile_alignment=alignment,
    )


def graph_manifest_fixture():
    return {
        "status": "passed",
        "profiles": {"nfv3_extended": SCHEMA_DEFINITION_HASH},
        "corrected_manifest": {"sha256": CORRECTED_MANIFEST_HASH},
        "artifacts": {
            "feature_schemas": {
                "nfv3_extended": {"sha256": SCHEMA_FILE_HASH},
            },
            "graph_collections": {
                "nfv3_extended": {"sha256": GRAPH_COLLECTION_HASH},
            },
        },
    }


def config_fixture(stage="mlp", candidate_index=0):
    calibration = calibration_fixture()
    candidate = calibration["candidates"][candidate_index]
    return build_screening_model_config(
        stage=stage,
        candidate=candidate,
        edge_dim=40,
        window_ms=30_000,
        correction_rule_version="rules-v1",
        calibration_manifest_sha256=CALIBRATION_HASH,
        calibration_code_revision=calibration["code_revision"],
    )


def run_record_fixture(candidate_index=0, ap=0.4):
    calibration = calibration_fixture()
    candidate = calibration["candidates"][candidate_index]
    config = config_fixture(candidate_index=candidate_index)
    config["model_name"] = f"phase4b_mlp_{candidate['candidate_id']}_seed42"
    config["extra_params"].update(
        {
            "seed": 42,
            "configuration_sha256": "7" * 64,
            "code_version": "8" * 40,
        }
    )
    metrics = {
        "best_validation_ap": ap,
        "AUC-PR": ap,
        "best_epoch": 4,
        "stopped_epoch": 14,
        "optimal_threshold": 0.37,
        "Precision": 0.5,
        "Recall": 0.4,
        "F1": 0.44,
        "F2": 0.42,
        "FPR": 0.01,
        "TP": 40,
        "FP": 40,
        "TN": 960,
        "FN": 60,
    }
    return candidate, {
        "run_id": f"run-{candidate['candidate_id']}",
        "configuration": config,
        "metrics": metrics,
    }


class CalibrationManifestScreeningTests(unittest.TestCase):
    def test_accepts_complete_phase4a_manifest(self):
        calibration = calibration_fixture()

        validate_calibration_manifest(calibration)

        self.assertEqual(len(calibration_candidates(calibration)), 5)

    def test_rejects_inconsistent_counts(self):
        calibration = calibration_fixture()
        calibration["counts"]["total_flows"] += 1

        with self.assertRaisesRegex(ValueError, "counts are inconsistent"):
            validate_calibration_manifest(calibration)

    def test_rejects_bias_not_paired_with_weight(self):
        calibration = calibration_fixture()
        calibration["candidates"][0]["output_bias_init"] += 0.1

        with self.assertRaisesRegex(ValueError, "output bias mismatch"):
            validate_calibration_manifest(calibration)

    def test_matches_calibration_to_exact_graph_sources(self):
        validate_calibration_sources(
            calibration_fixture(),
            graph_manifest_fixture(),
            actual_graph_manifest_sha256=GRAPH_HASH,
        )

        with self.assertRaisesRegex(ValueError, "graph-manifest hashes differ"):
            validate_calibration_sources(
                calibration_fixture(),
                graph_manifest_fixture(),
                actual_graph_manifest_sha256="wrong",
            )

    def test_mlp_requires_every_candidate_and_stgnn_requires_shortlist(self):
        calibration = calibration_fixture()

        self.assertEqual(len(select_stage_candidates(calibration, "mlp")), 5)
        with self.assertRaisesRegex(ValueError, "every frozen"):
            select_stage_candidates(calibration, "mlp", ["calibration_01"])
        with self.assertRaisesRegex(ValueError, "explicit candidate shortlist"):
            select_stage_candidates(calibration, "stgnn")
        selected = select_stage_candidates(
            calibration, "stgnn", ["calibration_01", "calibration_03"]
        )
        self.assertEqual(
            [item["candidate_id"] for item in selected],
            ["calibration_01", "calibration_03"],
        )


class ScreeningConfigurationTests(unittest.TestCase):
    def test_mlp_configuration_freezes_validation_only_protocol(self):
        config = config_fixture()

        self.assertFalse(config["temporal"])
        self.assertEqual(config["temporal_memory_policy"], "none")
        self.assertEqual(config["selection_metric"], "average_precision")
        self.assertEqual(config["threshold"], {"strategy": "max_f1"})
        self.assertEqual(config["extra_params"]["epochs"], 60)
        self.assertEqual(config["extra_params"]["pos_weight"], 1.0)
        self.assertEqual(
            config["model_params"]["output_bias_init"],
            calibration_fixture()["candidates"][0]["output_bias_init"],
        )

    def test_stgnn_configuration_records_all_phase3_controls(self):
        config = config_fixture(stage="stgnn")
        params = config["model_params"]

        self.assertTrue(config["temporal"])
        self.assertEqual(config["temporal_memory_policy"], "exponential_decay")
        self.assertEqual(params["identity_mode"], "current")
        self.assertTrue(params["use_memory"])
        self.assertTrue(params["use_topology"])
        self.assertTrue(params["use_direct_edge_attr"])
        self.assertEqual(params["time_scale_ms"], 30_000)
        self.assertEqual(params["window_ms"], 30_000)
        self.assertEqual(params["decay_half_life_windows"], 20.0)

    def test_screening_plan_is_deterministic_and_records_no_test_access(self):
        config = config_fixture()
        first = build_screening_plan(
            stage="mlp",
            calibration_manifest_sha256=CALIBRATION_HASH,
            graph_manifest_sha256=GRAPH_HASH,
            code_revision="9" * 40,
            configurations=[config],
            device="cuda",
            num_workers=2,
        )
        second = build_screening_plan(
            stage="mlp",
            calibration_manifest_sha256=CALIBRATION_HASH,
            graph_manifest_sha256=GRAPH_HASH,
            code_revision="9" * 40,
            configurations=[config],
            device="cuda",
            num_workers=2,
        )

        self.assertEqual(canonical_sha256(first), canonical_sha256(second))
        self.assertFalse(first["selection_policy"]["test_splits_accessed"])
        self.assertTrue(first["selection_policy"]["fixed_0_5_comparison_prohibited"])

    def test_frozen_plan_refuses_different_rerun(self):
        config = config_fixture()
        plan = build_screening_plan(
            stage="mlp",
            calibration_manifest_sha256=CALIBRATION_HASH,
            graph_manifest_sha256=GRAPH_HASH,
            code_revision="9" * 40,
            configurations=[config],
            device="cpu",
            num_workers=0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "screening_plan.json"
            self.assertEqual(write_json_artifact(plan, path), "written")
            self.assertEqual(write_json_artifact(plan, path), "unchanged")
            changed = copy.deepcopy(plan)
            changed["seed"] = 123
            with self.assertRaises(FileExistsError):
                write_json_artifact(changed, path)


class ScreeningResultTests(unittest.TestCase):
    def screening_result(self, candidate_index, ap):
        candidate, record = run_record_fixture(candidate_index, ap)
        return result_from_run_record(
            record,
            candidate=candidate,
            stage="mlp",
            calibration_manifest_sha256=CALIBRATION_HASH,
            screening_plan_sha256="c" * 64,
            run_record_path=f"record-{candidate_index}.json",
            run_record_sha256="a" * 64,
            checkpoint_path=f"checkpoint-{candidate_index}.pth",
            checkpoint_sha256="b" * 64,
        )

    def test_extracts_only_validation_selected_result(self):
        result = self.screening_result(0, 0.4)

        self.assertEqual(result["candidate_id"], "calibration_01")
        self.assertEqual(result["best_validation_ap"], 0.4)
        self.assertEqual(result["selected_validation_threshold"], 0.37)

    def test_resume_requires_the_exact_frozen_configuration(self):
        candidate, record = run_record_fixture()
        expected = config_fixture()

        self.assertTrue(
            _record_matches(
                record,
                stage="mlp",
                candidate_id=candidate["candidate_id"],
                calibration_manifest_sha256=CALIBRATION_HASH,
                expected_configuration=expected,
            )
        )
        changed = copy.deepcopy(expected)
        changed["extra_params"]["learning_rate"] = 0.01
        self.assertFalse(
            _record_matches(
                record,
                stage="mlp",
                candidate_id=candidate["candidate_id"],
                calibration_manifest_sha256=CALIBRATION_HASH,
                expected_configuration=changed,
            )
        )

    def test_rejects_fixed_threshold_run_record(self):
        candidate, record = run_record_fixture()
        record["configuration"]["threshold"] = {"strategy": "fixed_0.5"}

        with self.assertRaisesRegex(ValueError, "threshold strategy"):
            result_from_run_record(
                record,
                candidate=candidate,
                stage="mlp",
                calibration_manifest_sha256=CALIBRATION_HASH,
                screening_plan_sha256="c" * 64,
                run_record_path="record.json",
                run_record_sha256="a" * 64,
                checkpoint_path="checkpoint.pth",
                checkpoint_sha256="b" * 64,
            )

    def test_summary_applies_frozen_margin_and_prefers_smaller_weight(self):
        lower_weight = self.screening_result(0, 0.400)
        higher_weight = self.screening_result(1, 0.404)
        summary = build_screening_summary(
            stage="mlp",
            calibration_manifest_sha256=CALIBRATION_HASH,
            plan_sha256="c" * 64,
            expected_candidate_ids=["calibration_01", "calibration_02"],
            results=[higher_weight, lower_weight],
        )

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(
            [item["candidate_id"] for item in summary["ranking"]],
            ["calibration_02", "calibration_01"],
        )
        self.assertEqual(
            summary["recommended_stgnn_shortlist_ids"],
            ["calibration_02", "calibration_01"],
        )
        self.assertEqual(
            summary["preferred_smaller_weight_candidate_id"], "calibration_01"
        )

    def test_summary_reports_missing_candidates(self):
        summary = build_screening_summary(
            stage="mlp",
            calibration_manifest_sha256=CALIBRATION_HASH,
            plan_sha256="c" * 64,
            expected_candidate_ids=["calibration_01", "calibration_02"],
            results=[self.screening_result(0, 0.4)],
        )

        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["missing_candidate_ids"], ["calibration_02"])


if __name__ == "__main__":
    unittest.main()
