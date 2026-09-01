import copy
from pathlib import Path
import sys
import unittest


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from scripts.screen_staticgnn_factorial import _record_matches  # noqa: E402
from utils.staticgnn_factorial import (  # noqa: E402
    DEFAULT_HIDDEN_DIMS,
    DEFAULT_LEARNING_RATES,
    FACTORIAL_STAGE,
    build_staticgnn_factorial_configurations,
    build_staticgnn_factorial_plan,
    build_staticgnn_factorial_summary,
    factorial_configuration_id,
    factorial_result_from_run_record,
)


CALIBRATION_HASH = "a" * 64
GRAPH_HASH = "b" * 64


def candidates_fixture():
    return [
        {
            "candidate_id": f"calibration_{index:02d}",
            "pos_weight": weight,
            "output_bias_init": bias,
        }
        for index, (weight, bias) in enumerate(
            (
                (1.0, -3.3),
                (2.0, -2.6),
                (5.0, -1.7),
                (14.0, -0.7),
                (28.0, 0.0),
            ),
            start=1,
        )
    ]


def configurations_fixture():
    return build_staticgnn_factorial_configurations(
        candidates=candidates_fixture(),
        learning_rates=DEFAULT_LEARNING_RATES,
        hidden_dims=DEFAULT_HIDDEN_DIMS,
        edge_dim=40,
        node_dim=16,
        window_ms=30_000,
        correction_rule_version="rules-v1",
        calibration_manifest_sha256=CALIBRATION_HASH,
        calibration_code_revision="c" * 40,
    )


def run_record_fixture(configuration, ap=0.8, total_seconds=100.0):
    actual = copy.deepcopy(configuration)
    actual["model_name"] += "_seed42"
    actual["extra_params"].update(
        {
            "seed": 42,
            "configuration_sha256": "d" * 64,
            "code_version": "e" * 40,
        }
    )
    metrics = {
        "best_validation_ap": ap,
        "AUC-PR": ap,
        "best_epoch": 12,
        "stopped_epoch": 22,
        "optimal_threshold": 0.42,
        "threshold_selection": "max_f1",
        "Precision": 0.8,
        "Recall": 0.7,
        "F1": 0.75,
        "F2": 0.72,
        "FPR": 0.01,
        "TP": 70,
        "FP": 10,
        "TN": 990,
        "FN": 30,
        "time_total_sec": total_seconds,
        "time_train_sec": total_seconds * 0.8,
        "time_eval_sec": total_seconds * 0.19,
        "time_final_eval_sec": total_seconds * 0.005,
        "time_threshold_sec": total_seconds * 0.01,
    }
    return {
        "run_id": "run-" + actual["data_params"]["factorial_configuration_id"],
        "configuration": actual,
        "metrics": metrics,
    }


def result_fixture(configuration, ap=0.8, total_seconds=100.0):
    return factorial_result_from_run_record(
        run_record_fixture(configuration, ap=ap, total_seconds=total_seconds),
        expected_configuration=configuration,
        calibration_manifest_sha256=CALIBRATION_HASH,
        factorial_plan_sha256="f" * 64,
        run_record_path="record.json",
        run_record_sha256="1" * 64,
        checkpoint_path="checkpoint.pth",
        checkpoint_sha256="2" * 64,
    )


class StaticGNNFactorialConfigurationTests(unittest.TestCase):
    def test_configuration_identifier_is_stable(self):
        self.assertEqual(
            factorial_configuration_id("calibration_02", 0.005, 64),
            "calibration_02_lr0p005_h64",
        )

    def test_expands_complete_twenty_cell_factorial(self):
        configurations = configurations_fixture()

        self.assertEqual(len(configurations), 20)
        identifiers = [
            item["data_params"]["factorial_configuration_id"]
            for item in configurations
        ]
        self.assertEqual(len(set(identifiers)), 20)
        self.assertEqual(identifiers[0], "calibration_01_lr0p001_h32")
        self.assertEqual(identifiers[-1], "calibration_05_lr0p005_h64")
        self.assertTrue(all(not item["temporal"] for item in configurations))
        self.assertTrue(
            all(item["temporal_memory_policy"] == "none" for item in configurations)
        )
        self.assertTrue(
            all(item["model_params"]["node_dim"] == 16 for item in configurations)
        )
        self.assertTrue(
            all(item["extra_params"]["epochs"] == 100 for item in configurations)
        )

    def test_plan_freezes_grid_progress_and_no_test_access(self):
        configurations = configurations_fixture()
        plan = build_staticgnn_factorial_plan(
            calibration_manifest_sha256=CALIBRATION_HASH,
            graph_manifest_sha256=GRAPH_HASH,
            code_revision="3" * 40,
            configurations=configurations,
            learning_rates=DEFAULT_LEARNING_RATES,
            hidden_dims=DEFAULT_HIDDEN_DIMS,
            device="cpu",
            num_workers=2,
            graph_reads="checksum-verified local Colab cache",
        )

        self.assertEqual(plan["grid"]["configuration_count"], 20)
        self.assertEqual(plan["selection_policy"]["finalist_count"], 2)
        self.assertEqual(
            plan["selection_policy"]["finalist_confirmation_seeds"],
            [42, 123, 777],
        )
        self.assertFalse(plan["selection_policy"]["test_splits_accessed"])
        self.assertEqual(plan["execution"]["device"], "cpu")
        self.assertIn("tenth epoch", plan["execution"]["progress_output"])
        self.assertEqual(plan["execution"]["local_resume_every_epochs"], 10)
        self.assertEqual(plan["execution"]["durable_resume_sync_minutes"], 60.0)
        self.assertEqual(
            plan["execution"]["graph_reads"],
            "checksum-verified local Colab cache",
        )
        self.assertEqual(plan["fixed_parameters"]["node_dim"], 16)
        self.assertEqual(plan["fixed_parameters"]["maximum_epochs"], 100)

    def test_plan_rejects_incomplete_factorial(self):
        with self.assertRaisesRegex(ValueError, "all five calibration candidates"):
            build_staticgnn_factorial_plan(
                calibration_manifest_sha256=CALIBRATION_HASH,
                graph_manifest_sha256=GRAPH_HASH,
                code_revision="3" * 40,
                configurations=configurations_fixture()[:-1],
                learning_rates=DEFAULT_LEARNING_RATES,
                hidden_dims=DEFAULT_HIDDEN_DIMS,
                device="cpu",
                num_workers=2,
            )

    def test_plan_rejects_twenty_cells_that_are_not_the_declared_product(self):
        configurations = configurations_fixture()
        configurations[-1] = copy.deepcopy(configurations[0])
        configurations[-1]["data_params"]["factorial_configuration_id"] = "fake-cell"
        with self.assertRaisesRegex(ValueError, "declared grid"):
            build_staticgnn_factorial_plan(
                calibration_manifest_sha256=CALIBRATION_HASH,
                graph_manifest_sha256=GRAPH_HASH,
                code_revision="3" * 40,
                configurations=configurations,
                learning_rates=DEFAULT_LEARNING_RATES,
                hidden_dims=DEFAULT_HIDDEN_DIMS,
                device="cpu",
                num_workers=2,
            )

    def test_plan_rejects_a_changed_fixed_node_dimension(self):
        configurations = configurations_fixture()
        configurations[-1]["model_params"]["node_dim"] = 32
        with self.assertRaisesRegex(ValueError, "must be fixed"):
            build_staticgnn_factorial_plan(
                calibration_manifest_sha256=CALIBRATION_HASH,
                graph_manifest_sha256=GRAPH_HASH,
                code_revision="3" * 40,
                configurations=configurations,
                learning_rates=DEFAULT_LEARNING_RATES,
                hidden_dims=DEFAULT_HIDDEN_DIMS,
                device="cpu",
                num_workers=2,
            )


class StaticGNNFactorialResultTests(unittest.TestCase):
    def test_resume_requires_exact_factorial_configuration(self):
        configuration = configurations_fixture()[0]
        record = run_record_fixture(configuration)

        self.assertTrue(
            _record_matches(record, expected_configuration=configuration)
        )
        changed = copy.deepcopy(configuration)
        changed["model_params"]["hidden_dim"] = 64
        self.assertFalse(_record_matches(record, expected_configuration=changed))

    def test_result_records_validation_metrics_timing_and_artifacts(self):
        configuration = configurations_fixture()[0]
        result = result_fixture(configuration, ap=0.91, total_seconds=125.0)

        self.assertEqual(result["configuration_id"], "calibration_01_lr0p001_h32")
        self.assertEqual(result["best_validation_ap"], 0.91)
        self.assertEqual(result["threshold_selection"], "max_f1")
        self.assertEqual(result["timing_seconds"]["total"], 125.0)
        self.assertEqual(result["run_record"]["path"], "record.json")

    def test_result_rejects_changed_training_recipe(self):
        configuration = configurations_fixture()[0]
        record = run_record_fixture(configuration)
        record["configuration"]["extra_params"]["learning_rate"] = 0.5

        with self.assertRaisesRegex(ValueError, "frozen factorial configuration"):
            factorial_result_from_run_record(
                record,
                expected_configuration=configuration,
                calibration_manifest_sha256=CALIBRATION_HASH,
                factorial_plan_sha256="f" * 64,
                run_record_path="record.json",
                run_record_sha256="1" * 64,
                checkpoint_path="checkpoint.pth",
                checkpoint_sha256="2" * 64,
            )

    def test_summary_ranks_all_cells_and_selects_two_finalists(self):
        configurations = configurations_fixture()
        results = [
            result_fixture(configuration, ap=0.7 + index / 1000, total_seconds=100 + index)
            for index, configuration in enumerate(configurations)
        ]
        summary = build_staticgnn_factorial_summary(
            calibration_manifest_sha256=CALIBRATION_HASH,
            plan_sha256="f" * 64,
            expected_configuration_ids=[
                item["data_params"]["factorial_configuration_id"]
                for item in configurations
            ],
            results=results,
        )

        self.assertEqual(summary["status"], "complete")
        self.assertEqual(len(summary["ranking"]), 20)
        self.assertEqual(
            summary["recommended_finalist_configuration_ids"],
            [
                "calibration_05_lr0p005_h64",
                "calibration_05_lr0p005_h32",
            ],
        )
        self.assertEqual(
            summary["aggregate_completed_run_timing_seconds"]["total"],
            sum(100 + index for index in range(20)),
        )

    def test_partial_summary_does_not_recommend_finalists(self):
        configuration = configurations_fixture()[0]
        summary = build_staticgnn_factorial_summary(
            calibration_manifest_sha256=CALIBRATION_HASH,
            plan_sha256="f" * 64,
            expected_configuration_ids=[
                "calibration_01_lr0p001_h32",
                "calibration_01_lr0p001_h64",
            ],
            results=[result_fixture(configuration)],
        )

        self.assertEqual(summary["status"], "partial")
        self.assertEqual(summary["recommended_finalist_configuration_ids"], [])


if __name__ == "__main__":
    unittest.main()
