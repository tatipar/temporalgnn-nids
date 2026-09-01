import csv
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
import torch.nn as nn


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from utils.experiment import (  # noqa: E402
    EarlyStopping,
    ExperimentManager,
    load_model_checkpoint,
)
from utils.metrics import calculate_metrics_gnn  # noqa: E402
from utils.training import (  # noqa: E402
    SELECTION_METRIC,
    _atomic_torch_save,
    _load_latest_resume_state,
    _resume_contract,
    _validate_model_config,
    make_flow_criterion,
    select_optimal_threshold,
    train_epoch,
    validate_temporal_configuration,
)


class ResumeStateContractTests(unittest.TestCase):
    def setUp(self):
        self.model_config = {
            "model_params": {"edge_dim": 40},
            "data_params": {"feature_profile": "nfv3_extended"},
            "extra_params": {"learning_rate": 0.005},
        }
        self.metadata = {
            "graph_manifest_sha256": "a" * 64,
            "feature_profile": "nfv3_extended",
        }
        self.contract = _resume_contract(
            model_config=self.model_config,
            seed=42,
            epochs=100,
            experiment_name="factorial-cell",
            code_version="b" * 40,
            train_metadata=self.metadata,
        )

    def payload(self, next_epoch):
        return {
            "format_version": 1,
            "artifact_type": "epoch_boundary_training_resume",
            "contract": self.contract,
            "next_epoch": next_epoch,
        }

    def test_latest_compatible_epoch_is_selected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            local_path = root / "local.pth"
            durable_path = root / "durable.pth"
            _atomic_torch_save(self.payload(20), local_path)
            _atomic_torch_save(self.payload(10), durable_path)

            selected = _load_latest_resume_state(
                (local_path, durable_path), self.contract
            )

            self.assertIsNotNone(selected)
            self.assertEqual(selected[0], local_path)
            self.assertEqual(selected[1]["next_epoch"], 20)

    def test_changed_contract_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resume.pth"
            _atomic_torch_save(self.payload(10), path)
            changed = dict(self.contract, maximum_epochs=60)

            with self.assertRaisesRegex(ValueError, "frozen run contract"):
                _load_latest_resume_state((path,), changed)


class TinyGraph:
    """Minimal graph object for testing the shared training interface."""

    def __init__(self, targets):
        targets = torch.tensor(targets, dtype=torch.float32)
        flow_count = targets.numel()
        self.edge_index = torch.zeros((2, flow_count), dtype=torch.long)
        self.edge_attr = torch.zeros((flow_count, 1), dtype=torch.float32)
        self.y = targets
        self.num_nodes = 1
        self.global_node_ids = torch.tensor([10], dtype=torch.long)
        self.timestamp = 1_000

    def to(self, device):
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.y = self.y.to(device)
        self.global_node_ids = self.global_node_ids.to(device)
        return self


class ScalarFlowModel(nn.Module):
    """Emit one shared trainable logit for every flow."""

    temporal = False
    temporal_memory_policy = "none"

    def __init__(self):
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        edge_index,
        edge_attr,
        num_nodes,
        global_node_ids=None,
        timestamp=None,
    ):
        return self.logit.expand(edge_attr.shape[0], 1)


class TemporalStub(ScalarFlowModel):
    temporal = True
    temporal_memory_policy = "carry_no_decay"

    def reset_memory(self):
        pass

    def detach_all_memory(self):
        pass


class FlowWeightedLossTests(unittest.TestCase):
    def train_partition(self, windows):
        model = ScalarFlowModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        result = train_epoch(
            model,
            [TinyGraph(targets) for targets in windows],
            optimizer,
            make_flow_criterion(1.0, "cpu"),
            "cpu",
            temporal=False,
            batch_steps=len(windows),
        )
        return model, result

    def test_unequal_windows_are_weighted_by_flow_not_by_window(self):
        model, result = self.train_partition([[1.0], [0.0] * 999])

        self.assertEqual(result.flows, 1_000)
        self.assertEqual(result.graph_windows, 2)
        self.assertEqual(result.optimizer_steps, 1)
        self.assertAlmostEqual(result.loss_per_flow, math.log(2.0), places=6)
        self.assertAlmostEqual(model.logit.item(), -0.499, places=6)

    def test_update_is_invariant_to_flow_partition_across_windows(self):
        unequal_model, _ = self.train_partition([[1.0], [0.0] * 999])
        balanced_model, _ = self.train_partition(
            [[1.0] + [0.0] * 499, [0.0] * 500]
        )

        self.assertAlmostEqual(
            unequal_model.logit.item(), balanced_model.logit.item(), places=7
        )

    def test_mean_reduction_is_rejected(self):
        model = ScalarFlowModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        criterion = nn.BCEWithLogitsLoss(reduction="mean")

        with self.assertRaisesRegex(ValueError, "reduction='sum'"):
            train_epoch(
                model,
                [TinyGraph([0.0, 1.0])],
                optimizer,
                criterion,
                "cpu",
                temporal=False,
                batch_steps=1,
            )


class ExplicitProtocolTests(unittest.TestCase):
    def test_temporal_configuration_must_match_model_capability(self):
        validate_temporal_configuration(ScalarFlowModel(), False)
        validate_temporal_configuration(TemporalStub(), True)

        with self.assertRaisesRegex(ValueError, "Temporal configuration mismatch"):
            validate_temporal_configuration(ScalarFlowModel(), True)
        with self.assertRaisesRegex(ValueError, "Temporal configuration mismatch"):
            validate_temporal_configuration(TemporalStub(), False)

    def test_temporal_memory_policy_must_match_model(self):
        validate_temporal_configuration(
            TemporalStub(), True, "carry_no_decay"
        )
        with self.assertRaisesRegex(ValueError, "Memory-policy mismatch"):
            validate_temporal_configuration(
                TemporalStub(), True, "hard_reset"
            )

    def test_decay_configuration_requires_explicit_time_parameters(self):
        configuration = {
            "model_name": "edge_gru",
            "type": "temporal_baseline",
            "model_params": {
                "edge_dim": 4,
                "hidden_dim": 8,
                "dropout": 0.0,
                "memory_policy": "exponential_decay",
            },
            "extra_params": {
                "learning_rate": 1e-3,
                "pos_weight": 1.0,
                "batch_steps": 2,
            },
            "data_params": {"label_correction_version": "rules-v1"},
            "temporal": True,
            "temporal_memory_policy": "exponential_decay",
            "variant": "base",
            "threshold": {"strategy": "max_f1"},
            "selection_metric": "average_precision",
        }

        with self.assertRaisesRegex(ValueError, "time_scale_ms"):
            _validate_model_config(configuration)

        configuration["model_params"]["time_scale_ms"] = 30_000
        with self.assertRaisesRegex(ValueError, "decay_half_life_windows"):
            _validate_model_config(configuration)

        configuration["model_params"]["decay_half_life_windows"] = 20.0
        _validate_model_config(configuration)

    def test_probability_equal_to_threshold_is_positive(self):
        metrics = calculate_metrics_gnn(
            y_true=np.array([1, 0]),
            y_probs=np.array([0.5, 0.49]),
            prob_threshold=0.5,
        )

        self.assertEqual(metrics["TP"], 1)
        self.assertEqual(metrics["TN"], 1)

    def test_threshold_strategy_has_no_implicit_precision_constraint(self):
        y_true = np.array([0, 1, 0, 1])
        y_probs = np.array([0.1, 0.4, 0.6, 0.8])

        threshold, description = select_optimal_threshold(
            y_true, y_probs, strategy="max_f1"
        )
        self.assertIsInstance(threshold, float)
        self.assertEqual(description, "max_f1")
        with self.assertRaisesRegex(ValueError, "must be omitted"):
            select_optimal_threshold(
                y_true, y_probs, strategy="max_f1", min_precision=0.9
            )

    def test_early_stopping_record_names_average_precision(self):
        model = ScalarFlowModel()
        stopping = EarlyStopping(
            patience=2,
            min_delta=0.0,
            mode="max",
            metric_name=SELECTION_METRIC,
        )

        self.assertTrue(stopping(0.7, model, epoch=0))
        self.assertFalse(stopping(0.6, model, epoch=1))
        self.assertEqual(stopping.metric_name, "average_precision")
        self.assertEqual(stopping.best_epoch, 0)


class ExperimentPersistenceTests(unittest.TestCase):
    def test_checkpoint_and_run_record_preserve_full_configuration(self):
        configuration = {
            "model_name": "scalar_seed42",
            "type": "test",
            "model_params": {"edge_dim": 1},
            "temporal": False,
            "temporal_memory_policy": "none",
            "variant": "base",
            "selection_metric": "average_precision",
            "threshold": {"strategy": "max_f1"},
            "prob_threshold": 0.5,
            "data_params": {
                "label_correction_version": "rules-v1",
                "graph_manifest_sha256": "graph-hash",
                "corrected_data_sha256": "data-hash",
                "feature_schema_sha256": "schema-hash",
                "graph_collection_sha256": "collection-hash",
                "scaler_sha256": "scaler-hash",
                "mapping_sha256": {"day1": "mapping-hash"},
            },
            "extra_params": {
                "run_id": "test-run",
                "run_ts": "20260828_120000",
                "seed": 42,
                "code_version": "commit-hash",
            },
        }
        metrics = {
            "AUC-PR": 0.75,
            "best_validation_ap": 0.75,
            "final_temporal_diagnostics": {
                "memory_policy": "none",
                "gap_count": 0,
            },
        }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manager = ExperimentManager(
                log_file=root / "logs" / "runs.csv",
                model_dir=root / "models",
                record_dir=root / "records",
            )
            paths = manager.log_experiment(
                model_config=configuration,
                metrics=metrics,
                model_object=ScalarFlowModel(),
            )

            record = json.loads(Path(paths["run_record"]).read_text(encoding="utf-8"))
            self.assertEqual(record["configuration"], configuration)
            self.assertEqual(record["metrics"], metrics)
            with (root / "logs" / "runs.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                csv_record = next(csv.DictReader(handle))
            self.assertEqual(
                json.loads(csv_record["final_temporal_diagnostics"]),
                metrics["final_temporal_diagnostics"],
            )

            restored = ScalarFlowModel()
            payload = load_model_checkpoint(restored, paths["checkpoint"])
            self.assertEqual(payload["configuration"], configuration)
            self.assertEqual(payload["metrics"], metrics)


if __name__ == "__main__":
    unittest.main()
