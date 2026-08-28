from pathlib import Path
import sys
import unittest

import torch
from torch_geometric.data import Data


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from utils.models import (  # noqa: E402
    E_GraphSAGE,
    EdgeGRU_Baseline_NoX,
    ST_GNN_Identity,
    StaticGNN_Identity,
)
from utils.temporal_state import (  # noqa: E402
    LaggedIdentityState,
    TemporalNodeState,
)
from utils.training import (  # noqa: E402
    evaluate,
    make_flow_criterion,
    validate_model_instance_configuration,
)


WINDOW_MS = 30_000


def graph(timestamp=WINDOW_MS, edge_dim=4):
    data = Data(
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.tensor(
            [[0.2, -0.1, 0.4, 0.8], [0.5, 0.3, -0.2, 0.1]],
            dtype=torch.float32,
        )[:, :edge_dim],
        y=torch.tensor([0.0, 1.0]),
        num_nodes=2,
    )
    data.global_node_ids = torch.tensor([10, 20], dtype=torch.long)
    data.timestamp = timestamp
    return data


def graph_kwargs(data):
    return {
        "edge_index": data.edge_index,
        "edge_attr": data.edge_attr,
        "num_nodes": data.num_nodes,
        "global_node_ids": data.global_node_ids,
        "timestamp": data.timestamp,
    }


class TemporalNodeStateTests(unittest.TestCase):
    def test_scalar_exponential_decay_uses_the_configured_half_life(self):
        state = TemporalNodeState(
            2,
            policy="exponential_decay",
            time_scale_ms=WINDOW_MS,
            decay_half_life_windows=1.0,
        )
        node_ids = torch.tensor([7], dtype=torch.long)
        reference = torch.zeros(1)

        initial, timestamp = state.read(node_ids, WINDOW_MS, reference=reference)
        self.assertTrue(torch.equal(initial, torch.zeros(1, 2)))
        state.write(node_ids, torch.tensor([[2.0, 4.0]]), timestamp)

        recalled, _ = state.read(node_ids, 2 * WINDOW_MS, reference=reference)
        self.assertTrue(
            torch.allclose(recalled, torch.tensor([[1.0, 2.0]]), atol=1e-6)
        )
        self.assertAlmostEqual(
            state.decay_rate().item(),
            torch.log(torch.tensor(2.0)).item(),
            places=6,
        )

    def test_hard_reset_preserves_boundary_and_resets_larger_gap(self):
        node_ids = torch.tensor([7], dtype=torch.long)
        reference = torch.zeros(1)

        boundary = TemporalNodeState(
            1,
            policy="hard_reset",
            time_scale_ms=WINDOW_MS,
            max_gap_ms=2 * WINDOW_MS,
        )
        _, timestamp = boundary.read(node_ids, WINDOW_MS, reference=reference)
        boundary.write(node_ids, torch.tensor([[3.0]]), timestamp)
        recalled, _ = boundary.read(node_ids, 3 * WINDOW_MS, reference=reference)
        self.assertEqual(recalled.item(), 3.0)

        long_gap = TemporalNodeState(
            1,
            policy="hard_reset",
            time_scale_ms=WINDOW_MS,
            max_gap_ms=2 * WINDOW_MS,
        )
        _, timestamp = long_gap.read(node_ids, WINDOW_MS, reference=reference)
        long_gap.write(node_ids, torch.tensor([[3.0]]), timestamp)
        recalled, _ = long_gap.read(node_ids, 4 * WINDOW_MS, reference=reference)
        self.assertEqual(recalled.item(), 0.0)
        self.assertEqual(long_gap.diagnostics()["long_gap_resets"], 1)

    def test_carry_policy_does_not_decay_long_gap(self):
        state = TemporalNodeState(1, policy="carry_no_decay")
        node_ids = torch.tensor([7], dtype=torch.long)
        reference = torch.zeros(1)
        _, timestamp = state.read(node_ids, WINDOW_MS, reference=reference)
        state.write(node_ids, torch.tensor([[5.0]]), timestamp)

        recalled, _ = state.read(node_ids, 100 * WINDOW_MS, reference=reference)
        self.assertEqual(recalled.item(), 5.0)

    def test_reset_clears_hidden_state_timestamps_and_diagnostics(self):
        state = TemporalNodeState(1, policy="carry_no_decay")
        node_ids = torch.tensor([7], dtype=torch.long)
        reference = torch.zeros(1)
        _, timestamp = state.read(node_ids, WINDOW_MS, reference=reference)
        state.write(node_ids, torch.tensor([[5.0]]), timestamp)
        state.reset()

        self.assertEqual(state.node_memory, {})
        self.assertEqual(state.last_seen_ms, {})
        self.assertIsNone(state.last_graph_timestamp_ms)
        self.assertEqual(state.diagnostics()["graphs"], 0)

    def test_duplicate_or_regressing_graph_timestamp_is_rejected(self):
        state = TemporalNodeState(1, policy="carry_no_decay")
        node_ids = torch.tensor([7], dtype=torch.long)
        reference = torch.zeros(1)
        state.read(node_ids, 2 * WINDOW_MS, reference=reference)

        for invalid in (2 * WINDOW_MS, WINDOW_MS):
            with self.subTest(timestamp=invalid):
                with self.assertRaisesRegex(ValueError, "strictly increasing"):
                    state.read(node_ids, invalid, reference=reference)


class LaggedIdentityTests(unittest.TestCase):
    def test_lagged_identity_uses_only_the_immediately_previous_window(self):
        state = LaggedIdentityState(2, window_ms=WINDOW_MS)
        ids = torch.tensor([10, 20], dtype=torch.long)
        first = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        second = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        selected_first = state.select(first, ids, WINDOW_MS)
        selected_second = state.select(second, ids, 2 * WINDOW_MS)
        selected_after_gap = state.select(second, ids, 4 * WINDOW_MS)

        self.assertTrue(torch.equal(selected_first, torch.zeros_like(first)))
        self.assertTrue(torch.equal(selected_second, first))
        self.assertTrue(torch.equal(selected_after_gap, torch.zeros_like(first)))
        self.assertEqual(state.diagnostics()["identity_gap_invalidations"], 1)

    def test_node_absent_from_previous_window_has_zero_identity(self):
        state = LaggedIdentityState(2, window_ms=WINDOW_MS)
        state.select(
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([10]),
            WINDOW_MS,
        )
        selected = state.select(
            torch.tensor([[3.0, 4.0]]),
            torch.tensor([20]),
            2 * WINDOW_MS,
        )
        self.assertTrue(torch.equal(selected, torch.zeros(1, 2)))

    def test_reset_prevents_identity_transfer_between_sequences(self):
        state = LaggedIdentityState(2, window_ms=WINDOW_MS)
        node_ids = torch.tensor([10], dtype=torch.long)
        identity = torch.tensor([[1.0, 2.0]])
        state.select(identity, node_ids, WINDOW_MS)
        state.reset()

        selected = state.select(identity, node_ids, 2 * WINDOW_MS)

        self.assertTrue(torch.equal(selected, torch.zeros_like(identity)))
        self.assertEqual(state.diagnostics()["identity_cache_hits"], 0)


class TemporalModelIntegrationTests(unittest.TestCase):
    edge_dim = 4
    hidden_dim = 6
    node_dim = 3

    def temporal_factories(self):
        return {
            "edge_gru": lambda: EdgeGRU_Baseline_NoX(
                edge_dim=self.edge_dim,
                hidden_dim=self.hidden_dim,
                dropout=0.0,
                memory_policy="exponential_decay",
                time_scale_ms=WINDOW_MS,
                decay_half_life_windows=1.0,
            ),
            "st_gnn": lambda: ST_GNN_Identity(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                hidden_dim=self.hidden_dim,
                dropout=0.0,
                memory_policy="exponential_decay",
                time_scale_ms=WINDOW_MS,
                decay_half_life_windows=1.0,
            ),
        }

    def test_both_temporal_models_track_last_seen_and_decay(self):
        for name, factory in self.temporal_factories().items():
            with self.subTest(model=name):
                model = factory().eval()
                model(**graph_kwargs(graph(WINDOW_MS)))
                model(**graph_kwargs(graph(3 * WINDOW_MS)))

                self.assertEqual(
                    model.last_seen,
                    {10: 3 * WINDOW_MS, 20: 3 * WINDOW_MS},
                )
                diagnostics = model.get_temporal_diagnostics()
                self.assertEqual(diagnostics["decayed_nodes"], 2)
                self.assertEqual(diagnostics["gap_max_windows"], 2.0)
                self.assertIn("temporal_state.raw_decay_rate", model.state_dict())

    def test_both_temporal_models_require_a_timestamp(self):
        for name, factory in self.temporal_factories().items():
            with self.subTest(model=name):
                model = factory().eval()
                data = graph(WINDOW_MS)
                data.timestamp = None

                with self.assertRaisesRegex(ValueError, "decision timestamp"):
                    model(**graph_kwargs(data))

    def test_both_temporal_models_apply_hard_reset_after_long_gap(self):
        factories = {
            "edge_gru": lambda: EdgeGRU_Baseline_NoX(
                edge_dim=self.edge_dim,
                hidden_dim=self.hidden_dim,
                dropout=0.0,
                memory_policy="hard_reset",
                time_scale_ms=WINDOW_MS,
                max_gap_ms=WINDOW_MS,
            ),
            "st_gnn": lambda: ST_GNN_Identity(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                hidden_dim=self.hidden_dim,
                dropout=0.0,
                memory_policy="hard_reset",
                time_scale_ms=WINDOW_MS,
                max_gap_ms=WINDOW_MS,
            ),
        }
        for name, factory in factories.items():
            with self.subTest(model=name):
                model = factory().eval()
                model(**graph_kwargs(graph(WINDOW_MS)))
                model(**graph_kwargs(graph(3 * WINDOW_MS)))

                diagnostics = model.get_temporal_diagnostics()
                self.assertEqual(diagnostics["long_gap_resets"], 2)

    def test_reset_clears_both_temporal_models(self):
        for name, factory in self.temporal_factories().items():
            with self.subTest(model=name):
                model = factory().eval()
                model(**graph_kwargs(graph(WINDOW_MS)))
                model.reset_memory()

                self.assertEqual(model.node_memory, {})
                self.assertEqual(model.last_seen, {})
                self.assertEqual(model.get_temporal_diagnostics()["graphs"], 0)

    def test_evaluation_resets_state_between_splits(self):
        criterion = make_flow_criterion(1.0, "cpu")
        factories = {
            "edge_gru": lambda: EdgeGRU_Baseline_NoX(
                edge_dim=self.edge_dim,
                hidden_dim=self.hidden_dim,
                dropout=0.0,
                memory_policy="carry_no_decay",
            ),
            "st_gnn": lambda: ST_GNN_Identity(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                hidden_dim=self.hidden_dim,
                dropout=0.0,
                memory_policy="carry_no_decay",
            ),
        }
        for name, factory in factories.items():
            with self.subTest(model=name):
                model = factory().eval()
                evaluate(model, [graph(WINDOW_MS)], criterion, "cpu", temporal=True)
                first = model.get_temporal_diagnostics()
                evaluate(
                    model,
                    [graph(10 * WINDOW_MS)],
                    criterion,
                    "cpu",
                    temporal=True,
                )
                second = model.get_temporal_diagnostics()

                self.assertEqual(first["new_nodes"], 2)
                self.assertEqual(second["new_nodes"], 2)
                self.assertEqual(second["recalled_nodes"], 0)

    def test_st_gnn_without_memory_matches_static_gnn(self):
        static = StaticGNN_Identity(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            dropout=0.0,
        ).eval()
        no_memory = ST_GNN_Identity(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            dropout=0.0,
            use_memory=False,
            memory_policy="none",
        ).eval()
        no_memory.load_state_dict(static.state_dict(), strict=True)
        data = graph(WINDOW_MS)

        expected = static(**graph_kwargs(data))
        actual = no_memory(
            edge_index=data.edge_index,
            edge_attr=data.edge_attr,
            num_nodes=data.num_nodes,
        )

        self.assertFalse(no_memory.temporal)
        self.assertTrue(torch.allclose(actual, expected, atol=1e-7))

    def test_st_gnn_controls_preserve_one_logit_per_flow(self):
        variants = {
            "no_topology": {
                "use_topology": False,
                "memory_policy": "carry_no_decay",
            },
            "no_direct_edge_attr": {
                "use_direct_edge_attr": False,
                "memory_policy": "carry_no_decay",
            },
            "lagged_identity": {
                "identity_mode": "lagged",
                "memory_policy": "carry_no_decay",
            },
        }
        for name, options in variants.items():
            with self.subTest(variant=name):
                model = ST_GNN_Identity(
                    node_dim=self.node_dim,
                    edge_dim=self.edge_dim,
                    hidden_dim=self.hidden_dim,
                    dropout=0.0,
                    **options,
                ).eval()
                output = model(**graph_kwargs(graph(WINDOW_MS)))
                self.assertEqual(tuple(output.shape), (2, 1))
                self.assertTrue(torch.isfinite(output).all())

    def test_recorded_identity_and_controls_must_match_st_gnn(self):
        model = ST_GNN_Identity(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            dropout=0.0,
            identity_mode="current",
            memory_policy="carry_no_decay",
        )
        configuration = {
            "temporal": True,
            "temporal_memory_policy": "carry_no_decay",
            "model_params": {
                "identity_mode": "lagged",
                "use_memory": True,
                "use_topology": True,
                "use_direct_edge_attr": True,
            },
        }

        with self.assertRaisesRegex(ValueError, "identity_mode"):
            validate_model_instance_configuration(model, configuration)

    def test_egraphsage_constructs_internal_ones_state(self):
        model = E_GraphSAGE(
            node_dim=self.node_dim,
            edge_dim=self.edge_dim,
            hidden_dim=self.hidden_dim,
            dropout=0.0,
        )
        reference = torch.zeros(2, self.edge_dim)
        initial = model.initial_node_state(3, reference)

        self.assertEqual(tuple(initial.shape), (3, self.node_dim))
        self.assertTrue(torch.equal(initial, torch.ones_like(initial)))


if __name__ == "__main__":
    unittest.main()
