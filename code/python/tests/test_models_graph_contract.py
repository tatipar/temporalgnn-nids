from pathlib import Path
import sys
import unittest

import torch


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from utils.models import (  # noqa: E402
    E_GraphSAGE,
    EdgeGRU_Baseline_NoX,
    SimpleMLP,
    ST_GNN_Identity,
    StaticGNN_Identity,
)


class ProductionModelGraphContractTests(unittest.TestCase):
    edge_dim = 6
    node_dim = 4
    hidden_dim = 8

    def setUp(self):
        torch.manual_seed(42)
        self.graph = {
            "edge_index": torch.tensor(
                [[0, 0, 1, 2], [1, 2, 2, 0]], dtype=torch.long
            ),
            "edge_attr": torch.randn(4, self.edge_dim),
            "num_nodes": 3,
            "global_node_ids": torch.tensor([100, 200, 300], dtype=torch.long),
            "timestamp": 1519830000000,
        }

    def model_factories(self):
        return {
            "simple_mlp": lambda: SimpleMLP(
                edge_dim=self.edge_dim, hidden_dim=self.hidden_dim, dropout=0.0
            ),
            "edge_gru": lambda: EdgeGRU_Baseline_NoX(
                edge_dim=self.edge_dim, hidden_dim=self.hidden_dim, dropout=0.0
            ),
            "static_gnn": lambda: StaticGNN_Identity(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                hidden_dim=self.hidden_dim,
                dropout=0.0,
            ),
            "st_gnn": lambda: ST_GNN_Identity(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                hidden_dim=self.hidden_dim,
                dropout=0.0,
            ),
            "e_graphsage": lambda: E_GraphSAGE(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                hidden_dim=self.hidden_dim,
                dropout=0.0,
            ),
        }

    def test_all_production_models_accept_graphs_without_x(self):
        for name, factory in self.model_factories().items():
            with self.subTest(model=name):
                model = factory().eval()
                output = model(**self.graph)
                self.assertEqual(tuple(output.shape), (4, 1))
                self.assertTrue(torch.isfinite(output).all())

    def test_temporal_models_require_global_node_ids(self):
        for model in (
            EdgeGRU_Baseline_NoX(
                edge_dim=self.edge_dim, hidden_dim=self.hidden_dim, dropout=0.0
            ),
            ST_GNN_Identity(
                node_dim=self.node_dim,
                edge_dim=self.edge_dim,
                hidden_dim=self.hidden_dim,
                dropout=0.0,
            ),
        ):
            graph = self.graph | {"global_node_ids": None}
            with self.assertRaisesRegex(ValueError, "requires global_node_ids"):
                model(**graph)

    def test_temporal_memory_can_be_reset(self):
        model = EdgeGRU_Baseline_NoX(
            edge_dim=self.edge_dim, hidden_dim=self.hidden_dim, dropout=0.0
        )
        model(**self.graph)
        self.assertEqual(set(model.node_memory), {100, 200, 300})
        model.reset_memory()
        self.assertEqual(model.node_memory, {})


if __name__ == "__main__":
    unittest.main()
