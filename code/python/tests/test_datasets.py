import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch
from torch_geometric.data import Data


PYTHON_ROOT = Path(__file__).resolve().parents[1]
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from utils.datasets import NF_IDS_Dataset  # noqa: E402


def canonical_hash(payload):
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_artifact(root, path):
    serialized = path.read_bytes()
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "bytes": len(serialized),
    }


class GraphCollectionFixture:
    profile = "nfv3_extended"
    split = "train"
    edge_dim = 4

    def __init__(self, root):
        self.root = Path(root)
        self.profile_root = self.root / self.profile
        self.split_root = self.profile_root / self.split
        self.split_root.mkdir(parents=True)

        self.schema_payload = {
            "name": self.profile,
            "numeric_columns": ["a", "b", "c", "d"],
            "port_category_columns": [],
            "protocol_category_columns": [],
            "numeric_transform": "log1p_then_standard_scaler",
            "edge_attr_columns": ["a", "b", "c", "d"],
            "dimension": self.edge_dim,
        }
        self.schema_hash = canonical_hash(self.schema_payload)
        self.schema_path = self.profile_root / "feature_schema.json"
        self.schema_path.write_text(
            json.dumps(self.schema_payload | {"sha256": self.schema_hash}, indent=2),
            encoding="utf-8",
        )
        self.graph_paths = []

    def add_graph(self, timestamp, *, edge_dim=None, schema_hash=None, with_x=False):
        dimension = self.edge_dim if edge_dim is None else edge_dim
        graph = Data(
            edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long),
            edge_attr=torch.arange(3 * dimension, dtype=torch.float32).reshape(3, dimension),
            y=torch.tensor([0.0, 1.0, 0.0]),
            num_nodes=3,
        )
        graph.global_node_ids = torch.tensor([10, 11, 12], dtype=torch.long)
        graph.timestamp = timestamp
        graph.window_start = timestamp - 30_000
        graph.window_end = timestamp
        graph.feature_profile = self.profile
        graph.schema_hash = self.schema_hash if schema_hash is None else schema_hash
        if with_x:
            graph.x = torch.ones(3, 1)
        path = self.split_root / f"graph_{timestamp:013d}.pt"
        torch.save(graph, path)
        self.graph_paths.append(path)
        return path

    def publish(self, *, status="passed", graph_count=None):
        graph_count = len(self.graph_paths) if graph_count is None else graph_count
        checksums = {
            path.relative_to(self.root).as_posix(): {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
            for path in self.graph_paths
        }
        (self.root / "artifact_checksums.json").write_text(
            json.dumps({"algorithm": "sha256", "graphs": {self.profile: checksums}}, indent=2),
            encoding="utf-8",
        )
        manifest = {
            "status": status,
            "window_ms": 30_000,
            "profiles": {self.profile: self.schema_hash},
            "artifacts": {
                "feature_schemas": {
                    self.profile: file_artifact(self.root, self.schema_path),
                }
            },
            "audit": {
                "profiles": {
                    self.profile: {
                        "splits": {self.split: {"graphs": graph_count}},
                    }
                }
            },
        }
        (self.root / "graph_manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


class NFIDSDatasetTests(unittest.TestCase):
    def test_loads_graphs_in_chronological_order_without_x(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = GraphCollectionFixture(directory)
            fixture.add_graph(200_000)
            fixture.add_graph(100_000)
            fixture.publish()

            dataset = NF_IDS_Dataset(directory, fixture.profile, fixture.split)

            self.assertEqual(dataset.timestamps, [100_000, 200_000])
            self.assertEqual(dataset.edge_dim, fixture.edge_dim)
            self.assertEqual(len(dataset), 2)
            graph = dataset[0]
            self.assertIsNone(getattr(graph, "x", None))
            self.assertEqual(tuple(graph.edge_attr.shape), (3, fixture.edge_dim))
            dataset.validate_all()

    def test_rejects_unpublished_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = GraphCollectionFixture(directory)
            fixture.add_graph(100_000)
            fixture.publish(status="partial")

            with self.assertRaisesRegex(ValueError, "status='passed'"):
                NF_IDS_Dataset(directory, fixture.profile, fixture.split)

    def test_rejects_graph_count_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = GraphCollectionFixture(directory)
            fixture.add_graph(100_000)
            fixture.publish(graph_count=2)

            with self.assertRaisesRegex(ValueError, "Graph count mismatch"):
                NF_IDS_Dataset(directory, fixture.profile, fixture.split)

    def test_rejects_wrong_edge_dimension_on_access(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = GraphCollectionFixture(directory)
            fixture.add_graph(100_000, edge_dim=3)
            fixture.publish()
            dataset = NF_IDS_Dataset(directory, fixture.profile, fixture.split)

            with self.assertRaisesRegex(ValueError, "edge_attr shape"):
                _ = dataset[0]

    def test_rejects_graph_schema_mismatch_on_access(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = GraphCollectionFixture(directory)
            fixture.add_graph(100_000, schema_hash="wrong")
            fixture.publish()
            dataset = NF_IDS_Dataset(directory, fixture.profile, fixture.split)

            with self.assertRaisesRegex(ValueError, "profile/schema"):
                _ = dataset[0]

    def test_rejects_persisted_node_features(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = GraphCollectionFixture(directory)
            fixture.add_graph(100_000, with_x=True)
            fixture.publish()
            dataset = NF_IDS_Dataset(directory, fixture.profile, fixture.split)

            with self.assertRaisesRegex(ValueError, "must not persist node features"):
                _ = dataset[0]

    def test_verifies_graph_checksums_when_requested(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = GraphCollectionFixture(directory)
            graph_path = fixture.add_graph(100_000)
            fixture.publish()
            dataset = NF_IDS_Dataset(
                directory, fixture.profile, fixture.split, verify_checksums=True
            )

            serialized = bytearray(graph_path.read_bytes())
            serialized[-1] ^= 1
            graph_path.write_bytes(serialized)

            with self.assertRaisesRegex(ValueError, "checksum"):
                _ = dataset[0]


if __name__ == "__main__":
    unittest.main()
