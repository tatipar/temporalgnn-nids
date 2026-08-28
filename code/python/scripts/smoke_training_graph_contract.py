#!/usr/bin/env python3
"""Run no-gradient forward checks against an audited NF-v3 graph split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))

from utils.datasets import NF_IDS_Dataset  # noqa: E402
from utils.models import (  # noqa: E402
    E_GraphSAGE,
    EdgeGRU_Baseline_NoX,
    SimpleMLP,
    ST_GNN_Identity,
    StaticGNN_Identity,
)
from utils.training import forward_graph  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-root", type=Path, required=True)
    parser.add_argument(
        "--profile",
        choices=("nfv3_extended", "portable_core"),
        default="nfv3_extended",
    )
    parser.add_argument(
        "--split", choices=("train", "val", "test1", "test2"), default="train"
    )
    parser.add_argument("--max-graphs", type=int, default=1)
    parser.add_argument("--node-dim", type=int, default=16)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verify-checksums", action="store_true")
    return parser.parse_args()


def make_models(edge_dim: int, node_dim: int, hidden_dim: int, window_ms: int):
    common = {"edge_dim": edge_dim, "hidden_dim": hidden_dim, "dropout": 0.0}
    temporal = {
        "memory_policy": "exponential_decay",
        "time_scale_ms": window_ms,
        "decay_half_life_windows": 20.0,
    }
    return {
        "simple_mlp": SimpleMLP(**common),
        "edge_gru": EdgeGRU_Baseline_NoX(**common, **temporal),
        "static_gnn": StaticGNN_Identity(
            node_dim=node_dim,
            identity_mode="current",
            window_ms=window_ms,
            **common,
        ),
        "st_gnn": ST_GNN_Identity(
            node_dim=node_dim,
            identity_mode="current",
            use_memory=True,
            use_topology=True,
            use_direct_edge_attr=True,
            window_ms=window_ms,
            **common,
            **temporal,
        ),
        "e_graphsage": E_GraphSAGE(node_dim=node_dim, **common),
    }


def main() -> None:
    args = parse_args()
    if args.max_graphs <= 0 or args.node_dim <= 0 or args.hidden_dim <= 0:
        raise ValueError("max-graphs, node-dim, and hidden-dim must be positive.")

    device = torch.device(args.device)
    dataset = NF_IDS_Dataset(
        graph_root=args.graph_root,
        profile=args.profile,
        split=args.split,
        verify_checksums=args.verify_checksums,
    )
    models = make_models(
        dataset.edge_dim,
        args.node_dim,
        args.hidden_dim,
        dataset.window_ms,
    )
    inspected_graphs = min(args.max_graphs, len(dataset))
    summaries = {}

    for model_name, model in models.items():
        model = model.to(device).eval()
        if hasattr(model, "reset_memory"):
            model.reset_memory()
        edge_count = 0
        with torch.inference_mode():
            for index in range(inspected_graphs):
                graph = dataset[index].to(device)
                logits = forward_graph(model, graph)
                expected_shape = (int(graph.edge_attr.shape[0]), 1)
                if tuple(logits.shape) != expected_shape:
                    raise AssertionError(
                        f"{model_name} returned shape {tuple(logits.shape)}; expected {expected_shape}."
                    )
                if not torch.isfinite(logits).all():
                    raise AssertionError(f"{model_name} returned non-finite logits.")
                edge_count += expected_shape[0]
        summaries[model_name] = {"graphs": inspected_graphs, "edges": edge_count, "status": "passed"}

    print(json.dumps({
        "graph_root": str(dataset.graph_root),
        "profile": dataset.profile,
        "split": dataset.split,
        "schema_hash": dataset.expected_schema_hash,
        "edge_dim": dataset.edge_dim,
        "available_graphs": len(dataset),
        "inspected_graphs": inspected_graphs,
        "models": summaries,
        "status": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
