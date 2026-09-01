# Training graph contract

## Scope

This document defines the Phase-1 boundary between the audited NF-v3 graph
artifacts and the production model family. It supplements the reconstruction
and retraining guide; it does not replace the loss, temporal-memory, calibration,
or experiment-freezing requirements in later phases.

## Dataset loading

`NF_IDS_Dataset` receives three explicit coordinates:

```python
from utils.datasets import NF_IDS_Dataset

dataset = NF_IDS_Dataset(
    graph_root="/path/to/graphs/infiltration_v1_w30_tcpflags_episode_split_v1",
    profile="nfv3_extended",
    split="train",
)
```

The graph root is the directory containing `graph_manifest.json`. The loader
will not accept a partial or failed manifest. It validates:

- the requested profile and split;
- the feature-schema self-hash and its manifest declaration;
- the schema dimension and ordered feature-column count;
- the number and chronological filename order of graph windows;
- required tensors and metadata whenever a graph is loaded;
- edge, target, node-map, timestamp, profile, and schema consistency.

`verify_checksums=True` additionally verifies graph files on first access
against `artifact_checksums.json`. This option is recommended after copying a
collection between storage systems. It is intentionally optional for repeated
Drive-backed training because hashing every graph adds remote I/O.

For long Colab screening, stage only the required train and validation graph
files plus `graph_manifest.json`, `artifact_checksums.json`, the selected
feature schema, and its scaler under `/content`. Check free space before the
copy, retain a safety reserve, and verify every staged graph against the
checksum index. Train from the local copy while keeping frozen manifests,
completion artifacts, and low-frequency resume state in Drive. This avoids
reloading every graph from Drive on every epoch without weakening provenance:
the local manifest hash must equal the reviewed Drive manifest hash.

`validate_all()` eagerly loads every graph and validates the complete split.
Normal indexed access performs the same per-graph checks lazily.

After installing the project dependencies, the forward-contract smoke check can
be run directly against the published Drive collection:

```bash
python code/python/scripts/smoke_training_graph_contract.py \
  --graph-root /path/to/graphs/infiltration_v1_w30_tcpflags_episode_split_v1 \
  --profile nfv3_extended \
  --split train \
  --max-graphs 2 \
  --device cpu
```

This command performs inference only. It does not optimize parameters or write
results into the graph collection.

## Model input

Production graphs intentionally contain no `data.x`. The five target models
share this call contract:

```python
logits = model(
    edge_index=data.edge_index,
    edge_attr=data.edge_attr,
    num_nodes=data.num_nodes,
    global_node_ids=data.global_node_ids,
    timestamp=data.timestamp,
)
```

The model policies are:

- `SimpleMLP` uses only `edge_attr`.
- `EdgeGRU_Baseline_NoX` derives node updates from edge attributes and uses
  global node IDs as temporal-memory keys.
- `StaticGNN_Identity` induces node identity from incoming and outgoing edge
  aggregates.
- `ST_GNN_Identity` combines induced identity, graph convolution, and per-node
  temporal memory.
- `E_GraphSAGE` creates its all-ones constant initial node state internally;
  the constant is not persisted as an input feature.

The shared `forward_graph()` helper in `utils.training` is the only dispatch
needed by training, validation, and threshold selection. Batch size must remain
one graph window and shuffling must remain disabled for temporal models.

## Phase-1 completion test

Before loss calibration or temporal-policy experiments, run the dataset and
model contract tests and then perform a one-seed MLP smoke run on
`nfv3_extended`. The loader must report 40 edge features for that profile and
18 for `portable_core`. The smoke run is a technical compatibility check, not
a reportable model comparison.
