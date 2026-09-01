# Temporal state and ablation contract

## Scope

This document defines the Phase-3 temporal-state, identity-context, and model
ablation protocol. It applies to `EdgeGRU_Baseline_NoX`,
`ST_GNN_Identity`, the lagged variant of `StaticGNN_Identity`, and the
constant initial state used by `E_GraphSAGE`.

## Timestamp-aware recurrent memory

EdgeGRU and ST-GNN store a hidden vector and `last_seen` decision timestamp
for every global node ID. Graph timestamps must be strictly increasing during
one chronological sequence. When a node reappears, its elapsed time is:

```text
delta_windows = (current_timestamp_ms - last_seen_ms) / time_scale_ms
```

The graph-manifest `window_ms`, `model_params.time_scale_ms`, and, when used,
`model_params.window_ms` must agree. Empty windows are not emitted as graphs,
but their elapsed time is preserved by the timestamp difference when a node
reappears.

The supported recurrent-memory policies are:

| Policy | Retrieved hidden state |
|---|---|
| `exponential_decay` | `h * exp(-softplus(raw_rate) * delta_windows)` |
| `hard_reset` | zero when `delta_ms > max_gap_ms`; otherwise unchanged |
| `carry_no_decay` | unchanged for every positive gap |

`exponential_decay` is the primary Phase-3 policy. It uses one learned positive
scalar rate shared across nodes and hidden dimensions. The configured
`decay_half_life_windows` initializes that rate; it is not held fixed during
training. The default initialization is 20 windows and must be frozen in the
experiment configuration before confirmatory runs.

`hard_reset` and `carry_no_decay` are ablations. A hard-reset cutoff must be
chosen from training-only gap diagnostics and recorded as `max_gap_ms`; a gap
equal to the cutoff is retained and a larger gap is reset.

The model constructors retain `carry_no_decay` as a direct-call compatibility
default. The experiment runner does not accept an implicit policy: both
`model_params.memory_policy` and the top-level `temporal_memory_policy` must be
present and equal.

## Sequence boundaries and diagnostics

Training resets all temporal state before every epoch. Validation and test
evaluation reset it before every complete split pass. Therefore hidden state,
`last_seen`, and lagged identity never transfer between train, validation, and
test splits. TBPTT detachment preserves numeric state and timestamps while
cutting only the autograd graph.

Every temporal pass records bounded JSON-safe diagnostics in the training
history and final run metrics:

- graph, new-node, and recalled-node counts;
- decayed-node and long-gap-reset counts;
- maximum and mean gap in windows;
- fixed gap-histogram buckets;
- learned decay rate and equivalent half-life;
- lagged-identity cache hits, misses, and gap invalidations when applicable.

## Current and lagged identity

`identity_mode="current"` builds source and destination aggregates from the
edges in the graph being classified. This preserves the historical identity
definition.

`identity_mode="lagged"` uses identity cached from exactly the immediately
preceding wall-clock window. A node receives zero identity when it was absent
from that preceding window, on the first graph in a sequence, or after a
timestamp gap larger than one window. The current graph identity is published
only after selecting the input for the current prediction, so the edge being
classified cannot enter its own lagged identity context. Current edge
attributes remain available to GATv2 and to the direct classifier path unless
their corresponding ablation is enabled.

Lagged identity is stateful even without recurrent memory. Such a model must
declare `temporal=true` and
`temporal_memory_policy="lagged_identity_only"`.

## ST-GNN controls

ST-GNN records three independent boolean controls:

| Control | Disabled behavior |
|---|---|
| `use_memory` | bypass the GRU and recurrent node state |
| `use_topology` | bypass both GATv2 layers and feed local identity aggregates to the next stage |
| `use_direct_edge_attr` | remove only the classifier's direct edge-attribute shortcut |

With current identity, topology enabled, direct edge attributes enabled, and
`use_memory=false`, ST-GNN is exactly equivalent to `StaticGNN_Identity` and
shares its state-dictionary layout. Disabling topology does not remove edge
attributes from the identity aggregates. Disabling direct edge attributes does
not remove them from identity construction or GATv2 messages.

Recommended confirmatory variants are:

1. full ST-GNN with current identity and exponential decay;
2. lagged identity;
3. no memory;
4. no topology;
5. no direct edge attributes;
6. hard-reset and carry-without-decay gap-policy ablations.

## Explicit configurations

Primary EdgeGRU memory parameters:

```python
"model_params": {
    "edge_dim": train_dataset.edge_dim,
    "hidden_dim": 64,
    "dropout": 0.2,
    "memory_policy": "exponential_decay",
    "time_scale_ms": train_dataset.window_ms,
    "decay_half_life_windows": 20.0,
},
"temporal": True,
"temporal_memory_policy": "exponential_decay",
```

Primary ST-GNN architecture and memory parameters:

```python
"model_params": {
    "node_dim": 16,
    "edge_dim": train_dataset.edge_dim,
    "hidden_dim": 64,
    "dropout": 0.2,
    "identity_mode": "current",
    "use_memory": True,
    "use_topology": True,
    "use_direct_edge_attr": True,
    "memory_policy": "exponential_decay",
    "time_scale_ms": train_dataset.window_ms,
    "window_ms": train_dataset.window_ms,
    "decay_half_life_windows": 20.0,
},
"temporal": True,
"temporal_memory_policy": "exponential_decay",
```

For `hard_reset`, replace the memory policy and record `max_gap_ms`. For the
ST-GNN no-memory/current-identity ablation, set `use_memory=False`,
`memory_policy="none"`, `temporal=False`, and
`temporal_memory_policy="none"`. If identity is lagged, the model remains
temporal regardless of recurrent memory.

Every StaticGNN and ST-GNN run must record `identity_mode` explicitly. Every
ST-GNN run must also record all three architecture-control booleans; training
and evaluation reject a constructed model that disagrees with those values.

## E-GraphSAGE initial state

`E_GraphSAGE` constructs its all-ones node state internally from `num_nodes`
and the edge tensor's device and dtype. Graph artifacts remain free of a
persisted `data.x` tensor.

## Verification

Run the complete test suite from the repository root:

```bash
PYTHONPATH=code/python python -m unittest discover \
  -s code/python/tests -p "test_*.py" -v
```

The Phase-3 tests cover scalar decay, hard-reset boundaries, long gaps,
timestamp ordering, state reset, split isolation, strict lagged identity,
ST-GNN controls, StaticGNN equivalence, and E-GraphSAGE initialization.
