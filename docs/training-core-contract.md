# Training core contract

## Scope

This document defines the Phase-2 optimization, model-selection, threshold,
and run-persistence protocol. It applies to all five production models and both
feature profiles. The Phase-3 policies that supersede the original carry-only
temporal behavior are defined in
[`temporal-state-and-ablation-contract.md`](temporal-state-and-ablation-contract.md).

## Flow-level optimization

The prediction unit is one completed flow, represented by one graph edge.
Training and validation therefore use:

```python
criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([pos_weight], device=device),
    reduction="sum",
)
```

For each truncated-backpropagation block, graph loss sums are accumulated and
then divided by the total number of flows in the block before `backward()`.
Loss histories are also stored as loss per flow. They are never averaged by
graph window. This makes an edge contribute the same weight whether it occurs
in a small or large window.

`train_epoch()` returns a `TrainingEpochResult` with the flow count, graph
count, optimizer-step count, loss per flow, and gradient-norm diagnostics.
Passing a BCE criterion with any reduction other than `sum` is an error.

## Explicit model protocol

Every production model declares a boolean `temporal` capability. Every run
configuration must contain a matching `temporal` value and an explicit
`temporal_memory_policy`. No model behavior is inferred from an experiment
name.

The available Phase-3 base policies are:

| Model | `temporal` | `temporal_memory_policy` |
|---|---:|---|
| `SimpleMLP` | `false` | `none` |
| `EdgeGRU_Baseline_NoX` | `true` | `exponential_decay`, `hard_reset`, or `carry_no_decay` |
| `StaticGNN_Identity` | `false` for current; `true` for lagged | `none` or `lagged_identity_only` |
| `ST_GNN_Identity` | depends on memory and identity controls | recurrent policy, `lagged_identity_only`, or `none` |
| `E_GraphSAGE` | `false` | `none` |

Here, a sequence is one complete chronological pass through a split: one
training epoch or one validation/test evaluation. Temporal models must
implement `reset_memory()` and `detach_all_memory()`. Training and evaluation
fail immediately if the configuration and model capability or policy disagree.

## Validation selection and thresholds

Checkpoint selection uses validation average precision, computed with
`sklearn.metrics.average_precision_score`. The early-stopping mode is `max`,
and the run record names `average_precision` as the selection metric.

Threshold selection also uses validation data only and must declare one of:

- `{"strategy": "max_f1"}`;
- `{"strategy": "constrained", "min_precision": 0.9}`.

`min_precision` is invalid for `max_f1` and mandatory for `constrained`.
Every metric and evaluation path predicts the positive class when
`probability >= threshold`, matching the threshold semantics of the
precision-recall curve.

## Required run configuration

A base configuration passed to `run_multiple_seeds()` has this shape:

```python
model_config = {
    "model_name": "simple_mlp",
    "type": "edge_baseline",
    "variant": "base",
    "model_params": {
        "edge_dim": train_dataset.edge_dim,
        "hidden_dim": 64,
        "dropout": 0.2,
    },
    "temporal": False,
    "temporal_memory_policy": "none",
    "selection_metric": "average_precision",
    "threshold": {"strategy": "max_f1"},
    "data_params": {
        "label_correction_version": "cse-cic-ids2018-infiltration-v1",
    },
    "extra_params": {
        "learning_rate": 1e-3,
        "pos_weight": 1.0,
        "batch_steps": 10,
        "patience": 10,
        "min_delta": 1e-4,
        "max_grad_norm": None,
    },
}
```

The runner validates the train/validation manifests and augments each run with:

- graph-manifest and corrected-data-manifest SHA-256 values;
- corrected CSV, source CSV, feature schema, graph collection, scaler,
  mapping, checksum-index, and provenance-collection SHA-256 values;
- feature profile and split;
- label-correction rule version;
- temporal-memory policy and threshold strategy;
- seed, code revision, run ID, and timestamp;
- a canonical SHA-256 of the complete per-run configuration before that hash
  field is inserted;
- best validation AP, best/stopped epochs, selected threshold, timing, and
  loss/gradient histories.

The full nested configuration and metrics are written to a JSON run record and
embedded in the PyTorch checkpoint. The CSV is a compact index, not the
authoritative run record. New checkpoints contain `model_state_dict`; the
shared loader can still read historical raw state dictionaries.

## Verification

Run the complete suite from the repository root:

```bash
PYTHONPATH=code/python python -m unittest discover \
  -s code/python/tests -p "test_*.py" -v
```

The Phase-2 tests include two windows containing one and 999 flows. At a zero
logit, their combined update is checked against the analytical flow-weighted
gradient and against a different partition of the same 1,000 flows. A
window-weighted implementation fails these tests.
