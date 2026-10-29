# ST-GNN memory-reset correction and retraining history

## Finding

The memory-transfer problem had already been corrected. The identity-projection
ST-GNN notebook added a training-epoch memory reset in commit `550b90c` on
2026-03-20. Five **entropy-feature** ST-GNN runs accompanied that change.
The no-entropy historical checkpoint used in the later comparisons, however,
still identifies the February run that predates the correction.

The available Git history does not establish a later, replacement five-seed
no-entropy run on the original graphs before the newly supplied factorial
experiment. Updating the training implementation and reproducing inference
from old checkpoints were separate operations; the latter did not update the
weights or the protocol under which those weights were learned.

## Scope and evidence limits

- Refreshed remote references with `git fetch --all --prune`; branch tips did
  not change.
- Inspected all available local/remote branches, including
  `feat/fair-comparison-experiments`, `feat/fair-retrain-clean`,
  `feat/capture-feasibility`, `main`, `refactor-utils`, `tati-entropy`,
  `origin/tati-branch`, and `origin/tati-flows`.
- Scanned 294 distinct historical notebook blobs across all reachable refs.
  Saved outputs contain ten distinct timestamped identity-projection ST-GNN
  training run IDs: five February no-entropy runs and five March entropy runs.
  This inventory is evidence about committed outputs, not a complete inventory
  of Drive, deleted branches, or uncommitted Colab executions.
- Also reviewed the factorial/baseline/postanalysis
  notebooks and RESULTS files supplied in this workspace.

## Timeline

| Date | Evidence | What it establishes |
| --- | --- | --- |
| 2026-02-20/21 | `769dcb0`, historical `staticGNN_STGNN_projection.ipynb` | Five no-entropy ST-GNN runs, including seed 42 run `ST_GNN_BiasOn_robust_Identity_clone_seed42_20260220_213225`. Its `train_epoch` lacks a memory reset; `evaluate` resets memory before validation and leaves the resulting state in the model. |
| 2026-03-20 | `550b90c` | Adds `if is_temporal and hasattr(model, 'reset_memory'): model.reset_memory()` immediately after `model.train()`. Commit message explicitly says “also, reset memory during training.” The same commit changes the model to use entropy features and records five entropy ST-GNN runs dated March 19/20. |
| 2026-04-14 | `eeef519` | Consolidates the corrected training function into `utils/training.py`. This is a code refactor, not evidence that historical no-entropy checkpoints were retrained. |
| 2026-04-16 | `7cf20ea` | Creates `staticGNN_STGNN_projection_noentropy.ipynb` importing the updated utilities. The commit says results already exist in Drive under `results_earlystopping/`. The initial saved training output was copied from the entropy variant. |
| 2026-04-17 | `76b9ed9` | Removes those stale entropy outputs. The commit explicitly says: “Actual results are stored in Drive; no rerun needed.” No new no-entropy training output is recorded. |
| 2026-08-21 | `8fbfc80`, `d1278eb` | Opens the clean retraining plan and restores reference notebooks. Restoration is not a completed training run. |
| 2026-08-28 | `8a4bb30`, `ae5ff17` on the clean branch | Implements the new flow-weighted protocol, explicit temporal configuration, timestamp-aware state, sequence resets, diagnostics and tests. These experiments also change labels/graphs/features and cannot silently replace the original-label benchmark. |
| 2026-08-31 / 09-01 | `923341c`, `d70cc7c`, `bd50826`, `aab4646` | Records calibration and preliminary MLP screening; prepares the StaticGNN calibration factorial. The handoff retains full model screening/retraining as Phase 5 and explicitly defers ST-GNN until calibration/finalist confirmation. No completed NF-v3 ST-GNN matrix is established by the committed evidence. |
| 2026-09/10, other dataset | `feat/capture-feasibility`, e.g. `20fed7d` | Does contain completed ST-GNN fold-A/fold-B jobs for the cAPTure graph pilot. Its runner resets temporal state between scenario sequences. These are different data, features, tasks and folds, not replacement CICIDS2018 no-entropy checkpoints. |
| 2026-10-20/21 | `6f6d021`, `dd98a0b`, `06225bd`, `74fe12c` | Audits and analyzes the historical checkpoint. The audit explicitly retains the February run IDs. Successful replay establishes inference reproducibility, not retraining under the corrected protocol. |
| October executed notebooks | `code/python/notebook/baseline_retraining_factorial_protocol_colab.ipynb`, `code/python/notebook/retrained_models_postanalysis_colab.ipynb` | Reports completed five-seed factorial `M1_P1` as the retrained ST-GNN and keeps `ST-GNN (historical)` separate. The factorial uses the corrected pinned training utility with an epoch-start memory reset. |

## The corrected March runs

The five recorded run IDs are:

```text
ST_GNN_BiasOn_robust_Identity_clone_entropy_seed42_20260319_212814
ST_GNN_BiasOn_robust_Identity_clone_entropy_seed123_20260319_221154
ST_GNN_BiasOn_robust_Identity_clone_entropy_seed777_20260319_230618
ST_GNN_BiasOn_robust_Identity_clone_entropy_seed2024_20260319_235751
ST_GNN_BiasOn_robust_Identity_clone_entropy_seed99_20260320_010744
```

They use the entropy experiment name and `results_earlystopping_entropy/`.
They demonstrate that training happened around the correction, but do not
isolate the reset's effect or supply a corrected no-entropy counterpart.

## Where the workflow appears incomplete

The April no-entropy notebook had corrected code while the comparison pipeline
continued resolving the existing `results_earlystopping/` artifacts. Those
artifacts identify February weights. The historical-reference restoration in
August and the inference audit in October did not replace them.

The separate clean NF-v3 branch addressed a broader set of methodological
changes and reached calibration/screening infrastructure. It does not provide
the missing like-for-like, corrected no-entropy historical benchmark in the
committed results inspected here.

This supports the user's hypothesis that the code correction was not followed
through into a replacement no-entropy result set used by the old comparison
tables. It does **not** prove that no additional run exists anywhere in Drive.

## What the recent runtime checks establish

- Original ST-GNN and factorial M1_P1 match exactly for one controlled complete
  epoch and validation, including initial weights, logits, gradients, updated
  weights and validation probabilities.
- With fresh loaders and the current epoch-reset policy, epoch 1 matches the
  historical seed-42 log, but epoch 2 differs substantially.
- Loading the historical `train_epoch`/`evaluate` from `769dcb0` reproduces both
  historical epochs exactly in the supplied check. Epoch 2 starts with 18,092
  stored host states from the preceding validation pass.

These checks distinguish a known historical protocol difference from an
architectural mismatch. Extending the historical replay to all epochs would
complete that particular reconstruction, but it is not needed to establish
when the fix was committed or which run ID the old tables still use.

## Recommended treatment

Preserve February checkpoints as explicitly labeled historical artifacts.
Use the corrected memory-reset protocol for the primary factorial comparison.
Do not substitute entropy or cAPTure runs as if they were no-entropy CICIDS2018
replacements. Before asserting that a corrected no-entropy run exists in Drive,
require its exact run ID, training history, code/protocol identity and checkpoint;
a later evaluation date or a changed threshold is insufficient.

Loader RNG isolation remains an additional reproducibility issue to standardize
for future runs. It does not erase the independently confirmed historical
validation-to-next-training memory transfer.

## Baseline impact assessment

For the models actually used by the current original-label comparison:

| Model | Impact of the historical epoch-start memory omission | Other evidence / remaining review |
| --- | --- | --- |
| ST-GNN / `ST_GNN_BiasOn_robust_Identity_clone` | Confirmed for the February weights. | M1_P1 is the same full architecture retrained using the epoch/split reset policy. This description does not imply that every other protocol issue has been corrected. |
| Edge GRU / `EdgeGRU_NoX_BiasOn` | No evidence of that omission in this April experiment. | Recorded runs are dated April 19, after the shared helper incorporated the reset. The user's full seed-42 rerun matches the historical printed trajectory. Four other seed metrics match the newer baseline readout; its different seed-42 trajectory is associated with the loader-initialization investigation. This assessment does not automatically cover older, differently named GRU experiments. |
| StaticGNN_Identity | Not applicable: no persistent temporal state. | Historical versus newer champion differences do not demonstrate a memory bug. Loader lifecycle, seed selection and restart behavior remain relevant. |
| SimpleMLP | Not applicable: no persistent temporal state. | Historical and retrained AP values reproduce closely. Original precision-constrained thresholds and later max-F1 thresholds must remain distinguished. |
| E_GraphSAGE | Not applicable: no persistent temporal state. | Its model implementation underwent April fixes, and training runs are dated April 21. The historical inference audit loads compatible checkpoints and reproduces their evaluation metrics; the causes of remaining retraining differences have not been established by the ST-GNN memory diagnosis. |

The original-label factorial and baseline notebooks intentionally retain other
historical choices: mean BCE per window accumulated across windows, original
flow-start graphs/labels, and strict score-greater-than-threshold decisions.
The clean NF-v3 branch changes several of these choices. Consequently,
"memory-reset-corrected ST-GNN" is a justified label for M1_P1;
"all-methodology-corrected ST-GNN" is not established.

### Follow-up on the complete historical comparison

- Static GNN differs only for seeds 42 and 99. The February saved output starts
  seed 42 at train loss 0.2235 / validation AP 0.0313; the new shared-loader run
  starts it at 0.2254 / 0.0498. Historical seed 99 starts at 0.2074 / 0.1767,
  exactly matching the first, interrupted new attempt at the displayed
  precision. Its completed restart starts at 0.2084 / 0.1700. This is direct
  evidence of changed execution conditions; full recovery under an explicitly
  matched loader lifecycle remains to be tested. No persistent-memory bug
  applies to this static architecture.
- E-GraphSAGE differs for all five seeds. The historical and retrained first
  epoch logs match at four decimals; subsequent differences appear early, with
  seed 42 differing in epoch 2 and some other seeds initially differing only
  slightly. The cause remains unresolved. Rounded logs cannot establish exact
  equality, and train-loader versus validation-loader initialization must be
  considered separately when auditing RNG consumption.
- The additional `EdgeGRU_BiasOn` February experiment is **not** the April
  `EdgeGRU_NoX_BiasOn` used in the main comparison. Commit `ea7fe16` contains
  the exact `EdgeGRU_BiasOn_seed42_20260220_115040` output and a `train_epoch`
  without an epoch-start memory reset. Its runner does not add such a reset.
  This older temporal variant therefore has the same source-level issue as
  February ST-GNN and should not be treated as an audited, corrected NoX run.
- The additional `EGraphSAGE_BiasOn_OLD` run IDs are dated April 19, before
  `c592a68` changed the message-passing implementation on April 21. They belong
  to a different implementation and should remain separate from the April-21
  and newly retrained E-GraphSAGE comparisons.

### E-GraphSAGE implementation comparison

An AST comparison excluding documentation confirms that `E_GraphSAGE` in
`5ba36b4` (April 21 notebook training outputs) and in `2262b53` (the pinned
retraining implementation) is identical. `_ESAGEConv` differs only in the
compatibility signature `update(self, aggr_out, **kwargs)` versus
`update(self, aggr_out, x=None)`; both return `aggr_out`. From `d4f4145` onward,
both classes are AST-identical to the pinned version, excluding documentation.
The significant architectural change separates April-19 OLD results from the
April-21 results, rather than April-21 results from the current retraining.
This establishes the versioned architecture history, not the exact Python/PyG
objects that happened to be loaded in an unrecorded historical Colab runtime.

The simple explanation that the first architecture starts fresh persistent
workers while later architectures reuse them cannot by itself account for all
five E-GraphSAGE seeds if both old and new executions consistently retained
their iterators: later seeds would encounter initialized workers on both sides.
Independent train/validation iterator lifecycle, exact first-epoch values,
historical loaded code, library versions, and numerical settings therefore
remain audit targets rather than established causes.
