# Operational guide for the cAPTure graph pilot

## 1. Purpose and scope

This document turns the recommendation for a preliminary graph experiment into
an executable and auditable protocol. The pilot must answer one focused
question:

> Do the five development scenarios contain relational variation that a GNN
> can exploit, and does that information improve early detection at a
> reasonable operational cost?

The pilot is **exploratory and development-only**. It supports a decision about
whether topology should be central to the proposal; it does not confirm the
superiority of an architecture or authorize final generalization claims.

The following are out of scope:

- accessing Test1 or Test2;
- inspecting `train_pub_exf` or `train_user_prop`;
- changing the primary window using model results;
- selecting the best seed;
- claiming that an observed one-seed difference is stable;
- interpreting an aggregate improvement as topological evidence without the
  required ablation controls.

This guide complements, but does not replace, the
[cAPTure experimental plan](capture-experimental-plan.md), the
[versioned manifest](../configs/capture_experiment_v1.yaml), and the
[temporal-state and ablation contract](temporal-state-and-ablation-contract.md).
If they disagree, revise and version the manifest first; do not resolve the
difference through an implicit notebook parameter.

Stage 1 runs through
[`capture_graph_structural_audit.ipynb`](../code/python/notebook/capture_graph_structural_audit.ipynb)
under the contract in
[`capture_graph_structural_audit_v1.yaml`](../configs/capture_graph_structural_audit_v1.yaml).
After the structural gate is recorded, model-ready development graphs are
materialized through
[`capture_graph_materialization.ipynb`](../code/python/notebook/capture_graph_materialization.ipynb)
under
[`capture_graph_materialization_v1.yaml`](../configs/capture_graph_materialization_v1.yaml).

## 2. Frozen contract for the entire pilot

| Element | Decision |
|---|---|
| Data | Only the five development scenarios |
| Scenarios | `train_empty_conn`, `train_qos_mid`, `train_dollar_char`, `train_slash_char`, `train_sub_exf` |
| Fold A | train on `empty_conn`, `qos_mid`; evaluate on `dollar_char`, `slash_char`, `sub_exf` |
| Fold B | train on `dollar_char`, `slash_char`, `sub_exf`; evaluate on `empty_conn`, `qos_mid` |
| Seed | `42` |
| Prediction unit | one packet represented by one edge |
| Windows | fixed, non-overlapping, half-open, five seconds wide |
| Primary origin | first timestamp in each scenario |
| Decision time | `decision_time = window_end` for every model |
| Graph | directed multigraph with parallel edges preserved |
| Nodes | normalized Ethernet MAC; identities used only to construct topology |
| Features | the same preprocessed packet vector within each fold |
| Preprocessing | fitted only on the fold-training scenarios |
| Training weights | declared scenario/class policy fitted only on fold training data |
| Temporal state | chronological order without shuffling; reset at scenario, epoch, evaluation pass, and fold boundaries |
| Window alert | maximum packet score in the window greater than or equal to the threshold |
| Primary budget | one false-alert window per hour |
| Aggregation | scenarios within folds, then folds; packet micro-averages cannot stand alone |

Empty windows are not materialized as graphs, but they count as temporal
exposure in the false-alert denominator. Window-index gaps must be preserved so
that a temporal model sees the actual elapsed time.

## 3. Decisions that must be closed before training

The manifest still marks the memory policy and minimum effect sizes as
unresolved. Before inspecting pilot model results, create a manifest revision
that freezes:

- exact architecture, hidden dimension, dropout, optimizer, learning rate, and
  weight decay for each model;
- memory policy and its half-life or gap cutoff;
- maximum epochs, patience, and minimum early-stopping improvement;
- the rule for the chronological inner checkpoint subset;
- checkpoint metric, independently of the operational model-selection rule;
- maximum parameter count or the capacity-matching rule;
- minimum improvements considered practically meaningful;
- maximum acceptable construction, inference, and memory costs;
- tolerance to the shifted window origin.

Do not choose these values after comparing scores. If the already frozen MLP
configuration is reused as a starting point, record that as a deliberate reuse
rather than a GNN hyperparameter search.

### Early stopping without contaminating the outer fold

The outer fold produces the OOF predictions that will be reported. It must not
select the epoch or checkpoint.

For each fold and each training scenario:

1. order windows chronologically;
2. select the latest admissible chronological validation block using the
   frozen per-scenario rule;
3. train only on the prefix before that block while selecting the epoch;
4. leave the suffix after that block unused during epoch selection;
5. select one `best_epoch_count` per model and fold with the frozen internal
   metric;
6. discard the selection-run weights and reinitialize the same architecture
   with the declared seed;
7. train for exactly `best_epoch_count` epochs on all fold-training scenarios,
   including the former validation block and excluded suffix;
8. evaluate the outer scenarios without readjusting weights, epoch, or
   threshold.

Freeze the block rule, metric, maximum epochs, minimum epochs, patience, and
minimum improvement before the first training run. The inner split selects a
training duration; it is not a third reportable generalization estimate. The
outer scenarios remain inaccessible until the complete-data refit finishes.

## 4. Stage 1: structural audit without training

### 4.1 Auditable construction

Construct graphs one scenario at a time with the same logic that downstream
models will consume. The following mapping must be verifiable for every packet:

```text
packet_id -> scenario -> window_id -> edge_index -> edge_attr -> y
```

Mandatory controls:

- one edge and one target per canonical packet;
- no duplicated, lost, or multiply assigned packet;
- edge order reconstructable through `packet_id` or `source_row_id`;
- non-null endpoints normalized under the schema;
- parallel edges preserved;
- strictly increasing graph timestamps within each scenario;
- no graph or state shared across scenarios or folds;
- exact equality between packet and edge labels;
- hashes for prepared packets, preprocessor, configuration, and constructed
  artifacts.

Run SMOKE first with one scenario from each benign block. Construct all five
scenarios only after reviewing its invariants.

### 4.2 Per-window table

Store one row per nonempty window plus a summary of empty windows, including at
least these fields:

| Group | Minimum fields |
|---|---|
| Identity | scenario, benign block, index, start, end, duration |
| Size | packets/edges, nodes, unique directed pairs, unique undirected pairs |
| Connectivity | weak components, fraction of nodes in the largest component |
| Relational load | directed simple density, mean and maximum multiplicity, fraction of parallel edges |
| Change | node and pair Jaccard against the previous window, new nodes, retained nodes |
| Labels | benign and attack counts, window type, present steps and iterations |
| Protocol | counts by layer and broadcast/multicast, ARP, IPv4/IPv6, TCP/UDP, MQTT/SSH fractions |
| Resources | construction time, serialized bytes, attributable peak RAM |

Recommended definitions, where `m` is the number of edges, `n` the number of
nodes, and `q` the number of unique directed pairs without multiplicity:

```text
simple_density = q / (n * (n - 1))          when n > 1 and self-loops are excluded
mean_multiplicity = m / q                    when q > 0
parallel_edge_fraction = (m - q) / m         when m > 0
node_jaccard(t) = |V_t intersect V_t-1| / |V_t union V_t-1|
pair_jaccard(t) = |P_t intersect P_t-1| / |P_t union P_t-1|
```

Do not calculate multigraph density directly with `m`: it may exceed one and
conflates connectivity with traffic volume. Report simple-graph density and
multiplicity separately. Declare the self-loop policy even when no self-loops
are observed.

Calculate endpoint change in two ways:

- between adjacent wall-clock window indexes, explicitly accounting for gaps;
- between consecutive nonempty graphs, recording the number of empty windows
  between them.

This prevents a comparison that skipped minutes of real time from being
misinterpreted as continuity.

### 4.3 Required summaries

For every scenario and the hierarchical total, produce:

- total, empty, and nonempty window counts;
- percentiles 0, 1, 5, 25, 50, 75, 95, 99, and 100 for nodes, edges, pairs,
  components, multiplicity, and serialized size;
- fraction of benign-only, mixed, and attack-only windows;
- continuous covered duration and occupied-window density;
- distributions of new and retained nodes and pairs;
- frequency of every exactly repeated endpoint or pair set;
- most frequent topologies and the fraction of windows they explain;
- topology metrics split by benign-only, mixed, attack-only, and step, always
  accompanied by the number of windows;
- total time, time per million packets, storage, and peak RAM.

Step comparisons are descriptive. Present them by scenario and window type so
that attack topology is not confused with traffic volume, protocol, or benign
block. If standardized differences or bootstrap intervals are calculated,
resampling must respect scenario and iteration; windows are not independent
observations.

### 4.4 Trivial-shortcut audit

This section does not train the GNN. It searches for signals that could explain
a result without generalizable relational learning:

- attack prevalence by protocol and simple indicator combinations;
- prevalence by MAC direction type: unicast, multicast, broadcast, and group
  node;
- fraction of attack packets identified by a single protocol or port-role rule;
- attack-exclusive endpoints and pairs within each scenario;
- outer-fold coverage of endpoints or pairs seen with attack in fold training;
- endpoint/pair memorization rules constructed only from fold-training data;
- MQTT/non-MQTT and dominant-protocol stratification;
- exact repetition of benign backgrounds or structurally identical windows.

Raw MAC addresses, OUIs, global IDs, scenario names, and step names never enter
`edge_attr`. They may be used as diagnostic metadata in this audit as long as
the report clearly distinguishes leakage or shortcuts from permitted features.

### 4.5 Stage-1 outputs

Stage 1 is complete when the following artifacts exist:

```text
graph_pilot/<run_id>/
  resolved_config.yaml
  provenance.json
  construction_summary.json
  window_metrics.parquet
  scenario_summary.csv
  percentile_summary.csv
  step_topology_summary.csv
  shortcut_audit.json
  resource_summary.json
  figures/
  review.md
```

`review.md` must answer with evidence:

1. Do nodes and links change across windows?
2. Is most traffic carried by a small set of repeated pairs?
3. Is there enough connected structure for message passing?
4. Do step-level differences survive scenario and protocol breakdowns?
5. Does complete construction fit the intended environment?
6. Could a MAC or protocol shortcut dominate any apparent gain?

### 4.6 Structural gate

Proceed to Stage 2 only when construction is correct and fits the resource
budget. Nearly constant topology, trivial two-node components, no shared
neighborhoods, or a simple rule that explains almost all attacks are reasons to
**deprioritize** a GNN. They are not reasons to discard temporal memory, the
MLP, or the existing causal summaries.

Structural variation is a favorable condition, not proof that it is
predictive. The Stage-2 ablations are required to decide whether topology is
useful.

### 4.7 Model-ready materialization after the gate

When Stage 1 is recorded as `PASS_WITH_LIMITATIONS`, materialize the graph
inputs before implementing or running training. This is a separate auditable
stage; it must not refit preprocessing, select a model, or inspect held-out
data.

The materialization contract stores compressed NumPy shards rather than one
file per window. Each shard contains concatenated arrays plus `edge_ptr` and
`node_ptr`, so every graph can be reconstructed without losing its local node
index. For every fold and development scenario it stores:

- one directed edge, label, and `source_row_id` per prepared packet;
- the 103-dimensional `float32` edge view produced by that fold's audited
  training-only preprocessor;
- local `edge_index`, scenario-scoped global node IDs, window index, window
  boundaries, and decision time;
- hashes for topology, labels, features, source-row order, node mapping, and
  every shard.

Raw endpoint identifiers, endpoint hashes, and node features are not persisted;
the node-mapping table contains only global IDs and first-seen source rows. The
same five scenario topologies are materialized under both fold preprocessors.
Topology, labels, source rows, feature names, and the semantic node mapping
must match exactly across folds; transformed feature values are allowed and
expected to differ.

Run materialization as `SMOKE` first and inspect its cross-fold invariants,
storage, time, and peak memory. Approve `FULL_DEV` manually, then keep its run
ID immutable: Stage 2 must consume these shards instead of rebuilding graphs
inside each model implementation.

The durable output is:

```text
graph_materialization_runs/<run_id>/
  run_config.json
  graph_materialization_manifest.json
  run_status.json
  fold_A/<scenario>/
    graph_shard_*.npz
    node_mapping.parquet
    scenario_fold_report.json
    artifact_checksums.json
    run_status.json
  fold_B/<scenario>/
    ...
```

### 4.8 Stage-2 graph-input boundary

Before implementing the training runner, validate the immutable FULL_DEV
materialization with
[`capture_graph_input_audit.ipynb`](../code/python/notebook/capture_graph_input_audit.ipynb)
and
[`capture_graph_stage2_input_v2.yaml`](../configs/capture_graph_stage2_input_v2.yaml).
The notebook copies the small compressed collection from Drive to local Colab
storage and verifies every artifact checksum before repeated access.

The loader exposes one fold/scenario sequence at a time. It reconstructs local
`edge_index`, `edge_attr`, targets, scenario-scoped global node IDs,
`source_row_id`, window coordinates, and decision time as a PyTorch Geometric
`Data` object. Stored nanosecond boundaries remain available for provenance;
the model timestamp is the integer-millisecond floor required by the temporal
state implementation. Because all window boundaries share one scenario origin,
this conversion preserves every five-second interval and empty-window gap.

The physical presence of every scenario below both fold directories does not
authorize mixing them. The loader derives and enforces these roles:

| Scenario | Fold A | Fold B | OOF fold |
|---|---|---|---|
| `train_empty_conn` | train | validation | B |
| `train_qos_mid` | train | validation | B |
| `train_dollar_char` | validation | train | A |
| `train_slash_char` | validation | train | A |
| `train_sub_exf` | validation | train | A |

Run the bounded first/middle/last sample before approving the full loader scan.
The full scan must establish all of the following before training code is
enabled:

- one valid graph object for every materialized window representation;
- contiguous `source_row_id` values and exact packet-edge conservation;
- strictly increasing windows and timestamps within every scenario;
- exactly one training role and one OOF validation role per development packet
  across the two folds;
- an exact per-edge join from `source_row_id` to the bound prepared packets,
  yielding one bijective scenario-scoped `global_node_id` to normalized MAC
  mapping with no unresolved nodes or conflicts;
- equality between the mapping contract reconstructed independently from graph
  edges and prepared packets and the semantic mapping hash stored by both folds;
- no persisted node features, held-out access, or model optimization.

The recovered raw-MAC lookup is diagnostic provenance, not a model input. It is
stored only below the immutable graph-input audit run in
`node_identity_lookup/`, with a checksum manifest. Training loaders must not
read that directory. This prevents the CIC2018 failure mode in which a decoder
was regenerated independently with a different ID permutation.

Temporal state must be reset between the scenario datasets. A single loader
spanning multiple scenarios without an explicit reset is outside the contract.

### 4.9 Training preflight and inner-checkpoint split

Before implementing the optimizer, run
[`capture_graph_training_preflight.ipynb`](../code/python/notebook/capture_graph_training_preflight.ipynb)
under
[`capture_graph_training_preflight_v2.yaml`](../configs/capture_graph_training_preflight_v2.yaml).
This preflight performs no optimization and never uses an outer-validation
scenario to choose a split.

The v1 preflight run `20260928T201318_972670Z_graph_training_preflight`
demonstrated that a common terminal tail is unsuitable: late benign-only
periods leave some scenarios without enough attack support. The v2
split remains scenario-local and chronological but uses a bounded block. For
each scenario it first searches for the latest admissible block spanning 20%
of wall-clock windows and falls back to 25% only when no 20% block passes. Both
block boundaries must preserve every `(attack_step, sequence_id)` iteration.
The training portion is the prefix before the block. The suffix after the block
is excluded from checkpoint selection and is restored only during the final
complete-data refit.

Every selected block must pass the declared packet-label, window, complete
iteration, and distinct-step minimums. Selection uses labels and iteration
metadata only; it never uses model scores. The latest admissible location is
chosen so the internal check remains as temporally late as the support
constraints permit without expanding validation toward half the scenario.

For each model and fold, checkpoint selection is capped at 60 epochs, cannot
stop before 10 epochs, uses patience 10 and an absolute AP improvement of
`0.0001`, and maximizes the unweighted mean scenario AP. The resulting
one-based `best_epoch_count` is the only value transferred to the final refit;
selection-run weights are not reused.

The selection-run preprocessor and loss weights must be fit only on the
inner-training prefixes. After `best_epoch_count` is frozen, the final-refit
preprocessor and weights are fit again on all fold-training scenarios. The
existing fold graph materialization remains authoritative for topology,
labels, ordering, and complete-data refit features; separate selection feature
tensors are required so the internal validation block and excluded suffix do
not influence normalization or loss weighting.

The preflight also verifies prepared-packet/graph counts at each boundary and
runs inference-only synthetic forward checks for every planned variant. It
must demonstrate one finite logit per edge, unchanged shared inputs, temporal
state creation, and complete scenario reset. A passing run remains
`review_required` until its descriptive split tables are approved manually.
Approval freezes the inner split and authorizes the full training contract and
runner to be implemented; it does not itself authorize optimization.

### 4.10 Selection-feature materialization

The approved preflight run is
`20260929T131750_353987Z_graph_training_preflight_v2`, with report SHA-256
`478b6c4cae0ee27595c2ddd7fea73e2c0c3914f14c78f60ac4534af77badafed`.
Before implementing optimization, run
[`capture_graph_selection_materialization.ipynb`](../code/python/notebook/capture_graph_selection_materialization.ipynb)
under
[`capture_graph_selection_materialization_v1.yaml`](../configs/capture_graph_selection_materialization_v1.yaml).

This data-only stage fits one preprocessor per fold using only the approved
inner-training prefixes. It materializes complete sequences only for that
fold's training scenarios so the runner can later slice the inner-training and
validation blocks without rebuilding topology. The post-validation suffix is
transformed for alignment auditing but remains forbidden during epoch
selection. Outer-validation scenarios are not rematerialized because their
already reviewed complete-fold features are used only after the final refit.

Every selection sequence must exactly match the complete-fold reference in
graph count, edge count, topology hash, target hash, source-row hash, and node
mapping contract hash. A feature hash difference is recorded diagnostically,
not required: two fitted preprocessors could theoretically produce identical
tensors without violating the fit-scope contract. This stage performs no model
optimization and requires a separate manual decision before runner binding.

The completed immutable run is
`20260929T134737_484622Z_graph_selection_materialization`, with manifest
SHA-256
`846d21671d6c657903427fefe665e8d1919c74859703a068a4940d02d8c09633`.
It was approved for runner binding after all required alignment invariants
passed. The approval explicitly leaves model training unauthorized.

### 4.11 Training-runner binding and authorization

Run
[`capture_graph_training_binding.ipynb`](../code/python/notebook/capture_graph_training_binding.ipynb)
under
[`capture_graph_training_v1.yaml`](../configs/capture_graph_training_v1.yaml)
before the first optimizer step. This gate stages and checksum-verifies both
graph collections, reconstructs the exact graph ranges for inner training,
inner validation, and the excluded suffix, and freezes twelve jobs: six models
crossed with two folds.

The contract deliberately reuses the existing untuned neural recipe: seed 42,
hidden dimension 64, node projection dimension 16, dropout 0.2, AdamW with
learning rate `0.001` and weight decay `0.00001`, and ten chronological graph
windows per truncated-backpropagation block. Recurrent models use the primary
timestamp-aware `exponential_decay` policy initialized to a 20-window
half-life. These choices are fixed before graph-model scores and do not
constitute a cAPTure hyperparameter search.

The binding report also records separate scenario/class weights for epoch
selection and complete-data refit. Each `(scenario, binary class)` cell has
equal total training mass and the weights have mean one in their respective
fit scope. Inner validation and outer validation remain unweighted.

No model is instantiated by this gate. A manual
`graph_training_authorization.json`, bound to the exact binding-report and
job-plan hashes, is required before any backward pass. That authorization can
permit only the development pilot; Test1 and Test2 remain prohibited.

The approved binding run is
`20260929T144234_934044Z_graph_training_binding`. Its binding-report SHA-256 is
`710593fe1a10cbe2342fe6a4dc52c3fa971eeff90bdae6edacb6f3d333457995`
and its job-plan SHA-256 is
`c3d64778323832919ad1f778d42a50a4d626ae80ee6c074bddc0e45cff5407c8`.
The associated authorization permits the one-seed development jobs and
explicitly denies held-out-scenario access.

## 5. Stage 2: one-seed development comparison

### 5.1 Minimum matrix

Every model receives exactly the same packets, features, folds, training
weights, windows, and decision times.

| Variant | Conceptual class | Available information | Question |
|---|---|---|---|
| Edge MLP | `SimpleMLP` | current packet | What can a network do without time or relations? |
| EdgeGRU | `EdgeGRU_Baseline_NoX` | packet plus per-node memory | Does memory add value without GAT? |
| StaticGNN | `StaticGNN_Identity` | packet plus current topology | Does current message passing add value without recurrent memory? |
| ST-GNN | `ST_GNN_Identity` | packet plus topology plus memory | Do structure and time complement each other? |
| ST-GNN without GAT | `ST_GNN_Identity(use_topology=false)` | packet plus endpoint aggregation plus memory, without GAT message passing | Main control for the topological layers |
| ST-GNN without direct `edge_attr` | `ST_GNN_Identity(use_direct_edge_attr=false)` | edge features only through identity/aggregation and GAT | Does the classifier depend on the direct packet shortcut? |

The last variant is conditional on cost and runs only if declared before
inspecting the minimum-matrix results.

**Important conceptual detail:** in the current implementation,
`use_topology=false` bypasses GATv2 layers but retains local incoming/outgoing
edge aggregates when constructing node identity. Call it “without GAT” or
“without message passing,” not “without all topology.” Similarly,
`use_direct_edge_attr=false` does not remove packet features from identity
construction or GAT messages. Edge MLP and EdgeGRU provide the controls without
message passing needed for interpretation.

### 5.2 Experimental equality

Before launching a run, a contract check must demonstrate that all variants
have:

- identical packet IDs and ordering in training and evaluation;
- identical `edge_attr`, target, and weight per packet;
- identical window assignment;
- identical origin and `window_end`;
- identical fold and inner checkpoint subset;
- the same operational threshold procedure based on OOF, never on Test;
- correct resets and temporal ordering;
- reported parameters and FLOPs, even when not identical.

Do not add XGB-P+T summaries to only one variant. If a second matrix with the
six causal history features is desired, give the same vector to every variant
and label it as a separate experiment.

### 5.3 Training and persistence

The executable Colab entry point is
[`capture_graph_training.ipynb`](../code/python/notebook/capture_graph_training.ipynb).
It runs one `model + fold` job at a time and defaults to a no-op until
`RUN_TRAINING=True` is set. Start with `edge_mlp__fold_A`; use its measured
cost to schedule the rest of the matrix.

For every fold/model combination:

1. set seed `42` for Python, NumPy, PyTorch, and CUDA;
2. load only the fold-training scenarios;
3. fit selection preprocessing and weights only on the inner-training prefixes;
4. train chronologically whenever temporal state exists;
5. select `best_epoch_count` using only the predeclared inner blocks;
6. reinitialize with the same seed, refit preprocessing and weights on all
   fold-training scenarios, and train for exactly `best_epoch_count` epochs
   without early stopping;
7. reset all memory;
8. run inference exactly once on every outer scenario;
9. persist one score per packet with its temporal and evaluation keys;
10. record duration, peak RAM/VRAM, parameter count, and checkpoint size.

Each job persists local recovery state after every epoch and a durable Drive
recovery point every five epochs and at phase completion. Recovery is allowed
only when the job contract, binding report, authorization, data manifests,
training configuration, and runner code hashes still match. Selection weights
are never reused: the runner instantiates a fresh seed-42 model for the
complete-data refit. Outer-fold graph tensors and evaluation metadata are read
only after that refit checkpoint exists.

Non-temporal models may shuffle examples for optimization only if the example
set and weights remain unchanged. For inference comparability, every model is
evaluated as the same ordered sequence of windows.

Measure pure inference time and end-to-end inference time separately. At a
minimum, report seconds per million packets and per-window latency percentiles
after an explicit warm-up. Synchronize CUDA before and after each measurement.

### 5.4 Threshold and operational semantics

A window alerts when:

```text
max(packet_score in the window) >= threshold
```

For each model, obtain its development threshold from its OOF predictions: the
smallest threshold whose worst fold mean of scenario false-alert rates meets
one false-alert window per hour. Preserve the declared `nextafter` tie handling
and `float64` comparison rules.

This calibration and its evaluation use the same OOF predictions, so pilot
operational results may be optimistic. They are suitable for screening, not a
confirmatory estimate.

## 6. Required metrics

### 6.1 Detection and operation

Report by model, fold, scenario, and step:

- packet ROC-AUC and PR-AUC as diagnostics;
- packet precision, recall, and FPR at the operational threshold;
- false-alert windows per hour;
- total, detected, and missed iterations;
- iteration coverage with a correct alert strictly before the final malicious
  packet in the iteration;
- coverage before the declared terminal action;
- coverage by step type;
- first alert time and lead time;
- number of scenarios and steps in which the difference against the control
  changes sign.

Temporal definitions:

```text
alert_time = window_end of the first window containing a malicious packet above threshold
timely_step = alert_time < timestamp of the iteration's final malicious packet
early_terminal = alert_time < onset of the first declared terminal action
terminal_lead_time = terminal_onset - alert_time
```

Misses remain misses; they are not converted to detections at step completion.
Summarize lead time only alongside coverage and miss counts. Report medians and
percentiles in addition to the mean because the five-second window quantizes
time.

### 6.2 Cost

Always separate:

- one-time graph-construction time;
- training time through the selected checkpoint;
- pure inference time;
- end-to-end inference time;
- peak RAM and VRAM;
- graph and checkpoint disk size;
- parameter count and windows/packets processed per second.

### 6.3 Window-origin sensitivity

After closing the primary matrix, reconstruct windows with a `+2.5 s` shift.
The minimum sensitivity includes Edge MLP and the variants supporting a
topological conclusion; repeating the entire ladder is not automatically
required if cost is prohibitive.

Do not retune architecture or window width. Apply the predeclared protocol and
report:

- absolute change in timely coverage;
- change in false alerts per hour;
- change in lead time;
- change in graph size and cost;
- whether the sign of the main topological comparisons is preserved.

The shifted origin is a development sensitivity, not an opportunity to replace
the primary origin retrospectively.

## 7. Causal interpretation of comparisons

| Comparison | Main evidence | Limitation |
|---|---|---|
| EdgeGRU - Edge MLP | value of per-endpoint memory | architecture also changes |
| StaticGNN - Edge MLP | value of current aggregation/message passing | capacity and optimization may differ |
| ST-GNN - StaticGNN | incremental value of memory with structure | requires aligned configurations |
| ST-GNN - EdgeGRU | incremental value of GAT in a temporal model | does not perfectly isolate interactions |
| ST-GNN - ST-GNN without GAT | value of GATv2 layers within one family | the control still uses endpoint aggregation |
| ST-GNN without direct `edge_attr` - ST-GNN | dependence on the classifier's direct shortcut | features remain in identity and messages |

An AUC-only gain does not demonstrate operational utility. Relevant evidence
is a consistent improvement in coverage or lead time at the same false-alert
budget, accompanied by scenario/step breakdowns and cost.

## 8. Pilot decision rule

Before training, complete the bracketed manifest fields:

```text
minimum_timely_coverage_improvement = [value]
minimum_terminal_coverage_improvement = [value]
minimum_lead_time_improvement = [value and statistic]
minimum_scenarios_with_positive_sign = [value out of 5]
maximum_shifted_origin_degradation = [value]
maximum_inference_cost = [value]
maximum_peak_ram_vram = [value]
```

### Topology should advance

Only when all of the following hold:

- StaticGNN or ST-GNN beats its non-topological control on a predeclared
  operational metric at the same false-alert budget;
- the difference reaches the frozen minimum practical effect;
- the sign does not depend on one scenario, step, or protocol;
- the main conclusion survives the shifted origin;
- raw MAC identities or a trivial protocol rule do not explain the result;
- construction and inference meet the resource budget.

The next step is multi-seed development confirmation before freezing the final
pipeline. Test1 and Test2 remain closed.

### Evidence favors time but not topology

If EdgeGRU improves over Edge MLP but StaticGNN/ST-GNN do not improve over the
controls without GAT, prioritize temporal memory or the existing `history`
baseline. The GNN may remain a secondary analysis rather than a central
contribution.

### Topology should be deprioritized

If the audit shows nearly constant or trivial graphs and the ablations do not
provide robust operational gains, document the negative result and avoid a
broad GNN search. This is a valid pilot conclusion, not an execution failure.

### Inconclusive result

Mark the result inconclusive if there are contract failures, insufficient
memory, severe origin sensitivity, dependence on one scenario, or differences
smaller than the predeclared minimum effects. Do not resolve it by looking at
Test.

## 9. Stage-2 deliverables

```text
graph_pilot/<run_id>/
  resolved_config.yaml
  provenance.json
  graph_contract_report.json
  models/<model>/<fold>/
    checkpoint.pt
    training_history.json
    timing.json
    resource_usage.json
    predictions.parquet
  threshold_selection.json
  packet_metrics.csv
  window_metrics.csv
  iteration_metrics.csv
  step_metrics.csv
  scenario_metrics.csv
  origin_shift_metrics.csv
  comparison_table.csv
  figures/
  pilot_decision.md
```

Every `predictions.parquet` row must retain at least:

```text
model, fold, scenario, packet_id, source_row_id, window_id,
window_start, window_end, packet_timestamp, binary_label,
attack_step, sequence_id, score
```

`attack_step` and `sequence_id` are evaluation-only metadata. They are never
given to the model.

## 10. Execution checklist

### Before construction

- [ ] Test1, Test2, and prohibited scenarios remain inaccessible.
- [ ] Manifest, schemas, and prepared artifacts have verified hashes.
- [ ] Self-loop, broadcast, multicast, and non-IP policies match the manifest.
- [ ] Structural metrics and formulas are frozen.
- [ ] Storage, RAM, and time budgets are declared.

### Before training

- [ ] Stage 1 has been reviewed and the structural gate documented.
- [ ] SMOKE and FULL_DEV graph materialization passed all cross-fold invariants.
- [ ] The immutable FULL_DEV materialization run ID is recorded.
- [ ] The Stage-2 sample and full graph-input audits passed.
- [ ] The training preflight and its manual decision are complete and bound to
      the reviewed graph-input audit.
- [ ] Training and validation datasets are selected only through declared fold roles.
- [ ] Exact configurations and minimum effect sizes are versioned.
- [ ] The bounded chronological blocks and complete-data epoch-refit rule are frozen.
- [ ] Inner-training-only selection features passed alignment review.
- [ ] The runner-binding report and twelve-job plan passed manual review.
- [ ] Training authorization is bound to the exact binding-report and job-plan hashes.
- [ ] A synthetic contract check verifies packet-edge-output correspondence.
- [ ] A synthetic contract check verifies resets, gaps, and temporal order.
- [ ] Every variant receives the same feature view.
- [ ] The exact meaning of every ablation has been verified.

### Before deciding

- [ ] There is one OOF prediction per packet and model.
- [ ] Thresholds use development OOF only and the one-false-alert-per-hour rule.
- [ ] Timely and terminal coverage use strict inequality.
- [ ] Misses, denominators, and sample sizes are visible.
- [ ] Results are separated by fold, scenario, step, and protocol.
- [ ] Time, RAM/VRAM, and storage are reported.
- [ ] The predeclared origin sensitivity has been executed.
- [ ] The decision uses the frozen criteria, not AUC alone.
- [ ] The report explicitly states that the pilot uses one seed.

## 11. Compact table for `pilot_decision.md`

| Question | Evidence | Result | Decision |
|---|---|---|---|
| Is there real node and link variation? | percentiles, Jaccard, repeated topologies |  |  |
| Are there non-trivial neighborhoods? | components, largest component, degrees, multiplicity |  |  |
| Are there MAC/protocol shortcuts? | rule and stratum audit |  |  |
| Does topology improve timely coverage? | StaticGNN/MLP and ST-GNN/without GAT |  |  |
| Does memory improve coverage or lead time? | EdgeGRU/MLP and ST-GNN/StaticGNN |  |  |
| Does detection precede the terminal action? | terminal coverage and lead time |  |  |
| Does it meet one false alert per hour? | worst fold mean |  |  |
| Is it stable under the `+2.5 s` origin? | window sensitivity |  |  |
| Does it fit resource budgets? | time, throughput, RAM/VRAM, disk |  |  |
| Should it advance to multi-seed confirmation? | all predeclared criteria |  |  |
