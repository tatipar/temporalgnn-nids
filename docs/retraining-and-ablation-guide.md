# Reconstruction, Retraining, and Ablation Guide

## Purpose and scope

This guide defines a reproducible restart of the binary intrusion-detection
experiments on the corrected **NF-CSE-CIC-IDS2018-v3** dataset. Its outcome is a
fair comparison of flow-classification models.

The core retraining branch must not contain heuristic MITRE attribution or
synthetic lateral movement. Once training and evaluation are frozen, a separate
post-hoc analysis may measure alerts preceding the documented Day-2 Nmap scan;
its requirements are defined in section 8. No lateral-movement claim is allowed
without independent ground truth.

More general attack-phase or lateral-movement research belongs in a separate
line of work using a dataset with a phase field independent of model inputs,
such as the DAPT2020 pilot with `Stage`.

## 0. Non-negotiable principles

- The prediction unit is a **completed flow**. Final flow statistics are not
  available at flow start time.
- Feature selection, hyperparameter selection, checkpoint selection, and
  threshold selection use only train/validation data. Test1 and Test2 are never
  used to make those decisions.
- Every compared model receives the same graphs, splits, corrected labels,
  features, and evaluation protocol.
- Test1 is reported as an *intra-campaign chronological holdout*; Test2 is an
  *out-of-day holdout*. Neither proves generalization to independent campaigns.
- Flow-level metrics require a loss weighted by flow, not by graph window.

## 1. Create a clean branch without destroying previous work

Do not delete or rewrite `feat/fair-comparison-experiments`; it is the
historical record. Do not add the local PDF or `Dapt2020_pilot/` to the clean
NF-v3 branch without an explicit decision.

```bash
git switch main
git switch -c feat/fair-retrain-clean
```

The branch should retain or rewrite only:

- auditable NF-v3 graph construction;
- shared models and training code;
- comparative evaluation;
- tests and data manifests.

`main` already contains `code/python/utils/mitre.py` and
`code/python/notebook/attack_profile_extractor.ipynb`. Remove them explicitly
in the first cleanup commit; this does not alter `main`:

```bash
git rm code/python/utils/mitre.py \
       code/python/notebook/attack_profile_extractor.ipynb
git commit -m "chore: remove heuristic MITRE analysis from retraining branch"
```

Do not restore or add:

- the historical heuristic MITRE notebook;
- `utils/synthetic_lm.py`, its tests, or the synthetic experiment notebook;
- historical results, checkpoints, and notebook outputs as if they belonged to
  the new protocol.

Create a distinct branch only when the DAPT phase-ground-truth study resumes:

```bash
git switch main
git switch -c feat/dapt2020-forensic-pilot
```

## 2. Freeze label corrections before generating graphs

Label correction must be versioned code, not manual CSV editing. Implement a
`prepare_nfv3.py` step, or equivalent, that produces a corrected dataset and a
manifest.

The manifest must contain:

- SHA-256 hashes of every input CSV;
- correction-rule version/source;
- rule applied and affected-row counts per label;
- class counts before and after correction;
- exact binary-target definition;
- identity, time, label, and metadata columns excluded from `edge_attr`;
- hash of the corrected output CSV.

Review samples of each corrected label manually and freeze the rule before
continuing. Any later rule change defines a new dataset: regenerate graphs,
scaler, hyperparameters, and results.

## 3. Temporal contract and graph construction

### 3.1 Edge timing and window assignment

The historical builder groups flows by `FLOW_START_TIME`. That is incorrect when
features include final bytes, packets, duration, or IAT statistics. For each
row define:

```text
flow_start     = FLOW_START_TIME
flow_end       = FLOW_END_MILLISECONDS
window_start   = start of the 30-second half-open window containing flow_end
window_end     = window_start + 30 seconds
decision_time  = window_end
```

`FLOW_END_MILLISECONDS` is authoritative for graph assignment. An audit of the
frozen corrected input found that it differs from
`flow_start + FLOW_DURATION_MILLISECONDS` by at most one millisecond,
consistent with exporter rounding. Duration remains an independent input
feature.

Windows are aligned to Unix epoch boundaries (`00` and `30` seconds of each
minute), not to the first row in a CSV or to a chunk boundary.

The edge is included in the window closing at `decision_time`. For example, a
flow running from 13:57:00 to 14:03:40 belongs in
`[14:03:30, 14:04:00)` and is classified at 14:04:00, never in a 13:57 window.
Even if a flow spans several windows, it appears as exactly one edge: in the
window containing its completion time. Final bytes, packets, duration, and IAT
statistics are not available before then. With half-open windows, a flow ending
exactly at 14:04:00 belongs to `[14:04:00, 14:04:30)` and has
`decision_time = 14:04:30`.

Duration remains a valid feature and retains the existing `log1p` plus scaling
transformation. Only its graph-assignment time changes.

Persist per-flow provenance in a table or graph metadata:

```text
flow_id, source_file, source_row_id, flow_start, flow_end,
decision_time, window_start, window_end, split
```

`flow_id` must be globally unique across all corrected input files. When
`source_row_id` is only unique within an input file, use the pair
`(source_file, source_row_id)` as its stable provenance key or derive a
separate globally unique `flow_id` before graph construction.

`data.timestamp` must represent the decision/window-close time, not only the
window start.

Do not introduce an assumed inference or serving latency into graph assignment.
The historical CSVs cannot measure exporter, ingestion, queueing, or alerting
delay. Report instead `window_wait = decision_time - flow_end` (the delay
introduced by the 30-second batching policy) and measure model forward-pass
latency separately on documented hardware. An end-to-end operational latency
requires a separate streaming or replay study.

### 3.2 Splits and scaling

- Define train, validation, and Test1 using `decision_time`, not `flow_start`.
- Derive provisional chronological cutoffs from the Day-1 decision-time span,
  then round each cutoff upward to the next 30-second window boundary. With
  the convention `decision_time < cutoff`, this preserves the intended split
  membership while making every split boundary a graph boundary.
- Fit `StandardScaler` only with flows available in train.
- Apply the frozen scaler to validation, Test1, and Test2.
- A flow ending after the train/validation cutoff belongs to the split determined
  by its `decision_time`, even if it started earlier.
- Keep 30-second windows initially. Window duration is a separate experiment
  after this protocol has stabilized.

### 3.3 IP-to-node-ID mapping contract

Global node IDs are opaque keys for graph-local nodes and temporal memory;
they are not numerical features, learned IP embeddings, or behavioural labels.
Their only purpose is to identify a node when it reappears within one temporal
stream.

- Construct one append-only map for all Day-1 splits (`train`, `val`, and
  `Test1`) and a separate append-only map for Day 2 (`Test2`). Do not transfer
  temporal memory between splits or days.
- Start each day map empty and assign an ID when a valid, canonicalized IP first
  appears while processing that day's flows in chronological order. Never
  reuse or reassign an ID. This mirrors online operation without exposing
  future flow content to an earlier graph.
- Persist the exact final map used by the builder in both directions
  (`ip_to_id` and `id_to_ip`), as JSON plus a human-readable two-column table.
  Record their hashes, entry counts, creation policy, and IP-normalization
  policy in the graph manifest.
- Decode an edge only with the map declared by the manifest for that graph
  collection. Never regenerate a map independently for a later analysis.
- Canonicalize and validate endpoint addresses before mapping. The manifest
  must count rows excluded for missing, unparsable, or explicitly disallowed
  endpoint values (including any policy for `0.0.0.0`); exclusions must never
  be silent.

### 3.4 Feature profiles

Define each profile by name and ordered columns, store it as JSON, and use the
same order for every model.

`nfv3_extended` uses an enriched 40-dimensional profile:

- 19 numerical features: bytes, packets, duration, IAT, IP lengths,
  retransmissions, TCP windows, and TTL;
- eight destination-port categories;
- five protocol categories;
- eight unscaled TCP control-bit indicators: `FIN`, `SYN`, `RST`, `PSH`,
  `ACK`, `URG`, `ECE`, and `CWR`.

`TCP_FLAGS` is a cumulative bitmask, not an ordered magnitude. The extended
profile therefore decodes it into the eight multi-hot indicators above; more
than one bit may be active for an edge. These indicators remain in `{0,1}` and
are not passed through `log1p` or `StandardScaler`.

`portable_core` is the first minimal, deployable profile:

- `IN_BYTES`, `OUT_BYTES`, `IN_PKTS`, `OUT_PKTS`;
- `FLOW_DURATION_MILLISECONDS`;
- protocol and destination-port category;

It therefore has 18 dimensions: five numerical values, eight destination-port
categories, and five protocol categories. `TCP_FLAGS` is deliberately excluded
from this profile: it cannot be reliably reconstructed from the retained flow
statistics and is not consistently available in compatible exporters. It
remains an `nfv3_extended` feature only.

The fixed destination-port taxonomy is: `web_http_proxy`, `admin_remote`,
`windows_smb_rpc`, `infrastructure`, `database`, `other_privileged`,
`other_high`, and `not_applicable_or_zero`. The final category is reserved for
destination port zero; it must not be interpreted as a privileged service.
`infrastructure` includes both network services and identity services such as
Kerberos and LDAP. The schema JSON is authoritative for the explicit port lists
in each named category.
Ports must be valid integers in `0..65535`; protocol
values must be valid integer IANA protocol numbers in `0..255`. `TCP_FLAGS`
must be an integer bitmask in `0..255`, and a non-zero mask on a non-TCP flow is
treated as a data-quality failure. Invalid values are data-quality failures,
not members of an `other` category.

Generate and store separate graph collections for each profile in the same
builder run. A profile's JSON schema must record its name, exact ordered
`edge_attr` columns, categorical encoding definitions, transformations,
dimension, and SHA-256 hash. The schema order, rather than an informal list in
documentation, is authoritative for every model. Each profile has its own
train-fitted scaler and scaler hash.

Bytes/s and packets/s may be optional derived features, with explicit handling
of zero duration. Do not mix CICFlowMeter and NF-v3 extractors within one run:
adopting CICFlowMeter requires rebuilding all flows and retraining from scratch.

### 3.5 Graph artefacts and storage contract

Version graph output directories and keep the profile name explicit, for
example:

```text
graphs/<graph_version>/nfv3_extended/{train,val,test1,test2}/
graphs/<graph_version>/portable_core/{train,val,test1,test2}/
graphs/<graph_version>/mappings/
graphs/<graph_version>/manifests/
```

Each non-empty graph must contain `edge_index`, `edge_attr`, `y`,
`global_node_ids`, `timestamp`, `window_start`, `window_end`, the feature
profile name, and the schema hash. The per-flow provenance table is stored next
to the graph collections rather than discarded after serialization. The graph
manifest records corrected-data and label-manifest hashes, feature-profile and
scaler hashes, mapping hash, split cutoffs, window policy, row and class counts,
and graph-file hashes.

Do not serialize empty windows as graph files. Their elapsed time is represented
by the difference between consecutive non-empty graph timestamps when an IP
reappears. The builder audit must still verify that a gap produces no duplicate
or misassigned flow window.

Do not persist an all-ones dummy node-feature matrix. The new graph schema has
no node attributes: node identity for StaticGNN and ST-GNN is induced from
edge attributes, while models that require a constant initial state must create
it internally and document that model policy. This keeps dummy constants from
being mistaken for input features.

The production builder must be a tested script, with a thin Colab notebook only
for Drive paths and execution. It must write versioned output, maintain a
resumable checkpoint/state for long Drive-backed runs, and publish the final
manifest only after all audits pass. Historical graph-creation and entropy
notebooks are not valid production builders.

To avoid rewriting a growing IP map for every graph, persist the day-scoped
IP-to-ID map together with build state every 200 newly built windows and at
every day boundary. Publish the map before the state. Resume must treat the
saved decision time as authoritative, reconstruct the append-only map by
replaying endpoint registration for earlier windows in chronological order,
and then rebuild any graph files written after the last published checkpoint.
This must produce exactly the same IDs as an uninterrupted build.

### 3.6 Mandatory automated checks

Implement these both as unit tests and as a graph-builder audit:

- every provenance key and `flow_id` occurs in exactly one graph;
- for every edge, `flow_end < decision_time`, the flow end belongs to the
  declared half-open window, and the assigned window closes at
  `decision_time`;
- graph timestamps are strictly increasing;
- splits are disjoint and chronologically ordered by `decision_time`;
- scaler fitting uses train indices only;
- feature order/dimensions, `edge_index`, `edge_attr`, and `y` agree;
- every graph's stored profile/schema hash agrees with its collection manifest;
- every `global_node_ids` entry resolves through the declared day-specific map,
  and sampled decoded edge endpoints match the raw provenance rows;
- before any graph files are written, a full corrected-input preflight reports
  invalid ports, protocols, and numeric values; the destination-port category
  distribution; port-zero counts by protocol; and the most common ports in
  each residual category. Invalid port, protocol, or numeric values fail the
  preflight rather than silently entering an `other` category;
- the preflight reports invalid source endpoints, invalid destination endpoints,
  and their row-level union separately. IPv4 and IPv6 endpoints are
  canonicalized and retained; it reports the exclusion reasons (`missing`,
  `non_parseable`, `zero_ipv4`, or `unspecified_ipv6`) so endpoint counts
  cannot be double-counted across both columns;
- no `NaN`, infinity, or invalid `log1p` inputs are present;
- no label, IP, timestamp, flow ID, or metadata column is used as input;
- total-flow and positive-flow counts are conserved from corrected CSV through
  graph splits, including documented endpoint exclusions.

The production preflight records total, retained, and excluded row/positive
counts globally and per source file. The final graph audit must match retained
rows and positives against serialized edges for every profile and day; a
conservation mismatch prevents publication of the graph manifest.

Unit-test a short flow, a flow spanning several windows, a flow crossing a split
cutoff, an empty window, an IP returning after a long idle period, a new IP
first appearing in validation or Test1, and source-row identifiers repeated
across two input files.

## 4. Fix the training core before final training

### 4.1 Loss and optimization

Use:

```python
criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=..., reduction="sum"
)
```

Accumulate `loss_sum` and divide by the total number of edges in the TBPTT block
before `backward()`. A 5,000-flow window then has the appropriate weight
relative to a 3-flow window, consistent with flow-level metrics. Record
loss-per-flow as well.

Use gradient clipping and record gradient norms when stability is a concern. The
DAPT2020 pilot already demonstrates this loss pattern; use it as an
implementation reference without mixing its data or labels into NF-v3.

### 4.2 Explicit temporal configuration

Remove logic such as:

```python
is_temporal = 'GRU' in experiment_name or 'ST' in experiment_name
```

Every model specification must declare `temporal: true|false`; train,
validation, test, and inference receive this explicit value. Add a test that
fails when a model signature and this configuration disagree.

### 4.3 Thresholds and evaluation

- Use `>= threshold` both when selecting and applying a threshold.
- State the strategy explicitly: `max_f1` or `constrained`/operational. Do not
  pass `min_precision` while the strategy remains `max_f1`.
- When relevant, store both thresholds and state which one produced each metric.

### 4.4 Supporting utility modules

The following modules did not differ from the historical branch, but require
small changes to support the new protocol:

- `metrics.py`: apply predictions with `probs >= threshold`, matching
  `precision_recall_curve` threshold semantics.
- `visualization.py`: extend `MODEL_NAME_MAPPING`, model order, and colour
  mappings for `EdgeGRU_Baseline_NoX` and `E_GraphSAGE`; use one fixed mapping
  for every comparison figure.
- `datasets.py`: keep its ordered graph loader, but validate the expected graph
  schema and optionally assert that timestamps are monotonic before temporal
  training/evaluation.
- `experiment.py`: persist the dataset-manifest hash, label-correction version,
  feature-profile name/hash, temporal-memory policy, and graph/scaler hashes in
  every run record.

### 4.5 Temporal memory and elapsed time

Per-IP memory must store both hidden state and last observed `decision_time`.
When retrieving a node state:

```text
delta_t = current_decision_time - last_seen_time
h_prev  = h_prev * exp(-softplus(lambda) * delta_t / time_scale)
```

`lambda` may be learned or initially fixed. At minimum, before learned decay:

- reset memory at train/validation/test boundaries;
- reset it at day boundaries;
- reset an IP when `delta_t` exceeds a documented maximum.

Empty windows need not execute the model: their elapsed time is represented by
`delta_t` when an IP reappears. Add train/evaluation assertions that timestamps
never decrease, and record reset counts and gap distributions.

### 4.6 Node identity and the target edge

The current identity representation averages incoming/outgoing edge attributes
from the same window, including the edge being classified. This is not label
leakage, but it mixes the edge's own evidence with neighbourhood context.

For this round implement the efficient **lagged identity** variant:

- to classify window `t`, construct each IP identity from aggregates of edges
  available at `t-1`;
- use edges from `t` in GATv2 and in the edge classifier as usual;
- use zero identity or a learned fixed vector for the first window;
- never transfer identity/memory state across a split reset.

Perfect target-edge exclusion would require a different context for every edge
and is not required now. The lagged-identity ablation quantifies its impact.

## 5. Recalibrate after graph regeneration

Historical `pos_weight` and bias values are invalid after changing labels,
availability time, and potentially split prevalence.

For every candidate `pos_weight`, derive initial output bias from train positive
prevalence `p_train`:

```text
bias_init = log(pos_weight * p_train / (1 - p_train))
```

Bias is not a free hyperparameter. Use a small predeclared `pos_weight` grid,
for example:

```text
1, sqrt(n_negative / n_positive), n_negative / n_positive
```

Select only with validation using AP for checkpoint selection and a
validation-selected threshold. Run one development seed first; after freezing
the choice, never revisit it using Test1 or Test2.

## 6. Experiment matrix

### 6.1 Base models

The final model family is:

1. SimpleMLP: edge attributes only.
2. EdgeGRU: temporal memory without graph convolution.
3. StaticGNN: spatial topology without temporal memory.
4. ST-GNN: topology, induced identity, and temporal memory.
5. E-GraphSAGE: external spatial baseline without temporal memory.

All use matched capacity where feasible, the same early-stopping protocol,
scaler, and calibration approach. Record trainable parameter counts and
inference latency consistently.

### 6.2 Technical screening: one seed

Use one fixed seed (for example, 42) to identify failures, confirm audits, and
discard non-viable configurations. These results are not scientific comparisons.

Order:

1. MLP on `nfv3_extended`.
2. All five base models on `nfv3_extended`.
3. ST-GNN plus structural controls on `nfv3_extended`.
4. All five base models on `portable_core`.

### 6.3 Confirmatory experiments: five seeds

Once code and configuration are frozen, run `[42, 123, 777, 2024, 99]`.

**A. Feature-profile comparison — all models, five seeds per profile**

| Profile | Models | Objective |
|---|---|---|
| `nfv3_extended` | All five | Performance with enriched NF-v3 telemetry |
| `portable_core` | All five | Robustness using common/deployable telemetry |

This prevents feature-profile conclusions that apply only to ST-GNN.

**B. Architecture ablations — ST-GNN, five seeds per variant**

Use the primary profile selected before examining test results (by default,
`nfv3_extended`). Keep every configuration fixed except the named factor.

| Variant | Removed or changed factor | Question |
|---|---|---|
| Full ST-GNN | fixed identity policy, GAT, and delta-time memory | Reference |
| No memory | replace GRU/memory with current-window `z` | Does temporal history help? |
| No topology | omit GAT; feed local edge aggregates to identity/GRU | Does GAT add value beyond local temporal state? |
| Lagged identity | use identity from `t-1` | Does target-edge participation in identity matter? |
| No direct edge attributes in classifier | classifier sees endpoint embeddings only | How much does the direct `edge_attr` path contribute? |
| Gap reset vs delta-time decay | compare temporal policies | Does elapsed time modelling matter? |

The final two are important extra ablations: the direct feature shortcut and
gap-handling policy. With limited compute, screen every variant with one seed,
but repeat at least full, no-memory, no-topology, and lagged-identity variants
with five seeds.

Do not treat MLP or EdgeGRU as substitutes for controlled ST-GNN ablations:
they are essential baselines, but they change multiple factors simultaneously.

### 6.4 Do not vary everything at once

Do not simultaneously change window length, feature profile, labels,
`pos_weight`, architecture, and split. If window length is explored later, use
the frozen pipeline, select on validation with ST-GNN first, and then repeat the
necessary final matrix.

## 7. Execution order

1. Run label correction and produce its manifest.
2. Run graph construction on a small sample; pass all temporal-contract tests.
3. Generate complete graphs, scaler, edge provenance, and graph manifest.
4. Run an MLP smoke test for dimensions, timing, and splits.
5. Calibrate `pos_weight` and bias on train/validation only.
6. Run one-seed screening for models and ablations.
7. Freeze JSON/YAML configuration, hashes, and the experiment list.
8. Run five-seed confirmatory experiments.
9. Run Test1/Test2 once per final checkpoint; save probabilities and metrics
   without retuning any hyperparameter.
10. Generate comparison tables, PR curves, latency measurements, and
    between-seed variance summaries.

Each run ID must include the dataset-manifest hash, feature profile, model,
variant, seed, and code version.

## 8. Post-hoc alerts preceding the documented Nmap scan

This analysis happens **after** labels, graphs, model, hyperparameters, and
thresholds are frozen. It can answer whether a correct model alert preceded the
documented late Discovery/Nmap stage; it is not called lateral-movement lead
time.

The reportable question is:

> How far in advance did the model issue a correct alert associated with
> documented prior activity before the documented internal Nmap scan began?

### 8.1 Allowed external anchors

The historical `mitre_comparison_corrected.ipynb` identifies three stages from
documented endpoints and intervals:

1. Dropbox-related activity;
2. victim-attacker communication;
3. documented internal Nmap scan, conservatively interpreted as candidate
   `Discovery`/T1046.

Stage 3 is the late event. Its `onset_time` remains the documented Nmap start
(`2018-03-01 14:09:48` for the current scenario), not the model's first
prediction. Attribution must rely only on external evidence: documented IPs or
endpoints, documented intervals, and provenance metadata. It must never use
port category, protocol, model score, or an internal-to-internal rule.

### 8.2 Required corrections to the corrected notebook

Before using new checkpoints:

- migrate its logic to a new script or notebook consuming rebuilt graph
  provenance; do not reuse caches or windows from the historical builder;
- determine `Campaign_Stage` from raw `flow_start`, endpoints, and documented
  intervals before inspecting predictions;
- use each flow's `decision_time` as the time at which an alert can exist, never
  `Window_Start` or `FLOW_START_TIME`;
- keep Nmap onset as the external stage-start time; measure Nmap detection delay
  from onset to the first corresponding positive `decision_time`;
- constrain victim-attacker communication to a documented interval, or document
  evidence that all traffic for the pair belongs to the campaign; an unrestricted
  endpoint pair can capture unrelated traffic;
- persist the exact IP-to-global-node-ID map from graph construction. Ideally,
  per-edge provenance removes the need to reconstruct IDs for endpoint
  attribution;
- exclude all historical rules that turn SMB/Admin or internal-to-internal
  traffic into lateral movement;
- exclude Nmap/Discovery flows from their own precursor search and exclude a
  previous late-stage event as a precursor for another episode.

### 8.3 Precursor definition and metrics

A valid precursor alert must simultaneously:

- be a model positive using a validation-selected threshold;
- have a corrected positive ground-truth label;
- belong to a documented prior stage (Dropbox or victim-attacker);
- have `decision_time` earlier than documented Nmap onset;
- share the scenario's defined victim or endpoint.

Report per model, seed, and threshold variant:

- earliest valid precursor and its `decision_time`;
- maximum lead and lead for the nearest valid alert before onset;
- precursor coverage, i.e. scenarios/episodes with at least one precursor;
- delay to the first detection of documented Nmap;
- positive alerts, true positives, and false positives in the campaign interval;
- attribution basis for every reported flow.

Interpret this as a **retrospective intervention opportunity**, not causal proof
that blocking an alert would have prevented the scan.

Run it once on Test2 for each final checkpoint. It must not modify features,
architecture, weights, thresholds, or model selection. Any change motivated by
its result begins a new experimental cycle.

## 9. Artefacts to preserve

For every final run, retain:

- best-epoch checkpoint and full configuration;
- seed, `pos_weight`, initial bias, threshold strategy, and threshold value;
- code, corrected-data, graph-manifest, and scaler hashes;
- feature schema and exact feature order;
- train/validation history;
- per-flow Test1/Test2 probabilities and predictions;
- provenance: `edge -> source row -> flow_start/flow_end/decision_time`;
- metrics, parameter count, latency, and memory policy.

This allows later explainability work without retraining: GATv2 spatial attention
weights can be extracted at inference from saved checkpoints. Adding intrinsic
temporal attention to the GRU, as in the reference paper, changes the model
architecture and requires new training. Attention interpretations must be
validated with perturbation tests; attention alone is not evidence of faithful
explanation.

## 10. Deliverables and completion criteria

`feat/fair-retrain-clean` is ready for review only when it includes:

- reproducible scripts and tests rather than manual notebook-only steps;
- label-correction, feature, scaler, and graph manifests;
- no heuristic MITRE/LM code; a campaign analysis, if included, may use only the
  documented anchors in section 8;
- temporal, loss, threshold, and memory corrections;
- complete results for the confirmatory matrix;
- a report clearly separating validation, Test1, and Test2;
- explicit limitations: single campaign, available labels, extractor choice, and
  no unsupported phase/lead-time claims.

Do not delete historical branches or artefacts until this protocol has been
reproduced and reviewed.
