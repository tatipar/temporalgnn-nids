# cAPTure experimental decision record and execution plan

## Purpose and status

This document is the source of truth for the cAPTure feasibility study. It
records the decisions already made, their rationale, the planned experiments,
the leakage controls, and the remaining open questions. Future work should
update this document whenever a decision changes rather than relying on chat
history.

The immediate objective is not to demonstrate that a temporal GNN is better.
It is to test, without data leakage, whether temporal and/or structural context
provides useful and earlier attack-chain detection than current-packet
features alone.

Status as of 2026-09-16:

- benign-background provenance has been audited from the authors' notebooks;
- the development, Test1, and Test2 attack-chain assignments are fixed;
- the primary prediction unit is a packet;
- the first implementation will use fixed, non-overlapping graph windows;
- the exact packet schema and window duration remain open pending Gate 0;
- no cAPTure classifier has been trained yet.

Implementation update (2026-09-17): the initial machine-readable manifest is
available at [capture_experiment_v1.yaml](../configs/capture_experiment_v1.yaml).
It records the locked assignments and explicit unresolved decisions (`null`).
Initial Gate-0 window candidates are 1, 5, 10, and 30 seconds, with seed 42 as
the initial reproducibility seed. Neither the selected duration nor its
selection rule is frozen. Packet CSV IDs and published filenames still require
source-metadata verification; the existing audit's IDs identify merge notebooks,
not packet CSVs. Readiness requirements are declarative until runners implement
their checks. The data audit notebook is named `capture_data_gate0.ipynb`.

Gate-0 implementation: [capture_data_gate0.ipynb](../code/python/notebook/capture_data_gate0.ipynb)
uses the CPU-only [capture_data.py](../code/python/utils/capture_data.py) module.
It supports sequential local staging, mounted-Drive persistence with checksum
verification, explicit source/schema bindings, and a reviewed-SMOKE requirement
before FULL_DEV. Packet CSV bindings are supplied in the notebook until verified
metadata is available in the manifest; the resolved configuration is saved with
each run. It preserves raw fields in audit Parquet, without freezing model
features or selecting a window width. Runtime validation is pending execution
of the notebook's synthetic checks and smoke audit in Colab. The general model
and final-test readiness requirements remain declarative.

## Why cAPTure is being evaluated

The previous NF-CSE-CIC-IDS2018-v3 Infiltration study did not provide a sound
test of the research hypothesis. The scenario did not contain the assumed
lateral movement, the original IP-to-ID mapping contained an error, and label
corrections made binary detection nearly trivial for simple classifiers.

cAPTure is a better candidate because it contains multi-stage attack paths,
packet-level timestamps, attack-step and sequence annotations, and separate
executions for the authors' train and test sets. It is still semi-synthetic:
the authors executed attacks separately and inserted them into benign
captures. This study will use their published, pre-merged scenario files as
immutable episodes. It will not regenerate the mixes or choose new insertion
points.

The main research question is:

> Does temporal and/or graph-structural context improve packet detection,
> attack-step coverage, or detection latency over a strong packet-only
> classifier under unseen benign conditions and unseen attack paths?

## Benign-background provenance audit

The lightweight Colab audit is stored in
[capture_benign_source_audit_colab.ipynb](../code/python/notebook/capture_benign_source_audit_colab.ipynb).
It downloads and inspects the 14 official merge notebooks without downloading
the multi-gigabyte CSV or PCAP files.

The audit found exactly one declared benign source in every notebook:

| Attack chain | Author train benign capture | Author test benign capture |
|---|---|---|
| `dollar_char` | `normal_15_16_17` | `normal_5_6_7` |
| `empty_conn` | `normal_2_3_4` | `normal_5_6_7` |
| `pub_exf` | `normal_2_3_4` | `normal_5_6_7` |
| `qos_mid` | `normal_2_3_4` | `normal_5_6_7` |
| `slash_char` | `normal_15_16_17` | `normal_5_6_7` |
| `sub_exf` | `normal_15_16_17` | `normal_5_6_7` |
| `user_prop` | `normal_15_16_17` | `normal_5_6_7` |

These names are treated as benign capture blocks, not inferred as single
calendar days. The audit establishes provenance declared by the published
code. It does not prove that each distributed CSV is the exact bit-for-bit
output of the audited notebook version.

Important consequences:

- the authors' train split contains two benign capture blocks;
- the authors' test split contains a third, disjoint benign block;
- all seven attack-chain types occur in both author splits;
- attack-chain identity and benign background are confounded inside the
  author train scenarios;
- scenario files that use the same benign source likely repeat much of the
  same benign traffic if they were generated exactly as declared.

## Locked attack-chain split

### Development chains

The following five author-train scenarios form the development pool:

| Benign capture | Development chains |
|---|---|
| `normal_2_3_4` | `empty_conn`, `qos_mid` |
| `normal_15_16_17` | `dollar_char`, `slash_char`, `sub_exf` |

### Held-out chains

The following chains are held out from all training, feature selection,
calibration, and hyperparameter selection:

| Chain | Author train benign capture | Selection rationale |
|---|---|---|
| `pub_exf` | `normal_2_3_4` | network-observable exfiltration path |
| `user_prop` | `normal_15_16_17` | distributed/resource-exhaustion path with strong temporal and structural behavior |

Their author-train CSV files must not be used. The pair was chosen before
training to cover both author-train benign backgrounds and two distinct attack
objectives while leaving related, but not identical, objectives represented in
development.

This is not a claim of completely novel adversarial behavior. cAPTure paths
share reconnaissance and other precursor steps. Test2 evaluates an unseen
complete path and terminal branch, not an attack whose every step is absent
from training.

### Development validation

Validation uses two leave-one-benign-block-out folds:

| Fold | Train | Validate |
|---|---|---|
| A | `empty_conn`, `qos_mid` | `dollar_char`, `slash_char`, `sub_exf` |
| B | `dollar_char`, `slash_char`, `sub_exf` | `empty_conn`, `qos_mid` |

No individual chain is selected as a permanent validation chain. Every
development chain is validated exactly once through out-of-fold predictions.
Model and hyperparameter comparisons must use macro summaries across folds and
chains so that packet-rich chains do not dominate.

This validation is deliberately conservative but imperfect. Each fold changes
both the benign background and the attack paths, so it measures joint
background/path generalization. It cannot isolate those effects. Splitting
scenario files from the same benign source across train and validation would
risk repeating the same benign packets across both sides.

### Final tests

| Set | Scenarios | Purpose |
|---|---|---|
| Test1 | author-test executions of the five development chains | known path types under a new execution, timing, and benign capture |
| Test2 | author-test executions of `pub_exf` and `user_prop` | unseen complete attack paths under the same new benign capture |

Test1 and Test2 both use `normal_5_6_7`. They are two views of one final
evaluation, not statistically independent benign environments. Both must be
run with the same frozen pipeline. Test1 must not be used to modify the model
before Test2 is evaluated.

## Prediction and temporal protocol

### Prediction unit

The primary target is binary packet classification:

```text
normal packet -> 0
packet belonging to any labeled attack step -> 1
```

Every evaluated model must produce exactly one score for every target packet.
Attack-step, sequence, scenario, and timing fields remain available for
evaluation but are not model inputs.

### Primary window protocol

The first experiment uses fixed, non-overlapping, half-open windows within
each scenario:

```text
[window_start, window_end)
```

Windows must never cross scenario or dataset-split boundaries. A packet is an
edge in exactly one graph and receives one prediction. The exact window width
will be selected from development-only diagnostics and validation results.

The current ST-GNN processes a sequence of graph snapshots. Within one
window, it uses the complete graph, updates its recurrent node memory, and
classifies the window's edges. Its prediction is therefore available at the
window close, not at each packet timestamp.

The primary aligned decision time for every model is:

```text
decision_time = window_end
```

This avoids assigning an early packet timestamp to a prediction that used
later packets from the same window. XGB-P may also receive a secondary
"native" latency report in which its score is available at packet arrival,
but the aligned window-close result is the primary representation comparison.

Temporal state must:

- carry across consecutive windows of the same scenario according to the
  declared memory policy;
- reset at every scenario boundary;
- reset between train, validation, and test passes;
- never transfer across folds or independent episodes.

### Deferred continuous-time protocol

A packet-event TGN or TGAT would use a strict
`read memory -> predict packet -> update memory` sequence and would be the
cleanest way to obtain packet-time latency. It is deferred until the simpler
study shows that temporal or structural context is promising.

Do not feed overlapping rolling graphs into the current ST-GNN unchanged.
The model would process repeated context edges multiple times and repeatedly
update memory with the same packets. A future rolling implementation would
need either an event-based TGN or explicit context and target edge masks, with
memory updated only by new target edges.

### Window robustness

Window duration is not yet fixed. Gate 0 must report packets, nodes, and edges
per candidate duration as well as attack-step durations. Candidate widths and
the selection rule must be recorded before final testing.

At least one fixed-window sensitivity check should shift the window origin by
half a window. A large performance change under that shift is evidence that
the result depends on arbitrary boundary alignment.

## Common canonical data representation

All models must derive from the same audited packet records. The canonical
prepared data should retain at least:

```text
scenario
attack_chain
benign_source
author_split
packet_timestamp
window_id
window_start
window_end
src_endpoint
dst_endpoint
binary_label
attack_step
phase
sequence_id
packet_features...
```

The provenance fields are required even when a particular model does not use
them. They allow exact reconstruction of folds, windows, sequence metrics, and
decision timestamps.

Raw scenario CSV files should be processed one at a time into compressed,
scenario-specific Parquet artifacts. Record source IDs, source sizes, content
hashes when feasible, row counts, schema versions, and preprocessing
configuration. Large raw files and derived datasets must not be committed to
Git.

## Feature and leakage contract

### Metadata that is never a model feature

The following fields are for grouping, ordering, or evaluation only:

- binary and multiclass labels;
- `phase_name`, attack-step names, and related annotations;
- `sequence_id`;
- scenario and chain identifiers;
- benign-source and split identifiers;
- window identifiers;
- absolute timestamps.

### Endpoint identity

Endpoint identifiers are required to construct graph topology, but raw IP or
MAC identities are not primary packet or node features. Otherwise a model may
memorize attacker, victim, or scenario identities. The exact stable endpoint
key, such as IP, MAC, or another device identifier, remains a Gate-0 decision.

A separately labeled identity-feature ablation may be run later, but it must
not replace the identity-free primary result.

### Base packet features

The primary XGB-P and graph models should use the same technically valid
current-packet feature family wherever their architectures allow it. Gate 0
must inspect constant, missing, malformed, identifier-like, and suspiciously
label-correlated columns before the feature schema is frozen.

The main initial baseline should avoid learned feature selection if the full
valid schema is computationally manageable. XGBoost can determine useful
splits internally, while this choice avoids an additional source of held-out
chain leakage.

### Feature selection and preprocessing

Every fitted preprocessing step belongs inside each training fold. This
includes:

- imputation parameters;
- categorical vocabularies or encoders;
- scaling, when required;
- variance and correlation filters;
- mutual-information ranking;
- model-based feature ranking;
- the identities of learned top-k features.

Validation may choose among predefined feature procedures or values such as
`top_k`. It must not participate in fitting the selector being evaluated.
After the procedure is selected, it may be refit on all five development
chains before final evaluation.

The authors' published top-100 feature list is not automatically valid for
the primary held-out-chain experiment. It was derived using their full train
set, which includes `pub_exf` and `user_prop`. Using it would allow held-out
chain labels to influence feature selection. It may be reported only as a
clearly labeled reproduction or secondary ablation.

## Model ladder

Models must be added in an order that answers one question at a time.

### 1. XGB-P: packet-only baseline

Input:

- current-packet features only.

Purpose:

- measure how far a strong tabular classifier can go without history or
  topology;
- detect whether binary classification is already trivial;
- expose fingerprint or leakage shortcuts before investing in a GNN.

One global XGBoost model is trained per fold, not one model per window. Rows
may be shuffled after folds and window provenance have been assigned.

Required sanity checks include shuffled labels, per-feature diagnostics,
feature importance, results with endpoint identities excluded, and metrics
separated by fold and chain.

### 2. XGB-P+T: manually summarized temporal/structural baseline

Input:

- the same current-packet features as XGB-P;
- online-computable summaries available by the declared decision time.

Candidate summaries include:

- packets sent and received by each endpoint;
- unique peers and node degree;
- source-destination pair frequency;
- protocol and packet-size distributions;
- interarrival-time summaries;
- changes relative to previous windows.

The selected summaries and horizons must be computed identically in train,
validation, and test. This model answers whether history or summarized
structure adds information. It does not by itself establish that a GNN is
necessary.

### 3. Graph baselines and ST-GNN

The minimum graph comparison should distinguish direct edge information,
topology, and recurrent memory. Candidate models are:

- edge MLP or equivalent packet neural baseline;
- static GNN;
- EdgeGRU temporal baseline;
- the existing ST-GNN.

The central comparison is:

```text
XGB-P
  -> value of temporal/manual structural summaries: XGB-P+T
  -> value of learned graph structure and memory: static/temporal GNNs
```

Do not interpret an ST-GNN gain unless the simpler neural and graph ablations
use the same packet records, features, windows, labels, and decision times.

## Initial graph construction contract

The provisional packet graph is a directed temporal multigraph:

- nodes represent stable devices or endpoints;
- every packet is one directed edge from source to destination;
- repeated packets between the same pair remain distinct edges;
- edge features are the approved current-packet features;
- edge targets are binary packet labels;
- graph timestamps and decision times are window closes;
- stable global node IDs support temporal memory within a scenario;
- temporal memory resets at scenario boundaries.

The exact node identifier, handling of broadcasts/multicasts, missing
endpoints, non-IP traffic, and categorical protocol fields must be settled by
Gate 0 before graph artifacts are built.

## Metrics and alert semantics

Accuracy is not an acceptable primary metric because normal traffic dominates.
At minimum, report:

- packet PR-AUC;
- packet recall and precision at declared thresholds;
- packet FPR;
- false-alert windows per hour;
- sequence detection rate (SDR);
- latency to the first correctly detected packet in each attack-step
  iteration;
- per-chain and per-step results;
- macro summaries across chains and folds.

For the fixed-window primary protocol, an alert becomes available at the
window close. Sequence latency is therefore:

```text
first_detecting_window_end - first_malicious_packet_timestamp
```

Missed sequences must remain explicit misses rather than being removed from
latency summaries. Operational window alerts and their aggregation rule, such
as at least one packet score above threshold, must be frozen before final
testing.

Thresholds and target false-alert budgets must be selected from development
out-of-fold predictions only. Test1 and Test2 must not determine thresholds.
Exact target budgets remain an open protocol choice and should include values
comparable to the cAPTure paper when feasible.

## Execution plan

### Phase 0: record the experiment manifest

Create and version a machine-readable manifest containing:

- all scenario assignments and benign sources;
- the two validation folds;
- held-out-chain policy;
- label rule;
- feature-policy version;
- window policy and candidate durations;
- random seeds;
- metric and threshold-selection rules;
- input file IDs and expected names.

This manifest must exist before model training.

### Phase 1: Gate-0 smoke audit

Create `capture_data_gate0.ipynb` with a smoke mode that processes one
non-held-out scenario from each author-train benign background:

- `empty_conn` from `normal_2_3_4`;
- `dollar_char` from `normal_15_16_17`.

The smoke run is an engineering check, not scientific evidence. It must report:

- schema and data types;
- label values and binary-label mapping;
- timestamp parsing, ordering, and ranges;
- attack steps, phases, and sequence identifiers;
- normal and attack packet counts;
- missing, constant, malformed, and duplicate columns or rows;
- unique endpoints and communication pairs;
- packet-rate and graph-size distributions at candidate window widths;
- attack-step durations;
- estimated raw and derived storage requirements.

### Phase 2: prepare the full development pool

If the smoke audit passes, process the five development scenario CSVs one at
a time. Write canonical Parquet artifacts and audit reports, then delete each
ephemeral raw download if storage is constrained.

The approximate author-train transfer volume for the five development
scenarios is about 10 GB. The largest individual file is approximately 4.75
GB, so sequential processing should keep peak local storage far below the
full dataset size.

Do not process or inspect the held-out author-train scenario contents.

### Phase 3: train and audit XGB-P

Run both development folds with a small, predefined hyperparameter search.
Generate one out-of-fold score per development packet and retain full
provenance.

Before accepting results:

- run shuffled-label and leakage sanity checks;
- inspect single-feature and feature-importance behavior;
- verify that fitted preprocessing used only the fold's train scenarios;
- report both folds and every chain separately;
- investigate any near-perfect result before proceeding.

### Phase 4: train XGB-P+T

Implement only causal or window-close-available summaries. Use the same folds,
base packet features, packet targets, and decision times as XGB-P. Compare
against XGB-P using out-of-fold predictions and the same alert budgets.

### Phase 5: build graphs and run graph models

Freeze the graph schema and selected window duration from development only.
Build exactly one graph per fixed window and verify a one-to-one correspondence
between canonical packet records, graph edges, labels, and model outputs.

Train the graph baselines and ST-GNN under the same folds. Verify memory resets,
strict graph timestamp ordering, scenario isolation, and aligned decision
times.

### Phase 6: freeze the final pipeline

Before accessing final scenario contents, freeze and record:

- preprocessing and feature schema;
- window width and origin;
- graph construction;
- model architectures and hyperparameters;
- class weighting or sampling policy;
- threshold or false-alert selection method;
- checkpoint-selection rule;
- all primary and secondary metrics.

Retrain the selected configurations on all five development chains as defined
by the protocol. Any deviation after this point turns the tests into
development data and must be documented as exploratory.

### Phase 7: execute Test1 and Test2

Process the seven author-test scenario files with the frozen pipeline. Evaluate
Test1 and Test2 without tuning between them. Report:

- each chain independently;
- macro averages across chains;
- packet and sequence metrics;
- aligned latency at window close;
- XGB-P native latency as a secondary result if desired;
- uncertainty or variability across sequences when sample counts permit.

Avoid a packet-micro aggregate as the sole conclusion because high-volume
chains such as `user_prop` can dominate it.

### Phase 8: robustness and deferred experiments

Only after the confirmatory evaluation is preserved, consider:

- shifted fixed-window origins;
- alternative development-selected window widths;
- temporal jitter sensitivity;
- author top-100 feature reproduction;
- raw endpoint-identity ablation;
- a packet-event TGN or TGAT;
- alternative admissible held-out-chain pairs as a sensitivity analysis.

These are secondary analyses and must not be used retroactively to redefine
the primary test.

## Decision gates

The study should stop or change course when a gate fails.

### Gate 0: data integrity

Stop before modeling if packet ordering, labels, scenario boundaries, endpoint
identity, or attack-step reconstruction cannot be verified well enough to
support packet and sequence metrics.

### Gate 1: packet-only non-triviality

If XGB-P is nearly perfect under the out-of-background folds, first search for
metadata leakage, endpoint fingerprints, deterministic protocol artifacts, or
duplicated records. If the result remains legitimate, the dataset may be too
easy for demonstrating the value of a temporal GNN, even if it is useful for
other NIDS studies.

### Gate 2: value of temporal summaries

If XGB-P+T does not improve coverage or latency at comparable false-alert
budgets, the proposed temporal hypothesis receives weaker support. A GNN may
still learn structure that manual summaries miss, but the reason to invest in
it is reduced.

### Gate 3: value of graph learning

An ST-GNN is justified only if it improves relevant metrics over XGB-P+T and
the static/temporal neural ablations under the same information and decision
constraints. A gain limited to one window alignment or one packet-rich chain
is not robust evidence.

Exact minimum effect sizes for these gates are not yet fixed. They must be
declared before final testing.

## Known limitations

- cAPTure is semi-synthetic and uses attack insertion into benign captures.
- Benign background and chain identity are confounded in author train data.
- Only two author-train benign backgrounds are available for development
  validation.
- Test1 and Test2 share the same benign capture block.
- Many attack paths share precursor steps, so Test2 is not entirely novel at
  the step level.
- `sub_exf` has an exploitation action that the paper describes as not
  directly generating detectable network traffic; its surrounding path and
  exfiltration activity remain usable, but interpretation must be careful.
- Fixed windows quantize alert time and may create boundary sensitivity.
- Full packet-level temporal graph learning may be expensive in Colab.
- Row-subsampled reduced datasets are useful for reproduction and engineering
  checks but may not preserve the temporal evidence required for the primary
  experiment.

## Open decisions

The following items must be resolved from development data without consulting
final-test performance:

1. stable node identifier: IP, MAC, or another device key;
2. exact base packet feature schema;
3. categorical encoding and missing-value rules;
4. candidate and selected fixed-window duration;
5. class weighting or training-only benign subsampling;
6. target false-alert budgets and threshold-selection procedure;
7. graph handling for broadcast, multicast, missing endpoints, and non-IP
   packets;
8. minimum effect sizes for proceeding through the model ladder;
9. final ST-GNN memory policy and limited hyperparameter search;
10. whether data-level benign-source signature verification is necessary.

## Reproducibility checklist

Every run must record:

- Git commit and working-tree status;
- experiment-manifest hash;
- source file IDs and hashes when available;
- prepared-data schema and hashes;
- scenario and fold assignments;
- feature-policy version and selected columns;
- window duration, origin, and decision-time rule;
- seeds and software versions;
- fitted preprocessing provenance;
- model configuration and checkpoint-selection rule;
- threshold-selection provenance;
- per-packet predictions with scenario, window, sequence, and timestamp keys;
- per-chain, per-step, and aggregate metrics.

## Immediate next artifacts

The next implementation work should produce, in this order:

1. a machine-readable cAPTure experiment manifest;
2. `capture_data_gate0.ipynb` with `SMOKE` and `FULL_DEV` modes;
3. canonical per-scenario Parquet and audit-report schemas;
4. an XGB-P training and out-of-fold evaluation notebook or script;
5. XGB-P+T feature generation only after XGB-P passes its sanity gate;
6. packet-graph construction and ST-GNN adaptation only after the tabular
   baselines are trustworthy.

## Related project documents

- [cAPTure benign-source audit notebook](../code/python/notebook/capture_benign_source_audit_colab.ipynb)
- [Temporal state and ablation contract](temporal-state-and-ablation-contract.md)
- [Training graph contract](training-graph-contract.md)
- [Training core contract](training-core-contract.md)
- [cAPTure paper](https://doi.org/10.1016/j.comnet.2026.112570)
