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

Status as of 2026-09-20:

- benign-background provenance has been audited from the authors' notebooks;
- the development, Test1, and Test2 attack-chain assignments are fixed;
- the primary prediction unit is a packet;
- the first implementation will use fixed, non-overlapping graph windows;
- the identity-free canonical packet extraction completed its SMOKE and
  five-scenario FULL_DEV runs;
- the primary graph window is fixed at five seconds before model results;
- the SMOKE and five-scenario FULL_DEV audits completed without automatic
  data-integrity blockers;
- fold-aware feature profiling and preprocessing review completed;
- XGB-P, full XGB-P+T, and both context ablations completed development OOF runs;
- the thresholded development OOF comparison completed; its step-timeliness
  interpretation requires an audit before graph-model training.

Implementation update (2026-09-19): the primary depth-5 XGB-P development run
completed both folds. Its hierarchical macro OOF packet ROC-AUC is
0.9698542978982521. The sampled shuffled-label control returned
0.4998326345817774, below its 0.6 investigation line. The per-fold feature
importance, single-feature ranking, and MQTT-stratified OOF results identify a
strong protocol-prevalence shortcut: `is_mqtt` is the top gain-ranked feature
in both folds, while MQTT-only ROC-AUC is 0.685 for `dollar_char` and 0.718
for `qos_mid`. The strong aggregate result is therefore not uniform across
traffic groups or attack steps. The reviewed packet view excludes endpoint
identities; preprocessing was fitted on fold-training scenarios only. No
evidence from these controls requires stopping the next development
comparison. The XGB-P sanity gate is recorded as passed for XGB-P+T training,
with the protocol shortcut retained as an interpretation limit. This decision
does not approve test access or a final detection claim. The prepared data,
fold preprocessing audit, model, and OOF outputs are on Drive, not in this
repository. The notebook display alone does not verify their current
availability or checksums; the runners validate those artifacts when used.

The XGB-P+T implementation is now in
[capture_xgb_p_t.py](../code/python/utils/capture_xgb_p_t.py) and
[capture_xgb_p_t.ipynb](../code/python/notebook/capture_xgb_p_t.ipynb).
The training runner enforces the documented sanity decision. The first
real-data context run should be
reviewed for row counts, nonempty window counts, source-artifact checksums,
and plausible feature ranges before the two training folds run.

Development OOF update (user-reported Colab output, 2026-09-19): full XGB-P+T
reached hierarchical macro packet ROC-AUC 0.9975545445964986, compared with
0.9698542978982521 for XGB-P. All five scenario differences were positive;
the largest were `dollar_char` (+0.076349) and `qos_mid` (+0.036737). The
run artifacts and run ID are not stored in this repository, so the follow-up
notebooks verify their Drive checksums and fold provenance before using them.
The near-ceiling aggregate ranking motivates the predeclared context ablation
and thresholded operational evaluation below. It does not establish that
preceding history or learned graph structure contributed the improvement.

Development follow-up (user-reported Colab output, 2026-09-20): the
`current_window` and `history` ablations reached hierarchical macro OOF packet
ROC-AUC 0.998623 and 0.998556, respectively, versus 0.997555 for the
predeclared `full` variant. The ablation run ID is
`20260920T000745_916321Z_xgb_p_t_ablation`. The operational evaluation used
the declared one-false-alert-window-per-hour budget and model-specific
thresholds selected from development OOF predictions:

| Model | Macro false-alert windows/hour | Macro score-positive attack-step iterations | Macro packet recall | Mean duration-assigned latency, seconds |
|---|---:|---:|---:|---:|
| XGB-P | 0.209 | 0.529 | 0.600 | 63.81 |
| Current window | 0.658 | 0.910 | 0.827 | 2.76 |
| History | 0.919 | 0.985 | 0.816 | 3.08 |
| Full XGB-P+T | 0.433 | 0.845 | 0.771 | 5.91 |

The worst fold mean false-alert rate was below one per hour for all four
models; this constraint does not require every scenario rate to be below one.
At the tighter one-per-12-hours budget, `history` had qualifying scores in
0.861 of iterations and `full` in 0.827. The `history` model had qualifying
scores in 306/335 iterations in `train_sub_exf`, versus 206/335 for `full`,
and all iterations in
`train_dollar_char`, `train_empty_conn`, and `train_qos_mid`. Its weakest
listed attack step was `mqtt_cat` in `train_sub_exf` (8/27). The `full` model
had qualifying scores in only 6/117 `scp_exf` iterations in that scenario.
These are development diagnostics, not final-test performance claims.

The `history` variant is the strongest observed score-positive iteration
comparator at the primary one-per-hour budget. Its 0.919 false-alert
windows/hour exceeds the
0.433 rate of `full`, but both satisfy that budget. At the tighter
one-per-12-hours budget, both reported 0.036 false-alert windows/hour and
`history` still had higher iteration detection (0.861 versus 0.827). This
does not establish dominance at every possible false-alert rate. `Full`
remains the originally predeclared XGB-P+T reference, not an automatic choice
for final deployment. The operational results make `history` the development
candidate to beat; any change to the final-model selection rule must be
recorded before Test1 or Test2 is accessed. Graph-model development should
compare against both after the timeliness audit. The results motivate
matched-input neural/graph baselines; they do not by themselves establish a
benefit from learned topology or
recurrent memory. Thresholds and the metrics above use the same development
OOF predictions, so the operational estimates can be optimistic. Preserve
Test1 and Test2 for evaluation after the graph protocol and model choices are
frozen. The reported latency summary assigns each missed iteration its
last-minus-first malicious-packet duration; interpret it alongside detection
rate and the explicit miss flags, not as detection time for missed attacks.
The current iteration rule counts any qualifying malicious-packet score,
including one whose alert becomes available at the five-second window close
after the iteration's last malicious packet. Thus the displayed iteration
rates do not establish detection before an attack step completes. Audit the
stored per-iteration alert and final-packet timestamps before interpreting
these results as early warning, especially for subsecond steps. A chain-level
first-warning deadline relative to a declared terminal step is also needed
for a claim about early detection of an attack chain.

The development-only audit is implemented in
[capture_oof_early_warning_audit.ipynb](../code/python/notebook/capture_oof_early_warning_audit.ipynb)
and [capture_early_warning.py](../code/python/utils/capture_early_warning.py).
It consumes the completed operational JSON report and keeps its model-specific
OOF thresholds. The separate
[early-warning policy](../configs/capture_early_warning_audit_v1.yaml)
declares the observable terminal action steps for the five development
scenarios. An iteration is timely only when its first correct window-close
alert strictly precedes its last malicious packet. A scenario's first correct
chain alert is early only when it strictly precedes the first malicious packet
of any declared terminal action. The latter is one descriptive observation
per scenario, not an independent sample of attack campaigns. This audit
does not reopen packet predictions, retrain, or access final-test scenarios.

Implementation update (2026-09-17): the initial machine-readable manifest is
available at [capture_experiment_v1.yaml](../configs/capture_experiment_v1.yaml).
It records the locked assignments and explicit unresolved decisions (`null`).
Gate-0 audited 1-, 5-, 10-, and 30-second windows. Five seconds is now the
predeclared primary duration; the remaining widths are development-only
ablations. Seed 42 remains the initial reproducibility seed. Development packet
CSV IDs, published filenames, and
byte sizes have now been verified from the authors' public folder listing at
`dataset_creation/raw_traffic/normal_attack/train` and recorded in the manifest.
The existing benign-source audit's IDs identify merge notebooks, not packet CSVs.
Readiness requirements are declarative until runners implement
their checks. The data audit notebook is named `capture_data_gate0.ipynb`.

Gate-0 implementation: [capture_data_gate0.ipynb](../code/python/notebook/capture_data_gate0.ipynb)
uses the CPU-only [capture_data.py](../code/python/utils/capture_data.py) module.
It supports sequential local staging, mounted-Drive persistence with checksum
verification, explicit source/schema bindings, and a reviewed-SMOKE requirement
before FULL_DEV. The notebook downloads individual development CSVs directly
from the authors' Drive using manifest IDs. It previews the first scenario and
reuses that verified local download during the audit, then processes subsequent
scenarios sequentially. No raw CSV copy in the user's Drive is required. The
resolved configuration is saved with each run. Metadata verification alone
does not establish content integrity. The SMOKE and FULL_DEV executions
subsequently audited every source row and persisted checksum-verified artifacts
on Drive. The audit Parquet files preserve raw fields without freezing model
features or selecting a window width. The general model and final-test
readiness requirements remain declarative.

The post-audit decision module
[capture_gate0_review.py](../code/python/utils/capture_gate0_review.py) reuses a
completed FULL_DEV artifact collection to compare repeated benign backgrounds,
audit MAC and network-layer topology cases, and generate a reviewable feature
inventory. Its actions are diagnostic proposals; the module does not freeze
the endpoint, feature, sampling, or window policies. The runtime audit completed
without automatic blockers and confirmed exact normal-packet equality inside
both declared benign-source groups.

Canonical preparation (2026-09-17):
[capture_packet_schema_v1.yaml](../configs/capture_packet_schema_v1.yaml) records
the proposed identity-free packet features, Ethernet-address topology, and
special group-address handling. The canonical packet table retains timestamps
but is independent of graph windows and training weights.
[capture_prepare.ipynb](../code/python/notebook/capture_prepare.ipynb)
uses [capture_prepare.py](../code/python/utils/capture_prepare.py) to validate the
proposal first on a real-data SMOKE run and then, after explicit review, on all
five development scenarios. Both runs completed and the resulting counts,
timestamps, node roles, and structural null patterns were reviewed without
blockers. The canonical extraction is still kept separate from the model-input
preprocessing contract; no classifier is trained by this preparation step.

Feature-profile candidate (2026-09-18):
[capture_preprocessing_v1.yaml](../configs/capture_preprocessing_v1.yaml)
declares semantic feature roles, deterministic bidirectional TCP-port roles,
and layered protocol indicators. The CPU-only
[capture_feature_profile.ipynb](../code/python/notebook/capture_feature_profile.ipynb)
uses [capture_feature_profile.py](../code/python/utils/capture_feature_profile.py)
to validate the completed prepared artifacts and report numeric ranges,
categorical values, fixed port-role coverage, fold-training constants, and
validation-only categorical codes. It writes only a compact report; it does
not materialize transformed packets or fit a model.

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

The primary selection metric is hierarchical macro out-of-fold packet ROC-AUC.
Compute an unweighted packet ROC-AUC separately for each validation scenario,
average scenarios inside each fold, and then average the two fold means. This
keeps both benign blocks equally represented and gives every scenario equal
weight inside its fold. Packet PR-AUC remains a diagnostic metric rather than a
selection criterion because its random baseline varies with each scenario's
attack prevalence.

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
edge in exactly one graph and receives one prediction. The primary width is
the predeclared five-second duration; model results do not select it.

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

The primary development window is five seconds, selected before model results
as an engineering compromise among decision latency, occupied-window continuity,
context, and graph size. The canonical packet table does not store window IDs;
the graph builder deterministically assigns them from the scenario's first
packet timestamp and the declared width. One-, ten-, and thirty-second windows
are reserved for a later development-only width ablation on the selected
contextual model, not a complete repetition of the model ladder.

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
memorize attacker, victim, or scenario identities. The manifest now selects
normalized Ethernet MAC addresses as stable endpoint keys and specifies
group-address and non-IP handling. Verify this contract in the cAPTure graph
builder before model training.

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

The canonical packet table remains nullable and model-independent. Before the
packet-only baseline, a separate FULL_DEV feature profile must validate
semantic types, numeric ranges, fold-training constants, structural missingness,
and categorical coverage. This profile is descriptive: validation categories
must never be added to a fitted training vocabulary.

The primary port representation reuses the deterministic design used by the
NF-v3/CIC2018 pipeline, adapted to packet direction and MQTT. Source and
destination TCP ports are encoded separately with the same fixed exhaustive
roles: MQTT messaging, web/proxy, remote administration, Windows SMB/RPC,
infrastructure, database, residual privileged, registered and dynamic ranges,
port zero, and not-applicable-to-TCP. Raw port numbers are not primary model
features. Because this taxonomy is fixed before model fitting, a valid port
cannot become an unseen validation category.

Protocol indicators remain multi-hot by layer. ARP/IPv4/IPv6, TCP/UDP, and
MQTT/SSH describe different layers and may legitimately coexist; they must not
be collapsed into the single IANA-protocol one-hot representation used by the
older flow pipeline. The exact candidate contract is
`configs/capture_preprocessing_v1.yaml`.

The reviewed primary model view has 103 fixed output columns: 23 direct binary
indicators, two presence indicators, eight transformed numeric magnitudes, 22
source/destination port-role indicators, and 48 deterministic encodings of
protocol fields. `ethernet_type` is excluded because it is exactly redundant
with the ARP/IPv4/IPv6 indicators. Raw TCP ports are replaced by port roles.
`mqtt_version` and `mqtt_connack_reason_code` remain in the canonical packet
artifacts for auditing and secondary ablations but are excluded from the
primary view because they are respectively near-constant and extremely sparse
with suspicious invalid values.

The output order remains fixed across models and folds. Each fold fits numeric
means and population standard deviations only on its training scenarios. It
also identifies output columns that are constant after candidate encoding and
applies the resulting zero mask to both training and validation. A value that
appears only in validation therefore cannot activate an untrained neural input
weight. The fixed protocol-domain encoders themselves do not learn a vocabulary
from either partition.

The FULL_DEV preprocessing audit in
`capture_gate0/preprocessing_runs/20260919T004143_161396Z_preprocessing`
completed both folds. The committed notebook at `b9962d2` records 103 output
columns and full row conservation in all five scenarios per fold, with no
non-finite transformed values. Fold A retains 89 active columns and masks 14;
Fold B retains 92 and masks 11. The maximum absolute transformed values are
4.35 and 4.77 respectively. This approves the packet preprocessing contract,
not the XGB-P training protocol or the final study pipeline. The artifact
contract hash excludes only the administrative transition from audited
candidate to frozen status, so the two saved fold preprocessors remain valid.

Positive packet-length and TCP-window magnitudes use `log1p` followed by
fold-training standardization. TTL, SSH padding length, and TCP header length
use fold-training standardization directly. IPv4 fragment offset uses `log1p`
without standardization: the development profile is overwhelmingly zero, so
division by its small fold standard deviation would turn the observed offset
of two into a value above 45 and create an avoidable neural-input outlier.

Validation may choose among predefined feature procedures or values such as
`top_k`. It must not participate in fitting the selector being evaluated.
After the procedure is selected, it may be refit on all five development
chains before final evaluation.

Training weights are not canonical packet fields. For each fold, compute them
using only its training scenarios so that every `(scenario, binary class)` cell
has the same total weight, then normalize the training weights to mean one.
XGBoost receives them through `sample_weight`; neural models apply them to the
per-edge loss before reduction. Validation and test metrics are unweighted
inside each scenario and use the hierarchical macro aggregation described
above. This fold-local policy prevents packet-rich attack chains and repeated
benign backgrounds from dominating without changing graph structure or labels.

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

Missed sequences remain explicit misses. The development OOF operational
protocol is now fixed before running the follow-up notebooks:

- a nonempty window alerts when its maximum packet score is at least the
  model-specific threshold;
- a false-alert opportunity is a five-second wall-clock window containing no
  malicious packet, including empty windows from the scenario origin through
  the final packet window; empty windows cannot alert;
- the denominator is five seconds per such window, expressed in hours;
- the primary budget is one false-alert window per hour; sensitivity budgets
  are one per 12 hours and one per five minutes;
- for each model and budget, choose the smallest score threshold whose worst
  of the two fold means of per-scenario false-alert-window rates meets the
  budget; score ties are handled by moving just above the relevant negative
  window score;
- an attack-step iteration is detected only when a malicious packet score
  crosses the threshold; its detection time is that packet's window end;
- missed iterations retain a miss flag and use their last-minus-first
  malicious packet time in the mean duration-assigned latency summary.

This uses the same human-readable false-alert frequencies as selected points
in [the cAPTure paper](https://doi.org/10.1016/j.comnet.2026.112570), but
counts alerting windows rather than the paper's false-positive packets. Its
numerical FPR results are therefore not directly comparable. Thresholds are
selected only from development OOF predictions. Test1 and Test2 do not
determine thresholds. Scenario and fold breakdowns accompany hierarchical
macro summaries, so repeated benign backgrounds do not let packet-rich
scenarios determine the comparison.

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

First run the fold-aware feature profile against the completed canonical
FULL_DEV artifacts and review the fixed port-role coverage, remaining
categorical code domains, numeric ranges, and fold-training constants. Freeze
the semantic preprocessing proposal after this review, then run the separate
fold-local preprocessing audit. That audit must verify the 103-column order,
training-only numeric parameters and masks, finite transformed values, and row
conservation without materializing another full packet dataset. Freeze the
preprocessing contract only after both fold artifacts and their validation
transform reports have been reviewed.

Run the predeclared XGB-P depth-5 primary configuration on both development
folds using `capture_xgb_p.ipynb`. The optional depth-10 configuration is a
matched capacity sensitivity from the authors' published search grid, not an
automatic replacement for the primary result. Both configurations use 200
boosting rounds, learning rate 0.1, fold-local scenario/class weights, and
the same frozen 103-column packet representation. The fixed number of rounds
avoids selecting a checkpoint on the outer validation fold. Run one fold at a
time on Colab CPU; training matrices are temporary local artifacts, while
models, reports, and one out-of-fold score per validation packet are verified
and retained on Drive. No threshold or final model is fitted in this phase.

The depth-10 sensitivity is conditional on a resource-only CPU and memory
check. Any change to the declared primary configuration or both-model
comparison protocol requires a manifest revision before inspecting the
corresponding validation metrics. Do not treat the best seed or fold-trained
model as the final test model.

Before accepting results:

- run shuffled-label and leakage sanity checks;
- inspect single-feature and feature-importance behavior;
- verify that fitted preprocessing used only the fold's train scenarios;
- report both folds and every chain separately;
- investigate any near-perfect result before proceeding.

The XGB-P sanity notebook uses a deterministic one-in-100 source-row sample
from each development scenario for a negative control. It permutes sampled
training labels separately inside each training scenario, preserves class
counts, fits the same depth-5/200-round model using the already audited
fold-local preprocessor, and evaluates sampled held-out rows against their
original labels. This is a deliberately inexpensive pipeline check, not a
replacement for full-data validation. A hierarchical macro ROC-AUC above 0.6
requires investigation; a value near 0.5 is reassuring but cannot rule out
all leakage. The same notebook reports univariate ROC-AUC diagnostics for a
fixed set of current-packet features highlighted by the primary model. Their
best-direction scores are descriptive, not eligible for model selection.
They measure a single feature's scalar ranking only; a nonlinear one-feature
tree could behave differently.

### Phase 4: train XGB-P+T

Implement only causal or window-close-available summaries. Use the same folds,
base packet features, packet targets, and decision times as XGB-P. Compare
against XGB-P using out-of-fold predictions and the same alert budgets.

The initial context contract adds eight complete-current-window summaries and
six summaries over the union of the six preceding five-second wall-clock
windows (30 seconds): 14 context fields in addition to the 103 packet-model
columns before any fold-local context preprocessing. The historical summaries
exclude the current window;
distinct peers are recomputed over the entire horizon rather than summed
across windows. Missing history is zero. Raw endpoint identities are grouping
keys only, never model values. Summary computation resets at scenario
boundaries and uses no labels, attack annotations, or evaluation metadata.
Numeric preprocessing is fitted on each training fold only. The same derived
context fields must be available to graph models for matched-input
comparisons. A static GNN with these fields has explicit temporal input but
no learned recurrent memory; a temporal GNN with them has both. An unassisted
temporal GNN remains a separate representation-learning comparison.

For the first XGB-P+T run, all 14 nonnegative context fields receive a fixed
`log1p` transform followed by mean/standard-deviation normalization fitted
only on the current fold's training scenarios. Zero-variance fields use unit
scale. Context artifacts contain only source-row IDs and the 14 derived
fields. They are generated separately for each scenario from timestamp,
Ethernet endpoint keys, MQTT indicator, and frame length; labels and attack
metadata are never read by the context generator. A complete current window
is scored at its close. The preceding 30-second history consists of six
wall-clock windows, including empty windows as zero contribution. Distinct
peers are counted over the union of that history. Context state resets for
each scenario. Training joins context and packets by verified source-row ID.

The XGB-P+T primary model uses the XGB-P depth-5, 200-round configuration,
scenario/class weights, and folds. Its 117 ordered inputs are the existing
103 packet columns followed by the 14 context fields. Context preprocessing
is saved per fold. The same hierarchical macro OOF ROC-AUC is reported before
any threshold or final-test evaluation. A real-data run requires Drive access,
the completed prepared/preprocessing artifacts, and an explicit sanity-gate
decision in the manifest. No new context horizon or model-selection rule is
needed for the primary run.

Follow-up ablation (frozen before ablation results): reuse XGB-P and the full
XGB-P+T model, then train only two additional depth-5/200-round variants on
the same two folds and prepared context artifacts. `current_window` receives
the 103 packet columns plus the eight complete-current-window summaries;
`history` receives the 103 packet columns plus the six preceding-30-second
summaries. Each variant fits only its selected context columns on its current
training fold. The implementation is in
[capture_xgb_p_t_ablation.ipynb](../code/python/notebook/capture_xgb_p_t_ablation.ipynb).
The separate
[operational OOF notebook](../code/python/notebook/capture_oof_operational_evaluation.ipynb)
uses [capture_oof_operational.py](../code/python/utils/capture_oof_operational.py)
to compare all four models using the frozen alert protocol above. Neither
notebook accesses held-out author-train or final-test scenarios.

### Phase 5: build graphs and run graph models

Before constructing graph artifacts, run the matched packet MLP screen in
[capture_mlp_training.ipynb](../code/python/notebook/capture_mlp_training.ipynb).
The `MLP-P` variant receives the same 103 fold-preprocessed packet features as
XGB-P. The `MLP-History` variant adds only the six causal preceding-30-second
features. Both variants use the same two outer folds, scenario/class weights,
one declared seed, fixed epoch count, OOF threshold selection, and operational
budgets. This screen distinguishes the value of the engineered causal history
from the value of tree learning before graph structure is introduced.

Confirm the graph schema after preparation and use the predeclared five-second
window. Build exactly one graph per fixed window and verify a one-to-one
correspondence between canonical packet records, graph edges, labels, and model
outputs.

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
- predeclared one-, ten-, and thirty-second development ablation widths;
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

1. stable node identifier (resolved in the manifest as normalized Ethernet MAC);
2. exact base packet feature schema;
3. categorical encoding and missing-value rules;
4. candidate and selected fixed-window duration;
5. class weighting or training-only benign subsampling;
6. target false-alert budgets and threshold-selection procedure (resolved for
   the development OOF comparison above);
7. graph handling for broadcast, multicast, missing endpoints, and non-IP
   packets (resolved in the manifest; implementation verification pending);
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

## Implementation sequence and next gate

The experiment proceeds in this order:

1. completed Gate-0 and canonical per-scenario preparation artifacts;
2. a fold-aware FULL_DEV feature profile and reviewed preprocessing contract;
3. an XGB-P training and out-of-fold evaluation notebook or script (implemented;
   real-data primary folds, sampled sanity controls, and review completed);
4. XGB-P+T feature generation and training (implemented; the user reported
   completed real-data OOF results, with Drive artifacts verified by follow-up
   consumers);
5. current-window and history ablations followed by the operational OOF
   evaluation (user-reported Colab execution completed; development results
   recorded above);
6. window-close, packet-arrival, non-Nmap, and precursor-deadline early-warning
   audits (implemented and executed in Colab on development OOF predictions);
7. matched packet and history MLP screening (implemented for Colab execution),
   followed by packet-graph construction and the matched static and temporal
   graph baselines. Compare them with XGB-P, the predeclared full XGB-P+T model,
   and the stronger observed history ablation before adapting the ST-GNN.

## Related project documents

- [cAPTure benign-source audit notebook](../code/python/notebook/capture_benign_source_audit_colab.ipynb)
- [Temporal state and ablation contract](temporal-state-and-ablation-contract.md)
- [Training graph contract](training-graph-contract.md)
- [Training core contract](training-core-contract.md)
- [cAPTure paper](https://doi.org/10.1016/j.comnet.2026.112570)
