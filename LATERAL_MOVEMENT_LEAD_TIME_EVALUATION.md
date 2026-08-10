# Evaluation of Lateral-Movement Lead Time and Dataset Options

## Purpose

This document evaluates the current graph-generation, model-training, and post-analysis pipeline in relation to the following research hypothesis:

> A binary network-flow classifier may detect a malicious precursor involving a host sufficiently early that an administrator could investigate or isolate the host before it participates in verified lateral movement.

The central conclusion is that this is a valid **detection lead-time** question, but it is not the same task as directly forecasting lateral movement. The current NF-CSE-CIC-IDS2018 Day 2 sequence does not contain verified lateral movement, so it cannot be the main dataset used to test the hypothesis.

## 1. Classification and forecasting are related but not equivalent

The current models are binary flow classifiers. For a flow edge `e` observed in graph window `t`, they estimate whether that flow is malicious:

```text
malicious_score(e) = f(flow features, current graph, previous graph state)
```

The temporal models use previous graph windows, but their training target is still the label of the **current flow**. They are not trained using a target such as “this host will perform lateral movement in the next 10 minutes.”

The post-analysis performs a separate retrospective calculation. For a host `h`:

```text
t_alert(h) = earliest operationally available positive detection involving h
             that occurs before verified lateral movement

lead_time(h) = t_lateral_movement(h) - t_alert(h)
```

This measures whether the classifier created an intervention opportunity before an event that is known, after the fact, to be lateral movement. Appropriate names include:

- **Precursor-alert lead time**
- **Operational warning lead time**
- **Retrospective intervention lead time**

A lateral-movement forecasting model would instead use a host-time target such as:

```text
y(h, t, H) = 1 if host h participates in lateral movement during (t, t + H]
```

That task would require hosts that do and do not progress to lateral movement, would penalize early alerts that never progress, and would be evaluated as a prediction of a future outcome. The current classifier does not optimize that objective.

Therefore, the current experiment can support a statement such as:

> The classifier detected a malicious precursor involving the compromised host X minutes before its observed lateral movement.

It cannot, without an additional forecasting experiment, support a statement that a positive flow predicts lateral movement with a particular probability.

It is also preferable to say that an administrator **could have investigated or isolated the host within the available interval**, rather than claiming that lateral movement would definitely have been prevented.

## 2. Audit of the current pipeline

The principal files reviewed were:

- [Day 1 graph construction](code/python/notebook/cic_ids2018_v3_graph_creation.ipynb)
- [Day 2 graph construction](code/python/notebook/cic_ids2018_v3_graph_creation_TEST2.ipynb)
- [Corrected campaign analysis](mitre_comparison_corrected.ipynb)
- [Model implementations](code/python/utils/models.py)
- [Training and evaluation](code/python/utils/training.py)
- [Chronological graph loader](code/python/utils/datasets.py)

### 2.1 Graph construction

The graph representation is appropriate for an IP-level temporal-flow study:

- Each snapshot covers 30 seconds.
- Nodes are source or destination IPs active during that window.
- Directed edges are flows and parallel edges are allowed.
- A global IP-to-ID mapping preserves host identity across windows in the same sequence.
- Each graph locally reindexes the active global nodes.
- Node attributes are constant vectors, so the useful information comes primarily from edge features, topology, and temporal state.
- Each edge has 32 features: port-role encoding, protocol encoding, and 20 numeric flow measurements.
- Numeric features are transformed and standardized using a scaler fitted only on the training period.
- Graph files are loaded numerically and processed in chronological order.
- Empty 30-second intervals are represented when the graphs are generated.

The binary training label is attached to every edge. In the inspected pipeline, a flow is positive when its attack label is `Infilteration`; it is not a lateral-movement or future-progression label.

The corrected notebook appears to fix the post-hoc association between global node IDs and IP addresses. The graph sequence itself uses a consistent mapping; consequently, the mapping error primarily affects host attribution, campaign-stage analysis, and lead-time results rather than necessarily changing the already produced binary logits. The corrected mapping must always be used when reporting host-level results.

### 2.2 Models

| Model | Information used for classification |
|---|---|
| `SimpleMLP` | Current edge features only; no graph or temporal context. |
| `E_GraphSAGE` | Current edge features and spatial aggregation in the current graph. |
| `StaticGNN_Identity` | Current-window incoming/outgoing edge context and GATv2 message passing. |
| `EdgeGRU_Baseline_NoX` | Current edge information plus persistent per-global-IP GRU state from previous windows; no graph convolution. |
| `ST_GNN_Identity` | Current-window graph context plus persistent per-global-IP GRU state. |

All models output one binary logit per flow edge. The temporal models use the persistent global node ID to associate a host with its memory across snapshots.

The temporal computation is causal with respect to graph-window order: the loader does not shuffle snapshots and future snapshots are not supplied to the model. Temporal state is reset at the start of a training epoch or evaluation split.

### 2.3 Training and evaluation

The current training process includes:

- Adam optimization with a learning rate of `0.005`.
- Weighted binary cross-entropy with `pos_weight=2`.
- A negative initial output bias.
- Five training seeds.
- Early stopping based on validation average precision/AUC-PR.
- Validation-based threshold selection.
- A frozen validation threshold for test evaluation.
- Truncated backpropagation through time over ten valid graph snapshots.

Temporal memory values can continue beyond ten windows, but gradient credit is truncated at each TBPTT step. In addition, “ten valid windows” does not necessarily mean exactly five minutes because empty graphs are skipped.

There are three methodological details to address in future experiments:

1. **Elapsed gaps are not modeled.** Empty windows are skipped during training and evaluation. A temporal model can therefore treat a short and a long inactive interval similarly. A time-gap feature, explicit empty-window update, or time-decay mechanism should be added for longer and low-and-slow campaigns.

2. **A chronological split can divide one campaign.** Splitting a single day by wall-clock percentage is better than randomizing flows, but a campaign that crosses a boundary can still contribute related hosts and behavior to both training and validation. New datasets should be split by complete campaign, scenario, or capture day where possible.

3. **The configured precision constraint is not automatically active.** `find_optimal_threshold` defaults to `strategy="max_f1"`. Passing `min_precision=0.90` only affects the result when `strategy="constrained"` is selected. The chosen threshold strategy should be explicitly documented.

## 3. Correcting the alert timestamp

The largest issue for the operational interpretation is the time assigned to an alert.

Flows are placed into a graph according to `FLOW_START_TIME`, but their edge attributes include completed-flow statistics such as duration, bytes, packet counts, and inter-arrival-time measurements. The GNN models also classify a flow using the other edges in the completed 30-second snapshot.

An offline prediction therefore cannot be treated as available at the start of the flow or at the start of the graph window. A more defensible alert-availability time is:

```text
t_available(e) = max(window end, flow end or export time) + processing latency
```

If only start time and duration are available, flow end can be estimated as:

```text
flow_end = flow_start + flow_duration
```

Any precursor flow whose required features were not available until after lateral movement must not be counted as an early alert.

The corrected Day 2 analysis currently identifies:

- First documented campaign precursor: approximately `13:53:03.289`.
- Start of the internal scan: approximately `14:10:03.289`.
- Ground-truth interval between these activities: 17 minutes.
- Earliest qualifying EdgeGRU precursor detection: approximately `13:57:33.289`.
- Nominal window-start lead: 12.5 minutes.

Because a complete 30-second snapshot is consumed, the nominal 12.5-minute value becomes at most approximately 12 minutes after window closure, and possibly less after accounting for flow completion/export and processing.

Both of the following may be reported, but they must be distinguished:

- **Nominal lead:** event onset minus flow/window start.
- **Operational lead:** event onset minus estimated alert-availability time.

The second is the relevant quantity for the administrator-response argument.

## 4. What counts as a qualifying precursor

An earlier alert involving the same IP is not automatically evidence that the model detected an earlier stage of the same attack. Busy servers, shared infrastructure, false positives, NAT, and changing IP ownership can create accidental temporal associations.

The strongest analysis should require:

- The earlier flow is truly malicious.
- It belongs to the same documented campaign.
- It involves the same stable host identity.
- It precedes the verified onset of lateral movement.
- The host's role in the chain is known.

Two complementary lead-time variants should be maintained:

### Campaign-validated precursor lead

The first earlier true-positive flow that is attributed to the same campaign and relevant host. This is the primary scientific result.

### Operational host-alert lead

The first earlier positive alert involving the host, regardless of whether it is a campaign true positive. This approximates what an administrator would see, but it must be reported together with false-alert burden.

Host roles should distinguish at least:

- Compromised pivot/source that later moves to another system.
- Target of the lateral movement.
- Source or destination of a precursor without an established pivot role.

This distinction affects the intervention claim. Isolating a compromised pivot before it moves is different from alerting on the machine that is later targeted.

## 5. The Day 2 limitation

The internal Nmap activity in NF-CSE-CIC-IDS2018 Day 2 is reconnaissance or remote-system discovery. It does not establish that an attacker authenticated to or executed something on another internal machine. Scanning may be a precursor to lateral movement, but it is not itself verified lateral movement.

Consequently, the present sequence can demonstrate a lead before the documented scan. It cannot provide a valid `t_lateral_movement` for the main hypothesis. The Day 2 result should therefore be retained as:

- A pipeline and corrected-IP-mapping validation.
- A case study of precursor detection before internal discovery.
- A preliminary example of the proposed lead-time methodology.

It should not be the primary evidence for lead time before lateral movement.

MITRE ATT&CK defines Lateral Movement as adversaries moving through an environment, whereas remote-system discovery is a Discovery activity that may help prepare movement [1].

## 6. Requirements for a replacement dataset

Literal dotted IP addresses are not mandatory. Stable anonymized host identifiers are sufficient graph nodes, provided that the same machine has the same identifier throughout the trace. For longer captures, stable host IDs may be preferable to IP addresses affected by DHCP or NAT.

A suitable dataset should provide:

### Mandatory telemetry

- Flow start timestamps.
- Flow end, last-seen, or export timestamps whenever possible.
- Stable source and destination IPs or host IDs.
- Source/destination ports, protocol, and direction.
- Sufficient flow statistics to build comparable edge attributes, or raw PCAP from which they can be re-extracted.
- Continuous benign/background traffic as well as attack traffic.

### Mandatory ground truth

- Per-flow malicious/benign labels or reliable time/IP labeling rules.
- Campaign or scenario identifier.
- Attack activity/stage labels.
- Verified lateral-movement onset.
- Lateral-movement source/pivot and target.
- Evidence that the remote access or execution succeeded; scanning alone is insufficient.
- Precursor stages captured before lateral movement.

### Strongly desirable properties

- Multiple complete, independent chains.
- Campaigns that do and do not progress to lateral movement.
- Benign remote administration to challenge the classifier.
- Different movement mechanisms, delays, topologies, and compromised hosts.
- Raw packet or host telemetry that can verify the network labels.

The IP-to-global-ID map used when generating graphs should be stored as an artifact next to each processed sequence. Post-analysis should load that exact map rather than reconstructing it separately.

## 7. Candidate datasets

### 7.1 Unraveled / DAPT 2021 — recommended first option

Unraveled is the closest immediate match to the current pipeline. Its processed network-flow data includes:

- Source and destination IPs.
- Bidirectional first-seen and last-seen timestamps in milliseconds.
- Directional timestamps and flow duration.
- Ports, protocol, packet/byte counts, IATs, flags, and other flow statistics.
- `Activity`, `Stage`, `DefenderResponse`, and `Signature` labels.

Its scenario documents a chain that progresses from reconnaissance and foothold establishment through internal reconnaissance, credential attacks, remote access to a second victim, and exfiltration [2][3].

Important capture limitations are documented:

- Intra-subnet network traffic was not captured at the subnet gateways.
- Some internet traffic appears at more than one capture point and may need vantage-point selection or deduplication.

Before using it, confirm that the relevant movement crosses an observed capture point. Its raw features do not exactly match the current 32-dimensional edge vector. Either re-extract its PCAPs with the same flow exporter or define a common feature subset and retrain every model. Existing NF-CSE checkpoints should not be applied directly to the new schema.

### 7.2 DAPT 2020 — smaller pilot option

DAPT 2020 is another APT-chain dataset with timestamps, endpoint information, activity labels, and stage information [4]. It is a reasonable smaller pilot, but the exact number of independently verified lateral-movement events and the detail of its ground truth should be audited before it becomes the main evaluation dataset.

### 7.3 LMDG — strongest lateral-movement focus, high cost

LMDG was designed specifically around multi-stage and multi-hop lateral movement, with multiple virtual machines, days, and attack executions [5]. It is attractive for evaluating lead time and hop progression, but its size and preprocessing cost are much greater than Unraveled's.

### 7.4 CICAPT-IIoT — useful secondary validation

CICAPT-IIoT provides packet captures and timestamped attack information from emulated APT chains [6]. Its lateral-movement observations are relatively sparse, so it is better suited to secondary validation than to the sole statistical evaluation.

### 7.5 LANL enterprise data — chronology but weak stage attribution

The LANL authentication/network datasets contain timestamps and stable anonymized computer identifiers [7]. These identifiers can be used as graph nodes even though they are not IP strings. However, the red-team labels do not provide sufficiently rich per-flow campaign-stage attribution for the proposed precursor analysis, so the dataset would require considerable ground-truth reconstruction.

## 8. Generating lateral-movement data

There is an established way to generate defensible examples: execute complete attack chains in an isolated, authorized emulation environment and capture their telemetry. Generating or oversampling isolated tabular rows is not enough.

### Recommended emulation design

1. Create an isolated network with at least an initial victim, a second internal target, monitoring infrastructure, and an attacker/emulation controller.
2. Produce continuous benign activity during the full capture.
3. Establish a foothold on the first victim.
4. Execute internal discovery and, where appropriate, credential-access activity.
5. Use a remote service to authenticate to and successfully execute on another machine.
6. Optionally continue from the second host to a third host to create multi-hop movement.
7. Capture PCAP/network flows and host logs throughout the entire chain.
8. Record exact start/end timestamps, endpoint identities, ATT&CK techniques, success/failure, and campaign identifiers from the orchestrator.
9. Export flows with the same tool and configuration used for the other datasets.
10. Generate several campaigns with varied timing, protocols, hosts, paths, and attempts that fail or do not progress.

LADEMU is a published implementation of this approach. It integrates MITRE CALDERA for attack emulation with GHOSTS for benign behavior, captures host and network logs, and labels individual attack steps [8]. Atomic Red Team can provide individual remote-execution tests, such as PsExec-based actions, but individual atomic tests must still be sequenced, monitored, and labelled as a complete campaign [9].

All emulation must be performed only in a controlled environment where the activity is explicitly authorized.

### Using NF-CSE Day 2 as background

If retaining Day 2 is important, its traffic could be replayed or reproduced as part of the background in the isolated environment while a real emulated attack chain executes. The combined traffic should then be captured and processed again as a new emulated dataset.

This is more defensible than adding artificial CSV rows because it preserves:

- Real endpoint interactions.
- Temporal topology.
- Protocol state and flow statistics.
- Required precursor stages.
- Successful remote actions.
- A ground-truth timeline from the emulation controller.

The output should be described as a newly generated emulation dataset, not as an unmodified NF-CSE capture.

### Why tabular synthetic injection is insufficient

SMOTE, GAN-based flow generation, or manual row insertion may reproduce marginal feature distributions but cannot by itself establish:

- That credentials or prerequisites existed.
- That remote authentication or execution succeeded.
- That the same compromised host progressed through the chain.
- That topology and timing are causally coherent.
- That the generated flow was observable at the claimed time.

Such augmentation may be used in a clearly labelled training ablation, but the main test set should contain captured, held-out attack chains.

## 9. Recommended evaluation protocol

For each verified lateral-movement event:

1. Identify its onset, pivot/source, target, campaign, and success evidence.
2. Search only earlier model detections that were operationally available before the onset.
3. Determine whether each earlier alert is a true campaign precursor or merely a host-level alert.
4. Select the earliest qualifying alert under each lead-time definition.
5. Calculate nominal and operational lead separately.
6. Record the precursor stage and host role.

Report at least:

- **LM warning coverage:** fraction of verified LM events with an earlier campaign-valid detection.
- Median, quartiles, and range of operational lead time.
- Fraction of LM events with at least 1, 5, and 10 minutes of lead.
- Lead time by attack stage and host role.
- False alerts per hour/day and number of falsely alerted hosts.
- Number of complete campaigns represented.
- Results per campaign as well as aggregate results.

Campaigns, rather than individual flows, should be the primary statistical unit. Otherwise, a scan containing thousands of flows can dominate an experiment despite representing only one attack step.

If making a future-risk claim, additionally report **progression precision**: the fraction of alerted hosts that later participate in lateral movement within a specified horizon. That is optional for the detection-opportunity claim but necessary if alerts are interpreted as predictions of lateral movement.

## 10. Recommended next steps

1. Preserve the current Day 2 result as a precursor-before-scan case study, not a lateral-movement result.
2. Modify the post-analysis to use `t_available` rather than graph-window start as the primary alert timestamp.
3. Select Unraveled as the first replacement dataset and verify visibility of its documented remote movement.
4. Re-extract flows or agree on a common feature subset, then retrain all architectures.
5. Save the exact host-ID mapping with every graph sequence.
6. Split training, validation, and test data by complete campaign or capture day.
7. Add elapsed-time handling to temporal models before evaluating longer campaigns.
8. If the number of independent Unraveled chains is inadequate, evaluate LMDG or generate multiple controlled campaigns with LADEMU/CALDERA.
9. Keep synthetic row-level augmentation out of the primary test set.
10. Report intervention opportunity, alert workload, and uncertainty across campaigns rather than only the largest observed lead time.

## 11. Implemented Day-2 synthetic sensitivity experiment

The repository now includes an explicitly diagnostic synthetic-flow experiment:

- [Colab experiment notebook](code/python/notebook/synthetic_lateral_movement_experiment.ipynb)
- [Synthetic generation and sparse-overlay utilities](code/python/utils/synthetic_lm.py)
- [Structural tests](code/python/tests/test_synthetic_lm.py)

The experiment does not alter the original Day-2 CSV or graphs and does not retrain or retune the models. It creates sparse graph overlays for 54 attack scenarios:

```text
3 protocol mechanisms × 3 discovered targets × 3 horizons × 2 access paths
```

The mechanisms are SMB/RPC, RDP, and SSH. Each selected target must first appear in a real exact-port Discovery flow from `172.31.69.13`. The two access paths represent:

1. Credentials assumed to have been obtained previously, followed directly by a remote-service session.
2. Eight completed authentication-attempt flows before the remote-service session.

Each synthetic numeric vector is copied as a complete 20-feature vector from an empirical Day-2 donor. This preserves correlations between bytes, packets, duration, IATs, flags, windows, and TTL values. Donors can be benign or attack-labelled: the resulting counterfactual is assigned to an attack scenario because of its assumed endpoint, time, service, and role in the constructed sequence, not because its numeric vector is inherently malicious. The original donor label and endpoints are retained as provenance. If exact-port or same-category donor support is insufficient, the protocol fails preflight rather than using arbitrary independent random values.

The first remote-service flow is the declared synthetic LM onset. A target-originated external connection 60 seconds later acts only as constructed confirmation. Neither event proves actual authentication or execution because the experiment contains aggregate flows rather than packet or host telemetry.

Every attack scenario now has three endpoint-permuted controls, for 162 controls in total. Each control reuses the linked attack's flow features, target, service, and timestamps while substituting one of three campaign-unrelated mapped internal endpoints. Selection prefers source hosts whose retained source flows are all labelled benign. If Day 2's broad native `Infilteration` interval leaves fewer than three such hosts, the selector falls back to actual source hosts with the lowest native attack-label rate, then to mapped internal endpoints seen only as destinations. The selection tier, retained source/destination counts, and native label rate are saved as provenance. These are constructed controls, not verified benign administrative sessions. They help distinguish responses to flow/port features from responses to campaign-linked endpoint identity, topology, or temporal host state. The notebook produces an exact paired probability table; the MLP must remain invariant because it does not consume endpoint identity or graph topology.

The experiment uses operational availability:

```text
t_available = max(30-second window end, flow start + flow duration)
```

Original precursor alerts are restricted to corrected documented campaign event IDs that explicitly involve the same pivot used by the later synthetic movement. Synthetic authentication alerts and warning provenance are reported separately. The revised gate requires LM coverage, same-pivot precursor coverage, joint precursor-plus-LM coverage, operational lead, end-to-end protocol/target breadth, and scenario-macro advantage over the full control set. Results are also stratified by protocol, horizon, access path, and donor label.

For the temporal models, two fixed-checkpoint sensitivity ablations are reported. `reset_memory` removes pre-scenario temporal state. `remove_focus_context` retains memory but removes original current-window edges involving the pivot or target. The latter changes both graph structure and the local flow context; neither ablation is a causal proof of topology because the checkpoints were not trained under the intervention.

A passing result means that a fixed model reacts coherently to the constructed aggregate-flow sequence. It remains a sensitivity result and must not be presented as validation on real lateral movement. A failed preflight or failed diagnostic gate is the decision point for moving to a chain-labelled dataset.

## References

1. MITRE ATT&CK, “Lateral Movement, TA0008.” <https://attack.mitre.org/tactics/TA0008/>
2. Unraveled processed-data documentation. <https://gitlab.com/asu22/unraveled/-/raw/master/data/README.md>
3. Unraveled project and attack-scenario documentation. <https://gitlab.com/asu22/unraveled/-/raw/master/README.md>
4. “DAPT 2020 – Constructing a Benchmark Dataset for Advanced Persistent Threats.” <https://link.springer.com/book/10.1007/978-3-030-59621-7>
5. “LMDG: A Large-Scale Dataset for Lateral Movement Detection.” <https://arxiv.org/abs/2508.02942>
6. “CICAPT-IIoT: A Comprehensive Dataset for APT Detection in IIoT.” <https://arxiv.org/abs/2407.11278>
7. Los Alamos National Laboratory Cyber Security Dataset. <https://lanl.ma.ic.ac.uk/data/cyber1/>
8. FFI, “LADEMU: a modular & continuous approach for generating labelled APT datasets from emulations.” <https://www.ffi.no/en/publications-archive/lademu-a-modular-continuous-approach-for-generating-labelled-apt-datasets-from-emulations>
9. Atomic Red Team, “System Services: Service Execution — T1569.002.” <https://www.atomicredteam.io/docs/atomics/T1569.002>
10. MITRE ATT&CK, “Remote Services, T1021.” <https://attack.mitre.org/techniques/T1021/>
11. MITRE ATT&CK, “Valid Accounts, T1078.” <https://attack.mitre.org/techniques/T1078/>
12. MITRE ATT&CK, “Brute Force, T1110.” <https://attack.mitre.org/techniques/T1110/>
