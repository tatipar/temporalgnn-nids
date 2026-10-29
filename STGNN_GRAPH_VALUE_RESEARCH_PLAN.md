# When does a temporal GNN add value for flow intrusion detection?

## Purpose and scope

This document is an executable research plan for the binary, flow-level
`Infilteration` experiments in this repository. It separates work already done
from experiments that would identify whether temporal memory, inter-host message
passing, graph construction, or their interaction accounts for performance.
The current ground truth is the **original** label in the CICIDS2018-v3 CSVs.
It is not ground truth for lateral movement, attack stages, or successful host
compromise. Results from a revised label set or another dataset must be
reported as a separate experiment.

The research question is:

> Given the attributes of a completed flow and the prior behavior of its
> endpoints, when does information propagated from other endpoints improve
> attack detection, source coverage, and alert workload?

An overall Test 2 AP lead is not, by itself, evidence that message passing is
generally useful. Equally, an Edge GRU lead on some sources does not establish
that graph structure is unnecessary. `EdgeGRU_Baseline_NoX` still aggregates
flow embeddings by source and uses the temporal states of both endpoints;
`ST No Message` still uses mean summaries of incoming and outgoing flows at
each endpoint. Neither is a model of independent flows plus time alone.

## Existing protocol and evidence

The construction notebooks are
[`cic_ids2018_v3_graph_creation.ipynb`](code/python/notebook/cic_ids2018_v3_graph_creation.ipynb)
and
[`cic_ids2018_v3_graph_creation_TEST2.ipynb`](code/python/notebook/cic_ids2018_v3_graph_creation_TEST2.ipynb).
They create directed host-to-host multigraphs in 30-second windows, with one
edge per flow, complete-flow statistics as edge attributes, and constant
placeholder node features. ST-GNN builds node features by averaging projected
outgoing and incoming edge attributes, applies two GATv2 layers, updates a GRU
state per IP, and classifies each edge from its endpoint states and own edge
attributes. The code is in [`models.py`](code/python/utils/models.py).
Wednesday is split chronologically into train, validation, and Test 1;
Thursday is Test 2. The Thursday IP-to-ID map is built separately, so a numeric
`source_gid` should only be interpreted within that day. Do not treat an ID
match across days as a host match without an explicit IP mapping.
The four executed Colab notebooks below pinned their original model code to
commit `2262b53d155f9e8cf5221516f47e4170e6cd0d6e`; future runs must
record their own code commit and preprocessing version.

The audit notebook
[`audit_original_infiltration_colab.ipynb`](code/python/notebook/audit_original_infiltration_colab.ipynb)
reproduced all 25 Test 1 model-by-seed evaluations and all five
validation-selected Test 2 champions by checkpoint inference. Test 1 has
191,679 flows and 3,434 positive labels; Test 2 has 2,135,957 flows and
103,825 positive labels. These figures establish reproducibility of the
original code, checkpoints, graphs, and thresholds. They do not audit label
correctness or causal feature availability.

| Test 2 champion | AP | F1 | Recall | FPR | Attack sources detected |
| --- | ---: | ---: | ---: | ---: | ---: |
| ST-GNN | 0.8384 | 0.7576 | 0.9419 | 0.0278 | 141/380 |
| Edge GRU | 0.7307 | 0.6502 | 0.7110 | 0.0243 | 113/380 |
| ST Uniform | 0.6883 | 0.6256 | 0.6993 | 0.0274 | 188/380 |
| ST No Message | 0.6601 | 0.7115 | 0.6681 | 0.0107 | 111/380 |

The gain analysis
[`analyze_stgnn_gain_colab.ipynb`](code/python/notebook/analyze_stgnn_gain_colab.ipynb)
already reports paired flow outcomes, history/degree strata, source 1195 and
top-ten exclusions, equal alert budgets, source-level AP for the 18 sources
with at least 50 positive and 50 negative flows, and the first detected attack
from source 1195. Of ST-GNN's 25,558 true positives that Edge GRU misses,
22,310 (87.29%) originate from source 1195. Excluding that source, ST-GNN AP
is 0.7495 versus Edge GRU's 0.7652. Among the 18 assessable sources, ST-GNN
has higher AP on four, Edge GRU on 13, and one is tied. A reset-memory pass on
the *trained* ST-GNN checkpoint lowers full Test 2 AP from 0.8384 to 0.4507;
this is checkpoint sensitivity, not a trained no-memory ablation.

The spatial ablation notebook
[`stgnn_spatial_ablation_colab.ipynb`](code/python/notebook/stgnn_spatial_ablation_colab.ipynb)
already trains five seeds each of uniform attention and no inter-node message
passing, selects champions on validation, compares all five Test 1 seeds, and
reports Test 2 subsets and equal alert budgets. ST No Message has slightly
higher five-seed mean Test 1 AP than ST-GNN (0.6902 versus 0.6710); excluding
source 1195 on Test 2, it has AP 0.7866 versus ST-GNN's 0.7495. ST Uniform's
seed 123 fails badly, so its mean and variance must both be reported.

The operational notebook
[`temporal_operational_postanalysis_colab.ipynb`](code/python/notebook/temporal_operational_postanalysis_colab.ipynb)
already reports source detection, first-hit window, 5- and 30-minute counts,
attack-window coverage, longest miss streak, support strata, flow and
source-window alert workload, hourly workload, and an attribute-availability
audit. Source 1195 first receives a true positive 113.5 nominal minutes after
its first attack-labeled window with ST-GNN, versus 97.0 minutes with Edge GRU.
Some 16.34% of Test 2 flows finish after their start window ends. These nominal
delays are retrospective and are not deployment latencies.

The paper by Liu et al. documents errors and revised labeling for
CSE-CIC-IDS2018; matching its revised records to this repository's
CICIDS2018-v3 flows requires an explicit alignment audit before reuse. See
[the authors' dataset documentation](https://intrusion-detection.distrinet-research.be/CNS2022/CSECICIDS2018.html).

### Working hypotheses, not assumed properties of the graph

| Hypothesis | Observable prediction | Main test |
| --- | --- | --- |
| Other hosts carry incremental attack information | Real-neighbor messages outperform matched no-message and randomized-message controls | Experiments 2–4 |
| The advantage requires a sustained, high-volume source | Gains appear late or in high-support sources, but not at first detection | Experiment 1 |
| Direction or graph representation hides useful behavior | A role-aware or flow-centric graph improves over its own no-message control | Experiment 4 |
| Retrospective attributes inflate the apparent gain | The gain shrinks when only completed, available flows form the context | Experiment 5 |
| The effect is specific to the original capture or labels | Model ranking changes on revised labels or independent campaigns | Experiment 6 |

No fixed degree, density, homophily, or window length guarantees good GNN
performance. In this edge-classification task, multiple benign and malicious
flows may share an endpoint. Whether that mixture is useful must be measured
relative to target-flow attributes and endpoint history, rather than inferred
from node-label homophily results alone.

## Shared experimental rules

1. **Freeze the research question before a confirmatory test.** Existing Test 2
   analyses are exploratory. Choose representations, windows, and thresholds
   with train/validation only. Use a held-out campaign, day, or dataset for the
   final confirmation; do not call another selection on Test 2 confirmation.
2. **Preserve exact alignment.** Every prediction artifact must include or be
   verifiably joined to a stable flow key, source and destination identifiers,
   window or event time, label, model score, threshold, checkpoint hash,
   graph/preprocessing version, and split. Recheck AP/F1 and flow counts
   against the audited originals before extending them.
3. **Keep operating points honest.** Choose thresholds on validation only and
   carry them unchanged to test. A threshold found by maximizing recall under
   1% FPR *on Test 2* is an oracle diagnostic, not an operational result.
   Equal-top-K tables are ranking diagnostics, not fixed-threshold deployment
   claims. Report prevalence for every subset because AP changes with it.
4. **Report multiple units.** Primary: flow AP, source detection fraction, and
   true-positive attack flows within a predeclared alert budget. Also report
   first-hit time with never-detected sources retained as censored,
   source-window alerts without labeled attacks, and false alerts per day.
   Per-source AP is defined only where both classes have enough support;
   report how much of the positive population those sources cover.
5. **Use meaningful uncertainty.** Pair models on the same flows and sources.
   Do not bootstrap individual flows as independent observations. Prefer
   campaign or day blocks when available; a source bootstrap on this one
   campaign is exploratory and may understate dependence. Repeat new training
   across the five established seeds and show individual runs and mean/spread.
6. **Track time of availability.** In the current data, full duration, packet,
   IAT, and byte statistics are only known when a flow ends. A claim about
   online detection must score after all required inputs are available and
   must not use flows beginning later than the decision time.

## Experiment 1 — Extend, do not repeat, source-level diagnosis

**Status:** The source split, exclusions, source AP, first detection, 5/30-minute
coverage, support groups, and alert workload are complete in the two analysis
notebooks above. Start from their verified probability caches and CSVs. There
is no reason to rerun the original inference simply to reproduce these tables.
At the time this plan was written, [`early_source_dynamics_colab.ipynb`](code/python/notebook/early_source_dynamics_colab.ipynb) had not yet been executed. It implements the remaining retrospective early-window analysis from cached predictions. The repository now includes its executed outputs; the findings in this plan remain the original planning snapshot.

**Remaining question:** Is ST-GNN's advantage concentrated late in a sustained
attack, or does it help detect the first few attack-containing windows of a
source?

**Protocol:** For each attacked source, define time zero as its first
attack-labeled window. With each model's validation threshold, count true
positives in that source's first 1, 5, and 20 attack-containing **windows**
and in the first 5, 30, and 120 nominal minutes. The saved window IDs do not
establish a chronological order among flows *within* a window; a first-N-flow
analysis additionally requires exact raw timestamps and verified row-to-graph
alignment. Report cumulative detection curves at the source level; include a
separate never-detected category rather than computing a median only among
detected sources. Repeat the curves with
source 1195 excluded and by attack-count group (1, 2–9, 10–49, 50+). Compute
paired ST-GNN minus Edge GRU and ST-GNN minus ST No Message differences. The
existing top-K analysis gives equal *global* budgets; optionally add a
predeclared time-varying budget, such as N source-window alerts per nominal
hour, selected without test labels. Do not spend a separate oracle budget on
each attacked source at test time: a detector cannot know which sources will
be attacked.

**Inputs/outputs:** Use the verified `.npz` probability caches under
`MyDrive/nids-mitre/results_stgnn_spatial_ablation_v1/diagnostic_test2_spatial/`
and `MyDrive/nids-mitre/results_comparison/temporal_operational_v1/`, plus
`window_ids_test2.npz` in the latter directory. These large Colab/Drive
artifacts are not versioned in this repository.
Write one aligned per-flow table with window and within-graph edge index, one
per-source/horizon table, and one figure showing cumulative coverage with
denominators. Preserve a manifest of hashes
and thresholds. This is still a retrospective start-window analysis until
Experiment 5 rebuilds the time axis.

**Interpretation:** Higher late recall with no early gain would support a
"mature, high-volume campaign" explanation. A reproducible first-window gain
across many sources would be stronger evidence of useful
context for early detection. Neither result by itself attributes the gain to
GAT message passing.

## Experiment 2 — Audit whether graph neighborhoods contain usable signal

**Question:** What additional information is present in the actual graph
neighborhood of a target flow, and how does this vary across sources?

**Protocol:** Read graph files in chronological order. For each edge, record
source/destination in-degree, out-degree, distinct peers, parallel-edge count,
component size, directed one- and two-hop reachability, previous appearances
of both endpoints, and time since their last appearance. Compute these
properties for the complete 30-second snapshot. To compute the **strictly
prior, already completed** subgraph at the edge's available time, first join
raw flow start/duration records to graph edges with an audited unique key; the
saved graphs alone contain only a window-level timestamp. If the join fails,
report snapshot statistics and defer the causal subgraph to Experiment 5. For
diagnosis only, compute the fraction of other incident edges with positive
labels, separating shared source from shared destination, past from current
window, and excluding the target edge. Show the distributions for positive and
negative flows, including cold-start sources and source 1195 separately.
Do not feed these diagnostic label fractions to a model.

Check whether a target edge's original attributes already predict attack well
and whether adding label-free neighbor summaries improves a simple train/val
baseline. The key comparison is conditional: information about *other* flows
given the target's attributes and its endpoints' own prior activity. Use only
training data to choose any cutoffs, feature subsets, or graph variants.

**Outputs:** A machine-readable per-flow diagnostic table; per-window degree,
component, and positive-support distributions; plots stratified by degree,
history, source, and attack age. State the share of positive flows with zero
usable prior neighbors and the share whose only apparent "context" is the
target flow itself. Check that counts reconstruct every graph's number of
edges and that no target edge enters its own neighbor summary.

**Decision:** If useful prior neighbors are rare, prioritize temporal/flow
features and a better event representation. If they are common but mixed with
many benign flows, prioritize role-aware or selective aggregation. The label
mix is a diagnostic, not proof that a GNN can learn it.

## Experiment 3 — Controlled memory-by-message-passing factorial ablation

**Question:** Is any gain attributable to temporal memory, inter-node message
passing, or their interaction under matched inputs and training?

**Design:** Train all four cells of a 2 × 2 design:

| | No inter-node messages | Inter-node messages |
| --- | --- | --- |
| No persistent memory | Nodewise encoder + GRU with zero prior state + edge classifier | GAT encoder + GRU with zero prior state + edge classifier |
| Persistent memory | Nodewise encoder + persistent GRU + edge classifier | GAT encoder + persistent GRU + edge classifier |

Use the same projected edge attributes, incoming/outgoing node summaries,
GRU cell, edge classifier, loss, optimizer, early stopping, training windows,
and seed list in all cells. The no-memory control should be **trained** with
zero prior state each window and without carrying state forward, not obtained
by resetting a memory-trained checkpoint at inference. Where practical, hold
widths and parameter counts close and report
their exact values; if equalization changes architecture, report both the
matched-width and matched-parameter comparison. Keep Edge GRU as an external
baseline because its encoder and decoder differ from the four controlled
cells. Retain ST Uniform as an attention-weighting diagnostic, not as the
message-passing-off cell.

Select checkpoints and thresholds from validation only. Evaluate all five
seeds on Test 1. Evaluate validation-selected champions on the reserved test,
and if a new independent confirmation set is available, apply the frozen
selection there. Report paired seed differences, source coverage, alert
workload, and the interaction between memory and messages; avoid claiming
that differences in pooled AP alone establish causality.

**Outputs:** A configuration file for every cell, seed-level validation and
Test 1 results, checkpoint hashes, champion-selection record, aligned test
scores, and a single table with factor contrasts. A failed run stays in the
table; do not discard ST Uniform-like bad seeds without a documented reason.

## Experiment 4 — Test graph construction and direction

**Question:** Does the present IP-to-IP directed graph expose the behavior that
message passing needs, or is a different relation more informative?

**Stage A: negative control.** Preserve the real edge endpoints for the
classifier, node summaries, temporal memory, target edge attributes, and
labels, but randomize **only the messages used by the propagation layer**
within predeclared time/degree strata. Train and evaluate the corresponding
control from scratch. Frozen-checkpoint shuffling is a sensitivity check only:
it also creates distribution shift. If message randomization leaves performance
unchanged, evidence for informative topology is weak.

**Stage B: representation candidates.** Guided by Experiment 2, compare the
original directed graph with one predeclared alternative, such as separate
incoming and outgoing message channels or a bidirectional relation with edge
direction as a feature. If that is inconclusive and resources permit, test a
flow-as-node or communication-channel-as-node graph in which target flow
attributes remain explicit and relations connect flows sharing an endpoint.
Constrain each representation to the same observation horizon and information
availability. Record graph size, degree distribution, training cost, inference
cost, and parameters. Do not select the best topology on Test 2 and then call
its Test 2 score confirmatory.

**Outputs and interpretation:** For each representation, report performance
against its own no-message control at fixed alert budgets and by source.
Improvement over both the original graph and a rewired control would support
the proposed relation; improvement over the original alone could reflect
capacity, preprocessing, or a different target-flow encoding.

## Experiment 5 — Causal availability and temporal resolution

**Question:** How much of the observed performance survives when predictions
use only attributes available at alert time?

**Protocol:** Compute `available_at = flow_end + exporter_delay` where the raw
data support it. If exporter delay is unknown, report zero delay as an optimistic
lower bound and treat larger plausible delays as sensitivity analyses. Build
snapshots from flows that have become available, not from all flows that began
in the interval. For a target completed flow, only prior available flows may
contribute to its context; its own attributes may enter the decoder at its
completion. Persist and expire memory explicitly across empty windows and
long gaps. Rebuild train/validation/test under the same rule, and retrain every
model: the causal data distribution differs from the existing one.

Compare a small predeclared set of windows, for example 5, 30, and 120 seconds,
chosen on validation. Track the number of observed neighbors, first-alert
time measured from **availability**, detection within fixed horizons after
attack onset, source coverage, and false alerts per day. State whether the
decision is made per completed flow or at window closure. Do not describe a
start-window timestamp as online detection latency. Use the existing DAPT2020
pilot's completed-flow construction only as an implementation reference;
its task and labels differ from this Infiltration study.

**Outputs:** A versioned causal graph builder, row-level timestamp audit,
per-window and per-model results, and a comparison against the retrospective
protocol. If an apparent gain disappears under causal availability, report
that as a central methodological finding.

## Experiment 6 — Independent labels, sources, and campaigns

**Question:** Do identified conditions and model rankings persist beyond the
original Thursday Infiltration capture?

**Labels:** First audit exact record-level alignment between CICIDS2018-v3
flows and the revised CSE-CIC-IDS2018 labels. Report unmatched, ambiguous, and
changed labels before training. Do not merge the original and revised labels
under one metric. If exact matching is infeasible, treat the revised dataset
as a new benchmark with its own preprocessing and protocol.

**External data:** The current workspace contains a
[`DAPT2020 pilot`](Dapt2020_pilot/DAPT2020_GRAPH_PILOT_REPORT.md)
(check it into a reproducible artifact before citing it in a manuscript): a single
campaign and one seed in which Edge GRU AP is 0.9622, Static GNN AP is 0.9590,
and ST-GNN AP is 0.9309. This is evidence that rankings can change, not a
multi-campaign estimate. AIT-LDS processing notes are in
[`ait_nds/README.md`](ait_nds/README.md); they do not yet constitute an aligned
external evaluation. Choose datasets only after confirming flow granularity,
source/destination identities, attack annotations, timestamps, and feature
availability. Prefer multiple independent campaigns/networks; if only one
campaign exists, describe it as a case study.

**Confirmation protocol:** Freeze model family, window, metrics, and
threshold-selection rule from earlier stages. Train/validate without any
test-campaign examples, test all predeclared models on the same episodes, and
report campaign-level paired differences and interval estimates. Source-holdout
within one campaign is a useful stress test but does not substitute for a new
campaign or network. Avoid asserting lateral movement detection unless that
event is independently annotated; `Infilteration` is not such an annotation.

## Recommended execution order and publication decision

1. **Now, no retraining:** Experiment 1's *remaining* early-flow/attack-age
   analyses and Experiment 2's neighborhood audit. Their purpose is to choose
   a small number of falsifiable hypotheses.
2. **Controlled training:** Experiment 3, followed by Experiment 4's negative
   control and one graph alternative selected from the audit.
3. **Operational confirmation:** Experiment 5, then Experiment 6 on independent
   labels/campaigns. Freeze choices before each new held-out evaluation.

The strongest positive result would be a message-passing gain over a matched
no-message model, across sources or campaigns, at a fixed alert workload and
under causal feature availability. A strong negative result would be that
measured neighborhood information is sparse or unhelpful, and controlled
propagation does not improve on node-summary memory models. Either outcome can
support a paper about the **conditions under which temporal graph context has
incremental value**. The current pooled Test 2 lead alone cannot establish
that claim.

## Primary literature guiding the plan

- Lo et al., [E-GraphSAGE](https://arxiv.org/abs/2103.16329): edge attributes
  and topology in flow NIDS.
- Santos Escriche et al., [CCSTGN](https://people.kth.se/~gyuri/Pub/SantosNKD-ChannelCentricNIDS-CNS2024.pdf):
  channel-centric graph, continuous time, and the strong effect of preprocessing.
- Van Langendonck et al., [PPT-GNN](https://arxiv.org/html/2406.13365v1):
  flow-as-node relations, short windows, and multi-dataset evaluation.
- Venturi et al., [Practical Evaluation of GNNs in NIDS](https://ceur-ws.org/Vol-3488/paper29.pdf):
  graph-window completeness and detection-delay trade-offs.
- Rossi et al., [Temporal Graph Networks](https://arxiv.org/abs/2006.10637):
  memory and event-sequence modeling.
- Ma et al., [Is Homophily a Necessity for GNNs?](https://arxiv.org/abs/2106.06134):
  why a simple homophily rule does not guarantee success. That work primarily
  studies node classification, so it is motivation rather than a theorem for
  our edge-classification problem.
