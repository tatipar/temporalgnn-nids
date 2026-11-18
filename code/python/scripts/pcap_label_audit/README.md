# PCAP-level audit of NF-v3 Infiltration labels

Scripts used to audit the NF-CSE-CIC-IDS2018-v3 Infiltration labels against the raw
CSE-CIC-IDS2018 packet captures (Wednesday 28-02 and Thursday 01-03), and to test whether
the trained models score identical traffic differently depending on the host.

Data lives outside the repository, in `~/data/cic-ids2018/`:

- `thursday-01-03/pcap/`, `wednesday-28-02/pcap/`: per-host pcaps fetched with `pcap_fetch.py`.
- `analysis/`: outputs (`analysis/matched/` for the matched-behaviour test).
- NF-v3 day CSVs: `~/Downloads/cicids2018v3_thu0103.csv`, `~/Downloads/cicids2018v3_wed2802.csv`.
- Model scores and IP attribution: `~/data/nids-mitre-results/extracted/`, produced by
  `code/python/notebook/paper_eight_models_ip_label_audit_colab.ipynb`.

Several scripts hard-code these paths; edit them before running elsewhere.

| Script | Purpose |
|---|---|
| `audit_config.py` | Shared paths, victims, noise hosts, controls and documented attack spans per day |
| `pcaplib.py` | Minimal pure-Python pcap/Ethernet/IPv4 reader used by the pcap scripts |
| `zip_list.py` | List members of a remote S3 `pcap.zip`/`logs.zip` without downloading it |
| `pcap_fetch.py`, `pcap_fetch_lib.py` | Download single zip members via HTTP range requests |
| `nfv3_relabel.py` | Liu et al. correction rules (copy of the module from commit 2086405, `feat/fair-retrain-clean`) |
| `load_nfv3.py` | Load a day CSV with timestamps, original/corrected labels and duplicate counts |
| `nfv3_label_audit.py <day>` | Labels vs Liu rules, labeled sources, timing vs attack span, tail sources, matched-behaviour label rates, duplicates by host group, controls |
| `pcap_dns_join.py <day>` | Join NF-v3 DNS flows to pcap DNS queries (host, source port, query ID); label rate per domain |
| `pcap_host_checks.py wed thu` | Per-host capture span, MAC, contacts with the victim (scan time, connections initiated), packets with 13.58.225.34, duplicate packets; MAC comparison across days |
| `cicflowmeter_compare.py` | Label timing in CIC's own Thursday CSV vs the documented attack span |
| `pcap_content_compare.py` | DNS domain mix and telemetry SNI of noise hosts vs same-image controls (cosine similarity) |
| `pcap_content.py` | DNS query names and TLS SNI per source host (pure-Python pcap reader) |
| `flow_stats_compare.py` | Flow statistics of noise-labeled hosts vs same-image controls vs others |
| `matched_pairs.py` | Matched-behaviour test: alert rates at frozen thresholds by host group |
| `matched_pairs_dedup.py` | Same test for experiment 3 (duplicates removed: M0_P0, M1_P0, MLP); rebuilds and verifies flow attribution from the CSV |
| `pcap_find_executables.py` | Locate packets carrying a Windows executable signature |
| `evtx_summary.py` | Summarize Windows `.evtx` logs (needs `python-evtx`) |

Python dependencies: pandas, pyarrow, numpy, python-evtx (a venv lives in `~/data/cic-ids2018/.venv`).
Run scripts that read downloaded data with `python3 -I`.

**Safety:** the victim pcaps (172.31.69.13 on Thursday) contain a Metasploit stage sent by
13.58.225.34:31337. Do not reassemble or export TCP streams to files, and do not copy the pcaps
to Windows machines or shared drives.

## Reproducing the audit

```bash
S=code/python/scripts/pcap_label_audit
for d in wed thu; do python3 -I $S/nfv3_label_audit.py $d; python3 -I $S/pcap_dns_join.py $d; done
python3 -I $S/pcap_host_checks.py wed thu        # reads ~3 GB of pcaps, ~10-20 min
python3 -I $S/cicflowmeter_compare.py
python3 -I $S/flow_stats_compare.py
~/data/cic-ids2018/.venv/bin/python -I $S/matched_pairs.py      # needs pyarrow
~/data/cic-ids2018/.venv/bin/python -I $S/matched_pairs_dedup.py   # after matched_pairs.py; reads the dedup results
cd ~/data/cic-ids2018/thursday-01-03/pcap && python3 -I $OLDPWD/$S/pcap_content.py \
  capEC2AMAZ-O4EL3NG-172.31.69.{5,8,9,10,12,14,16,17}.pcap capEC2AMAZ-O4EL3NG-172.31.{65.93,66.97,66.78,69.23}.pcap \
  > ~/data/cic-ids2018/analysis/matched/pcap_content_thu.json && cd -
python3 -I $S/pcap_content_compare.py
```

Outputs go to `~/data/cic-ids2018/analysis/label_audit/<day>/` (and `analysis/matched/` for the
model-score scripts).

## Choices behind the matched-behaviour test

Recovered from the session transcripts and re-checked on 2026-11-18.

**Host groups (Thursday, Test 2).**
- *Victim:* 172.31.69.13, the documented infected host. It is excluded from every group.
- *Noise hosts:* the eight other `.69` hosts whose flows carry original labels (.5 .8 .9 .10 .12 .14 .16 .17).
- *Same-image controls:* 172.31.65.93, 172.31.66.97, 172.31.66.78 and 172.31.69.23.
- *Other hosts:* every remaining internal (172.31.x.x) host.

**How the controls were chosen.** CIC names each per-host capture in `pcap.zip` as `cap<HOSTNAME>-<IP>`. The noise hosts and the victim all have the host name `EC2AMAZ-O4EL3NG`. On Wednesday about 200 IPs share it (201 in `~/data/cic-ids2018/wed_members.tsv`), so these are presumably instances cloned from one Windows image. Other groups are `DESKTOP-AN3U28N` (76), `PC1` (75) and `WIN-J6GMIG1DQE5` (75); Ubuntu hosts are `UCAP` files. The controls were picked in two steps:
1. Among Thursday `EC2AMAZ-O4EL3NG` hosts outside 172.31.69.0/24 with zero labeled flows (189 hosts), they were ranked by DNS flows to 172.31.0.2. The top three were kept: .65.93 (11,268 DNS flows, 24 telemetry), .66.97 (11,057; 76) and .66.78 (10,605; 219). Each has about 16k flows, more than any noise host, which rules out activity volume as the explanation.
2. 172.31.69.23 was added as a same-subnet control. It was also the Wednesday control. Its own outbound flows are unlabeled. On Thursday it receives 265 labeled flows, all from the victim's documented Nmap scan.

**Representative runs and thresholds.**
- *Seeds:* the seeds in `matched_pairs.py` and `matched_pairs_dedup.py` are the champions in each results folder's `champions_by_validation.csv`, all with `selection_status == selected` under the one-SD rule. In the main comparison they are MLP 777, Global 42, Host-history 2024, E-GraphSAGE 2024, M0_P0 2024, M0_P1 123, M1_P0 2024 and M1_P1 2024. In the deduplicated comparison they are M0_P0 99, M1_P0 123 and MLP 2024.
- *Thresholds:* each threshold is that run's validation max-F1 threshold. For the main comparison it is `frozen_thresholds` in the analysis manifest, which equals `Optimal_Threshold`. For the deduplicated comparison it is `record.json`.
