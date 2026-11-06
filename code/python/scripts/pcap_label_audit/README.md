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
