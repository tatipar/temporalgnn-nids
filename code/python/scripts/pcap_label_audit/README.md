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
| `zip_list.py` | List members of a remote S3 `pcap.zip`/`logs.zip` without downloading it |
| `pcap_fetch.py`, `pcap_fetch_lib.py` | Download single zip members via HTTP range requests |
| `nfv3_relabel.py` | Liu et al. correction rules (copy of the module from commit 2086405, `feat/fair-retrain-clean`) |
| `load_nfv3.py` | Load a day CSV with timestamps, original/corrected labels and duplicate counts |
| `pcap_content.py` | DNS query names and TLS SNI per source host (pure-Python pcap reader) |
| `flow_stats_compare.py` | Flow statistics of noise-labeled hosts vs same-image controls vs others |
| `matched_pairs.py` | Matched-behaviour test: alert rates at frozen thresholds by host group |
| `pcap_find_executables.py` | Locate packets carrying a Windows executable signature |
| `evtx_summary.py` | Summarize Windows `.evtx` logs (needs `python-evtx`) |

Python dependencies: pandas, pyarrow, numpy, python-evtx (a venv lives in `~/data/cic-ids2018/.venv`).
Run scripts that read downloaded data with `python3 -I`.

**Safety:** the victim pcaps (172.31.69.13 on Thursday) contain a Metasploit stage sent by
13.58.225.34:31337. Do not reassemble or export TCP streams to files, and do not copy the pcaps
to Windows machines or shared drives.

Some steps of the audit (label counts by host and time, scan timing, contacts with
13.58.225.34, MAC comparison across days) were run as one-off commands and still need
to be scripted.
