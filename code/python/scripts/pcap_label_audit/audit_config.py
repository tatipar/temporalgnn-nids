"""Shared configuration for the NF-v3 / PCAP label audit: paths, hosts and documented attack windows.

Paths default to the layout used during the audit and can be overridden with environment variables:
CICIDS_DATA (data root), NFV3_WED_CSV and NFV3_THU_CSV (NF-v3 day CSVs).
"""
import os
import re
from collections import defaultdict

DATA = os.environ.get('CICIDS_DATA', os.path.expanduser('~/data/cic-ids2018'))
OUT = os.path.join(DATA, 'analysis', 'label_audit')

KEY = ['FLOW_START_MILLISECONDS', 'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'PROTOCOL']

ATTACKER = '13.58.225.34'
DNS_RESOLVER = '172.31.0.2'
TELEMETRY_RE = r'^(65\.52\.108\.|52\.165\.231\.|13\.89\.190\.)'   # Microsoft telemetry endpoints (port 443)
RDP_SCANNERS = ['5.101.40.105', '5.101.40.43']


def _hosts(*last):
    return {f'172.31.69.{i}' for i in last}


# Documented attack span per day: earliest start and latest end over the Dropbox, victim-attacker
# communication and Nmap intervals in nfv3_relabel.py (Unix seconds, UTC).
DAYS = {
    'wed': dict(
        name='Wednesday-28-02-2018',
        csv=os.environ.get('NFV3_WED_CSV', os.path.expanduser('~/Downloads/cicids2018v3_wed2802.csv')),
        pcap_dir=os.path.join(DATA, 'wednesday-28-02', 'pcap'),
        victim='172.31.69.24',
        noise=_hosts(5, 8, 9, 10, 12, 13, 14, 16, 17),
        controls=_hosts(23, 26),
        attack_span=(1519828404, 1519843200),
    ),
    'thu': dict(
        name='Thursday-01-03-2018',
        csv=os.environ.get('NFV3_THU_CSV', os.path.expanduser('~/Downloads/cicids2018v3_thu0103.csv')),
        pcap_dir=os.path.join(DATA, 'thursday-01-03', 'pcap'),
        victim='172.31.69.13',
        noise=_hosts(5, 8, 9, 10, 12, 14, 16, 17),
        controls={'172.31.65.93', '172.31.66.97', '172.31.66.78', '172.31.69.23'},
        attack_span=(1519912390, 1519933092.182726),
    ),
}


def pcap_files(day):
    """Map host IP -> sorted list of pcap files for that host (multi-part captures are grouped)."""
    d = DAYS[day]['pcap_dir']
    out = defaultdict(list)
    for name in sorted(os.listdir(d)):
        m = re.search(r'(\d+\.\d+\.\d+\.\d+)', name)
        if m and name.endswith('.pcap'):
            out[m[1]].append(os.path.join(d, name))
    return dict(out)


def out_dir(day):
    p = os.path.join(OUT, day)
    os.makedirs(p, exist_ok=True)
    return p
