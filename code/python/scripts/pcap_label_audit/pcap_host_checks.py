"""Per-host pcap checks for one or both days: capture span, MAC address, contacts with the victim
(first scan time, connections initiated toward it), packets with the attacker address, and
adjacent duplicate packets. With both days, also compares MAC addresses across days.

Usage: python3 -I pcap_host_checks.py wed thu
Writes host_checks.csv per day and mac_wed_vs_thu.csv to <CICIDS_DATA>/analysis/label_audit/.
"""
import collections
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_config import ATTACKER, DAYS, OUT, out_dir, pcap_files  # noqa: E402
from pcaplib import ACK, SYN, ipv4, packets  # noqa: E402

DUP_WINDOW = 64       # duplicates from capture/SPAN artifacts appear next to each other


def check(host, paths, victim):
    r = dict(host=host, files=len(paths), packets=0, first=None, last=None,
             pkts_with_victim=0, first_from_victim=None, syn_to_victim=0, synack_to_victim=0,
             pkts_with_attacker=0, first_with_attacker=None, duplicate_packets=0)
    macs = collections.Counter()
    for path in paths:
        recent, order = set(), collections.deque()
        for ts, fr in packets(path):
            r['packets'] += 1
            r['first'] = ts if r['first'] is None else min(r['first'], ts)
            r['last'] = ts if r['last'] is None else max(r['last'], ts)
            k = (ts, hash(fr))
            if k in recent:
                r['duplicate_packets'] += 1
            else:
                recent.add(k); order.append(k)
                if len(order) > DUP_WINDOW:
                    recent.discard(order.popleft())
            p = ipv4(fr)
            if not p:
                continue
            if p['src'] == host:
                macs[p['src_mac']] += 1
            if host != victim and victim in (p['src'], p['dst']):
                r['pkts_with_victim'] += 1
                if p['src'] == victim and r['first_from_victim'] is None:
                    r['first_from_victim'] = ts
                if p['src'] == host and p['tcp_flags'] is not None:
                    if p['tcp_flags'] & SYN and not p['tcp_flags'] & ACK:
                        r['syn_to_victim'] += 1
                    elif p['tcp_flags'] & SYN and p['tcp_flags'] & ACK:
                        r['synack_to_victim'] += 1
            if ATTACKER in (p['src'], p['dst']):
                r['pkts_with_attacker'] += 1
                if r['first_with_attacker'] is None:
                    r['first_with_attacker'] = ts
    r['mac'] = macs.most_common(1)[0][0] if macs else None
    r['mac_share'] = round(macs.most_common(1)[0][1] / sum(macs.values()), 4) if macs else None
    for c in ('first', 'last', 'first_from_victim', 'first_with_attacker'):
        if r[c] is not None:
            r[c] = pd.to_datetime(r[c], unit='s')
    return r


res = {}
for day in sys.argv[1:]:
    C = DAYS[day]
    rows = []
    for host, paths in sorted(pcap_files(day).items(), key=lambda kv: tuple(map(int, kv[0].split('.')))):
        r = check(host, paths, C['victim'])
        r['role'] = 'victim' if host == C['victim'] else 'noise host' if host in C['noise'] else \
            'control' if host in C['controls'] else 'other'
        rows.append(r)
        print(day, {k: v for k, v in r.items() if k not in ('files',)}, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(os.path.join(out_dir(day), 'host_checks.csv'), index=False)
    res[day] = t
    scan = t[t.role == 'noise host'].sort_values('first_from_victim')[['host', 'first_from_victim']]
    print(f'\n{day}: first packet from victim {C["victim"]} per noise host (scan order)\n', scan.to_string(index=False))

if {'wed', 'thu'} <= res.keys():
    m = res['wed'][['host', 'mac', 'role']].merge(res['thu'][['host', 'mac', 'role']], on='host', suffixes=('_wed', '_thu'))
    m['same_mac'] = m.mac_wed == m.mac_thu
    m.to_csv(os.path.join(OUT, 'mac_wed_vs_thu.csv'), index=False)
    print('\nMAC comparison\n', m.to_string(index=False))
