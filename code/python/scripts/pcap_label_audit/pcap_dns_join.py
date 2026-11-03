"""Join NF-v3 DNS flows of the noise-labeled hosts to the DNS queries in their pcaps
(key: source host, source port, DNS query ID) and measure label rates per queried domain.

Usage: python3 -I pcap_dns_join.py wed|thu
Writes dns_join_summary.json, dns_label_by_domain.csv, dns_label_by_name.csv to
<CICIDS_DATA>/analysis/label_audit/<day>/.
"""
import json
import os
import struct
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_config import DAYS, KEY, out_dir, pcap_files  # noqa: E402
from load_nfv3 import load  # noqa: E402
from pcaplib import packets, ipv4  # noqa: E402


def qname(buf, off=12):
    labels = []
    while off < len(buf):
        n = buf[off]
        if n == 0 or n & 0xC0:
            break
        labels.append(buf[off + 1:off + 1 + n].decode('ascii', 'replace'))
        off += n + 1
    return '.'.join(labels).lower()


day = sys.argv[1]
C = DAYS[day]
OUT = out_dir(day)
files = pcap_files(day)
rows = []
for host in sorted(C['noise'], key=lambda x: int(x.split('.')[-1])):
    for path in files.get(host, []):
        for ts, fr in packets(path):
            p = ipv4(fr)
            if not p or p['proto'] != 17 or p['dport'] != 53 or p['src'] != host:
                continue
            d = fr[p['payload']:]
            if len(d) <= 12 or d[2] & 0x80:          # need a query (QR bit = 0)
                continue
            rows.append((host, p['sport'], struct.unpack('>H', d[0:2])[0], ts, qname(d)))
q = pd.DataFrame(rows, columns=['src', 'sport', 'qid', 't', 'qname'])
n_query_packets = len(q)
q = q.drop_duplicates(['src', 'sport', 'qid'])

df = load(C['csv'])
n = df[df.IPV4_SRC_ADDR.isin(q.src.unique()) & (df.L4_DST_PORT == 53)].drop_duplicates(KEY)
m = n.merge(q, left_on=['IPV4_SRC_ADDR', 'L4_SRC_PORT', 'DNS_QUERY_ID'], right_on=['src', 'sport', 'qid'], how='left')
S = dict(hosts=sorted(q.src.unique().tolist()), query_packets=n_query_packets,
         nfv3_dns_flows=len(n), matched_share=round(float(m.qname.notna().mean()), 4),
         median_abs_time_offset_s=round(float((m.FLOW_START_MILLISECONDS / 1000 - m.t).abs().median()), 3))
m = m[m.qname.notna()].copy()
S['label_rate_per_host'] = m.groupby('IPV4_SRC_ADDR').Label.mean().round(3).to_dict()
m['domain'] = m.qname.str.split('.').str[-2:].str.join('.')
g = m.groupby('domain').agg(flows=('Label', 'size'), label_rate=('Label', 'mean'), hosts=('IPV4_SRC_ADDR', 'nunique'))
g.sort_values('flows', ascending=False).round(3).to_csv(f'{OUT}/dns_label_by_domain.csv')
gn = m.groupby('qname').Label.agg(flows='size', label_rate='mean')
gn.sort_values('flows', ascending=False).round(3).to_csv(f'{OUT}/dns_label_by_name.csv')
big = gn[gn.flows >= 10]
S['names_with_10plus_queries'] = dict(
    names=len(big), mixed_label_share=round(float(((big.label_rate > 0) & (big.label_rate < 1)).mean()), 3),
    never_labeled=int((big.label_rate == 0).sum()), always_labeled=int((big.label_rate == 1).sum()))
dom5 = g[g.flows >= 5]
S['domains_with_5plus_queries'] = dict(domains=len(dom5), never_labeled=int((dom5.label_rate == 0).sum()),
                                      always_labeled=int((dom5.label_rate == 1).sum()))
S['top_domains'] = g.sort_values('flows', ascending=False).head(15).round(3).to_dict('index')
json.dump(S, open(f'{OUT}/dns_join_summary.json', 'w'), indent=2)
print(json.dumps(S, indent=2))
