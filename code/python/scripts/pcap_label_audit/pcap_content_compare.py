"""Compare DNS domain mix and TLS SNI of the noise-labeled hosts with same-image controls (Thursday),
using the JSON written by pcap_content.py.

Usage: python3 -I pcap_content_compare.py [pcap_content_thu.json]
Writes pcap_content_compare.json and dns_domain_host_cosine.csv to <CICIDS_DATA>/analysis/label_audit/thu/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_config import DATA, DAYS, TELEMETRY_RE, out_dir  # noqa: E402

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, 'analysis', 'matched', 'pcap_content_thu.json')
OUT = out_dir('thu')
NOISE, CT = DAYS['thu']['noise'], DAYS['thu']['controls']
d = json.load(open(path))
dns = pd.DataFrame([r for v in d.values() for r in v['dns']], columns=['src', 'qname', 'n'])
tls = pd.DataFrame([r for v in d.values() for r in v['tls']], columns=['src', 'dst', 'sni', 'n'])
for t in (dns, tls):
    t['grp'] = np.where(t.src.isin(NOISE), 'noise', np.where(t.src.isin(CT), 'ctrl', 'other'))
dns, tls = dns[dns.grp != 'other'], tls[tls.grp != 'other']
dns['domain'] = dns.qname.str.split('.').str[-2:].str.join('.')

g = dns.groupby(['grp', 'domain']).n.sum().unstack(0).fillna(0)
g = g / g.sum()
n_dom, c_dom = set(dns[dns.grp == 'noise'].domain), set(dns[dns.grp == 'ctrl'].domain)
nq, cq = dns[dns.grp == 'noise'], dns[dns.grp == 'ctrl']
S = dict(queries_per_host=dns.groupby('src').n.sum().to_dict(),
         noise_queries_with_domain_seen_in_controls=round(float(nq[nq.domain.isin(c_dom)].n.sum() / nq.n.sum()), 3),
         control_queries_with_domain_seen_in_noise=round(float(cq[cq.domain.isin(n_dom)].n.sum() / cq.n.sum()), 3),
         cosine_noise_vs_controls=round(float(g.noise @ g.ctrl / np.linalg.norm(g.noise) / np.linalg.norm(g.ctrl)), 3))

h = dns.groupby(['src', 'domain']).n.sum().unstack(0).fillna(0)
h = h / h.sum()
hn = h / np.linalg.norm(h, axis=0)
M = pd.DataFrame(hn.T.values @ hn.values, index=h.columns, columns=h.columns)
nz = sorted(NOISE & set(M.index), key=lambda x: int(x.split('.')[-1]))
ct = sorted(CT & set(M.index))
M = M.loc[nz + ct, nz + ct]
M.round(3).to_csv(os.path.join(OUT, 'dns_domain_host_cosine.csv'))


def upper(A):
    return A.values[np.triu_indices(len(A), 1)]


S['pairwise_cosine'] = dict(noise_noise=round(float(upper(M.loc[nz, nz]).mean()), 3),
                            ctrl_ctrl=round(float(upper(M.loc[ct, ct]).mean()), 3),
                            noise_ctrl=round(float(M.loc[nz, ct].values.mean()), 3))
ms = tls[tls.dst.str.match(TELEMETRY_RE)]
S['telemetry_sni'] = ms.groupby(['grp', 'sni']).n.sum().unstack(0).fillna(0).astype(int) \
    .sort_values('noise', ascending=False).head(10).to_dict('index')
json.dump(S, open(os.path.join(OUT, 'pcap_content_compare.json'), 'w'), indent=2)
print(json.dumps(S, indent=2))
