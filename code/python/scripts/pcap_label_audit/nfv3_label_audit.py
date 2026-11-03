"""NF-v3 label audit for one day (wed|thu): original labels vs Liu et al. rules, where the labels sit
(hosts, time), label rates for matched behaviours, internet tail sources and duplicate rows.

Usage: python3 -I nfv3_label_audit.py wed|thu
Writes CSVs and summary.json to <CICIDS_DATA>/analysis/label_audit/<day>/.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_config import DAYS, KEY, TELEMETRY_RE, RDP_SCANNERS, out_dir  # noqa: E402
from load_nfv3 import load  # noqa: E402

day = sys.argv[1]
C = DAYS[day]
OUT = out_dir(day)
V, N, CT = C['victim'], C['noise'], C['controls']
df = load(C['csv'])
src, dst = df.IPV4_SRC_ADDR.astype(str), df.IPV4_DST_ADDR.astype(str)
internal_src = src.str.startswith('172.31.')
S = {}


def r3(x):
    return None if pd.isna(x) else round(float(x), 3)


# 1. Original labels vs the corrected (Liu et al.) rules, on all rows and on unique flows.
for tag, d in (('rows', df), ('dedup', df.drop_duplicates(KEY))):
    v = d.IPV4_SRC_ADDR == V
    lab, conf = d.Label == 1, d.binary_target == 1
    S[f'labels_{tag}'] = dict(
        total=len(d), labeled=int(lab.sum()),
        victim_labeled_confirmed=int((lab & conf & v).sum()),
        victim_labeled_unconfirmed=int((lab & ~conf & v).sum()),
        other_sources_labeled=int((lab & ~v).sum()),
        labeled_not_confirmed=int((lab & ~conf).sum()),
        labeled_not_confirmed_share=r3((lab & ~conf).sum() / lab.sum()),
        confirmed_labeled_benign=int((~lab & conf).sum()),
        corrected_positives=int(conf.sum()),
        corrected_positive_sources=sorted(d[conf].IPV4_SRC_ADDR.unique().tolist()),
        labeled_sources=int(d[lab].IPV4_SRC_ADDR.nunique()),
    )
S['corrected_positives_by_rule'] = df[df.binary_target == 1].correction_rule.value_counts().to_dict()

# 2. Per labeled source.
a = df[df.Label == 1]
tot = df.groupby('IPV4_SRC_ADDR').size()
per_src = a.groupby('IPV4_SRC_ADDR').agg(
    labeled_rows=('Label', 'size'), first=('ts', 'min'), last=('ts', 'max'),
    confirmed=('binary_target', 'sum'),
    top_dst=('IPV4_DST_ADDR', lambda x: x.value_counts().index[0]),
    top_dst_port=('L4_DST_PORT', lambda x: x.value_counts().index[0]))
per_src['labeled_dedup'] = a.drop_duplicates(KEY).groupby('IPV4_SRC_ADDR').size()
per_src['share_of_source_rows'] = (per_src.labeled_rows / tot[per_src.index]).round(3)
per_src['internal'] = per_src.index.str.startswith('172.31.')
per_src['role'] = np.where(per_src.index == V, 'victim', np.where(per_src.index.isin(N), 'noise host', ''))
per_src.sort_values('labeled_rows', ascending=False).to_csv(f'{OUT}/labeled_sources.csv')

# 3. Do all labels touch the victim or the noise hosts?
touch = src.isin(N | {V}) | dst.isin(N | {V})
rest = df[(df.Label == 1) & ~touch]
S['labeled_rows_not_touching_victim_or_noise_hosts'] = len(rest)
S['labeled_rows_not_touching_examples'] = {
    f'{k[0]}->{k[1]}:{k[2]}': int(n) for k, n in
    rest.groupby(['IPV4_SRC_ADDR', 'IPV4_DST_ADDR', 'L4_DST_PORT']).size().sort_values(ascending=False).head(10).items()}
ends = pd.concat([rest.IPV4_SRC_ADDR, rest.IPV4_DST_ADDR])
S['labeled_rows_not_touching_internal_endpoints'] = ends[ends.str.startswith('172.31.')].value_counts().head(10).to_dict()

# 4. Timing: labels on the noise hosts vs the documented attack span.
lo, hi = (pd.to_datetime(t, unit='s') for t in C['attack_span'])
S['documented_attack_span_utc'] = [str(lo), str(hi)]
S['capture_span_nfv3_utc'] = [str(df.ts.min()), str(df.ts.max())]
host_rows = []
for h in sorted(N, key=lambda x: int(x.split('.')[-1])):
    d = df[src.eq(h) | dst.eq(h)].drop_duplicates(KEY)
    lab = d[d.Label == 1]
    inside = lab.ts.between(lo, hi)
    host_rows.append(dict(host=h, flows_dedup=len(d), labeled_dedup=len(lab), label_rate=r3(d.Label.mean()),
                          first_labeled=lab.ts.min(), last_labeled=lab.ts.max(),
                          labeled_outside_attack_span_share=r3((~inside).mean()) if len(lab) else None))
pd.DataFrame(host_rows).to_csv(f'{OUT}/noise_hosts_label_timing.csv', index=False)
h = df[src.isin(N)].drop_duplicates(KEY)
h.groupby(h.ts.dt.hour).Label.agg(flows='size', label_rate='mean').round(3).to_csv(f'{OUT}/noise_hosts_label_rate_by_hour.csv')
nc = df[(df.Label == 1) & (df.binary_target == 0)]
S['unconfirmed_labels_outside_attack_span_share'] = r3((~nc.ts.between(lo, hi)).mean())

# 5. Long tail of labeled sources (< 10 labeled rows).
cnt = a.groupby('IPV4_SRC_ADDR').size()
tail = cnt[cnt < 10].index
t = a[a.IPV4_SRC_ADDR.isin(tail)]
S['tail_sources'] = dict(sources=len(tail), rows=len(t),
                         external_sources=int((~pd.Series(tail).str.startswith('172.31.')).sum()),
                         dst_in_noise_or_victim_share=r3(t.IPV4_DST_ADDR.isin(N | {V}).mean()),
                         top_dst_ports=t.L4_DST_PORT.value_counts().head(10).to_dict())

# 6. Same behaviour, different hosts: label rates (unique flows).
u = df.drop_duplicates(KEY)
us, ud = u.IPV4_SRC_ADDR.astype(str), u.IPV4_DST_ADDR.astype(str)
match = []
tel = ud.str.match(TELEMETRY_RE) & (u.L4_DST_PORT == 443) & us.str.startswith('172.31.') & (us != V)
for g, m in (('noise hosts', tel & us.isin(N)), ('other internal hosts', tel & ~us.isin(N))):
    match.append(dict(behaviour='Microsoft telemetry 443 (host as source)', group=g, flows=int(m.sum()),
                      hosts=int(us[m].nunique()), label_rate=r3(u.Label[m].mean())))
for s in RDP_SCANNERS:
    sm = (us == s) & (ud != V)
    for g, m in (('noise hosts', sm & ud.isin(N)), ('other hosts', sm & ~ud.isin(N))):
        match.append(dict(behaviour=f'RDP scanner {s} (host as target)', group=g, flows=int(m.sum()),
                          hosts=int(ud[m].nunique()), label_rate=r3(u.Label[m].mean())))
ext = ~us.str.startswith('172.31.')
for g, m in (('noise hosts or victim', ext & ud.isin(N | {V})), ('other hosts', ext & ~ud.isin(N | {V}))):
    match.append(dict(behaviour='any external source (host as target)', group=g, flows=int(m.sum()),
                      hosts=int(ud[m].nunique()), label_rate=r3(u.Label[m].mean())))
pd.DataFrame(match).to_csv(f'{OUT}/matched_behaviour_label_rates.csv', index=False)

# 7. Duplicate rows (same 5-tuple + start time) by host group.
grp = np.where(src.isin(N) | dst.isin(N), 'noise hosts',
               np.where(src.isin(CT) | dst.isin(CT), 'controls',
                        np.where(src.eq(V) | dst.eq(V), 'victim', 'other')))
dup = df.ncopy > 1
dg = pd.DataFrame(dict(g=grp, dup=dup, Label=df.Label)).groupby('g').agg(
    rows=('dup', 'size'), duplicated_share=('dup', 'mean'), label_rate=('Label', 'mean')).round(3)
dg.to_csv(f'{OUT}/duplicates_by_group.csv')
nz = grp == 'noise hosts'
S['duplicates'] = dict(rows=len(df), unique_flows=int(len(u)),
                       labeled_rows_duplicated_share=r3(dup[df.Label == 1].mean()),
                       noise_hosts_label_rate_duplicated=r3(df.Label[nz & dup].mean()),
                       noise_hosts_label_rate_single=r3(df.Label[nz & ~dup].mean()),
                       copies_per_group_labeled=df[df.Label == 1].ncopy.value_counts().sort_index().to_dict())

# 8. Control hosts: labeled outside the victim's (documented) traffic to them?
nv = ~(src.eq(V) | dst.eq(V))
S['controls'] = {c: dict(rows=int((src.eq(c) | dst.eq(c)).sum()),
                         labeled_with_victim=int(df.Label[(src.eq(c) | dst.eq(c)) & ~nv].sum()),
                         labeled_without_victim=int(df.Label[(src.eq(c) | dst.eq(c)) & nv].sum()))
                 for c in sorted(CT)}

json.dump(S, open(f'{OUT}/summary.json', 'w'), indent=2, default=str)
print(json.dumps(S, indent=2, default=str))
print('\n', pd.DataFrame(host_rows).to_string(index=False))
print('\n', pd.DataFrame(match).to_string(index=False))
print('\n', dg.to_string())
print('\noutputs in', OUT)
