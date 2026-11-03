"""Compare NF-v3 flow statistics of the same traffic type between noise-labeled hosts, same-image controls, and all other hosts (Thursday)."""
import numpy as np, pandas as pd
df = pd.read_csv('/home/tati/Downloads/cicids2018v3_thu0103.csv')
KEY = ['FLOW_START_MILLISECONDS', 'IPV4_SRC_ADDR', 'L4_SRC_PORT', 'IPV4_DST_ADDR', 'L4_DST_PORT', 'PROTOCOL']
df = df.drop_duplicates(KEY)
NOISE = {f'172.31.69.{i}' for i in (5, 8, 9, 10, 12, 14, 16, 17)}
CTRL = {'172.31.65.93', '172.31.66.97', '172.31.66.78', '172.31.69.23'}
src, dst = df.IPV4_SRC_ADDR, df.IPV4_DST_ADDR
internal = src.str.startswith('172.31.') & (src != '172.31.69.13')
df['grp'] = np.where(src.isin(NOISE) | dst.isin(NOISE), 'noise hosts', np.where(src.isin(CTRL) | dst.isin(CTRL), 'same-image controls', 'other hosts'))
types = {
    'DNS to 172.31.0.2': internal & (dst == '172.31.0.2') & (df.L4_DST_PORT == 53),
    'Microsoft telemetry 443': internal & dst.str.match(r'^(65\.52\.108\.|52\.165\.231\.|13\.89\.190\.)') & (df.L4_DST_PORT == 443),
    'RDP scanners 5.101.40.x': src.isin(['5.101.40.105', '5.101.40.43']) & (dst != '172.31.69.13'),
}
cols = ['IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS', 'FLOW_DURATION_MILLISECONDS', 'LONGEST_FLOW_PKT', 'MIN_TTL', 'TCP_WIN_MAX_IN']
out = []
for t, m in types.items():
    for g, d in df[m].groupby('grp'):
        row = dict(type=t, group=g, flows=len(d), hosts=d.IPV4_SRC_ADDR.nunique() if 'scanner' not in t else d.IPV4_DST_ADDR.nunique(), label_rate=round(d.Label.mean(), 3))
        for c in cols:
            row[c] = f"{d[c].median():g} [{d[c].quantile(.25):g}-{d[c].quantile(.75):g}]"
        if t.startswith('DNS'):
            row['qtype A/AAAA'] = f"{(d.DNS_QUERY_TYPE == 1).mean():.2f}/{(d.DNS_QUERY_TYPE == 28).mean():.2f}"
        if t.startswith('Microsoft'):
            gaps = d.sort_values('FLOW_START_MILLISECONDS').groupby(['IPV4_SRC_ADDR', 'IPV4_DST_ADDR']).FLOW_START_MILLISECONDS.diff().dropna() / 1000
            row['median gap s'] = round(gaps.median(), 1)
        out.append(row)
res = pd.DataFrame(out)
res.to_csv('/home/tati/data/cic-ids2018/analysis/matched/flow_stats_by_group_thu.csv', index=False)
pd.set_option('display.width', 250)
for t in types:
    print('\n==', t); print(res[res.type == t].drop(columns='type').set_index('group').T.to_string())
