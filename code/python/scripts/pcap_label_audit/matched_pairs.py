"""Matched-behaviour test: do models score identical traffic differently depending on the host?

Compares alert rates (frozen validation thresholds) and median scores for the same behaviour
coming from the nine noise-labeled .69 hosts versus all other hosts, on Test 2 (Thursday).
"""
import glob, json, sys
import numpy as np, pandas as pd

BASE = '/home/tati/data/nids-mitre-results/extracted'
OUT = '/home/tati/data/cic-ids2018/analysis/matched'
run = glob.glob(f'{BASE}/paper_eight_models_ip_label_audit_v1/run_*')[0]
man = json.load(open(f'{run}/analysis_manifest.json'))
paths = {
    'MLP': 'results_baseline_retraining_factorial_protocol_v1/mlp/seed_777',
    'Global traffic MLP-GRU': 'results_temporal_mlp_global_host_factorial_protocol_v1/global_traffic_mlp_gru/seed_42',
    'Host history MLP-GRU': 'results_temporal_mlp_global_host_factorial_protocol_v1/host_history_mlp_gru/seed_2024',
    'E-GraphSAGE': 'results_baseline_retraining_factorial_protocol_v1/e_graphsage/seed_2024',
    'M0_P0': 'results_stgnn_memory_message_factorial_v1/M0_P0/seed_2024',
    'M0_P1': 'results_stgnn_memory_message_factorial_v1/M0_P1/seed_123',
    'M1_P0': 'results_stgnn_memory_message_factorial_v1/M1_P0/seed_2024',
    'M1_P1': 'results_stgnn_memory_message_factorial_v1/M1_P1/seed_2024',
}
f = pd.read_parquet(f'{run}/flow_attribution_test2.parquet',
                    columns=['flow_index', 'flow_start', 'src_ip', 'dst_ip', 'dst_port', 'protocol', 'native_label'])
f = f.sort_values('flow_index').reset_index(drop=True)
for name, p in paths.items():
    z = np.load(f'{BASE}/{p}/test2_scores.npz')
    assert np.array_equal(z['flow_index'], f.flow_index.values), name
    f[name] = z['probabilities']
src, dst = f.src_ip.astype(str), f.dst_ip.astype(str)
NOISE = {f'172.31.69.{i}' for i in (5, 8, 9, 10, 12, 14, 16, 17)}
internal = src.str.startswith('172.31.')
ms = dst.str.match(r'^(65\.52\.108\.|52\.165\.231\.|13\.89\.190\.)') & (f.dst_port == 443)
scanners = src.isin(['5.101.40.105', '5.101.40.43'])
pairs = {
    'Microsoft telemetry 443 (src)': (ms & src.isin(NOISE), ms & internal & ~src.isin(NOISE | {'172.31.69.13'})),
    'DNS to 172.31.0.2 (src)': ((dst == '172.31.0.2') & src.isin(NOISE), (dst == '172.31.0.2') & internal & ~src.isin(NOISE | {'172.31.69.13'})),
    'RDP scanners 5.101.40.x (dst)': (scanners & dst.isin(NOISE), scanners & ~dst.isin(NOISE | {'172.31.69.13'})),
}
rows = []
for pair, (a, b) in pairs.items():
    for name in paths:
        t = man['frozen_thresholds'][name]
        for grp, m in (('noise hosts', a), ('other hosts', b)):
            s = f.loc[m, name]
            rows.append(dict(behaviour=pair, model=name, group=grp, flows=int(m.sum()),
                             native_label_rate=f.loc[m, 'native_label'].mean(),
                             alert_rate=(s >= t).mean(), median_score=s.median()))
res = pd.DataFrame(rows)
res.to_csv(f'{OUT}/matched_pairs_test2.csv', index=False)
wide = res.pivot_table(index=['behaviour', 'model'], columns='group', values='alert_rate').round(3)
print(res.groupby(['behaviour', 'group'])[['flows', 'native_label_rate']].first().round(3).to_string())
print(wide.to_string())
# timing: alert rate on noise hosts' own (outbound) flows before vs after first scan of each host
first_scan = f[src.eq('172.31.69.13') & dst.isin(NOISE)].groupby(dst)['flow_start'].min()
own = f[src.isin(NOISE)].copy()
own['after_scan'] = own.flow_start >= own.src_ip.astype(str).map(first_scan)
tim = []
for name in paths:
    t = man['frozen_thresholds'][name]
    g = own.groupby(['after_scan', 'native_label'])[name].apply(lambda s: (s >= t).mean()).rename('alert_rate').reset_index()
    g['model'] = name; tim.append(g)
tim = pd.concat(tim)
tim.to_csv(f'{OUT}/noise_hosts_before_after_scan_test2.csv', index=False)
print(tim.pivot_table(index='model', columns=['after_scan', 'native_label'], values='alert_rate').round(3).to_string())
print('flows before/after scan:', own.after_scan.value_counts().to_dict())
# control: other .69 hosts (never labeled) — outbound alert rates
ctrl = [f'172.31.69.{i}' for i in (6, 23, 26, 29, 30)] + ['172.31.65.93', '172.31.66.97', '172.31.66.78']
crow = []
for h in ctrl + sorted(NOISE):
    m = src.eq(h)
    if m.sum() == 0: continue
    d = dict(host=h, noise=h in NOISE, flows=int(m.sum()), native_label_rate=round(f.loc[m, 'native_label'].mean(), 3))
    for name in paths:
        d[name] = round((f.loc[m, name] >= man['frozen_thresholds'][name]).mean(), 3)
    crow.append(d)
ctab = pd.DataFrame(crow)
ctab.to_csv(f'{OUT}/same_subnet_controls_test2.csv', index=False)
print(ctab[['host', 'noise', 'flows', 'native_label_rate', 'M1_P0', 'M1_P1', 'Host history MLP-GRU', 'M0_P0', 'E-GraphSAGE']].to_string())
# same three behaviours, split three ways: noise hosts / same-image controls / all other hosts
CT = {'172.31.65.93', '172.31.66.97', '172.31.66.78', '172.31.69.23'}
def grp3(ip):
    return np.where(ip.isin(NOISE), 'noise', np.where(ip.isin(CT), 'same-image ctrl', 'other'))
r3 = []
for pair, sel, side in (('DNS to 172.31.0.2', (dst == '172.31.0.2') & internal & (src != '172.31.69.13'), src),
                        ('Microsoft telemetry 443', ms & internal & (src != '172.31.69.13'), src),
                        ('RDP scanners 5.101.40.x', scanners & (dst != '172.31.69.13'), dst)):
    g = pd.Series(grp3(side[sel]), index=side[sel].index)
    for name in paths:
        a = (f.loc[sel, name] >= man['frozen_thresholds'][name])
        for k, v in a.groupby(g).mean().items():
            r3.append(dict(behaviour=pair, model=name, group=k, alert_rate=round(v, 3), flows=int((g == k).sum())))
r3 = pd.DataFrame(r3)
r3.to_csv(f'{OUT}/matched_pairs_3groups_test2.csv', index=False)
print(r3.pivot_table(index=['behaviour', 'model'], columns='group', values='alert_rate').reindex(columns=['noise', 'same-image ctrl', 'other']).to_string())
print(r3.drop_duplicates(['behaviour', 'group'])[['behaviour', 'group', 'flows']].to_string())
