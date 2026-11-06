"""Matched-behaviour test on the deduplicated experiment (original labels, duplicates removed).

Same comparisons as ``matched_pairs.py`` for the three models of experiment 3 (M0_P0, M1_P0, MLP).
Flow attribution is rebuilt from the NF-v3 Thursday CSV with the graph builder's rules
(duplicate removal, 0.0.0.0 filter, 30-second windows, file order within a window) and checked
edge by edge against ``alignment/test2_flows.parquet`` before any score is used.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from utils.nfv3_dedup import DEDUP_KEY  # noqa: E402

CSV = Path.home() / 'Downloads/cicids2018v3_thu0103.csv'
BASE = Path.home() / 'data/nids-mitre-results/extracted'
OUT = Path.home() / 'data/cic-ids2018/analysis/matched'
FACT = BASE / 'results_stgnn_factorial_dedup_historical_split_original_v1'
MODELS = {
    'M0_P0': FACT / 'M0_P0/seed_99',
    'M1_P0': FACT / 'M1_P0/seed_123',
    'MLP': BASE / 'results_baselines_dedup_historical_split_original_v1/mlp/seed_2024',
}
WINDOW = pd.Timedelta(seconds=30)

# --- rebuild attribution -------------------------------------------------------------------
cols = DEDUP_KEY + ['FLOW_START_TIME', 'Attack']
raw = pd.read_csv(CSV, usecols=cols)
f = raw.loc[~raw.duplicated(DEDUP_KEY, keep='first')].copy()
f['FLOW_START_TIME'] = pd.to_datetime(f['FLOW_START_TIME'])
start = f['FLOW_START_TIME'].min()           # builder: minimum over the deduplicated file
f = f[(f.IPV4_SRC_ADDR != '0.0.0.0') & (f.IPV4_DST_ADDR != '0.0.0.0')]
f['window_id'] = ((f.FLOW_START_TIME - start) // WINDOW).astype(int)
f = f.sort_values('window_id', kind='stable').reset_index(drop=True)
ip_map = json.load(open(BASE / 'dataset_processed_thu0103_dedup_historical_original/ip_map.json'))
align = pd.read_parquet(FACT / 'alignment/test2_flows.parquet').sort_values('flow_index').reset_index(drop=True)
assert len(align) == len(f), (len(align), len(f))
assert np.array_equal(align.flow_index.to_numpy(), np.arange(len(f)))
assert np.array_equal(align.window_id.to_numpy(), f.window_id.to_numpy()), 'window order differs'
assert np.array_equal(align.source_gid.to_numpy(), f.IPV4_SRC_ADDR.map(ip_map).to_numpy()), 'source differs'
assert np.array_equal(align.destination_gid.to_numpy(), f.IPV4_DST_ADDR.map(ip_map).to_numpy()), 'destination differs'
assert np.array_equal(align.label.to_numpy(), f.Attack.eq('Infilteration').astype(np.int8).to_numpy()), 'label differs'
print(f'attribution verified: {len(f):,} flows, {int(align.label.sum()):,} positives')

f = f.rename(columns={'IPV4_SRC_ADDR': 'src_ip', 'IPV4_DST_ADDR': 'dst_ip', 'L4_DST_PORT': 'dst_port',
                      'FLOW_START_TIME': 'flow_start'})
f['native_label'] = align.label.to_numpy()
thresholds = {}
for name, folder in MODELS.items():
    z = np.load(folder / 'test2_scores.npz')
    assert np.array_equal(z['flow_index'], align.flow_index.to_numpy()), name
    f[name] = z['probabilities']
    thresholds[name] = json.load(open(folder / 'record.json'))['threshold']
print('validation thresholds:', {k: round(v, 4) for k, v in thresholds.items()})

# --- matched behaviours (same definitions as matched_pairs.py) -------------------------------
src, dst = f.src_ip.astype(str), f.dst_ip.astype(str)
NOISE = {f'172.31.69.{i}' for i in (5, 8, 9, 10, 12, 14, 16, 17)}
CT = {'172.31.65.93', '172.31.66.97', '172.31.66.78', '172.31.69.23'}
internal = src.str.startswith('172.31.')
ms = dst.str.match(r'^(65\.52\.108\.|52\.165\.231\.|13\.89\.190\.)') & (f.dst_port == 443)
scanners = src.isin(['5.101.40.105', '5.101.40.43'])


def grp3(ip):
    return np.where(ip.isin(NOISE), 'noise', np.where(ip.isin(CT), 'same-image ctrl', 'other'))


rows = []
for pair, sel, side in (('DNS to 172.31.0.2', (dst == '172.31.0.2') & internal & (src != '172.31.69.13'), src),
                        ('Microsoft telemetry 443', ms & internal & (src != '172.31.69.13'), src),
                        ('RDP scanners 5.101.40.x', scanners & (dst != '172.31.69.13'), dst)):
    g = pd.Series(grp3(side[sel]), index=side[sel].index)
    label_rate = f.loc[sel, 'native_label'].groupby(g).mean()
    for name in MODELS:
        alerts = f.loc[sel, name] >= thresholds[name]
        for k, v in alerts.groupby(g).mean().items():
            rows.append(dict(behaviour=pair, model=name, group=k, alert_rate=round(v, 3),
                             flows=int((g == k).sum()), native_label_rate=round(label_rate[k], 3)))
res = pd.DataFrame(rows)
OUT.mkdir(parents=True, exist_ok=True)
res.to_csv(OUT / 'matched_pairs_3groups_test2_dedup.csv', index=False)
order = ['noise', 'same-image ctrl', 'other']
print('\nlabel rate and flows per group')
print(res.drop_duplicates(['behaviour', 'group']).pivot_table(
    index='behaviour', columns='group', values=['native_label_rate', 'flows']).reindex(columns=order, level=1).to_string())
print('\nalert rate, deduplicated experiment')
print(res.pivot_table(index=['behaviour', 'model'], columns='group', values='alert_rate').reindex(columns=order).to_string())
raw_res = pd.read_csv(OUT / 'matched_pairs_3groups_test2.csv')
raw_res = raw_res[raw_res.model.isin(MODELS)]
print('\nalert rate, data as distributed (matched_pairs.py)')
print(raw_res.pivot_table(index=['behaviour', 'model'], columns='group', values='alert_rate').reindex(columns=order).to_string())

# --- noise hosts' own flows before / after the victim first scans them -----------------------
first_scan = f[src.eq('172.31.69.13') & dst.isin(NOISE)].groupby(dst)['flow_start'].min()
own = f[src.isin(NOISE)].copy()
own['after_scan'] = own.flow_start >= own.src_ip.astype(str).map(first_scan)
tim = []
for name in MODELS:
    g = own.groupby(['after_scan', 'native_label'])[name].apply(lambda s: (s >= thresholds[name]).mean())
    tim.append(g.rename('alert_rate').reset_index().assign(model=name))
tim = pd.concat(tim)
tim.to_csv(OUT / 'noise_hosts_before_after_scan_test2_dedup.csv', index=False)
print('\nnoise hosts, own flows: alert rate by (after first scan, label)')
print(tim.pivot_table(index='model', columns=['after_scan', 'native_label'], values='alert_rate').round(3).to_string())
print('flows before/after scan:', own.after_scan.value_counts().to_dict())
