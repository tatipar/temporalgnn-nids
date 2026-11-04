"""Flow-level baselines (option 1): how much of each label set can single-flow features explain?

Uses the deduplicated CSVs from prepare_corrected_inputs.py and reproduces the graph builder's
edge features (20 log1p + train-fitted StandardScaler numeric features, 7-way port role and
5-way protocol one-hot = 32 dimensions), its 0.0.0.0 filter and its window-start split rule with the
episode-aware cutoffs. Models: decision trees (depth 1, 2, 4, 8) and logistic regression, trained on
Wednesday train only and scored on validation, Test 1 and Thursday Test 2. No graph, no history.

Usage: python3 -I flow_baselines.py WED.csv THU.csv OUTPUT_DIR [episode|historical]

`episode` (default) uses the 15:44/16:56 UTC cutoffs; `historical` reproduces the fair-comparison
70%/85%-of-day cutoffs, under which Test 1 has no corrected positives (its AP is reported as NaN).
Inputs may be the deduplicated CSVs or the raw NF-v3 CSVs (targets are then added on the fly).
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

# Identical to cic_ids2018_v3_graph_creation.ipynb.
NUMERIC_FEATURES = [
    'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
    'FLOW_DURATION_MILLISECONDS', 'DURATION_IN', 'DURATION_OUT',
    'SRC_TO_DST_IAT_AVG', 'DST_TO_SRC_IAT_AVG',
    'SRC_TO_DST_IAT_STDDEV', 'DST_TO_SRC_IAT_STDDEV',
    'MIN_IP_PKT_LEN', 'MAX_IP_PKT_LEN',
    'RETRANSMITTED_IN_PKTS', 'RETRANSMITTED_OUT_PKTS',
    'TCP_WIN_MAX_IN', 'TCP_WIN_MAX_OUT',
    'TCP_FLAGS', 'MIN_TTL', 'MAX_TTL',
]
PORT_ROLES = ['port_web', 'port_admin', 'port_windows', 'port_infra', 'port_db', 'port_priv', 'port_high']
PROTOCOLS = ['proto_tcp', 'proto_udp', 'proto_icmp', 'proto_igmp', 'proto_other']
FEATURES = PORT_ROLES + PROTOCOLS + NUMERIC_FEATURES
WINDOW = pd.Timedelta(seconds=30)
SPLIT = sys.argv[4] if len(sys.argv) > 4 else 'episode'
assert SPLIT in ('episode', 'historical'), SPLIT
TARGETS = ['target_corrected', 'target_original']


def encode(df):
    port, proto = df.L4_DST_PORT.to_numpy(), df.PROTOCOL.to_numpy()
    roles = np.select(
        [np.isin(port, [80, 443, 8080, 8443, 81, 3128, 8545]),
         np.isin(port, [22, 222, 2222, 23, 2323, 3389, 3390, 3394, 5900, 5901, 5555, 21, 2131]),
         np.isin(port, [445, 135, 137, 138, 139]),
         np.isin(port, [53, 5355, 67, 547, 123, 1900, 5060]),
         np.isin(port, [1433, 3306, 5432, 6379, 27017]),
         port < 1024], range(6), 6)
    protos = np.select([proto == 6, proto == 17, np.isin(proto, [1, 58]), proto == 2], range(4), 4)
    out = pd.DataFrame(np.eye(7, dtype=np.float32)[roles], columns=PORT_ROLES, index=df.index)
    out[PROTOCOLS] = np.eye(5, dtype=np.float32)[protos]
    out[NUMERIC_FEATURES] = np.log1p(df[NUMERIC_FEATURES].to_numpy(dtype=np.float64))
    return out


def load(path):
    df = pd.read_csv(path)
    if 'target_corrected' not in df:              # raw NF-v3 CSV: add both targets, keep duplicates
        from utils.nfv3_relabel import relabel_chunk
        df['target_original'] = df['Attack'].eq('Infilteration').astype('int8')
        df['target_corrected'] = relabel_chunk(df)['binary_target'].astype('int8')
    df['FLOW_START_TIME'] = pd.to_datetime(df['FLOW_START_TIME'])
    start, end = df['FLOW_START_TIME'].min(), df['FLOW_START_TIME'].max()  # all rows, before the IP filter
    df = df[(df.IPV4_SRC_ADDR != '0.0.0.0') & (df.IPV4_DST_ADDR != '0.0.0.0')]
    window_start = start + ((df['FLOW_START_TIME'] - start) // WINDOW) * WINDOW
    return df, window_start, start, end


wed, wed_window, wed_start, wed_end = load(sys.argv[1])
thu, *_ = load(sys.argv[2])
if SPLIT == 'episode':      # episode-aware cutoffs (see docs/fair-comparison-corrected.md)
    LIMIT_TRAIN, LIMIT_VAL = pd.Timestamp('2018-02-28 15:44:00'), pd.Timestamp('2018-02-28 16:56:00')
else:                       # historical fair-comparison rule, as in cic_ids2018_v3_graph_creation.ipynb
    LIMIT_TRAIN = wed_start + (wed_end - wed_start) * 0.70
    LIMIT_VAL = wed_start + (wed_end - wed_start) * 0.85
out_dir = sys.argv[3]
os.makedirs(out_dir, exist_ok=True)
split = np.where(wed_window < LIMIT_TRAIN, 'train', np.where(wed_window < LIMIT_VAL, 'val', 'test1'))
X = {'wed': encode(wed), 'thu': encode(thu)}
scaler = StandardScaler().fit(X['wed'].loc[split == 'train', NUMERIC_FEATURES])
for k in X:
    X[k][NUMERIC_FEATURES] = scaler.transform(X[k][NUMERIC_FEATURES])
parts = {s: (X['wed'][split == s], wed.loc[split == s]) for s in ('train', 'val', 'test1')}
parts['test2'] = (X['thu'], thu)

models = {f'decision_tree_depth_{d}': (lambda d=d: DecisionTreeClassifier(max_depth=d, random_state=0))
          for d in (1, 2, 4, 8)}
models['logistic_regression'] = lambda: LogisticRegression(max_iter=2000)
rows, rules = [], {}
for target in TARGETS:
    y_train = parts['train'][1][target].to_numpy()
    for name, make in models.items():
        model = make().fit(parts['train'][0][FEATURES], y_train)
        for s in ('val', 'test1', 'test2'):
            xs, frame = parts[s]
            y = frame[target].to_numpy()
            p = model.predict_proba(xs[FEATURES])[:, 1]
            ap = round(float(average_precision_score(y, p)), 4) if y.any() else float('nan')
            rows.append(dict(target=target, model=name, split=s, flows=len(y), positives=int(y.sum()),
                             prevalence=round(float(y.mean()), 5), average_precision=ap))
        if name in ('decision_tree_depth_1', 'decision_tree_depth_2'):
            rules[f'{target}/{name}'] = export_text(model, feature_names=FEATURES)
result = pd.DataFrame(rows)
result.to_csv(os.path.join(out_dir, 'flow_baselines.csv'), index=False)
summary = dict(split=SPLIT, cutoffs=[str(LIMIT_TRAIN), str(LIMIT_VAL)], features=FEATURES,
               split_counts={s: dict(flows=len(f), **{t: int(f[t].sum()) for t in TARGETS})
                             for s, (_, f) in parts.items()},
               tree_rules=rules)
json.dump(summary, open(os.path.join(out_dir, 'flow_baselines_summary.json'), 'w'), indent=2)
print(json.dumps(summary['split_counts'], indent=2))
print(result.pivot_table(index=['target', 'model'], columns='split', values='average_precision')
      .reindex(columns=['val', 'test1', 'test2']).to_string())
print('\nprevalence:', result.drop_duplicates(['target', 'split']).set_index(['target', 'split']).prevalence.to_dict())
for k, v in rules.items():
    print(f'\n== {k}\n{v}')
