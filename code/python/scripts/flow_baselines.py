"""Flow-level baselines (option 1): how much of each label set can single-flow features explain?

Uses the deduplicated CSVs from prepare_corrected_inputs.py and reproduces the graph builder's
edge features (20 log1p + train-fitted StandardScaler numeric features, 7-way port role and
5-way protocol one-hot = 32 dimensions), its 0.0.0.0 filter and its window-start split rule with the
episode-aware cutoffs. Models: decision trees (depth 1, 2, 4, 8) and logistic regression, trained on
Wednesday train only and scored on validation, Test 1 and Thursday Test 2. No graph, no history.

Usage: python3 -I flow_baselines.py WED_DEDUP.csv THU_DEDUP.csv OUTPUT_DIR
"""
import json
import os
import sys

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
LIMIT_TRAIN = pd.Timestamp('2018-02-28 15:44:00')   # episode-aware cutoffs (see docs)
LIMIT_VAL = pd.Timestamp('2018-02-28 16:56:00')
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
    df['FLOW_START_TIME'] = pd.to_datetime(df['FLOW_START_TIME'])
    start = df['FLOW_START_TIME'].min()           # the builder uses all rows, before the IP filter
    df = df[(df.IPV4_SRC_ADDR != '0.0.0.0') & (df.IPV4_DST_ADDR != '0.0.0.0')]
    window_start = start + ((df['FLOW_START_TIME'] - start) // WINDOW) * WINDOW
    return df, window_start


wed, wed_window = load(sys.argv[1])
thu, _ = load(sys.argv[2])
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
            rows.append(dict(target=target, model=name, split=s, flows=len(y), positives=int(y.sum()),
                             prevalence=round(float(y.mean()), 5),
                             average_precision=round(float(average_precision_score(y, p)), 4)))
        if name in ('decision_tree_depth_1', 'decision_tree_depth_2'):
            rules[f'{target}/{name}'] = export_text(model, feature_names=FEATURES)
result = pd.DataFrame(rows)
result.to_csv(os.path.join(out_dir, 'flow_baselines.csv'), index=False)
summary = dict(cutoffs=[str(LIMIT_TRAIN), str(LIMIT_VAL)], features=FEATURES,
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
