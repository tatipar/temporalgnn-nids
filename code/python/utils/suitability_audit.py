"""Training-free suitability audit of multi-stage intrusion datasets.

The same checks are applied to every dataset, so the results can be compared in one table.
Each dataset loader returns a canonical flow table with these columns:

- ``campaign``: unit of independence (capture day, run or the whole capture);
- ``time``: flow start (datetime64);
- ``src``, ``dst``, ``sport``, ``dport``, ``proto``: endpoints;
- ``src_internal``, ``dst_internal``: whether each endpoint is inside the defended network;
- ``label`` (0/1), ``stage`` (``'benign'`` for negatives) and ``activity``;
- the dataset's native single-flow numeric features (names listed in ``features``).

The checks answer five questions:

1. Structure: how many campaigns, attack episodes, attacking hosts and attack host pairs are there?
2. Late stage: is lateral movement labeled, and between which internal hosts?
3. Identity shortcut: how well does a lookup on (source, destination) predict the label?
4. Single-flow separability: how well does a depth-1 or depth-4 tree on one flow's features do,
   overall and per stage?
5. Host-based labeling: are some hosts labeled through most of their active hours?

Nothing here trains a neural model. Ports are kept as features (they describe behaviour); IP
addresses, MAC addresses, identifiers and timestamps are never features.
"""
from __future__ import annotations

import glob
import io
import ipaddress
import json
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.tree import DecisionTreeClassifier

CANONICAL = ['campaign', 'time', 'src', 'dst', 'sport', 'dport', 'proto',
             'src_internal', 'dst_internal', 'label', 'stage', 'activity']
EPISODE_GAP = pd.Timedelta(hours=6)
MIN_STAGE_FLOWS = 20


@dataclass
class Flows:
    name: str
    unit: str
    frame: pd.DataFrame
    features: list[str]
    notes: dict = field(default_factory=dict)


# ----------------------------------------------------------------------------------------------
# Loaders
# ----------------------------------------------------------------------------------------------

def _numeric_features(frame: pd.DataFrame, exclude: set[str]) -> list[str]:
    feats = []
    for column in frame.columns:
        if column in exclude or column in CANONICAL:
            continue
        values = pd.to_numeric(frame[column], errors='coerce')
        if values.notna().mean() > 0.5:
            frame[column] = values.astype('float32')
            feats.append(column)
    return feats


def _slim(frame: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
    """Keep only the canonical columns and the features, to bound memory."""
    return frame[CANONICAL + feats].reset_index(drop=True)


def _is_private(ip: str, prefixes: tuple[str, ...]) -> bool:
    return any(str(ip).startswith(p) for p in prefixes)


NFV3_EXCLUDE = {'FLOW_START_MILLISECONDS', 'FLOW_END_MILLISECONDS', 'IPV4_SRC_ADDR', 'IPV4_DST_ADDR',
                'L4_SRC_PORT', 'DNS_QUERY_ID', 'Label', 'Attack', 'FLOW_START_TIME'}
NFV3_CORRECTED_STAGE = {'confirmed_dropbox_download': 'dropbox download',
                        'confirmed_victim_attacker_communication': 'victim-attacker communication',
                        'confirmed_nmap_portscan': 'internal nmap scan'}


def load_cic_nfv3(day_csvs: dict[str, str], label_set: str, nrows: int | None = None) -> Flows:
    """NF-CSE-CIC-IDS2018-v3 Infiltration days. ``label_set`` is 'original' or 'corrected'."""
    from utils.nfv3_relabel import relabel_chunk
    parts = []
    for day, path in day_csvs.items():
        frame = pd.read_csv(path, nrows=nrows, low_memory=False)
        frame = frame[(frame.IPV4_SRC_ADDR != '0.0.0.0') & (frame.IPV4_DST_ADDR != '0.0.0.0')]
        attack = frame['Attack'].astype(str).str.strip()
        if label_set == 'original':
            frame['label'] = attack.eq('Infilteration').astype('int8')
            frame['stage'] = np.where(frame.label == 1, 'infiltration (original label)', 'benign')
        else:
            rel = relabel_chunk(frame)
            frame['label'] = rel['binary_target'].astype('int8')
            frame['stage'] = rel['correction_rule'].map(NFV3_CORRECTED_STAGE).fillna('benign')
            frame.loc[frame.label == 0, 'stage'] = 'benign'
        frame['activity'] = frame['stage']
        frame['campaign'] = day
        parts.append(frame)
    frame = pd.concat(parts, ignore_index=True)
    frame['time'] = pd.to_datetime(frame.FLOW_START_MILLISECONDS, unit='ms')
    frame['src'], frame['dst'] = frame.IPV4_SRC_ADDR.astype(str), frame.IPV4_DST_ADDR.astype(str)
    frame['sport'], frame['dport'] = frame.L4_SRC_PORT, frame.L4_DST_PORT
    frame['proto'] = frame.PROTOCOL
    frame['src_internal'] = frame.src.str.startswith('172.31.')
    frame['dst_internal'] = frame.dst.str.startswith('172.31.')
    feats = _numeric_features(frame, NFV3_EXCLUDE)
    return Flows(f'CIC-IDS2018 NF-v3 Infiltration ({label_set} labels)', 'flow', _slim(frame, feats), feats,
                 {'internal': '172.31.0.0/16', 'campaigns': 'capture days'})


DAPT_FILES = [
    'enp0s3-monday-pvt.pcap_Flow.csv', 'enp0s3-monday.pcap_Flow.csv',
    'enp0s3-public-tuesday.pcap_Flow.csv', 'enp0s3-pvt-tuesday.pcap_Flow.csv',
    'enp0s3-public-wednesday.pcap_Flow.csv', 'enp0s3-pvt-wednesday.pcap_Flow.csv',
    'enp0s3-public-thursday.pcap_Flow.csv', 'enp0s3-pvt-thursday.pcap_Flow.csv',
    'enp0s3-tcpdump-friday.pcap_Flow.csv', 'enp0s3-tcpdump-pvt-friday.pcap_Flow.csv',
]
DAPT_URL = 'https://gitlab.com/asu22/dapt2020/-/raw/main/csv/'
DAPT_EXCLUDE = {'Flow ID', 'Src IP', 'Dst IP', 'Src Port', 'Timestamp', 'Activity', 'Stage'}
BENIGN_WORDS = {'benign', 'normal', 'none', 'background', '', 'nan', '-'}


def load_dapt2020(data_dir: str) -> Flows:
    parts, columns = [], None
    for name in DAPT_FILES:
        path = Path(data_dir) / name
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = [str(c).strip() for c in frame.columns]
        if 'Stage' not in frame.columns:  # the Thursday private file has no header
            frame = pd.read_csv(path, header=None, names=columns, low_memory=False)
        columns = list(frame.columns)
        frame['file'] = name
        parts.append(frame)
    frame = pd.concat(parts, ignore_index=True)
    frame['time'] = pd.to_datetime(frame['Timestamp'].astype(str).str.strip(), dayfirst=True,
                                   format='mixed', errors='coerce')
    frame = frame.dropna(subset=['time'])
    frame['src'], frame['dst'] = frame['Src IP'].astype(str).str.strip(), frame['Dst IP'].astype(str).str.strip()
    frame['sport'], frame['dport'], frame['proto'] = frame['Src Port'], frame['Dst Port'], frame['Protocol']
    stage = frame['Stage'].astype(str).str.strip().str.lower()
    frame['label'] = (~stage.isin(BENIGN_WORDS)).astype('int8')
    frame['stage'] = np.where(frame.label == 1, stage, 'benign')
    frame['activity'] = frame['Activity'].astype(str).str.strip().str.lower()
    frame['campaign'] = 'single campaign'
    frame['src_internal'] = frame.src.str.startswith('192.168.3.')
    frame['dst_internal'] = frame.dst.str.startswith('192.168.3.')
    feats = _numeric_features(frame, DAPT_EXCLUDE | {'file'})
    return Flows('DAPT2020', 'flow (CICFlowMeter biflow)', _slim(frame, feats), feats,
                 {'internal': '192.168.3.0/24', 'campaigns': 'one five-day campaign'})


UNRAVELED_EXCLUDE = {'id', 'expiration_id', 'src_ip', 'src_mac', 'src_oui', 'src_port', 'dst_ip', 'dst_mac',
                     'dst_oui', 'vlan_id', 'tunnel_id', 'Activity', 'Stage', 'DefenderResponse', 'Signature',
                     'bidirectional_first_seen_ms', 'bidirectional_last_seen_ms', 'src2dst_first_seen_ms',
                     'src2dst_last_seen_ms', 'dst2src_first_seen_ms', 'dst2src_last_seen_ms'}
MIME = re.compile(r'^[a-z]+/[a-z0-9.+-]+$')
WORDS = re.compile(r'^[a-z][a-z \-_:]*$')


UNRAVELED_TEXT = {'src_mac', 'dst_mac', 'src_oui', 'dst_oui', 'application_name', 'application_category_name',
                  'requested_server_name', 'client_fingerprint', 'server_fingerprint', 'user_agent', 'content_type'}
UNRAVELED_KEY = ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'bidirectional_first_seen_ms']


def load_unraveled(data_dir: str) -> Flows:
    """Unraveled (DAPT 2021) labeled network flows: data/network-flows/<day>/<sensor>_Flow_labeled.csv.

    Each file is reduced to the endpoints, the labels and float32 features as soon as it is read,
    so the whole capture (about 6.9 million rows) fits in a 12-16 GB machine.
    """
    paths = sorted(glob.glob(str(Path(data_dir) / '**' / '*_Flow_labeled.csv'), recursive=True))
    if not paths:
        raise FileNotFoundError(f'no *_Flow_labeled.csv under {data_dir}')
    header = pd.read_csv(paths[0], nrows=0).columns.tolist()
    feature_columns = [c for c in header if c not in UNRAVELED_EXCLUDE | UNRAVELED_TEXT]
    keep = set(UNRAVELED_KEY) | {'Stage', 'Activity'} | set(feature_columns)
    parts, malformed = [], 0
    for number, path in enumerate(paths, 1):
        frame = pd.read_csv(path, usecols=lambda c: c in keep, low_memory=False, on_bad_lines='skip')
        stage = frame['Stage'].astype(str).str.strip().str.lower()
        activity = frame['Activity'].astype(str).str.strip().str.lower()
        bad = ~stage.str.match(WORDS) | activity.str.match(MIME)
        malformed += int(bad.sum())
        out = pd.DataFrame({'src': frame.src_ip.astype(str).str.strip(), 'dst': frame.dst_ip.astype(str).str.strip()})
        for column in ('src_port', 'dst_port', 'protocol', 'bidirectional_first_seen_ms'):
            out[column] = pd.to_numeric(frame[column], errors='coerce')
        out['stage_raw'] = stage
        out['activity'] = activity
        out['gateway'] = Path(path).name.startswith('netgw')
        for column in feature_columns:
            if column in frame.columns:
                out[column] = pd.to_numeric(frame[column], errors='coerce').astype('float32')
        parts.append(out[~bad.to_numpy()])
        del frame
        if number % 40 == 0:
            print(f'  Unraveled: read {number}/{len(paths)} files', flush=True)
    frame = pd.concat(parts, ignore_index=True)
    del parts
    frame = frame.dropna(subset=['bidirectional_first_seen_ms'])
    for column in ('stage_raw', 'activity'):
        frame[column] = frame[column].astype('category')
    rows_before = len(frame)
    # The same flow is recorded by a subnet sensor and by the gateway: keep one, preferring the subnet.
    frame = frame.sort_values('gateway', kind='stable').drop_duplicates(
        ['src', 'dst', 'src_port', 'dst_port', 'protocol', 'bidirectional_first_seen_ms'], keep='first')
    cross_sensor_duplicates = rows_before - len(frame)
    frame['time'] = pd.to_datetime(frame.bidirectional_first_seen_ms, unit='ms')
    frame['sport'], frame['dport'], frame['proto'] = frame.src_port, frame.dst_port, frame.protocol
    stage = frame.stage_raw.astype(str)
    frame['label'] = (~stage.isin(BENIGN_WORDS)).astype('int8')
    frame['stage'] = np.where(frame.label == 1, stage, 'benign')
    frame['activity'] = frame.activity.astype(str)
    frame['campaign'] = 'single campaign'
    frame['src_internal'] = frame.src.str.startswith('10.1.')
    frame['dst_internal'] = frame.dst.str.startswith('10.1.')
    feats = [c for c in feature_columns if c in frame.columns and frame[c].notna().mean() > 0.5]
    return Flows('Unraveled (DAPT 2021)', 'flow (NFStream biflow)', _slim(frame, feats), feats,
                 {'internal': '10.1.0.0/16', 'campaigns': 'one six-week campaign',
                  'malformed_rows_dropped': malformed, 'cross_sensor_duplicates_dropped': int(cross_sensor_duplicates)})


AIT_URL = 'https://zenodo.org/records/13168643/files/'
AIT_RUNS = ['fox', 'harrison', 'russellmitchell', 'santos', 'shaw', 'wardbeck', 'wheeler', 'wilson']
AIT_BENIGN = {
    'browsing/update', 'benign_share', 'broken flow - benign', 'mail', 'monitoring', 'HTTP(S) intra',
    'HTTP(S) DMZ', 'HTTP', 'HTTPS', 'DNS', 'SSH', 'proxy', 'update/command on unassigned port', 'mDNSv6',
    'mDNS', 'bootp', 'NTP', 'systemd-resolver (local)', 'benign DNS - attacker requesting updates',
    'benign DNS', 'OpenVPN', 'QUIC', 'STUN VoIP', 'NetBIOS', 'GoogleDNS'}
# Analytical grouping of AIT's per-command labels into stages (not an official AIT annotation).
AIT_STAGE = {**{k: 'recon/discovery' for k in ('service_scan', 'host_discover_dmz', 'host_discover_local',
                                               'wpscan', 'dirb_scan', 'dns_brute_force_start')},
             'online_cracking': 'credential access', 'upload_rce_shell': 'initial access/execution',
             'vpn_connect': 'initial access/execution', 'data exfiltration': 'exfiltration'}
AIT_TIME_LIKE = re.compile(r'^(first|last|c_first|s_first|c_last|s_last|c_first_ack|s_first_ack|c_first_abs|'
                           r's_first_abs|req_tm|res_tm|c_last_handshakeT|s_last_handshakeT|c_appdataT|s_appdataT|'
                           r'c_tls_sesid|c_port|c_ip|s_ip)$')
AIT_INTERNAL = {'dmz', 'intranet', 'firewall'}


def _ait_canonical(name: str) -> str:
    return re.sub(r':\d+$', '', re.sub(r'^#(?:15#)?', '', str(name).strip()))


AIT_FEATURES = [
    'c_pkts_all', 'c_rst_cnt', 'c_ack_cnt', 'c_bytes_uniq', 'c_pkts_data', 'c_bytes_all', 'c_pkts_retx',
    'c_syn_cnt', 'c_fin_cnt', 's_pkts_all', 's_rst_cnt', 's_ack_cnt', 's_bytes_uniq', 's_pkts_data',
    's_bytes_all', 's_pkts_retx', 's_syn_cnt', 's_fin_cnt', 'durat', 'c_isint', 's_isint', 'c_iscrypto',
    's_iscrypto', 'con_t', 'http_t', 'c_rtt_avg', 's_rtt_avg', 'c_ttl_min', 'c_ttl_max', 's_ttl_min',
    's_ttl_max', 'c_win_max', 's_win_max', 'c_mss', 's_mss', 'http_req_cnt', 'http_res_cnt', 'c_pkts_push',
    's_pkts_push', 'c_type', 's_type', 's_port_feature', 'is_udp', 'is_complete']
AIT_META = ['c_ip', 's_ip', 'c_port', 's_port', 'timestamp', 'network_cli', 'network_serv', 'label']


def load_ait(data_dir: str) -> Flows:
    """AIT-NDS Tstat flows of the eight runs. Only metadata and a fixed feature subset are read."""
    wanted = set(AIT_FEATURES) | set(AIT_META) | {'c_durat'}
    parts = []
    for run in AIT_RUNS:
        with zipfile.ZipFile(Path(data_dir) / f'{run}_netflows.zip') as zf:
            members = {Path(n).name: n for n in zf.namelist()}
            for member in ('tcp_complete.csv', 'tcp_nocomplete.csv', 'udp_complete.csv'):
                if member not in members:
                    continue
                header = pd.read_csv(zf.open(members[member]), nrows=0).columns
                use = [c for c in header if _ait_canonical(c) in wanted]
                frame = pd.read_csv(zf.open(members[member]), usecols=use, low_memory=False)
                frame.columns = [_ait_canonical(c) for c in frame.columns]
                if 'c_durat' in frame.columns:
                    frame = frame.rename(columns={'c_durat': 'durat'})
                frame['is_udp'] = float(member.startswith('udp'))
                frame['is_complete'] = float(member != 'tcp_nocomplete.csv')
                frame['campaign'] = run
                for column in AIT_FEATURES:
                    if column in frame.columns:
                        frame[column] = pd.to_numeric(frame[column], errors='coerce').astype('float32')
                parts.append(frame)
    frame = pd.concat(parts, ignore_index=True)
    frame['time'] = pd.to_datetime(frame['timestamp'], errors='coerce')
    frame = frame.dropna(subset=['time'])
    label = frame['label'].astype(str).str.strip()
    unlabeled = label.isin({'nan', ''})
    frame, label = frame[~unlabeled].copy(), label[~unlabeled]
    frame['label'] = (~label.isin(AIT_BENIGN)).astype('int8')
    frame['stage'] = np.where(frame.label == 1, label.map(AIT_STAGE).fillna('post-compromise commands'), 'benign')
    frame['activity'] = label.to_numpy()
    frame['src'], frame['dst'] = frame.c_ip.astype(str), frame.s_ip.astype(str)
    frame['sport'], frame['dport'] = frame.c_port, frame.s_port
    frame['s_port_feature'] = pd.to_numeric(frame.s_port, errors='coerce').astype('float32')
    frame['proto'] = np.where(frame.is_udp == 1, 17, 6)
    frame['src_internal'] = frame.network_cli.isin(AIT_INTERNAL)
    frame['dst_internal'] = frame.network_serv.isin(AIT_INTERNAL)
    feats = [c for c in AIT_FEATURES if c in frame.columns]
    frame[feats] = frame[feats].fillna(-1)
    return Flows('AIT-NDS', 'flow (Tstat)', _slim(frame, feats), feats,
                 {'internal': 'zones dmz/intranet/firewall', 'campaigns': 'eight simulated companies (runs)',
                  'stage_mapping': 'analytical grouping of per-command labels; post-compromise commands '
                                   'are all other malicious labels',
                  'unlabeled_rows_dropped': int(unlabeled.sum())})


# ----------------------------------------------------------------------------------------------
# Checks
# ----------------------------------------------------------------------------------------------

def structure(fl: Flows) -> dict:
    f = fl.frame
    att = f[f.label == 1]
    pairs = att.src + ' -> ' + att.dst
    pair_counts = pairs.value_counts()
    internal_pairs = att[att.src_internal & att.dst_internal]
    episodes = 0
    for _, group in att.sort_values('time').groupby('campaign'):
        episodes += int((group.time.diff() > EPISODE_GAP).sum()) + 1
    # A duplicate repeats the start time, 5-tuple and every feature value of an earlier row.
    dup = f.duplicated(['campaign', 'time', 'src', 'sport', 'dst', 'dport', 'proto'] + fl.features, keep='first')
    stages = []
    for stage, g in att.groupby('stage'):
        sp = (g.src + ' -> ' + g.dst).value_counts()
        stages.append(dict(stage=stage, flows=len(g), campaigns=int(g.campaign.nunique()),
                           sources=int(g.src.nunique()), pairs=int(len(sp)),
                           top_pair=sp.index[0], top_pair_share=round(float(sp.iloc[0] / len(g)), 3),
                           internal_pairs=int(((g.src_internal & g.dst_internal)).groupby(g.src + ' -> ' + g.dst).any().sum()),
                           first=str(g.time.min()), last=str(g.time.max()),
                           top_activities=g.activity.value_counts().head(5).to_dict()))
    lateral = att[att.stage.str.contains('lateral')]
    lm_internal = lateral[lateral.src_internal & lateral.dst_internal]
    return dict(
        flows=len(f), attack_flows=len(att), prevalence=round(float(f.label.mean()), 5),
        span_days=round((f.time.max() - f.time.min()).total_seconds() / 86400, 2),
        campaigns=int(f.campaign.nunique()), attack_episodes_6h_gap=episodes,
        attack_sources=int(att.src.nunique()), attack_hosts=int(pd.concat([att.src, att.dst]).nunique()),
        attack_pairs=int(len(pair_counts)),
        internal_attack_pairs=int((internal_pairs.src + ' -> ' + internal_pairs.dst).nunique()),
        top1_pair_share=round(float(pair_counts.iloc[0] / len(att)), 3) if len(att) else None,
        top3_pair_share=round(float(pair_counts.iloc[:3].sum() / len(att)), 3) if len(att) else None,
        lateral_movement_flows=len(lateral),
        lateral_movement_internal_pairs=sorted((lm_internal.src + ' -> ' + lm_internal.dst).unique().tolist())[:20],
        lateral_movement_activities=lateral.activity.value_counts().head(10).to_dict(),
        duplicate_rows=int(dup.sum()), duplicate_rows_labeled=int((dup & (f.label == 1)).sum()),
        stages=stages)


def _splits(f: pd.DataFrame, stage: str | None):
    """Yield (train_index, test_index, description). Campaign split when there are several campaigns."""
    data = f if stage is None else f[(f.label == 0) | (f.stage == stage)]
    pos = data[data.label == 1]
    campaigns = sorted(data.campaign.unique(), key=lambda c: data.time[data.campaign == c].min())
    if len(campaigns) >= 3:
        for c in campaigns:
            test = data.campaign == c
            if pos.campaign.eq(c).any() and pos.campaign.ne(c).any():
                yield data.index[~test], data.index[test], f'leave-one-campaign-out ({c})'
    elif len(campaigns) == 2:
        first = data.campaign == campaigns[0]
        if pos.campaign.eq(campaigns[0]).any() and pos.campaign.eq(campaigns[1]).any():
            yield data.index[first], data.index[~first], f'train {campaigns[0]} / test {campaigns[1]}'
        else:
            yield from _time_split(data, pos)
    else:
        yield from _time_split(data, pos)


def _time_split(data, pos):
    cut = pos.time.median()
    early = data.time < cut
    yield data.index[early], data.index[~early], f'chronological at median attack time ({cut})'


def _matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = frame[features].to_numpy(np.float64)
    return np.nan_to_num(X, nan=-1.0, posinf=1e12, neginf=-1e12).astype(np.float32)


def _lookup_scores(train: pd.DataFrame, test: pd.DataFrame, key: str) -> np.ndarray:
    """Score each test flow by the training attack rate of its key; unseen keys get the training prevalence."""
    rate = train.groupby(key).label.mean()
    return test[key].map(rate).fillna(train.label.mean()).to_numpy()


MIN_TEST_POSITIVES = 10
METHODS = ('pair_lookup', 'source_lookup', 'tree_depth1', 'tree_depth4')


def separability(fl: Flows, stage: str | None = None, max_rows: int = 1_000_000, seed: int = 0) -> dict:
    """AP of identity lookups and shallow trees under the dataset's split, overall or for one stage.

    With several folds (leave-one-campaign-out), the out-of-fold scores are pooled and one AP is
    computed, so that stages with few flows per campaign are still measured on all of them.
    """
    f = fl.frame
    folds, labels, scores = [], [], {m: [] for m in METHODS}
    for tr, te, how in _splits(f, stage):
        train, test = f.loc[tr], f.loc[te]
        if test.label.sum() == 0 or train.label.sum() == 0:
            continue
        print(f'    {stage or "all"}: {how}', flush=True)
        y = test.label.to_numpy()
        fold = dict(split=how, test_flows=len(test), test_positives=int(y.sum()), train_positives=int(train.label.sum()))
        # Lookups use the full training set; only the trees are fitted on a subsample.
        fold_scores = {'pair_lookup': _lookup_scores(train, test, '_pair'),
                       'source_lookup': _lookup_scores(train, test, '_src')}
        if len(train) > max_rows:  # keep every positive, subsample negatives
            n_neg = max(max_rows - int(train.label.sum()), min(len(train), max_rows) // 2)
            neg = train[train.label == 0].sample(min(n_neg, int((train.label == 0).sum())), random_state=seed)
            train = pd.concat([train[train.label == 1], neg])
        X_tr, X_te = _matrix(train, fl.features), _matrix(test, fl.features)
        for depth in (1, 4):
            tree = DecisionTreeClassifier(max_depth=depth, random_state=seed).fit(X_tr, train.label)
            fold_scores[f'tree_depth{depth}'] = tree.predict_proba(X_te)[:, 1]
            if depth == 4:
                importances = pd.Series(tree.feature_importances_, index=fl.features)
                fold['tree4_top_features'] = importances[importances > 0].sort_values(ascending=False).head(4).round(3).to_dict()
        for m in METHODS:
            fold[f'AP_{m}'] = round(float(average_precision_score(y, fold_scores[m])), 3)
            scores[m].append(fold_scores[m])
        labels.append(y)
        folds.append(fold)
    out = dict(stage=stage or 'all', folds=len(folds))
    if not folds:
        return dict(out, evaluable=False, reason='no split with positives in both training and test')
    y = np.concatenate(labels)
    out.update(test_positives=int(y.sum()), test_prevalence=round(float(y.mean()), 5),
               split=folds[0]['split'] if len(folds) == 1 else f'{len(folds)} folds, leave-one-campaign-out (pooled)')
    if y.sum() < MIN_TEST_POSITIVES:
        return dict(out, evaluable=False, reason=f'fewer than {MIN_TEST_POSITIVES} test positives', folds_detail=folds)
    out['evaluable'] = True
    for m in METHODS:
        out[f'AP_{m}'] = round(float(average_precision_score(y, np.concatenate(scores[m]))), 3)
    out['tree4_top_features'] = folds[0].get('tree4_top_features')
    out['folds_detail'] = folds
    return out


def host_labeling(fl: Flows, min_flows: int = 100, min_hours: int = 6) -> dict:
    """Hosts whose outgoing flows carry the attack label through most of their active hours."""
    f = fl.frame
    att_sources = f.loc[f.label == 1, 'src'].value_counts()
    hosts = att_sources.index[att_sources >= min_flows]
    rows = []
    for src, own in f[f.src.isin(hosts)].groupby('src'):
        hours = own.time.dt.floor('h')
        active = hours.nunique()
        labeled = hours[own.label == 1].nunique()
        rows.append(dict(host=src, flows=len(own), label_rate=round(float(own.label.mean()), 3),
                         active_hours=int(active), labeled_hours=int(labeled),
                         labeled_hour_share=round(labeled / active, 3),
                         distinct_services=int(own.loc[own.label == 1, 'dport'].nunique())))
    table = pd.DataFrame(rows)
    if table.empty:
        return dict(hosts_checked=0, hosts_labeled_most_hours=0, table=[])
    flagged = table[(table.active_hours >= min_hours) & (table.labeled_hour_share >= 0.8)
                    & (table.label_rate < 0.95)]
    return dict(hosts_checked=len(table), hosts_labeled_most_hours=len(flagged),
                flagged_hosts=flagged.host.tolist()[:20],
                table=table.sort_values('flows', ascending=False).head(30).to_dict('records'))


def audit(fl: Flows) -> dict:
    print(f'[{fl.name}] {len(fl.frame):,} flows, {int(fl.frame.label.sum()):,} positives, '
          f'{len(fl.features)} features', flush=True)
    report = dict(dataset=fl.name, unit=fl.unit, notes=fl.notes, features=fl.features)
    fl.frame['_pair'] = pd.factorize(fl.frame.src + '>' + fl.frame.dst)[0]
    fl.frame['_src'] = pd.factorize(fl.frame.src)[0]
    report['structure'] = structure(fl)
    report['separability'] = [separability(fl)]
    for s in report['structure']['stages']:
        if s['flows'] >= MIN_STAGE_FLOWS:
            report['separability'].append(separability(fl, s['stage']))
    report['host_labeling'] = host_labeling(fl)
    return report


def summary_row(report: dict) -> dict:
    s, sep, hl = report['structure'], report['separability'][0], report['host_labeling']
    stages = [x for x in report['separability'][1:] if x.get('evaluable')]
    return {
        'dataset': report['dataset'], 'unit': report['unit'], 'flows': s['flows'],
        'attack flows': s['attack_flows'], 'prevalence': s['prevalence'], 'campaigns': s['campaigns'],
        'attack episodes (6 h gap)': s['attack_episodes_6h_gap'], 'attacking sources': s['attack_sources'],
        'attack host pairs': s['attack_pairs'], 'internal attack pairs': s['internal_attack_pairs'],
        'top-3 pair share': s['top3_pair_share'], 'LM flows': s['lateral_movement_flows'],
        'LM internal pairs': len(s['lateral_movement_internal_pairs']),
        'split': sep.get('split'), 'test prevalence': sep.get('test_prevalence'),
        'AP pair lookup': sep.get('AP_pair_lookup'), 'AP source lookup': sep.get('AP_source_lookup'),
        'AP tree d1': sep.get('AP_tree_depth1'), 'AP tree d4': sep.get('AP_tree_depth4'),
        'min stage AP tree d4': min((x['AP_tree_depth4'] for x in stages), default=None),
        'stages evaluated': f"{len(stages)}/{len(s['stages'])}",
        'hosts labeled most hours': hl['hosts_labeled_most_hours'],
    }


def stage_rows(report: dict) -> list[dict]:
    """One row per labeled stage; stages below MIN_STAGE_FLOWS are listed but not evaluated."""
    sep_by_stage = {s['stage']: s for s in report['separability'][1:]}
    rows = []
    for s in report['structure']['stages']:
        sep = sep_by_stage.get(s['stage'], {'evaluable': False, 'reason': f'fewer than {MIN_STAGE_FLOWS} flows'})
        rows.append({'dataset': report['dataset'], 'stage': s['stage'], 'flows': s['flows'],
                     'campaigns': s['campaigns'], 'host pairs': s['pairs'], 'top pair share': s['top_pair_share'],
                     'internal pairs': s['internal_pairs'], 'evaluable': sep.get('evaluable'),
                     'AP pair lookup': sep.get('AP_pair_lookup'), 'AP tree d1': sep.get('AP_tree_depth1'),
                     'AP tree d4': sep.get('AP_tree_depth4'), 'test positives': sep.get('test_positives'),
                     'test prevalence': sep.get('test_prevalence'), 'note': sep.get('reason', '')})
    return rows


def save(report: dict, out_dir: str | Path, name: str | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = name or re.sub(r'[^a-z0-9]+', '_', report['dataset'].lower()).strip('_')
    path = out_dir / f'{slug}.json'
    path.write_text(json.dumps(report, indent=2, default=str))
    return path


# ----------------------------------------------------------------------------------------------
# Downloads (public sources)
# ----------------------------------------------------------------------------------------------

def _fetch(url: str, target: Path, retries: int = 8) -> bool:
    """Download ``url`` to ``target`` unless it exists. Returns True when a request was made.

    Waits and retries on errors; on HTTP 429 (rate limit) it honours Retry-After or waits a minute.
    """
    import time
    import urllib.error
    import urllib.request
    if target.exists() and target.stat().st_size > 0:
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + '.part')
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(request, timeout=300) as response, open(tmp, 'wb') as out:
                while chunk := response.read(1 << 20):
                    out.write(chunk)
            tmp.replace(target)
            return True
        except Exception as error:  # noqa: BLE001
            if attempt == retries - 1:
                raise
            wait = 5 * (attempt + 1)
            if isinstance(error, urllib.error.HTTPError) and error.code == 429:
                wait = int(error.headers.get('Retry-After') or 60) + 5
                print(f'  rate limited, waiting {wait} s', flush=True)
            time.sleep(wait)


def download_dapt2020(data_dir: str) -> None:
    for name in DAPT_FILES:
        _fetch(DAPT_URL + name, Path(data_dir) / name)


def download_ait(data_dir: str) -> None:
    for run in AIT_RUNS:
        _fetch(f'{AIT_URL}{run}_netflows.zip', Path(data_dir) / f'{run}_netflows.zip')


def unraveled_file_list(project: str = 'asu22%2Funraveled', path: str = 'data/network-flows',
                        ref: str = 'master') -> list[str]:
    import urllib.request
    files, page = [], '1'
    while page:
        url = (f'https://gitlab.com/api/v4/projects/{project}/repository/tree?path={path}'
               f'&ref={ref}&recursive=true&per_page=100&page={page}')
        with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'}),
                                    timeout=60) as response:
            items = json.load(response)
            page = response.headers.get('X-Next-Page')
        files += [i['path'] for i in items if i['type'] == 'blob' and i['path'].endswith('_Flow_labeled.csv')]
    return sorted(files)


def download_unraveled(data_dir: str, ref: str = 'master') -> list[str]:
    files = unraveled_file_list(ref=ref)
    import time
    for number, path in enumerate(files, 1):
        if _fetch(f'https://gitlab.com/asu22/unraveled/-/raw/{ref}/{path}', Path(data_dir) / path):
            time.sleep(2)  # stay under GitLab's rate limit
        if number % 20 == 0:
            print(f'  Unraveled: {number}/{len(files)} files')
    return files
