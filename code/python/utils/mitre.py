import ipaddress

import numpy as np
import pandas as pd
import torch


def get_mitre_tactic(src_ip, dst_ip, c2_servers, external_attackers,
                     port_category, internal_net="172.31.0.0/16"):
    """
    Infer the most likely MITRE ATT&CK tactic for a network flow.

    Uses source/destination IP roles (internal vs external, C2, known attacker)
    and port category to classify the tactic. Returns a descriptive label string.

    Parameters
    ----------
    src_ip             : str — source IP address
    dst_ip             : str — destination IP address
    c2_servers         : list[str] — known C2 server IPs
    external_attackers : list[str] — known external attacker IPs
    port_category      : str — port role label (e.g. 'Web', 'DNS', 'Windows/SMB')
    internal_net       : str — CIDR of the internal network (default: 172.31.0.0/16)

    Returns
    -------
    str — MITRE tactic label
    """
    internal_network = ipaddress.ip_network(internal_net)
    src_is_internal = ipaddress.ip_address(str(src_ip)) in internal_network
    dst_is_internal = ipaddress.ip_address(str(dst_ip)) in internal_network

    # 1. INITIAL ACCESS
    # External attacker → internal host
    if src_ip in external_attackers and dst_is_internal:
        if port_category in ['Web', 'High Ports / Ephemeral (>= 1024)']:
            return "1. Initial Access / Exploitation (Medium Confidence)"
        return f"1. Initial Access (Port: {port_category})"

    # 2. COMMAND & CONTROL
    # Internal machine → known C2 server
    elif src_is_internal and dst_ip in c2_servers:
        if port_category in ['Web', 'DNS']:
            return "2. Command & Control (Stealth/Web/DNS)"
        elif port_category == 'High Ports / Ephemeral (>= 1024)':
            return "2. Command & Control (Custom/High Port)"
        return f"2. Command & Control (Port: {port_category})"

    # 3. DISCOVERY / LATERAL MOVEMENT
    # Internal → internal (post-compromise behaviour)
    elif src_is_internal and dst_is_internal:
        if port_category in ['Windows/SMB', 'Admin/Remote']:
            return f"3. Lateral Movement ({port_category})"
        elif port_category == 'Database':
            return "3. Discovery / Targeting Internal Data Stores"
        elif port_category in ['Other Privileged (< 1024)', 'High Ports / Ephemeral (>= 1024)']:
            return "3. Discovery / Possible Internal Reconnaissance"
        return f"3. Internal Malicious Activity (Port: {port_category})"

    # 4. EXTERNAL MALICIOUS INFRASTRUCTURE COMMUNICATION
    elif src_ip in external_attackers and not dst_is_internal:
        return "4. External Malicious Infrastructure Communication"

    else:
        return "Unknown Malicious Activity"


def canonical_tactic(tactic):
    """Map a full MITRE tactic label to its base category name."""
    if tactic.startswith("1. Initial Access"):
        return "Initial Access"
    elif tactic.startswith("2. Command & Control"):
        return "Command & Control"
    elif tactic.startswith("3. Lateral Movement"):
        return "Lateral Movement"
    elif tactic.startswith("3. Discovery"):
        return "Discovery"
    elif tactic.startswith("3. Internal Malicious Activity"):
        return "Internal Malicious Activity"
    elif tactic.startswith("4. External Malicious Infrastructure Communication"):
        return "External Malicious Infrastructure Communication"
    else:
        return "Unknown"


@torch.no_grad()
def extract_mitre_events(loader, device, model, optimal_threshold,
                         is_temporal, id_to_ip_dict,
                         c2_servers, external_attackers, internal_net):
    """
    Run inference over a DataLoader and return a flow-level MITRE events DataFrame.

    Captures every flow that is a real attack (y=1) OR triggers a model alert
    (prob >= optimal_threshold), enriched with MITRE tactic labels.

    Parameters
    ----------
    loader            : DataLoader
    device            : torch.device
    model             : trained GNN model
    optimal_threshold : float
    is_temporal       : bool
    id_to_ip_dict     : dict — global_node_id → IP string
    c2_servers        : list[str]
    external_attackers: list[str]
    internal_net      : str — CIDR string

    Returns
    -------
    pd.DataFrame with columns: Graph_Window_Idx, Source_IP, Dest_IP,
        Port_Category, y_real, y_pred, Probability,
        MITRE_Tactic, MITRE_Tactic_Base
    """
    port_roles = [
        'Web', 'Admin/Remote', 'Windows/SMB', 'DNS', 'Database',
        'Other Privileged (< 1024)', 'High Ports / Ephemeral (>= 1024)'
    ]

    model.eval()
    if hasattr(model, 'reset_memory'):
        model.reset_memory()

    results = []

    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        if data.x.shape[0] == 0:
            continue

        use_stats = getattr(model, 'use_node_stats', False)

        if is_temporal:
            if use_stats:
                out = model(data.x, data.edge_index, data.edge_attr,
                            data.global_node_ids, data.node_stats)
            else:
                out = model(data.x, data.edge_index, data.edge_attr,
                            data.global_node_ids)
        else:
            if use_stats:
                out = model(data.x, data.edge_index, data.edge_attr,
                            data.node_stats)
            else:
                out = model(data.x, data.edge_index, data.edge_attr)

        probs = torch.sigmoid(out.view(-1)).cpu().numpy()
        trues = data.y.cpu().numpy()
        edges = data.edge_index.cpu().numpy()
        global_ids = data.global_node_ids.cpu().numpy()

        relevant_indices = np.where((probs >= optimal_threshold) | (trues == 1))[0]

        for i in relevant_indices:
            local_src, local_dst = edges[0, i], edges[1, i]
            src_ip = id_to_ip_dict.get(global_ids[local_src], "Unknown")
            dst_ip = id_to_ip_dict.get(global_ids[local_dst], "Unknown")

            port_vector = data.edge_attr[i, 0:7].cpu().numpy()
            port_category = port_roles[port_vector.argmax()]

            tactic = get_mitre_tactic(
                src_ip, dst_ip, c2_servers, external_attackers,
                port_category, internal_net=internal_net
            )

            results.append({
                'Graph_Window_Idx': batch_idx,
                'Source_IP': src_ip,
                'Dest_IP': dst_ip,
                'Port_Category': port_category,
                'y_real': int(trues[i]),
                'y_pred': int(probs[i] >= optimal_threshold),
                'Probability': probs[i],
                'MITRE_Tactic': tactic,
                'MITRE_Tactic_Base': canonical_tactic(tactic)
            })

    return pd.DataFrame(results)


def count_total_flows(loader):
    """Count total edges (flows) across all batches in a DataLoader."""
    total_flows = 0
    for batch in loader:
        total_flows += batch.edge_index.shape[1]
    return total_flows


def metrics_per_tactic(df_flows, total_flows, ignore_unknown=False):
    """
    Compute Precision, Recall, F1, F2, and FPR per MITRE tactic.

    Parameters
    ----------
    df_flows      : pd.DataFrame — output of extract_mitre_events
    total_flows   : int — total edge count from count_total_flows
    ignore_unknown: bool — skip rows with MITRE_Tactic_Base == 'Unknown'

    Returns
    -------
    pd.DataFrame sorted by F1-Score descending
    """
    tactics = df_flows['MITRE_Tactic_Base'].unique()
    results = []

    for tactic in tactics:
        if tactic == 'Unknown' and ignore_unknown:
            continue

        df_tactic = df_flows[df_flows['MITRE_Tactic_Base'] == tactic]

        tp = len(df_tactic[(df_tactic['y_real'] == 1) & (df_tactic['y_pred'] == 1)])
        fp = len(df_tactic[(df_tactic['y_real'] == 0) & (df_tactic['y_pred'] == 1)])
        fn = len(df_tactic[(df_tactic['y_real'] == 1) & (df_tactic['y_pred'] == 0)])
        tn = total_flows - tp - fp - fn

        total_real   = tp + fn
        total_alerts = tp + fp

        if total_real == 0 and total_alerts == 0:
            continue

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1  = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f2  = (5 * precision * recall) / ((4 * precision) + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        results.append({
            'MITRE_Tactic': tactic,
            'Real Flows (Support)': total_real,
            'Alerts Generated (Preds)': total_alerts,
            'True Positives (TP)': tp,
            'False Positives (FP)': fp,
            'Precision (%)': round(precision * 100, 2),
            'Recall (%)': round(recall * 100, 2),
            'F1-Score (%)': round(f1 * 100, 2),
            'F2-Score (%)': round(f2 * 100, 2),
            'FPR (%)': round(fpr * 100, 6)
        })

    return pd.DataFrame(results).sort_values('F1-Score (%)', ascending=False)


def analyze_timeline(df_real, df_tp, time_window_sec=30):
    """
    Reconstruct the attack timeline and compute Time-To-Detect (TTD) per tactic.

    Parameters
    ----------
    df_real          : pd.DataFrame — ground truth flows (y_real == 1)
    df_tp            : pd.DataFrame — true positive predictions
    time_window_sec  : int — duration of each graph window in seconds

    Returns
    -------
    pd.DataFrame with TTD, span, and consistency metrics per MITRE tactic,
    sorted chronologically by real attack start time
    """
    real_stats = df_real.groupby('MITRE_Tactic_Base').agg(
        Real_Start_Idx=('Graph_Window_Idx', 'min'),
        Real_Active_Windows=('Graph_Window_Idx', 'nunique')
    ).reset_index()

    tp_stats = df_tp.groupby('MITRE_Tactic_Base').agg(
        First_Detection_Idx=('Graph_Window_Idx', 'min'),
        Last_Detection_Idx=('Graph_Window_Idx', 'max'),
        Total_Alerts=('Graph_Window_Idx', 'count'),
        Active_Windows=('Graph_Window_Idx', 'nunique'),
        Mean_Probability=('Probability', 'mean'),
        Max_Probability=('Probability', 'max')
    ).reset_index()

    timeline_df = pd.merge(real_stats, tp_stats, on='MITRE_Tactic_Base', how='left')
    timeline_df = timeline_df.dropna(subset=['First_Detection_Idx']).copy()

    timeline_df['Real_Start_Minute']      = timeline_df['Real_Start_Idx'] * (time_window_sec / 60.0)
    timeline_df['First_Detection_Minute'] = timeline_df['First_Detection_Idx'] * (time_window_sec / 60.0)
    timeline_df['Last_Detection_Minute']  = timeline_df['Last_Detection_Idx'] * (time_window_sec / 60.0)
    timeline_df['TTD_Minutes']            = timeline_df['First_Detection_Minute'] - timeline_df['Real_Start_Minute']
    timeline_df['Span_Windows']           = timeline_df['Last_Detection_Idx'] - timeline_df['First_Detection_Idx'] + 1
    timeline_df['Span_Minutes']           = timeline_df['Span_Windows'] * (time_window_sec / 60.0)
    timeline_df['Window_Consistency_Pct'] = (
        timeline_df['Active_Windows'] / timeline_df['Real_Active_Windows'] * 100
    ).round(2)

    timeline_df = timeline_df.sort_values('Real_Start_Idx')

    return timeline_df[[
        'MITRE_Tactic_Base',
        'Real_Active_Windows', 'Real_Start_Idx',
        'First_Detection_Idx', 'Last_Detection_Idx',
        'Real_Start_Minute', 'First_Detection_Minute', 'Last_Detection_Minute',
        'TTD_Minutes', 'Span_Windows', 'Span_Minutes',
        'Active_Windows', 'Window_Consistency_Pct',
        'Total_Alerts', 'Mean_Probability', 'Max_Probability'
    ]]


def analyze_early_warning(df_mitre, time_window_sec):
    """
    Identify the first alert and compute lead time before the first severe tactic.

    Parameters
    ----------
    df_mitre        : pd.DataFrame — output of extract_mitre_events (TP rows)
    time_window_sec : int

    Returns
    -------
    dict with keys: first_alert_idx, first_alert_minute, and optionally
        first_severe_idx, first_severe_minute, lead_time_sec, lead_time_min
    """
    df = df_mitre.copy()
    df["MITRE_Tactic_Base"] = df["MITRE_Tactic"].apply(canonical_tactic)

    first_alert_idx = df["Graph_Window_Idx"].min()
    severe_tactics  = ["Lateral Movement"]
    severe_df       = df[df["MITRE_Tactic_Base"].isin(severe_tactics)]
    first_severe_idx = severe_df["Graph_Window_Idx"].min() if not severe_df.empty else None

    result = {
        "first_alert_idx":    first_alert_idx,
        "first_alert_minute": first_alert_idx * time_window_sec / 60.0
    }

    if first_severe_idx is not None:
        result["first_severe_idx"]    = first_severe_idx
        result["first_severe_minute"] = first_severe_idx * time_window_sec / 60.0
        result["lead_time_sec"]       = (first_severe_idx - first_alert_idx) * time_window_sec
        result["lead_time_min"]       = result["lead_time_sec"] / 60.0

    return result


def analyze_all_lateral_movements_pairs(df_tp, time_window_sec=30):
    """
    Track all IP pairs involved in Lateral Movement and compute per-pair lead time.

    For each (attacker → victim) pair in Lateral Movement flows, looks back in the
    alert history to find the earliest alert involving either IP, then computes the
    lead time before the lateral movement began.

    Parameters
    ----------
    df_tp           : pd.DataFrame — true positive flows from extract_mitre_events
    time_window_sec : int

    Returns
    -------
    pd.DataFrame sorted by Lead_Time_Min descending
    """
    df = df_tp.copy()
    lm_df   = df[df["MITRE_Tactic_Base"] == "Lateral Movement"]
    lm_pairs = lm_df[["Source_IP", "Dest_IP"]].drop_duplicates().values

    print(f"{len(lm_pairs)} pairs of IPs interacting in Lateral Movement were found.")

    results = []

    for src, dst in lm_pairs:
        pair_lm_df   = lm_df[(lm_df["Source_IP"] == src) & (lm_df["Dest_IP"] == dst)]
        first_lm_idx = pair_lm_df["Graph_Window_Idx"].min()

        history = df[
            ((df["Source_IP"] == src) | (df["Dest_IP"] == src) |
             (df["Source_IP"] == dst) | (df["Dest_IP"] == dst)) &
            (df["Graph_Window_Idx"] <= first_lm_idx)
        ]

        if not history.empty:
            first_alert_idx = history["Graph_Window_Idx"].min()
            early_warnings  = history[history["Graph_Window_Idx"] == first_alert_idx]
            first_tactic    = early_warnings["MITRE_Tactic_Base"].mode()[0]

            src_in_early = (early_warnings["Source_IP"] == src).any() or (early_warnings["Dest_IP"] == src).any()
            dst_in_early = (early_warnings["Source_IP"] == dst).any() or (early_warnings["Dest_IP"] == dst).any()

            trigger_ip   = []
            alert_detail = ""

            if src_in_early:
                trigger_ip.append(f"Attack_LM ({src})")

            if dst_in_early:
                trigger_ip.append(f"Victim_LM ({dst})")

                attacks_received   = early_warnings[early_warnings["Dest_IP"] == dst]
                connections_started = early_warnings[early_warnings["Source_IP"] == dst]

                if not attacks_received.empty:
                    attackers    = attacks_received["Source_IP"].unique()
                    alert_detail = "Attacked by: " + ", ".join(attackers)
                elif not connections_started.empty:
                    destinations = connections_started["Dest_IP"].unique()
                    alert_detail = "Beaconing towards: " + ", ".join(destinations)
                else:
                    alert_detail = f"Prior alert in Attacker ({src})"

            lead_time_min = (first_lm_idx - first_alert_idx) * time_window_sec / 60.0

            results.append({
                "LM_Source":            src,
                "LM_Dest":              dst,
                "First_Alert_Min":      first_alert_idx * time_window_sec / 60.0,
                "Alert_Caused_By":      " and ".join(trigger_ip),
                "Forensic_Detail":      alert_detail,
                "Early_Warning_Tactic": first_tactic,
                "Lateral_Movement_Min": first_lm_idx * time_window_sec / 60.0,
                "Lead_Time_Min":        lead_time_min
            })

    df_results = pd.DataFrame(results)

    if not df_results.empty:
        avg_lead_time = df_results["Lead_Time_Min"].mean()
        print(f"\n Global Average Lead Time (per pair): {avg_lead_time:.1f} minutes")
        print("\n Top Tactics That Gave Early Warning:")
        print(df_results["Early_Warning_Tactic"].value_counts().to_string())
        print("\n")

    return df_results.sort_values("Lead_Time_Min", ascending=False)
