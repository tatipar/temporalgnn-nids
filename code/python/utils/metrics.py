import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    average_precision_score, roc_auc_score, confusion_matrix,
)


# ---------------------------------------------------------------------------
# Flow-level metrics
# ---------------------------------------------------------------------------

def calculate_metrics_gnn(y_true, y_probs, prob_threshold=0.5):
    """
    Compute classification metrics from probabilities.

    Parameters
    ----------
    y_true  : array-like of int (0 or 1)
    y_probs : array-like of float — PROBABILITIES (0.0–1.0), not logits
    prob_threshold : float

    Returns
    -------
    dict with Precision, Recall, F1, F2, AUC-PR, AUC-ROC, FPR, TP, FP, TN, FN
    """
    y_true = np.array(y_true)
    probs = np.array(y_probs)

    preds = (probs > prob_threshold).astype(int)

    prec = precision_score(y_true, preds, zero_division=0)
    rec  = recall_score(y_true, preds, zero_division=0)
    f1   = f1_score(y_true, preds, zero_division=0)
    f2   = fbeta_score(y_true, preds, beta=2, zero_division=0)

    try:
        ap  = average_precision_score(y_true, probs)
        roc = roc_auc_score(y_true, probs)
    except ValueError:
        # Only one class present in y_true
        ap = 0.0
        roc = 0.5

    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    return {
        "Precision": prec, "Recall": rec, "F1": f1, "F2": f2,
        "AUC-PR": ap, "AUC-ROC": roc, "FPR": fpr,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }


def calc_metrics(TP, FP, TN, FN, phase_name):
    """
    Compute Precision, Recall, F1, F2, FPR from raw confusion matrix counts.
    Used in multi-phase SOC evaluations.
    """
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1   = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    f2   = (5 * prec * rec) / ((4 * prec) + rec) if (prec + rec) > 0 else 0.0
    fpr  = FP / (FP + TN) if (FP + TN) > 0 else 0.0

    return {
        "Phase": phase_name,
        "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        "Precision": prec, "Recall": rec, "F1": f1, "F2": f2, "FPR": fpr,
    }


# ---------------------------------------------------------------------------
# Temporal / tactic-level metrics
# ---------------------------------------------------------------------------

def calculate_full_temporal_metrics(df_ground_truth, df_tp, df_fp, ignore_unknown=False):
    """
    Per-tactic temporal recall and global precision over detection windows.

    Parameters
    ----------
    df_ground_truth : DataFrame with columns [MITRE_Tactic_Base, Graph_Window_Idx]
    df_tp           : DataFrame — true-positive detections (same columns)
    df_fp           : DataFrame — false-positive detections (Graph_Window_Idx)
    ignore_unknown  : bool — skip rows where MITRE_Tactic_Base == 'Unknown'

    Returns
    -------
    DataFrame with per-tactic TP_Windows, FN_Windows, Temporal_Recall (%)
    Also prints global precision to stdout.
    """
    metrics = []
    tactics = set(df_ground_truth['MITRE_Tactic_Base'])

    for tactic in tactics:
        if tactic == 'Unknown' and ignore_unknown:
            continue

        real_windows = set(
            df_ground_truth[df_ground_truth['MITRE_Tactic_Base'] == tactic]['Graph_Window_Idx']
        )
        tp_windows = set(
            df_tp[df_tp['MITRE_Tactic_Base'] == tactic]['Graph_Window_Idx']
        )

        tp_win = len(tp_windows)
        fn_win = len(real_windows - tp_windows)
        recall = (tp_win / (tp_win + fn_win) * 100) if (tp_win + fn_win) > 0 else 0.0

        metrics.append({
            'MITRE_Tactic_Base': tactic,
            'TP_Windows': tp_win,
            'FN_Windows': fn_win,
            'Temporal_Recall (%)': round(recall, 2),
        })

    df_metrics = pd.DataFrame(metrics).sort_values('Temporal_Recall (%)', ascending=False)

    # Global FP precision (FPs have no tactic — they're benign traffic)
    global_fp_windows = df_fp['Graph_Window_Idx'].nunique()
    global_tp_windows = df_tp['Graph_Window_Idx'].nunique()
    global_precision = (
        global_tp_windows / (global_tp_windows + global_fp_windows) * 100
        if (global_tp_windows + global_fp_windows) > 0
        else 0.0
    )

    print("=" * 50)
    print(" GLOBAL SYSTEM METRICS:")
    print(f"   -> Total FP Windows (False Alarms): {global_fp_windows}")
    print(f"   -> Precision Temporal Global: {global_precision:.2f}%")
    print("=" * 50)
    print()

    return df_metrics


def calculate_mttd_metrics(df_real, df_tp, time_window_sec=30, max_gap_windows=5):
    """
    Calculate the Mean Time-To-Detect (MTTD) by grouping attacks into
    instances separated by periods of inactivity (gaps).

    Parameters
    ----------
    df_real          : DataFrame with [MITRE_Tactic_Base, Graph_Window_Idx]
    df_tp            : DataFrame — true-positive detections (same columns)
    time_window_sec  : int — duration of each graph window in seconds
    max_gap_windows  : int — max gap between windows to still be same instance

    Returns
    -------
    DataFrame with per-tactic MTTD, median TTD, instance detection rate, etc.
    """
    df_real = df_real.sort_values(['MITRE_Tactic_Base', 'Graph_Window_Idx']).copy()

    # Identify contiguous attack instances (separated by gaps)
    df_real['Window_Diff'] = df_real.groupby('MITRE_Tactic_Base')['Graph_Window_Idx'].diff()
    df_real['New_Instance_Flag'] = (
        (df_real['Window_Diff'] > max_gap_windows) | df_real['Window_Diff'].isna()
    )
    df_real['Instance_ID'] = df_real.groupby('MITRE_Tactic_Base')['New_Instance_Flag'].cumsum()

    instances_df = df_real.groupby(['MITRE_Tactic_Base', 'Instance_ID']).agg(
        Real_Start_Idx=('Graph_Window_Idx', 'min'),
        Real_End_Idx=('Graph_Window_Idx', 'max'),
    ).reset_index()

    def get_first_detection(row):
        tps = df_tp[
            (df_tp['MITRE_Tactic_Base'] == row['MITRE_Tactic_Base']) &
            (df_tp['Graph_Window_Idx'] >= row['Real_Start_Idx']) &
            (df_tp['Graph_Window_Idx'] <= row['Real_End_Idx'])
        ]
        return tps['Graph_Window_Idx'].min() if not tps.empty else pd.NA

    instances_df['First_Detection_Idx'] = instances_df.apply(get_first_detection, axis=1)

    instances_df['Real_Start_Min'] = instances_df['Real_Start_Idx'] * (time_window_sec / 60.0)
    instances_df['First_Detection_Min'] = (
        pd.to_numeric(instances_df['First_Detection_Idx']) * (time_window_sec / 60.0)
    )
    instances_df['TTD_Instance_Min'] = (
        instances_df['First_Detection_Min'] - instances_df['Real_Start_Min']
    )

    mttd_summary = instances_df.groupby('MITRE_Tactic_Base').agg(
        Total_Instances=('Instance_ID', 'count'),
        Detected_Instances=('TTD_Instance_Min', 'count'),
        MTTD_Minutes=('TTD_Instance_Min', 'mean'),
        Median_TTD_Minutes=('TTD_Instance_Min', 'median'),
        Max_TTD_Minutes=('TTD_Instance_Min', 'max'),
        Std_TTD_Minutes=('TTD_Instance_Min', 'std'),
        Real_Start_Minute=('Real_Start_Min', 'min'),
    ).reset_index()

    mttd_summary['Std_TTD_Minutes'] = mttd_summary['Std_TTD_Minutes'].fillna(0).round(2)
    mttd_summary['Instance_Detection_Rate_%'] = (
        mttd_summary['Detected_Instances'] / mttd_summary['Total_Instances'] * 100
    ).round(2)
    mttd_summary['MTTD_Minutes'] = mttd_summary['MTTD_Minutes'].round(2)
    mttd_summary['Median_TTD_Minutes'] = mttd_summary['Median_TTD_Minutes'].round(2)

    return mttd_summary


# ---------------------------------------------------------------------------
# SOC evaluation (multi-phase: raw → DNS correction → SMB whitelist)
# ---------------------------------------------------------------------------

def evaluation_soc(
    loader,
    device,
    model,
    optimal_threshold,
    id_to_ip_dict,
    is_temporal,
    focus_ips=None,
    dns_suspicious_ips=None,
    smb_whitelist_ips=None,
    return_flow_df=False,
    verbose=True,
):
    """
    Evaluate the model and simulate SOC rules.

    Parameters
    ----------
    loader            : DataLoader
    device            : torch.device
    model             : trained GNN model
    optimal_threshold : float — classification threshold
    id_to_ip_dict     : dict mapping global node IDs to IP strings
    is_temporal       : bool
    focus_ips         : str | list[str] | None — IPs to check for innocence
    dns_suspicious_ips: str | list[str] | None — DNS dest IPs for FP→TP correction
    smb_whitelist_ips : str | list[str] | None — source IPs for SMB FP→TN correction
                        Defaults to focus_ips if None.
    return_flow_df    : bool — if True, include per-flow DataFrame in result
    verbose           : bool

    Returns
    -------
    dict with summary_df, counts_df, focus_ip_df, dns_fp2tp, smb_fp2tn,
    whitelist_safe, and optionally flow_df
    """
    def _to_set(x):
        if x is None:
            return set()
        if isinstance(x, str):
            return {x}
        return set(x)

    if smb_whitelist_ips is None:
        smb_whitelist_ips = focus_ips

    focus_ips         = _to_set(focus_ips)
    dns_suspicious_ips = _to_set(dns_suspicious_ips)
    smb_whitelist_ips  = _to_set(smb_whitelist_ips)

    port_roles = [
        'Web',
        'Admin/Remote',
        'Windows/SMB',
        'DNS',
        'Database',
        'Other Privileged (< 1024)',
        'High Ports / Ephemeral (>= 1024)',
    ]

    model.eval()
    if hasattr(model, 'reset_memory'):
        model.reset_memory()

    TP_raw, FP_raw, TN_raw, FN_raw = 0, 0, 0, 0
    dns_fp2tp = 0
    smb_fp2tn = 0
    real_attacks_by_focus_ip = {ip: 0 for ip in focus_ips}
    flow_rows = []

    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        if data.x.shape[0] == 0:
            continue

        if is_temporal:
            out = model(data.x, data.edge_index, data.edge_attr, data.global_node_ids)
        else:
            out = model(data.x, data.edge_index, data.edge_attr)

        probs      = torch.sigmoid(out.view(-1)).cpu().numpy()
        trues      = data.y.cpu().numpy()
        edges      = data.edge_index.cpu().numpy()
        global_ids = data.global_node_ids.cpu().numpy()

        for i in range(len(probs)):
            is_attack_pred  = probs[i] >= optimal_threshold
            is_real_attack  = trues[i] == 1

            local_src, local_dst = edges[0, i], edges[1, i]
            src_ip = id_to_ip_dict.get(global_ids[local_src], "Unknown")
            dst_ip = id_to_ip_dict.get(global_ids[local_dst], "Unknown")

            port_vector   = data.edge_attr[i, 0:7].cpu().numpy()
            port_category = port_roles[port_vector.argmax()]

            for ip in focus_ips:
                if src_ip == ip and is_real_attack:
                    real_attacks_by_focus_ip[ip] += 1

            if is_attack_pred and is_real_attack:
                TP_raw += 1
            elif is_attack_pred and not is_real_attack:
                FP_raw += 1
            elif not is_attack_pred and not is_real_attack:
                TN_raw += 1
            else:
                FN_raw += 1

            dns_rule_applied = (
                dst_ip in dns_suspicious_ips
                and port_category == 'DNS'
                and not is_real_attack
                and is_attack_pred
            )
            if dns_rule_applied:
                dns_fp2tp += 1

            smb_rule_applied = (
                src_ip in smb_whitelist_ips
                and port_category == 'Windows/SMB'
                and not is_real_attack
                and is_attack_pred
            )
            if smb_rule_applied:
                smb_fp2tn += 1

            if return_flow_df:
                flow_rows.append({
                    'Graph_Window_Idx': batch_idx,
                    'Source_IP':        src_ip,
                    'Dest_IP':          dst_ip,
                    'Port_Category':    port_category,
                    'Probability':      float(probs[i]),
                    'y_real':           int(is_real_attack),
                    'y_pred':           int(is_attack_pred),
                    'dns_rule_applied': int(dns_rule_applied),
                    'smb_rule_applied': int(smb_rule_applied),
                    'focus_src':        int(src_ip in focus_ips),
                    'focus_dst':        int(dst_ip in focus_ips),
                })

    # Phase 1: raw
    phase_1 = calc_metrics(TP_raw, FP_raw, TN_raw, FN_raw, "PHASE 1: RAW MATRIX")

    # Phase 2: DNS label correction (FP → TP)
    phase_2 = calc_metrics(
        TP_raw + dns_fp2tp, FP_raw - dns_fp2tp, TN_raw, FN_raw,
        "PHASE 2: LABEL CORRECTION DNS",
    )

    # Phase 3: SMB whitelist (FP → TN), only if no focus IP was an attacker
    whitelist_safe = (
        all(real_attacks_by_focus_ip[ip] == 0 for ip in focus_ips)
        if focus_ips else True
    )
    TP_dns, FP_dns, TN_dns, FN_dns = (
        phase_2['TP'], phase_2['FP'], phase_2['TN'], phase_2['FN']
    )
    if whitelist_safe:
        phase_3 = calc_metrics(
            TP_dns, FP_dns - smb_fp2tn, TN_dns + smb_fp2tn, FN_dns,
            "PHASE 3: SOC + WHITELIST SMB",
        )
    else:
        phase_3 = calc_metrics(
            TP_dns, FP_dns, TN_dns, FN_dns,
            "PHASE 3: SOC WITHOUT WHITELIST (Unsecured IPs)",
        )

    counts_df  = pd.DataFrame([phase_1, phase_2, phase_3])[["Phase", "TP", "FP", "TN", "FN"]]
    summary_df = pd.DataFrame([phase_1, phase_2, phase_3])[
        ["Phase", "Precision", "Recall", "F1", "F2", "FPR"]
    ]
    focus_ip_df = pd.DataFrame([
        {
            "IP": ip,
            "Real_Attack_Flows": real_attacks_by_focus_ip[ip],
            "Whitelist_Safe": real_attacks_by_focus_ip[ip] == 0,
        }
        for ip in sorted(real_attacks_by_focus_ip)
    ]) if focus_ips else pd.DataFrame(columns=["IP", "Real_Attack_Flows", "Whitelist_Safe"])

    if verbose:
        print("\n=== METRICS SUMMARY ===")
        print(summary_df.to_string(index=False))
        print("\n=== COUNTS ===")
        print(counts_df.to_string(index=False))
        if dns_suspicious_ips:
            print(f"\n DNS correction applied: {dns_fp2tp} flows FP -> TP")
        if focus_ips:
            print("\n=== IPs FOCUS VERIFICATION ===")
            print(focus_ip_df.to_string(index=False))
        if smb_whitelist_ips:
            if whitelist_safe:
                print(f"\n Whitelist SMB applied: {smb_fp2tn} flows FP -> TN")
            else:
                print("\n SMB whitelisting not applied: at least one focus IP participated as attacker.")

    result = {
        "summary_df":    summary_df,
        "counts_df":     counts_df,
        "focus_ip_df":   focus_ip_df,
        "dns_fp2tp":     dns_fp2tp,
        "smb_fp2tn":     smb_fp2tn,
        "whitelist_safe": whitelist_safe,
    }
    if return_flow_df:
        result["flow_df"] = pd.DataFrame(flow_rows)

    return result
