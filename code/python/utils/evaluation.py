import glob
import json
import os
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve

from .metrics import calculate_metrics_gnn
from .training import evaluate


def extract_seed(text):
    """Extract the seed number from a string (e.g. 'seed_42' or 'seed42' → 42)."""
    match = re.search(r'seed_?(\d+)', str(text), re.IGNORECASE)
    return int(match.group(1)) if match else None


def gather_metrics(base_log_path, model_name_mapping):
    """
    Collect per-seed metrics from training log directories.

    Reads JSON files for best_epoch and CSV files for threshold/F1 data,
    then cross-references them by seed.

    Parameters
    ----------
    base_log_path      : str — path to the logs directory
                               (contains one subdirectory per experiment)
    model_name_mapping : dict — raw directory name → display name

    Returns
    -------
    pd.DataFrame with columns: Architecture, Raw_Dir_Name, Seed,
        F1_Score, Best_Epoch, Optimal_Threshold
    """
    results = []

    for raw_model_name in os.listdir(base_log_path):
        model_dir = os.path.join(base_log_path, raw_model_name)
        if not os.path.isdir(model_dir):
            continue

        clean_model_name = model_name_mapping.get(raw_model_name, raw_model_name)

        epoch_dict = {}
        for jf in glob.glob(os.path.join(model_dir, "*.json")):
            with open(jf) as f:
                data = json.load(f)
            seed = data.get('seed')
            if seed is None:
                seed = extract_seed(os.path.basename(jf))
            best_ep = data.get('early_stopping', {}).get('best_epoch', np.nan)
            if seed is not None:
                epoch_dict[seed] = best_ep

        csv_files = glob.glob(os.path.join(model_dir, "metrics_newth_*.csv"))
        if not csv_files:
            csv_files = glob.glob(os.path.join(model_dir, "run_metrics_*.csv"))
        csv_file = csv_files[0] if csv_files else None

        if csv_file is None:
            print(f" Warning: No metrics CSV found in {model_dir}")
            continue

        df_csv = pd.read_csv(csv_file)
        df_csv.columns = df_csv.columns.str.lower().str.strip()

        for _, row in df_csv.iterrows():
            col_name = 'model_name' if 'model_name' in df_csv.columns else df_csv.columns[0]
            seed     = extract_seed(row[col_name])
            f1_col   = next((c for c in df_csv.columns if 'f1' in c.lower()), None)
            th_col   = next((c for c in df_csv.columns if 'threshold' in c.lower() or 'th' in c.lower()), None)

            if seed is not None and f1_col is not None:
                results.append({
                    'Architecture':      clean_model_name,
                    'Raw_Dir_Name':      raw_model_name,
                    'Seed':              seed,
                    'F1_Score':          row[f1_col],
                    'Best_Epoch':        epoch_dict.get(seed, np.nan),
                    'Optimal_Threshold': row[th_col] if th_col else 0.5
                })

    return pd.DataFrame(results)


def apply_1sd_rule(df_results):
    """
    Select one champion model per architecture using the 1-SD rule.

    Within each architecture, finds seeds whose F1 falls within one standard
    deviation of the maximum F1. Among those candidates, picks the most
    parsimonious (fewest training epochs), breaking ties by highest F1.

    Parameters
    ----------
    df_results : pd.DataFrame — output of gather_metrics

    Returns
    -------
    pd.DataFrame — one row per architecture (the champion)
    """
    print("=" * 80)
    print(f"{'SELECTION OF CHAMPIONS (RULE 1-SD)':^80}")
    print("=" * 80)

    champions = []

    for arch in df_results['Architecture'].unique():
        df_arch = df_results[df_results['Architecture'] == arch].copy()
        if len(df_arch) == 0:
            continue

        max_f1 = df_arch['F1_Score'].max()
        std_f1 = df_arch['F1_Score'].std()
        if pd.isna(std_f1):
            std_f1 = 0.0

        threshold_1sd = max_f1 - std_f1
        candidates    = df_arch[df_arch['F1_Score'] >= threshold_1sd]
        champion      = candidates.sort_values(
            by=['Best_Epoch', 'F1_Score'], ascending=[True, False]
        ).iloc[0]

        champions.append(champion)

        print(f"\nArchitecture: {arch.upper()} (Seeds analyzed: {len(df_arch)})")
        print(f"  -> F1 Range   : Max = {max_f1:.4f} | Std = {std_f1:.4f}")
        print(f"  -> Zone 1-SD  : Models with F1 >= {threshold_1sd:.4f}")
        print(f"  -> Candidates : {len(candidates)} seed(s) entered the safe zone.")
        print(f"     CHAMPION   : Seed {int(champion['Seed'])} | F1: {champion['F1_Score']:.4f} | Training epochs: {int(champion['Best_Epoch'])}")

    return pd.DataFrame(champions)


def change_threshold(model_class, model_config, val_loader, experiment_name,
                     device='cpu', results_dir="./results_earlystopping",
                     verbose=False):
    """
    Re-compute optimal thresholds for all seeds of an experiment using max-F1.

    Loads each trained model, re-evaluates on the validation set, finds the
    threshold that maximises F1, saves the updated metrics CSV and a .npz
    with per-seed thresholds.

    Parameters
    ----------
    model_class     : class
    model_config    : dict — must contain 'model_params' and 'extra_params'
    val_loader      : DataLoader
    experiment_name : str
    device          : str or torch.device
    results_dir     : str
    verbose         : bool

    Returns
    -------
    pd.DataFrame — updated metrics (read from saved CSV)
    """
    df_metrics = pd.read_csv(f"{results_dir}/logs/{experiment_name}/run_metrics_{experiment_name}.csv")

    all_optimal_thresholds = {}
    filepath = None

    for exp_id in range(len(df_metrics)):
        df_exp = df_metrics.iloc[exp_id, :].copy()
        seed   = df_exp['seed']

        model_config['model_name'] = df_exp['model_name']
        model_config['type']       = df_exp['type']
        run_id                     = df_exp['extra_run_id']

        pos_weight  = torch.tensor([model_config['extra_params']['pos_weight']]).to(device)
        criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        is_temporal = 'GRU' in experiment_name or 'ST' in experiment_name

        if verbose:
            print(f"\nChanging threshold for: {model_config['model_name']}")
            print("\n OLD METRICS (MAX RECALL FOR PRECISION=0.9):")
            print(f" Precision: {df_exp['Precision']:.4f} | Recall: {df_exp['Recall']:.4f} | "
                  f"F1: {df_exp['F1']:.4f} | F2: {df_exp['F2']:.4f} | "
                  f"AUC-PR: {df_exp['AUC-PR']:.4f} | AUC-ROC: {df_exp['AUC-ROC']:.4f} | "
                  f"FPR: {df_exp['FPR']:.4f}")

        model_paths = glob.glob(os.path.join(results_dir, "saved_models", experiment_name, f"{run_id}_*.pth"))
        model_path  = model_paths[0] if model_paths else None
        if model_path is None:
            print(f"Error: No model found for run_id {run_id} in experiment {experiment_name}")
            continue

        model = model_class(**model_config['model_params']).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        _, y_true, y_probs = evaluate(model, val_loader, criterion, device, is_temporal)

        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        precisions = precisions[:-1]
        recalls    = recalls[:-1]
        f1_scores  = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_idx   = np.argmax(f1_scores)
        new_best_th = thresholds[best_idx]
        all_optimal_thresholds[f"seed_{seed}"] = new_best_th

        new_metrics = calculate_metrics_gnn(y_true, y_probs, prob_threshold=new_best_th)
        new_metrics['optimal_threshold'] = new_best_th

        if verbose:
            print("\n NEW METRICS (MAX F1):")
            print(f" Precision: {new_metrics['Precision']:.4f} | Recall: {new_metrics['Recall']:.4f} | "
                  f"F1: {new_metrics['F1']:.4f} | F2: {new_metrics['F2']:.4f} | "
                  f"AUC-PR: {new_metrics['AUC-PR']:.4f} | AUC-ROC: {new_metrics['AUC-ROC']:.4f} | "
                  f"FPR: {new_metrics['FPR']:.4f}\n")
            print("-" * 60)

        df_new_row = pd.DataFrame([new_metrics])
        for key, value in df_exp[['seed', 'run_id', 'model_name', 'type']].to_dict().items():
            df_new_row[key] = value

        filepath = f"{results_dir}/logs/{experiment_name}/metrics_newth_{experiment_name}.csv"
        if os.path.exists(filepath):
            df_new_row.to_csv(filepath, mode="a", header=False, index=False, float_format='%.16g')
        else:
            df_new_row.to_csv(filepath, mode="w", header=True, index=False, float_format='%.16g')

    np.savez(
        f"{results_dir}/logs/{experiment_name}/thresholds_{experiment_name}.npz",
        **all_optimal_thresholds
    )

    return pd.read_csv(filepath) if filepath and os.path.exists(filepath) else pd.DataFrame()


def compute_and_save_thresholds(model_class, model_config, val_loader,
                                experiment_name, device,
                                results_dir="./results_earlystopping"):
    """
    Re-evaluate each seed on the validation set, compute the max-F1 threshold,
    and save results to thresholds_<experiment_name>.npz.

    Use this for experiments trained before run_multiple_seeds saved the .npz
    automatically (i.e. experiments that have run_metrics_*.csv but no .npz).

    Parameters
    ----------
    model_class     : class
    model_config    : dict — must contain 'model_params' and 'extra_params'
    val_loader      : DataLoader
    experiment_name : str
    device          : torch.device or str
    results_dir     : str
    """
    csv_files = glob.glob(f"{results_dir}/logs/{experiment_name}/metrics_newth_{experiment_name}.csv")
    if not csv_files:
        csv_files = glob.glob(f"{results_dir}/logs/{experiment_name}/run_metrics_{experiment_name}.csv")
    if not csv_files:
        print(f"No metrics CSV found for {experiment_name}")
        return

    df_metrics = pd.read_csv(csv_files[0])
    is_temporal = 'GRU' in experiment_name or 'ST' in experiment_name
    pos_weight  = torch.tensor([model_config['extra_params']['pos_weight']]).to(device)
    criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    all_thresholds = {}

    for exp_id in range(len(df_metrics)):
        df_exp = df_metrics.iloc[exp_id, :].copy()
        seed   = int(df_exp['seed'])
        run_id = df_exp['run_id'] if 'run_id' in df_exp else df_exp.get('extra_run_id', None)

        model_config['model_name'] = df_exp['model_name']
        model_config['type']       = df_exp['type']

        model_paths = glob.glob(os.path.join(results_dir, "saved_models",
                                             experiment_name, f"{run_id}_*.pth"))
        model_path  = model_paths[0] if model_paths else None
        if model_path is None:
            print(f"  Error: No model found for run_id {run_id}")
            continue

        model = model_class(**model_config['model_params']).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        from sklearn.metrics import precision_recall_curve
        _, y_true, y_probs = evaluate(model, val_loader, criterion, device, is_temporal)
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        precisions = precisions[:-1]
        recalls    = recalls[:-1]
        f1_scores  = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        best_th    = thresholds[np.argmax(f1_scores)]

        all_thresholds[f"seed_{seed}"] = best_th
        print(f"  seed {seed}: threshold = {best_th:.16f}")

    npz_path = f"{results_dir}/logs/{experiment_name}/thresholds_{experiment_name}.npz"
    np.savez(npz_path, **all_thresholds)
    print(f"\nSaved: {npz_path}")


def _load_metrics_csv(results_gral_dir, experiment_name):
    """
    Load the metrics CSV for an experiment, falling back from metrics_newth_*.csv
    to run_metrics_*.csv.

    Also normalises column names so callers always see 'run_id' (not 'extra_run_id')
    and 'optimal_threshold' (set to NaN if absent — threshold comes from .npz).

    Returns
    -------
    pd.DataFrame
    """
    log_dir = os.path.join(results_gral_dir, "logs", experiment_name)
    path = os.path.join(log_dir, f"metrics_newth_{experiment_name}.csv")
    if not os.path.exists(path):
        path = os.path.join(log_dir, f"run_metrics_{experiment_name}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No metrics CSV found for {experiment_name} in {log_dir}"
        )

    df = pd.read_csv(path)

    # Normalise run_id column name (run_metrics uses 'extra_run_id')
    if 'run_id' not in df.columns and 'extra_run_id' in df.columns:
        df = df.rename(columns={'extra_run_id': 'run_id'})

    # Ensure optimal_threshold column exists (may be absent in run_metrics)
    if 'optimal_threshold' not in df.columns:
        df['optimal_threshold'] = float('nan')

    return df


def evaluate_test1(model_class, model_config, test_loader, experiment_name,
                   device, results_gral_dir, results_test_dirname=None,
                   verbose=False):
    """
    Evaluate all seeds of an experiment on test set 1.

    Loads each trained model, applies the saved optimal threshold, and computes
    metrics. Optionally saves results to a CSV file.

    If the output CSV already exists, it is loaded directly and inference is
    skipped — allows resuming interrupted evaluation runs without re-running
    completed experiments.

    Parameters
    ----------
    model_class          : class
    model_config         : dict
    test_loader          : DataLoader
    experiment_name      : str
    device               : torch.device or str
    results_gral_dir     : str
    results_test_dirname : str or None — if provided, saves CSV to this subdirectory
    verbose              : bool

    Returns
    -------
    pd.DataFrame — one row per seed
    """
    if results_test_dirname is not None:
        os.makedirs(os.path.join(results_gral_dir, results_test_dirname), exist_ok=True)
        cached = os.path.join(results_gral_dir, results_test_dirname,
                              f"test1_metrics_{experiment_name}.csv")
        if os.path.exists(cached):
            print(f"  Skipping inference — loaded from cache: {cached}")
            return pd.read_csv(cached)

    df_metrics     = _load_metrics_csv(results_gral_dir, experiment_name)
    opt_thresholds = np.load(f"{results_gral_dir}/logs/{experiment_name}/thresholds_{experiment_name}.npz")
    all_results    = []

    for exp_id in range(len(df_metrics)):
        df_exp         = df_metrics.iloc[exp_id, :].copy()
        seed           = df_exp['seed']
        opt_th         = opt_thresholds[f"seed_{seed}"]
        raw_model_name = df_exp['model_name']

        model_config['model_name'] = raw_model_name
        model_config['type']       = df_exp['type']
        run_id                     = df_exp['run_id']

        if verbose:
            print(f"\nEvaluating: {raw_model_name}")

        pos_weight  = torch.tensor([model_config['extra_params']['pos_weight']]).to(device)
        criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        is_temporal = 'GRU' in experiment_name or 'ST' in experiment_name

        model_paths = glob.glob(os.path.join(results_gral_dir, "saved_models", experiment_name, f"{run_id}_*.pth"))
        model_path  = model_paths[0] if model_paths else None
        if model_path is None:
            print(f"Error: No model found for run_id {run_id} in experiment {experiment_name}")
            continue

        model = model_class(**model_config['model_params']).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        _, y_true, y_probs = evaluate(model, test_loader, criterion, device, is_temporal)
        metrics = calculate_metrics_gnn(y_true, y_probs, prob_threshold=opt_th)

        if verbose:
            print(f" Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f} | "
                  f"F1: {metrics['F1']:.4f} | F2: {metrics['F2']:.4f} | "
                  f"AUC-PR: {metrics['AUC-PR']:.4f} | AUC-ROC: {metrics['AUC-ROC']:.4f} | "
                  f"FPR: {metrics['FPR']:.4f}\n")
            print("-" * 60)

        df_new_row = pd.DataFrame([metrics])
        for key, value in df_exp[['optimal_threshold', 'seed', 'run_id', 'model_name', 'type']].to_dict().items():
            df_new_row[key] = value
        all_results.append(df_new_row)

    if not all_results:
        return pd.DataFrame()

    final_df = pd.concat(all_results, ignore_index=True)
    if results_test_dirname is not None:
        filepath = f"{results_gral_dir}/{results_test_dirname}/test1_metrics_{experiment_name}.csv"
        final_df.to_csv(filepath, mode="w", header=True, index=False, float_format='%.16g')
    return final_df


def evaluate_test2(model_class, model_config, test_loader, experiment_name,
                   df_champions_filtered, device, results_gral_dir,
                   results_test_dirname=None, verbose=False):
    """
    Evaluate champion models on test set 2.

    Unlike evaluate_test1 (which loops over all seeds), this function
    evaluates only the champion(s) passed in df_champions_filtered
    (typically one per architecture, from apply_1sd_rule).

    Parameters
    ----------
    model_class          : class
    model_config         : dict
    test_loader          : DataLoader
    experiment_name      : str
    df_champions_filtered: pd.DataFrame — output of apply_1sd_rule filtered to one architecture
    device               : torch.device or str
    results_gral_dir     : str
    results_test_dirname : str or None — if provided, saves CSV to this subdirectory
    verbose              : bool

    Returns
    -------
    pd.DataFrame — one row per champion
    """
    if results_test_dirname is not None:
        os.makedirs(os.path.join(results_gral_dir, results_test_dirname), exist_ok=True)
        cached = os.path.join(results_gral_dir, results_test_dirname,
                              f"test2_metrics_{experiment_name}.csv")
        if os.path.exists(cached):
            print(f"  Skipping inference — loaded from cache: {cached}")
            return pd.read_csv(cached)

    df_metrics     = _load_metrics_csv(results_gral_dir, experiment_name)
    opt_thresholds = np.load(f"{results_gral_dir}/logs/{experiment_name}/thresholds_{experiment_name}.npz")
    all_results    = []

    for exp_id in range(len(df_champions_filtered)):
        seed   = df_champions_filtered['Seed'].iloc[exp_id]
        opt_th = opt_thresholds[f"seed_{seed}"]

        df_exp         = df_metrics[df_metrics['seed'] == seed].copy()
        raw_model_name = df_exp['model_name'].iloc[0]

        model_config['model_name'] = raw_model_name
        model_config['type']       = df_exp['type'].iloc[0]
        run_id                     = df_exp['run_id'].iloc[0]

        if verbose:
            print(f"\nEvaluating: {raw_model_name}")

        pos_weight  = torch.tensor([model_config['extra_params']['pos_weight']]).to(device)
        criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        is_temporal = 'GRU' in experiment_name or 'ST' in experiment_name

        model_paths = glob.glob(os.path.join(results_gral_dir, "saved_models", experiment_name, f"{run_id}_*.pth"))
        model_path  = model_paths[0] if model_paths else None
        if model_path is None:
            print(f"Error: No model found for run_id {run_id} in experiment {experiment_name}")
            continue

        model = model_class(**model_config['model_params']).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        _, y_true, y_probs = evaluate(model, test_loader, criterion, device, is_temporal)
        metrics = calculate_metrics_gnn(y_true, y_probs, prob_threshold=opt_th)

        if verbose:
            print(f" Precision: {metrics['Precision']:.4f} | Recall: {metrics['Recall']:.4f} | "
                  f"F1: {metrics['F1']:.4f} | F2: {metrics['F2']:.4f} | "
                  f"AUC-PR: {metrics['AUC-PR']:.4f} | AUC-ROC: {metrics['AUC-ROC']:.4f} | "
                  f"FPR: {metrics['FPR']:.4f}\n")
            print("-" * 60)

        df_new_row = pd.DataFrame([metrics])
        for key, value in df_exp[['optimal_threshold', 'seed', 'run_id', 'model_name', 'type']].iloc[0].to_dict().items():
            df_new_row[key] = value
        all_results.append(df_new_row)

    if not all_results:
        return pd.DataFrame()

    final_df = pd.concat(all_results, ignore_index=True)
    if results_test_dirname is not None:
        filepath = f"{results_gral_dir}/{results_test_dirname}/test2_metrics_{experiment_name}.csv"
        final_df.to_csv(filepath, mode="w", header=True, index=False, float_format='%.16g')
    return final_df


def generate_summary_table(df):
    """
    Print a mean ± std summary table grouped by model architecture.

    Parameters
    ----------
    df : pd.DataFrame — must contain a 'model' column and metric columns
    """
    possible     = ['Precision', 'Recall', 'F1', 'F2', 'AUC-PR', 'AUC-ROC', 'FPR', 'optimal_threshold']
    metrics      = [c for c in df.columns if c in possible]
    model_order  = ["Simple MLP", "Edge GRU", "Static GNN", "ST-GNN (Ours)"]

    df['model'] = pd.Categorical(df['model'], categories=model_order, ordered=True)
    summary     = df.groupby('model', observed=False)[metrics].agg(['mean', 'std'])

    print("\n" + "=" * 80)
    print(" SUMMARY TABLE (Mean ± Std)")
    print("=" * 80)
    print(summary.round(4))
