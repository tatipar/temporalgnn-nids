import gc
import json
import os
import random
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve

from .experiment import EarlyStopping, NumpyEncoder
from .metrics import calculate_metrics_gnn
from .visualization import save_plots


def train_epoch(model, loader, optimizer, criterion, device,
                is_temporal=False, batch_steps=10):
    """
    Train one epoch using Truncated Backpropagation Through Time (TBPTT).

    Accumulates loss over `batch_steps` valid graph windows before calling
    optimizer.step(), then detaches temporal memory to cut the gradient graph.
    Supports all model variants via use_node_stats and is_temporal flags.

    Parameters
    ----------
    model       : GNN or baseline model
    loader      : DataLoader over graph windows (sequential order)
    optimizer   : torch optimizer
    criterion   : loss function (BCEWithLogitsLoss)
    device      : torch.device
    is_temporal : bool — True for ST-GNN and EdgeGRU models
    batch_steps : int  — number of valid windows per TBPTT step

    Returns
    -------
    float — average loss per valid window
    """
    model.train()
    if is_temporal and hasattr(model, 'reset_memory'):
        model.reset_memory()

    use_stats = getattr(model, 'use_node_stats', False)
    total_loss = 0
    steps = 0
    batch_loss = 0

    for batch_idx, data in enumerate(loader):
        data = data.to(device)
        if data.x.shape[0] == 0:
            continue

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

        batch_loss += criterion(out.view(-1), data.y)
        steps += 1

        is_last_batch = (batch_idx == len(loader) - 1)
        if (steps > 0 and steps % batch_steps == 0) or is_last_batch:
            if batch_loss > 0:
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()

                total_loss += batch_loss.item()
                batch_loss = 0

                if is_temporal:
                    model.detach_all_memory()

    return total_loss / steps if steps > 0 else 0.0


@torch.no_grad()
def evaluate(model, loader, criterion, device, is_temporal=False):
    """
    Run inference over a DataLoader and return loss + raw probabilities.

    Applies sigmoid to convert logits to probabilities. Callers are
    responsible for applying a threshold and computing metrics via
    calculate_metrics_gnn().

    Parameters
    ----------
    model       : GNN or baseline model
    loader      : DataLoader
    criterion   : loss function (BCEWithLogitsLoss)
    device      : torch.device
    is_temporal : bool

    Returns
    -------
    (avg_loss, y_true, y_probs) — all numpy arrays
    """
    model.eval()
    if is_temporal and hasattr(model, 'reset_memory'):
        model.reset_memory()

    use_stats = getattr(model, 'use_node_stats', False)
    all_probs = []
    all_trues = []
    total_loss = 0
    steps = 0

    for data in loader:
        data = data.to(device)
        if data.x.shape[0] == 0:
            continue

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

        total_loss += criterion(out.view(-1), data.y).item()
        all_probs.extend(torch.sigmoid(out.view(-1)).cpu().numpy())
        all_trues.extend(data.y.cpu().numpy())
        steps += 1

    avg_loss = total_loss / steps if steps > 0 else 0.0
    return avg_loss, np.array(all_trues), np.array(all_probs)


def find_optimal_threshold(model, loader, device,
                           is_temporal=False,
                           strategy='max_f1',
                           min_precision=0.90):
    """
    Find the optimal decision threshold on a validation set.

    Parameters
    ----------
    model         : trained model
    loader        : DataLoader (typically validation set)
    device        : torch.device
    is_temporal   : bool
    strategy      : 'max_f1'        — maximise F1 (default, used for final evaluation)
                    'constrained'   — maximise recall subject to precision >= min_precision,
                                      fallback to max-F1 if constraint is not met
    min_precision : float — precision floor for strategy='constrained' (default 0.90)

    Returns
    -------
    (best_threshold, y_true, y_probs) — threshold as float, labels and
    probabilities as numpy arrays (for downstream calibration plots)
    """
    model.eval()
    if is_temporal and hasattr(model, 'reset_memory'):
        model.reset_memory()

    use_stats = getattr(model, 'use_node_stats', False)
    all_probs, all_trues = [], []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if data.x.shape[0] == 0:
                continue

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

            all_probs.extend(torch.sigmoid(out.view(-1)).cpu().numpy())
            all_trues.extend(data.y.cpu().numpy())

    y_true  = np.array(all_trues)
    y_probs = np.array(all_probs)

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    precisions = precisions[:-1]
    recalls    = recalls[:-1]
    f1_scores  = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    if strategy == 'constrained':
        valid = np.where(precisions >= min_precision)[0]
        if len(valid) > 0:
            best_idx     = valid[np.argmax(recalls[valid])]
            strategy_msg = f"Max Recall @ Prec>={min_precision}"
        else:
            best_idx     = np.argmax(f1_scores)
            strategy_msg = "Max F1 (Fallback — precision constraint not met)"
    else:  # 'max_f1'
        best_idx     = np.argmax(f1_scores)
        strategy_msg = "Max F1"

    best_th = thresholds[best_idx]
    print(f"\nOptimal Threshold: {best_th:.4f} ({strategy_msg})")

    return best_th, y_true, y_probs


# Backwards-compatible alias
find_optimal_threshold_constrained = lambda *a, **kw: find_optimal_threshold(
    *a, strategy='constrained', **kw
)


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def run_multiple_seeds(model_class, model_config, train_loader, val_loader,
                       manager,
                       seeds=[42, 123, 777, 2024, 99],
                       epochs=60,
                       device='cpu',
                       experiment_name="Exp_Optimized",
                       json_dir="./logs",
                       plots_dir="./plots"):
    """
    Train a model across multiple random seeds and log results.

    For each seed: instantiates a fresh model, runs training with early stopping,
    finds the optimal threshold, computes metrics, saves plots and a JSON history,
    and logs to the ExperimentManager. Prints a summary table at the end.

    Parameters
    ----------
    model_class     : class — model constructor
    model_config    : dict  — must contain 'model_params' and 'extra_params'
                              (learning_rate, pos_weight, batch_steps)
    train_loader    : DataLoader
    val_loader      : DataLoader
    manager         : ExperimentManager
    seeds           : list of ints
    epochs          : int — max training epochs per seed
    device          : torch.device or str
    experiment_name : str — used for file names and log filtering
    json_dir        : str — directory for training history JSON files
    plots_dir       : str — directory for plot files
    """
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    print(f" Starting Multi-Seed Run: {experiment_name}")
    print(f"   Seeds: {seeds}")
    print("-" * 60)

    all_thresholds = {}

    for seed in seeds:
        t0_seed = time.perf_counter()
        t_train_total = 0.0
        t_eval_total = 0.0
        t_threshold_total = 0.0

        tz = timezone(timedelta(hours=-3))  # Argentina
        run_ts = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
        run_id = f"{experiment_name}_seed{seed}_{run_ts}"

        print(f"\nRunning seed {seed} | run_id={run_id}")

        exp_id = f"{experiment_name}_seed{seed}"
        print(f"\n{exp_id}")

        # Preventive memory cleaning before loading anything
        gc.collect()
        torch.cuda.empty_cache()

        set_seed(seed)

        # Update model_config
        model_config['model_name'] = exp_id
        model_config.setdefault("extra_params", {})
        model_config["extra_params"]["run_ts"] = run_ts
        model_config["extra_params"]["run_id"] = run_id

        # Instantiate model
        model = model_class(**model_config['model_params']).to(device)

        # Training setup
        optimizer = torch.optim.Adam(model.parameters(), lr=model_config['extra_params']['learning_rate'])
        pos_weight = torch.tensor([model_config['extra_params']['pos_weight']]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        early_stopping = EarlyStopping(patience=10, mode='max', min_delta=0.0001)

        train_hist, val_loss_hist, val_auc_hist = [], [], []

        is_temporal = 'GRU' in experiment_name or 'ST' in experiment_name

        for epoch in range(epochs):
            # --- TRAIN ---
            t0 = time.perf_counter()
            loss_train = train_epoch(model, train_loader, optimizer, criterion,
                                     device=device, is_temporal=is_temporal,
                                     batch_steps=model_config['extra_params']["batch_steps"])
            t_train_total += time.perf_counter() - t0

            # --- EVAL ---
            t0 = time.perf_counter()
            val_loss, y_true, y_probs = evaluate(model, val_loader, criterion, device, is_temporal=is_temporal)
            t_eval_total += time.perf_counter() - t0

            val_auc = average_precision_score(y_true, y_probs)

            train_hist.append(float(loss_train))
            val_loss_hist.append(float(val_loss))
            val_auc_hist.append(float(val_auc))

            improved = early_stopping(val_auc, model, epoch)

            if improved or (epoch + 1) % 10 == 0:
                mark = "(*)" if improved else ""
                print(f"   Ep {epoch+1} | Loss: {loss_train:.4f} | Val Loss: {val_loss:.4f} | Val AUC-PR: {val_auc:.4f} {mark}")

            if early_stopping.early_stop:
                print(f"   Early Stopping in epoch {epoch}")
                break

        model.load_state_dict(early_stopping.best_model_state)

        # Re-evaluate best model to get clean probs
        _, y_true_best, y_probs_best = evaluate(model, val_loader, criterion, device, is_temporal=is_temporal)

        # Find optimal threshold
        t0 = time.perf_counter()
        best_th, _, _ = find_optimal_threshold(
            model, val_loader, device, is_temporal, min_precision=0.90
        )
        all_thresholds[f"seed_{seed}"] = best_th
        t_threshold_total += time.perf_counter() - t0

        # Compute all metrics
        final_metrics = calculate_metrics_gnn(y_true_best, y_probs_best, prob_threshold=best_th)
        final_metrics['optimal_threshold'] = best_th
        final_metrics['stopped_epoch'] = len(train_hist)
        final_metrics['seed'] = seed
        final_metrics["run_ts"] = run_ts
        final_metrics["run_id"] = run_id
        final_metrics["time_total_sec"] = time.perf_counter() - t0_seed
        final_metrics["time_train_sec"] = t_train_total
        final_metrics["time_eval_sec"] = t_eval_total
        final_metrics["time_threshold_sec"] = t_threshold_total

        print(f"   Final: Precision={final_metrics['Precision']:.4f} | "
              f"Recall={final_metrics['Recall']:.4f} | F1={final_metrics['F1']:.4f} | "
              f"F2={final_metrics['F2']:.4f} | AUC-PR={final_metrics['AUC-PR']:.4f} | "
              f"FPR={final_metrics['FPR']:.5f}")

        save_plots(train_hist, val_loss_hist, y_true_best, y_probs_best,
                   seed, experiment_name, run_ts,
                   save_dir=os.path.join(plots_dir, experiment_name))

        manager.log_experiment(
            model_config=model_config,
            metrics=final_metrics,
            model_object=model
        )

        # Save training history JSON
        filename_json = os.path.join(
            json_dir, experiment_name,
            f"training_history_{experiment_name}_seed{seed}_{run_ts}.json"
        )

        history_payload = {
            "experiment_name": experiment_name,
            "seed": seed,
            "run_ts": run_ts,
            "run_id": run_id,
            "training": {
                "train_loss": train_hist,
                "val_loss": val_loss_hist,
                "val_aucpr": val_auc_hist
            },
            "early_stopping": {
                "best_epoch": int(early_stopping.best_epoch),
                "best_val_aucpr": float(early_stopping.best_score),
                "stopped_epoch": int(len(train_hist))
            }
        }

        try:
            with open(filename_json, 'w') as f:
                json.dump(history_payload, f, cls=NumpyEncoder, indent=4)
            print(f"Training history saved in: {filename_json}")
        except Exception as e:
            print(f"\nWarning: JSON could not be saved: {e}")

        print(f"\n End seed {seed}")
        print("-" * 60)

    # Save thresholds for all seeds to .npz (preserves full float precision)
    npz_path = os.path.join(json_dir, experiment_name,
                            f"thresholds_{experiment_name}.npz")
    np.savez(npz_path, **all_thresholds)
    print(f"\nThresholds saved: {npz_path}")

    # Print summary statistics across all seeds
    df = pd.read_csv(manager.log_file)
    df_exp = df[df["model_name"].str.contains(experiment_name)]

    if len(df_exp) == 0:
        print("No records found for this experiment.")
        return

    def mean_std(series):
        return series.mean(), series.std()

    auc_mean,   auc_std   = mean_std(df_exp["AUC-PR"])
    rec_mean,   rec_std   = mean_std(df_exp["Recall"])
    ttot_mean,  ttot_std  = mean_std(df_exp["time_total_sec"])
    ttrain_mean, ttrain_std = mean_std(df_exp["time_train_sec"])

    print("=" * 50)
    print(f" AVERAGE RESULT: {experiment_name}")
    print(f"AUC-PR:      {auc_mean:.4f} ± {auc_std:.4f}")
    print(f"Recall:      {rec_mean:.4f} ± {rec_std:.4f}")
    print(f"Total Time:  {ttot_mean:.2f} ± {ttot_std:.2f} sec")
    print(f"Train Time:  {ttrain_mean:.2f} ± {ttrain_std:.2f} sec")
    print("=" * 50)
