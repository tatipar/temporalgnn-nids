import numpy as np
import torch
from sklearn.metrics import precision_recall_curve


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
