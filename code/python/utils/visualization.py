import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi
from sklearn.calibration import calibration_curve


# ---------------------------------------------------------------------------
# Default style settings — can be overridden by callers
# ---------------------------------------------------------------------------

# Display names for raw model keys used in experiment logs
MODEL_NAME_MAPPING = {
    'SimpleMLP_BiasOn':                     'Simple MLP',
    'EdgeGRU_BiasOn':                       'Edge GRU',
    'StaticGNN_BiasOn_robust_Identity':     'Static GNN',
    'ST_GNN_BiasOn_robust_Identity_clone':  'ST-GNN (Ours)',
}

# Consistent color palette across all plots
MODEL_COLORS = {
    'Simple MLP':    '#95a5a6',  # Gray
    'Edge GRU':      '#3498db',  # Blue
    'Static GNN':    '#e67e22',  # Orange
    'ST-GNN (Ours)': '#2ecc71',  # Green
}

sns.set_theme(style='whitegrid')


# ---------------------------------------------------------------------------
# Multi-metric comparison plots
# ---------------------------------------------------------------------------

def plot_radar_chart(df, plots_dir='./plots',
                     colors=None, model_name_mapping=None):
    """
    Radar (spider) chart comparing models across six metrics.

    Parameters
    ----------
    df                 : DataFrame with columns [model, precision, recall,
                         f1, f2, auc-pr, auc-roc]
    plots_dir          : output directory
    colors             : dict model_name→color (defaults to MODEL_COLORS)
    model_name_mapping : dict raw_name→display_name (defaults to MODEL_NAME_MAPPING)
    """
    os.makedirs(plots_dir, exist_ok=True)
    if colors is None:
        colors = MODEL_COLORS
    if model_name_mapping is None:
        model_name_mapping = MODEL_NAME_MAPPING

    # Normalize column names to lowercase to handle both capitalize (from
    # calculate_metrics_gnn) and lowercase (from CSV) variants.
    df = df.copy()
    df.columns = [c.lower() if c != 'model' else c for c in df.columns]

    df_avg = df.groupby('model', observed=False).mean(numeric_only=True).reset_index()
    metrics_to_plot = ['precision', 'recall', 'f1', 'f2', 'auc-pr', 'auc-roc']
    categories = [m.upper() for m in metrics_to_plot]
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, color='black', size=12)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0],
               ['0.2', '0.4', '0.6', '0.8', '1.0'], color='grey', size=10)
    plt.ylim(0, 1)

    draw_order = list(colors.keys())

    for model_name in draw_order:
        if model_name not in df_avg['model'].values:
            continue
        values = df_avg[df_avg['model'] == model_name][metrics_to_plot].values.flatten().tolist()
        values += values[:1]
        col = colors.get(model_name, 'black')
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=model_name, color=col)
        ax.fill(angles, values, color=col, alpha=0.1)

    plt.title('Multidimensional Performance Analysis', size=16, y=1.05, weight='bold')
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/radar_chart_comparison.png', dpi=300)
    plt.show()


def plot_comparison(df, plots_dir='./plots', colors=None):
    """
    Grouped bar chart comparing models across six metrics.

    Parameters
    ----------
    df        : DataFrame with columns [model, precision, recall, f1, f2, auc-pr, auc-roc]
    plots_dir : output directory
    colors    : dict model_name→color (defaults to MODEL_COLORS)
    """
    os.makedirs(plots_dir, exist_ok=True)
    if colors is None:
        colors = MODEL_COLORS

    # Normalize column names to lowercase so this works whether columns come
    # from calculate_metrics_gnn (capitalized) or loaded from CSV (lowercase).
    df = df.copy()
    df.columns = [c.lower() if c != 'model' else c for c in df.columns]

    metrics_to_plot = ['precision', 'recall', 'f1', 'f2', 'auc-pr', 'auc-roc']
    available = [m for m in metrics_to_plot if m in df.columns]
    if not available:
        return

    df_melted = df.melt(id_vars=['model'], value_vars=available,
                        var_name='Metric', value_name='Value')
    df_melted['Metric'] = df_melted['Metric'].str.upper()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x='Metric', y='Value', hue='model',
                palette=colors, errorbar='sd', capsize=0.1)
    plt.title('Model Comparison (Optimal Threshold)', fontsize=14)
    plt.ylim(0.0, 1.1)
    plt.legend(title='Architecture')
    plt.ylabel('Score')
    plt.xlabel('')
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/metrics_comparison_barplot.png', dpi=300)
    plt.show()


# ---------------------------------------------------------------------------
# Training loss plots
# ---------------------------------------------------------------------------

def plot_training_losses(losses_data, plots_dir='./plots',
                         colors=None, model_name_mapping=None):
    """
    Plot mean ± std training loss curves for each model over multiple seeds.

    Parameters
    ----------
    losses_data        : dict  raw_model_name → {seed → {train: [...], val: [...]}}
    plots_dir          : output directory
    colors             : dict display_name→color
    model_name_mapping : dict raw_name→display_name
    """
    os.makedirs(plots_dir, exist_ok=True)
    if colors is None:
        colors = MODEL_COLORS
    if model_name_mapping is None:
        model_name_mapping = MODEL_NAME_MAPPING

    plt.figure(figsize=(12, 6))

    for raw_model_name, seed_dict in losses_data.items():
        if not seed_dict:
            continue
        clean_name = model_name_mapping.get(raw_model_name, raw_model_name)

        all_train = [data['train'] for data in seed_dict.values() if 'train' in data]
        if not all_train:
            continue

        min_len = min(len(lst) for lst in all_train)
        matrix  = np.array([lst[:min_len] for lst in all_train])
        mean_loss = np.mean(matrix, axis=0)
        std_loss  = np.std(matrix, axis=0)
        epochs    = range(1, len(mean_loss) + 1)

        color = colors.get(clean_name, '#333333')
        plt.plot(epochs, mean_loss, label=clean_name, color=color, linewidth=2)
        plt.fill_between(epochs,
                         mean_loss - std_loss, mean_loss + std_loss,
                         color=color, alpha=0.15)

    plt.title('Training Stability & Convergence (Mean ± Std)', fontsize=14)
    plt.xlabel('Epochs')
    plt.ylabel('Training Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{plots_dir}/training_losses_comparison.png', dpi=300)
    plt.show()


def plot_losses_train_val_per_model(losses_data, raw_model_name, plots_dir='./plots',
                                    model_name_mapping=None):
    """
    Plot train vs validation loss for all seeds of a single model,
    with a star marker at the best epoch (early stopping point).

    Parameters
    ----------
    losses_data     : dict  raw_model_name → {seed → {train, val, best_epoch}}
    raw_model_name  : str — key into losses_data
    plots_dir       : output directory
    model_name_mapping : dict raw_name→display_name
    """
    os.makedirs(plots_dir, exist_ok=True)
    if model_name_mapping is None:
        model_name_mapping = MODEL_NAME_MAPPING

    if raw_model_name not in losses_data or not losses_data[raw_model_name]:
        print(f'No data for {raw_model_name}')
        return

    clean_name = model_name_mapping.get(raw_model_name, raw_model_name)
    seed_data  = losses_data[raw_model_name]
    palette    = sns.color_palette('husl', len(seed_data))

    plt.figure(figsize=(12, 6))

    for i, (seed_name, metrics) in enumerate(seed_data.items()):
        if 'train' not in metrics or 'val' not in metrics:
            continue
        train_loss = metrics['train']
        val_loss   = metrics['val']
        best_epoch = metrics.get('best_epoch', None)
        color      = palette[i]

        epochs = range(len(train_loss))
        plt.plot(epochs, train_loss, linestyle='-',  color=color, alpha=0.4)
        plt.plot(epochs, val_loss,   linestyle='--', color=color, linewidth=2, label=seed_name)

        if best_epoch is not None and best_epoch < len(val_loss):
            plt.scatter(best_epoch, val_loss[best_epoch],
                        color=color, s=100, zorder=5, marker='*', edgecolor='black')

    plt.title(f'{clean_name} — Train vs Validation Loss (Stars = Best Epoch)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Seeds')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

    safe_name = clean_name.replace(' ', '_')
    out_dir   = os.path.join(plots_dir, raw_model_name)
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f'{out_dir}/{safe_name}_train_val_loss.png', dpi=300)
    plt.show()


# ---------------------------------------------------------------------------
# Per-run plots (loss + calibration)
# ---------------------------------------------------------------------------

def save_plots(train_loss, val_loss, y_true, y_probs, seed, exp_name, run_ts,
               save_dir='./plots'):
    """
    Save two diagnostic plots for a single training run:
    1. Train/val loss curve
    2. Calibration curve

    Parameters
    ----------
    train_loss : list of float
    val_loss   : list of float
    y_true     : array-like of int labels
    y_probs    : array-like of float probabilities
    seed       : int or str — used in filenames
    exp_name   : str — experiment identifier
    run_ts     : str — timestamp string used in filenames
    save_dir   : output directory
    """
    os.makedirs(save_dir, exist_ok=True)

    # Loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss,   label='Val Loss', linestyle='--')
    plt.title(f'{exp_name} — Loss Seed {seed}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/loss_{exp_name}_seed{seed}_{run_ts}.png')
    plt.close()

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_probs, n_bins=10)
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly Calibrated')
    plt.plot(prob_pred, prob_true, marker='.', label='Model')
    plt.title(f'{exp_name} — Calibration Seed {seed}')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{save_dir}/calib_{exp_name}_seed{seed}_{run_ts}.png')
    plt.close()
