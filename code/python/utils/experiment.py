import os
import copy
import json
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar and array types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


class EarlyStopping:
    """
    Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience  : int   — epochs to wait after last improvement
    min_delta : float — minimum change to qualify as improvement
    mode      : 'max' for metrics that should increase (AUC, F1),
                'min' for metrics that should decrease (Loss)
    """

    def __init__(self, patience=10, min_delta=0.0001, mode='max'):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.counter   = 0
        self.early_stop = False
        self.best_score       = -np.inf if mode == 'max' else np.inf
        self.best_model_state = None
        self.best_epoch       = None

    def __call__(self, current_score, model, epoch=None):
        """
        Returns True if the metric improved, False otherwise.
        Sets self.early_stop = True when patience is exhausted.
        """
        if self.mode == 'max':
            improved = current_score > (self.best_score + self.min_delta)
        else:
            improved = current_score < (self.best_score - self.min_delta)

        if improved:
            self.best_score       = current_score
            self.best_model_state = copy.deepcopy(model.state_dict())
            self.best_epoch       = epoch
            self.counter          = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False


class ExperimentManager:
    """
    Log experiment results to CSV and optionally save model checkpoints.

    Usage
    -----
    manager = ExperimentManager(log_file="./logs/experiments_log.csv",
                                model_dir="./saved_models")
    manager.log_experiment(
        model_config={
            "model_name": "ST_GNN_Identity",
            "type": "temporal_gnn",
            "model_params": {"hidden_dim": 64, "dropout": 0.2},
            "prob_threshold": 0.5,
            "data_params": {"dataset": "CIC-IDS2018"},
            "extra_params": {"run_ts": "20240101_120000"},
        },
        metrics={"F1": 0.92, "AUC-PR": 0.95},
        model_object=model,
    )
    """

    def __init__(self,
                 log_file="./logs/experiments_log.csv",
                 model_dir="./saved_models"):
        self.log_file  = log_file
        self.model_dir = model_dir
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)

    def log_experiment(self,
                       model_config=None,
                       model_name=None,
                       params=None,
                       metrics=None,
                       model_object=None):
        """
        Record an experiment in CSV format and optionally save the model.

        model_config (recommended):
          - model_name   (str)
          - type         (str)
          - model_params (dict) — hyperparameters only
          - prob_threshold (float)
          - data_params  (dict) — optional
          - extra_params (dict) — optional, may include run_ts, run_id

        Legacy mode: pass model_name + params dict directly.
        metrics: dict of evaluation results
        model_object: PyTorch or sklearn model to save
        """
        if metrics is None:
            metrics = {}
        if params is None:
            params = {}

        tz  = timezone(timedelta(hours=-3))  # Argentina
        now = datetime.now(tz)

        if model_config is not None:
            mname        = model_config.get("model_name", model_name)
            mtype        = model_config.get("type", None)
            model_params = model_config.get("model_params", {})
            threshold    = model_config.get("prob_threshold", None)
            data_params  = model_config.get("data_params", {})
            extra_params = model_config.get("extra_params", {})
        else:
            mname        = model_name
            mtype        = params.get("type", None)
            threshold    = params.get("prob_threshold", None)
            model_params = params
            data_params  = {}
            extra_params = params

        run_ts = extra_params.get("run_ts", None)
        run_id = extra_params.get("run_id", None)

        if run_ts is not None:
            run_dt = datetime.strptime(run_ts, "%Y%m%d_%H%M%S").replace(tzinfo=tz)
        else:
            run_dt = now
            run_ts = run_dt.strftime("%Y%m%d_%H%M%S")

        entry = {
            "timestamp": run_dt.strftime("%Y-%m-%d %H:%M:%S"),
            "run_ts":    run_ts,
            "run_id":    run_id,
            "model_name": mname,
        }

        if mtype is not None:
            entry["type"] = mtype
        if threshold is not None:
            entry["prob_threshold"] = threshold

        entry.update({f"hp_{k}": v    for k, v in (model_params or {}).items()})
        entry.update({f"data_{k}": v  for k, v in (data_params  or {}).items()})
        entry.update({f"extra_{k}": v for k, v in {**extra_params, **params}.items()
                      if k not in ("type", "prob_threshold")})
        entry.update(metrics)

        df_new = pd.DataFrame([entry])
        if os.path.exists(self.log_file):
            df_new.to_csv(self.log_file, mode="a", header=False, index=False)
        else:
            df_new.to_csv(self.log_file, mode="w", header=True,  index=False)

        print(f"\nExperiment recorded in {self.log_file}")

        if model_object is not None:
            metric_key = "AUC-PR" if "AUC-PR" in metrics else ("F1" if "F1" in metrics else None)
            metric_val = metrics.get(metric_key, 0) if metric_key else 0
            safe_key   = metric_key or "metric"

            os.makedirs(self.model_dir, exist_ok=True)

            if run_id:
                filename = f"{run_id}_{safe_key}_{float(metric_val):.4f}"
            else:
                filename = f"{mname}_{run_ts}_{safe_key}_{float(metric_val):.4f}"

            filepath = os.path.join(self.model_dir, filename)

            if "sklearn" in str(type(model_object)):
                import joblib
                joblib.dump(model_object, f"{filepath}.joblib")
            else:
                import torch
                torch.save(model_object.state_dict(), f"{filepath}.pth")

            print(f"Saved model: {filepath}")
