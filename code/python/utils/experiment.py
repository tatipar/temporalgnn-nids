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

    def __init__(self, patience=10, min_delta=0.0001, mode='max', metric_name=None):
        self.patience  = patience
        self.min_delta = min_delta
        self.mode      = mode
        self.metric_name = metric_name
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


def load_model_checkpoint(model, path, map_location="cpu"):
    """Load a self-describing checkpoint or a historical raw state dictionary."""
    import torch

    try:
        payload = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=map_location)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        model.load_state_dict(payload["model_state_dict"])
        return payload
    model.load_state_dict(payload)
    return {"format_version": 1, "model_state_dict": payload}


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
                 model_dir="./saved_models",
                 record_dir=None):
        self.log_file  = log_file
        self.model_dir = model_dir
        log_parent = os.path.dirname(log_file) or "."
        self.record_dir = record_dir or os.path.join(log_parent, "run_records")
        os.makedirs(log_parent, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(self.record_dir, exist_ok=True)

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

        def csv_value(value):
            if isinstance(value, (dict, list, tuple)):
                return json.dumps(value, cls=NumpyEncoder, sort_keys=True)
            return value

        entry.update({f"hp_{k}": csv_value(v) for k, v in (model_params or {}).items()})
        entry.update({f"data_{k}": csv_value(v) for k, v in (data_params or {}).items()})
        if model_config is not None:
            for key in (
                "variant",
                "temporal",
                "temporal_memory_policy",
                "selection_metric",
                "threshold",
            ):
                if key in model_config:
                    entry[f"protocol_{key}"] = csv_value(model_config[key])
        entry.update({
            f"extra_{k}": csv_value(v)
            for k, v in {**extra_params, **params}.items()
            if k not in ("type", "prob_threshold")
        })
        entry.update(metrics)

        df_new = pd.DataFrame([entry])
        if os.path.exists(self.log_file):
            df_new.to_csv(self.log_file, mode="a", header=False, index=False)
        else:
            df_new.to_csv(self.log_file, mode="w", header=True,  index=False)

        print(f"\nExperiment recorded in {self.log_file}")

        record_name = run_id or f"{mname}_{run_ts}"
        record_path = os.path.join(self.record_dir, f"{record_name}.json")
        run_record = {
            "format_version": 2,
            "run_id": run_id,
            "timestamp": entry["timestamp"],
            "configuration": copy.deepcopy(model_config) if model_config is not None else {
                "model_name": mname,
                "params": copy.deepcopy(params),
            },
            "metrics": copy.deepcopy(metrics),
        }
        with open(record_path, "w", encoding="utf-8") as handle:
            json.dump(run_record, handle, cls=NumpyEncoder, indent=2, sort_keys=True)
        print(f"Saved run record: {record_path}")

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
                checkpoint_path = f"{filepath}.joblib"
            else:
                import torch
                checkpoint_path = f"{filepath}.pth"
                torch.save({
                    "format_version": 2,
                    "model_state_dict": model_object.state_dict(),
                    "configuration": copy.deepcopy(model_config),
                    "metrics": copy.deepcopy(metrics),
                    "run_id": run_id,
                }, checkpoint_path)

            print(f"Saved model: {checkpoint_path}")
            return {"run_record": record_path, "checkpoint": checkpoint_path}
        return {"run_record": record_path, "checkpoint": None}
