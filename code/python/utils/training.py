"""Shared training, validation, and threshold-selection utilities."""

from __future__ import annotations

import copy
from dataclasses import dataclass
import gc
import json
import math
import os
import random
import re
import subprocess
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, precision_recall_curve

from .experiment import EarlyStopping, NumpyEncoder
from .metrics import calculate_metrics_gnn
from .temporal_state import IDENTITY_MODES, MEMORY_POLICIES
from .visualization import save_plots


SELECTION_METRIC = "average_precision"
THRESHOLD_STRATEGIES = frozenset({"max_f1", "constrained"})
TEMPORAL_POLICIES = MEMORY_POLICIES | {"lagged_identity_only"}


@dataclass(frozen=True)
class TrainingEpochResult:
    """Flow-level optimization diagnostics for one epoch."""

    loss_per_flow: float
    flows: int
    graph_windows: int
    optimizer_steps: int
    gradient_norm_mean: float | None
    gradient_norm_max: float | None
    temporal_diagnostics: dict[str, Any] | None


def forward_graph(model, data):
    """Run a model through the shared no-node-feature graph interface."""
    return model(
        edge_index=data.edge_index,
        edge_attr=data.edge_attr,
        num_nodes=data.num_nodes,
        global_node_ids=getattr(data, "global_node_ids", None),
        timestamp=getattr(data, "timestamp", None),
    )


def validate_temporal_configuration(
    model,
    temporal: bool,
    temporal_memory_policy: str | None = None,
) -> None:
    """Fail when an explicit run configuration disagrees with the model."""
    if not isinstance(temporal, bool):
        raise TypeError("temporal must be declared explicitly as a boolean.")
    declared = getattr(model, "temporal", None)
    if not isinstance(declared, bool):
        raise ValueError(
            f"Model {type(model).__name__} does not declare a boolean temporal capability."
        )
    if declared != temporal:
        raise ValueError(
            f"Temporal configuration mismatch for {type(model).__name__}: "
            f"config={temporal}, model={declared}."
        )
    declared_policy = getattr(model, "temporal_memory_policy", None)
    if temporal_memory_policy is not None:
        if not isinstance(declared_policy, str):
            raise ValueError(
                f"Model {type(model).__name__} does not declare a memory policy."
            )
        if declared_policy != temporal_memory_policy:
            raise ValueError(
                f"Memory-policy mismatch for {type(model).__name__}: "
                f"config={temporal_memory_policy!r}, model={declared_policy!r}."
            )
    if temporal:
        missing = [
            method for method in ("reset_memory", "detach_all_memory")
            if not callable(getattr(model, method, None))
        ]
        if missing:
            raise ValueError(
                f"Temporal model {type(model).__name__} lacks required memory methods: {missing}."
            )


def temporal_diagnostics(model) -> dict[str, Any] | None:
    """Return bounded diagnostics exposed by a stateful model."""
    getter = getattr(model, "get_temporal_diagnostics", None)
    return getter() if callable(getter) else None


def make_flow_criterion(pos_weight: float, device: str | torch.device):
    """Create the only loss reduction accepted by the flow-level protocol."""
    if not math.isfinite(pos_weight) or pos_weight <= 0:
        raise ValueError("pos_weight must be a positive finite number.")
    weight = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    return nn.BCEWithLogitsLoss(pos_weight=weight, reduction="sum")


def _validate_flow_criterion(criterion) -> None:
    if not isinstance(criterion, nn.BCEWithLogitsLoss):
        raise TypeError("Flow training requires torch.nn.BCEWithLogitsLoss.")
    if criterion.reduction != "sum":
        raise ValueError("Flow training requires BCEWithLogitsLoss(reduction='sum').")


def _gradient_norm(parameters) -> torch.Tensor:
    norms = [parameter.grad.detach().norm(2) for parameter in parameters if parameter.grad is not None]
    if not norms:
        return torch.tensor(0.0)
    return torch.stack(norms).norm(2)


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    *,
    temporal: bool,
    batch_steps: int = 10,
    max_grad_norm: float | None = None,
) -> TrainingEpochResult:
    """Train one epoch with TBPTT blocks weighted by their flow counts.

    Each graph contributes the sum of its per-flow losses. Before backward, the
    accumulated block loss is divided by the total number of flows in that
    block. Consequently, a 5,000-flow graph has the same per-flow influence as
    5,000 one-flow observations rather than the influence of one graph window.
    """
    _validate_flow_criterion(criterion)
    validate_temporal_configuration(model, temporal)
    if batch_steps <= 0:
        raise ValueError("batch_steps must be positive.")
    if max_grad_norm is not None and (
        not math.isfinite(max_grad_norm) or max_grad_norm <= 0
    ):
        raise ValueError("max_grad_norm must be a positive finite number or None.")

    model.train()
    if temporal:
        model.reset_memory()

    total_loss_sum = 0.0
    total_flows = 0
    graph_windows = 0
    optimizer_steps = 0
    gradient_norms: list[float] = []
    block_loss_sum = None
    block_flows = 0
    block_windows = 0

    def optimize_block() -> None:
        nonlocal block_loss_sum, block_flows, block_windows, optimizer_steps
        if block_loss_sum is None or block_flows == 0:
            return
        optimizer.zero_grad()
        (block_loss_sum / block_flows).backward()
        if max_grad_norm is None:
            norm = _gradient_norm(model.parameters())
        else:
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        gradient_norms.append(float(norm.detach().cpu()))
        optimizer.step()
        optimizer_steps += 1
        if temporal:
            model.detach_all_memory()
        block_loss_sum = None
        block_flows = 0
        block_windows = 0

    for data in loader:
        data = data.to(device)
        flow_count = int(data.edge_attr.shape[0])
        if flow_count == 0:
            continue
        logits = forward_graph(model, data).view(-1)
        targets = data.y.view(-1)
        if logits.numel() != flow_count or targets.numel() != flow_count:
            raise ValueError("Model logits and targets must contain one value per flow.")

        graph_loss_sum = criterion(logits, targets)
        block_loss_sum = graph_loss_sum if block_loss_sum is None else block_loss_sum + graph_loss_sum
        block_flows += flow_count
        block_windows += 1
        total_loss_sum += float(graph_loss_sum.detach().cpu())
        total_flows += flow_count
        graph_windows += 1

        if block_windows == batch_steps:
            optimize_block()

    optimize_block()
    mean_gradient_norm = float(np.mean(gradient_norms)) if gradient_norms else None
    max_gradient_norm = float(np.max(gradient_norms)) if gradient_norms else None
    return TrainingEpochResult(
        loss_per_flow=total_loss_sum / total_flows if total_flows else 0.0,
        flows=total_flows,
        graph_windows=graph_windows,
        optimizer_steps=optimizer_steps,
        gradient_norm_mean=mean_gradient_norm,
        gradient_norm_max=max_gradient_norm,
        temporal_diagnostics=temporal_diagnostics(model),
    )


@torch.no_grad()
def evaluate(model, loader, criterion, device, *, temporal: bool):
    """Return validation loss per flow, binary targets, and probabilities."""
    _validate_flow_criterion(criterion)
    validate_temporal_configuration(model, temporal)
    model.eval()
    if temporal:
        model.reset_memory()

    all_probs = []
    all_targets = []
    total_loss_sum = 0.0
    total_flows = 0
    for data in loader:
        data = data.to(device)
        flow_count = int(data.edge_attr.shape[0])
        if flow_count == 0:
            continue
        logits = forward_graph(model, data).view(-1)
        targets = data.y.view(-1)
        if logits.numel() != flow_count or targets.numel() != flow_count:
            raise ValueError("Model logits and targets must contain one value per flow.")
        total_loss_sum += float(criterion(logits, targets).detach().cpu())
        total_flows += flow_count
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
    return (
        total_loss_sum / total_flows if total_flows else 0.0,
        np.asarray(all_targets),
        np.asarray(all_probs),
    )


def select_optimal_threshold(
    y_true,
    y_probs,
    *,
    strategy: str,
    min_precision: float | None = None,
) -> tuple[float, str]:
    """Select a threshold using validation predictions only."""
    if strategy not in THRESHOLD_STRATEGIES:
        choices = ", ".join(sorted(THRESHOLD_STRATEGIES))
        raise ValueError(f"Unknown threshold strategy {strategy!r}. Expected: {choices}.")
    if min_precision is not None:
        try:
            min_precision = float(min_precision)
        except (TypeError, ValueError) as error:
            raise ValueError("min_precision must be a number.") from error
    if strategy == "max_f1" and min_precision is not None:
        raise ValueError("min_precision must be omitted when strategy='max_f1'.")
    if strategy == "constrained" and (
        min_precision is None
        or not math.isfinite(min_precision)
        or not 0 <= min_precision <= 1
    ):
        raise ValueError(
            "strategy='constrained' requires min_precision in the closed interval [0, 1]."
        )

    y_true = np.asarray(y_true)
    y_probs = np.asarray(y_probs)
    if y_true.size == 0 or y_probs.size != y_true.size:
        raise ValueError("Threshold selection requires aligned non-empty targets and probabilities.")
    if np.unique(y_true).size != 2:
        raise ValueError("Threshold selection requires both binary classes in validation.")

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
    precisions = precisions[:-1]
    recalls = recalls[:-1]
    if thresholds.size == 0:
        raise ValueError("Validation predictions did not produce any candidate thresholds.")
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-12)

    if strategy == "constrained":
        valid = np.flatnonzero(precisions >= min_precision)
        if valid.size:
            best_index = valid[np.argmax(recalls[valid])]
            description = f"max_recall_at_precision_gte_{min_precision:g}"
        else:
            best_index = int(np.argmax(f1_scores))
            description = "max_f1_fallback_constraint_not_met"
    else:
        best_index = int(np.argmax(f1_scores))
        description = "max_f1"
    return float(thresholds[best_index]), description


def find_optimal_threshold(
    model,
    loader,
    device,
    *,
    temporal: bool,
    strategy: str,
    min_precision: float | None = None,
):
    """Run validation inference and select an explicitly configured threshold."""
    validate_temporal_configuration(model, temporal)
    model.eval()
    if temporal:
        model.reset_memory()

    all_probs, all_targets = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if data.edge_attr.shape[0] == 0:
                continue
            logits = forward_graph(model, data).view(-1)
            all_probs.extend(torch.sigmoid(logits).cpu().numpy())
            all_targets.extend(data.y.view(-1).cpu().numpy())
    y_true = np.asarray(all_targets)
    y_probs = np.asarray(all_probs)
    threshold, description = select_optimal_threshold(
        y_true,
        y_probs,
        strategy=strategy,
        min_precision=min_precision,
    )
    print(f"\nOptimal threshold: {threshold:.6f} ({description})")
    return threshold, y_true, y_probs


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch random seeds."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def _git_code_version() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip())
        return f"{commit}-dirty" if dirty else commit
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _dataset_metadata(loader) -> dict[str, Any]:
    dataset = getattr(loader, "dataset", None)
    required_attributes = (
        "manifest", "manifest_path", "graph_root", "profile", "split",
        "expected_schema_hash", "edge_dim", "window_ms",
    )
    missing = [name for name in required_attributes if not hasattr(dataset, name)]
    if missing:
        raise ValueError(f"Loader dataset lacks reproducibility metadata: {missing}.")
    manifest = dataset.manifest
    profile = dataset.profile
    try:
        corrected_manifest_artifact = manifest["corrected_manifest"]
        corrected_manifest_path = corrected_manifest_artifact["path"]
        if _file_sha256(corrected_manifest_path) != corrected_manifest_artifact["sha256"]:
            raise ValueError("Corrected-data manifest hash does not match the graph manifest.")
        with open(corrected_manifest_path, "r", encoding="utf-8") as handle:
            corrected_manifest = json.load(handle)
        metadata = {
            "graph_root": str(dataset.graph_root),
            "graph_manifest_sha256": _file_sha256(dataset.manifest_path),
            "corrected_manifest_sha256": corrected_manifest_artifact["sha256"],
            "corrected_data_sha256": corrected_manifest["output"]["sha256"],
            "correction_rule_version": corrected_manifest["correction_rule_version"],
            "feature_profile": profile,
            "feature_schema_sha256": dataset.expected_schema_hash,
            "edge_dim": int(dataset.edge_dim),
            "window_ms": int(dataset.window_ms),
            "graph_collection_sha256": manifest["artifacts"]["graph_collections"][profile]["sha256"],
            "scaler_sha256": manifest["artifacts"]["scalers"][profile]["sha256"],
            "mapping_sha256": {
                name: artifact["sha256"]
                for name, artifact in manifest["artifacts"]["mappings"].items()
            },
            "checksum_index_sha256": manifest["artifacts"]["checksum_index"]["sha256"],
            "provenance_collection_sha256": manifest["artifacts"]["provenance_collection"]["sha256"],
            "input_csv_sha256": {
                artifact["path"]: artifact["sha256"]
                for artifact in manifest["input_csvs"]
            },
            "split": dataset.split,
        }
    except (KeyError, TypeError, OSError, json.JSONDecodeError) as error:
        raise ValueError("Graph manifest lacks required training artifact hashes.") from error
    return metadata


def _file_sha256(path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_model_config(model_config: dict[str, Any]) -> None:
    required = {
        "model_name", "type", "model_params", "extra_params", "data_params",
        "temporal", "temporal_memory_policy", "variant", "threshold",
        "selection_metric",
    }
    missing = sorted(required - set(model_config))
    if missing:
        raise ValueError(f"Model configuration is missing required fields: {missing}.")
    for key in ("model_params", "extra_params", "data_params"):
        if not isinstance(model_config[key], dict):
            raise TypeError(f"model_config.{key} must be a dictionary.")
    if not isinstance(model_config["temporal"], bool):
        raise TypeError("model_config.temporal must be a boolean.")
    if not isinstance(model_config["variant"], str) or not model_config["variant"]:
        raise ValueError("model_config.variant must be a non-empty string.")
    memory_policy = model_config["temporal_memory_policy"]
    if not isinstance(memory_policy, str) or not memory_policy:
        raise ValueError("model_config.temporal_memory_policy must be a non-empty string.")
    if memory_policy not in TEMPORAL_POLICIES | {"none"}:
        choices = ", ".join(sorted(TEMPORAL_POLICIES | {"none"}))
        raise ValueError(f"Unknown temporal memory policy {memory_policy!r}. Expected: {choices}.")
    if model_config["temporal"] and memory_policy == "none":
        raise ValueError("Temporal models must declare a temporal memory policy.")
    if not model_config["temporal"] and memory_policy != "none":
        raise ValueError("Non-temporal models must use temporal_memory_policy='none'.")
    if model_config["selection_metric"] != SELECTION_METRIC:
        raise ValueError(f"Checkpoint selection metric must be {SELECTION_METRIC!r}.")

    model_params = model_config["model_params"]
    configured_model_policy = model_params.get("memory_policy")
    if memory_policy in MEMORY_POLICIES:
        if configured_model_policy != memory_policy:
            raise ValueError(
                "model_params.memory_policy must match temporal_memory_policy."
            )
        time_scale_ms = model_params.get("time_scale_ms")
        if not isinstance(time_scale_ms, int) or time_scale_ms <= 0:
            raise ValueError(
                "Timestamp-aware memory requires a positive integer "
                "model_params.time_scale_ms."
            )
        if memory_policy == "exponential_decay":
            half_life = model_params.get("decay_half_life_windows")
            if (
                not isinstance(half_life, (int, float))
                or not math.isfinite(float(half_life))
                or float(half_life) <= 0
            ):
                raise ValueError(
                    "exponential_decay requires a positive finite "
                    "model_params.decay_half_life_windows."
                )
        if memory_policy == "hard_reset":
            max_gap_ms = model_params.get("max_gap_ms")
            if not isinstance(max_gap_ms, int) or max_gap_ms <= 0:
                raise ValueError(
                    "hard_reset requires a positive integer model_params.max_gap_ms."
                )
    elif configured_model_policy is not None and configured_model_policy != "none":
        raise ValueError(
            "model_params.memory_policy must be 'none' when recurrent memory is disabled."
        )

    identity_mode = model_params.get("identity_mode")
    if identity_mode is not None and identity_mode not in IDENTITY_MODES:
        choices = ", ".join(sorted(IDENTITY_MODES))
        raise ValueError(f"Unknown identity_mode {identity_mode!r}. Expected: {choices}.")
    if identity_mode == "lagged":
        window_ms = model_params.get("window_ms")
        if not isinstance(window_ms, int) or window_ms <= 0:
            raise ValueError(
                "Lagged identity requires a positive integer model_params.window_ms."
            )
    if memory_policy == "lagged_identity_only" and identity_mode != "lagged":
        raise ValueError(
            "lagged_identity_only requires model_params.identity_mode='lagged'."
        )
    for flag in ("use_memory", "use_topology", "use_direct_edge_attr"):
        if flag in model_params and not isinstance(model_params[flag], bool):
            raise TypeError(f"model_params.{flag} must be a boolean.")

    extra = model_config["extra_params"]
    for key in ("learning_rate", "pos_weight", "batch_steps"):
        if key not in extra:
            raise ValueError(f"model_config.extra_params is missing {key!r}.")
    learning_rate = float(extra["learning_rate"])
    pos_weight = float(extra["pos_weight"])
    if (
        not math.isfinite(learning_rate)
        or not math.isfinite(pos_weight)
        or learning_rate <= 0
        or pos_weight <= 0
    ):
        raise ValueError("learning_rate and pos_weight must be positive.")
    if not isinstance(extra["batch_steps"], int) or extra["batch_steps"] <= 0:
        raise ValueError("batch_steps must be positive.")
    if not model_config["data_params"].get("label_correction_version"):
        raise ValueError("data_params.label_correction_version must be recorded.")

    threshold = model_config["threshold"]
    if not isinstance(threshold, dict) or "strategy" not in threshold:
        raise ValueError("model_config.threshold must declare a strategy.")
    strategy = threshold["strategy"]
    min_precision = threshold.get("min_precision")
    if strategy == "max_f1" and min_precision is not None:
        raise ValueError("max_f1 threshold configuration must omit min_precision.")
    if strategy == "constrained" and (
        min_precision is None or not 0 <= float(min_precision) <= 1
    ):
        raise ValueError(
            "constrained threshold configuration requires min_precision in [0, 1]."
        )
    if strategy not in THRESHOLD_STRATEGIES:
        raise ValueError(f"Unknown threshold strategy {strategy!r}.")


def validate_model_instance_configuration(
    model, model_config: dict[str, Any]
) -> None:
    """Match architecture controls recorded in a run to the built model."""
    validate_temporal_configuration(
        model,
        model_config["temporal"],
        model_config["temporal_memory_policy"],
    )
    model_params = model_config["model_params"]
    for attribute in (
        "identity_mode",
        "use_memory",
        "use_topology",
        "use_direct_edge_attr",
    ):
        if not hasattr(model, attribute):
            continue
        if attribute not in model_params:
            raise ValueError(
                f"model_params.{attribute} must be recorded explicitly for "
                f"{type(model).__name__}."
            )
        if getattr(model, attribute) != model_params[attribute]:
            raise ValueError(
                f"Configured {attribute} does not match {type(model).__name__}."
            )


def _safe_run_component(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-") or "unknown"


def _configuration_sha256(configuration: dict[str, Any]) -> str:
    import hashlib

    encoded = json.dumps(
        configuration,
        cls=NumpyEncoder,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def run_multiple_seeds(
    model_class,
    model_config,
    train_loader,
    val_loader,
    manager,
    seeds=(42, 123, 777, 2024, 99),
    epochs=60,
    device="cpu",
    experiment_name="experiment",
    json_dir="./logs",
    plots_dir="./plots",
):
    """Train multiple seeds using AP-selected checkpoints and frozen metadata."""
    _validate_model_config(model_config)
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    train_metadata = _dataset_metadata(train_loader)
    val_metadata = _dataset_metadata(val_loader)
    comparable_keys = (
        "graph_root", "graph_manifest_sha256", "corrected_manifest_sha256",
        "corrected_data_sha256", "correction_rule_version", "feature_profile",
        "feature_schema_sha256", "edge_dim", "window_ms",
        "graph_collection_sha256", "scaler_sha256", "mapping_sha256",
        "checksum_index_sha256", "provenance_collection_sha256", "input_csv_sha256",
    )
    if any(train_metadata[key] != val_metadata[key] for key in comparable_keys):
        raise ValueError("Train and validation loaders do not share the same graph artifacts.")
    if train_metadata["split"] != "train" or val_metadata["split"] != "val":
        raise ValueError("run_multiple_seeds requires train and val dataset splits.")
    configured_edge_dim = model_config["model_params"].get("edge_dim")
    if configured_edge_dim != train_metadata["edge_dim"]:
        raise ValueError(
            "model_params.edge_dim does not match the selected feature schema."
        )
    for field in ("time_scale_ms", "window_ms"):
        configured_value = model_config["model_params"].get(field)
        if (
            configured_value is not None
            and configured_value != train_metadata["window_ms"]
        ):
            raise ValueError(
                f"model_params.{field} must match the graph-manifest window_ms."
            )
    configured_rule = model_config["data_params"]["label_correction_version"]
    if configured_rule != train_metadata["correction_rule_version"]:
        raise ValueError(
            "Configured label correction version does not match the corrected-data manifest."
        )

    output_dir = os.path.join(json_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(plots_dir, experiment_name), exist_ok=True)
    code_version = _git_code_version()
    all_thresholds = {}

    for seed in seeds:
        started = time.perf_counter()
        train_seconds = 0.0
        validation_seconds = 0.0
        threshold_seconds = 0.0
        tz = timezone(timedelta(hours=-3))
        run_timestamp = datetime.now(tz).strftime("%Y%m%d_%H%M%S")
        code_id = code_version[:8] + ("-dirty" if code_version.endswith("-dirty") else "")
        run_id = "_".join((
            train_metadata["graph_manifest_sha256"][:8],
            _safe_run_component(train_metadata["feature_profile"]),
            _safe_run_component(experiment_name),
            _safe_run_component(model_config["variant"]),
            f"seed{seed}",
            _safe_run_component(code_id),
            run_timestamp,
        ))
        print(f"\nRunning seed {seed} | run_id={run_id}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        set_seed(seed)

        run_config = copy.deepcopy(model_config)
        run_config["model_name"] = f"{experiment_name}_seed{seed}"
        run_config["selection_metric"] = SELECTION_METRIC
        artifact_metadata = {
            key: value for key, value in train_metadata.items() if key != "split"
        }
        run_config["data_params"].update(artifact_metadata)
        run_config["data_params"].update({
            "train_split": train_metadata["split"],
            "validation_split": val_metadata["split"],
        })
        run_config["extra_params"].update({
            "run_ts": run_timestamp,
            "run_id": run_id,
            "seed": seed,
            "code_version": code_version,
        })
        model = model_class(**run_config["model_params"]).to(device)
        validate_model_instance_configuration(model, run_config)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=float(run_config["extra_params"]["learning_rate"])
        )
        criterion = make_flow_criterion(
            float(run_config["extra_params"]["pos_weight"]), device
        )
        early_stopping = EarlyStopping(
            patience=int(run_config["extra_params"].get("patience", 10)),
            mode="max",
            min_delta=float(run_config["extra_params"].get("min_delta", 0.0001)),
            metric_name=SELECTION_METRIC,
        )

        train_loss_history = []
        validation_loss_history = []
        validation_ap_history = []
        gradient_norm_history = []
        train_temporal_history = []
        validation_temporal_history = []
        for epoch in range(epochs):
            tick = time.perf_counter()
            train_result = train_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                temporal=run_config["temporal"],
                batch_steps=int(run_config["extra_params"]["batch_steps"]),
                max_grad_norm=run_config["extra_params"].get("max_grad_norm"),
            )
            train_seconds += time.perf_counter() - tick

            tick = time.perf_counter()
            validation_loss, y_true, y_probs = evaluate(
                model, val_loader, criterion, device, temporal=run_config["temporal"]
            )
            validation_seconds += time.perf_counter() - tick
            validation_ap = average_precision_score(y_true, y_probs)
            train_loss_history.append(train_result.loss_per_flow)
            validation_loss_history.append(float(validation_loss))
            validation_ap_history.append(float(validation_ap))
            gradient_norm_history.append(train_result.gradient_norm_max)
            train_temporal_history.append(train_result.temporal_diagnostics)
            validation_temporal_history.append(temporal_diagnostics(model))
            improved = early_stopping(validation_ap, model, epoch)
            if improved or (epoch + 1) % 10 == 0:
                marker = " (*)" if improved else ""
                print(
                    f"Epoch {epoch + 1} | train loss/flow={train_result.loss_per_flow:.6f} "
                    f"| val loss/flow={validation_loss:.6f} | val AP={validation_ap:.6f}{marker}"
                )
            if early_stopping.early_stop:
                print(f"Early stopping after epoch {epoch + 1}.")
                break

        if early_stopping.best_model_state is None:
            raise RuntimeError("Early stopping did not capture a valid checkpoint.")
        model.load_state_dict(early_stopping.best_model_state)
        _, y_true_best, y_probs_best = evaluate(
            model, val_loader, criterion, device, temporal=run_config["temporal"]
        )
        final_temporal_diagnostics = temporal_diagnostics(model)

        threshold_config = run_config["threshold"]
        tick = time.perf_counter()
        best_threshold, threshold_description = select_optimal_threshold(
            y_true_best,
            y_probs_best,
            strategy=threshold_config["strategy"],
            min_precision=threshold_config.get("min_precision"),
        )
        threshold_seconds += time.perf_counter() - tick
        all_thresholds[f"seed_{seed}"] = best_threshold

        final_metrics = calculate_metrics_gnn(
            y_true_best, y_probs_best, prob_threshold=best_threshold
        )
        final_metrics.update({
            "optimal_threshold": best_threshold,
            "threshold_strategy": threshold_config["strategy"],
            "threshold_selection": threshold_description,
            "selection_metric": SELECTION_METRIC,
            "best_validation_ap": float(early_stopping.best_score),
            "best_epoch": int(early_stopping.best_epoch),
            "stopped_epoch": len(train_loss_history),
            "seed": seed,
            "run_ts": run_timestamp,
            "run_id": run_id,
            "time_total_sec": time.perf_counter() - started,
            "time_train_sec": train_seconds,
            "time_eval_sec": validation_seconds,
            "time_threshold_sec": threshold_seconds,
            "final_temporal_diagnostics": final_temporal_diagnostics,
        })
        run_config["prob_threshold"] = best_threshold
        run_config["extra_params"]["configuration_sha256"] = _configuration_sha256(
            run_config
        )
        save_plots(
            train_loss_history,
            validation_loss_history,
            y_true_best,
            y_probs_best,
            seed,
            experiment_name,
            run_timestamp,
            save_dir=os.path.join(plots_dir, experiment_name),
        )
        manager.log_experiment(
            model_config=run_config,
            metrics=final_metrics,
            model_object=model,
        )

        history_payload = {
            "experiment_name": experiment_name,
            "run_id": run_id,
            "configuration": run_config,
            "training": {
                "train_loss_per_flow": train_loss_history,
                "validation_loss_per_flow": validation_loss_history,
                "validation_average_precision": validation_ap_history,
                "gradient_norm_max": gradient_norm_history,
                "train_temporal_diagnostics": train_temporal_history,
                "validation_temporal_diagnostics": validation_temporal_history,
            },
            "early_stopping": {
                "metric": SELECTION_METRIC,
                "best_epoch": int(early_stopping.best_epoch),
                "best_score": float(early_stopping.best_score),
                "stopped_epoch": len(train_loss_history),
            },
            "final_validation_temporal_diagnostics": final_temporal_diagnostics,
        }
        history_path = os.path.join(output_dir, f"training_history_{run_id}.json")
        with open(history_path, "w", encoding="utf-8") as handle:
            json.dump(history_payload, handle, cls=NumpyEncoder, indent=2)
        print(f"Training history saved: {history_path}")

    threshold_path = os.path.join(output_dir, f"thresholds_{experiment_name}.npz")
    np.savez(threshold_path, **all_thresholds)
    print(f"\nThresholds saved: {threshold_path}")

    frame = pd.read_csv(manager.log_file)
    selected = frame[frame["model_name"].astype(str).str.contains(experiment_name, regex=False)]
    if selected.empty:
        return
    print("=" * 60)
    print(f"VALIDATION SUMMARY: {experiment_name}")
    print(f"AP: {selected['AUC-PR'].mean():.6f} ± {selected['AUC-PR'].std():.6f}")
    print(f"Recall: {selected['Recall'].mean():.6f} ± {selected['Recall'].std():.6f}")
    print("=" * 60)
