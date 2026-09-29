"""Resumable one-seed graph training for the cAPTure development pilot."""

from __future__ import annotations

import gc
import hashlib
import json
import math
from pathlib import Path
import random
import resource
import shutil
import tempfile
import time
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
import torch
import torch.nn.functional as F

from .capture_data import sha256_file, write_json
from .capture_graph_dataset import (
    CaptureGraphCollection,
    CaptureGraphScenarioDataset,
)
from .capture_graph_training_binding import (
    EXPECTED_VARIANTS,
    load_graph_training_config,
)
from .models import (
    EdgeGRU_Baseline_NoX,
    SimpleMLP,
    StaticGNN_Identity,
    ST_GNN_Identity,
)
from .training import forward_graph, validate_temporal_configuration


REPORT_VERSION = 1
MODULE_PATH = Path(__file__)
PREDICTION_SCHEMA = pa.schema([
    pa.field("model", pa.string(), nullable=False),
    pa.field("fold", pa.string(), nullable=False),
    pa.field("scenario", pa.string(), nullable=False),
    pa.field("packet_id", pa.string(), nullable=False),
    pa.field("source_row_id", pa.int64(), nullable=False),
    pa.field("window_index", pa.int64(), nullable=False),
    pa.field("window_start_ns", pa.int64(), nullable=False),
    pa.field("window_end_ns", pa.int64(), nullable=False),
    pa.field("decision_time_ns", pa.int64(), nullable=False),
    pa.field("packet_timestamp_ns", pa.int64(), nullable=False),
    pa.field("binary_label", pa.int8(), nullable=False),
    pa.field("attack_step", pa.string(), nullable=True),
    pa.field("sequence_id", pa.string(), nullable=True),
    pa.field("score", pa.float32(), nullable=False),
])
PREPARED_EVALUATION_COLUMNS = (
    "packet_id",
    "source_row_id",
    "packet_timestamp_ns",
    "binary_label",
    "attack_step",
    "sequence_id",
)


def _load_json(path: str | Path, label: str) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required {label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label}: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"Expected {label} to contain a JSON object: {path}")
    return value


def _canonical_sha256(value: dict) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class CaptureSelectionGraphCollection:
    """Expose approved inner-preprocessed sequences through the shared loader."""

    def __init__(self, root: str | Path, source_manifest: dict) -> None:
        self.root = Path(root).expanduser().resolve()
        self.feature_dim = 103
        self.window_ms = 5000
        self.manifest = json.loads(json.dumps(source_manifest))
        for fold, fold_report in self.manifest["folds"].items():
            scenarios = list(fold_report["training_scenarios"])
            if not scenarios:
                raise ValueError(f"Selection fold {fold} has no training scenarios.")
            feature_names = None
            for scenario in scenarios:
                relative = fold_report["scenario_reports"][scenario]
                report = _load_json(
                    self.root / relative,
                    f"selection scenario report for fold {fold}/{scenario}",
                )
                if feature_names is None:
                    feature_names = report["feature_names"]
                elif report["feature_names"] != feature_names:
                    raise ValueError(f"Selection feature order changed within fold {fold}.")
            if len(feature_names) != self.feature_dim:
                raise ValueError(f"Selection feature dimension changed in fold {fold}.")
            fold_report["feature_names"] = feature_names

    def scenario_dataset(
        self,
        fold: str,
        scenario: str,
        *,
        graph_start: int,
        graph_stop: int,
    ) -> CaptureGraphScenarioDataset:
        if scenario not in self.manifest["folds"][fold]["training_scenarios"]:
            raise ValueError(f"Scenario is not a selection-training input: {fold}/{scenario}")
        return CaptureGraphScenarioDataset(
            self,
            fold,
            scenario,
            partition="train",
            graph_start=graph_start,
            graph_stop=graph_stop,
            verify_shard_checksums=False,
        )


def _load_authorized_job_context(
    *,
    binding_dir: Path,
    training_config_path: Path,
    job_id: str,
) -> tuple[dict, dict, dict, dict]:
    config = load_graph_training_config(training_config_path)
    report_path = binding_dir / "training_runner_binding_report.json"
    plan_path = binding_dir / "training_job_plan.json"
    authorization_path = binding_dir / "graph_training_authorization.json"
    report = _load_json(report_path, "training-runner binding report")
    plan = _load_json(plan_path, "training job plan")
    authorization = _load_json(authorization_path, "graph-training authorization")
    if (
        binding_dir.name != report.get("runner_binding_run_id")
        or report.get("status") != "review_required"
        or report.get("training_performed") is not False
        or report.get("held_out_scenarios_accessed") is not False
        or report.get("training_contract_sha256") != sha256_file(training_config_path)
        or plan.get("training_contract_sha256") != sha256_file(training_config_path)
        or authorization.get("approved") is not True
        or authorization.get("model_training_authorized") is not True
        or authorization.get("held_out_scenarios_authorized") is not False
        or authorization.get("training_runner_binding_report_sha256")
        != sha256_file(report_path)
        or authorization.get("training_job_plan_sha256") != sha256_file(plan_path)
        or authorization.get("selection_materialization_manifest_sha256")
        != report.get("selection_materialization_manifest_sha256")
    ):
        raise ValueError("The graph-training authorization or binding changed.")
    jobs = {item["job_id"]: item for item in plan.get("jobs", [])}
    if set(jobs) != set(plan.get("execution_order", [])):
        raise ValueError("The training job list differs from its execution order.")
    if job_id not in jobs:
        raise ValueError(f"Unknown training job: {job_id}")
    job = jobs[job_id]
    if job.get("model") not in EXPECTED_VARIANTS or job.get("fold") not in {"A", "B"}:
        raise ValueError(f"Invalid training job declaration: {job_id}")
    return config, report, plan, job


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _build_model(specification: dict) -> torch.nn.Module:
    class_name = specification["class"]
    parameters = dict(specification["parameters"])
    factories = {
        "SimpleMLP": SimpleMLP,
        "EdgeGRU_Baseline_NoX": EdgeGRU_Baseline_NoX,
        "StaticGNN_Identity": StaticGNN_Identity,
        "ST_GNN_Identity": ST_GNN_Identity,
    }
    if class_name not in factories:
        raise ValueError(f"Unknown model class: {class_name}")
    model = factories[class_name](**parameters)
    validate_temporal_configuration(
        model,
        temporal=bool(specification["temporal"]),
        temporal_memory_policy=specification["temporal_memory_policy"],
    )
    return model


def _optimizer(model: torch.nn.Module, optimization: dict) -> torch.optim.Optimizer:
    if optimization["optimizer"] != "adamw":
        raise ValueError("Only the frozen AdamW optimizer is supported.")
    return torch.optim.AdamW(
        model.parameters(),
        lr=float(optimization["learning_rate"]),
        weight_decay=float(optimization["weight_decay"]),
    )


def _reset_memory(model: torch.nn.Module, temporal: bool) -> None:
    if temporal:
        model.reset_memory()


def _detach_memory(model: torch.nn.Module, temporal: bool) -> None:
    if temporal:
        model.detach_all_memory()


def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    norms = [
        parameter.grad.detach().norm(2)
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not norms:
        return 0.0
    return float(torch.stack(norms).norm(2).detach().cpu())


def _train_epoch(
    *,
    model: torch.nn.Module,
    scenario_datasets: list[tuple[str, CaptureGraphScenarioDataset]],
    scenario_weights: dict[str, dict[str, float]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    temporal: bool,
    batch_steps: int,
    maximum_gradient_norm: float | None,
) -> dict:
    model.train()
    total_weighted_loss = 0.0
    total_weight = 0.0
    total_edges = 0
    total_graphs = 0
    optimizer_steps = 0
    gradient_norms = []

    for scenario, dataset in scenario_datasets:
        _reset_memory(model, temporal)
        weights = scenario_weights[scenario]
        normal_weight = float(weights["normal"])
        attack_weight = float(weights["attack"])
        block_loss = None
        block_weight = None
        block_graphs = 0

        def optimize_block() -> None:
            nonlocal block_loss, block_weight, block_graphs, optimizer_steps
            if block_loss is None:
                return
            optimizer.zero_grad(set_to_none=True)
            (block_loss / block_weight).backward()
            if maximum_gradient_norm is None:
                gradient_norm = _gradient_norm(model.parameters())
            else:
                gradient_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        float(maximum_gradient_norm),
                    ).detach().cpu()
                )
            gradient_norms.append(gradient_norm)
            optimizer.step()
            optimizer_steps += 1
            _detach_memory(model, temporal)
            block_loss = None
            block_weight = None
            block_graphs = 0

        for data in dataset:
            data = data.to(device)
            targets = data.y.view(-1)
            logits = forward_graph(model, data).view(-1)
            if logits.shape != targets.shape or not torch.isfinite(logits).all():
                raise ValueError(f"Invalid training logits: {scenario}")
            edge_weights = torch.where(
                targets > 0.5,
                targets.new_tensor(attack_weight),
                targets.new_tensor(normal_weight),
            )
            losses = F.binary_cross_entropy_with_logits(
                logits,
                targets,
                reduction="none",
            )
            graph_loss = (losses * edge_weights).sum()
            graph_weight = edge_weights.sum()
            block_loss = graph_loss if block_loss is None else block_loss + graph_loss
            block_weight = (
                graph_weight if block_weight is None else block_weight + graph_weight
            )
            block_graphs += 1
            total_weighted_loss += float(graph_loss.detach().cpu())
            total_weight += float(graph_weight.detach().cpu())
            total_edges += int(targets.numel())
            total_graphs += 1
            if block_graphs == batch_steps:
                optimize_block()
        optimize_block()

    return {
        "weighted_loss_per_unit_weight": total_weighted_loss / total_weight,
        "weight_sum": total_weight,
        "edges": total_edges,
        "graphs": total_graphs,
        "optimizer_steps": optimizer_steps,
        "gradient_norm_mean": float(np.mean(gradient_norms)),
        "gradient_norm_max": float(np.max(gradient_norms)),
    }


@torch.no_grad()
def _evaluate_selection(
    *,
    model: torch.nn.Module,
    scenario_datasets: list[tuple[str, CaptureGraphScenarioDataset]],
    device: torch.device,
    temporal: bool,
) -> dict:
    model.eval()
    scenario_metrics = {}
    for scenario, dataset in scenario_datasets:
        _reset_memory(model, temporal)
        targets = []
        probabilities = []
        loss_sum = 0.0
        edges = graphs = 0
        for data in dataset:
            data = data.to(device)
            graph_targets = data.y.view(-1)
            logits = forward_graph(model, data).view(-1)
            if logits.shape != graph_targets.shape or not torch.isfinite(logits).all():
                raise ValueError(f"Invalid validation logits: {scenario}")
            loss_sum += float(
                F.binary_cross_entropy_with_logits(
                    logits,
                    graph_targets,
                    reduction="sum",
                ).cpu()
            )
            targets.append(graph_targets.cpu().numpy().astype(np.uint8, copy=False))
            probabilities.append(
                torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False)
            )
            edges += int(graph_targets.numel())
            graphs += 1
        y_true = np.concatenate(targets)
        y_score = np.concatenate(probabilities)
        if y_true.size != edges or np.unique(y_true).size != 2:
            raise ValueError(f"Validation requires both classes: {scenario}")
        scenario_metrics[scenario] = {
            "average_precision": float(average_precision_score(y_true, y_score)),
            "binary_cross_entropy_per_edge": loss_sum / edges,
            "edges": edges,
            "graphs": graphs,
        }
    mean_ap = float(
        np.mean([item["average_precision"] for item in scenario_metrics.values()])
    )
    return {
        "unweighted_mean_scenario_average_precision": mean_ap,
        "scenarios": scenario_metrics,
    }


def _cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


def _rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_torch_save(value: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _load_resume_state(
    paths: list[Path],
    *,
    phase: str,
    job_contract_sha256: str,
    device: torch.device,
) -> tuple[dict | None, Path | None]:
    candidates = []
    for path in paths:
        if not path.is_file():
            continue
        state = torch.load(path, map_location=device, weights_only=False)
        if (
            state.get("phase") != phase
            or state.get("job_contract_sha256") != job_contract_sha256
            or state.get("training_code_sha256") != sha256_file(MODULE_PATH)
        ):
            raise ValueError(f"Incompatible resume state: {path}")
        candidates.append((int(state["completed_epoch"]), path.stat().st_mtime_ns, path, state))
    if not candidates:
        return None, None
    _, _, path, state = max(candidates, key=lambda item: (item[0], item[1]))
    return state, path


def _save_phase_state(
    state: dict,
    *,
    local_path: Path,
    durable_path: Path,
    durable_every_epochs: int,
    force_durable: bool,
) -> None:
    _atomic_torch_save(state, local_path)
    if force_durable or int(state["completed_epoch"]) % durable_every_epochs == 0:
        _atomic_torch_save(state, durable_path)


def _selection_datasets(
    *,
    selection_collection: CaptureSelectionGraphCollection,
    fold: str,
    fold_plan: dict,
) -> tuple[
    list[tuple[str, CaptureGraphScenarioDataset]],
    list[tuple[str, CaptureGraphScenarioDataset]],
]:
    training = []
    validation = []
    for scenario in fold_plan["training_scenarios"]:
        sequence = fold_plan["selection_sequences"][scenario]
        train_start, train_stop = sequence["inner_train_graph_range"]
        validation_start, validation_stop = sequence[
            "inner_validation_graph_range"
        ]
        training.append((
            scenario,
            selection_collection.scenario_dataset(
                fold,
                scenario,
                graph_start=int(train_start),
                graph_stop=int(train_stop),
            ),
        ))
        validation.append((
            scenario,
            selection_collection.scenario_dataset(
                fold,
                scenario,
                graph_start=int(validation_start),
                graph_stop=int(validation_stop),
            ),
        ))
    return training, validation


def _complete_refit_datasets(
    *,
    collection: CaptureGraphCollection,
    fold: str,
    scenarios: list[str],
) -> list[tuple[str, CaptureGraphScenarioDataset]]:
    return [
        (
            scenario,
            collection.scenario_dataset(
                fold,
                scenario,
                expected_partition="train",
                verify_shard_checksums=False,
            ),
        )
        for scenario in scenarios
    ]


def _run_selection_phase(
    *,
    model_specification: dict,
    fold_plan: dict,
    selection_collection: CaptureSelectionGraphCollection,
    fold: str,
    optimization: dict,
    checkpointing: dict,
    device: torch.device,
    job_contract_sha256: str,
    local_state_path: Path,
    durable_state_path: Path,
    result_path: Path,
) -> dict:
    if result_path.is_file():
        result = _load_json(result_path, "selection result")
        if result.get("job_contract_sha256") != job_contract_sha256:
            raise ValueError("The completed selection result has a different contract.")
        return result

    seed = int(optimization["seed"])
    _set_seed(seed)
    model = _build_model(model_specification).to(device)
    parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
    temporal = bool(model_specification["temporal"])
    optimizer = _optimizer(model, optimization)
    train_datasets, validation_datasets = _selection_datasets(
        selection_collection=selection_collection,
        fold=fold,
        fold_plan=fold_plan,
    )
    state, resumed_path = _load_resume_state(
        [local_state_path, durable_state_path],
        phase="selection",
        job_contract_sha256=job_contract_sha256,
        device=device,
    )
    if state is None:
        completed_epoch = 0
        best_epoch = 0
        best_metric = -math.inf
        best_model_state = None
        non_improving_epochs = 0
        history = []
        resume_count = 0
        accumulated_seconds = 0.0
    else:
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        _restore_rng_state(state["rng_state"])
        completed_epoch = int(state["completed_epoch"])
        best_epoch = int(state["best_epoch"])
        best_metric = float(state["best_metric"])
        best_model_state = state["best_model_state_dict"]
        non_improving_epochs = int(state["non_improving_epochs"])
        history = list(state["history"])
        resume_count = int(state["resume_count"]) + 1
        accumulated_seconds = float(state["accumulated_seconds"])
        print(
            f"Resumed selection at epoch {completed_epoch} from {resumed_path}.",
            flush=True,
        )

    invocation_started = time.perf_counter()
    maximum_epochs = int(checkpointing["maximum_epochs"])
    minimum_epochs = int(checkpointing["minimum_epochs"])
    patience = int(checkpointing["patience_epochs"])
    minimum_improvement = float(checkpointing["minimum_absolute_improvement"])
    stopped_early = (
        completed_epoch >= minimum_epochs
        and non_improving_epochs >= patience
    )
    for epoch in range(completed_epoch + 1, maximum_epochs + 1):
        if stopped_early:
            break
        epoch_started = time.perf_counter()
        training_metrics = _train_epoch(
            model=model,
            scenario_datasets=train_datasets,
            scenario_weights=fold_plan["selection_scenario_class_weights"],
            optimizer=optimizer,
            device=device,
            temporal=temporal,
            batch_steps=int(optimization["truncated_backpropagation_windows"]),
            maximum_gradient_norm=optimization["maximum_gradient_norm"],
        )
        validation_metrics = _evaluate_selection(
            model=model,
            scenario_datasets=validation_datasets,
            device=device,
            temporal=temporal,
        )
        metric = float(
            validation_metrics["unweighted_mean_scenario_average_precision"]
        )
        improved = metric > best_metric + minimum_improvement
        if improved:
            best_metric = metric
            best_epoch = epoch
            best_model_state = _cpu_state_dict(model)
            non_improving_epochs = 0
        else:
            non_improving_epochs += 1
        epoch_seconds = float(time.perf_counter() - epoch_started)
        history.append({
            "epoch": epoch,
            "training": training_metrics,
            "validation": validation_metrics,
            "improved": improved,
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "non_improving_epochs": non_improving_epochs,
            "wall_seconds": epoch_seconds,
        })
        print(
            f"Selection epoch {epoch:02d}: mean scenario AP={metric:.6f}; "
            f"best={best_metric:.6f} at epoch {best_epoch}; "
            f"{epoch_seconds:.1f}s{' *' if improved else ''}",
            flush=True,
        )
        should_stop = epoch >= minimum_epochs and non_improving_epochs >= patience
        phase_state = {
            "phase": "selection",
            "job_contract_sha256": job_contract_sha256,
            "training_code_sha256": sha256_file(MODULE_PATH),
            "completed_epoch": epoch,
            "model_state_dict": _cpu_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_epoch": best_epoch,
            "best_metric": best_metric,
            "best_model_state_dict": best_model_state,
            "non_improving_epochs": non_improving_epochs,
            "history": history,
            "resume_count": resume_count,
            "accumulated_seconds": (
                accumulated_seconds + time.perf_counter() - invocation_started
            ),
            "rng_state": _rng_state(),
        }
        _save_phase_state(
            phase_state,
            local_path=local_state_path,
            durable_path=durable_state_path,
            durable_every_epochs=int(
                optimization.get("durable_resume_checkpoint_every_epochs", 5)
            ),
            force_durable=should_stop or epoch == maximum_epochs,
        )
        if should_stop:
            stopped_early = True
            break

    if best_epoch <= 0 or best_model_state is None:
        raise ValueError("Selection did not produce a valid best epoch.")
    result = {
        "report_version": REPORT_VERSION,
        "phase": "selection",
        "job_contract_sha256": job_contract_sha256,
        "best_epoch_count": best_epoch,
        "best_unweighted_mean_scenario_average_precision": best_metric,
        "parameter_count": parameter_count,
        "stopped_after_epoch": int(history[-1]["epoch"]),
        "stopped_early": stopped_early,
        "selection_weights_discarded_after_epoch_selection": True,
        "selection_model_weights_reused_for_refit": False,
        "outer_validation_accessed": False,
        "threshold_selected": False,
        "resume_count": resume_count,
        "wall_seconds": float(
            accumulated_seconds + time.perf_counter() - invocation_started
        ),
        "history": history,
    }
    write_json(result_path, result)
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _run_refit_phase(
    *,
    model_specification: dict,
    fold_plan: dict,
    collection: CaptureGraphCollection,
    fold: str,
    optimization: dict,
    best_epoch_count: int,
    device: torch.device,
    job_contract_sha256: str,
    local_state_path: Path,
    durable_state_path: Path,
    checkpoint_path: Path,
    result_path: Path,
) -> dict:
    if result_path.is_file():
        result = _load_json(result_path, "final-refit result")
        if (
            result.get("job_contract_sha256") != job_contract_sha256
            or result.get("checkpoint_sha256") != sha256_file(checkpoint_path)
        ):
            raise ValueError("The completed final-refit result has a different contract.")
        return result

    seed = int(optimization["seed"])
    _set_seed(seed)
    model = _build_model(model_specification).to(device)
    temporal = bool(model_specification["temporal"])
    optimizer = _optimizer(model, optimization)
    datasets = _complete_refit_datasets(
        collection=collection,
        fold=fold,
        scenarios=fold_plan["training_scenarios"],
    )
    state, resumed_path = _load_resume_state(
        [local_state_path, durable_state_path],
        phase="final_refit",
        job_contract_sha256=job_contract_sha256,
        device=device,
    )
    if state is None:
        completed_epoch = 0
        history = []
        resume_count = 0
        accumulated_seconds = 0.0
    else:
        if int(state["target_epochs"]) != int(best_epoch_count):
            raise ValueError("The resumed refit has a different selected epoch count.")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        _restore_rng_state(state["rng_state"])
        completed_epoch = int(state["completed_epoch"])
        history = list(state["history"])
        resume_count = int(state["resume_count"]) + 1
        accumulated_seconds = float(state["accumulated_seconds"])
        print(
            f"Resumed final refit at epoch {completed_epoch} from {resumed_path}.",
            flush=True,
        )

    invocation_started = time.perf_counter()
    for epoch in range(completed_epoch + 1, int(best_epoch_count) + 1):
        epoch_started = time.perf_counter()
        training_metrics = _train_epoch(
            model=model,
            scenario_datasets=datasets,
            scenario_weights=fold_plan["complete_refit_scenario_class_weights"],
            optimizer=optimizer,
            device=device,
            temporal=temporal,
            batch_steps=int(optimization["truncated_backpropagation_windows"]),
            maximum_gradient_norm=optimization["maximum_gradient_norm"],
        )
        epoch_seconds = float(time.perf_counter() - epoch_started)
        history.append({
            "epoch": epoch,
            "training": training_metrics,
            "wall_seconds": epoch_seconds,
        })
        print(
            f"Final-refit epoch {epoch:02d}/{best_epoch_count}: "
            f"weighted loss={training_metrics['weighted_loss_per_unit_weight']:.6f}; "
            f"{epoch_seconds:.1f}s",
            flush=True,
        )
        phase_state = {
            "phase": "final_refit",
            "job_contract_sha256": job_contract_sha256,
            "training_code_sha256": sha256_file(MODULE_PATH),
            "target_epochs": int(best_epoch_count),
            "completed_epoch": epoch,
            "model_state_dict": _cpu_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "resume_count": resume_count,
            "accumulated_seconds": (
                accumulated_seconds + time.perf_counter() - invocation_started
            ),
            "rng_state": _rng_state(),
        }
        _save_phase_state(
            phase_state,
            local_path=local_state_path,
            durable_path=durable_state_path,
            durable_every_epochs=5,
            force_durable=epoch == int(best_epoch_count),
        )

    checkpoint = {
        "report_version": REPORT_VERSION,
        "job_contract_sha256": job_contract_sha256,
        "training_code_sha256": sha256_file(MODULE_PATH),
        "best_epoch_count_from_selection": int(best_epoch_count),
        "seed": seed,
        "model_specification": model_specification,
        "model_state_dict": _cpu_state_dict(model),
        "selection_weights_reused": False,
        "refit_used_complete_fold_training_scenarios": True,
        "outer_validation_accessed_during_refit": False,
    }
    _atomic_torch_save(checkpoint, checkpoint_path)
    result = {
        "report_version": REPORT_VERSION,
        "phase": "final_refit",
        "job_contract_sha256": job_contract_sha256,
        "epochs": int(best_epoch_count),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "early_stopping_used": False,
        "checkpoint_selection_used": False,
        "outer_validation_accessed_during_refit": False,
        "resume_count": resume_count,
        "wall_seconds": float(
            accumulated_seconds + time.perf_counter() - invocation_started
        ),
        "history": history,
    }
    write_json(result_path, result)
    del model, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _prepared_scenario_artifacts(
    *,
    prepared_run_dir: Path,
    scenario: str,
    expected_prepared_run_id: str,
) -> tuple[Path, dict]:
    if prepared_run_dir.name != expected_prepared_run_id:
        raise ValueError("The prepared-run ID changed.")
    status = _load_json(prepared_run_dir / "run_status.json", "prepared-run status")
    if status.get("complete") is not True or status.get("mode") != "FULL_DEV":
        raise ValueError("The prepared development run is incomplete.")
    directory = prepared_run_dir / scenario
    report_path = directory / "preparation_report.json"
    checksums = _load_json(
        directory / "artifact_checksums.json",
        f"prepared checksums for {scenario}",
    )
    report = _load_json(report_path, f"preparation report for {scenario}")
    if checksums.get(report_path.name) != sha256_file(report_path):
        raise ValueError(f"Prepared report checksum changed: {scenario}")
    candidates = [
        directory / name
        for name in checksums
        if name.endswith(".parquet") and name != report_path.name
    ]
    if len(candidates) != 1:
        raise ValueError(f"Expected one prepared packet Parquet for {scenario}.")
    packet_path = candidates[0]
    if (
        report.get("scenario") != scenario
        or report.get("output_sha256") != checksums.get(packet_path.name)
        or sha256_file(packet_path) != checksums.get(packet_path.name)
    ):
        raise ValueError(f"Prepared packet artifact changed: {scenario}")
    return packet_path, report


@torch.no_grad()
def _infer_to_memmaps(
    *,
    model: torch.nn.Module,
    dataset: CaptureGraphScenarioDataset,
    device: torch.device,
    temporal: bool,
    temporary_dir: Path,
) -> tuple[dict[str, np.memmap], dict]:
    edges = int(dataset.report["edges"])
    arrays = {
        "score": np.lib.format.open_memmap(
            temporary_dir / "score.npy", mode="w+", dtype=np.float32, shape=(edges,)
        ),
        "target": np.lib.format.open_memmap(
            temporary_dir / "target.npy", mode="w+", dtype=np.uint8, shape=(edges,)
        ),
        "window_index": np.lib.format.open_memmap(
            temporary_dir / "window_index.npy", mode="w+", dtype=np.int64, shape=(edges,)
        ),
        "window_start_ns": np.lib.format.open_memmap(
            temporary_dir / "window_start_ns.npy", mode="w+", dtype=np.int64, shape=(edges,)
        ),
        "window_end_ns": np.lib.format.open_memmap(
            temporary_dir / "window_end_ns.npy", mode="w+", dtype=np.int64, shape=(edges,)
        ),
    }
    model.eval()
    _reset_memory(model, temporal)
    warmup = dataset[0].to(device)
    warmup_logits = forward_graph(model, warmup).view(-1)
    if not torch.isfinite(warmup_logits).all():
        raise ValueError(f"Invalid warm-up logits: {dataset.scenario}")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    _reset_memory(model, temporal)
    del warmup, warmup_logits
    started = time.perf_counter()
    offset = 0
    for data in dataset:
        source_rows = data.source_row_id.numpy()
        size = len(source_rows)
        expected = np.arange(offset, offset + size, dtype=np.int64)
        if not np.array_equal(source_rows, expected):
            raise ValueError(f"OOF source-row order changed: {dataset.scenario}")
        data = data.to(device)
        logits = forward_graph(model, data).view(-1)
        if logits.numel() != size or not torch.isfinite(logits).all():
            raise ValueError(f"Invalid OOF logits: {dataset.scenario}")
        arrays["score"][offset : offset + size] = (
            torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False)
        )
        arrays["target"][offset : offset + size] = (
            data.y.cpu().numpy().astype(np.uint8, copy=False)
        )
        arrays["window_index"][offset : offset + size] = int(data.window_index)
        arrays["window_start_ns"][offset : offset + size] = int(data.window_start_ns)
        arrays["window_end_ns"][offset : offset + size] = int(data.window_end_ns)
        offset += size
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = float(time.perf_counter() - started)
    if offset != edges or not np.isfinite(arrays["score"]).all():
        raise ValueError(f"Incomplete OOF inference: {dataset.scenario}")
    for value in arrays.values():
        value.flush()
    return arrays, {
        "graphs": len(dataset),
        "edges": edges,
        "warmup_graphs_excluded_from_timing": 1,
        "pure_inference_seconds": inference_seconds,
        "seconds_per_million_packets": inference_seconds * 1_000_000 / edges,
        "average_precision": float(
            average_precision_score(arrays["target"], arrays["score"])
        ),
    }


def _arrow_string(values) -> pa.Array:
    return pa.array(values, type=pa.string(), from_pandas=True)


def _write_enriched_predictions(
    *,
    output_path: Path,
    packet_path: Path,
    arrays: dict[str, np.memmap],
    model_name: str,
    fold: str,
    scenario: str,
    batch_size: int = 100_000,
) -> None:
    writer = pq.ParquetWriter(
        output_path,
        PREDICTION_SCHEMA,
        compression="zstd",
        use_dictionary=["model", "fold", "scenario", "attack_step", "sequence_id"],
    )
    offset = 0
    try:
        parquet = pq.ParquetFile(packet_path)
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=list(PREPARED_EVALUATION_COLUMNS),
        ):
            frame = batch.to_pandas()
            size = len(frame)
            source_rows = frame["source_row_id"].to_numpy(dtype=np.int64, copy=False)
            expected = np.arange(offset, offset + size, dtype=np.int64)
            if not np.array_equal(source_rows, expected):
                raise ValueError(f"Prepared evaluation row order changed: {scenario}")
            prepared_targets = frame["binary_label"].to_numpy(
                dtype=np.uint8,
                copy=False,
            )
            if not np.array_equal(
                prepared_targets,
                np.asarray(arrays["target"][offset : offset + size]),
            ):
                raise ValueError(f"Graph/prepared OOF targets differ: {scenario}")
            table = pa.Table.from_arrays(
                [
                    pa.array([model_name] * size, type=pa.string()),
                    pa.array([fold] * size, type=pa.string()),
                    pa.array([scenario] * size, type=pa.string()),
                    _arrow_string(frame["packet_id"]),
                    pa.array(source_rows, type=pa.int64()),
                    pa.array(arrays["window_index"][offset : offset + size]),
                    pa.array(arrays["window_start_ns"][offset : offset + size]),
                    pa.array(arrays["window_end_ns"][offset : offset + size]),
                    pa.array(arrays["window_end_ns"][offset : offset + size]),
                    pa.array(
                        frame["packet_timestamp_ns"].to_numpy(
                            dtype=np.int64,
                            copy=False,
                        ),
                        type=pa.int64(),
                    ),
                    pa.array(prepared_targets.astype(np.int8, copy=False)),
                    _arrow_string(frame["attack_step"]),
                    _arrow_string(frame["sequence_id"]),
                    pa.array(arrays["score"][offset : offset + size]),
                ],
                schema=PREDICTION_SCHEMA,
            )
            writer.write_table(table)
            offset += size
    finally:
        writer.close()
    if offset != len(arrays["score"]):
        raise ValueError(f"Prepared prediction enrichment is incomplete: {scenario}")


def _evaluate_outer_fold(
    *,
    model_specification: dict,
    fold_plan: dict,
    collection: CaptureGraphCollection,
    fold: str,
    model_name: str,
    device: torch.device,
    checkpoint_path: Path,
    prepared_run_dir: Path,
    expected_prepared_run_id: str,
    job_contract_sha256: str,
    oof_dir: Path,
    local_work_root: Path,
    status_path: Path,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("job_contract_sha256") != job_contract_sha256:
        raise ValueError("The final checkpoint has a different job contract.")
    model = _build_model(model_specification).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    temporal = bool(model_specification["temporal"])
    oof_dir.mkdir(parents=True, exist_ok=True)
    completed = _load_json(status_path, "OOF status") if status_path.is_file() else {
        "job_contract_sha256": job_contract_sha256,
        "scenarios": {},
    }
    if completed.get("job_contract_sha256") != job_contract_sha256:
        raise ValueError("The OOF status has a different job contract.")

    scenario_metrics = {}
    for scenario in fold_plan["outer_validation_scenarios"]:
        output_path = oof_dir / f"{scenario}.parquet"
        existing = completed["scenarios"].get(scenario)
        if existing is not None:
            if existing.get("predictions_sha256") != sha256_file(output_path):
                raise ValueError(f"Completed OOF predictions changed: {scenario}")
            scenario_metrics[scenario] = existing
            print(f"Reusing completed OOF inference: {scenario}", flush=True)
            continue
        dataset = collection.scenario_dataset(
            fold,
            scenario,
            expected_partition="validation",
            verify_shard_checksums=False,
        )
        with tempfile.TemporaryDirectory(
            dir=local_work_root,
            prefix=f"oof_{model_name}_{fold}_{scenario}_",
        ) as temporary:
            arrays, metrics = _infer_to_memmaps(
                model=model,
                dataset=dataset,
                device=device,
                temporal=temporal,
                temporary_dir=Path(temporary),
            )
            packet_path, prepared_report = _prepared_scenario_artifacts(
                prepared_run_dir=prepared_run_dir,
                scenario=scenario,
                expected_prepared_run_id=expected_prepared_run_id,
            )
            if int(prepared_report["counts"]["packets"]) != metrics["edges"]:
                raise ValueError(f"Prepared/graph OOF edge counts differ: {scenario}")
            temporary_output = Path(temporary) / f"{scenario}.parquet"
            _write_enriched_predictions(
                output_path=temporary_output,
                packet_path=packet_path,
                arrays=arrays,
                model_name=model_name,
                fold=fold,
                scenario=scenario,
            )
            shutil.copyfile(temporary_output, output_path)
            if sha256_file(output_path) != sha256_file(temporary_output):
                raise IOError(f"Durable OOF prediction checksum mismatch: {scenario}")
        metrics.update({
            "predictions": f"oof/{scenario}.parquet",
            "predictions_sha256": sha256_file(output_path),
            "threshold_selected": False,
        })
        completed["scenarios"][scenario] = metrics
        write_json(status_path, completed)
        scenario_metrics[scenario] = metrics
        print(
            f"Completed OOF inference for {scenario}: "
            f"AP={metrics['average_precision']:.6f}, "
            f"{metrics['pure_inference_seconds']:.1f}s",
            flush=True,
        )

    return {
        "scenarios": scenario_metrics,
        "unweighted_mean_scenario_average_precision": float(np.mean([
            item["average_precision"] for item in scenario_metrics.values()
        ])),
        "total_edges": sum(int(item["edges"]) for item in scenario_metrics.values()),
        "pure_inference_seconds": sum(
            float(item["pure_inference_seconds"])
            for item in scenario_metrics.values()
        ),
        "threshold_selected": False,
    }


def _resource_summary(device: torch.device, started: float) -> dict:
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    result = {
        "wall_seconds": float(time.perf_counter() - started),
        "peak_host_rss_bytes": peak_rss,
        "peak_host_rss_gib": peak_rss / 1024**3,
    }
    if device.type == "cuda":
        result.update({
            "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3,
            "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        })
    return result


def run_capture_graph_training_job(
    *,
    job_id: str,
    binding_dir: str | Path,
    collection: CaptureGraphCollection,
    selection_root: str | Path,
    selection_manifest: dict,
    prepared_run_dir: str | Path,
    training_config_path: str | Path,
    durable_output_root: str | Path,
    local_work_root: str | Path,
) -> dict:
    """Run or resume one authorized model/fold job through OOF prediction."""
    started = time.perf_counter()
    binding_dir = Path(binding_dir).expanduser().resolve()
    selection_root = Path(selection_root).expanduser().resolve()
    prepared_run_dir = Path(prepared_run_dir).expanduser().resolve()
    training_config_path = Path(training_config_path).expanduser().resolve()
    durable_output_root = Path(durable_output_root).expanduser().resolve()
    local_work_root = Path(local_work_root).expanduser().resolve()
    config, binding_report, plan, job = _load_authorized_job_context(
        binding_dir=binding_dir,
        training_config_path=training_config_path,
        job_id=job_id,
    )
    bindings = config["bindings"]
    if (
        collection.expected_run_id != bindings["graph_materialization_run_id"]
        or not collection.artifact_checksums_verified
        or selection_root.name != bindings["selection_materialization_run_id"]
        or sha256_file(
            selection_root / "selection_materialization_manifest.json"
        ) != bindings["selection_materialization_manifest_sha256"]
        or prepared_run_dir.name != bindings["prepared_run_id"]
    ):
        raise ValueError("Training inputs differ from the authorized binding.")
    if binding_report["selection_materialization_manifest_sha256"] != bindings[
        "selection_materialization_manifest_sha256"
    ]:
        raise ValueError("The binding and training selection hashes differ.")

    model_name = job["model"]
    fold = job["fold"]
    model_specification = plan["model_specifications"][model_name]
    fold_plan = plan["fold_plans"][fold]
    job_contract = {
        "job_id": job_id,
        "model": model_name,
        "fold": fold,
        "binding_run_id": binding_dir.name,
        "binding_report_sha256": sha256_file(
            binding_dir / "training_runner_binding_report.json"
        ),
        "training_authorization_sha256": sha256_file(
            binding_dir / "graph_training_authorization.json"
        ),
        "training_job_plan_sha256": sha256_file(
            binding_dir / "training_job_plan.json"
        ),
        "training_contract_sha256": sha256_file(training_config_path),
        "training_code_sha256": sha256_file(MODULE_PATH),
        "graph_materialization_manifest_sha256": sha256_file(
            collection.manifest_path
        ),
        "selection_materialization_manifest_sha256": sha256_file(
            selection_root / "selection_materialization_manifest.json"
        ),
        "model_specification": model_specification,
        "optimization": plan["optimization"],
        "checkpoint_selection": plan["checkpoint_selection"],
        "final_refit": plan["final_refit"],
        "sequence": plan["sequence"],
        "fold_plan": fold_plan,
    }
    job_contract_sha256 = _canonical_sha256(job_contract)
    job_dir = durable_output_root / job_id
    local_job_dir = local_work_root / binding_dir.name / job_id
    local_job_dir.mkdir(parents=True, exist_ok=True)

    completion_path = job_dir / "completion.json"
    if completion_path.is_file():
        completion = _load_json(completion_path, "training-job completion")
        if completion.get("job_contract_sha256") != job_contract_sha256:
            raise ValueError("The completed training job has a different contract.")
        for relative, expected in completion["artifact_checksums"].items():
            if sha256_file(job_dir / relative) != expected:
                raise ValueError(f"Completed training artifact changed: {relative}")
        print(f"Reusing completed immutable training job: {job_id}", flush=True)
        return completion

    if job_dir.exists():
        existing_contract = _load_json(job_dir / "job_contract.json", "job contract")
        if existing_contract != job_contract:
            raise FileExistsError(
                "The existing job directory has a different contract."
            )
    else:
        job_dir.mkdir(parents=True)
        write_json(job_dir / "job_contract.json", job_contract)
    local_work_root.mkdir(parents=True, exist_ok=True)
    device_name = config["optimization"]["device"]
    if device_name != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The frozen pilot requires an available CUDA device.")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)

    selection_collection = CaptureSelectionGraphCollection(
        selection_root,
        selection_manifest,
    )
    persistence = config["persistence"]
    optimization = dict(plan["optimization"])
    optimization["durable_resume_checkpoint_every_epochs"] = int(
        persistence["durable_resume_checkpoint_every_epochs"]
    )
    selection_result = _run_selection_phase(
        model_specification=model_specification,
        fold_plan=fold_plan,
        selection_collection=selection_collection,
        fold=fold,
        optimization=optimization,
        checkpointing=plan["checkpoint_selection"],
        device=device,
        job_contract_sha256=job_contract_sha256,
        local_state_path=local_job_dir / "selection_resume.pt",
        durable_state_path=job_dir / "selection_resume.pt",
        result_path=job_dir / "selection_result.json",
    )
    refit_result = _run_refit_phase(
        model_specification=model_specification,
        fold_plan=fold_plan,
        collection=collection,
        fold=fold,
        optimization=optimization,
        best_epoch_count=int(selection_result["best_epoch_count"]),
        device=device,
        job_contract_sha256=job_contract_sha256,
        local_state_path=local_job_dir / "final_refit_resume.pt",
        durable_state_path=job_dir / "final_refit_resume.pt",
        checkpoint_path=job_dir / "final_model.pt",
        result_path=job_dir / "final_refit_result.json",
    )
    outer_metrics = _evaluate_outer_fold(
        model_specification=model_specification,
        fold_plan=fold_plan,
        collection=collection,
        fold=fold,
        model_name=model_name,
        device=device,
        checkpoint_path=job_dir / "final_model.pt",
        prepared_run_dir=prepared_run_dir,
        expected_prepared_run_id=bindings["prepared_run_id"],
        job_contract_sha256=job_contract_sha256,
        oof_dir=job_dir / "oof",
        local_work_root=local_work_root,
        status_path=job_dir / "oof_status.json",
    )
    metrics = {
        "report_version": REPORT_VERSION,
        "job_id": job_id,
        "model": model_name,
        "fold": fold,
        "job_contract_sha256": job_contract_sha256,
        "best_epoch_count": selection_result["best_epoch_count"],
        "selection_best_mean_scenario_average_precision": selection_result[
            "best_unweighted_mean_scenario_average_precision"
        ],
        "parameter_count": selection_result["parameter_count"],
        "final_checkpoint_bytes": (job_dir / "final_model.pt").stat().st_size,
        "outer_oof": outer_metrics,
        "threshold_selected": False,
        "held_out_scenarios_accessed": False,
    }
    write_json(job_dir / "metrics.json", metrics)
    resources = _resource_summary(device, started)
    write_json(job_dir / "resource_usage.json", resources)
    artifact_paths = [
        "job_contract.json",
        "selection_result.json",
        "final_refit_result.json",
        "final_model.pt",
        "oof_status.json",
        "metrics.json",
        "resource_usage.json",
        *[
            f"oof/{scenario}.parquet"
            for scenario in fold_plan["outer_validation_scenarios"]
        ],
    ]
    completion = {
        "report_version": REPORT_VERSION,
        "status": "complete",
        "job_id": job_id,
        "model": model_name,
        "fold": fold,
        "job_contract_sha256": job_contract_sha256,
        "best_epoch_count": selection_result["best_epoch_count"],
        "parameter_count": selection_result["parameter_count"],
        "final_checkpoint_bytes": (job_dir / "final_model.pt").stat().st_size,
        "outer_oof_unweighted_mean_scenario_average_precision": outer_metrics[
            "unweighted_mean_scenario_average_precision"
        ],
        "artifact_checksums": {
            relative: sha256_file(job_dir / relative) for relative in artifact_paths
        },
        "model_training_performed": True,
        "outer_validation_evaluated_after_refit": True,
        "threshold_selected": False,
        "held_out_scenarios_accessed": False,
        "resource_usage": resources,
    }
    write_json(completion_path, completion)
    print(
        f"Completed {job_id}: best epoch={completion['best_epoch_count']}, "
        f"outer mean scenario AP="
        f"{completion['outer_oof_unweighted_mean_scenario_average_precision']:.6f}",
        flush=True,
    )
    return completion
