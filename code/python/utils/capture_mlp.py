"""One-seed packet MLP screening for the cAPTure development folds."""

from __future__ import annotations

from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .capture_data import load_manifest, sha256_file, write_json
from .capture_feature_profile import (
    load_prepared_full_dev,
    load_preprocessing_schema,
    preprocessing_schema_sha256,
)
from .capture_oof_operational import (
    _macro_metrics,
    _packet_counts_by_threshold,
    evaluate_scenario,
    operational_budgets,
    select_threshold,
    summarize_scenario_oof,
)
from .capture_xgb_p import (
    OOF_COLUMNS,
    OOF_SCHEMA,
    _load_fold_preprocessor,
    _scenario_metrics,
    _validated_labels,
    scenario_class_weights,
    window_coordinates,
)
from .capture_xgb_p_t import (
    _load_context_run,
    _paired_batches,
    context_features_for_variant,
    fit_context_scaler,
    transform_context,
)


REPORT_VERSION = 1
CONFIGURATION_NAME = "mlp_screen_v1"
VARIANTS = ("packet", "history")
MODEL_NAMES = {"packet": "mlp_p", "history": "mlp_history"}


def mlp_configuration(manifest: dict) -> dict:
    """Load and validate the frozen one-seed MLP screening configuration."""
    configuration = dict(
        manifest["training"]["hyperparameter_search"][CONFIGURATION_NAME]
    )
    expected = {
        "status": "frozen_for_one_seed_development_screen",
        "variants": ["packet", "history"],
        "architecture": "simple_mlp",
        "hidden_layers": 2,
        "normalization": "layer_norm",
        "activation": "relu",
        "optimizer": "adamw",
        "checkpoint_selection": (
            "fixed_final_epoch_no_outer_validation_early_stopping"
        ),
        "loss": (
            "per_packet_binary_cross_entropy_times_fold_local_scenario_class_weight"
        ),
        "loss_normalization": "sum_weighted_loss_divided_by_sum_batch_weights",
        "packet_variant_features": "same_103_fold_preprocessed_features_as_xgb_p",
        "history_variant_features": (
            "packet_variant_plus_six_preceding_30_second_features"
        ),
        "primary_decision_time": "window_end",
        "secondary_native_decision_time": "packet_timestamp",
        "device": "cuda",
    }
    for name, value in expected.items():
        if configuration.get(name) != value:
            raise ValueError(f"The MLP screening configuration changed: {name}")
    positive_integer_fields = (
        "hidden_dim",
        "fixed_epochs",
        "training_batch_size",
        "shuffle_block_rows",
        "materialization_batch_size",
        "inference_batch_size",
        "seed",
    )
    for name in positive_integer_fields:
        if not isinstance(configuration.get(name), int) or configuration[name] <= 0:
            raise ValueError(f"MLP configuration field {name} must be a positive integer.")
    for name in ("dropout", "learning_rate", "weight_decay"):
        value = float(configuration.get(name, -1))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"MLP configuration field {name} is invalid.")
    if not 0 <= float(configuration["dropout"]) < 1:
        raise ValueError("MLP dropout must be in [0, 1).")
    if float(configuration["learning_rate"]) <= 0:
        raise ValueError("MLP learning rate must be positive.")
    if manifest["training"]["seeds"] != [configuration["seed"]]:
        raise ValueError("The manifest seed and MLP screening seed differ.")
    return configuration


def _set_deterministic_seed(seed: int) -> None:
    """Set the random generators used by the fixed MLP screening run."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def _feature_contract(
    manifest: dict,
    variant_name: str,
    packet_feature_names: list[str],
) -> tuple[list[str], tuple[str, ...]]:
    if variant_name not in VARIANTS:
        raise ValueError(f"Undeclared MLP variant: {variant_name}")
    context_names = (
        ()
        if variant_name == "packet"
        else context_features_for_variant(manifest, "history")
    )
    if variant_name == "history" and len(context_names) != 6:
        raise ValueError("The MLP history variant requires six frozen history fields.")
    names = [*packet_feature_names, *context_names]
    if len(packet_feature_names) != 103 or len(names) != len(set(names)):
        raise ValueError("The MLP feature contract has an invalid order or dimension.")
    return names, tuple(context_names)


def _training_batches(
    *,
    packet_path: Path,
    context_path: Path | None,
    columns: list[str],
    batch_size: int,
):
    if context_path is None:
        parquet = pq.ParquetFile(packet_path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            yield batch.to_pandas(), None
        return
    yield from _paired_batches(packet_path, context_path, columns, batch_size)


def _materialize_training_fold(
    *,
    matrix_dir: Path,
    train_scenarios: list[str],
    packet_paths: dict[str, Path],
    context_paths: dict[str, Path] | None,
    reports: dict[str, dict],
    preprocessor,
    context_scaler: dict | None,
    feature_count: int,
    weights: dict[str, dict[int, float]],
    batch_size: int,
) -> tuple[np.memmap, np.memmap, np.memmap, dict]:
    """Materialize one fold on local storage without loading it into RAM."""
    rows = sum(int(reports[name]["counts"]["packets"]) for name in train_scenarios)
    required_bytes = rows * (feature_count * 4 + 1 + 4) + 1_000_000_000
    free_bytes = shutil.disk_usage(matrix_dir).free
    if free_bytes < required_bytes:
        raise OSError(
            "Insufficient local storage for the MLP training matrix: "
            f"need {required_bytes / 1e9:.2f} GB, free {free_bytes / 1e9:.2f} GB."
        )
    features = np.lib.format.open_memmap(
        matrix_dir / "features.npy",
        mode="w+",
        dtype=np.float32,
        shape=(rows, feature_count),
    )
    labels = np.lib.format.open_memmap(
        matrix_dir / "labels.npy", mode="w+", dtype=np.uint8, shape=(rows,)
    )
    sample_weights = np.lib.format.open_memmap(
        matrix_dir / "weights.npy", mode="w+", dtype=np.float32, shape=(rows,)
    )
    columns = list(
        dict.fromkeys([*preprocessor.required_columns, "source_row_id", "binary_label"])
    )
    offset = 0
    observed = {}
    for scenario in train_scenarios:
        scenario_rows = normal = attack = 0
        context_path = None if context_paths is None else context_paths[scenario]
        for packet_batch, context_batch in _training_batches(
            packet_path=packet_paths[scenario],
            context_path=context_path,
            columns=columns,
            batch_size=batch_size,
        ):
            size = len(packet_batch)
            row_ids = packet_batch["source_row_id"].to_numpy(dtype=np.int64)
            if not np.array_equal(
                row_ids, np.arange(scenario_rows, scenario_rows + size)
            ):
                raise ValueError(f"Training packet order changed in {scenario}.")
            batch_labels = _validated_labels(packet_batch, scenario)
            packet_values = preprocessor.transform(packet_batch).to_numpy(
                dtype=np.float32, copy=False
            )
            if context_batch is None:
                matrix = packet_values
            else:
                if context_scaler is None:
                    raise ValueError("The history matrix requires a fitted context scaler.")
                matrix = np.concatenate(
                    [packet_values, transform_context(context_batch, context_scaler)],
                    axis=1,
                )
            if matrix.shape != (size, feature_count) or not np.isfinite(matrix).all():
                raise ValueError("A transformed MLP training batch is invalid.")
            features[offset : offset + size] = matrix
            labels[offset : offset + size] = batch_labels
            sample_weights[offset : offset + size] = np.where(
                batch_labels == 0,
                weights[scenario][0],
                weights[scenario][1],
            ).astype(np.float32)
            normal += int(np.count_nonzero(batch_labels == 0))
            attack += int(np.count_nonzero(batch_labels == 1))
            scenario_rows += size
            offset += size
        counts = reports[scenario]["counts"]
        if (
            scenario_rows != int(counts["packets"])
            or normal != int(counts["normal_packets"])
            or attack != int(counts["attack_packets"])
        ):
            raise ValueError(f"Training row or label counts differ for {scenario}.")
        observed[scenario] = {
            "rows": scenario_rows,
            "normal": normal,
            "attack": attack,
        }
    if (
        offset != rows
        or not np.isfinite(sample_weights).all()
        or np.any(sample_weights <= 0)
    ):
        raise ValueError("The MLP training matrix or weights are incomplete.")
    features.flush()
    labels.flush()
    sample_weights.flush()
    return features, labels, sample_weights, observed


def _train_fixed_epoch_mlp(
    *,
    features: np.memmap,
    labels: np.memmap,
    sample_weights: np.memmap,
    configuration: dict,
    device: str,
):
    """Train without consulting outer-fold validation predictions."""
    import torch
    import torch.nn.functional as functional

    from .models import SimpleMLP

    if device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("The frozen MLP screening run requires a CUDA runtime.")
    seed = int(configuration["seed"])
    _set_deterministic_seed(seed)
    model = SimpleMLP(
        edge_dim=int(features.shape[1]),
        hidden_dim=int(configuration["hidden_dim"]),
        dropout=float(configuration["dropout"]),
        output_bias_init=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(configuration["learning_rate"]),
        weight_decay=float(configuration["weight_decay"]),
    )
    rows = int(features.shape[0])
    batch_size = int(configuration["training_batch_size"])
    block_rows = int(configuration["shuffle_block_rows"])
    blocks = [(start, min(start + block_rows, rows)) for start in range(0, rows, block_rows)]
    history = []
    started = time.monotonic()
    for epoch in range(int(configuration["fixed_epochs"])):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        block_order = rng.permutation(len(blocks))
        weighted_loss_sum = 0.0
        weight_sum = 0.0
        examples = 0
        for block_index in block_order:
            start, end = blocks[int(block_index)]
            indexes = np.arange(start, end, dtype=np.int64)
            rng.shuffle(indexes)
            for batch_start in range(0, len(indexes), batch_size):
                batch_indexes = indexes[batch_start : batch_start + batch_size]
                feature_tensor = torch.from_numpy(
                    np.asarray(features[batch_indexes], dtype=np.float32)
                ).to(device)
                target_tensor = torch.from_numpy(
                    np.asarray(labels[batch_indexes], dtype=np.float32)
                ).to(device)
                weight_tensor = torch.from_numpy(
                    np.asarray(sample_weights[batch_indexes], dtype=np.float32)
                ).to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(
                    edge_index=None,
                    edge_attr=feature_tensor,
                    num_nodes=0,
                ).view(-1)
                losses = functional.binary_cross_entropy_with_logits(
                    logits, target_tensor, reduction="none"
                )
                weighted = losses * weight_tensor
                denominator = weight_tensor.sum()
                if not torch.isfinite(weighted).all() or denominator <= 0:
                    raise FloatingPointError("The MLP loss became invalid.")
                loss = weighted.sum() / denominator
                loss.backward()
                optimizer.step()
                weighted_loss_sum += float(weighted.detach().sum().cpu())
                weight_sum += float(denominator.detach().cpu())
                examples += len(batch_indexes)
        epoch_record = {
            "epoch": epoch + 1,
            "weighted_loss": weighted_loss_sum / weight_sum,
            "examples": examples,
            "elapsed_seconds": time.monotonic() - started,
        }
        history.append(epoch_record)
        print(
            f"Epoch {epoch + 1}/{configuration['fixed_epochs']}: "
            f"weighted loss={epoch_record['weighted_loss']:.6f}",
            flush=True,
        )
    return model, history


def _predict_scores(model, values: np.ndarray, device: str) -> np.ndarray:
    import torch

    model.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).to(device)
        logits = model(edge_index=None, edge_attr=tensor, num_nodes=0).view(-1)
        scores = torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False)
    if (
        scores.shape != (len(values),)
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("The MLP produced invalid packet scores.")
    return scores


def _score_validation_scenario(
    *,
    model,
    scenario: str,
    packet_path: Path,
    context_path: Path | None,
    report: dict,
    preprocessor,
    context_scaler: dict | None,
    output_path: Path,
    width_seconds: int,
    batch_size: int,
    device: str,
) -> dict:
    columns = list(dict.fromkeys([*preprocessor.required_columns, *OOF_COLUMNS]))
    origin_ns = int(report["scenario_origin_timestamp_ns"])
    expected_rows = int(report["counts"]["packets"])
    rows = normal = attack = 0
    writer = None
    try:
        for packet_batch, context_batch in _training_batches(
            packet_path=packet_path,
            context_path=context_path,
            columns=columns,
            batch_size=batch_size,
        ):
            size = len(packet_batch)
            row_ids = packet_batch["source_row_id"].to_numpy(dtype=np.int64)
            if not np.array_equal(row_ids, np.arange(rows, rows + size)):
                raise ValueError(f"Validation packet order changed in {scenario}.")
            batch_labels = _validated_labels(packet_batch, scenario)
            packet_values = preprocessor.transform(packet_batch).to_numpy(
                dtype=np.float32, copy=False
            )
            if context_batch is None:
                values = packet_values
            else:
                if context_scaler is None:
                    raise ValueError("The history scorer requires a context scaler.")
                values = np.concatenate(
                    [packet_values, transform_context(context_batch, context_scaler)],
                    axis=1,
                )
            scores = _predict_scores(model, values, device)
            timestamps = packet_batch["packet_timestamp_ns"].to_numpy(dtype=np.int64)
            window_indexes, window_ends = window_coordinates(
                timestamps, origin_ns, width_seconds
            )
            output = packet_batch.loc[:, list(OOF_COLUMNS)].copy()
            output["window_index"] = window_indexes
            output["window_end_ns"] = window_ends
            output["score"] = scores
            table = pa.Table.from_pandas(
                output, schema=OOF_SCHEMA, preserve_index=False, safe=True
            )
            if writer is None:
                writer = pq.ParquetWriter(output_path, OOF_SCHEMA, compression="zstd")
            writer.write_table(table)
            normal += int(np.count_nonzero(batch_labels == 0))
            attack += int(np.count_nonzero(batch_labels == 1))
            rows += size
    finally:
        if writer is not None:
            writer.close()
    counts = report["counts"]
    if (
        rows != expected_rows
        or normal != int(counts["normal_packets"])
        or attack != int(counts["attack_packets"])
    ):
        raise ValueError(f"Validation row or label counts differ for {scenario}.")
    if not output_path.is_file() or pq.ParquetFile(output_path).metadata.num_rows != rows:
        raise ValueError(f"The MLP OOF artifact is incomplete for {scenario}.")
    return {
        "rows": rows,
        "normal_packets": normal,
        "attack_packets": attack,
        "oof_artifact": output_path.name,
        "oof_sha256": sha256_file(output_path),
    }


def run_capture_mlp_fold(
    *,
    manifest_path: Path,
    packet_schema_path: Path,
    preprocessing_schema_path: Path,
    prepared_run_dir: Path,
    preprocessing_audit_dir: Path,
    context_dir: Path,
    output_dir: Path,
    local_work_root: Path,
    fold: str,
    variant_name: str,
) -> dict:
    """Train one fixed MLP variant and persist outer-fold OOF packet scores."""
    import torch

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output_dir}")
    manifest_path = Path(manifest_path)
    packet_schema_path = Path(packet_schema_path)
    preprocessing_schema_path = Path(preprocessing_schema_path)
    prepared_run_dir = Path(prepared_run_dir)
    preprocessing_audit_dir = Path(preprocessing_audit_dir)
    context_dir = Path(context_dir)
    local_work_root = Path(local_work_root)
    manifest, packet_schema, reports, packet_paths = load_prepared_full_dev(
        prepared_run_dir, manifest_path, packet_schema_path
    )
    configuration = mlp_configuration(manifest)
    if fold not in manifest["validation"]["folds"]:
        raise ValueError(f"Undeclared development fold: {fold}")
    split = manifest["validation"]["folds"][fold]
    train_scenarios = list(split["train"])
    validation_scenarios = list(split["validate"])
    schema = load_preprocessing_schema(preprocessing_schema_path, packet_schema)
    if schema["status"] != "frozen":
        raise ValueError("MLP training requires the frozen packet preprocessing contract.")
    preprocessor, preprocessor_sha256 = _load_fold_preprocessor(
        preprocessing_audit_dir,
        schema,
        fold,
        train_scenarios,
        prepared_run_dir,
    )
    feature_names, context_feature_names = _feature_contract(
        manifest, variant_name, preprocessor.feature_names
    )
    context_report = None
    context_paths = None
    context_scaler = None
    if variant_name == "history":
        context_report = _load_context_run(
            context_dir, manifest, reports, packet_paths
        )
        context_paths = {
            scenario: context_dir / item["context_artifact"]
            for scenario, item in context_report["scenarios"].items()
        }
        context_scaler = fit_context_scaler(
            [context_paths[name] for name in train_scenarios],
            int(configuration["materialization_batch_size"]),
            context_feature_names,
        )
    training_rows = sum(
        int(reports[name]["counts"]["packets"]) for name in train_scenarios
    )
    if preprocessor.training_rows != training_rows:
        raise ValueError("The packet preprocessor was fitted on a different row count.")
    if context_scaler is not None and context_scaler["training_rows"] != training_rows:
        raise ValueError("The history scaler was fitted on a different row count.")
    weights = scenario_class_weights(train_scenarios, reports)
    width_seconds = int(manifest["windows"]["selected_duration_seconds"])
    if width_seconds != 5:
        raise ValueError("The MLP screen requires the frozen five-second windows.")
    if configuration["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Select a CUDA-enabled Colab runtime before MLP training.")
    local_work_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    repository_root = manifest_path.resolve().parent.parent
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    worktree = subprocess.run(
        ["git", "status", "--short"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    with tempfile.TemporaryDirectory(
        prefix=f"capture_mlp_{variant_name}_{fold}_", dir=local_work_root
    ) as temporary:
        stage = Path(temporary)
        print(
            f"Materializing {variant_name} fold {fold} with "
            f"{len(feature_names)} features...",
            flush=True,
        )
        features, labels, sample_weights, observed_training = (
            _materialize_training_fold(
                matrix_dir=stage,
                train_scenarios=train_scenarios,
                packet_paths=packet_paths,
                context_paths=context_paths,
                reports=reports,
                preprocessor=preprocessor,
                context_scaler=context_scaler,
                feature_count=len(feature_names),
                weights=weights,
                batch_size=int(configuration["materialization_batch_size"]),
            )
        )
        print(
            f"Training {variant_name} fold {fold} on {len(labels):,} packets...",
            flush=True,
        )
        model, training_history = _train_fixed_epoch_mlp(
            features=features,
            labels=labels,
            sample_weights=sample_weights,
            configuration=configuration,
            device="cuda",
        )
        checkpoint_path = stage / "model.pt"
        torch.save(
            {
                "model_state_dict": {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                "model_family": "capture_mlp",
                "variant_name": variant_name,
                "configuration_name": CONFIGURATION_NAME,
                "configuration": configuration,
                "feature_names": feature_names,
                "fold": fold,
            },
            checkpoint_path,
        )
        write_json(stage / "training_history.json", {"epochs": training_history})
        if context_scaler is not None:
            write_json(stage / "context_scaler.json", context_scaler)
        del features, labels, sample_weights
        gc.collect()
        validation = {}
        for scenario in validation_scenarios:
            print(f"Scoring held-out scenario {scenario}...", flush=True)
            oof_path = stage / f"oof_{scenario}.parquet"
            saved = _score_validation_scenario(
                model=model,
                scenario=scenario,
                packet_path=packet_paths[scenario],
                context_path=(
                    None if context_paths is None else context_paths[scenario]
                ),
                report=reports[scenario],
                preprocessor=preprocessor,
                context_scaler=context_scaler,
                output_path=oof_path,
                width_seconds=width_seconds,
                batch_size=int(configuration["inference_batch_size"]),
                device="cuda",
            )
            validation[scenario] = {**saved, **_scenario_metrics(oof_path)}
        result = {
            "report_version": REPORT_VERSION,
            "status": "development_oof_complete_thresholds_pending",
            "model_family": "capture_mlp",
            "variant_name": variant_name,
            "fold": fold,
            "configuration_name": CONFIGURATION_NAME,
            "configuration": configuration,
            "training_scenarios": train_scenarios,
            "validation_scenarios": validation_scenarios,
            "training_rows": training_rows,
            "observed_training_counts": observed_training,
            "scenario_class_weights": {
                scenario: {"normal": values[0], "attack": values[1]}
                for scenario, values in weights.items()
            },
            "preprocessor_sha256": preprocessor_sha256,
            "preprocessing_contract_sha256": preprocessing_schema_sha256(schema),
            "context_report_sha256": (
                None
                if context_report is None
                else sha256_file(context_dir / "context_report.json")
            ),
            "context_scaler_artifact": (
                None if context_scaler is None else "context_scaler.json"
            ),
            "context_scaler_sha256": (
                None
                if context_scaler is None
                else sha256_file(stage / "context_scaler.json")
            ),
            "context_feature_names": list(context_feature_names),
            "feature_count": len(feature_names),
            "feature_names": feature_names,
            "active_feature_count": len(preprocessor.active_features),
            "masked_features": preprocessor.masked_features,
            "window_width_seconds": width_seconds,
            "decision_time": configuration["primary_decision_time"],
            "native_decision_time": configuration["secondary_native_decision_time"],
            "validation": validation,
            "fold_macro_packet_roc_auc": float(
                np.mean([item["packet_roc_auc"] for item in validation.values()])
            ),
            "fold_macro_packet_pr_auc_diagnostic": float(
                np.mean(
                    [item["packet_pr_auc_diagnostic"] for item in validation.values()]
                )
            ),
            "model_artifact": checkpoint_path.name,
            "model_sha256": sha256_file(checkpoint_path),
            "training_history_artifact": "training_history.json",
            "training_history_sha256": sha256_file(stage / "training_history.json"),
            "torch_version": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "manifest_sha256": sha256_file(manifest_path),
            "packet_schema_sha256": sha256_file(packet_schema_path),
            "preprocessing_schema_sha256": sha256_file(preprocessing_schema_path),
            "trainer_code_sha256": sha256_file(Path(__file__)),
            "git_commit": (
                revision.stdout.strip() if revision.returncode == 0 else "unavailable"
            ),
            "git_worktree_status": (
                worktree.stdout if worktree.returncode == 0 else "unavailable"
            ),
            "prepared_packet_sha256": {
                scenario: reports[scenario]["output_sha256"]
                for scenario in [*train_scenarios, *validation_scenarios]
            },
            "context_artifact_sha256": (
                {}
                if context_report is None
                else {
                    scenario: context_report["scenarios"][scenario]["context_sha256"]
                    for scenario in [*train_scenarios, *validation_scenarios]
                }
            ),
            "elapsed_seconds": time.monotonic() - started,
            "thresholds_selected": False,
            "test_data_accessed": False,
        }
        write_json(stage / "fold_report.json", result)
        write_json(
            stage / "run_status.json",
            {
                "complete": True,
                "report": "fold_report.json",
                "report_sha256": sha256_file(stage / "fold_report.json"),
            },
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, output_dir, ignore=shutil.ignore_patterns("*.npy"))
        for name, checksum in (
            ("fold_report.json", sha256_file(stage / "fold_report.json")),
            ("model.pt", result["model_sha256"]),
            ("training_history.json", result["training_history_sha256"]),
        ):
            if sha256_file(output_dir / name) != checksum:
                raise IOError(f"Copied MLP artifact failed checksum verification: {name}")
        if context_scaler is not None and sha256_file(
            output_dir / "context_scaler.json"
        ) != result["context_scaler_sha256"]:
            raise IOError("Copied MLP context scaler failed checksum verification.")
        for item in validation.values():
            if sha256_file(output_dir / item["oof_artifact"]) != item["oof_sha256"]:
                raise IOError("Copied MLP OOF predictions failed checksum verification.")
    print(f"Saved {variant_name} fold {fold} to {output_dir}.", flush=True)
    return result


def validate_capture_mlp_fold_run(
    directory: Path, fold: str, variant_name: str
) -> dict:
    """Validate one immutable MLP fold run and all reportable artifacts."""
    if variant_name not in VARIANTS:
        raise ValueError(f"Undeclared MLP variant: {variant_name}")
    directory = Path(directory)
    report_path = directory / "fold_report.json"
    status_path = directory / "run_status.json"
    if not report_path.is_file() or not status_path.is_file():
        raise FileNotFoundError(f"Complete MLP {variant_name} fold {fold} is required.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("complete") is not True or status.get("report_sha256") != sha256_file(
        report_path
    ):
        raise ValueError("The MLP fold status is incomplete or changed.")
    if (
        report.get("model_family") != "capture_mlp"
        or report.get("variant_name") != variant_name
        or report.get("fold") != fold
        or report.get("configuration_name") != CONFIGURATION_NAME
    ):
        raise ValueError("The MLP fold report binding is invalid.")
    artifacts = {
        report["model_artifact"]: report["model_sha256"],
        report["training_history_artifact"]: report["training_history_sha256"],
    }
    if report.get("context_scaler_artifact") is not None:
        artifacts[report["context_scaler_artifact"]] = report["context_scaler_sha256"]
    for item in report["validation"].values():
        artifacts[item["oof_artifact"]] = item["oof_sha256"]
    for name, expected in artifacts.items():
        if sha256_file(directory / name) != expected:
            raise ValueError(f"The MLP artifact changed after training: {name}")
    return report


def summarize_capture_mlp_oof(run_dir: Path, variant_name: str) -> dict:
    """Compute the hierarchical development OOF ranking summary."""
    run_dir = Path(run_dir)
    reports = {
        fold: validate_capture_mlp_fold_run(
            run_dir / variant_name / f"fold_{fold}", fold, variant_name
        )
        for fold in ("A", "B")
    }
    if (
        reports["A"]["manifest_sha256"] != reports["B"]["manifest_sha256"]
        or reports["A"]["configuration"] != reports["B"]["configuration"]
        or reports["A"]["feature_names"] != reports["B"]["feature_names"]
    ):
        raise ValueError("The MLP folds do not share one frozen protocol.")
    scenarios_a = set(reports["A"]["validation"])
    scenarios_b = set(reports["B"]["validation"])
    if scenarios_a & scenarios_b or len(scenarios_a | scenarios_b) != 5:
        raise ValueError("Exactly one MLP OOF score is required per development scenario.")
    return {
        "variant_name": variant_name,
        "model_name": MODEL_NAMES[variant_name],
        "configuration_name": CONFIGURATION_NAME,
        "fold_packet_roc_auc": {
            fold: reports[fold]["fold_macro_packet_roc_auc"] for fold in ("A", "B")
        },
        "fold_packet_pr_auc_diagnostic": {
            fold: reports[fold]["fold_macro_packet_pr_auc_diagnostic"]
            for fold in ("A", "B")
        },
        "hierarchical_macro_oof_packet_roc_auc": float(
            np.mean([reports[fold]["fold_macro_packet_roc_auc"] for fold in ("A", "B")])
        ),
        "hierarchical_macro_oof_packet_pr_auc_diagnostic": float(
            np.mean(
                [
                    reports[fold]["fold_macro_packet_pr_auc_diagnostic"]
                    for fold in ("A", "B")
                ]
            )
        ),
        "scenario_metrics": {
            scenario: {
                "fold": fold,
                "packets": item["rows"],
                "packet_roc_auc": item["packet_roc_auc"],
                "packet_pr_auc_diagnostic": item["packet_pr_auc_diagnostic"],
            }
            for fold in ("A", "B")
            for scenario, item in reports[fold]["validation"].items()
        },
        "thresholds_selected": False,
        "test_data_accessed": False,
    }


def run_capture_mlp_operational_evaluation(
    *,
    manifest_path: Path,
    mlp_run_dir: Path,
    output_dir: Path,
    batch_size: int = 250_000,
) -> dict:
    """Apply the frozen false-alert-window policy to both MLP variants."""
    if batch_size <= 0:
        raise ValueError("Operational batch size must be positive.")
    manifest_path = Path(manifest_path)
    mlp_run_dir = Path(mlp_run_dir)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output_dir}")
    manifest = load_manifest(manifest_path)
    folds = manifest["validation"]["folds"]
    budgets = operational_budgets(manifest)
    fold_reports = {
        variant: {
            fold: validate_capture_mlp_fold_run(
                mlp_run_dir / variant / f"fold_{fold}", fold, variant
            )
            for fold in folds
        }
        for variant in VARIANTS
    }
    reference = fold_reports["packet"]
    for variant in VARIANTS:
        for fold, split in folds.items():
            report = fold_reports[variant][fold]
            if (
                report["training_scenarios"] != split["train"]
                or report["validation_scenarios"] != split["validate"]
                or report["window_width_seconds"] != 5
                or report["decision_time"] != "window_end"
            ):
                raise ValueError(
                    f"Fold or decision-time provenance differs for {variant}/{fold}."
                )
            if (
                report["prepared_packet_sha256"]
                != reference[fold]["prepared_packet_sha256"]
            ):
                raise ValueError(
                    f"Prepared packet provenance differs for {variant}/{fold}."
                )
            for scenario in split["validate"]:
                item = report["validation"][scenario]
                reference_item = reference[fold]["validation"][scenario]
                count_fields = ("rows", "normal_packets", "attack_packets")
                if tuple(item[name] for name in count_fields) != tuple(
                    reference_item[name] for name in count_fields
                ):
                    raise ValueError(
                        f"OOF packet counts differ for {variant}/{scenario}."
                    )
    result = {
        "report_version": REPORT_VERSION,
        "status": "development_oof_mlp_operational_evaluation_complete",
        "manifest_sha256": sha256_file(manifest_path),
        "evaluator_code_sha256": sha256_file(Path(__file__)),
        "mlp_run_dir": str(mlp_run_dir),
        "budget_order": [name for name, _ in budgets],
        "budgets_per_hour": {name: value for name, value in budgets},
        "models": {},
        "thresholds_selected_from": "development_oof_only",
        "test_data_accessed": False,
    }
    for variant in VARIANTS:
        model_name = MODEL_NAMES[variant]
        summaries = {}
        paths = {}
        expected = {}
        for fold, split in folds.items():
            fold_report = fold_reports[variant][fold]
            for scenario in split["validate"]:
                item = fold_report["validation"][scenario]
                path = mlp_run_dir / variant / f"fold_{fold}" / item["oof_artifact"]
                summaries[scenario] = summarize_scenario_oof(path, item)
                paths[scenario] = path
                expected[scenario] = item
        thresholds = {
            name: {
                "target_false_alert_windows_per_hour": budget,
                **select_threshold(summaries, folds, budget),
            }
            for name, budget in budgets
        }
        packet_counts = {
            scenario: _packet_counts_by_threshold(
                paths[scenario],
                {name: item["threshold"] for name, item in thresholds.items()},
                expected[scenario],
                batch_size,
            )
            for scenario in paths
        }
        per_budget = {}
        for budget_name, _ in budgets:
            threshold = thresholds[budget_name]["threshold"]
            scenario_metrics = {}
            iteration_rows = []
            step_groups = defaultdict(list)
            for fold, split in folds.items():
                for scenario in split["validate"]:
                    metrics, iterations = evaluate_scenario(
                        summaries[scenario],
                        packet_counts[scenario][budget_name],
                        threshold,
                        scenario,
                        fold,
                    )
                    scenario_metrics[scenario] = metrics
                    iteration_rows.extend(iterations)
                    for item in iterations:
                        step_groups[(scenario, item["attack_step"])].append(item)
            step_metrics = {
                f"{scenario}::{step}": {
                    "scenario": scenario,
                    "attack_step": step,
                    "iterations": len(items),
                    "detected_iterations": sum(item["detected"] for item in items),
                    "sequence_detection_rate": (
                        sum(item["detected"] for item in items) / len(items)
                    ),
                    "mean_miss_capped_latency_seconds": float(
                        np.mean(
                            [item["miss_capped_latency_seconds"] for item in items]
                        )
                    ),
                }
                for (scenario, step), items in step_groups.items()
            }
            per_budget[budget_name] = {
                "scenario_metrics": scenario_metrics,
                "step_metrics": step_metrics,
                "iteration_rows": iteration_rows,
                **_macro_metrics(scenario_metrics, folds),
            }
        result["models"][model_name] = {
            "variant_name": variant,
            "thresholds": thresholds,
            "budgets": per_budget,
            "fold_reports": {
                fold: str(mlp_run_dir / variant / f"fold_{fold}" / "fold_report.json")
                for fold in folds
            },
        }
    output_dir.mkdir(parents=True)
    write_json(output_dir / "operational_report.json", result)
    write_json(
        output_dir / "run_status.json",
        {
            "complete": True,
            "report_sha256": sha256_file(output_dir / "operational_report.json"),
        },
    )
    return result


def validate_capture_mlp_operational_evaluation(
    *, manifest_path: Path, mlp_run_dir: Path, output_dir: Path
) -> dict:
    """Validate a completed MLP operational report and its fold provenance."""
    manifest_path = Path(manifest_path)
    mlp_run_dir = Path(mlp_run_dir)
    output_dir = Path(output_dir)
    report_path = output_dir / "operational_report.json"
    status_path = output_dir / "run_status.json"
    if not report_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("A complete MLP operational evaluation is required.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if (
        status.get("complete") is not True
        or status.get("report_sha256") != sha256_file(report_path)
        or report.get("manifest_sha256") != sha256_file(manifest_path)
        or report.get("mlp_run_dir") != str(mlp_run_dir)
    ):
        raise ValueError("The MLP operational report is incomplete or changed.")
    for variant in VARIANTS:
        for fold in load_manifest(manifest_path)["validation"]["folds"]:
            validate_capture_mlp_fold_run(
                mlp_run_dir / variant / f"fold_{fold}", fold, variant
            )
    return report
