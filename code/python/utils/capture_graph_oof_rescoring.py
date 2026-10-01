"""Versioned OOF logit rescoring for the cAPTure graph-model pilot."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score
import torch
import yaml

from . import capture_graph_dataset, capture_graph_training, models, training
from .capture_data import sha256_file, write_json
from .capture_graph_comparison import MODEL_NAMES
from .capture_graph_dataset import CaptureGraphCollection
from .capture_graph_score_saturation import (
    validate_graph_score_saturation_audit,
)
from .capture_graph_training import _build_model, _set_seed
from .training import forward_graph


REPORT_VERSION = 1
MODULE_PATH = Path(__file__)
FOLDS = ("A", "B")
SOURCE_COLUMNS = tuple(field.name for field in capture_graph_training.PREDICTION_SCHEMA)
OUTPUT_SCHEMA = pa.schema([
    *[
        field
        for field in capture_graph_training.PREDICTION_SCHEMA
        if field.name != "score"
    ],
    pa.field("source_score_float32", pa.float32(), nullable=False),
    pa.field("recomputed_score_float32", pa.float32(), nullable=False),
    pa.field("raw_logit", pa.float32(), nullable=False),
    pa.field("score_float64", pa.float64(), nullable=False),
])


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


def load_graph_oof_rescoring_config(path: str | Path) -> dict:
    """Load and strictly validate the development-only rescoring contract."""
    path = Path(path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("The graph OOF rescoring contract must be a mapping.")
    if (
        config.get("rescoring_contract_version") != 1
        or config.get("scope") != "development_only"
        or config.get("stage") != "stage2_outer_oof_inference_only"
        or tuple(config.get("models", [])) != tuple(MODEL_NAMES)
    ):
        raise ValueError("Unsupported graph OOF rescoring contract.")
    expected_folds = {
        "A": {
            "outer_validation": [
                "train_dollar_char",
                "train_slash_char",
                "train_sub_exf",
            ]
        },
        "B": {"outer_validation": ["train_empty_conn", "train_qos_mid"]},
    }
    if config.get("folds") != expected_folds:
        raise ValueError("The rescoring fold assignments changed.")
    expected_inference = {
        "device": "cuda",
        "seed": 42,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "graph_order": "chronological",
        "shuffle": False,
        "reset_temporal_state_before_each_scenario": True,
        "warmup_graphs_excluded_from_timing": 1,
        "source_score_type": "float32",
        "raw_logit_type": "float32",
        "recomputed_score_type": "float32",
        "high_resolution_score_type": "float64",
        "high_resolution_score_definition": (
            "sigmoid_of_float32_logit_computed_in_float64"
        ),
        "operational_ranking_field": "raw_logit",
        "maximum_allowed_source_probability_absolute_difference": 0.0001,
    }
    if config.get("inference") != expected_inference:
        raise ValueError("The OOF inference contract changed.")
    expected_output = {
        "preserve_source_probability": True,
        "preserve_recomputed_probability": True,
        "preserve_raw_logit": True,
        "preserve_float64_sigmoid": True,
        "require_identical_packet_keys_and_labels": True,
        "completed_job_is_immutable": True,
        "job_unit": "model_and_fold",
    }
    if config.get("output") != expected_output:
        raise ValueError("The rescoring output contract changed.")
    if config.get("prohibitions") != {
        "model_training": True,
        "optimizer_creation": True,
        "checkpoint_modification": True,
        "threshold_selection": True,
        "held_out_scenario_access": True,
        "source_artifact_overwrite": True,
    }:
        raise ValueError("The rescoring prohibitions changed.")
    return config


def expected_rescoring_job_ids(config: dict) -> list[str]:
    """Return the frozen model/fold execution order."""
    return [
        f"{model_name}__fold_{fold}"
        for model_name in config["models"]
        for fold in FOLDS
    ]


def _dependency_hashes() -> dict[str, str]:
    dependencies = {
        "capture_graph_oof_rescoring.py": MODULE_PATH,
        "capture_graph_training.py": Path(capture_graph_training.__file__),
        "capture_graph_dataset.py": Path(capture_graph_dataset.__file__),
        "models.py": Path(models.__file__),
        "training.py": Path(training.__file__),
    }
    return {name: sha256_file(path) for name, path in dependencies.items()}


def _validate_source_context(
    *,
    config: dict,
    comparison_dir: Path,
    saturation_audit_dir: Path,
) -> tuple[dict, dict]:
    bindings = config["bindings"]
    if comparison_dir.name != bindings["four_model_comparison_run_id"]:
        raise ValueError("The source comparison run ID changed.")
    if saturation_audit_dir.name != bindings["score_saturation_audit_run_id"]:
        raise ValueError("The score-saturation audit run ID changed.")
    audit = validate_graph_score_saturation_audit(
        comparison_dir=comparison_dir,
        output_dir=saturation_audit_dir,
    )
    comparison = _load_json(
        comparison_dir / "comparison_report.json",
        "four-model comparison report",
    )
    if (
        audit.get("status") != "score_saturation_confirmed"
        or audit.get("oof_rescoring_recommended") is not True
        or sorted(audit.get("saturated_models", [])) != ["edge_gru", "st_gnn"]
        or comparison.get("models_compared") != list(MODEL_NAMES)
        or comparison.get("model_training_performed") is not False
        or comparison.get("held_out_scenarios_accessed") is not False
    ):
        raise ValueError("The source audit does not authorize the frozen rescoring task.")
    return comparison, audit


def _run_contract(
    *,
    output_root: Path,
    config_path: Path,
    config: dict,
    comparison_dir: Path,
    saturation_audit_dir: Path,
    training_root: Path,
    prepared_run_dir: Path,
    collection: CaptureGraphCollection,
) -> dict:
    bindings = config["bindings"]
    if (
        not training_root.is_dir()
        or not prepared_run_dir.is_dir()
        or training_root.name != bindings["training_binding_run_id"]
        or prepared_run_dir.name != bindings["prepared_run_id"]
        or collection.expected_run_id != bindings["graph_materialization_run_id"]
        or collection.contract_path.name != bindings["graph_input_contract"]
        or not collection.artifact_checksums_verified
    ):
        raise ValueError("The staged rescoring inputs differ from the frozen bindings.")
    return {
        "report_version": REPORT_VERSION,
        "run_id": output_root.name,
        "rescoring_contract_sha256": sha256_file(config_path),
        "source_comparison_run_id": comparison_dir.name,
        "source_comparison_report_sha256": sha256_file(
            comparison_dir / "comparison_report.json"
        ),
        "source_saturation_audit_run_id": saturation_audit_dir.name,
        "source_saturation_audit_report_sha256": sha256_file(
            saturation_audit_dir / "score_saturation_audit.json"
        ),
        "source_training_binding_run_id": training_root.name,
        "prepared_run_id": prepared_run_dir.name,
        "graph_materialization_run_id": collection.expected_run_id,
        "graph_materialization_manifest_sha256": sha256_file(
            collection.manifest_path
        ),
        "models": list(config["models"]),
        "folds": config["folds"],
        "inference": config["inference"],
        "output": config["output"],
        "prohibitions": config["prohibitions"],
        "dependency_sha256": _dependency_hashes(),
        "model_training_performed": False,
        "threshold_selected": False,
        "held_out_scenarios_accessed": False,
    }


def _bind_run_directory(output_root: Path, contract: dict) -> str:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "run_contract.json"
    if path.is_file():
        if _load_json(path, "rescoring run contract") != contract:
            raise ValueError("The existing rescoring run has a different contract.")
    else:
        unexpected = [item for item in output_root.iterdir() if item.name != path.name]
        if unexpected:
            raise FileExistsError("The rescoring output directory is not empty.")
        write_json(path, contract)
    return sha256_file(path)


def _source_job_context(
    *,
    job_id: str,
    model_name: str,
    fold: str,
    config: dict,
    comparison: dict,
    audit: dict,
    training_root: Path,
    collection: CaptureGraphCollection,
) -> dict:
    source_dir = training_root / job_id
    completion_path = source_dir / "completion.json"
    completion = _load_json(completion_path, f"{job_id} completion")
    contract_path = source_dir / "job_contract.json"
    contract = _load_json(contract_path, f"{job_id} training contract")
    metrics_path = source_dir / "metrics.json"
    metrics = _load_json(metrics_path, f"{job_id} metrics")
    checkpoint_path = source_dir / "final_model.pt"
    expected_scenarios = config["folds"][fold]["outer_validation"]
    scenario_metrics = metrics.get("outer_oof", {}).get("scenarios", {})
    if (
        completion.get("status") != "complete"
        or completion.get("model") != model_name
        or completion.get("fold") != fold
        or completion.get("model_training_performed") is not True
        or completion.get("held_out_scenarios_accessed") is not False
        or contract.get("job_id") != job_id
        or contract.get("model") != model_name
        or contract.get("fold") != fold
        or _canonical_sha256(contract) != completion.get("job_contract_sha256")
        or contract.get("graph_materialization_manifest_sha256")
        != sha256_file(collection.manifest_path)
        or set(scenario_metrics) != set(expected_scenarios)
        or sha256_file(completion_path)
        != comparison["source_completion_sha256"].get(job_id)
    ):
        raise ValueError(f"The immutable source job changed: {job_id}")
    required_artifacts = {
        "job_contract.json": contract_path,
        "metrics.json": metrics_path,
        "final_model.pt": checkpoint_path,
    }
    for relative, path in required_artifacts.items():
        if sha256_file(path) != completion["artifact_checksums"].get(relative):
            raise ValueError(f"The source artifact changed: {job_id}/{relative}")
    source_oof = {}
    for scenario in expected_scenarios:
        item = scenario_metrics[scenario]
        path = source_dir / item["predictions"]
        observed = sha256_file(path)
        expected = audit["source_input_oof_sha256"][model_name].get(scenario)
        if (
            observed != expected
            or observed != item.get("predictions_sha256")
            or observed != completion["artifact_checksums"].get(item["predictions"])
        ):
            raise ValueError(f"The source OOF artifact changed: {job_id}/{scenario}")
        parquet = pq.ParquetFile(path)
        if parquet.schema_arrow != capture_graph_training.PREDICTION_SCHEMA:
            raise ValueError(f"The source OOF schema changed: {job_id}/{scenario}")
        source_oof[scenario] = {
            "path": path,
            "sha256": observed,
            "rows": parquet.metadata.num_rows,
        }
    return {
        "source_dir": source_dir,
        "completion_path": completion_path,
        "completion": completion,
        "contract_path": contract_path,
        "contract": contract,
        "metrics_path": metrics_path,
        "metrics": metrics,
        "checkpoint_path": checkpoint_path,
        "source_oof": source_oof,
        "expected_scenarios": expected_scenarios,
    }


def _reset_memory(model: torch.nn.Module, temporal: bool) -> None:
    if temporal:
        model.reset_memory()


@torch.no_grad()
def _infer_high_resolution_scores(
    *,
    model: torch.nn.Module,
    dataset,
    device: torch.device,
    temporal: bool,
    temporary_dir: Path,
) -> tuple[dict[str, np.memmap], dict]:
    edges = int(dataset.report["edges"])
    arrays = {
        "raw_logit": np.lib.format.open_memmap(
            temporary_dir / "raw_logit.npy",
            mode="w+",
            dtype=np.float32,
            shape=(edges,),
        ),
        "score_float32": np.lib.format.open_memmap(
            temporary_dir / "score_float32.npy",
            mode="w+",
            dtype=np.float32,
            shape=(edges,),
        ),
        "score_float64": np.lib.format.open_memmap(
            temporary_dir / "score_float64.npy",
            mode="w+",
            dtype=np.float64,
            shape=(edges,),
        ),
        "target": np.lib.format.open_memmap(
            temporary_dir / "target.npy",
            mode="w+",
            dtype=np.uint8,
            shape=(edges,),
        ),
        "window_index": np.lib.format.open_memmap(
            temporary_dir / "window_index.npy",
            mode="w+",
            dtype=np.int64,
            shape=(edges,),
        ),
        "window_start_ns": np.lib.format.open_memmap(
            temporary_dir / "window_start_ns.npy",
            mode="w+",
            dtype=np.int64,
            shape=(edges,),
        ),
        "window_end_ns": np.lib.format.open_memmap(
            temporary_dir / "window_end_ns.npy",
            mode="w+",
            dtype=np.int64,
            shape=(edges,),
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
        arrays["raw_logit"][offset : offset + size] = (
            logits.cpu().numpy().astype(np.float32, copy=False)
        )
        arrays["score_float32"][offset : offset + size] = (
            torch.sigmoid(logits).cpu().numpy().astype(np.float32, copy=False)
        )
        arrays["score_float64"][offset : offset + size] = (
            torch.sigmoid(logits.double()).cpu().numpy().astype(np.float64, copy=False)
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
    if offset != edges:
        raise ValueError(f"Incomplete OOF inference: {dataset.scenario}")
    for name in ("raw_logit", "score_float32", "score_float64"):
        if not np.isfinite(arrays[name]).all():
            raise ValueError(f"Non-finite rescored values: {dataset.scenario}/{name}")
    for value in arrays.values():
        value.flush()
    return arrays, {
        "graphs": len(dataset),
        "edges": edges,
        "warmup_graphs_excluded_from_timing": 1,
        "pure_inference_seconds": inference_seconds,
        "seconds_per_million_packets": inference_seconds * 1_000_000 / edges,
        "average_precision_raw_logit": float(
            average_precision_score(arrays["target"], arrays["raw_logit"])
        ),
        "average_precision_score_float64": float(
            average_precision_score(arrays["target"], arrays["score_float64"])
        ),
        "raw_logit_minimum": float(np.min(arrays["raw_logit"])),
        "raw_logit_maximum": float(np.max(arrays["raw_logit"])),
        "score_float64_exact_one_count": int(
            np.count_nonzero(arrays["score_float64"] == 1.0)
        ),
    }


def _source_array(batch: pa.RecordBatch, name: str) -> pa.Array:
    index = batch.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"The source OOF batch is missing {name}.")
    return batch.column(index)


def _write_rescored_predictions(
    *,
    output_path: Path,
    source_path: Path,
    arrays: dict[str, np.memmap],
    model_name: str,
    fold: str,
    scenario: str,
    batch_size: int = 100_000,
) -> dict:
    writer = pq.ParquetWriter(
        output_path,
        OUTPUT_SCHEMA,
        compression="zstd",
        use_dictionary=["model", "fold", "scenario", "attack_step", "sequence_id"],
    )
    offset = 0
    exact_matches = 0
    absolute_difference_sum = 0.0
    maximum_absolute_difference = 0.0
    source_exact_one_count = 0
    recomputed_exact_one_count = 0
    try:
        parquet = pq.ParquetFile(source_path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=SOURCE_COLUMNS):
            size = batch.num_rows
            source_rows = _source_array(batch, "source_row_id").to_numpy(
                zero_copy_only=False
            )
            expected_rows = np.arange(offset, offset + size, dtype=np.int64)
            if not np.array_equal(source_rows, expected_rows):
                raise ValueError(f"Source OOF row order changed: {scenario}")
            source_targets = _source_array(batch, "binary_label").to_numpy(
                zero_copy_only=False
            ).astype(np.uint8, copy=False)
            if not np.array_equal(
                source_targets,
                np.asarray(arrays["target"][offset : offset + size]),
            ):
                raise ValueError(f"Source and rescored targets differ: {scenario}")
            for name in ("window_index", "window_start_ns", "window_end_ns"):
                source_values = _source_array(batch, name).to_numpy(
                    zero_copy_only=False
                )
                if not np.array_equal(
                    source_values,
                    np.asarray(arrays[name][offset : offset + size]),
                ):
                    raise ValueError(f"Source and graph {name} differ: {scenario}")
            source_scores = _source_array(batch, "score").to_numpy(
                zero_copy_only=False
            ).astype(np.float32, copy=False)
            recomputed_scores = np.asarray(
                arrays["score_float32"][offset : offset + size]
            )
            differences = np.abs(
                source_scores.astype(np.float64)
                - recomputed_scores.astype(np.float64)
            )
            exact_matches += int(np.count_nonzero(source_scores == recomputed_scores))
            absolute_difference_sum += float(np.sum(differences, dtype=np.float64))
            maximum_absolute_difference = max(
                maximum_absolute_difference,
                float(np.max(differences, initial=0.0)),
            )
            source_exact_one_count += int(np.count_nonzero(source_scores == 1.0))
            recomputed_exact_one_count += int(
                np.count_nonzero(recomputed_scores == 1.0)
            )
            output_arrays = [
                _source_array(batch, field.name)
                for field in capture_graph_training.PREDICTION_SCHEMA
                if field.name != "score"
            ]
            output_arrays.extend([
                pa.array(source_scores, type=pa.float32()),
                pa.array(recomputed_scores, type=pa.float32()),
                pa.array(arrays["raw_logit"][offset : offset + size], type=pa.float32()),
                pa.array(
                    arrays["score_float64"][offset : offset + size],
                    type=pa.float64(),
                ),
            ])
            writer.write_table(pa.Table.from_arrays(output_arrays, schema=OUTPUT_SCHEMA))
            offset += size
    finally:
        writer.close()
    if offset != len(arrays["raw_logit"]):
        raise ValueError(f"Rescored prediction enrichment is incomplete: {scenario}")
    return {
        "source_probability_exact_match_count": exact_matches,
        "source_probability_exact_match_fraction": exact_matches / offset,
        "source_probability_mean_absolute_difference": (
            absolute_difference_sum / offset
        ),
        "source_probability_maximum_absolute_difference": (
            maximum_absolute_difference
        ),
        "source_probability_exact_one_count": source_exact_one_count,
        "recomputed_probability_exact_one_count": recomputed_exact_one_count,
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
            "peak_cuda_allocated_gib": (
                torch.cuda.max_memory_allocated(device) / 1024**3
            ),
            "peak_cuda_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3,
        })
    return result


def _validate_completed_job(job_dir: Path, expected_contract_sha256: str) -> dict:
    completion_path = job_dir / "completion.json"
    completion = _load_json(completion_path, "rescoring-job completion")
    if (
        completion.get("status") != "complete"
        or completion.get("job_contract_sha256") != expected_contract_sha256
        or completion.get("model_training_performed") is not False
        or completion.get("threshold_selected") is not False
        or completion.get("held_out_scenarios_accessed") is not False
    ):
        raise ValueError(f"Invalid completed rescoring job: {job_dir.name}")
    for relative, expected in completion.get("artifact_checksums", {}).items():
        if sha256_file(job_dir / relative) != expected:
            raise ValueError(f"Completed rescoring artifact changed: {relative}")
    return completion


def run_capture_graph_oof_rescoring_job(
    *,
    job_id: str,
    config_path: str | Path,
    comparison_dir: str | Path,
    saturation_audit_dir: str | Path,
    training_root: str | Path,
    prepared_run_dir: str | Path,
    collection: CaptureGraphCollection,
    output_root: str | Path,
    local_work_root: str | Path,
) -> dict:
    """Repeat one immutable model/fold OOF inference and preserve raw logits."""
    started = time.perf_counter()
    config_path = Path(config_path).expanduser().resolve()
    comparison_dir = Path(comparison_dir).expanduser().resolve()
    saturation_audit_dir = Path(saturation_audit_dir).expanduser().resolve()
    training_root = Path(training_root).expanduser().resolve()
    prepared_run_dir = Path(prepared_run_dir).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    local_work_root = Path(local_work_root).expanduser().resolve()
    config = load_graph_oof_rescoring_config(config_path)
    if job_id not in expected_rescoring_job_ids(config):
        raise ValueError(f"Unknown rescoring job: {job_id}")
    model_name, fold_suffix = job_id.rsplit("__fold_", 1)
    fold = fold_suffix
    comparison, audit = _validate_source_context(
        config=config,
        comparison_dir=comparison_dir,
        saturation_audit_dir=saturation_audit_dir,
    )
    run_contract = _run_contract(
        output_root=output_root,
        config_path=config_path,
        config=config,
        comparison_dir=comparison_dir,
        saturation_audit_dir=saturation_audit_dir,
        training_root=training_root,
        prepared_run_dir=prepared_run_dir,
        collection=collection,
    )
    run_contract_sha256 = _bind_run_directory(output_root, run_contract)
    source = _source_job_context(
        job_id=job_id,
        model_name=model_name,
        fold=fold,
        config=config,
        comparison=comparison,
        audit=audit,
        training_root=training_root,
        collection=collection,
    )
    job_contract = {
        "report_version": REPORT_VERSION,
        "job_id": job_id,
        "model": model_name,
        "fold": fold,
        "run_contract_sha256": run_contract_sha256,
        "source_training_completion_sha256": sha256_file(
            source["completion_path"]
        ),
        "source_training_job_contract_sha256": sha256_file(source["contract_path"]),
        "source_training_metrics_sha256": sha256_file(source["metrics_path"]),
        "source_final_checkpoint_sha256": sha256_file(source["checkpoint_path"]),
        "source_oof_sha256": {
            scenario: item["sha256"]
            for scenario, item in source["source_oof"].items()
        },
        "model_specification": source["contract"]["model_specification"],
        "outer_validation_scenarios": source["expected_scenarios"],
        "inference": config["inference"],
        "model_training_performed": False,
        "threshold_selected": False,
        "held_out_scenarios_accessed": False,
    }
    job_contract_sha256 = _canonical_sha256(job_contract)
    job_dir = output_root / job_id
    completion_path = job_dir / "completion.json"
    if completion_path.is_file():
        completion = _validate_completed_job(job_dir, job_contract_sha256)
        print(f"Reusing completed immutable rescoring job: {job_id}", flush=True)
        return completion
    if job_dir.exists():
        if _load_json(job_dir / "job_contract.json", "rescoring job contract") != job_contract:
            raise FileExistsError("The existing rescoring job has a different contract.")
    else:
        job_dir.mkdir(parents=True)
        write_json(job_dir / "job_contract.json", job_contract)

    if config["inference"]["device"] != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Select a Colab GPU runtime for checkpoint rescoring.")
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    _set_seed(int(config["inference"]["seed"]))
    checkpoint = torch.load(
        source["checkpoint_path"], map_location=device, weights_only=False
    )
    if checkpoint.get("job_contract_sha256") != source["completion"].get(
        "job_contract_sha256"
    ):
        raise ValueError(f"The source checkpoint contract changed: {job_id}")
    model = _build_model(source["contract"]["model_specification"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    temporal = bool(source["contract"]["model_specification"]["temporal"])
    del checkpoint

    output_oof_dir = job_dir / "oof_logits"
    output_oof_dir.mkdir(exist_ok=True)
    status_path = job_dir / "rescore_status.json"
    status = _load_json(status_path, "rescoring status") if status_path.is_file() else {
        "job_contract_sha256": job_contract_sha256,
        "scenarios": {},
    }
    if status.get("job_contract_sha256") != job_contract_sha256:
        raise ValueError("The rescoring status has a different job contract.")
    local_work_root.mkdir(parents=True, exist_ok=True)
    scenario_metrics = {}
    tolerance = float(
        config["inference"][
            "maximum_allowed_source_probability_absolute_difference"
        ]
    )
    for scenario in source["expected_scenarios"]:
        output_path = output_oof_dir / f"{scenario}.parquet"
        existing = status["scenarios"].get(scenario)
        if existing is not None:
            if existing.get("predictions_sha256") != sha256_file(output_path):
                raise ValueError(f"Completed rescored predictions changed: {scenario}")
            scenario_metrics[scenario] = existing
            print(f"Reusing completed logit inference: {scenario}", flush=True)
            continue
        dataset = collection.scenario_dataset(
            fold,
            scenario,
            expected_partition="validation",
            verify_shard_checksums=False,
        )
        if int(dataset.report["edges"]) != source["source_oof"][scenario]["rows"]:
            raise ValueError(f"Graph/source OOF edge counts differ: {scenario}")
        with tempfile.TemporaryDirectory(
            dir=local_work_root,
            prefix=f"rescore_{model_name}_{fold}_{scenario}_",
        ) as temporary:
            temporary_dir = Path(temporary)
            arrays, metrics = _infer_high_resolution_scores(
                model=model,
                dataset=dataset,
                device=device,
                temporal=temporal,
                temporary_dir=temporary_dir,
            )
            temporary_output = temporary_dir / f"{scenario}.parquet"
            comparison_metrics = _write_rescored_predictions(
                output_path=temporary_output,
                source_path=source["source_oof"][scenario]["path"],
                arrays=arrays,
                model_name=model_name,
                fold=fold,
                scenario=scenario,
            )
            metrics.update(comparison_metrics)
            if metrics["source_probability_maximum_absolute_difference"] > tolerance:
                raise ValueError(
                    "Recomputed probabilities exceed the frozen numerical-drift "
                    f"tolerance for {job_id}/{scenario}: "
                    f"{metrics['source_probability_maximum_absolute_difference']:.9g}"
                )
            partial_path = output_oof_dir / f"{scenario}.parquet.partial"
            shutil.copyfile(temporary_output, partial_path)
            if sha256_file(partial_path) != sha256_file(temporary_output):
                raise IOError(f"Durable rescored prediction copy failed: {scenario}")
            partial_path.replace(output_path)
        metrics.update({
            "source_predictions": str(
                source["source_oof"][scenario]["path"].relative_to(training_root)
            ),
            "source_predictions_sha256": source["source_oof"][scenario]["sha256"],
            "predictions": f"oof_logits/{scenario}.parquet",
            "predictions_sha256": sha256_file(output_path),
            "operational_ranking_field": "raw_logit",
            "threshold_selected": False,
        })
        status["scenarios"][scenario] = metrics
        write_json(status_path, status)
        scenario_metrics[scenario] = metrics
        print(
            f"Completed logit inference for {scenario}: "
            f"AP={metrics['average_precision_raw_logit']:.6f}, "
            f"source max |delta|="
            f"{metrics['source_probability_maximum_absolute_difference']:.3g}, "
            f"{metrics['pure_inference_seconds']:.1f}s",
            flush=True,
        )

    job_report = {
        "report_version": REPORT_VERSION,
        "status": "complete",
        "job_id": job_id,
        "model": model_name,
        "fold": fold,
        "job_contract_sha256": job_contract_sha256,
        "source_final_checkpoint_sha256": job_contract[
            "source_final_checkpoint_sha256"
        ],
        "scenarios": scenario_metrics,
        "unweighted_mean_scenario_average_precision_raw_logit": float(np.mean([
            item["average_precision_raw_logit"]
            for item in scenario_metrics.values()
        ])),
        "total_edges": sum(int(item["edges"]) for item in scenario_metrics.values()),
        "pure_inference_seconds": sum(
            float(item["pure_inference_seconds"])
            for item in scenario_metrics.values()
        ),
        "source_rows_and_labels_verified": True,
        "source_packet_keys_copied_without_transformation": True,
        "model_training_performed": False,
        "optimizer_created": False,
        "checkpoint_modified": False,
        "threshold_selected": False,
        "held_out_scenarios_accessed": False,
    }
    write_json(job_dir / "job_report.json", job_report)
    resources = _resource_summary(device, started)
    resources["python"] = platform.python_version()
    resources["torch"] = torch.__version__
    resources["cuda_device"] = torch.cuda.get_device_name(device)
    write_json(job_dir / "resource_usage.json", resources)
    artifact_paths = [
        "job_contract.json",
        "rescore_status.json",
        "job_report.json",
        "resource_usage.json",
        *[f"oof_logits/{scenario}.parquet" for scenario in source["expected_scenarios"]],
    ]
    completion = {
        "report_version": REPORT_VERSION,
        "status": "complete",
        "job_id": job_id,
        "model": model_name,
        "fold": fold,
        "job_contract_sha256": job_contract_sha256,
        "source_final_checkpoint_sha256": job_contract[
            "source_final_checkpoint_sha256"
        ],
        "artifact_checksums": {
            relative: sha256_file(job_dir / relative) for relative in artifact_paths
        },
        "model_training_performed": False,
        "outer_validation_inference_repeated": True,
        "threshold_selected": False,
        "held_out_scenarios_accessed": False,
        "resource_usage": resources,
    }
    write_json(completion_path, completion)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Completed immutable OOF logit rescoring: {job_id}", flush=True)
    return completion


def finalize_capture_graph_oof_rescoring(
    *,
    config_path: str | Path,
    comparison_dir: str | Path,
    saturation_audit_dir: str | Path,
    training_root: str | Path,
    prepared_run_dir: str | Path,
    collection: CaptureGraphCollection,
    output_root: str | Path,
) -> dict:
    """Verify all eight jobs and seal the development-only rescoring run."""
    config_path = Path(config_path).expanduser().resolve()
    comparison_dir = Path(comparison_dir).expanduser().resolve()
    saturation_audit_dir = Path(saturation_audit_dir).expanduser().resolve()
    training_root = Path(training_root).expanduser().resolve()
    prepared_run_dir = Path(prepared_run_dir).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    config = load_graph_oof_rescoring_config(config_path)
    _validate_source_context(
        config=config,
        comparison_dir=comparison_dir,
        saturation_audit_dir=saturation_audit_dir,
    )
    contract = _run_contract(
        output_root=output_root,
        config_path=config_path,
        config=config,
        comparison_dir=comparison_dir,
        saturation_audit_dir=saturation_audit_dir,
        training_root=training_root,
        prepared_run_dir=prepared_run_dir,
        collection=collection,
    )
    run_contract_sha256 = _bind_run_directory(output_root, contract)
    existing_status = output_root / "run_status.json"
    if existing_status.is_file():
        return validate_capture_graph_oof_rescoring(
            config_path=config_path,
            comparison_dir=comparison_dir,
            saturation_audit_dir=saturation_audit_dir,
            output_root=output_root,
        )

    jobs = {}
    scenario_outputs = {model_name: {} for model_name in config["models"]}
    for job_id in expected_rescoring_job_ids(config):
        model_name, fold = job_id.rsplit("__fold_", 1)
        job_dir = output_root / job_id
        job_contract = _load_json(job_dir / "job_contract.json", f"{job_id} contract")
        if job_contract.get("run_contract_sha256") != run_contract_sha256:
            raise ValueError(f"The rescoring job belongs to another run: {job_id}")
        completion = _validate_completed_job(
            job_dir,
            _canonical_sha256(job_contract),
        )
        report = _load_json(job_dir / "job_report.json", f"{job_id} report")
        if report.get("job_contract_sha256") != completion["job_contract_sha256"]:
            raise ValueError(f"The rescoring report changed: {job_id}")
        jobs[job_id] = {
            "model": model_name,
            "fold": fold,
            "completion_sha256": sha256_file(job_dir / "completion.json"),
            "job_report_sha256": sha256_file(job_dir / "job_report.json"),
            "source_final_checkpoint_sha256": completion[
                "source_final_checkpoint_sha256"
            ],
            "pure_inference_seconds": report["pure_inference_seconds"],
            "total_edges": report["total_edges"],
        }
        for scenario, item in report["scenarios"].items():
            scenario_outputs[model_name][scenario] = {
                "fold": fold,
                "path": f"{job_id}/{item['predictions']}",
                "sha256": item["predictions_sha256"],
                "rows": item["edges"],
                "operational_ranking_field": item["operational_ranking_field"],
            }
    expected_scenarios = {
        scenario
        for fold in config["folds"].values()
        for scenario in fold["outer_validation"]
    }
    for model_name, outputs in scenario_outputs.items():
        if set(outputs) != expected_scenarios:
            raise ValueError(f"Incomplete scenario coverage for {model_name}.")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    manifest = {
        "report_version": REPORT_VERSION,
        "status": "complete",
        "run_id": output_root.name,
        "run_contract_sha256": run_contract_sha256,
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
        "models": list(config["models"]),
        "folds": config["folds"],
        "jobs": jobs,
        "scenario_outputs": scenario_outputs,
        "operational_ranking_field": "raw_logit",
        "source_probabilities_preserved": True,
        "float64_sigmoid_scores_preserved": True,
        "model_training_performed": False,
        "optimizer_created": False,
        "checkpoint_modified": False,
        "threshold_selected": False,
        "held_out_scenarios_accessed": False,
        "next_action": "run_the_versioned_operational_comparison_on_raw_logits",
    }
    manifest_path = output_root / "rescoring_manifest.json"
    write_json(manifest_path, manifest)
    write_json(output_root / "run_status.json", {
        "complete": True,
        "manifest_sha256": sha256_file(manifest_path),
        "run_contract_sha256": run_contract_sha256,
        "completed_jobs": expected_rescoring_job_ids(config),
        "model_training_performed": False,
        "threshold_selected": False,
        "held_out_scenarios_accessed": False,
    })
    return manifest


def validate_capture_graph_oof_rescoring(
    *,
    config_path: str | Path,
    comparison_dir: str | Path,
    saturation_audit_dir: str | Path,
    output_root: str | Path,
) -> dict:
    """Validate a sealed OOF logit-rescoring run without loading model data."""
    config_path = Path(config_path).expanduser().resolve()
    comparison_dir = Path(comparison_dir).expanduser().resolve()
    saturation_audit_dir = Path(saturation_audit_dir).expanduser().resolve()
    output_root = Path(output_root).expanduser().resolve()
    config = load_graph_oof_rescoring_config(config_path)
    _validate_source_context(
        config=config,
        comparison_dir=comparison_dir,
        saturation_audit_dir=saturation_audit_dir,
    )
    contract_path = output_root / "run_contract.json"
    contract = _load_json(contract_path, "rescoring run contract")
    status = _load_json(output_root / "run_status.json", "rescoring run status")
    manifest_path = output_root / "rescoring_manifest.json"
    manifest = _load_json(manifest_path, "rescoring manifest")
    if (
        contract.get("rescoring_contract_sha256") != sha256_file(config_path)
        or contract.get("source_comparison_report_sha256")
        != sha256_file(comparison_dir / "comparison_report.json")
        or contract.get("source_saturation_audit_report_sha256")
        != sha256_file(saturation_audit_dir / "score_saturation_audit.json")
        or contract.get("dependency_sha256") != _dependency_hashes()
        or status.get("complete") is not True
        or status.get("manifest_sha256") != sha256_file(manifest_path)
        or status.get("run_contract_sha256") != sha256_file(contract_path)
        or status.get("completed_jobs") != expected_rescoring_job_ids(config)
        or manifest.get("status") != "complete"
        or manifest.get("run_id") != output_root.name
        or manifest.get("run_contract_sha256") != sha256_file(contract_path)
        or manifest.get("operational_ranking_field") != "raw_logit"
        or manifest.get("model_training_performed") is not False
        or manifest.get("threshold_selected") is not False
        or manifest.get("held_out_scenarios_accessed") is not False
    ):
        raise ValueError("The sealed rescoring run has unexpected provenance.")
    for job_id in expected_rescoring_job_ids(config):
        job_dir = output_root / job_id
        job_contract = _load_json(job_dir / "job_contract.json", f"{job_id} contract")
        completion = _validate_completed_job(
            job_dir,
            _canonical_sha256(job_contract),
        )
        if sha256_file(job_dir / "completion.json") != manifest["jobs"][job_id][
            "completion_sha256"
        ]:
            raise ValueError(f"The sealed job completion changed: {job_id}")
        if completion["source_final_checkpoint_sha256"] != manifest["jobs"][job_id][
            "source_final_checkpoint_sha256"
        ]:
            raise ValueError(f"The sealed source checkpoint changed: {job_id}")
    return manifest
