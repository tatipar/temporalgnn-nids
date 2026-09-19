"""Window-close context features and fold-local XGB-P+T training."""

from __future__ import annotations

from collections import Counter, deque
import gc
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .capture_data import sha256_file, write_json
from .capture_feature_profile import (
    load_prepared_full_dev, load_preprocessing_schema, preprocessing_schema_sha256,
)
from .capture_xgb_p import (
    OOF_COLUMNS, OOF_SCHEMA, _load_fold_preprocessor, _scenario_metrics,
    _validated_labels, scenario_class_weights, window_coordinates,
    xgb_p_configuration,
    validate_xgb_p_fold_run, summarize_xgb_p_oof,
)


CONTEXT_COLUMNS = (
    "window_packet_count", "source_sent_packet_count",
    "destination_received_packet_count", "directed_pair_packet_count",
    "source_unique_destinations", "destination_unique_sources",
    "window_mqtt_fraction", "window_mean_frame_length",
    "previous_30s_packet_count", "previous_30s_source_sent_packet_count",
    "previous_30s_destination_received_packet_count",
    "previous_30s_directed_pair_packet_count",
    "previous_30s_source_unique_destinations",
    "previous_30s_destination_unique_sources",
)
CONTEXT_SCHEMA = pa.schema([("source_row_id", pa.int64())] + [
    (name, pa.float32()) for name in CONTEXT_COLUMNS
])
INPUT_COLUMNS = (
    "source_row_id", "packet_timestamp_ns", "src_endpoint", "dst_endpoint",
    "is_mqtt", "frame_length",
)
CONTEXT_VERSION = 1
ABLATION_VARIANTS = {
    "full": CONTEXT_COLUMNS,
    "current_window": CONTEXT_COLUMNS[:8],
    "history": CONTEXT_COLUMNS[8:],
}


def context_features_for_variant(manifest: dict, variant_name: str) -> tuple[str, ...]:
    """Resolve a declared context ablation without changing feature order."""
    validate_context_contract(manifest)
    if variant_name not in ABLATION_VARIANTS:
        raise ValueError(f"Undeclared XGB-P+T variant: {variant_name}")
    policy = manifest["training"]["xgb_p_t_ablation"]
    if (policy["status"] != "frozen_before_ablation_results"
            or policy["train_new_variants"] != ["current_window", "history"]
            or policy["reuse_existing_variants"] != ["xgb_p", "full"]):
        raise ValueError("The XGB-P+T ablation policy changed.")
    declared = policy["variants"]
    features = ABLATION_VARIANTS[variant_name]
    if list(features) != declared[variant_name]:
        raise ValueError("The context ablation differs from the declared feature set.")
    return features


def validate_context_contract(manifest: dict) -> dict:
    """Reject silent changes to the predeclared 5-second/30-second protocol."""
    spec = manifest["training"]["temporal_summary_features_and_horizons"]
    if (spec["status"] != "frozen_for_xgb_p_t_implementation"
            or spec["source"] != "canonical_packets_without_labels_or_evaluation_metadata"
            or spec["window_width_seconds"] != 5
            or spec["history_window_count"] != 6
            or spec["history_seconds"] != 30
            or spec["current_window_scope"] != "complete_half_open_window_available_at_window_end"
            or spec["history_scope"] != "six_complete_preceding_wall_clock_windows_excluding_current"
            or spec["unique_peer_rule"] != "count_distinct_over_entire_horizon_not_sum_of_window_counts"
            or spec["empty_history_value"] != 0
            or tuple(spec["current_window_features"] + spec["history_features"])
            != CONTEXT_COLUMNS
            or spec["state_reset_scope"] != "scenario_boundary"
            or spec["endpoint_identity_policy"] != "grouping_keys_only_never_model_values"
            or spec["numeric_transform"] != "log1p_then_training_fold_standardization_for_all_14_context_fields"
            or spec["shared_context_inputs_for_xgb_and_graph_models"] is not True
            or manifest["windows"]["origin_rule"] != "first_packet_timestamp_per_scenario"
            or manifest["windows"]["selected_duration_seconds"] != 5
            or manifest["windows"]["decision_time"] != "window_end"):
        raise ValueError("The XGB-P+T context contract differs from the frozen proposal.")
    return spec


def _context_contract_sha256(manifest: dict) -> str:
    spec = validate_context_contract(manifest)
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _window_summary(frame: pd.DataFrame) -> dict:
    sources = frame["src_endpoint"]
    destinations = frame["dst_endpoint"]
    pairs = pd.MultiIndex.from_arrays([sources, destinations])
    return {
        "packet_count": len(frame),
        "source_counts": Counter(sources),
        "destination_counts": Counter(destinations),
        "pair_counts": Counter(pairs),
        "pairs": set(pairs),
    }


def context_for_window(frame: pd.DataFrame, history: deque[tuple[int, dict]],
                       window_index: int) -> pd.DataFrame:
    """Derive current and preceding wall-clock-window context for each row."""
    if frame.empty:
        raise ValueError("A materialized window must contain packets.")
    if frame[list(INPUT_COLUMNS)].isna().any().any():
        raise ValueError("Context input contains null packet fields.")
    frame_length = pd.to_numeric(frame["frame_length"], errors="raise")
    mqtt = pd.to_numeric(frame["is_mqtt"], errors="raise")
    if (not np.isfinite(frame_length).all() or (frame_length < 0).any()
            or not mqtt.isin([0, 1]).all()):
        raise ValueError("Invalid frame length or MQTT indicator in context input.")
    sources = frame["src_endpoint"].tolist()
    destinations = frame["dst_endpoint"].tolist()
    pairs = list(zip(sources, destinations))
    current = _window_summary(frame)
    preceding = [item for index, item in history if window_index - 6 <= index < window_index]
    prior_sources = Counter()
    prior_destinations = Counter()
    prior_pairs = Counter()
    prior_unique_pairs = set()
    for item in preceding:
        prior_sources.update(item["source_counts"])
        prior_destinations.update(item["destination_counts"])
        prior_pairs.update(item["pair_counts"])
        prior_unique_pairs.update(item["pairs"])
    current_source_peers = Counter(source for source, _ in current["pairs"])
    current_destination_peers = Counter(destination for _, destination in current["pairs"])
    prior_source_peers = Counter(source for source, _ in prior_unique_pairs)
    prior_destination_peers = Counter(destination for _, destination in prior_unique_pairs)
    size = len(frame)
    values = {
        "window_packet_count": np.full(size, size),
        "source_sent_packet_count": [current["source_counts"][source] for source in sources],
        "destination_received_packet_count": [current["destination_counts"][destination] for destination in destinations],
        "directed_pair_packet_count": [current["pair_counts"][pair] for pair in pairs],
        "source_unique_destinations": [current_source_peers[source] for source in sources],
        "destination_unique_sources": [current_destination_peers[destination] for destination in destinations],
        "window_mqtt_fraction": np.full(size, mqtt.mean()),
        "window_mean_frame_length": np.full(size, frame_length.mean()),
        "previous_30s_packet_count": np.full(size, sum(item["packet_count"] for item in preceding)),
        "previous_30s_source_sent_packet_count": [prior_sources[source] for source in sources],
        "previous_30s_destination_received_packet_count": [prior_destinations[destination] for destination in destinations],
        "previous_30s_directed_pair_packet_count": [prior_pairs[pair] for pair in pairs],
        "previous_30s_source_unique_destinations": [prior_source_peers[source] for source in sources],
        "previous_30s_destination_unique_sources": [prior_destination_peers[destination] for destination in destinations],
    }
    result = pd.DataFrame({"source_row_id": frame["source_row_id"].to_numpy(dtype=np.int64)})
    for name in CONTEXT_COLUMNS:
        result[name] = np.asarray(values[name], dtype=np.float32)
    if not np.isfinite(result[list(CONTEXT_COLUMNS)].to_numpy()).all():
        raise ValueError("Context features must be finite.")
    history.append((window_index, current))
    while history and history[0][0] <= window_index - 6:
        history.popleft()
    return result


def build_context_scenario(packet_path: Path, output_path: Path, *,
                           scenario: str, report: dict, batch_size: int = 50_000) -> dict:
    """Stream one scenario and write one context row per source packet."""
    if batch_size <= 0 or output_path.exists():
        raise ValueError("Batch size must be positive and output must be new.")
    parquet = pq.ParquetFile(packet_path)
    if not set(INPUT_COLUMNS) <= set(parquet.schema_arrow.names):
        raise ValueError("Prepared packets lack required context fields.")
    origin = int(report["scenario_origin_timestamp_ns"])
    expected = int(report["counts"]["packets"])
    history = deque()
    pending = []
    active_index = None
    previous_timestamp = None
    rows_read = rows_written = windows = 0
    writer = None

    def flush() -> None:
        nonlocal rows_written, windows, writer
        if not pending:
            return
        frame = pd.concat(pending, ignore_index=True)
        result = context_for_window(frame, history, active_index)
        table = pa.Table.from_pandas(result, schema=CONTEXT_SCHEMA, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, CONTEXT_SCHEMA, compression="zstd")
        writer.write_table(table)
        rows_written += len(frame)
        windows += 1
        pending.clear()

    try:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=list(INPUT_COLUMNS)):
            frame = batch.to_pandas()
            row_ids = frame["source_row_id"].to_numpy(dtype=np.int64)
            if not np.array_equal(row_ids, np.arange(rows_read, rows_read + len(frame))):
                raise ValueError(f"Packet row order changed in {scenario}.")
            timestamps = frame["packet_timestamp_ns"].to_numpy(dtype=np.int64)
            if np.any(np.diff(timestamps) < 0) or (previous_timestamp is not None and timestamps[0] < previous_timestamp):
                raise ValueError(f"Packet time order changed in {scenario}.")
            previous_timestamp = int(timestamps[-1])
            indexes, _ = window_coordinates(timestamps, origin, 5)
            changes = np.flatnonzero(np.diff(indexes)) + 1
            for part in np.split(np.arange(len(frame)), changes):
                index = int(indexes[part[0]])
                if active_index is not None and index != active_index:
                    flush()
                active_index = index
                pending.append(frame.iloc[part].reset_index(drop=True))
            rows_read += len(frame)
        flush()
    finally:
        if writer is not None:
            writer.close()
    if rows_read != expected or rows_written != expected or not output_path.is_file():
        raise ValueError(f"Context row count differs from prepared packets in {scenario}.")
    return {"rows": rows_written, "nonempty_windows": windows,
            "context_artifact": output_path.name, "context_sha256": sha256_file(output_path)}


def _context_batches(path: Path, batch_size: int):
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow != CONTEXT_SCHEMA:
        raise ValueError("Context artifact has an unexpected schema or column order.")
    yield from (batch.to_pandas() for batch in parquet.iter_batches(batch_size=batch_size))


def fit_context_scaler(paths: list[Path], batch_size: int,
                       feature_names: tuple[str, ...] = CONTEXT_COLUMNS) -> dict:
    """Fit log1p and standardization parameters on training scenarios only."""
    if not feature_names or not set(feature_names) <= set(CONTEXT_COLUMNS):
        raise ValueError("Context scaler features must be a nonempty declared subset.")
    count = 0
    sums = np.zeros(len(feature_names), dtype=np.float64)
    squared = np.zeros_like(sums)
    minima = np.full(len(feature_names), np.inf)
    maxima = np.full(len(feature_names), -np.inf)
    for path in paths:
        for batch in _context_batches(path, batch_size):
            raw = batch[list(feature_names)].to_numpy(dtype=np.float64)
            if not np.isfinite(raw).all() or (raw < 0).any():
                raise ValueError("Context features must be finite and nonnegative.")
            values = np.log1p(raw)
            count += len(values)
            sums += values.sum(axis=0)
            squared += np.square(values).sum(axis=0)
            minima = np.minimum(minima, values.min(axis=0))
            maxima = np.maximum(maxima, values.max(axis=0))
    if count == 0:
        raise ValueError("Cannot fit context preprocessing on zero rows.")
    mean = sums / count
    standard_deviation = np.sqrt(np.maximum(squared / count - np.square(mean), 0))
    standard_deviation[(standard_deviation == 0) | (minima == maxima)] = 1
    return {"training_rows": count, "feature_names": list(feature_names),
            "transform": "log1p_then_training_fold_standardization",
            "mean": mean.tolist(), "standard_deviation": standard_deviation.tolist()}


def transform_context(batch: pd.DataFrame, scaler: dict) -> np.ndarray:
    values = batch[scaler["feature_names"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Context features must be finite and nonnegative.")
    result = (np.log1p(values) - scaler["mean"]) / scaler["standard_deviation"]
    if not np.isfinite(result).all():
        raise ValueError("Transformed context features must be finite.")
    return result.astype(np.float32)


def _paired_batches(packet_path: Path, context_path: Path, columns: list[str],
                    batch_size: int, right_columns: list[str] | None = None):
    packet_file = pq.ParquetFile(packet_path)
    context_file = pq.ParquetFile(context_path)
    if packet_file.metadata.num_rows != context_file.metadata.num_rows:
        raise ValueError("Packet and context artifacts have different row counts.")
    packets = packet_file.iter_batches(batch_size=batch_size, columns=columns)
    context = iter(context_file.iter_batches(
        batch_size=batch_size, columns=right_columns))
    pending = None
    for packet_batch in packets:
        packet_frame = packet_batch.to_pandas()
        parts = []
        remaining = len(packet_frame)
        while remaining:
            if pending is None or len(pending) == 0:
                pending = next(context, None)
            if pending is None:
                raise ValueError("Context artifact ended before packet artifact.")
            taken = min(remaining, len(pending))
            parts.append(pending.slice(0, taken))
            pending = pending.slice(taken)
            remaining -= taken
        context_frame = pa.Table.from_batches(parts).to_pandas()
        if not np.array_equal(packet_frame["source_row_id"].to_numpy(), context_frame["source_row_id"].to_numpy()):
            raise ValueError("Packet and context source-row IDs are misaligned.")
        yield packet_frame, context_frame
    if (pending is not None and len(pending)) or next(context, None) is not None:
        raise ValueError("Context artifact contains trailing rows.")


def _load_context_run(context_dir: Path, manifest: dict, reports: dict,
                      packet_paths: dict) -> dict:
    status_path = context_dir / "run_status.json"
    report_path = context_dir / "context_report.json"
    if not status_path.is_file() or not report_path.is_file():
        raise FileNotFoundError("Completed context artifacts are required.")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("complete") is not True or status.get("report_sha256") != sha256_file(report_path):
        raise ValueError("Context run status or report checksum is invalid.")
    result = json.loads(report_path.read_text(encoding="utf-8"))
    validate_context_contract(manifest)
    if result.get("context_version") != CONTEXT_VERSION:
        raise ValueError("Unsupported context artifact version.")
    if result.get("context_contract_sha256") != _context_contract_sha256(manifest):
        raise ValueError("Context artifacts use a different feature contract.")
    for scenario, packet_path in packet_paths.items():
        item = result["scenarios"][scenario]
        context_path = context_dir / item["context_artifact"]
        if (item["rows"] != reports[scenario]["counts"]["packets"]
                or item["prepared_packet_sha256"] != reports[scenario]["output_sha256"]
                or sha256_file(context_path) != item["context_sha256"]):
            raise ValueError(f"Context provenance differs for {scenario}.")
    return result


def build_context_run(*, manifest_path: Path, packet_schema_path: Path,
                      prepared_run_dir: Path, output_dir: Path,
                      batch_size: int = 50_000) -> dict:
    """Build immutable, label-free context for all five development scenarios."""
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite context run: {output_dir}")
    manifest, _, reports, packet_paths = load_prepared_full_dev(
        prepared_run_dir, manifest_path, packet_schema_path)
    validate_context_contract(manifest)
    output_dir.mkdir(parents=True)
    scenario_reports = {}
    for scenario, packet_path in packet_paths.items():
        artifact = output_dir / f"context_{scenario}.parquet"
        item = build_context_scenario(packet_path, artifact, scenario=scenario,
                                      report=reports[scenario], batch_size=batch_size)
        scenario_reports[scenario] = {
            **item, "prepared_packet_sha256": reports[scenario]["output_sha256"]}
    result = {"context_version": CONTEXT_VERSION,
              "context_contract_sha256": _context_contract_sha256(manifest),
              "generator_code_sha256": sha256_file(Path(__file__)),
              "manifest_sha256": sha256_file(manifest_path),
              "packet_schema_sha256": sha256_file(packet_schema_path),
              "prepared_run": str(prepared_run_dir),
              "context_columns": list(CONTEXT_COLUMNS),
              "scenarios": scenario_reports}
    write_json(output_dir / "context_report.json", result)
    write_json(output_dir / "run_status.json", {
        "complete": True, "report_sha256": sha256_file(output_dir / "context_report.json")})
    return result


def validate_context_run(*, manifest_path: Path, packet_schema_path: Path,
                         prepared_run_dir: Path, context_dir: Path) -> dict:
    """Verify a completed context run and all five source artifact bindings."""
    manifest, _, reports, packet_paths = load_prepared_full_dev(
        prepared_run_dir, manifest_path, packet_schema_path)
    return _load_context_run(context_dir, manifest, reports, packet_paths)


def validate_xgb_p_t_fold_run(directory: Path, fold: str,
                              variant_name: str = "full") -> dict:
    """Verify a persisted model, scaler, OOF files, and context provenance."""
    report = validate_xgb_p_fold_run(directory, fold, "depth5_primary")
    if variant_name not in ABLATION_VARIANTS:
        raise ValueError(f"Undeclared XGB-P+T variant: {variant_name}")
    expected_family = "xgb_p_t" if variant_name == "full" else "xgb_p_t_ablation"
    expected_features = list(ABLATION_VARIANTS[variant_name])
    if (report.get("model_family") != expected_family
            or report.get("variant_name", "full") != variant_name
            or report.get("feature_count") != 103 + len(expected_features)
            or report.get("feature_names", [])[103:] != expected_features):
        raise ValueError("Fold report does not describe the XGB-P+T model view.")
    if sha256_file(directory / report["context_scaler_artifact"]) != report["context_scaler_sha256"]:
        raise ValueError("The fold context scaler changed after training.")
    return report


def summarize_xgb_p_t_oof(run_dir: Path, variant_name: str = "full") -> dict:
    """Compute the same hierarchical development OOF summary as XGB-P."""
    reports = {
        fold: validate_xgb_p_t_fold_run(
            run_dir / "depth5_primary" / f"fold_{fold}", fold, variant_name)
        for fold in ("A", "B")
    }
    if reports["A"]["context_report_sha256"] != reports["B"]["context_report_sha256"]:
        raise ValueError("The folds used different context artifacts.")
    return summarize_xgb_p_oof(run_dir, "depth5_primary")


def compare_xgb_p_t_to_xgb_p(*, baseline_run_dir: Path, context_run_dir: Path,
                             batch_size: int = 50_000) -> dict:
    """Verify identical OOF packet keys and compare unthresholded ranking."""
    baseline_summary = summarize_xgb_p_oof(baseline_run_dir, "depth5_primary")
    context_summary = summarize_xgb_p_t_oof(context_run_dir)
    key_columns = ["packet_id", "source_row_id", "packet_timestamp_ns",
                   "binary_label", "window_index", "window_end_ns"]
    comparison = {}
    for fold in ("A", "B"):
        baseline = validate_xgb_p_fold_run(
            baseline_run_dir / "depth5_primary" / f"fold_{fold}",
            fold, "depth5_primary")
        context = validate_xgb_p_t_fold_run(
            context_run_dir / "depth5_primary" / f"fold_{fold}", fold)
        if set(baseline["validation"]) != set(context["validation"]):
            raise ValueError("XGB-P and XGB-P+T validated different scenarios.")
        for scenario, baseline_item in baseline["validation"].items():
            context_item = context["validation"][scenario]
            if baseline_item["rows"] != context_item["rows"]:
                raise ValueError(f"OOF row counts differ for {scenario}.")
            baseline_path = (baseline_run_dir / "depth5_primary" / f"fold_{fold}"
                             / baseline_item["oof_artifact"])
            context_path = (context_run_dir / "depth5_primary" / f"fold_{fold}"
                            / context_item["oof_artifact"])
            checked = 0
            for baseline_batch, context_batch in _paired_batches(
                    baseline_path, context_path, key_columns, batch_size,
                    right_columns=key_columns):
                for name in key_columns:
                    if not baseline_batch[name].equals(context_batch[name]):
                        raise ValueError(f"OOF {name} differs for {scenario}.")
                checked += len(baseline_batch)
            if checked != baseline_item["rows"]:
                raise ValueError(f"OOF packet coverage differs for {scenario}.")
            comparison[scenario] = {
                "fold": fold, "packets": checked,
                "xgb_p_packet_roc_auc": baseline_item["packet_roc_auc"],
                "xgb_p_t_packet_roc_auc": context_item["packet_roc_auc"],
                "delta_packet_roc_auc": (
                    context_item["packet_roc_auc"] - baseline_item["packet_roc_auc"]),
            }
    return {
        "aligned_oof_scenarios": comparison,
        "xgb_p_hierarchical_macro_oof_packet_roc_auc":
            baseline_summary["hierarchical_macro_oof_packet_roc_auc"],
        "xgb_p_t_hierarchical_macro_oof_packet_roc_auc":
            context_summary["hierarchical_macro_oof_packet_roc_auc"],
        "delta_hierarchical_macro_oof_packet_roc_auc": (
            context_summary["hierarchical_macro_oof_packet_roc_auc"]
            - baseline_summary["hierarchical_macro_oof_packet_roc_auc"]),
        "thresholds_selected": False,
    }


def run_xgb_p_t_fold(*, manifest_path: Path, packet_schema_path: Path,
                     preprocessing_schema_path: Path, prepared_run_dir: Path,
                     preprocessing_audit_dir: Path, context_dir: Path,
                     output_dir: Path, local_work_root: Path, fold: str,
                     configuration_name: str = "depth5_primary",
                     batch_size: int = 50_000, nthread: int = 2,
                     variant_name: str = "full") -> dict:
    """Train one fold using the frozen packet view plus reviewed context."""
    import xgboost as xgb

    if batch_size <= 0 or nthread <= 0:
        raise ValueError("Batch size and CPU thread count must be positive.")
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output_dir}")
    manifest, packet_schema, reports, packet_paths = load_prepared_full_dev(
        prepared_run_dir, manifest_path, packet_schema_path)
    if manifest["readiness"]["xgb_p_sanity_gate_passed"] is not True:
        raise ValueError("XGB-P sanity gate must be documented and passed before XGB-P+T training.")
    review = manifest["readiness"].get("xgb_p_sanity_review", {})
    if (review.get("decision") != "pass_for_xgb_p_t_development_training"
            or review.get("shuffled_label_hierarchical_macro_roc_auc", 1)
            > review.get("review_line", 0.6)):
        raise ValueError("XGB-P sanity review is missing or violates the negative-control line.")
    context_feature_names = context_features_for_variant(manifest, variant_name)
    if fold not in manifest["validation"]["folds"]:
        raise ValueError(f"Undeclared development fold: {fold}")
    split = manifest["validation"]["folds"][fold]
    train_scenarios = list(split["train"])
    validation_scenarios = list(split["validate"])
    parameters = xgb_p_configuration(manifest, configuration_name)
    if configuration_name != "depth5_primary":
        raise ValueError("The primary XGB-P+T comparison uses depth5_primary.")
    schema = load_preprocessing_schema(preprocessing_schema_path, packet_schema)
    if schema["status"] != "frozen":
        raise ValueError("XGB-P+T requires frozen packet preprocessing.")
    preprocessor, preprocessor_sha256 = _load_fold_preprocessor(
        preprocessing_audit_dir, schema, fold, train_scenarios, prepared_run_dir)
    context_report = _load_context_run(context_dir, manifest, reports, packet_paths)
    context_paths = {
        scenario: context_dir / item["context_artifact"]
        for scenario, item in context_report["scenarios"].items()
    }
    training_rows = sum(int(reports[name]["counts"]["packets"]) for name in train_scenarios)
    if preprocessor.training_rows != training_rows:
        raise ValueError("Packet preprocessing was fitted on a different fold row count.")
    scaler = fit_context_scaler(
        [context_paths[name] for name in train_scenarios], batch_size,
        context_feature_names)
    if scaler["training_rows"] != training_rows:
        raise ValueError("Context preprocessing was fitted on a different fold row count.")
    weights = scenario_class_weights(train_scenarios, reports)
    feature_names = [*preprocessor.feature_names, *context_feature_names]
    feature_count = 103 + len(context_feature_names)
    if len(feature_names) != feature_count or len(set(feature_names)) != feature_count:
        raise ValueError("The XGB-P+T model view has an invalid feature order.")
    local_work_root.mkdir(parents=True, exist_ok=True)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    repository_root = manifest_path.resolve().parent.parent
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root,
        capture_output=True, text=True, check=False)
    worktree = subprocess.run(
        ["git", "status", "--short"], cwd=repository_root,
        capture_output=True, text=True, check=False)
    with tempfile.TemporaryDirectory(prefix="capture_xgb_p_t_", dir=local_work_root) as temporary:
        stage = Path(temporary)
        required_bytes = training_rows * (feature_count * 4 + 1 + 4) + 1_000_000_000
        if shutil.disk_usage(stage).free < required_bytes:
            raise OSError("Insufficient local storage for XGB-P+T training matrix.")
        features = np.lib.format.open_memmap(
            stage / "features.npy", mode="w+", dtype=np.float32,
            shape=(training_rows, feature_count))
        labels = np.lib.format.open_memmap(
            stage / "labels.npy", mode="w+", dtype=np.uint8,
            shape=(training_rows,))
        sample_weights = np.lib.format.open_memmap(
            stage / "weights.npy", mode="w+", dtype=np.float32,
            shape=(training_rows,))
        offset = 0
        for scenario in train_scenarios:
            scenario_rows = normal = attack = 0
            columns = list(dict.fromkeys([*preprocessor.required_columns,
                                           "source_row_id", "binary_label"]))
            for packet_batch, context_batch in _paired_batches(
                    packet_paths[scenario], context_paths[scenario], columns, batch_size):
                size = len(packet_batch)
                if not np.array_equal(packet_batch["source_row_id"].to_numpy(),
                                      np.arange(scenario_rows, scenario_rows + size)):
                    raise ValueError(f"Training packet order changed in {scenario}.")
                batch_labels = _validated_labels(packet_batch, scenario)
                packet_values = preprocessor.transform(packet_batch).to_numpy(dtype=np.float32)
                context_values = transform_context(context_batch, scaler)
                features[offset:offset + size, :103] = packet_values
                features[offset:offset + size, 103:] = context_values
                labels[offset:offset + size] = batch_labels
                sample_weights[offset:offset + size] = np.where(
                    batch_labels == 0, weights[scenario][0], weights[scenario][1])
                normal += int(np.count_nonzero(batch_labels == 0))
                attack += int(np.count_nonzero(batch_labels == 1))
                scenario_rows += size
                offset += size
            expected_counts = reports[scenario]["counts"]
            if (scenario_rows != expected_counts["packets"]
                    or normal != expected_counts["normal_packets"]
                    or attack != expected_counts["attack_packets"]):
                raise ValueError(f"Training row or label counts differ for {scenario}.")
        if offset != training_rows or not np.isfinite(features).all():
            raise ValueError("Training matrix is incomplete or non-finite.")
        features.flush()
        labels.flush()
        sample_weights.flush()
        train_matrix = xgb.QuantileDMatrix(
            features, label=labels, weight=sample_weights,
            feature_names=feature_names, max_bin=int(parameters["max_bin"]),
            nthread=nthread)
        xgb_parameters = {
            "objective": parameters["objective"], "tree_method": parameters["tree_method"],
            "device": parameters["device"], "max_depth": int(parameters["max_depth"]),
            "eta": float(parameters["learning_rate"]),
            "min_child_weight": float(parameters["min_child_weight"]),
            "lambda": float(parameters["reg_lambda"]), "max_bin": int(parameters["max_bin"]),
            "subsample": float(parameters["subsample"]),
            "colsample_bytree": float(parameters["colsample_bytree"]),
            "colsample_bylevel": float(parameters["colsample_bylevel"]),
            "colsample_bynode": float(parameters["colsample_bynode"]),
            "seed": 42, "nthread": nthread,
        }
        model = xgb.train(xgb_parameters, train_matrix,
                          num_boost_round=int(parameters["num_boost_round"]))
        if model.num_boosted_rounds() != int(parameters["num_boost_round"]):
            raise ValueError("The XGB-P+T model has an unexpected number of trees.")
        model_path = stage / "model.json"
        model.save_model(model_path)
        write_json(stage / "context_scaler.json", scaler)
        write_json(stage / "feature_importance_gain.json", {
            name: float(model.get_score(importance_type="gain").get(name, 0))
            for name in feature_names})
        del train_matrix, features, labels, sample_weights
        gc.collect()
        validation = {}
        for scenario in validation_scenarios:
            path = stage / f"oof_{scenario}.parquet"
            columns = list(dict.fromkeys([*preprocessor.required_columns, *OOF_COLUMNS]))
            origin = int(reports[scenario]["scenario_origin_timestamp_ns"])
            writer = None
            rows = normal = attack = 0
            try:
                for packet_batch, context_batch in _paired_batches(
                        packet_paths[scenario], context_paths[scenario], columns, batch_size):
                    size = len(packet_batch)
                    if not np.array_equal(packet_batch["source_row_id"].to_numpy(),
                                          np.arange(rows, rows + size)):
                        raise ValueError(f"Validation packet order changed in {scenario}.")
                    batch_labels = _validated_labels(packet_batch, scenario)
                    values = np.concatenate([
                        preprocessor.transform(packet_batch).to_numpy(dtype=np.float32),
                        transform_context(context_batch, scaler)], axis=1)
                    scores = np.asarray(model.inplace_predict(values), dtype=np.float32)
                    if scores.shape != (size,) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                        raise ValueError(f"Invalid XGB-P+T scores in {scenario}.")
                    indexes, ends = window_coordinates(
                        packet_batch["packet_timestamp_ns"].to_numpy(dtype=np.int64),
                        origin, 5)
                    output = packet_batch.loc[:, list(OOF_COLUMNS)].copy()
                    output["window_index"] = indexes
                    output["window_end_ns"] = ends
                    output["score"] = scores
                    table = pa.Table.from_pandas(output, schema=OOF_SCHEMA,
                                                 preserve_index=False, safe=True)
                    if writer is None:
                        writer = pq.ParquetWriter(path, OOF_SCHEMA, compression="zstd")
                    writer.write_table(table)
                    normal += int(np.count_nonzero(batch_labels == 0))
                    attack += int(np.count_nonzero(batch_labels == 1))
                    rows += size
            finally:
                if writer is not None:
                    writer.close()
            counts = reports[scenario]["counts"]
            if (rows != counts["packets"] or normal != counts["normal_packets"]
                    or attack != counts["attack_packets"]):
                raise ValueError(f"Validation row or label counts differ for {scenario}.")
            validation[scenario] = {
                "rows": rows, "normal_packets": normal, "attack_packets": attack,
                "oof_artifact": path.name, "oof_sha256": sha256_file(path),
                **_scenario_metrics(path)}
        result = {
            "report_version": 1,
            "model_family": "xgb_p_t" if variant_name == "full" else "xgb_p_t_ablation",
            "variant_name": variant_name,
            "context_feature_names": list(context_feature_names),
            "status": "development_oof_complete_thresholds_pending",
            "fold": fold, "configuration_name": configuration_name,
            "configuration": parameters, "xgboost_parameters": xgb_parameters,
            "xgboost_version": xgb.__version__,
            "training_scenarios": train_scenarios,
            "validation_scenarios": validation_scenarios,
            "training_rows": training_rows,
            "scenario_class_weights": {name: {"normal": weights[name][0],
                                               "attack": weights[name][1]}
                                       for name in train_scenarios},
            "preprocessor_sha256": preprocessor_sha256,
            "preprocessing_contract_sha256": preprocessing_schema_sha256(schema),
            "context_report_sha256": sha256_file(context_dir / "context_report.json"),
            "context_scaler_artifact": "context_scaler.json",
            "context_scaler_sha256": sha256_file(stage / "context_scaler.json"),
            "feature_count": len(feature_names), "feature_names": feature_names,
            "window_width_seconds": 5, "decision_time": "window_end",
            "validation": validation,
            "fold_macro_packet_roc_auc": float(np.mean([
                item["packet_roc_auc"] for item in validation.values()])),
            "fold_macro_packet_pr_auc_diagnostic": float(np.mean([
                item["packet_pr_auc_diagnostic"] for item in validation.values()])),
            "model_artifact": model_path.name, "model_sha256": sha256_file(model_path),
            "manifest_sha256": sha256_file(manifest_path),
            "trainer_code_sha256": sha256_file(Path(__file__)),
            "git_commit": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
            "git_worktree_status": worktree.stdout if worktree.returncode == 0 else "unavailable",
            "prepared_packet_sha256": {
                name: reports[name]["output_sha256"]
                for name in [*train_scenarios, *validation_scenarios]},
            "context_artifact_sha256": {
                name: context_report["scenarios"][name]["context_sha256"]
                for name in [*train_scenarios, *validation_scenarios]},
            "elapsed_seconds": time.monotonic() - started,
            "thresholds_selected": False, "test_data_accessed": False,
        }
        write_json(stage / "fold_report.json", result)
        write_json(stage / "run_status.json", {
            "complete": True, "report_sha256": sha256_file(stage / "fold_report.json")})
        shutil.copytree(stage, output_dir, ignore=shutil.ignore_patterns("*.npy"))
        for name, checksum in [("fold_report.json", sha256_file(stage / "fold_report.json")),
                               ("model.json", result["model_sha256"]),
                               ("context_scaler.json", result["context_scaler_sha256"])]:
            if sha256_file(output_dir / name) != checksum:
                raise IOError(f"Copied XGB-P+T artifact failed checksum verification: {name}")
        for item in validation.values():
            if sha256_file(output_dir / item["oof_artifact"]) != item["oof_sha256"]:
                raise IOError("Copied XGB-P+T OOF predictions failed checksum verification.")
    return result
