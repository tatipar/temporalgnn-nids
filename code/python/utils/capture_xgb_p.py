"""CPU XGB-P development-fold training on prepared cAPTure packets."""

from __future__ import annotations

import gc
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
    load_prepared_full_dev,
    load_preprocessing_schema,
    preprocessing_schema_sha256,
)
from .capture_preprocess import CaptureFoldPreprocessor


NANOSECONDS_PER_SECOND = 1_000_000_000
REPORT_VERSION = 1
OOF_COLUMNS = (
    "packet_id", "source_row_id", "packet_timestamp_ns", "binary_label",
    "attack_step", "phase", "sequence_id",
)
OOF_SCHEMA = pa.schema([
    ("packet_id", pa.string()),
    ("source_row_id", pa.int64()),
    ("packet_timestamp_ns", pa.int64()),
    ("binary_label", pa.int8()),
    ("attack_step", pa.string()),
    ("phase", pa.string()),
    ("sequence_id", pa.string()),
    ("window_index", pa.int64()),
    ("window_end_ns", pa.int64()),
    ("score", pa.float32()),
])


def xgb_p_configuration(manifest: dict, configuration_name: str) -> dict:
    """Resolve one declared configuration without notebook-side overrides."""
    study = manifest["training"]["hyperparameter_search"]["xgb_p"]
    if configuration_name not in study["candidates"]:
        raise ValueError(f"Undeclared XGB-P configuration: {configuration_name}")
    common = dict(study["common_parameters"])
    common.update(study["candidates"][configuration_name])
    required = {
        "objective": "binary:logistic", "tree_method": "hist", "device": "cpu",
        "subsample": 1.0, "colsample_bytree": 1.0,
        "colsample_bylevel": 1.0, "colsample_bynode": 1.0,
    }
    for name, expected in required.items():
        if common.get(name) != expected:
            raise ValueError(f"XGB-P requires {name}={expected} in this protocol.")
    if int(common["num_boost_round"]) <= 0 or int(common["max_depth"]) <= 0:
        raise ValueError("Boosting rounds and tree depth must be positive.")
    if manifest["training"]["seeds"] != [42]:
        raise ValueError("This first XGB-P protocol requires the declared seed 42.")
    return common


def scenario_class_weights(
    train_scenarios: list[str], reports: dict[str, dict]
) -> dict[str, dict[int, float]]:
    """Give every scenario/class cell equal total weight and mean weight one."""
    if not train_scenarios or len(set(train_scenarios)) != len(train_scenarios):
        raise ValueError("Training scenarios must be a non-empty unique list.")
    total_rows = sum(int(reports[name]["counts"]["packets"]) for name in train_scenarios)
    cell_mass = total_rows / (2 * len(train_scenarios))
    weights = {}
    for scenario in train_scenarios:
        counts = reports[scenario]["counts"]
        normal = int(counts["normal_packets"])
        attack = int(counts["attack_packets"])
        if normal <= 0 or attack <= 0 or normal + attack != int(counts["packets"]):
            raise ValueError(f"Incomplete binary class counts for {scenario}.")
        weights[scenario] = {0: cell_mass / normal, 1: cell_mass / attack}
    return weights


def window_coordinates(
    timestamps_ns: np.ndarray, origin_ns: int, width_seconds: int
) -> tuple[np.ndarray, np.ndarray]:
    """Assign fixed half-open windows whose first boundary is the scenario origin."""
    if width_seconds <= 0:
        raise ValueError("Window width must be positive.")
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    if np.any(timestamps < origin_ns):
        raise ValueError("A packet timestamp precedes its scenario origin.")
    width_ns = width_seconds * NANOSECONDS_PER_SECOND
    indexes = (timestamps - origin_ns) // width_ns
    ends = origin_ns + (indexes + 1) * width_ns
    return indexes.astype(np.int64), ends.astype(np.int64)


def _validated_labels(frame: pd.DataFrame, scenario: str) -> np.ndarray:
    values = frame["binary_label"].to_numpy()
    if not np.isin(values, [0, 1]).all():
        raise ValueError(f"Invalid binary labels in {scenario}.")
    return values.astype(np.uint8)


def _load_fold_preprocessor(
    audit_dir: Path, schema: dict, fold: str, train_scenarios: list[str],
    prepared_run_dir: Path,
) -> tuple[CaptureFoldPreprocessor, str]:
    audit_path = audit_dir / "capture_preprocessing_audit.json"
    status_path = audit_dir / "run_status.json"
    if not audit_path.is_file() or not status_path.is_file():
        raise FileNotFoundError("The completed preprocessing audit is required.")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("complete") is not True or audit.get("preprocessing_contract_sha256") != preprocessing_schema_sha256(schema):
        raise ValueError("The preprocessing audit is incomplete or has a different contract.")
    if Path(audit["prepared_run"]).resolve() != prepared_run_dir.resolve():
        raise ValueError("The preprocessing audit belongs to a different prepared run.")
    fold_report = audit["folds"][fold]
    if fold_report["train_scenarios"] != train_scenarios:
        raise ValueError("The audited preprocessor belongs to a different training fold.")
    artifact_path = audit_dir / fold_report["preprocessor_artifact"]
    artifact_sha256 = sha256_file(artifact_path)
    if artifact_sha256 != fold_report["preprocessor_sha256"]:
        raise ValueError("The fold preprocessor checksum differs from the audit.")
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    if artifact["fold"] != fold or artifact["training_scenarios"] != train_scenarios:
        raise ValueError("The fold preprocessor has incorrect fold provenance.")
    preprocessor = CaptureFoldPreprocessor.from_dict(schema, artifact)
    if len(preprocessor.feature_names) != 103:
        raise ValueError("The frozen packet model view must have 103 columns.")
    return preprocessor, artifact_sha256


def _materialize_training_fold(
    matrix_dir: Path, train_scenarios: list[str], packet_paths: dict[str, Path],
    reports: dict[str, dict], preprocessor: CaptureFoldPreprocessor,
    weights: dict[str, dict[int, float]], batch_size: int,
) -> tuple[np.memmap, np.memmap, np.memmap, dict]:
    rows = sum(int(reports[name]["counts"]["packets"]) for name in train_scenarios)
    feature_count = len(preprocessor.feature_names)
    required_bytes = rows * (feature_count * 4 + 1 + 4) + 1_000_000_000
    free_bytes = shutil.disk_usage(matrix_dir).free
    if free_bytes < required_bytes:
        raise OSError(
            f"Insufficient local storage for the training matrix: "
            f"need {required_bytes / 1e9:.2f} GB, free {free_bytes / 1e9:.2f} GB."
        )
    features = np.lib.format.open_memmap(
        matrix_dir / "features.npy", mode="w+", dtype=np.float32,
        shape=(rows, feature_count),
    )
    labels = np.lib.format.open_memmap(
        matrix_dir / "labels.npy", mode="w+", dtype=np.uint8, shape=(rows,),
    )
    sample_weights = np.lib.format.open_memmap(
        matrix_dir / "weights.npy", mode="w+", dtype=np.float32, shape=(rows,),
    )
    offset = 0
    observed = {}
    columns = [*preprocessor.required_columns, "source_row_id", "binary_label"]
    for scenario in train_scenarios:
        expected_rows = int(reports[scenario]["counts"]["packets"])
        scenario_rows = 0
        normal = attack = 0
        parquet = pq.ParquetFile(packet_paths[scenario])
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            frame = batch.to_pandas()
            size = len(frame)
            row_ids = frame["source_row_id"].to_numpy(dtype=np.int64)
            if not np.array_equal(row_ids, np.arange(scenario_rows, scenario_rows + size)):
                raise ValueError(f"Packet order changed in {scenario}.")
            batch_labels = _validated_labels(frame, scenario)
            matrix = preprocessor.transform(frame).to_numpy(dtype=np.float32, copy=False)
            if matrix.shape != (size, feature_count):
                raise ValueError("Transformed packet matrix has an unexpected shape.")
            features[offset:offset + size] = matrix
            labels[offset:offset + size] = batch_labels
            sample_weights[offset:offset + size] = np.where(
                batch_labels == 0, weights[scenario][0], weights[scenario][1]
            ).astype(np.float32)
            normal += int(np.count_nonzero(batch_labels == 0))
            attack += int(np.count_nonzero(batch_labels == 1))
            scenario_rows += size
            offset += size
        if scenario_rows != expected_rows or normal != int(reports[scenario]["counts"]["normal_packets"]) or attack != int(reports[scenario]["counts"]["attack_packets"]):
            raise ValueError(f"Training row or label counts differ for {scenario}.")
        observed[scenario] = {"rows": scenario_rows, "normal": normal, "attack": attack}
    if offset != rows or not np.isfinite(sample_weights).all() or np.any(sample_weights <= 0):
        raise ValueError("The training matrix or sample weights are incomplete.")
    features.flush()
    labels.flush()
    sample_weights.flush()
    return features, labels, sample_weights, observed


def _score_validation_scenario(
    *, model: object, scenario: str, packet_path: Path, report: dict,
    preprocessor: CaptureFoldPreprocessor, output_path: Path,
    width_seconds: int, batch_size: int,
) -> dict:
    columns = list(dict.fromkeys([*preprocessor.required_columns, *OOF_COLUMNS]))
    parquet = pq.ParquetFile(packet_path)
    origin_ns = int(report["scenario_origin_timestamp_ns"])
    expected_rows = int(report["counts"]["packets"])
    rows = normal = attack = 0
    writer = None
    try:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            frame = batch.to_pandas()
            size = len(frame)
            row_ids = frame["source_row_id"].to_numpy(dtype=np.int64)
            if not np.array_equal(row_ids, np.arange(rows, rows + size)):
                raise ValueError(f"Validation packet order changed in {scenario}.")
            labels = _validated_labels(frame, scenario)
            matrix = preprocessor.transform(frame).to_numpy(dtype=np.float32, copy=False)
            scores = np.asarray(model.inplace_predict(matrix), dtype=np.float32)
            if scores.shape != (size,) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError(f"Invalid model scores in {scenario}.")
            timestamps = frame["packet_timestamp_ns"].to_numpy(dtype=np.int64)
            windows, window_ends = window_coordinates(timestamps, origin_ns, width_seconds)
            result = frame.loc[:, list(OOF_COLUMNS)].copy()
            result["window_index"] = windows
            result["window_end_ns"] = window_ends
            result["score"] = scores
            table = pa.Table.from_pandas(
                result, schema=OOF_SCHEMA, preserve_index=False, safe=True,
            )
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
            writer.write_table(table)
            normal += int(np.count_nonzero(labels == 0))
            attack += int(np.count_nonzero(labels == 1))
            rows += size
    finally:
        if writer is not None:
            writer.close()
    if rows != expected_rows or normal != int(report["counts"]["normal_packets"]) or attack != int(report["counts"]["attack_packets"]):
        raise ValueError(f"Validation row or label counts differ for {scenario}.")
    if pq.ParquetFile(output_path).metadata.num_rows != rows:
        raise ValueError(f"OOF Parquet row count differs for {scenario}.")
    return {
        "rows": rows, "normal_packets": normal, "attack_packets": attack,
        "oof_artifact": output_path.name, "oof_sha256": sha256_file(output_path),
    }


def _scenario_metrics(path: Path) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    table = pq.read_table(path, columns=["binary_label", "score"])
    labels = table.column("binary_label").to_numpy(zero_copy_only=False)
    scores = table.column("score").to_numpy(zero_copy_only=False)
    if np.unique(labels).tolist() != [0, 1]:
        raise ValueError("Both classes are required for packet ranking metrics.")
    return {
        "packet_roc_auc": float(roc_auc_score(labels, scores)),
        "packet_pr_auc_diagnostic": float(average_precision_score(labels, scores)),
    }


def run_xgb_p_fold(
    *, manifest_path: Path, packet_schema_path: Path,
    preprocessing_schema_path: Path, prepared_run_dir: Path,
    preprocessing_audit_dir: Path, output_dir: Path, local_work_root: Path,
    fold: str, configuration_name: str, batch_size: int = 50_000,
    nthread: int = 2,
) -> dict:
    """Train one declared fold/configuration and persist immutable OOF outputs."""
    import xgboost as xgb

    if batch_size <= 0 or nthread <= 0:
        raise ValueError("Batch size and CPU thread count must be positive.")
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {output_dir}")
    manifest_path = Path(manifest_path)
    packet_schema_path = Path(packet_schema_path)
    preprocessing_schema_path = Path(preprocessing_schema_path)
    prepared_run_dir = Path(prepared_run_dir)
    preprocessing_audit_dir = Path(preprocessing_audit_dir)
    manifest, packet_schema, reports, packet_paths = load_prepared_full_dev(
        prepared_run_dir, manifest_path, packet_schema_path,
    )
    if fold not in manifest["validation"]["folds"]:
        raise ValueError(f"Undeclared development fold: {fold}")
    split = manifest["validation"]["folds"][fold]
    train_scenarios = list(split["train"])
    validation_scenarios = list(split["validate"])
    parameters = xgb_p_configuration(manifest, configuration_name)
    schema = load_preprocessing_schema(preprocessing_schema_path, packet_schema)
    if schema["status"] != "frozen":
        raise ValueError("XGB-P training requires the frozen preprocessing contract.")
    preprocessor, preprocessor_sha256 = _load_fold_preprocessor(
        preprocessing_audit_dir, schema, fold, train_scenarios, prepared_run_dir,
    )
    expected_training_rows = sum(int(reports[name]["counts"]["packets"]) for name in train_scenarios)
    if preprocessor.training_rows != expected_training_rows:
        raise ValueError("The audited preprocessor was fitted on a different row count.")
    weights = scenario_class_weights(train_scenarios, reports)
    width_seconds = int(manifest["windows"]["selected_duration_seconds"])
    if width_seconds != 5 or manifest["windows"]["origin_rule"] != "first_packet_timestamp_per_scenario":
        raise ValueError("This XGB-P run requires the declared five-second window policy.")
    local_work_root = Path(local_work_root)
    local_work_root.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    repository_root = manifest_path.resolve().parent.parent
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository_root,
        capture_output=True, text=True, check=False,
    )
    with tempfile.TemporaryDirectory(prefix="capture_xgb_p_", dir=local_work_root) as temporary:
        stage = Path(temporary)
        print(f"Materializing fold {fold} training matrix on local storage...", flush=True)
        features, labels, sample_weights, observed_train = _materialize_training_fold(
            stage, train_scenarios, packet_paths, reports, preprocessor,
            weights, batch_size,
        )
        print(
            f"Training {configuration_name} on {len(labels):,} packets "
            f"with {features.shape[1]} features...", flush=True,
        )
        train_matrix = xgb.QuantileDMatrix(
            features, label=labels, weight=sample_weights,
            feature_names=preprocessor.feature_names,
            max_bin=int(parameters["max_bin"]), nthread=nthread,
        )
        xgb_parameters = {
            "objective": parameters["objective"],
            "tree_method": parameters["tree_method"],
            "device": parameters["device"],
            "max_depth": int(parameters["max_depth"]),
            "eta": float(parameters["learning_rate"]),
            "min_child_weight": float(parameters["min_child_weight"]),
            "lambda": float(parameters["reg_lambda"]),
            "max_bin": int(parameters["max_bin"]),
            "subsample": float(parameters["subsample"]),
            "colsample_bytree": float(parameters["colsample_bytree"]),
            "colsample_bylevel": float(parameters["colsample_bylevel"]),
            "colsample_bynode": float(parameters["colsample_bynode"]),
            "seed": 42,
            "nthread": nthread,
        }
        model = xgb.train(
            xgb_parameters, train_matrix,
            num_boost_round=int(parameters["num_boost_round"]),
        )
        if model.num_boosted_rounds() != int(parameters["num_boost_round"]):
            raise ValueError("The trained model has an unexpected number of trees.")
        model_path = stage / "model.json"
        model.save_model(model_path)
        write_json(stage / "feature_importance_gain.json", {
            name: float(model.get_score(importance_type="gain").get(name, 0.0))
            for name in preprocessor.feature_names
        })
        del train_matrix, features, labels, sample_weights
        gc.collect()
        validation_reports = {}
        for scenario in validation_scenarios:
            print(f"Scoring held-out scenario {scenario}...", flush=True)
            oof_path = stage / f"oof_{scenario}.parquet"
            saved = _score_validation_scenario(
                model=model, scenario=scenario, packet_path=packet_paths[scenario],
                report=reports[scenario], preprocessor=preprocessor,
                output_path=oof_path, width_seconds=width_seconds,
                batch_size=batch_size,
            )
            validation_reports[scenario] = {
                **saved, **_scenario_metrics(oof_path),
            }
        fold_roc_auc = float(np.mean([
            item["packet_roc_auc"] for item in validation_reports.values()
        ]))
        fold_pr_auc = float(np.mean([
            item["packet_pr_auc_diagnostic"] for item in validation_reports.values()
        ]))
        result = {
            "report_version": REPORT_VERSION,
            "status": "development_oof_complete_thresholds_pending",
            "model_family": "xgb_p",
            "fold": fold,
            "configuration_name": configuration_name,
            "configuration": parameters,
            "xgboost_parameters": xgb_parameters,
            "xgboost_version": xgb.__version__,
            "training_scenarios": train_scenarios,
            "validation_scenarios": validation_scenarios,
            "training_rows": sum(item["rows"] for item in observed_train.values()),
            "observed_training_counts": observed_train,
            "scenario_class_weights": {
                scenario: {"normal": values[0], "attack": values[1]}
                for scenario, values in weights.items()
            },
            "preprocessor_sha256": preprocessor_sha256,
            "preprocessing_contract_sha256": preprocessing_schema_sha256(schema),
            "feature_count": len(preprocessor.feature_names),
            "feature_names": preprocessor.feature_names,
            "active_feature_count": len(preprocessor.active_features),
            "masked_features": preprocessor.masked_features,
            "window_width_seconds": width_seconds,
            "decision_time": "window_end",
            "validation": validation_reports,
            "fold_macro_packet_roc_auc": fold_roc_auc,
            "fold_macro_packet_pr_auc_diagnostic": fold_pr_auc,
            "model_artifact": model_path.name,
            "model_sha256": sha256_file(model_path),
            "manifest_sha256": sha256_file(manifest_path),
            "trainer_code_sha256": sha256_file(Path(__file__)),
            "git_commit": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
            "packet_schema_sha256": sha256_file(packet_schema_path),
            "preprocessing_schema_sha256": sha256_file(preprocessing_schema_path),
            "prepared_packet_sha256": {
                scenario: reports[scenario]["output_sha256"]
                for scenario in [*train_scenarios, *validation_scenarios]
            },
            "prepared_run": str(prepared_run_dir),
            "preprocessing_audit_run": str(preprocessing_audit_dir),
            "elapsed_seconds": time.monotonic() - started,
            "thresholds_selected": False,
            "test_data_accessed": False,
        }
        write_json(stage / "fold_report.json", result)
        write_json(stage / "run_status.json", {
            "complete": True, "report": "fold_report.json",
            "report_sha256": sha256_file(stage / "fold_report.json"),
        })
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, output_dir, ignore=shutil.ignore_patterns("*.npy"))
        if sha256_file(output_dir / "fold_report.json") != sha256_file(stage / "fold_report.json"):
            raise IOError("Copied fold report failed checksum verification.")
        if sha256_file(output_dir / "model.json") != result["model_sha256"]:
            raise IOError("Copied XGB-P model failed checksum verification.")
        for item in validation_reports.values():
            if sha256_file(output_dir / item["oof_artifact"]) != item["oof_sha256"]:
                raise IOError("Copied OOF predictions failed checksum verification.")
    print(f"Saved {configuration_name} fold {fold} to {output_dir}.", flush=True)
    print("Removed temporary local training matrix after verified Drive copy.", flush=True)
    return result


def validate_xgb_p_fold_run(directory: Path, fold: str, configuration_name: str) -> dict:
    """Verify an existing immutable fold output before resuming or summarizing."""
    directory = Path(directory)
    status_path = directory / "run_status.json"
    report_path = directory / "fold_report.json"
    if not status_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(f"Complete fold {fold} is required for {configuration_name}.")
    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("complete") is not True or status.get("report_sha256") != sha256_file(report_path):
        raise ValueError(f"Incomplete or changed fold {fold} report.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["fold"] != fold or report["configuration_name"] != configuration_name:
        raise ValueError("Fold report binding mismatch.")
    if sha256_file(directory / report["model_artifact"]) != report["model_sha256"]:
        raise ValueError("The trained model changed after the fold run.")
    for item in report["validation"].values():
        if sha256_file(directory / item["oof_artifact"]) != item["oof_sha256"]:
            raise ValueError("An OOF prediction artifact changed after training.")
    return report


def summarize_xgb_p_oof(run_dir: Path, configuration_name: str) -> dict:
    """Read both fold reports and compute the predeclared hierarchical macro."""
    run_dir = Path(run_dir)
    reports = {}
    for fold in ("A", "B"):
        directory = run_dir / configuration_name / f"fold_{fold}"
        reports[fold] = validate_xgb_p_fold_run(directory, fold, configuration_name)
    if reports["A"]["manifest_sha256"] != reports["B"]["manifest_sha256"] or reports["A"]["configuration"] != reports["B"]["configuration"]:
        raise ValueError("The fold reports do not share one frozen protocol.")
    scenarios_a = set(reports["A"]["validation"])
    scenarios_b = set(reports["B"]["validation"])
    if scenarios_a & scenarios_b or len(scenarios_a | scenarios_b) != 5:
        raise ValueError("Exactly one OOF prediction set per development scenario is required.")
    return {
        "configuration_name": configuration_name,
        "primary_metric": "hierarchical_macro_oof_packet_roc_auc",
        "fold_packet_roc_auc": {
            fold: reports[fold]["fold_macro_packet_roc_auc"] for fold in ("A", "B")
        },
        "fold_packet_pr_auc_diagnostic": {
            fold: reports[fold]["fold_macro_packet_pr_auc_diagnostic"] for fold in ("A", "B")
        },
        "hierarchical_macro_oof_packet_roc_auc": float(np.mean([
            reports[fold]["fold_macro_packet_roc_auc"] for fold in ("A", "B")
        ])),
        "hierarchical_macro_oof_packet_pr_auc_diagnostic": float(np.mean([
            reports[fold]["fold_macro_packet_pr_auc_diagnostic"] for fold in ("A", "B")
        ])),
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
