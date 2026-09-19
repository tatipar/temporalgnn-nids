"""Bounded negative and univariate controls for the cAPTure XGB-P baseline."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .capture_data import sha256_file, write_json
from .capture_feature_profile import load_prepared_full_dev, load_preprocessing_schema
from .capture_xgb_p import (
    _load_fold_preprocessor,
    scenario_class_weights,
    validate_xgb_p_fold_run,
    xgb_p_configuration,
)


SINGLE_FEATURE_COLUMNS = {
    "is_mqtt": ("is_mqtt",),
    "tcp_destination_port_role_mqtt_messaging": ("is_tcp", "tcp_destination_port"),
    "frame_length": ("frame_length",),
    "is_arp": ("is_arp",),
    "destination_is_multicast": ("destination_is_multicast",),
    "tcp_flag_ack": ("tcp_flag_ack",),
    "mqtt_qos_0": ("mqtt_qos",),
}


def _sampled_scenario(
    packet_path: Path, columns: list[str], modulus: int, batch_size: int,
) -> pd.DataFrame:
    """Select rows by stable original position without examining labels."""
    if modulus <= 1 or batch_size <= 0:
        raise ValueError("Sample modulus must exceed one and batch size must be positive.")
    selected = []
    required = list(dict.fromkeys([*columns, "source_row_id", "binary_label"]))
    observed = 0
    for batch in pq.ParquetFile(packet_path).iter_batches(
        batch_size=batch_size, columns=required,
    ):
        frame = batch.to_pandas()
        row_ids = frame["source_row_id"].to_numpy(dtype=np.int64)
        if not np.array_equal(row_ids, np.arange(observed, observed + len(frame))):
            raise ValueError("Canonical packet order changed before the sanity sample.")
        observed += len(frame)
        sampled = frame.loc[row_ids % modulus == 0]
        if not sampled.empty:
            selected.append(sampled)
    if not selected:
        raise ValueError("A scenario produced no deterministic sanity-sample rows.")
    return pd.concat(selected, ignore_index=True)


def _binary_labels(frame: pd.DataFrame) -> np.ndarray:
    labels = frame["binary_label"].to_numpy(dtype=np.int8)
    if not np.isin(labels, (0, 1)).all() or len(np.unique(labels)) != 2:
        raise ValueError("The sanity sample must contain both binary classes.")
    return labels


def permute_training_labels(labels: np.ndarray, seed: int) -> np.ndarray:
    """Break feature-label association while preserving class counts."""
    original = np.asarray(labels, dtype=np.int8)
    if original.ndim != 1 or np.unique(original).tolist() != [0, 1]:
        raise ValueError("Training labels must contain both binary classes.")
    shuffled = np.random.default_rng(seed).permutation(original)
    if np.array_equal(shuffled, original):
        shuffled = np.roll(shuffled, 1)
    if np.array_equal(shuffled, original):
        first_zero = int(np.flatnonzero(original == 0)[0])
        first_one = int(np.flatnonzero(original == 1)[0])
        shuffled[first_zero], shuffled[first_one] = shuffled[first_one], shuffled[first_zero]
    if not np.array_equal(np.bincount(original), np.bincount(shuffled)):
        raise AssertionError("Label permutation changed the scenario class counts.")
    return shuffled


def run_shuffled_label_control(
    *, manifest_path: Path, packet_schema_path: Path,
    preprocessing_schema_path: Path, prepared_run_dir: Path,
    preprocessing_audit_dir: Path, baseline_run_dir: Path,
    output_dir: Path, fold: str, batch_size: int = 100_000,
    nthread: int = 2,
) -> dict:
    """Fit a small same-protocol model with training labels shuffled by scenario."""
    import xgboost as xgb
    from sklearn.metrics import average_precision_score, roc_auc_score

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite a sanity run: {output_dir}")
    if nthread <= 0:
        raise ValueError("CPU thread count must be positive.")
    manifest, packet_schema, prepared_reports, packet_paths = load_prepared_full_dev(
        Path(prepared_run_dir), Path(manifest_path), Path(packet_schema_path),
    )
    if fold not in manifest["validation"]["folds"]:
        raise ValueError(f"Undeclared development fold: {fold}")
    split = manifest["validation"]["folds"][fold]
    baseline_dir = Path(baseline_run_dir) / "depth5_primary" / f"fold_{fold}"
    baseline = validate_xgb_p_fold_run(baseline_dir, fold, "depth5_primary")
    if baseline["training_scenarios"] != split["train"] or baseline["validation_scenarios"] != split["validate"]:
        raise ValueError("The baseline fold does not match the current development split.")
    if baseline["prepared_packet_sha256"] != {
        name: prepared_reports[name]["output_sha256"] for name in [*split["train"], *split["validate"]]
    }:
        raise ValueError("The baseline used different prepared packet artifacts.")
    configuration = xgb_p_configuration(manifest, "depth5_primary")
    if baseline["configuration"] != configuration:
        raise ValueError("The declared primary configuration changed after baseline training.")
    schema = load_preprocessing_schema(Path(preprocessing_schema_path), packet_schema)
    preprocessor, preprocessor_hash = _load_fold_preprocessor(
        Path(preprocessing_audit_dir), schema, fold, split["train"],
        Path(prepared_run_dir),
    )
    if preprocessor_hash != baseline["preprocessor_sha256"]:
        raise ValueError("The baseline and negative control use different preprocessors.")
    policy = manifest["training"]["xgb_p_sanity_controls"]["shuffled_training_labels"]
    modulus = int(policy["sample_modulus"])
    seed = int(policy["permutation_seed"])
    if (policy["model_configuration"] != "depth5_primary" or modulus != 100
            or policy["selection_rule"] != "source_row_id_modulo_100_equals_zero"):
        raise ValueError("The frozen XGB-P negative-control policy changed.")

    sampled_frames = {}
    sampled_reports = {}
    for scenario in split["train"]:
        frame = _sampled_scenario(
            packet_paths[scenario], preprocessor.required_columns, modulus, batch_size,
        )
        labels = _binary_labels(frame)
        sampled_frames[scenario] = frame
        sampled_reports[scenario] = {"counts": {
            "packets": len(labels),
            "normal_packets": int(np.count_nonzero(labels == 0)),
            "attack_packets": int(np.count_nonzero(labels == 1)),
        }}
    cell_weights = scenario_class_weights(split["train"], sampled_reports)
    matrices = []
    labels_by_scenario = []
    weights_by_scenario = []
    changed_labels = {}
    for index, scenario in enumerate(split["train"]):
        frame = sampled_frames.pop(scenario)
        original = _binary_labels(frame)
        shuffled = permute_training_labels(original, seed + index)
        changed_labels[scenario] = int(np.count_nonzero(shuffled != original))
        if changed_labels[scenario] == 0:
            raise ValueError("The negative control did not change any training labels.")
        matrices.append(preprocessor.transform(frame).to_numpy(dtype=np.float32))
        labels_by_scenario.append(shuffled)
        weights_by_scenario.append(np.where(
            shuffled == 0, cell_weights[scenario][0], cell_weights[scenario][1],
        ).astype(np.float32))
    features = np.concatenate(matrices)
    labels = np.concatenate(labels_by_scenario)
    weights = np.concatenate(weights_by_scenario)
    del matrices, labels_by_scenario, weights_by_scenario
    if not np.isfinite(features).all():
        raise ValueError("The negative-control training matrix has non-finite values.")
    train_matrix = xgb.QuantileDMatrix(
        features, label=labels, weight=weights,
        feature_names=preprocessor.feature_names,
        max_bin=int(configuration["max_bin"]), nthread=nthread,
    )
    xgb_parameters = dict(baseline["xgboost_parameters"])
    xgb_parameters["nthread"] = nthread
    model = xgb.train(
        xgb_parameters, train_matrix,
        num_boost_round=int(configuration["num_boost_round"]),
    )
    del train_matrix, features, labels, weights

    validation = {}
    for scenario in split["validate"]:
        frame = _sampled_scenario(
            packet_paths[scenario], preprocessor.required_columns, modulus, batch_size,
        )
        true_labels = _binary_labels(frame)
        scores = model.inplace_predict(
            preprocessor.transform(frame).to_numpy(dtype=np.float32),
        )
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError("The negative-control validation scores are invalid.")
        validation[scenario] = {
            "sampled_packets": len(true_labels),
            "normal_packets": int(np.count_nonzero(true_labels == 0)),
            "attack_packets": int(np.count_nonzero(true_labels == 1)),
            "packet_roc_auc": float(roc_auc_score(true_labels, scores)),
            "packet_pr_auc_diagnostic": float(average_precision_score(true_labels, scores)),
        }
    fold_macro = float(np.mean([item["packet_roc_auc"] for item in validation.values()]))
    result = {
        "status": "sampled_shuffled_label_negative_control_complete",
        "fold": fold,
        "sample_modulus": modulus,
        "permutation_seed": seed,
        "permutation_scope": "within_each_sampled_training_scenario",
        "validation_labels_shuffled": False,
        "training_scenarios": split["train"],
        "validation_scenarios": split["validate"],
        "sampled_training_counts": {name: item["counts"] for name, item in sampled_reports.items()},
        "changed_training_labels": changed_labels,
        "validation": validation,
        "fold_macro_packet_roc_auc": fold_macro,
        "review_if_hierarchical_macro_roc_auc_above": policy["review_if_hierarchical_macro_roc_auc_above"],
        "baseline_report_sha256": sha256_file(baseline_dir / "fold_report.json"),
        "baseline_model_sha256": baseline["model_sha256"],
        "preprocessor_sha256": preprocessor_hash,
        "prepared_packet_sha256": baseline["prepared_packet_sha256"],
        "manifest_sha256": sha256_file(Path(manifest_path)),
        "code_sha256": sha256_file(Path(__file__)),
        "thresholds_selected": False,
        "test_data_accessed": False,
    }
    with tempfile.TemporaryDirectory(prefix="capture_sanity_") as temporary:
        stage = Path(temporary)
        model_path = stage / "shuffled_model.json"
        model.save_model(model_path)
        result["model_sha256"] = sha256_file(model_path)
        write_json(stage / "sanity_report.json", result)
        write_json(stage / "run_status.json", {
            "complete": True,
            "report_sha256": sha256_file(stage / "sanity_report.json"),
        })
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage, output_dir)
        if (sha256_file(output_dir / "sanity_report.json")
                != sha256_file(stage / "sanity_report.json")
                or sha256_file(output_dir / "shuffled_model.json") != result["model_sha256"]):
            raise IOError("Copied negative-control artifacts failed verification.")
    return result


def validate_shuffled_label_control(directory: Path, fold: str) -> dict:
    """Verify a completed negative-control run before reuse."""
    directory = Path(directory)
    status = json.loads((directory / "run_status.json").read_text(encoding="utf-8"))
    report_path = directory / "sanity_report.json"
    if status.get("complete") is not True or status.get("report_sha256") != sha256_file(report_path):
        raise ValueError("The shuffled-label control report is incomplete or changed.")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report["fold"] != fold or report["model_sha256"] != sha256_file(directory / "shuffled_model.json"):
        raise ValueError("The shuffled-label control model or fold binding changed.")
    return report


def summarize_shuffled_label_controls(run_dir: Path) -> dict:
    """Compute the same scenario-then-fold macro used by the baseline."""
    reports = {
        fold: validate_shuffled_label_control(Path(run_dir) / f"fold_{fold}", fold)
        for fold in ("A", "B")
    }
    if reports["A"]["manifest_sha256"] != reports["B"]["manifest_sha256"]:
        raise ValueError("Negative-control folds used different manifests.")
    if set(reports["A"]["validation"]) & set(reports["B"]["validation"]):
        raise ValueError("Negative-control validation scenarios overlap.")
    macro = float(np.mean([reports[fold]["fold_macro_packet_roc_auc"] for fold in ("A", "B")]))
    return {
        "fold_packet_roc_auc": {
            fold: reports[fold]["fold_macro_packet_roc_auc"] for fold in ("A", "B")
        },
        "scenario_metrics": {
            scenario: {"fold": fold, **item}
            for fold in ("A", "B")
            for scenario, item in reports[fold]["validation"].items()
        },
        "hierarchical_macro_oof_packet_roc_auc": macro,
        "review_required": macro > float(reports["A"]["review_if_hierarchical_macro_roc_auc_above"]),
        "sampled_negative_control_not_full_data_estimate": True,
    }


def single_feature_values(frame: pd.DataFrame, feature: str) -> np.ndarray:
    """Return one fixed current-packet diagnostic without fitted transforms."""
    if feature not in SINGLE_FEATURE_COLUMNS:
        raise ValueError(f"Undeclared univariate diagnostic: {feature}")
    if feature == "tcp_destination_port_role_mqtt_messaging":
        tcp = pd.to_numeric(frame["is_tcp"], errors="raise").eq(1)
        port = pd.to_numeric(frame["tcp_destination_port"], errors="raise")
        values = (tcp & port.isin((1883, 8883))).to_numpy(dtype=np.float32)
    elif feature == "mqtt_qos_0":
        values = pd.to_numeric(frame["mqtt_qos"], errors="raise").eq(0).to_numpy(dtype=np.float32)
    else:
        values = pd.to_numeric(frame[feature], errors="raise").fillna(0).to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite values in diagnostic feature {feature}.")
    return values


def run_single_feature_diagnostics(
    *, manifest_path: Path, packet_schema_path: Path,
    prepared_run_dir: Path, baseline_run_dir: Path,
    output_path: Path, batch_size: int = 100_000,
) -> dict:
    """Describe each declared feature's individual held-out ranking strength."""
    from sklearn.metrics import roc_auc_score

    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic output: {output_path}")
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    manifest, _, reports, packet_paths = load_prepared_full_dev(
        Path(prepared_run_dir), Path(manifest_path), Path(packet_schema_path),
    )
    features = manifest["training"]["xgb_p_sanity_controls"]["single_feature_diagnostics"]["features"]
    if len(features) != len(set(features)) or not set(features) <= set(SINGLE_FEATURE_COLUMNS):
        raise ValueError("The univariate diagnostic feature list is invalid.")
    rows = []
    baseline_report_hashes = {}
    for fold, split in manifest["validation"]["folds"].items():
        baseline = validate_xgb_p_fold_run(
            Path(baseline_run_dir) / "depth5_primary" / f"fold_{fold}",
            fold, "depth5_primary",
        )
        if baseline["validation_scenarios"] != split["validate"]:
            raise ValueError("The baseline validation scenarios changed.")
        baseline_report_hashes[fold] = sha256_file(
            Path(baseline_run_dir) / "depth5_primary" / f"fold_{fold}" / "fold_report.json"
        )
        for scenario in split["validate"]:
            if baseline["prepared_packet_sha256"][scenario] != reports[scenario]["output_sha256"]:
                raise ValueError("The baseline and diagnostic input artifacts differ.")
            columns = list(dict.fromkeys([
                "binary_label", *(
                    column for feature in features for column in SINGLE_FEATURE_COLUMNS[feature]
                ),
            ]))
            labels_parts = []
            value_parts = {feature: [] for feature in features}
            for batch in pq.ParquetFile(packet_paths[scenario]).iter_batches(
                batch_size=batch_size, columns=columns,
            ):
                frame = batch.to_pandas()
                labels_parts.append(frame["binary_label"].to_numpy(dtype=np.int8))
                for feature in features:
                    value_parts[feature].append(single_feature_values(frame, feature))
            labels = np.concatenate(labels_parts)
            if len(labels) != reports[scenario]["counts"]["packets"] or np.unique(labels).tolist() != [0, 1]:
                raise ValueError("The univariate diagnostic labels or row count changed.")
            for feature in features:
                values = np.concatenate(value_parts[feature])
                auc = float(roc_auc_score(labels, values))
                rows.append({
                    "fold": fold,
                    "scenario": scenario,
                    "feature": feature,
                    "packets": len(labels),
                    "distinct_values": int(len(np.unique(values))),
                    "raw_direction_roc_auc": auc,
                    "best_direction_roc_auc_diagnostic": max(auc, 1.0 - auc),
                    "baseline_packet_roc_auc": baseline["validation"][scenario]["packet_roc_auc"],
                })
    result = {
        "status": "single_feature_oof_diagnostics_complete",
        "features": features,
        "rows": rows,
        "interpretation": "Descriptive best-direction strength only; not a fitted model or feature-selection rule.",
        "baseline_report_sha256": baseline_report_hashes,
        "prepared_packet_sha256": {
            scenario: reports[scenario]["output_sha256"] for scenario in packet_paths
        },
        "manifest_sha256": sha256_file(Path(manifest_path)),
        "code_sha256": sha256_file(Path(__file__)),
        "test_data_accessed": False,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result
