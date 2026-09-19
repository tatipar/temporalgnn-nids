"""Fold-local model preprocessing for canonical cAPTure packet features."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .capture_data import selected_scenarios, sha256_file, write_json
from .capture_feature_profile import (
    encode_tcp_port_roles,
    load_prepared_full_dev,
    load_preprocessing_schema,
    preprocessing_schema_sha256,
)
from .capture_prepare import ordered_feature_names


REPORT_VERSION = 1


def _stable_name_hash(names: list[str]) -> str:
    payload = json.dumps(names, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _parse_nullable_numeric(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_numeric(values, errors="coerce")
    invalid = values.notna() & parsed.isna()
    if invalid.any():
        examples = values.loc[invalid].drop_duplicates().head(5).tolist()
        raise ValueError(f"{name} contains non-numeric values: {examples}")
    finite = parsed.dropna().to_numpy(dtype=np.float64)
    if not np.isfinite(finite).all():
        raise ValueError(f"{name} contains non-finite values.")
    return parsed.astype("float64")


def _parse_nullable_integer(values: pd.Series, name: str,
                            minimum: int, maximum: int) -> pd.Series:
    parsed = _parse_nullable_numeric(values, name)
    present = parsed.notna()
    observed = parsed.loc[present]
    invalid = (np.floor(observed) != observed) | observed.lt(minimum) | observed.gt(maximum)
    if invalid.any():
        examples = observed.loc[invalid].drop_duplicates().head(5).tolist()
        raise ValueError(
            f"{name} must contain integers in {minimum}..{maximum}; examples: {examples}"
        )
    return parsed.astype("Int64")


def model_feature_names(schema: dict) -> list[str]:
    """Return the stable 103-column primary model feature order."""
    exclusions = set(schema["primary_exclusions"])
    binary = [
        name for name in schema["feature_roles"]["binary"] if name not in exclusions
    ]
    presence = list(schema["numeric_policy"]["presence_indicators"])
    numeric = [
        name for name in schema["feature_roles"]["numeric_magnitude"]
        if name not in exclusions
    ]
    ports = []
    for direction in ("source", "destination"):
        ports.extend(
            f"tcp_{direction}_port_role_{role}"
            for role in schema["port_encoding"]["ordered_roles"]
        )
    categorical = []
    for source, encoding in schema["categorical_code_policy"]["encodings"].items():
        if source in exclusions:
            raise ValueError(f"An excluded feature has a model encoding: {source}")
        representation = encoding["representation"]
        if representation == "fixed_binary_bits":
            categorical.extend(
                f"{encoding['output_prefix']}_{bit}"
                for bit in range(int(encoding["bit_count"]))
            )
        elif representation == "fixed_one_hot":
            categorical.extend(f"{source}_{value}" for value in encoding["values"])
        else:
            raise ValueError(f"Unsupported categorical representation: {representation}")
    names = [*binary, *presence, *numeric, *ports, *categorical]
    if len(names) != len(set(names)):
        raise ValueError("The model feature order contains duplicate names.")
    expected = int(schema["primary_model_view"]["final_feature_count"])
    if len(names) != expected:
        raise ValueError(f"Expected {expected} model features, found {len(names)}.")
    return names


def validate_model_preprocessing_schema(schema: dict) -> list[str]:
    """Validate policies that define the fixed-width primary model view."""
    roles = schema["feature_roles"]
    exclusions = schema.get("primary_exclusions", {})
    expected_exclusions = {
        "ethernet_type", "mqtt_version", "mqtt_connack_reason_code",
        "tcp_source_port", "tcp_destination_port",
    }
    if set(exclusions) != expected_exclusions:
        raise ValueError("Primary exclusions do not match the reviewed feature contract.")
    numeric = set(roles["numeric_magnitude"])
    numeric_policy = schema["numeric_policy"]
    transform_groups = [
        set(numeric_policy["log1p_then_standardize"]),
        set(numeric_policy["log1p_without_scaling"]),
        set(numeric_policy["standardize"]),
    ]
    transformed = set().union(*transform_groups)
    overlaps = any(
        transform_groups[left] & transform_groups[right]
        for left in range(len(transform_groups))
        for right in range(left + 1, len(transform_groups))
    )
    if transformed != numeric or overlaps:
        raise ValueError("Every numeric magnitude must have exactly one transform.")
    presence_sources = set(numeric_policy["presence_indicators"].values())
    if not presence_sources <= set(roles["binary"] + roles["numeric_magnitude"]):
        raise ValueError("A presence indicator references an unknown canonical feature.")
    binary_policy = schema["binary_policy"]
    nonnullable = set(binary_policy["nonnullable"])
    nullable = set(binary_policy["nullable_fill_zero"])
    if (nonnullable & nullable) or ((nonnullable | nullable) != set(roles["binary"])):
        raise ValueError("Binary null policies must partition the canonical binary features.")
    encoded = set(schema["categorical_code_policy"]["encodings"])
    expected_encoded = set(roles["categorical_code"]) - set(exclusions)
    if encoded != expected_encoded:
        raise ValueError("Categorical encodings do not cover the primary categorical fields.")
    if schema["fold_variance_policy"].get("apply_same_mask_to_training_and_validation") is not True:
        raise ValueError("The fold variance mask must also be applied to validation.")
    return model_feature_names(schema)


def _batch_frames(paths: Iterable[Path], columns: list[str],
                  batch_size: int) -> Iterator[pd.DataFrame]:
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    for path in paths:
        parquet = pq.ParquetFile(Path(path))
        missing = sorted(set(columns) - set(parquet.schema_arrow.names))
        if missing:
            raise ValueError(f"Prepared packet artifact is missing columns: {missing}")
        for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
            yield batch.to_pandas()


class CaptureFoldPreprocessor:
    """Fit numeric statistics and a variance mask using one training fold only."""

    def __init__(self, schema: dict):
        self.schema = schema
        self.feature_names = validate_model_preprocessing_schema(schema)
        self.numeric_features = list(schema["feature_roles"]["numeric_magnitude"])
        self.numeric_parameters: dict[str, dict[str, float | int | str]] = {}
        self.active_features: list[str] = []
        self.masked_features: list[str] = []
        self._active_mask: np.ndarray | None = None
        self.training_rows = 0
        self.is_fitted = False

    @property
    def required_columns(self) -> list[str]:
        return [
            name for role in self.schema["feature_roles"].values() for name in role
        ]

    def _encode_unscaled(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.required_columns) - set(frame.columns))
        if missing:
            raise ValueError(f"Canonical packet batch is missing features: {missing}")
        result = pd.DataFrame(index=frame.index)
        exclusions = set(self.schema["primary_exclusions"])
        nonnullable_binary = set(self.schema["binary_policy"]["nonnullable"])

        for name in self.schema["feature_roles"]["binary"]:
            if name in exclusions:
                continue
            parsed = _parse_nullable_numeric(frame[name], name)
            if name in nonnullable_binary and parsed.isna().any():
                raise ValueError(f"{name} must not contain null values.")
            if not parsed.dropna().isin([0.0, 1.0]).all():
                raise ValueError(f"{name} must contain only binary values or nulls.")
            result[name] = parsed.fillna(0.0).astype("float32")

        for output, source in self.schema["numeric_policy"]["presence_indicators"].items():
            result[output] = frame[source].notna().astype("float32")

        log_features = (
            set(self.schema["numeric_policy"]["log1p_then_standardize"])
            | set(self.schema["numeric_policy"]["log1p_without_scaling"])
        )
        for name in self.numeric_features:
            parsed = _parse_nullable_numeric(frame[name], name)
            if name in log_features:
                if parsed.dropna().lt(0).any():
                    raise ValueError(f"{name} must be nonnegative before log1p.")
                parsed = np.log1p(parsed)
            result[name] = parsed.astype("float64")

        port_schema = self.schema["port_encoding"]
        for direction, source in (
            ("source", port_schema["source_feature"]),
            ("destination", port_schema["destination_feature"]),
        ):
            encoded = encode_tcp_port_roles(
                frame[source], frame[port_schema["tcp_indicator"]], self.schema,
                prefix=f"tcp_{direction}_port_role",
            )
            result = pd.concat([result, encoded], axis=1)

        for source, encoding in self.schema["categorical_code_policy"]["encodings"].items():
            representation = encoding["representation"]
            if representation == "fixed_binary_bits":
                values = _parse_nullable_integer(
                    frame[source], source, int(encoding["minimum"]), int(encoding["maximum"]),
                )
                present = values.notna()
                integers = values.fillna(0).astype("int64")
                for bit in range(int(encoding["bit_count"])):
                    result[f"{encoding['output_prefix']}_{bit}"] = (
                        present & integers.floordiv(2 ** bit).mod(2).eq(1)
                    ).astype("float32")
            elif representation == "fixed_one_hot":
                domain = [int(value) for value in encoding["values"]]
                values = _parse_nullable_integer(frame[source], source, min(domain), max(domain))
                unexpected = values.notna() & ~values.isin(domain)
                if unexpected.any():
                    examples = values.loc[unexpected].drop_duplicates().head(5).tolist()
                    raise ValueError(f"{source} contains out-of-domain values: {examples}")
                for value in domain:
                    result[f"{source}_{value}"] = values.eq(value).fillna(False).astype("float32")
            else:
                raise AssertionError(f"Unexpected categorical representation: {representation}")

        if result.columns.tolist() != self.feature_names:
            raise AssertionError("Encoded model features do not match the declared stable order.")
        return result

    def fit(self, frames: Iterable[pd.DataFrame]) -> "CaptureFoldPreprocessor":
        """Fit numeric moments and the final constant mask from training batches."""
        if self.is_fitted:
            raise RuntimeError("A fitted preprocessor cannot be refit.")
        counts = {name: 0 for name in self.numeric_features}
        means = {name: 0.0 for name in self.numeric_features}
        m2 = {name: 0.0 for name in self.numeric_features}
        minima = {name: np.inf for name in self.feature_names}
        maxima = {name: -np.inf for name in self.feature_names}

        for frame in frames:
            encoded = self._encode_unscaled(frame)
            self.training_rows += len(encoded)
            for name in self.feature_names:
                observed = encoded[name].dropna().to_numpy(dtype=np.float64)
                if observed.size:
                    minima[name] = min(minima[name], float(observed.min()))
                    maxima[name] = max(maxima[name], float(observed.max()))
            for name in self.numeric_features:
                values = encoded[name].dropna().to_numpy(dtype=np.float64)
                if not values.size:
                    continue
                batch_count = int(values.size)
                batch_mean = float(values.mean())
                batch_m2 = float(np.square(values - batch_mean).sum())
                previous = counts[name]
                total = previous + batch_count
                delta = batch_mean - means[name]
                means[name] += delta * batch_count / total
                m2[name] += batch_m2 + delta * delta * previous * batch_count / total
                counts[name] = total

        if self.training_rows == 0:
            raise ValueError("The training fold contains no packets.")
        for name in self.numeric_features:
            count = counts[name]
            variance = m2[name] / count if count else 0.0
            standard_deviation = float(np.sqrt(max(variance, 0.0)))
            if name in self.schema["numeric_policy"]["log1p_then_standardize"]:
                transform = "log1p_standardize"
                centering_value = means[name] if count else 0.0
                scaling_divisor = standard_deviation if standard_deviation > 0.0 else 1.0
            elif name in self.schema["numeric_policy"]["log1p_without_scaling"]:
                transform = "log1p_no_scaling"
                centering_value = 0.0
                scaling_divisor = 1.0
            else:
                transform = "standardize"
                centering_value = means[name] if count else 0.0
                scaling_divisor = standard_deviation if standard_deviation > 0.0 else 1.0
            self.numeric_parameters[name] = {
                "transform": transform,
                "training_nonnull": count,
                "training_mean": means[name] if count else 0.0,
                "training_standard_deviation": standard_deviation,
                "centering_value": centering_value,
                "scaling_divisor": scaling_divisor,
            }

        active = []
        numeric = set(self.numeric_features)
        for name in self.feature_names:
            if name in numeric:
                variable = self.numeric_parameters[name]["training_standard_deviation"] > 0.0
            else:
                variable = np.isfinite(minima[name]) and minima[name] != maxima[name]
            if variable:
                active.append(name)
        active_set = set(active)
        self.active_features = active
        self.masked_features = [name for name in self.feature_names if name not in active_set]
        self._active_mask = np.asarray(
            [1.0 if name in active_set else 0.0 for name in self.feature_names],
            dtype=np.float32,
        )
        self.is_fitted = True
        return self

    def fit_parquet(self, paths: Iterable[Path], *, batch_size: int = 100_000
                    ) -> "CaptureFoldPreprocessor":
        return self.fit(_batch_frames(paths, self.required_columns, batch_size))

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Transform a packet batch without modifying fitted fold state."""
        if not self.is_fitted or self._active_mask is None:
            raise RuntimeError("The preprocessor must be fitted before transform.")
        result = self._encode_unscaled(frame)
        for name, parameters in self.numeric_parameters.items():
            values = result[name].to_numpy(dtype=np.float64)
            values = (values - float(parameters["centering_value"])) / float(
                parameters["scaling_divisor"]
            )
            result[name] = np.nan_to_num(values, nan=0.0).astype("float32")
        matrix = result.to_numpy(dtype=np.float32, copy=False)
        matrix *= self._active_mask
        if not np.isfinite(matrix).all():
            raise ValueError("Transformed model features contain NaN or infinity.")
        return pd.DataFrame(matrix, index=frame.index, columns=self.feature_names)

    def to_dict(self, *, fold: str, training_scenarios: list[str]) -> dict:
        if not self.is_fitted:
            raise RuntimeError("Cannot serialize an unfitted preprocessor.")
        return {
            "artifact_version": 1,
            "fold": fold,
            "training_scenarios": list(training_scenarios),
            "training_rows": self.training_rows,
            "preprocessing_contract_sha256": preprocessing_schema_sha256(self.schema),
            "feature_count": len(self.feature_names),
            "feature_names_sha256": _stable_name_hash(self.feature_names),
            "feature_names": self.feature_names,
            "numeric_parameters": self.numeric_parameters,
            "active_feature_count": len(self.active_features),
            "active_features": self.active_features,
            "masked_feature_count": len(self.masked_features),
            "masked_features": self.masked_features,
            "validation_was_used_for_fit": False,
        }

    @classmethod
    def from_dict(cls, schema: dict, artifact: dict) -> "CaptureFoldPreprocessor":
        instance = cls(schema)
        if artifact.get("artifact_version") != 1:
            raise ValueError("Unsupported preprocessor artifact version.")
        if artifact.get("preprocessing_contract_sha256") != preprocessing_schema_sha256(schema):
            raise ValueError("Preprocessor artifact and preprocessing contract differ.")
        if artifact.get("feature_names") != instance.feature_names:
            raise ValueError("Preprocessor artifact has a different feature order.")
        if artifact.get("feature_names_sha256") != _stable_name_hash(instance.feature_names):
            raise ValueError("Preprocessor feature-order hash mismatch.")
        instance.numeric_parameters = artifact["numeric_parameters"]
        instance.active_features = artifact["active_features"]
        instance.masked_features = artifact["masked_features"]
        active = set(instance.active_features)
        masked = set(instance.masked_features)
        if (active & masked) or ((active | masked) != set(instance.feature_names)):
            raise ValueError("Preprocessor active and masked features do not partition the model view.")
        instance._active_mask = np.asarray(
            [1.0 if name in active else 0.0 for name in instance.feature_names],
            dtype=np.float32,
        )
        instance.training_rows = int(artifact["training_rows"])
        instance.is_fitted = True
        return instance


def _audit_scenario(path: Path, preprocessor: CaptureFoldPreprocessor,
                    batch_size: int) -> dict:
    rows = 0
    nonzero = 0
    maximum_absolute_value = 0.0
    active_nonzero = {name: 0 for name in preprocessor.active_features}
    for frame in _batch_frames([path], preprocessor.required_columns, batch_size):
        transformed = preprocessor.transform(frame)
        values = transformed.to_numpy(dtype=np.float32, copy=False)
        rows += len(transformed)
        nonzero += int(np.count_nonzero(values))
        if values.size:
            maximum_absolute_value = max(
                maximum_absolute_value, float(np.abs(values).max())
            )
        for name in active_nonzero:
            active_nonzero[name] += int(np.count_nonzero(transformed[name].to_numpy()))
    zero_active = sorted(name for name, count in active_nonzero.items() if count == 0)
    return {
        "rows": rows,
        "feature_count": len(preprocessor.feature_names),
        "finite_values": True,
        "nonzero_values": nonzero,
        "maximum_absolute_value": maximum_absolute_value,
        "active_features_zero_in_this_scenario": zero_active,
    }


def run_capture_preprocessing_audit(*, manifest_path: Path, packet_schema_path: Path,
                                    preprocessing_schema_path: Path,
                                    prepared_run_dir: Path, output_dir: Path,
                                    batch_size: int = 100_000) -> dict:
    """Fit both fold preprocessors and audit transformed batches without saving them."""
    manifest_path = Path(manifest_path)
    packet_schema_path = Path(packet_schema_path)
    preprocessing_schema_path = Path(preprocessing_schema_path)
    prepared_run_dir = Path(prepared_run_dir)
    output_dir = Path(output_dir)
    manifest, packet_schema, reports, packet_paths = load_prepared_full_dev(
        prepared_run_dir, manifest_path, packet_schema_path,
    )
    schema = load_preprocessing_schema(preprocessing_schema_path, packet_schema)
    names = validate_model_preprocessing_schema(schema)
    output_dir.mkdir(parents=True, exist_ok=False)
    for path in (manifest_path, packet_schema_path, preprocessing_schema_path):
        shutil.copyfile(path, output_dir / path.name)

    fold_reports = {}
    scenarios = selected_scenarios(manifest, "FULL_DEV")
    for fold, split in manifest["validation"]["folds"].items():
        print(f"Fitting preprocessing fold {fold}...", flush=True)
        preprocessor = CaptureFoldPreprocessor(schema).fit_parquet(
            [packet_paths[name] for name in split["train"]], batch_size=batch_size,
        )
        artifact = preprocessor.to_dict(fold=fold, training_scenarios=split["train"])
        artifact_name = f"fold_{fold}_preprocessor.json"
        artifact_path = output_dir / artifact_name
        write_json(artifact_path, artifact)
        restored = CaptureFoldPreprocessor.from_dict(schema, artifact)
        scenario_audits = {}
        for scenario in scenarios:
            partition = "train" if scenario in split["train"] else "validation"
            print(f"Auditing fold {fold} {partition} scenario {scenario}...", flush=True)
            audit = _audit_scenario(packet_paths[scenario], restored, batch_size)
            expected_rows = int(reports[scenario]["counts"]["packets"])
            if audit["rows"] != expected_rows:
                raise AssertionError(f"Transformed row-count mismatch for {scenario}.")
            audit["partition"] = partition
            scenario_audits[scenario] = audit
        fold_reports[fold] = {
            "train_scenarios": split["train"],
            "validation_scenarios": split["validate"],
            "preprocessor_artifact": artifact_name,
            "preprocessor_sha256": sha256_file(artifact_path),
            "feature_count": len(names),
            "active_feature_count": len(preprocessor.active_features),
            "active_features": preprocessor.active_features,
            "masked_feature_count": len(preprocessor.masked_features),
            "masked_features": preprocessor.masked_features,
            "scenario_audits": scenario_audits,
        }

    repository = manifest_path.resolve().parent.parent
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository,
        capture_output=True, text=True, check=False,
    )
    tree = subprocess.run(
        ["git", "status", "--short"], cwd=repository,
        capture_output=True, text=True, check=False,
    )
    result = {
        "report_version": REPORT_VERSION,
        "status": "review_required",
        "prepared_run": str(prepared_run_dir),
        "manifest_sha256": sha256_file(manifest_path),
        "packet_schema_sha256": sha256_file(packet_schema_path),
        "preprocessing_schema_sha256": sha256_file(preprocessing_schema_path),
        "preprocessing_contract_sha256": preprocessing_schema_sha256(schema),
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
        "working_tree_status": tree.stdout.strip() if tree.returncode == 0 else "unavailable",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
        "model_feature_count": len(names),
        "model_feature_names_sha256": _stable_name_hash(names),
        "model_feature_names": names,
        "transformed_packet_artifacts_written": False,
        "folds": fold_reports,
        "next_action": (
            "Review fold masks, numeric parameters, finite-value checks, and scenario row "
            "conservation before freezing preprocessing and building five-second windows."
        ),
    }
    write_json(output_dir / "capture_preprocessing_audit.json", result)
    write_json(output_dir / "run_status.json", {
        "complete": True,
        "status": "review_required",
        "report": "capture_preprocessing_audit.json",
    })
    return result
