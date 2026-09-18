"""Fold-aware feature profiling for prepared cAPTure packet artifacts."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .capture_data import load_manifest, selected_scenarios, sha256_file, write_json
from .capture_prepare import (
    load_packet_schema,
    ordered_feature_names,
    validate_audit_manifest_compatibility,
)


MISSING_CATEGORY = "__MISSING__"


def load_preprocessing_schema(path: Path, packet_schema: dict) -> dict:
    """Load and validate the candidate semantic preprocessing contract."""
    schema = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if schema.get("schema_version") != "capture_preprocessing_v1":
        raise ValueError("Unsupported cAPTure preprocessing schema version.")
    if schema.get("status") not in {"candidate_pending_feature_profile", "frozen"}:
        raise ValueError("The preprocessing schema has an unsupported status.")
    if schema.get("canonical_packet_schema_version") != packet_schema["schema_version"]:
        raise ValueError("The preprocessing and canonical packet schemas do not match.")

    roles = schema.get("feature_roles", {})
    required_roles = {"binary", "numeric_magnitude", "categorical_code", "tcp_port"}
    if set(roles) != required_roles:
        raise ValueError(f"Feature roles must be exactly {sorted(required_roles)}.")
    assigned = [name for role in required_roles for name in roles[role]]
    canonical = ordered_feature_names(packet_schema)
    if len(assigned) != len(set(assigned)) or set(assigned) != set(canonical):
        missing = sorted(set(canonical) - set(assigned))
        extra = sorted(set(assigned) - set(canonical))
        raise ValueError(f"Feature-role coverage mismatch; missing={missing}, extra={extra}.")

    port = schema.get("port_encoding", {})
    ordered = port.get("ordered_roles", [])
    named = port.get("roles", {})
    residual = port.get("residual_roles", {})
    if len(ordered) != len(set(ordered)) or set(ordered) != set(named) | set(residual):
        raise ValueError("Port roles must form one unique ordered taxonomy.")
    declared_ports: list[int] = []
    for role, values in named.items():
        if not values or any(not isinstance(value, int) or value <= 0 or value > 65535
                             for value in values):
            raise ValueError(f"Invalid named port role: {role}")
        declared_ports.extend(values)
    if len(declared_ports) != len(set(declared_ports)):
        raise ValueError("Named port roles overlap.")
    if roles["tcp_port"] != [port.get("source_feature"), port.get("destination_feature")]:
        raise ValueError("TCP-port features must be ordered as source then destination.")
    if port.get("tcp_indicator") not in roles["binary"]:
        raise ValueError("The TCP applicability indicator must be a binary feature.")

    protocol = schema.get("protocol_encoding", {})
    protocol_columns = [
        name
        for group in ("network_layer", "transport_layer", "application_layer", "special_state")
        for name in protocol.get(group, [])
    ]
    if len(protocol_columns) != len(set(protocol_columns)) or not set(protocol_columns) <= set(roles["binary"]):
        raise ValueError("Protocol-layer indicators must be unique binary features.")

    expected_after_ports = len(canonical) - len(roles["tcp_port"]) + 2 * len(ordered)
    if schema["primary_model_view"].get("feature_count_after_port_encoding_only") != expected_after_ports:
        raise ValueError("The declared post-port feature count is inconsistent.")
    return schema


def preprocessing_schema_sha256(schema: dict) -> str:
    encoded = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _binary_indicator(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_numeric(values, errors="coerce")
    if parsed.isna().any() or not parsed.isin([0, 1]).all():
        raise ValueError(f"{name} must contain only non-null binary values.")
    return parsed.astype("int8")


def encode_tcp_port_roles(values: pd.Series, is_tcp: pd.Series, schema: dict,
                          *, prefix: str = "tcp_port_role") -> pd.DataFrame:
    """Encode one TCP port direction with the fixed exhaustive taxonomy."""
    indicator = _binary_indicator(is_tcp, schema["port_encoding"]["tcp_indicator"])
    numeric = pd.to_numeric(values, errors="coerce")
    nonnull_input = values.notna()
    parse_failure = nonnull_input & numeric.isna()
    if parse_failure.any() or not numeric.dropna().map(np.isfinite).all():
        raise ValueError("TCP ports contain non-numeric or non-finite values.")
    present = numeric.notna()
    if (indicator.eq(1) & ~present).any():
        raise ValueError("A TCP packet has a null TCP port.")
    if (indicator.eq(0) & present).any():
        raise ValueError("A non-TCP packet has a non-null TCP port.")
    valid = numeric.loc[present]
    if ((np.floor(valid) != valid) | (valid < 0) | (valid > 65535)).any():
        raise ValueError("TCP ports must be integers in the range 0..65535.")

    roles = schema["port_encoding"]["ordered_roles"]
    encoded = pd.DataFrame(
        np.zeros((len(values), len(roles)), dtype=np.float32),
        index=values.index,
        columns=[f"{prefix}_{role}" for role in roles],
    )
    unassigned = present.copy()
    integer = numeric.fillna(0).astype("int64")
    for role, ports in schema["port_encoding"]["roles"].items():
        mask = present & integer.isin(ports)
        encoded.loc[mask, f"{prefix}_{role}"] = 1.0
        unassigned &= ~mask

    residual_masks = {
        "other_privileged": unassigned & integer.between(1, 1023),
        "other_registered": unassigned & integer.between(1024, 49151),
        "other_dynamic": unassigned & integer.between(49152, 65535),
        "zero_or_reserved": unassigned & integer.eq(0),
        "not_applicable_to_tcp": ~present & indicator.eq(0),
    }
    for role, mask in residual_masks.items():
        encoded.loc[mask, f"{prefix}_{role}"] = 1.0
    if not encoded.sum(axis=1).eq(1.0).all():
        raise AssertionError("Every packet must receive exactly one TCP-port role.")
    return encoded


def tcp_port_role_labels(values: pd.Series, is_tcp: pd.Series, schema: dict) -> pd.Series:
    """Return stable role labels for reporting without exposing raw port identity."""
    encoded = encode_tcp_port_roles(values, is_tcp, schema, prefix="role")
    roles = schema["port_encoding"]["ordered_roles"]
    indexes = encoded.to_numpy().argmax(axis=1)
    return pd.Series([roles[index] for index in indexes], index=values.index, dtype="string")


def validate_protocol_indicators(frame: pd.DataFrame, schema: dict) -> dict[str, int]:
    """Validate the multi-layer protocol contract and return indicator counts."""
    protocol = schema["protocol_encoding"]
    groups = {
        name: protocol[name]
        for name in ("network_layer", "transport_layer", "application_layer", "special_state")
    }
    parsed = {
        column: _binary_indicator(frame[column], column)
        for columns in groups.values() for column in columns
    }
    if pd.DataFrame({name: parsed[name] for name in groups["network_layer"]}).sum(axis=1).gt(1).any():
        raise ValueError("A packet activates multiple network-layer indicators.")
    if pd.DataFrame({name: parsed[name] for name in groups["transport_layer"]}).sum(axis=1).gt(1).any():
        raise ValueError("A packet activates multiple transport-layer indicators.")
    if (parsed["is_mqtt"].gt(parsed["is_tcp"])).any():
        raise ValueError("An MQTT packet is not marked as TCP.")
    if (parsed["is_ssh"].gt(parsed["is_tcp"])).any():
        raise ValueError("An SSH packet is not marked as TCP.")
    return {column: int(values.sum()) for column, values in parsed.items()}


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _json_number(value: object) -> int | float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return int(numeric) if numeric.is_integer() else numeric


def _category_key(value: object) -> str:
    if value is None or pd.isna(value):
        return MISSING_CATEGORY
    numeric = float(value)
    return str(int(numeric)) if numeric.is_integer() else format(numeric, ".12g")


def _feature_statistics(path: Path, features: list[str], quantiles: list[float]) -> dict[str, dict]:
    expressions = ["count(*) AS packet_count"]
    for index, name in enumerate(features):
        column = _quote_identifier(name)
        expressions.extend([
            f"count({column}) AS f{index}_nonnull",
            f"count(DISTINCT {column}) AS f{index}_distinct",
            f"min({column}) AS f{index}_min",
            f"max({column}) AS f{index}_max",
            f"avg({column}) AS f{index}_mean",
            f"stddev_pop({column}) AS f{index}_std",
            f"approx_quantile({column}, {quantiles}) AS f{index}_quantiles",
        ])
    query = f"SELECT {', '.join(expressions)} FROM read_parquet({_sql_literal(str(path))})"
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 2")
        connection.execute("SET memory_limit = '4GB'")
        row = connection.execute(query).fetchone()
        columns = [item[0] for item in connection.description]
    finally:
        connection.close()
    result = dict(zip(columns, row))
    packets = int(result["packet_count"])
    statistics = {}
    for index, name in enumerate(features):
        values = result[f"f{index}_quantiles"]
        if values is None:
            values = [None] * len(quantiles)
        statistics[name] = {
            "nonnull": int(result[f"f{index}_nonnull"]),
            "null": packets - int(result[f"f{index}_nonnull"]),
            "distinct_nonnull": int(result[f"f{index}_distinct"]),
            "minimum": _json_number(result[f"f{index}_min"]),
            "maximum": _json_number(result[f"f{index}_max"]),
            "mean": _json_number(result[f"f{index}_mean"]),
            "standard_deviation": _json_number(result[f"f{index}_std"]),
            "quantiles": {
                str(quantile): _json_number(value)
                for quantile, value in zip(quantiles, values)
            },
        }
    return statistics


def _top_counter(counter: Counter, limit: int) -> dict[str, int]:
    return {
        str(key): int(value)
        for key, value in sorted(counter.items(), key=lambda item: (-item[1], str(item[0])))[:limit]
    }


def profile_scenario(path: Path, report: dict, packet_schema: dict,
                     preprocessing_schema: dict, *, batch_size: int = 250_000) -> tuple[dict, dict[str, Counter]]:
    """Profile one prepared scenario with bounded memory."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    features = ordered_feature_names(packet_schema)
    quantiles = [float(value) for value in preprocessing_schema["profile"]["quantiles"]]
    statistics = _feature_statistics(path, features, quantiles)
    categorical = preprocessing_schema["feature_roles"]["categorical_code"]
    ports = preprocessing_schema["feature_roles"]["tcp_port"]
    protocol_columns = [
        name
        for group in ("network_layer", "transport_layer", "application_layer", "special_state")
        for name in preprocessing_schema["protocol_encoding"][group]
    ]
    read_columns = list(dict.fromkeys([*categorical, *ports, *protocol_columns]))
    category_counts = {name: Counter() for name in categorical}
    port_role_counts = {"source": Counter(), "destination": Counter()}
    raw_port_counts = {"source": Counter(), "destination": Counter()}
    protocol_counts = Counter()
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=read_columns):
        frame = batch.to_pandas()
        for name in categorical:
            category_counts[name].update(_category_key(value) for value in frame[name].array)
        protocol_counts.update(validate_protocol_indicators(frame, preprocessing_schema))
        for direction, name in zip(("source", "destination"), ports):
            labels = tcp_port_role_labels(frame[name], frame["is_tcp"], preprocessing_schema)
            port_role_counts[direction].update(labels.tolist())
            raw_port_counts[direction].update(
                _category_key(value) for value in frame[name].dropna().array
            )

    packets = int(report["counts"]["packets"])
    if any(sum(counter.values()) != packets for counter in port_role_counts.values()):
        raise AssertionError("Port-role profiling did not conserve packet rows.")
    if any(value > packets for value in protocol_counts.values()):
        raise AssertionError("Protocol-indicator profiling exceeded packet rows.")
    limit = int(preprocessing_schema["profile"]["maximum_reported_values_per_categorical_feature"])
    profile = {
        "scenario": report["scenario"],
        "packets": packets,
        "feature_statistics": statistics,
        "categorical_counts": {
            name: _top_counter(counter, limit) for name, counter in category_counts.items()
        },
        "categorical_counts_truncated": {
            name: len(counter) > limit for name, counter in category_counts.items()
        },
        "port_role_counts": {
            direction: {role: int(counter.get(role, 0))
                        for role in preprocessing_schema["port_encoding"]["ordered_roles"]}
            for direction, counter in port_role_counts.items()
        },
        "top_raw_nonnull_ports": {
            direction: _top_counter(counter, 20) for direction, counter in raw_port_counts.items()
        },
        "protocol_indicator_counts": dict(sorted(protocol_counts.items())),
    }
    return profile, category_counts


def _merge_counters(counters: Iterable[Counter]) -> Counter:
    result = Counter()
    for counter in counters:
        result.update(counter)
    return result


def build_fold_profiles(manifest: dict, scenario_profiles: dict[str, dict],
                        category_counts: dict[str, dict[str, Counter]],
                        preprocessing_schema: dict) -> dict[str, dict]:
    """Summarize train-only constancy and validation categorical coverage."""
    result = {}
    all_features = [
        name for role in preprocessing_schema["feature_roles"].values() for name in role
    ]
    categorical = preprocessing_schema["feature_roles"]["categorical_code"]
    for fold_name, split in manifest["validation"]["folds"].items():
        train, validate = split["train"], split["validate"]
        train_packets = sum(scenario_profiles[name]["packets"] for name in train)
        constants = []
        all_missing = []
        feature_fit = {}
        for feature in all_features:
            stats = [scenario_profiles[name]["feature_statistics"][feature] for name in train]
            nonnull = sum(item["nonnull"] for item in stats)
            null = sum(item["null"] for item in stats)
            minima = [item["minimum"] for item in stats if item["minimum"] is not None]
            maxima = [item["maximum"] for item in stats if item["maximum"] is not None]
            constant = bool(nonnull and min(minima) == max(maxima))
            if nonnull == 0:
                all_missing.append(feature)
            elif constant:
                constants.append(feature)
            feature_fit[feature] = {
                "training_nonnull": nonnull,
                "training_null": null,
                "training_null_fraction": null / train_packets,
                "constant_nonmissing": constant,
                "all_missing": nonnull == 0,
            }

        coverage = {}
        for feature in categorical:
            train_counter = _merge_counters(category_counts[name][feature] for name in train)
            validation_counter = _merge_counters(category_counts[name][feature] for name in validate)
            train_values = set(train_counter) - {MISSING_CATEGORY}
            unknown = (set(validation_counter) - {MISSING_CATEGORY}) - train_values
            validation_nonnull = sum(
                count for value, count in validation_counter.items() if value != MISSING_CATEGORY
            )
            unknown_rows = sum(validation_counter[value] for value in unknown)
            coverage[feature] = {
                "training_distinct_nonnull": len(train_values),
                "validation_distinct_nonnull": len(set(validation_counter) - {MISSING_CATEGORY}),
                "validation_unseen_values": sorted(unknown),
                "validation_unseen_rows": int(unknown_rows),
                "validation_unseen_fraction_of_nonnull": (
                    unknown_rows / validation_nonnull if validation_nonnull else 0.0
                ),
                "encoder_refit_on_validation": False,
            }
        result[fold_name] = {
            "train_scenarios": train,
            "validation_scenarios": validate,
            "training_packets": train_packets,
            "training_constants": sorted(constants),
            "training_all_missing": sorted(all_missing),
            "feature_fit_summary": feature_fit,
            "categorical_validation_coverage": coverage,
        }
    return result


def load_prepared_full_dev(prepared_run_dir: Path, current_manifest_path: Path,
                           current_packet_schema_path: Path) -> tuple[dict, dict, dict[str, dict], dict[str, Path]]:
    """Validate a completed canonical preparation run and its artifact hashes."""
    prepared_run_dir = Path(prepared_run_dir)
    current_manifest = load_manifest(current_manifest_path)
    current_packet_schema = load_packet_schema(current_packet_schema_path)
    archived_manifest_path = prepared_run_dir / Path(current_manifest_path).name
    archived_schema_path = prepared_run_dir / Path(current_packet_schema_path).name
    for path in (archived_manifest_path, archived_schema_path,
                 prepared_run_dir / "run_config.json", prepared_run_dir / "run_status.json"):
        if not path.is_file():
            raise FileNotFoundError(f"Prepared run is missing {path.name}.")
    archived_manifest = load_manifest(archived_manifest_path)
    archived_schema = load_packet_schema(archived_schema_path)
    validate_audit_manifest_compatibility(current_manifest, archived_manifest)
    if current_packet_schema != archived_schema:
        raise ValueError("The current canonical packet schema differs from the prepared-run schema.")
    config = json.loads((prepared_run_dir / "run_config.json").read_text(encoding="utf-8"))
    status = json.loads((prepared_run_dir / "run_status.json").read_text(encoding="utf-8"))
    scenarios = selected_scenarios(current_manifest, "FULL_DEV")
    if (status.get("complete") is not True or status.get("mode") != "FULL_DEV"
            or config.get("mode") != "FULL_DEV" or config.get("scenarios") != scenarios):
        raise ValueError("The prepared run is not a complete FULL_DEV run for these scenarios.")
    if config.get("manifest_sha256") != sha256_file(archived_manifest_path):
        raise ValueError("Prepared-run manifest hash mismatch.")
    if config.get("packet_schema_sha256") != sha256_file(archived_schema_path):
        raise ValueError("Prepared-run packet-schema hash mismatch.")

    reports, packet_paths = {}, {}
    for scenario in scenarios:
        directory = prepared_run_dir / scenario
        report_path = directory / "preparation_report.json"
        packet_path = directory / archived_schema["output_artifact"]
        checksum_path = directory / "artifact_checksums.json"
        if not all(path.is_file() for path in (report_path, packet_path, checksum_path)):
            raise FileNotFoundError(f"Prepared artifacts are incomplete for {scenario}.")
        checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
        if checksums.get(report_path.name) != sha256_file(report_path):
            raise ValueError(f"Prepared report checksum mismatch: {scenario}")
        if checksums.get(packet_path.name) != sha256_file(packet_path):
            raise ValueError(f"Prepared packet checksum mismatch: {scenario}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (report.get("scenario") != scenario
                or report.get("schema_version") != archived_schema["schema_version"]
                or report.get("output_sha256") != checksums[packet_path.name]):
            raise ValueError(f"Prepared report binding mismatch: {scenario}")
        parquet_columns = pq.read_schema(packet_path).names
        missing = sorted(set(ordered_feature_names(archived_schema)) - set(parquet_columns))
        if missing:
            raise ValueError(f"Prepared packet features are missing for {scenario}: {missing}")
        reports[scenario], packet_paths[scenario] = report, packet_path
    return current_manifest, current_packet_schema, reports, packet_paths


def run_capture_feature_profile(*, manifest_path: Path, packet_schema_path: Path,
                                preprocessing_schema_path: Path, prepared_run_dir: Path,
                                output_dir: Path, batch_size: int = 250_000) -> dict:
    """Profile all development scenarios and write a provenance-bound report."""
    manifest_path = Path(manifest_path)
    packet_schema_path = Path(packet_schema_path)
    preprocessing_schema_path = Path(preprocessing_schema_path)
    prepared_run_dir, output_dir = Path(prepared_run_dir), Path(output_dir)
    manifest, packet_schema, reports, packet_paths = load_prepared_full_dev(
        prepared_run_dir, manifest_path, packet_schema_path,
    )
    preprocessing_schema = load_preprocessing_schema(preprocessing_schema_path, packet_schema)
    output_dir.mkdir(parents=True, exist_ok=False)
    for path in (manifest_path, packet_schema_path, preprocessing_schema_path):
        shutil.copyfile(path, output_dir / path.name)

    scenario_profiles = {}
    category_counts = {}
    for scenario in selected_scenarios(manifest, "FULL_DEV"):
        print(f"Profiling {scenario}...", flush=True)
        profile, counters = profile_scenario(
            packet_paths[scenario], reports[scenario], packet_schema,
            preprocessing_schema, batch_size=batch_size,
        )
        scenario_profiles[scenario] = profile
        category_counts[scenario] = counters
        print(f"Profiled {scenario}: {profile['packets']:,} packets", flush=True)

    fold_profiles = build_fold_profiles(
        manifest, scenario_profiles, category_counts, preprocessing_schema,
    )
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
        "report_version": 1,
        "status": "review_required",
        "prepared_run": str(prepared_run_dir),
        "prepared_run_config_sha256": sha256_file(prepared_run_dir / "run_config.json"),
        "manifest_sha256": sha256_file(manifest_path),
        "packet_schema_sha256": sha256_file(packet_schema_path),
        "preprocessing_schema_sha256": sha256_file(preprocessing_schema_path),
        "preprocessing_contract_sha256": preprocessing_schema_sha256(preprocessing_schema),
        "git_commit": revision.stdout.strip() if revision.returncode == 0 else "unavailable",
        "working_tree_status": tree.stdout.strip() if tree.returncode == 0 else "unavailable",
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "duckdb": duckdb.__version__,
        },
        "candidate_dimensions": preprocessing_schema["primary_model_view"],
        "scenario_profiles": scenario_profiles,
        "fold_profiles": fold_profiles,
        "next_action": (
            "Review fixed port roles, categorical code domains, fold constants, "
            "validation coverage, and numeric ranges before freezing preprocessing."
        ),
    }
    write_json(output_dir / "capture_feature_profile.json", result)
    write_json(output_dir / "run_status.json", {
        "complete": True,
        "status": "review_required",
        "report": "capture_feature_profile.json",
    })
    return result
