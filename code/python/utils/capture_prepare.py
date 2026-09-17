"""Disk-backed canonical packet preparation for the cAPTure study."""

from __future__ import annotations

from collections import Counter
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from . import capture_gate0_review
from .capture_data import load_manifest, selected_scenarios, sha256_file, write_json


METADATA_TYPES = {
    "packet_id": pa.string(),
    "source_row_id": pa.int64(),
    "packet_timestamp": pa.timestamp("ns", tz="UTC"),
    "packet_timestamp_ns": pa.int64(),
    "scenario": pa.string(),
    "attack_chain": pa.string(),
    "benign_source": pa.string(),
    "author_split": pa.string(),
    "src_endpoint": pa.string(),
    "dst_endpoint": pa.string(),
    "src_node_role": pa.string(),
    "dst_node_role": pa.string(),
    "binary_label": pa.int8(),
    "attack_step": pa.string(),
    "phase": pa.string(),
    "sequence_id": pa.string(),
    "raw_row_sha256": pa.string(),
}

SOURCE_METADATA_COLUMNS = tuple(
    column for column in METADATA_TYPES
    if column not in {"src_node_role", "dst_node_role"}
)
MAC_PATTERN = re.compile(r"^[0-9a-f]{2}(?::[0-9a-f]{2}){5}$")
MULTICAST_SECOND_NIBBLES = frozenset("13579bdf")
AUDIT_SCENARIO_BINDING_KEYS = (
    "attack_chain", "author_split", "benign_source", "assignment",
    "source_file_id", "expected_filename", "expected_size_bytes",
)


def load_packet_schema(path: Path) -> dict:
    """Load and validate the candidate canonical packet schema."""
    schema = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if schema.get("schema_version") != "capture_packet_v1":
        raise ValueError("Unsupported cAPTure packet schema version.")
    if schema.get("status") not in {"candidate_pending_runtime_validation", "frozen"}:
        raise ValueError("The packet schema has an unsupported status.")
    output_artifact = schema.get("output_artifact")
    if not output_artifact or Path(output_artifact).name != output_artifact:
        raise ValueError("The output artifact must be a plain filename.")
    features = schema.get("features", {})
    expected_groups = {"indicators", "numeric", "bit_flags"}
    if set(features) != expected_groups:
        raise ValueError(f"Feature groups must be exactly {sorted(expected_groups)}.")
    names = [name for group in expected_groups for name in features[group]]
    if len(names) != len(set(names)) or not names:
        raise ValueError("Canonical feature names must be non-empty and unique.")
    for name, item in features["indicators"].items():
        if item.get("transform") not in {
            "present", "mac_broadcast", "mac_multicast_excluding_broadcast",
        } or not item.get("source"):
            raise ValueError(f"Invalid indicator definition: {name}")
    for group in ("numeric", "bit_flags"):
        for name, item in features[group].items():
            if item.get("parser") not in {"decimal", "integer_auto_base"}:
                raise ValueError(f"Invalid numeric parser: {name}")
            if not item.get("source"):
                raise ValueError(f"Missing source column: {name}")
            if group == "bit_flags" and (
                not isinstance(item.get("mask"), int) or item["mask"] <= 0
            ):
                raise ValueError(f"Invalid bit mask: {name}")
    temporal = schema.get("temporal_metadata", {})
    if (temporal.get("canonical_packet_table_is_window_independent") is not True
            or temporal.get("window_indexes_are_created_during_graph_construction") is not True):
        raise ValueError("Canonical packet preparation must remain window-independent.")
    return schema


def ordered_feature_names(schema: dict) -> list[str]:
    """Return the stable feature order recorded by the schema."""
    return [
        *schema["features"]["indicators"],
        *schema["features"]["numeric"],
        *schema["features"]["bit_flags"],
    ]


def validate_audit_manifest_compatibility(current: dict, archived: dict) -> None:
    """Ensure downstream decisions did not alter the audited data bindings."""
    if current.get("dataset") != archived.get("dataset"):
        raise ValueError("The current and archived manifests name different datasets.")
    for mode in ("SMOKE", "FULL_DEV"):
        if selected_scenarios(current, mode) != selected_scenarios(archived, mode):
            raise ValueError(f"The current manifest changed the audited {mode} scenarios.")
    for scenario in selected_scenarios(archived, "FULL_DEV"):
        current_binding = {
            key: current["scenarios"][scenario].get(key)
            for key in AUDIT_SCENARIO_BINDING_KEYS
        }
        archived_binding = {
            key: archived["scenarios"][scenario].get(key)
            for key in AUDIT_SCENARIO_BINDING_KEYS
        }
        if current_binding != archived_binding:
            raise ValueError(f"The current manifest changed the audited binding for {scenario}.")
    for key in ("strategy", "folds"):
        if current.get("validation", {}).get(key) != archived.get("validation", {}).get(key):
            raise ValueError("The current manifest changed the audited validation folds.")
    current_widths = current.get("windows", {}).get("candidate_durations_seconds")
    archived_widths = archived.get("windows", {}).get("candidate_durations_seconds")
    if current_widths != archived_widths:
        raise ValueError("The current manifest changed the audited window candidates.")


def _clean_strings(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().replace("", pd.NA)


def _parse_integer(value: object) -> int | None:
    if value is None or value is pd.NA or pd.isna(value):
        return None
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text, 10)
    if re.fullmatch(r"[+-]?0[xX][0-9a-fA-F]+", text):
        sign = -1 if text.startswith("-") else 1
        unsigned = text[1:] if text[:1] in "+-" else text
        return sign * int(unsigned, 16)
    raise ValueError(text)


def parse_numeric(values: pd.Series, parser: str, feature_name: str) -> pd.Series:
    """Parse one nullable numeric feature and reject non-empty malformed values."""
    cleaned = _clean_strings(values)
    if parser == "decimal":
        parsed = pd.to_numeric(cleaned, errors="coerce")
        invalid = cleaned.notna() & parsed.isna()
    elif parser == "integer_auto_base":
        converted = []
        invalid_values = []
        for value in cleaned:
            try:
                converted.append(_parse_integer(value))
            except ValueError:
                converted.append(None)
                if len(invalid_values) < 5:
                    invalid_values.append(str(value))
        if invalid_values:
            raise ValueError(
                f"Cannot parse {feature_name} using {parser}; examples: {invalid_values}"
            )
        parsed = pd.Series(converted, index=values.index, dtype="Float64")
        invalid = pd.Series(False, index=values.index)
    else:
        raise ValueError(f"Unsupported parser for {feature_name}: {parser}")
    if invalid.any():
        examples = cleaned[invalid].drop_duplicates().head(5).tolist()
        raise ValueError(f"Cannot parse {feature_name} using {parser}; examples: {examples}")
    numeric = pd.to_numeric(parsed, errors="coerce")
    finite = numeric.dropna().to_numpy(dtype=np.float64)
    if not np.isfinite(finite).all():
        raise ValueError(f"Non-finite values in {feature_name}.")
    return numeric.astype("Float32")


def _normalize_mac(values: pd.Series, column: str) -> pd.Series:
    normalized = _clean_strings(values).str.lower()
    invalid = normalized.isna() | ~normalized.str.fullmatch(MAC_PATTERN.pattern, na=False)
    if invalid.any():
        examples = values[invalid].head(5).tolist()
        raise ValueError(f"Missing or invalid MAC values in {column}: {examples}")
    return normalized


def _mac_roles(values: pd.Series, broadcast: str) -> pd.Series:
    broadcast_mask = values.eq(broadcast)
    multicast_mask = values.str[1].isin(MULTICAST_SECOND_NIBBLES) & ~broadcast_mask
    roles = pd.Series("unicast", index=values.index, dtype="string")
    roles.loc[multicast_mask] = "multicast"
    roles.loc[broadcast_mask] = "broadcast"
    return roles


def required_source_columns(schema: dict) -> list[str]:
    columns = set(SOURCE_METADATA_COLUMNS)
    for group in schema["features"].values():
        columns.update(item["source"] for item in group.values())
    return sorted(columns)


def _minimum_timestamp_ns(path: Path, batch_size: int) -> int:
    minimum = None
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["packet_timestamp_ns"]):
        values = batch.column(0).to_numpy(zero_copy_only=False)
        if len(values):
            candidate = int(values.min())
            minimum = candidate if minimum is None else min(minimum, candidate)
    if minimum is None:
        raise ValueError("The audit Parquet contains no packet timestamps.")
    return minimum


def _output_arrow_schema(schema: dict) -> pa.Schema:
    fields = [pa.field(name, dtype) for name, dtype in METADATA_TYPES.items()]
    fields.extend(pa.field(name, pa.float32()) for name in ordered_feature_names(schema))
    return pa.schema(fields)


def transform_packet_batch(frame: pd.DataFrame, schema: dict, *, origin_ns: int) -> pd.DataFrame:
    """Transform one audit batch into the canonical identity-free packet schema."""
    missing = sorted(set(required_source_columns(schema)) - set(frame.columns))
    if missing:
        raise ValueError(f"Audit batch is missing source columns: {missing}")
    result = frame.loc[:, list(SOURCE_METADATA_COLUMNS)].copy()
    required_metadata = [
        "packet_id", "source_row_id", "packet_timestamp", "packet_timestamp_ns",
        "scenario", "attack_chain", "benign_source", "author_split",
        "src_endpoint", "dst_endpoint", "binary_label", "raw_row_sha256",
    ]
    incomplete = [column for column in required_metadata if result[column].isna().any()]
    if incomplete:
        raise ValueError(f"Canonical metadata contains missing values: {incomplete}")
    if result["packet_id"].duplicated().any():
        raise ValueError("Duplicate packet identifiers inside an audit batch.")
    result["src_endpoint"] = _normalize_mac(result["src_endpoint"], "src_endpoint")
    result["dst_endpoint"] = _normalize_mac(result["dst_endpoint"], "dst_endpoint")
    broadcast = schema["topology"]["broadcast_address"]
    result["src_node_role"] = _mac_roles(result["src_endpoint"], broadcast)
    result["dst_node_role"] = _mac_roles(result["dst_endpoint"], broadcast)

    labels = pd.to_numeric(result["binary_label"], errors="coerce").astype("Int8")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError("Canonical preparation requires complete binary labels.")
    attack_rows = labels.eq(1)
    if result.loc[attack_rows, ["attack_step", "phase", "sequence_id"]].isna().any().any():
        raise ValueError("Attack packets require complete evaluation annotations.")
    result["binary_label"] = labels
    benign_sources = _clean_strings(result["benign_source"])
    if benign_sources.isna().any():
        raise ValueError("Canonical preparation requires a benign source for every packet.")

    timestamps = pd.to_numeric(result["packet_timestamp_ns"], errors="coerce").astype("Int64")
    if timestamps.isna().any() or (timestamps < origin_ns).any():
        raise ValueError("Missing timestamp or timestamp preceding the scenario origin.")
    result["packet_timestamp_ns"] = timestamps.astype("int64")

    indicator_definitions = schema["features"]["indicators"]
    for name, item in indicator_definitions.items():
        transform = item["transform"]
        if transform == "present":
            value = _clean_strings(frame[item["source"]]).notna()
        elif transform == "mac_broadcast":
            value = result["dst_node_role"].eq("broadcast")
        elif transform == "mac_multicast_excluding_broadcast":
            value = result["dst_node_role"].eq("multicast")
        else:
            raise AssertionError(f"Unexpected indicator transform: {transform}")
        result[name] = value.astype("float32")

    parsed_cache: dict[tuple[str, str], pd.Series] = {}
    for name, item in schema["features"]["numeric"].items():
        key = (item["source"], item["parser"])
        if key not in parsed_cache:
            parsed_cache[key] = parse_numeric(frame[item["source"]], item["parser"], name)
        result[name] = parsed_cache[key]
    for name, item in schema["features"]["bit_flags"].items():
        key = (item["source"], item["parser"])
        if key not in parsed_cache:
            parsed_cache[key] = parse_numeric(frame[item["source"]], item["parser"], name)
        integers = parsed_cache[key].astype("Int64")
        result[name] = ((integers & item["mask"]) != 0).astype("Float32")

    return result.loc[:, _output_arrow_schema(schema).names]


def prepare_scenario(source_path: Path, output_dir: Path, schema: dict, scenario: str, *,
                     batch_size: int = 100_000, source_sha256: str | None = None) -> dict:
    """Prepare one scenario without loading the complete Parquet into memory."""
    if batch_size <= 0:
        raise ValueError("Batch size must be positive.")
    source_path, output_dir = Path(source_path), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    output_path = output_dir / schema["output_artifact"]
    source_columns = required_source_columns(schema)
    available = set(pq.read_schema(source_path).names)
    missing = sorted(set(source_columns) - available)
    if missing:
        raise ValueError(f"Audit Parquet is missing required columns: {missing}")
    origin_ns = _minimum_timestamp_ns(source_path, batch_size)
    arrow_schema = _output_arrow_schema(schema)
    writer = pq.ParquetWriter(output_path, arrow_schema, compression="zstd")
    counts = Counter()
    role_counts = Counter()
    null_counts = Counter()
    seen_benign_sources = set()
    previous_timestamp = None
    expected_source_row_id = 0
    try:
        parquet = pq.ParquetFile(source_path)
        for batch in parquet.iter_batches(batch_size=batch_size, columns=source_columns):
            if shutil.disk_usage(output_dir).free < 512 * 1024**2:
                raise OSError("Local disk reserve exhausted during canonical preparation.")
            transformed = transform_packet_batch(batch.to_pandas(), schema, origin_ns=origin_ns)
            row_ids = transformed["source_row_id"].to_numpy(dtype=np.int64)
            expected_ids = np.arange(
                expected_source_row_id, expected_source_row_id + len(transformed), dtype=np.int64,
            )
            if not np.array_equal(row_ids, expected_ids):
                raise ValueError(f"Non-contiguous source row identifiers in {scenario}.")
            expected_source_row_id += len(transformed)
            timestamps = transformed["packet_timestamp_ns"]
            if (timestamps.diff().dropna() < 0).any() or (
                previous_timestamp is not None and int(timestamps.iloc[0]) < previous_timestamp
            ):
                raise ValueError(f"Timestamp order inversion while preparing {scenario}.")
            previous_timestamp = int(timestamps.iloc[-1])
            if set(transformed["scenario"].dropna().unique()) != {scenario}:
                raise ValueError(f"Scenario binding mismatch while preparing {scenario}.")
            counts["packets"] += len(transformed)
            counts["normal_packets"] += int(transformed["binary_label"].eq(0).sum())
            counts["attack_packets"] += int(transformed["binary_label"].eq(1).sum())
            seen_benign_sources.update(transformed["benign_source"].unique().tolist())
            role_counts.update(transformed["dst_node_role"].value_counts().to_dict())
            for name in ordered_feature_names(schema):
                null_counts[name] += int(transformed[name].isna().sum())
            arrays = [pa.array(transformed[field.name], type=field.type, from_pandas=True)
                      for field in arrow_schema]
            writer.write_table(pa.Table.from_arrays(arrays, schema=arrow_schema))
            print(f"{scenario}: prepared {counts['packets']:,} packets", flush=True)
    finally:
        writer.close()
    if not counts["packets"]:
        raise ValueError(f"No packets were prepared for {scenario}.")
    if sum(role_counts.values()) != counts["packets"]:
        raise AssertionError("Every packet must have one destination node role.")
    if len(seen_benign_sources) != 1:
        raise ValueError(f"Expected one benign source in {scenario}: {seen_benign_sources}")
    if previous_timestamp is None or previous_timestamp < origin_ns:
        raise AssertionError("Invalid canonical timestamp range.")
    report = {
        "report_version": 1,
        "schema_version": schema["schema_version"],
        "scenario": scenario,
        "status": "review_required",
        "source_artifact": str(source_path),
        "source_sha256": source_sha256 or sha256_file(source_path),
        "output_artifact": output_path.name,
        "output_sha256": sha256_file(output_path),
        "output_size_bytes": output_path.stat().st_size,
        "scenario_origin_timestamp_ns": origin_ns,
        "last_packet_timestamp_ns": previous_timestamp,
        "duration_seconds": (previous_timestamp - origin_ns) / 1_000_000_000,
        "counts": {key: int(value) for key, value in counts.items()},
        "destination_node_role_counts": {
            key: int(value) for key, value in sorted(role_counts.items())
        },
        "feature_columns": ordered_feature_names(schema),
        "feature_null_counts": dict(sorted(null_counts.items())),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
    }
    write_json(output_dir / "preparation_report.json", report)
    return report


def validate_preparation_smoke_review(path: Path, manifest_hash: str,
                                      schema_hash: str, manifest: dict) -> None:
    """Require an explicit real-data transformation review before FULL_DEV."""
    review = json.loads(Path(path).read_text(encoding="utf-8"))
    if (review.get("approved") is not True
            or review.get("manifest_sha256") != manifest_hash
            or review.get("packet_schema_sha256") != schema_hash):
        raise ValueError("FULL_DEV preparation requires a matching approved smoke review.")
    if not review.get("review_notes", "").strip():
        raise ValueError("Record the preparation smoke-review rationale.")
    for scenario in selected_scenarios(manifest, "SMOKE"):
        entry = review.get("reports", {}).get(scenario, {})
        report_path = Path(entry.get("path", ""))
        if not report_path.is_file() or sha256_file(report_path) != entry.get("sha256"):
            raise ValueError(f"Preparation smoke report mismatch: {scenario}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("scenario") != scenario or report.get("schema_version") != "capture_packet_v1":
            raise ValueError(f"Invalid preparation smoke report binding: {scenario}")


def run_capture_preparation(*, manifest_path: Path, packet_schema_path: Path,
                            audit_run_dir: Path, decision_audit_path: Path,
                            mode: str, local_root: Path, drive_run_dir: Path,
                            smoke_review_path: Path | None = None,
                            batch_size: int = 100_000) -> dict:
    """Prepare selected scenarios sequentially and persist verified Drive artifacts."""
    manifest_path = Path(manifest_path)
    packet_schema_path = Path(packet_schema_path)
    audit_run_dir = Path(audit_run_dir)
    decision_audit_path = Path(decision_audit_path)
    local_root, drive_run_dir = Path(local_root), Path(drive_run_dir)
    manifest = load_manifest(manifest_path)
    schema = load_packet_schema(packet_schema_path)
    manifest_hash, schema_hash = sha256_file(manifest_path), sha256_file(packet_schema_path)
    archived_manifest_path = audit_run_dir / "experiment_manifest.yaml"
    if not archived_manifest_path.is_file():
        raise FileNotFoundError("The source audit run is missing its archived manifest.")
    archived_manifest = load_manifest(archived_manifest_path)
    archived_manifest_hash = sha256_file(archived_manifest_path)
    validate_audit_manifest_compatibility(manifest, archived_manifest)
    decision = json.loads(decision_audit_path.read_text(encoding="utf-8"))
    if (decision.get("manifest_sha256") != archived_manifest_hash
            or decision.get("module_sha256") != sha256_file(Path(capture_gate0_review.__file__))
            or decision.get("status") != "review_required" or decision.get("blockers")
            or Path(decision.get("full_dev_run", "")) != audit_run_dir):
        raise ValueError("The Gate-0 decision audit is missing, blocked, or bound to another run.")
    _, audit_reports, packet_paths = capture_gate0_review.load_full_dev_artifacts(
        audit_run_dir, archived_manifest_path,
    )
    scenarios = selected_scenarios(manifest, mode)
    if mode == "FULL_DEV":
        if smoke_review_path is None:
            raise ValueError("FULL_DEV preparation requires a reviewed preparation SMOKE run.")
        validate_preparation_smoke_review(
            smoke_review_path, manifest_hash, schema_hash, manifest,
        )
    local_root.mkdir(parents=True, exist_ok=True)
    drive_run_dir.mkdir(parents=True, exist_ok=False)
    config = {
        "manifest_sha256": manifest_hash,
        "current_manifest_sha256": manifest_hash,
        "source_audit_manifest_sha256": archived_manifest_hash,
        "packet_schema_sha256": schema_hash,
        "decision_audit_sha256": sha256_file(decision_audit_path),
        "source_audit_run": str(audit_run_dir),
        "mode": mode,
        "scenarios": scenarios,
        "batch_size": batch_size,
        "module_sha256": sha256_file(Path(__file__)),
    }
    repository = manifest_path.resolve().parent.parent
    for name, command in [
        ("git_commit", ["git", "rev-parse", "HEAD"]),
        ("working_tree_status", ["git", "status", "--short"]),
    ]:
        result = subprocess.run(command, cwd=repository, capture_output=True, text=True, check=False)
        config[name] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    shutil.copyfile(manifest_path, drive_run_dir / manifest_path.name)
    shutil.copyfile(packet_schema_path, drive_run_dir / packet_schema_path.name)
    write_json(drive_run_dir / "run_config.json", config)
    results = {}
    for scenario in scenarios:
        work = Path(tempfile.mkdtemp(prefix=f"{scenario}_", dir=local_root))
        print(f"Preparing {scenario}. Local workspace: {work}", flush=True)
        try:
            output = work / "prepared"
            report = prepare_scenario(
                packet_paths[scenario], output, schema, scenario,
                batch_size=batch_size,
                source_sha256=audit_reports[scenario]["audit_parquet_sha256"],
            )
            expected_counts = audit_reports[scenario]["counts"]
            for key in ("packets", "normal_packets", "attack_packets"):
                if report["counts"][key] != expected_counts[key]:
                    raise ValueError(f"Prepared count mismatch for {scenario}/{key}.")
            destination = drive_run_dir / scenario
            destination.mkdir()
            checksums = {}
            for artifact in sorted(output.iterdir()):
                copied = destination / artifact.name
                shutil.copyfile(artifact, copied)
                checksum = sha256_file(artifact)
                if sha256_file(copied) != checksum:
                    raise OSError(f"Drive artifact checksum mismatch: {copied}")
                checksums[artifact.name] = checksum
            write_json(destination / "artifact_checksums.json", checksums)
            results[scenario] = {
                "status": report["status"],
                "report": str(destination / "preparation_report.json"),
            }
            write_json(drive_run_dir / "run_status.json", {
                "complete": False, "mode": mode, "scenarios": results,
            })
            shutil.rmtree(work)
            print(f"Saved and verified {scenario} on Drive. Removed its local copy.", flush=True)
        except Exception as error:
            write_json(drive_run_dir / "failure.json", {
                "scenario": scenario, "error": str(error), "local_workspace": str(work),
            })
            raise
    write_json(drive_run_dir / "run_status.json", {
        "complete": True,
        "mode": mode,
        "scenarios": results,
        "next_action": "Review canonical counts, timestamp ranges, node roles, and feature nulls.",
    })
    return results
