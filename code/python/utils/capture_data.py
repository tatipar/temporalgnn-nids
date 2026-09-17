"""CPU-only, disk-backed cAPTure Gate-0 auditing. No model fitting occurs here."""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def load_manifest(path: Path) -> dict:
    manifest = yaml.safe_load(Path(path).read_text())
    if manifest["dataset"] != "cAPTure" or manifest["manifest_version"] != 1:
        raise ValueError("Unsupported experiment manifest.")
    scenarios = manifest["scenarios"]
    development = {key for key, item in scenarios.items() if item["assignment"] == "development"}
    validations = []
    for fold in manifest["validation"]["folds"].values():
        train, validate = set(fold["train"]), set(fold["validate"])
        if train & validate or train | validate != development:
            raise ValueError("Invalid development fold assignments.")
        if {scenarios[key]["benign_source"] for key in train} & {
            scenarios[key]["benign_source"] for key in validate
        }:
            raise ValueError("Benign backgrounds overlap across a fold.")
        validations.extend(validate)
    if len(validations) != len(development) or set(validations) != development:
        raise ValueError("Each development scenario must be validated exactly once.")
    return manifest


def selected_scenarios(manifest: dict, mode: str) -> list[str]:
    if mode not in {"SMOKE", "FULL_DEV"}:
        raise ValueError("Gate 0 supports only SMOKE and FULL_DEV.")
    selected = manifest["gate0"]["modes"][mode]
    if len(set(selected)) != len(selected):
        raise ValueError("Repeated scenario in audit mode.")
    for scenario in selected:
        item = manifest["scenarios"][scenario]
        if item["assignment"] != "development" or item["author_split"] != "train":
            raise ValueError(f"Gate 0 cannot access scenario: {scenario}")
    return selected


@dataclass(frozen=True)
class AuditSchema:
    timestamp: str
    timestamp_unit: str  # s, ms, us, ns, or datetime
    label: str
    label_mapping: dict[str, int]
    source_endpoint: str
    destination_endpoint: str
    attack_step: str
    phase: str
    sequence_id: str
    separator: str = ","
    encoding: str = "utf-8-sig"

    def validate(self, columns: list[str]) -> None:
        required = [self.timestamp, self.label, self.source_endpoint,
                    self.destination_endpoint, self.attack_step, self.phase, self.sequence_id]
        missing = sorted(set(required) - set(columns))
        if missing:
            raise ValueError(f"Unresolved or missing schema columns: {missing}")
        if self.timestamp_unit not in {"s", "ms", "us", "ns", "datetime"}:
            raise ValueError("Timestamp unit must be explicitly declared.")
        if not self.label_mapping or set(self.label_mapping.values()) != {0, 1}:
            raise ValueError("Label mapping must explicitly include normal and attack values.")
        if any(not isinstance(key, str) for key in self.label_mapping):
            raise ValueError("Raw label mapping keys must be strings.")


def inspect_csv(path: Path, *, separator: str = ",", encoding: str = "utf-8-sig",
                sample_rows: int = 2000) -> dict:
    """Inspect a bounded prefix. This is not a full-scenario audit."""
    with Path(path).open(encoding=encoding, newline="") as stream:
        header = next(csv.reader(stream, delimiter=separator))
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate CSV column names: {duplicates}")
    if not header or any(not name.strip() for name in header):
        raise ValueError("Empty CSV header or column name.")
    frame = pd.read_csv(path, sep=separator, encoding=encoding, nrows=sample_rows,
                        dtype="string", keep_default_na=False)
    return {
        "filename": Path(path).name, "source_size_bytes": Path(path).stat().st_size,
        "sample_rows": len(frame), "scope": "prefix_only_not_representative",
        "columns": header,
        "sample_values": {column: frame[column].drop_duplicates().head(8).tolist()
                          for column in frame},
    }


def stage_source(source: dict, local_dir: Path) -> Path:
    """Stage one CSV, reusing only a complete, checksum-verified local download."""
    if source.get("metadata_verified") is not True:
        raise ValueError("Verify the source scenario, published filename, and location first.")
    name = source.get("expected_filename")
    if not name or Path(name).name != name or not name.lower().endswith(".csv"):
        raise ValueError("An explicit CSV basename is required.")
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    target = local_dir / name
    receipt_path = local_dir / (name + ".source.json")
    binding = {key: source.get(key) for key in (
        "drive_path", "source_file_id", "expected_filename", "expected_size_bytes", "sha256")}
    if target.exists():
        if not receipt_path.exists():
            raise FileExistsError(f"Unverified existing file; use a fresh staging directory: {target}")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("binding") != binding or sha256_file(target) != receipt.get("sha256"):
            raise ValueError(f"Cached source binding or checksum mismatch: {target}")
        print(f"Reusing verified local source: {target.name}", flush=True)
        return target
    partial = local_dir / (name + ".part")
    if partial.exists():
        raise FileExistsError(f"Incomplete previous download; use a fresh staging directory: {partial}")
    if source.get("drive_path"):
        original = Path(source["drive_path"])
        if original.name != name:
            raise ValueError("Mounted Drive filename does not match the verified filename.")
        required = original.stat().st_size + 2 * 1024**3
        if shutil.disk_usage(local_dir).free < required:
            raise OSError("Insufficient local disk space to stage the source with a reserve.")
        shutil.copyfile(original, partial)
    elif source.get("source_file_id"):
        size = source.get("expected_size_bytes")
        if not isinstance(size, int) or size <= 0:
            raise ValueError("Declare expected_size_bytes before downloading a source.")
        if shutil.disk_usage(local_dir).free < size + 2 * 1024**3:
            raise OSError("Insufficient local disk space for the declared download.")
        import gdown
        result = gdown.download(id=source["source_file_id"], output=str(partial), quiet=False)
        if result is None:
            raise RuntimeError("Source download failed.")
    else:
        raise ValueError("Provide either drive_path or a verified source_file_id.")
    if source.get("expected_size_bytes") is not None and partial.stat().st_size != source["expected_size_bytes"]:
        raise ValueError("Downloaded byte size differs from the verified source metadata.")
    source_hash = sha256_file(partial)
    if source.get("sha256") and source_hash != source["sha256"]:
        raise ValueError("Source checksum does not match.")
    inspect_csv(partial, separator=source.get("separator", ","),
                encoding=source.get("encoding", "utf-8-sig"), sample_rows=1)
    partial.rename(target)
    write_json(receipt_path, {"binding": binding, "sha256": source_hash})
    return target


def _identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def audit_scenario(source_path: Path, output_dir: Path, manifest: dict,
                   scenario: str, schema: AuditSchema, *, chunksize: int = 25000,
                   memory_limit: str = "2GB", threads: int = 2) -> dict:
    """Audit every row, preserving raw strings and canonical diagnostic metadata.

    The Parquet is an audit artifact, not a frozen model-ready feature table.
    SQL aggregation spills to local disk. No entire scenario is loaded into RAM.
    """
    if scenario not in selected_scenarios(manifest, "FULL_DEV"):
        raise ValueError("Only development scenario contents may be audited.")
    if chunksize <= 0 or threads <= 0:
        raise ValueError("Chunk size and thread count must be positive.")
    inspection = inspect_csv(source_path, separator=schema.separator, encoding=schema.encoding)
    schema.validate(inspection["columns"])
    widths = manifest["windows"]["candidate_durations_seconds"]
    if not widths or any(not isinstance(w, int) or w <= 0 for w in widths):
        raise ValueError("Candidate widths must be positive integer seconds.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    parquet_path = output_dir / "packets.audit.parquet"
    source_hash = sha256_file(source_path)
    profile = {name: {"missing": 0, "numeric_values": 0, "first_value": None,
                      "constant_nonmissing": True} for name in inspection["columns"]}
    row_count, inversions, previous = 0, 0, None
    writer = None
    try:
        for frame in pd.read_csv(source_path, sep=schema.separator, encoding=schema.encoding,
                                 dtype="string", keep_default_na=False, chunksize=chunksize):
            if shutil.disk_usage(output_dir).free < 512 * 1024**2:
                raise OSError("Local disk reserve exhausted during packet conversion.")
            raw = frame.copy()
            for column in frame:
                values = frame[column].str.strip().replace("", pd.NA)
                stats = profile[column]
                stats["missing"] += int(values.isna().sum())
                stats["numeric_values"] += int(pd.to_numeric(values, errors="coerce").notna().sum())
                present = values.dropna()
                if len(present):
                    if stats["first_value"] is None:
                        stats["first_value"] = str(present.iloc[0])
                    stats["constant_nonmissing"] &= bool(present.eq(stats["first_value"]).all())
                frame[column] = values
            timestamp_values = frame[schema.timestamp]
            if schema.timestamp_unit == "datetime":
                times = pd.to_datetime(timestamp_values, errors="coerce", utc=True, format="mixed")
            else:
                # Preserve integer epoch precision; fractional seconds are rounded by pandas.
                times = pd.to_datetime(pd.to_numeric(timestamp_values, errors="coerce"),
                                       unit=schema.timestamp_unit, errors="coerce", utc=True)
            valid_times = times.dropna()
            inversions += int((valid_times.diff().dt.total_seconds() < 0).sum())
            if len(valid_times):
                inversions += int(previous is not None and valid_times.iloc[0] < previous)
                previous = valid_times.iloc[-1]
            canonical = pd.DataFrame(index=frame.index)
            canonical["packet_id"] = [f"{source_hash}:{i}" for i in range(row_count, row_count + len(frame))]
            canonical["source_row_id"] = pd.Series(range(row_count, row_count + len(frame)), index=frame.index, dtype="int64")
            canonical["packet_timestamp"] = times
            timestamp_ns = times.dt.as_unit("ns").astype("int64").astype("Int64")
            canonical["packet_timestamp_ns"] = timestamp_ns.mask(times.isna())
            canonical["binary_label"] = frame[schema.label].map(schema.label_mapping).astype("Int8")
            for key, column in [("src_endpoint", schema.source_endpoint), ("dst_endpoint", schema.destination_endpoint),
                                ("attack_step", schema.attack_step), ("phase", schema.phase),
                                ("sequence_id", schema.sequence_id)]:
                canonical[key] = frame[column]
            for key in ["attack_chain", "benign_source", "author_split"]:
                canonical[key] = manifest["scenarios"][scenario][key]
            canonical["scenario"] = scenario
            # Length-delimited JSON preserves column boundaries and raw empty values.
            canonical["raw_row_sha256"] = [hashlib.sha256(json.dumps(row, ensure_ascii=False,
                separators=(",", ":")).encode()).hexdigest() for row in raw.itertuples(index=False, name=None)]
            for column in raw:
                canonical[f"raw::{column}"] = raw[column]
            arrays = {}
            for column in canonical:
                dtype = (pa.timestamp("ns", tz="UTC") if column == "packet_timestamp" else
                         pa.int64() if column in {"source_row_id", "packet_timestamp_ns"} else
                         pa.int8() if column == "binary_label" else pa.string())
                arrays[column] = pa.array(canonical[column], type=dtype, from_pandas=True)
            table = pa.table(arrays)
            if writer is None:
                writer = pq.ParquetWriter(parquet_path, table.schema, compression="zstd")
            writer.write_table(table)
            row_count += len(frame)
            print(f"{scenario}: converted {row_count:,} packets", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if not row_count:
        raise ValueError("The scenario contains no packets.")
    connection = duckdb.connect()
    try:
        connection.execute(f"SET memory_limit={_literal(memory_limit)}")
        connection.execute(f"SET threads={threads}")
        connection.execute(f"SET temp_directory={_literal(str(output_dir / 'spill'))}")
        connection.execute("SET preserve_insertion_order=false")
        connection.execute(f"CREATE VIEW packets AS SELECT * FROM read_parquet({_literal(str(parquet_path))})")

        def records(sql: str) -> list[dict]:
            cursor = connection.execute(sql)
            names = [item[0] for item in cursor.description]
            return [dict(zip(names, row)) for row in cursor.fetchall()]

        counts = records("""SELECT count(*) AS packets,
            count(*) FILTER (WHERE binary_label=0) AS normal_packets,
            count(*) FILTER (WHERE binary_label=1) AS attack_packets,
            count(*) FILTER (WHERE binary_label IS NULL) AS unmapped_labels,
            count(*) FILTER (WHERE packet_timestamp IS NULL) AS invalid_timestamps,
            count(*) FILTER (WHERE src_endpoint IS NULL OR dst_endpoint IS NULL) AS missing_endpoints,
            count(*) FILTER (WHERE binary_label=1 AND
                (attack_step IS NULL OR phase IS NULL OR sequence_id IS NULL)) AS incomplete_attack_annotations,
            epoch(min(packet_timestamp)) AS first_timestamp_seconds,
            epoch(max(packet_timestamp)) AS last_timestamp_seconds,
            count(*) - count(DISTINCT raw_row_sha256) AS duplicate_raw_rows_sha256
            FROM packets""")[0]
        endpoint_counts = records("""SELECT count(*) AS unique_endpoints FROM (
            SELECT src_endpoint AS endpoint FROM packets UNION SELECT dst_endpoint FROM packets
            ) WHERE endpoint IS NOT NULL""")[0]
        pair_counts = records("""SELECT count(*) AS unique_directed_pairs FROM (
            SELECT DISTINCT src_endpoint, dst_endpoint FROM packets
            WHERE src_endpoint IS NOT NULL AND dst_endpoint IS NOT NULL)""")[0]
        label_column = _identifier(f"raw::{schema.label}")
        raw_labels = records(f"SELECT {label_column} AS raw_label, count(*) AS packets FROM packets GROUP BY 1 ORDER BY 2 DESC")
        sequences_sql = """SELECT attack_step, phase, sequence_id, count(*) AS packets,
            epoch(min(packet_timestamp)) AS start_seconds, epoch(max(packet_timestamp)) AS end_seconds,
            epoch(max(packet_timestamp))-epoch(min(packet_timestamp)) AS duration_seconds
            FROM packets WHERE binary_label=1 GROUP BY 1,2,3"""
        connection.execute(f"COPY ({sequences_sql}) TO {_literal(str(output_dir / 'attack_iterations.parquet'))} (FORMAT PARQUET)")
        windows = []
        for width in widths:
            for offset in [0.0, width * 0.5]:
                # Unix epoch is only a diagnostic origin, not the selected experimental origin.
                # Integer arithmetic avoids rounding packets across window boundaries.
                width_ns, offset_ns = width * 1_000_000_000, int(offset * 1_000_000_000)
                relative = f"(packet_timestamp_ns - {offset_ns})"
                window_expression = (f"({relative} // {width_ns}) - CASE WHEN {relative} < 0 "
                                     f"AND {relative} % {width_ns} != 0 THEN 1 ELSE 0 END")
                connection.execute(f"""CREATE OR REPLACE TEMP VIEW window_packets AS
                    SELECT *, {window_expression} AS window_id
                    FROM packets WHERE packet_timestamp_ns IS NOT NULL""")
                sql = """WITH edges AS (
                        SELECT window_id, count(*) AS packets,
                        count(*) FILTER (WHERE binary_label=1) AS attack_packets
                        FROM window_packets GROUP BY 1),
                    endpoints AS (SELECT window_id, src_endpoint AS endpoint FROM window_packets
                        UNION SELECT window_id, dst_endpoint FROM window_packets),
                    nodes AS (SELECT window_id, count(*) AS nodes FROM endpoints
                        WHERE endpoint IS NOT NULL GROUP BY 1),
                    pairs AS (SELECT window_id, count(*) AS directed_pairs FROM (
                        SELECT DISTINCT window_id, src_endpoint, dst_endpoint FROM window_packets
                        WHERE src_endpoint IS NOT NULL AND dst_endpoint IS NOT NULL) GROUP BY 1)
                    SELECT edges.*, coalesce(nodes.nodes,0) AS nodes,
                        coalesce(pairs.directed_pairs,0) AS directed_pairs
                    FROM edges LEFT JOIN nodes USING(window_id) LEFT JOIN pairs USING(window_id)"""
                name = f"windows_{width}s_offset_{offset:g}s.parquet"
                connection.execute(f"COPY ({sql}) TO {_literal(str(output_dir / name))} (FORMAT PARQUET)")
                summary = records(f"""SELECT count(*) AS occupied_windows,
                    max(window_id)-min(window_id)+1-count(*) AS empty_windows_between_first_and_last,
                    avg(packets) AS mean_packets_occupied, max(packets) AS max_packets,
                    quantile_cont(packets, [0.5,0.95,0.99]) AS packet_quantiles_occupied,
                    max(nodes) AS max_nodes, quantile_cont(nodes, [0.5,0.95,0.99]) AS node_quantiles_occupied,
                    max(directed_pairs) AS max_directed_pairs,
                    quantile_cont(directed_pairs, [0.5,0.95,0.99]) AS pair_quantiles_occupied
                    FROM ({sql})""")[0]
                windows.append({"width_seconds": width, "origin_offset_seconds": offset,
                                "artifact": name, **summary})
        # Exact equality checks are restricted to columns with the same streaming digest.
        column_hashes = {column: hashlib.sha256() for column in inspection["columns"]}
        for batch in pq.ParquetFile(parquet_path).iter_batches(batch_size=chunksize,
                columns=[f"raw::{column}" for column in column_hashes]):
            for index, column in enumerate(column_hashes):
                for value in batch.column(index).to_pylist():
                    encoded = value.encode()
                    column_hashes[column].update(len(encoded).to_bytes(8, "big") + encoded)
        digest_groups = {}
        for column, digest in column_hashes.items():
            digest_groups.setdefault(digest.hexdigest(), []).append(column)
        duplicates = [group for group in digest_groups.values() if len(group) > 1]
        for column, stats in profile.items():
            stats["numeric_parse_failures_nonmissing"] = row_count - stats["missing"] - stats["numeric_values"]
            stats["interpretation"] = "Parse failures may be valid categorical values; manual schema review is required."
            stats.pop("first_value")
        suspicious_columns = [column for column in inspection["columns"] if re.search(
            r"label|phase|sequence|attack|scenario|timestamp|(^|[._])(?:ip|mac|id|time)([._]|$)",
            column, re.IGNORECASE)]
        blockers = [key for key in ["unmapped_labels", "invalid_timestamps", "missing_endpoints",
                                   "incomplete_attack_annotations"] if counts[key]]
        if inversions:
            blockers.append("timestamp_order_inversions")
        if not counts["normal_packets"] or not counts["attack_packets"]:
            blockers.append("missing_normal_or_attack_class")
        report = {
            "report_version": 1, "scenario": scenario, "status": "blocked" if blockers else "review_required",
            "blockers": blockers, "automatic_gate_pass": False,
            "schema": asdict(schema), "source_sha256": source_hash,
            "source_size_bytes": Path(source_path).stat().st_size,
            "audit_parquet_size_bytes": parquet_path.stat().st_size,
            "audit_parquet_sha256": sha256_file(parquet_path),
            "counts": {**counts, **endpoint_counts, **pair_counts},
            "timestamp_order_inversions": inversions, "raw_labels": raw_labels,
            "column_profiles": profile, "duplicate_column_groups_sha256": duplicates,
            "raw_storage_type": "string_preserving_source_values",
            "canonical_arrow_schema": str(pq.read_schema(parquet_path)),
            "identifier_or_annotation_name_candidates": suspicious_columns,
            "name_screen_is_not_a_feature_allowlist": True,
            "attack_step_phase_counts": records("""SELECT attack_step, phase,
                binary_label, count(*) AS packets FROM packets GROUP BY 1,2,3 ORDER BY 4 DESC"""),
            "windows": windows, "window_origin": "unix_epoch_diagnostic_only",
            "parquet_role": "audit_only_not_model_ready",
            "versions": {"python": platform.python_version(), "pandas": pd.__version__,
                         "pyarrow": pa.__version__, "duckdb": duckdb.__version__},
            "required_manual_review": ["endpoint_identity_and_non_ip_broadcast_policy",
                "sequence_iteration_semantics", "feature_leakage_and_identifier_columns",
                "duplicate_rows_and_columns", "timestamp_units_and_plausible_ranges",
                "storage_feasibility_and_window_selection"],
        }
        write_json(output_dir / "audit_report.json", report)
        return report
    finally:
        connection.close()


def validate_smoke_review(path: Path, manifest_hash: str, manifest: dict) -> None:
    """Require an explicit review tied to the exact manifest and smoke reports."""
    review = json.loads(Path(path).read_text())
    if review.get("manifest_sha256") != manifest_hash or review.get("approved") is not True:
        raise ValueError("FULL_DEV requires an approved smoke review for this manifest.")
    if not review.get("review_notes", "").strip():
        raise ValueError("Record the smoke review rationale.")
    for scenario in selected_scenarios(manifest, "SMOKE"):
        entry = review["reports"][scenario]
        report_path = Path(entry["path"])
        if sha256_file(report_path) != entry["sha256"]:
            raise ValueError("Smoke report checksum mismatch.")
        report = json.loads(report_path.read_text())
        if (report["scenario"] != scenario or report["blockers"]
                or report.get("manifest_sha256") != manifest_hash or report.get("mode") != "SMOKE"):
            raise ValueError("Smoke reports contain unresolved blockers.")


def run_gate0(*, manifest_path: Path, mode: str, sources: dict,
              schemas: dict[str, AuditSchema], local_root: Path, drive_run_dir: Path,
              smoke_review_path: Path | None = None, chunksize: int = 25000,
              memory_limit: str = "2GB", threads: int = 2,
              keep_audit_parquet: bool = True, source_cache_root: Path | None = None) -> dict:
    """Run scenarios sequentially and persist completed artifacts to mounted Drive.

    The caller chooses a fresh run directory. Local temporary files are removed
    only after their artifacts have been copied and checksum-verified on Drive.
    Failed work remains locally available until the Colab runtime is discarded.
    """
    manifest_path, local_root, drive_run_dir = map(Path, (manifest_path, local_root, drive_run_dir))
    manifest = load_manifest(manifest_path)
    manifest_hash = sha256_file(manifest_path)
    selected = selected_scenarios(manifest, mode)
    if mode == "FULL_DEV" and manifest["gate0"]["full_dev_requires_smoke_pass"]:
        if smoke_review_path is None:
            raise ValueError("FULL_DEV requires a reviewed SMOKE run.")
        validate_smoke_review(smoke_review_path, manifest_hash, manifest)
    for scenario in selected:
        if scenario not in sources or scenario not in schemas:
            raise ValueError(f"Missing source or schema configuration: {scenario}")
        source = sources[scenario]
        if source.get("metadata_verified") is not True:
            raise ValueError(f"Source metadata must be verified: {scenario}")
        for key in ("source_file_id", "expected_filename", "expected_size_bytes"):
            frozen = manifest["scenarios"][scenario].get(key)
            if frozen is not None and source.get(key) != frozen:
                raise ValueError(f"Source configuration disagrees with manifest: {scenario}/{key}")
    local_root.mkdir(parents=True, exist_ok=True)
    drive_run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(manifest_path, drive_run_dir / "experiment_manifest.yaml")
    config = {
        "manifest_sha256": manifest_hash, "mode": mode, "sources": sources,
        "schemas": {key: asdict(value) for key, value in schemas.items()},
        "chunksize": chunksize, "memory_limit": memory_limit, "threads": threads,
        "keep_audit_parquet": keep_audit_parquet,
        "module_sha256": sha256_file(Path(__file__)),
    }
    repository = manifest_path.resolve().parent.parent
    for name, command in [("git_commit", ["git", "rev-parse", "HEAD"]),
                          ("working_tree_status", ["git", "status", "--short"])]:
        result = subprocess.run(command, cwd=repository, capture_output=True, text=True, check=False)
        config[name] = result.stdout.strip() if result.returncode == 0 else "unavailable"
    write_json(drive_run_dir / "run_config.json", config)
    results = {}
    for scenario in selected:
        work = Path(tempfile.mkdtemp(prefix=f"{scenario}_", dir=local_root))
        print(f"Starting {scenario}. Local workspace: {work}", flush=True)
        try:
            source_dir = Path(source_cache_root) / scenario if source_cache_root is not None else work / "raw"
            source_path = stage_source(sources[scenario], source_dir)
            output = work / "audit"
            report = audit_scenario(source_path, output, manifest, scenario, schemas[scenario],
                                    chunksize=chunksize, memory_limit=memory_limit, threads=threads)
            report["manifest_sha256"] = manifest_hash
            report["mode"] = mode
            write_json(output / "audit_report.json", report)
            destination = drive_run_dir / scenario
            destination.mkdir()
            checksums = {}
            for artifact in sorted(output.iterdir()):
                if not artifact.is_file() or (artifact.name == "packets.audit.parquet" and not keep_audit_parquet):
                    continue
                copied = destination / artifact.name
                shutil.copyfile(artifact, copied)
                checksum = sha256_file(artifact)
                if sha256_file(copied) != checksum:
                    raise OSError(f"Drive artifact checksum mismatch: {copied}")
                checksums[artifact.name] = checksum
            write_json(destination / "artifact_checksums.json", checksums)
            results[scenario] = {"status": report["status"], "blockers": report["blockers"],
                                 "report": str(destination / "audit_report.json")}
            write_json(drive_run_dir / "run_status.json", {"complete": False, "scenarios": results})
            # Only this function's isolated, successfully persisted workspace is deleted.
            if source_cache_root is not None:
                source_path.unlink()
                source_path.with_name(source_path.name + ".source.json").unlink()
            shutil.rmtree(work)
            print(f"Saved and verified {scenario} on Drive. Removed its temporary local copy.", flush=True)
            if report["blockers"]:
                print("Stopping after data-integrity blockers. Review the saved report before continuing.")
                return results
        except Exception as error:
            write_json(drive_run_dir / "failure.json", {"scenario": scenario,
                       "error": str(error), "local_workspace": str(work)})
            raise
    write_json(drive_run_dir / "run_status.json", {"complete": True, "scenarios": results,
               "gate0_passed": False, "next_action": "Review scientific and schema decisions."})
    return results
