"""Deterministic class-weight calibration for frozen NF-v3 training graphs.

This module deliberately keeps the mathematical and serialization helpers free
of PyTorch imports.  Only :func:`calibrate_graph_collection` loads production
graphs, which keeps the formulas easy to test in lightweight environments.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence


REQUIRED_PROFILES = ("nfv3_extended", "portable_core")
CANDIDATE_DECLARATION = ("1", "2", "sqrt(R)", "R/2", "R")
DEFAULT_REL_TOL = 1e-12
DEFAULT_ABS_TOL = 1e-12
TARGET_DIGEST_CONTRACT = (
    "sha256 of repeated ASCII '<timestamp_ms>:<target_count>:' prefixes, "
    "followed by one byte per binary target in edge order and a newline"
)


def sha256_file(path: str | Path, block_size: int = 1024 * 1024) -> str:
    """Return the streaming SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def output_bias(pos_weight: float, negative_flows: int, positive_flows: int) -> float:
    """Return the weighted-BCE intercept matching the training prevalence."""
    weight = float(pos_weight)
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("pos_weight must be a positive finite number.")
    if not isinstance(negative_flows, int) or not isinstance(positive_flows, int):
        raise TypeError("Class counts must be integers.")
    if negative_flows <= 0 or positive_flows <= 0:
        raise ValueError("Both negative and positive training classes must be non-empty.")
    return math.log(weight * positive_flows / negative_flows)


def candidate_weight_bias_pairs(
    negative_flows: int,
    positive_flows: int,
    *,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> list[dict[str, Any]]:
    """Generate and near-deduplicate the predeclared weight/bias grid."""
    if not isinstance(negative_flows, int) or not isinstance(positive_flows, int):
        raise TypeError("Class counts must be integers.")
    if negative_flows <= 0 or positive_flows <= 0:
        raise ValueError("Both negative and positive training classes must be non-empty.")
    if rel_tol < 0 or abs_tol < 0 or not math.isfinite(rel_tol + abs_tol):
        raise ValueError("Candidate deduplication tolerances must be finite and non-negative.")

    ratio = negative_flows / positive_flows
    declared = (
        ("1", 1.0),
        ("2", 2.0),
        ("sqrt(R)", math.sqrt(ratio)),
        ("R/2", ratio / 2.0),
        ("R", ratio),
    )
    unique: list[dict[str, Any]] = []
    for anchor, weight in declared:
        duplicate = next(
            (
                item
                for item in unique
                if math.isclose(
                    weight,
                    item["pos_weight"],
                    rel_tol=rel_tol,
                    abs_tol=abs_tol,
                )
            ),
            None,
        )
        if duplicate is not None:
            duplicate["anchors"].append(anchor)
            continue
        unique.append({"anchors": [anchor], "pos_weight": float(weight)})

    for index, item in enumerate(unique, start=1):
        item["candidate_id"] = f"calibration_{index:02d}"
        item["output_bias_init"] = output_bias(
            item["pos_weight"], negative_flows, positive_flows
        )
    return unique


def _target_values(values: Any, *, profile: str, graph_index: int) -> list[int]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    else:
        try:
            values = list(values)
        except TypeError as error:
            raise ValueError(
                f"Targets for {profile} graph {graph_index} are not one-dimensional."
            ) from error
    if not values:
        raise ValueError(f"Targets for {profile} graph {graph_index} are empty.")

    binary: list[int] = []
    for edge_index, value in enumerate(values):
        if not isinstance(value, numbers.Real):
            raise ValueError(
                f"Invalid target for {profile} graph {graph_index}, edge {edge_index}: {value!r}."
            )
        numeric = float(value)
        if not math.isfinite(numeric) or numeric not in (0.0, 1.0):
            raise ValueError(
                f"Invalid target for {profile} graph {graph_index}, edge {edge_index}: {value!r}."
            )
        binary.append(int(numeric))
    return binary


def collect_aligned_training_targets(
    datasets: Mapping[str, Any],
) -> tuple[dict[str, int | float], dict[str, Any]]:
    """Count train labels while proving all supplied profiles are aligned."""
    if len(datasets) < 2:
        raise ValueError("Calibration requires at least two aligned feature profiles.")
    profiles = sorted(datasets)
    reference_profile = profiles[0]
    reference = datasets[reference_profile]

    for profile in profiles:
        dataset = datasets[profile]
        if getattr(dataset, "split", None) != "train":
            raise ValueError(f"Calibration may read only the train split; got {profile!r}.")
        if getattr(dataset, "profile", profile) != profile:
            raise ValueError(f"Dataset/profile mismatch for {profile!r}.")
        if not hasattr(dataset, "timestamps"):
            raise ValueError(f"Dataset {profile!r} does not expose graph timestamps.")
        if list(dataset.timestamps) != list(reference.timestamps):
            raise ValueError(
                f"Feature profiles {reference_profile!r} and {profile!r} have different train windows."
            )
        if len(dataset) != len(reference):
            raise ValueError(
                f"Feature profiles {reference_profile!r} and {profile!r} have different graph counts."
            )

    digests = {profile: hashlib.sha256() for profile in profiles}
    flow_counts = {profile: 0 for profile in profiles}
    negative_flows = 0
    positive_flows = 0

    for graph_index, timestamp in enumerate(reference.timestamps):
        targets_by_profile: dict[str, list[int]] = {}
        for profile in profiles:
            graph = datasets[profile][graph_index]
            if not hasattr(graph, "y"):
                raise ValueError(f"Graph {graph_index} for {profile!r} has no targets.")
            targets = _target_values(graph.y, profile=profile, graph_index=graph_index)
            targets_by_profile[profile] = targets
            flow_counts[profile] += len(targets)
            digests[profile].update(f"{int(timestamp)}:{len(targets)}:".encode("ascii"))
            digests[profile].update(bytes(targets))
            digests[profile].update(b"\n")

        reference_targets = targets_by_profile[reference_profile]
        for profile in profiles[1:]:
            if targets_by_profile[profile] != reference_targets:
                raise ValueError(
                    "Aligned feature profiles disagree on targets at "
                    f"train graph {graph_index} (timestamp {timestamp}): "
                    f"{reference_profile!r} != {profile!r}."
                )
        positive_flows += sum(reference_targets)
        negative_flows += len(reference_targets) - sum(reference_targets)

    if negative_flows <= 0 or positive_flows <= 0:
        raise ValueError("Both negative and positive training classes must be non-empty.")
    total_flows = negative_flows + positive_flows
    counts: dict[str, int | float] = {
        "total_flows": total_flows,
        "negative_flows": negative_flows,
        "positive_flows": positive_flows,
        "positive_prevalence": positive_flows / total_flows,
        "class_ratio_negative_to_positive": negative_flows / positive_flows,
    }
    profile_records = {
        profile: {
            "graphs": len(datasets[profile]),
            "flows": flow_counts[profile],
            "target_sha256": digests[profile].hexdigest(),
        }
        for profile in profiles
    }
    target_hashes = {record["target_sha256"] for record in profile_records.values()}
    if len(target_hashes) != 1:
        raise AssertionError("Equal targets unexpectedly produced different target hashes.")
    alignment = {
        "status": "passed",
        "split": "train",
        "target_digest_contract": TARGET_DIGEST_CONTRACT,
        "profiles": profile_records,
    }
    return counts, alignment


def build_calibration_manifest(
    *,
    graph_manifest_sha256: str,
    feature_schemas: Mapping[str, Mapping[str, Any]],
    corrected_manifest_sha256: str,
    corrected_data_sha256: str,
    correction_rule_version: str,
    code_revision: str,
    counts: Mapping[str, int | float],
    profile_alignment: Mapping[str, Any],
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> dict[str, Any]:
    """Build the complete deterministic Phase-4A manifest payload."""
    negative_flows = int(counts["negative_flows"])
    positive_flows = int(counts["positive_flows"])
    candidates = candidate_weight_bias_pairs(
        negative_flows,
        positive_flows,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
    if not correction_rule_version or not code_revision:
        raise ValueError("Correction version and code revision must be non-empty.")
    return {
        "format_version": 1,
        "artifact_type": "nfv3_training_class_calibration",
        "scope": {
            "split": "train",
            "profiles": sorted(feature_schemas),
            "label_access_policy": "training_split_only",
        },
        "source_artifacts": {
            "graph_manifest_sha256": graph_manifest_sha256,
            "feature_schemas": dict(feature_schemas),
            "corrected_manifest_sha256": corrected_manifest_sha256,
            "corrected_data_sha256": corrected_data_sha256,
            "correction_rule_version": correction_rule_version,
        },
        "code_revision": code_revision,
        "counts": dict(counts),
        "profile_alignment": dict(profile_alignment),
        "candidate_grid": {
            "declaration": list(CANDIDATE_DECLARATION),
            "class_ratio_symbol": "R = negative_flows / positive_flows",
            "output_bias_formula": (
                "log(pos_weight * positive_flows / negative_flows)"
            ),
            "equivalent_prevalence_formula": (
                "log(pos_weight * positive_prevalence / (1 - positive_prevalence))"
            ),
            "deduplication": {
                "method": "math.isclose",
                "relative_tolerance": rel_tol,
                "absolute_tolerance": abs_tol,
                "order": "first_declared_candidate_wins; all equivalent anchors retained",
            },
        },
        "candidates": candidates,
    }


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Serialize a manifest deterministically, rejecting non-finite floats."""
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def write_calibration_manifest(
    payload: Mapping[str, Any], output_path: str | Path, *, overwrite: bool = False
) -> str:
    """Atomically write a manifest, accepting an identical existing artifact."""
    path = Path(output_path)
    serialized = canonical_json_bytes(payload)
    if path.exists():
        if path.read_bytes() == serialized:
            return "unchanged"
        if not overwrite:
            raise FileExistsError(
                f"Calibration manifest differs from existing artifact: {path}. "
                "Use --overwrite only for a deliberate replacement."
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(serialized)
    os.replace(temporary, path)
    return "written"


def git_code_revision(repository_root: str | Path) -> str:
    """Return the exact Git commit, with ``-dirty`` for tracked modifications."""
    root = Path(repository_root).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "status",
                    "--porcelain",
                    "--untracked-files=no",
                ],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("Could not determine the calibration code revision.") from error
    if not commit:
        raise RuntimeError("Git returned an empty calibration code revision.")
    return f"{commit}-dirty" if dirty else commit


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {label}: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"Expected {label} to contain a JSON object: {path}")
    return payload


def _corrected_manifest_metadata(
    graph_root: Path,
    graph_manifest: Mapping[str, Any],
    override_path: str | Path | None,
) -> dict[str, str]:
    try:
        artifact = graph_manifest["corrected_manifest"]
        expected_sha256 = str(artifact["sha256"])
        declared_path = Path(artifact["path"])
    except (KeyError, TypeError) as error:
        raise ValueError("Graph manifest does not declare the corrected-data manifest.") from error
    path = Path(override_path) if override_path is not None else declared_path
    if not path.is_absolute():
        path = graph_root / path
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Corrected-data manifest not found: {path}. Supply --corrected-manifest after moving artifacts."
        )
    if sha256_file(path) != expected_sha256:
        raise ValueError("Corrected-data manifest hash does not match the graph manifest.")
    corrected = _load_json_object(path, "corrected-data manifest")
    try:
        return {
            "manifest_sha256": expected_sha256,
            "corrected_data_sha256": str(corrected["output"]["sha256"]),
            "correction_rule_version": str(corrected["correction_rule_version"]),
        }
    except (KeyError, TypeError) as error:
        raise ValueError("Corrected-data manifest lacks required calibration metadata.") from error


def calibrate_graph_collection(
    graph_root: str | Path,
    *,
    profiles: Sequence[str] = REQUIRED_PROFILES,
    corrected_manifest_path: str | Path | None = None,
    code_revision: str,
    verify_checksums: bool = False,
    rel_tol: float = DEFAULT_REL_TOL,
    abs_tol: float = DEFAULT_ABS_TOL,
) -> dict[str, Any]:
    """Read frozen train targets and produce the complete Phase-4A payload."""
    requested_profiles = tuple(profiles)
    if len(requested_profiles) != len(set(requested_profiles)):
        raise ValueError("Feature profiles must not be repeated.")
    if set(requested_profiles) != set(REQUIRED_PROFILES):
        raise ValueError(
            "Phase 4A requires exactly the aligned profiles: "
            + ", ".join(REQUIRED_PROFILES)
            + "."
        )
    root = Path(graph_root).expanduser().resolve()
    graph_manifest_path = root / "graph_manifest.json"
    graph_manifest = _load_json_object(graph_manifest_path, "graph manifest")
    if graph_manifest.get("status") != "passed":
        raise ValueError("Calibration requires a graph manifest with status='passed'.")

    # Imported lazily so formula/serialization tests do not require PyTorch.
    from .datasets import NF_IDS_Dataset

    datasets = {
        profile: NF_IDS_Dataset(
            graph_root=root,
            profile=profile,
            split="train",
            verify_checksums=verify_checksums,
        )
        for profile in sorted(requested_profiles)
    }
    counts, alignment = collect_aligned_training_targets(datasets)

    feature_schemas: dict[str, dict[str, Any]] = {}
    for profile, dataset in datasets.items():
        try:
            declared_file_hash = graph_manifest["artifacts"]["feature_schemas"][profile][
                "sha256"
            ]
            graph_collection_hash = graph_manifest["artifacts"]["graph_collections"][
                profile
            ]["sha256"]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Graph manifest lacks required artifact hashes for {profile!r}."
            ) from error
        actual_file_hash = sha256_file(dataset.schema_path)
        if actual_file_hash != declared_file_hash:
            raise ValueError(f"Feature-schema file hash mismatch for {profile!r}.")
        feature_schemas[profile] = {
            "schema_definition_sha256": dataset.expected_schema_hash,
            "schema_file_sha256": actual_file_hash,
            "graph_collection_sha256": str(graph_collection_hash),
            "edge_dim": int(dataset.edge_dim),
        }

    corrected = _corrected_manifest_metadata(
        root, graph_manifest, corrected_manifest_path
    )
    return build_calibration_manifest(
        graph_manifest_sha256=sha256_file(graph_manifest_path),
        feature_schemas=feature_schemas,
        corrected_manifest_sha256=corrected["manifest_sha256"],
        corrected_data_sha256=corrected["corrected_data_sha256"],
        correction_rule_version=corrected["correction_rule_version"],
        code_revision=code_revision,
        counts=counts,
        profile_alignment=alignment,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )
