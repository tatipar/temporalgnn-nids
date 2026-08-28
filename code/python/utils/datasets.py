"""Manifest-aware datasets for versioned NF-v3 graph collections."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch
from torch.utils.data import Dataset


GRAPH_FILE_PATTERN = re.compile(r"^graph_(\d+)\.pt$")
VALID_SPLITS = frozenset({"train", "val", "test1", "test2"})
REQUIRED_GRAPH_FIELDS = frozenset({
    "edge_index",
    "edge_attr",
    "y",
    "global_node_ids",
    "timestamp",
    "window_start",
    "window_end",
    "feature_profile",
    "schema_hash",
})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _scalar_int(value: Any, field_name: str) -> int:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Graph field {field_name!r} must contain exactly one value.")
        return int(value.item())
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Graph field {field_name!r} must be an integer scalar.") from error


class NF_IDS_Dataset(Dataset):
    """Load one audited feature-profile/split graph collection.

    Parameters
    ----------
    graph_root:
        Root of one completed graph version. It must contain
        ``graph_manifest.json`` and ``<profile>/<split>/graph_*.pt``.
    profile:
        Feature profile declared by the graph manifest, for example
        ``nfv3_extended`` or ``portable_core``.
    split:
        One of ``train``, ``val``, ``test1``, or ``test2``.
    verify_checksums:
        If true, verify the feature schema at initialization and each graph on
        first access against ``artifact_checksums.json``. This is useful after
        copying artifacts from Drive, but can be expensive on remote storage.

    Notes
    -----
    The production graph schema intentionally has no ``x`` node-feature
    tensor. Models must derive node identity from edge attributes or construct
    their documented constant initial node state internally.
    """

    def __init__(
        self,
        graph_root: str | Path,
        profile: str,
        split: str,
        *,
        verify_checksums: bool = False,
    ) -> None:
        self.graph_root = Path(graph_root).expanduser().resolve()
        self.profile = profile
        self.split = split
        self.verify_checksums = verify_checksums
        self._verified_graphs: set[Path] = set()

        if split not in VALID_SPLITS:
            choices = ", ".join(sorted(VALID_SPLITS))
            raise ValueError(f"Unknown split {split!r}. Expected one of: {choices}.")

        self.manifest_path = self.graph_root / "graph_manifest.json"
        self.manifest = self._load_json(self.manifest_path, "graph manifest")
        if self.manifest.get("status") != "passed":
            raise ValueError(
                f"Graph manifest is not publishable: expected status='passed' in {self.manifest_path}."
            )
        try:
            self.window_ms = int(self.manifest["window_ms"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Graph manifest must declare an integer window_ms.") from error
        if self.window_ms <= 0:
            raise ValueError("Graph manifest window_ms must be positive.")

        manifest_profiles = self.manifest.get("profiles", {})
        if profile not in manifest_profiles:
            available = ", ".join(sorted(manifest_profiles)) or "none"
            raise ValueError(
                f"Feature profile {profile!r} is absent from the graph manifest. Available: {available}."
            )
        self.expected_schema_hash = str(manifest_profiles[profile])

        self.profile_root = self.graph_root / profile
        self.split_dir = self.profile_root / split
        self.schema_path = self.profile_root / "feature_schema.json"
        self.schema = self._load_json(self.schema_path, "feature schema")
        self.edge_dim = self._validate_schema()
        self._validate_schema_artifact()

        if not self.split_dir.is_dir():
            raise FileNotFoundError(f"Graph split directory not found: {self.split_dir}")

        timestamped_files: list[tuple[int, Path]] = []
        for path in self.split_dir.glob("*.pt"):
            match = GRAPH_FILE_PATTERN.fullmatch(path.name)
            if match is None:
                raise ValueError(f"Unexpected graph filename in {self.split_dir}: {path.name}")
            timestamped_files.append((int(match.group(1)), path))
        timestamped_files.sort(key=lambda item: item[0])

        self.timestamps = [timestamp for timestamp, _ in timestamped_files]
        self.files = [path for _, path in timestamped_files]
        if not self.files:
            raise ValueError(f"Graph split contains no graph files: {self.split_dir}")
        if any(current <= previous for previous, current in zip(self.timestamps, self.timestamps[1:])):
            raise ValueError(f"Graph timestamps are not strictly increasing in {self.split_dir}.")

        self._validate_manifest_graph_count()
        self._checksum_index = self._load_checksum_index() if verify_checksums else {}

    @staticmethod
    def _load_json(path: Path, label: str) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(f"Required {label} not found: {path}")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read {label}: {path}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"Expected {label} to contain a JSON object: {path}")
        return payload

    def _validate_schema(self) -> int:
        stored_hash = self.schema.get("sha256")
        schema_payload = {key: value for key, value in self.schema.items() if key != "sha256"}
        computed_hash = _canonical_sha256(schema_payload)
        if stored_hash != computed_hash:
            raise ValueError(f"Feature schema self-hash is invalid: {self.schema_path}")
        if stored_hash != self.expected_schema_hash:
            raise ValueError(
                f"Feature schema hash does not match the graph manifest for {self.profile!r}."
            )
        if self.schema.get("name") != self.profile:
            raise ValueError(f"Feature schema name does not match profile {self.profile!r}.")

        dimension = self.schema.get("dimension")
        columns = self.schema.get("edge_attr_columns")
        if not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("Feature schema dimension must be a positive integer.")
        if not isinstance(columns, list) or len(columns) != dimension:
            raise ValueError("Feature schema columns do not agree with its declared dimension.")
        if len(columns) != len(set(columns)):
            raise ValueError("Feature schema contains duplicate edge-attribute columns.")
        return dimension

    def _validate_schema_artifact(self) -> None:
        artifact = (
            self.manifest.get("artifacts", {})
            .get("feature_schemas", {})
            .get(self.profile)
        )
        if not isinstance(artifact, dict):
            raise ValueError(f"Graph manifest does not declare the schema artifact for {self.profile!r}.")
        expected_path = self.schema_path.relative_to(self.graph_root).as_posix()
        if artifact.get("path") != expected_path:
            raise ValueError(f"Feature schema path does not match the graph manifest: {expected_path}")
        if int(artifact.get("bytes", -1)) != self.schema_path.stat().st_size:
            raise ValueError(f"Feature schema byte size does not match the graph manifest: {self.schema_path}")
        if self.verify_checksums and artifact.get("sha256") != _sha256_file(self.schema_path):
            raise ValueError(f"Feature schema file checksum does not match the graph manifest: {self.schema_path}")

    def _validate_manifest_graph_count(self) -> None:
        try:
            expected_count = int(
                self.manifest["audit"]["profiles"][self.profile]["splits"][self.split]["graphs"]
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"Graph manifest does not declare a graph count for {self.profile}/{self.split}."
            ) from error
        if expected_count != len(self.files):
            raise ValueError(
                f"Graph count mismatch for {self.profile}/{self.split}: "
                f"manifest={expected_count}, files={len(self.files)}."
            )

    def _load_checksum_index(self) -> dict[str, dict[str, Any]]:
        checksum_path = self.graph_root / "artifact_checksums.json"
        payload = self._load_json(checksum_path, "artifact checksum index")
        try:
            graph_checksums = payload["graphs"][self.profile]
        except (KeyError, TypeError) as error:
            raise ValueError(
                f"Artifact checksum index does not contain profile {self.profile!r}."
            ) from error
        if not isinstance(graph_checksums, dict):
            raise ValueError(f"Invalid graph checksum collection for profile {self.profile!r}.")
        return graph_checksums

    def _verify_graph_checksum(self, path: Path) -> None:
        if not self.verify_checksums or path in self._verified_graphs:
            return
        relative_path = path.relative_to(self.graph_root).as_posix()
        artifact = self._checksum_index.get(relative_path)
        if not isinstance(artifact, dict):
            raise ValueError(f"Graph is absent from the artifact checksum index: {relative_path}")
        if int(artifact.get("bytes", -1)) != path.stat().st_size:
            raise ValueError(f"Graph byte size does not match the checksum index: {relative_path}")
        if artifact.get("sha256") != _sha256_file(path):
            raise ValueError(f"Graph checksum does not match the checksum index: {relative_path}")
        self._verified_graphs.add(path)

    def _validate_graph(self, data: Any, expected_timestamp: int, path: Path) -> None:
        missing = sorted(field for field in REQUIRED_GRAPH_FIELDS if not hasattr(data, field))
        if missing:
            raise ValueError(f"Graph is missing required fields {missing}: {path}")
        if getattr(data, "x", None) is not None:
            raise ValueError(f"Production graphs must not persist node features in x: {path}")

        if data.edge_index.ndim != 2 or int(data.edge_index.shape[0]) != 2:
            raise ValueError(f"edge_index must have shape [2, num_edges]: {path}")
        if data.edge_index.dtype != torch.long:
            raise ValueError(f"edge_index must use torch.long indices: {path}")
        num_edges = int(data.edge_index.shape[1])
        if num_edges <= 0:
            raise ValueError(f"Production graph files must not be empty: {path}")
        if data.edge_attr.ndim != 2 or tuple(data.edge_attr.shape) != (num_edges, self.edge_dim):
            raise ValueError(
                f"edge_attr shape does not match {self.profile} dimension {self.edge_dim}: {path}"
            )
        if data.y.ndim != 1 or int(data.y.shape[0]) != num_edges:
            raise ValueError(f"y must contain one binary target per edge: {path}")
        if not torch.isfinite(data.edge_attr).all() or not torch.isfinite(data.y).all():
            raise ValueError(f"Graph contains non-finite edge attributes or targets: {path}")
        if not torch.all((data.y == 0) | (data.y == 1)):
            raise ValueError(f"Graph targets must be binary: {path}")

        num_nodes = int(data.num_nodes)
        if num_nodes <= 0 or data.global_node_ids.ndim != 1:
            raise ValueError(f"Graph must contain a non-empty one-dimensional global node map: {path}")
        if int(data.global_node_ids.numel()) != num_nodes:
            raise ValueError(f"global_node_ids must contain one ID per local node: {path}")
        if data.global_node_ids.dtype != torch.long:
            raise ValueError(f"global_node_ids must use torch.long IDs: {path}")
        if int(torch.unique(data.global_node_ids).numel()) != num_nodes:
            raise ValueError(f"global_node_ids must be unique within each graph: {path}")
        if data.edge_index.numel() and (
            int(data.edge_index.min()) < 0 or int(data.edge_index.max()) >= num_nodes
        ):
            raise ValueError(f"edge_index references a local node outside the graph: {path}")

        timestamp = _scalar_int(data.timestamp, "timestamp")
        window_start = _scalar_int(data.window_start, "window_start")
        window_end = _scalar_int(data.window_end, "window_end")
        if timestamp != expected_timestamp or window_end != expected_timestamp:
            raise ValueError(f"Graph timestamp metadata does not match its filename: {path}")
        if window_start >= window_end:
            raise ValueError(f"Graph window_start must be earlier than window_end: {path}")
        if window_end - window_start != self.window_ms:
            raise ValueError(f"Graph window duration does not match manifest window_ms: {path}")
        if data.feature_profile != self.profile or data.schema_hash != self.expected_schema_hash:
            raise ValueError(f"Graph profile/schema metadata does not match its collection: {path}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        path = self.files[index]
        self._verify_graph_checksum(path)
        data = torch.load(path, weights_only=False)
        self._validate_graph(data, self.timestamps[index], path)
        return data

    def validate_all(self) -> None:
        """Eagerly load and validate every graph in chronological order."""
        previous_timestamp: int | None = None
        for index in range(len(self)):
            data = self[index]
            timestamp = _scalar_int(data.timestamp, "timestamp")
            if previous_timestamp is not None and timestamp <= previous_timestamp:
                raise ValueError(f"Loaded graph timestamps are not strictly increasing in {self.split_dir}.")
            previous_timestamp = timestamp
