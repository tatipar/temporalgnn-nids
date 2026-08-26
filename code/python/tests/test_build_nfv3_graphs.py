"""Tests for the streaming NF-v3 graph builder."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from scripts.build_nfv3_graphs import (
    audit_output, build_artifact_summary, collection_checksum_summary,
    iter_complete_windows, register_window_endpoints, save_day_checkpoint,
    save_profile_artifacts, summarize_split_class_coverage,
)
from utils.graph_construction import (
    DAY1, DaySpec, IpIdMap, atomic_json_dump, atomic_torch_save, build_graph,
    prepare_chunk,
)
from utils.graph_schema import NFV3_EXTENDED, PORTABLE_CORE


class CompleteWindowIteratorTests(unittest.TestCase):
    COLUMNS = [
        "source_file",
        "source_row_id",
        "FLOW_START_MILLISECONDS",
        "FLOW_END_MILLISECONDS",
        "FLOW_DURATION_MILLISECONDS",
        "IPV4_SRC_ADDR",
        "IPV4_DST_ADDR",
    ]

    def test_varied_durations_remain_ordered_across_chunk_boundaries(self) -> None:
        # The third flow in chunk 1 ends much later than the first flow in
        # chunk 2. Buffering only the maximum decision time would emit the
        # 930000 window twice and fail at that chunk boundary.
        frame = pd.DataFrame({
            "source_file": [DAY1.source_file] * 9,
            "source_row_id": list(range(9)),
            "FLOW_START_MILLISECONDS": [
                0, 900_000, 910_000,
                920_000, 930_000, 940_000,
                950_000, 960_000, 970_000,
            ],
            "FLOW_DURATION_MILLISECONDS": [
                0, 0, 120_000,
                0, 0, 0,
                0, 0, 0,
            ],
            "FLOW_END_MILLISECONDS": [
                0, 900_000, 1_030_000,
                920_000, 930_000, 940_000,
                950_000, 960_000, 970_000,
            ],
            "IPV4_SRC_ADDR": ["192.0.2.1"] * 9,
            "IPV4_DST_ADDR": ["192.0.2.2"] * 9,
        })

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ordered.csv"
            frame.to_csv(path, index=False)
            windows = list(iter_complete_windows([path], DAY1, self.COLUMNS, chunksize=3))

        decision_times = [decision_time for decision_time, _ in windows]
        source_rows = [
            int(source_row_id)
            for _, group in windows
            for source_row_id in group["source_row_id"]
        ]
        self.assertEqual(decision_times, sorted(set(decision_times)))
        self.assertEqual(sorted(source_rows), list(range(9)))
        self.assertEqual(len(source_rows), len(set(source_rows)))

    def test_rejects_flow_start_regression_between_chunks(self) -> None:
        frame = pd.DataFrame({
            "source_file": [DAY1.source_file] * 4,
            "source_row_id": list(range(4)),
            "FLOW_START_MILLISECONDS": [0, 60_000, 30_000, 90_000],
            "FLOW_END_MILLISECONDS": [0, 60_000, 30_000, 90_000],
            "FLOW_DURATION_MILLISECONDS": [0, 0, 0, 0],
            "IPV4_SRC_ADDR": ["192.0.2.1"] * 4,
            "IPV4_DST_ADDR": ["192.0.2.2"] * 4,
        })

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unordered.csv"
            frame.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "FLOW_START_MILLISECONDS"):
                list(iter_complete_windows([path], DAY1, self.COLUMNS, chunksize=2))

    def test_empty_windows_are_not_emitted(self) -> None:
        frame = pd.DataFrame({
            "source_file": [DAY1.source_file] * 2,
            "source_row_id": [0, 1],
            "FLOW_START_MILLISECONDS": [0, 120_000],
            "FLOW_END_MILLISECONDS": [0, 120_000],
            "FLOW_DURATION_MILLISECONDS": [0, 0],
            "IPV4_SRC_ADDR": ["192.0.2.1", "192.0.2.1"],
            "IPV4_DST_ADDR": ["192.0.2.2", "192.0.2.2"],
        })

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gapped.csv"
            frame.to_csv(path, index=False)
            windows = list(iter_complete_windows([path], DAY1, self.COLUMNS, chunksize=1))

        self.assertEqual([decision_time for decision_time, _ in windows], [30_000, 150_000])
        self.assertEqual([len(group) for _, group in windows], [1, 1])


class ResumeMappingTests(unittest.TestCase):
    @staticmethod
    def group(sources: list[str], destinations: list[str]) -> pd.DataFrame:
        return pd.DataFrame({"source_ip": sources, "destination_ip": destinations})

    def test_resume_replay_matches_uninterrupted_mapping(self) -> None:
        windows = [
            (30_000, self.group(["192.0.2.1", "192.0.2.3"], ["192.0.2.2", "192.0.2.4"])),
            (60_000, self.group(["192.0.2.5"], ["192.0.2.1"])),
            (90_000, self.group(["192.0.2.6"], ["192.0.2.3"])),
        ]

        uninterrupted = IpIdMap()
        for _, group in windows:
            register_window_endpoints(uninterrupted, group)

        resumed = IpIdMap()
        last_completed = 60_000
        for decision_time, group in windows:
            if decision_time <= last_completed:
                register_window_endpoints(resumed, group)
                continue
            # Building the remaining graph performs the same registration.
            register_window_endpoints(resumed, group)

        self.assertEqual(resumed.ip_to_id, uninterrupted.ip_to_id)
        self.assertEqual(resumed.id_to_ip, uninterrupted.id_to_ip)
        self.assertEqual(resumed.ip_to_id["192.0.2.6"], 5)

    def test_replay_uses_sources_then_destinations_like_graph_building(self) -> None:
        mapping = IpIdMap()
        register_window_endpoints(
            mapping,
            self.group(["192.0.2.10", "192.0.2.11"], ["192.0.2.12", "192.0.2.10"]),
        )
        self.assertEqual(mapping.ip_to_id, {
            "192.0.2.10": 0,
            "192.0.2.11": 1,
            "192.0.2.12": 2,
        })

    def test_checkpoint_publishes_mapping_before_state(self) -> None:
        mapping = IpIdMap({"192.0.2.1": 0})
        state = {
            "days": {DAY1.name: {"last_completed_decision_time_ms": 30_000}},
            "completed": False,
        }
        map_path = Path("mapping.json")
        state_path = Path("state.json")

        with patch("scripts.build_nfv3_graphs.atomic_json_dump") as dump:
            save_day_checkpoint(mapping, map_path, DAY1, state, state_path)

        self.assertEqual(dump.call_count, 2)
        self.assertEqual(dump.call_args_list[0].args[1], map_path)
        self.assertEqual(dump.call_args_list[1].args, (state, state_path))


class OutputAuditTests(unittest.TestCase):
    def test_full_audit_requires_both_binary_classes(self) -> None:
        with self.assertRaisesRegex(AssertionError, "must contain both binary classes"):
            summarize_split_class_coverage(
                {"graphs": 2, "edges": 5, "negative_edges": 5, "positive_edges": 0},
                required=True,
                partial=False,
                split_label="portable_core/day1/test1",
            )

        partial = summarize_split_class_coverage(
            {"graphs": 2, "edges": 5, "negative_edges": 5, "positive_edges": 0},
            required=True,
            partial=True,
            split_label="portable_core/day1/test1",
        )
        self.assertEqual(partial["class_coverage_status"], "partial")

    def test_collection_digest_is_order_independent_and_path_sensitive(self) -> None:
        first = {
            "b.pt": {"sha256": "b" * 64, "bytes": 20},
            "a.pt": {"sha256": "a" * 64, "bytes": 10},
        }
        reordered = {"a.pt": first["a.pt"], "b.pt": first["b.pt"]}
        renamed = {"c.pt": first["a.pt"], "b.pt": first["b.pt"]}

        summary = collection_checksum_summary(first)

        self.assertEqual(summary, collection_checksum_summary(reordered))
        self.assertNotEqual(summary["sha256"], collection_checksum_summary(renamed)["sha256"])
        self.assertEqual(summary["files"], 2)
        self.assertEqual(summary["bytes"], 30)

    def test_aligned_profiles_read_provenance_only_once(self) -> None:
        frame = pd.DataFrame({column: [1.0] for column in NFV3_EXTENDED.numeric_columns})
        frame["source_file"] = "day.csv"
        frame["source_row_id"] = 7
        frame["FLOW_START_MILLISECONDS"] = 29_998
        frame["FLOW_END_MILLISECONDS"] = 29_999
        frame["FLOW_DURATION_MILLISECONDS"] = 1
        frame["IPV4_SRC_ADDR"] = "192.0.2.1"
        frame["IPV4_DST_ADDR"] = "192.0.2.2"
        frame["L4_DST_PORT"] = 443
        frame["PROTOCOL"] = 6
        frame["TCP_FLAGS"] = 18
        frame["binary_target"] = 1
        prepared, _ = prepare_chunk(frame)

        profiles = (NFV3_EXTENDED, PORTABLE_CORE)
        scalers = {}
        for profile in profiles:
            numeric = prepared.loc[:, profile.numeric_columns].to_numpy(dtype=np.float64)
            scalers[profile.name] = StandardScaler().fit(np.log1p(numeric))

        day = DaySpec("day1", "day.csv", ("train",))
        mapping = IpIdMap()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_profile_artifacts(root, profiles, scalers)
            provenance = None
            for profile in profiles:
                graph, provenance = build_graph(
                    prepared, profile, scalers[profile.name], mapping, "train",
                )
                atomic_torch_save(
                    graph,
                    root / profile.name / "train" / "graph_0000000030000.pt",
                )
            provenance_path = root / "provenance" / day.name / "graph_0000000030000.csv"
            provenance_path.parent.mkdir(parents=True)
            provenance.to_csv(provenance_path, index=False)
            atomic_json_dump(
                mapping.payload(day.name),
                root / "mappings" / f"{day.name}_ip_to_id.json",
            )
            preflight = {
                "input_rows": 1,
                "positive_rows": 1,
                "retained_rows": 1,
                "retained_positive_rows": 1,
                "excluded_rows": 0,
                "excluded_positive_rows": 0,
                "by_source_file": {
                    day.source_file: {"retained_rows": 1, "retained_positive_rows": 1},
                },
            }
            real_read_csv = pd.read_csv
            artifact_checksums = {
                "algorithm": "sha256",
                "graphs": {profile.name: {} for profile in profiles},
                "provenance": {},
            }
            with patch(
                "utils.graph_construction.pd.read_csv", side_effect=real_read_csv,
            ) as read_csv:
                audit = audit_output(
                    root, profiles, (day,), preflight, partial=False,
                    artifact_checksums=artifact_checksums,
                )
            checksums_path = root / "artifact_checksums.json"
            atomic_json_dump(artifact_checksums, checksums_path)
            artifact_summary = build_artifact_summary(
                root, profiles, (day,), checksums_path, artifact_checksums,
            )
            orphan = root / "provenance" / day.name / "graph_0000000060000.csv"
            orphan.write_text("orphan\n", encoding="utf-8")
            with self.assertRaisesRegex(AssertionError, "exactly match"):
                audit_output(root, profiles, (day,), preflight, partial=False)

        self.assertEqual(read_csv.call_count, 1)
        self.assertEqual(artifact_summary["provenance_collection"]["files"], 1)
        self.assertEqual(len(artifact_summary["checksum_index"]["sha256"]), 64)
        for profile in profiles:
            self.assertEqual(artifact_summary["graph_collections"][profile.name]["files"], 1)
            self.assertEqual(len(artifact_summary["scalers"][profile.name]["sha256"]), 64)
            self.assertEqual(
                audit["profiles"][profile.name]["conservation_by_day"][day.name]["status"],
                "passed",
            )


if __name__ == "__main__":
    unittest.main()
