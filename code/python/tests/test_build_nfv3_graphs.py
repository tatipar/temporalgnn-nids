"""Tests for the streaming NF-v3 graph builder."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from scripts.build_nfv3_graphs import (
    iter_complete_windows, register_window_endpoints, save_day_checkpoint,
)
from utils.graph_construction import DAY1, IpIdMap


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


if __name__ == "__main__":
    unittest.main()
