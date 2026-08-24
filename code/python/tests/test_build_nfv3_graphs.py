"""Tests for the streaming NF-v3 graph builder."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.build_nfv3_graphs import iter_complete_windows
from utils.graph_construction import DAY1


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


if __name__ == "__main__":
    unittest.main()
