"""Prepare an NF-v3 day CSV for the corrected-label retraining: remove duplicate rows and add
both binary targets.

NF-v3 repeats some flows as identical rows (the packet captures contain no matching duplicate
packets). A duplicate is a row with the same start time and 5-tuple as an earlier row; the first
row of each group is kept, so file order (chronological) is preserved. Copies never disagree on the
label, and only a few dozen to a few hundred groups differ in any feature value.

Added columns:
- ``target_original``: 1 when ``Attack == 'Infilteration'`` (the rule used by the historical builder);
- ``target_corrected``: the Liu et al. corrected binary target from ``nfv3_relabel.relabel_chunk``;
- ``correction_rule``: the rule that produced the corrected target (audit only).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from utils.nfv3_relabel import RULE_VERSION, relabel_chunk

DEDUP_KEY = ['FLOW_START_MILLISECONDS', 'IPV4_SRC_ADDR', 'L4_SRC_PORT',
             'IPV4_DST_ADDR', 'L4_DST_PORT', 'PROTOCOL']
DEDUP_POLICY = 'drop_later_rows_with_same_start_ms_and_5tuple_keep_first_v1'


def sha256_file(path, block=8 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(block), b''):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Return the deduplicated frame with both targets, plus summary statistics."""
    rows_in = len(frame)
    duplicated = frame.duplicated(DEDUP_KEY, keep='first')
    groups = frame[frame.duplicated(DEDUP_KEY, keep=False)].groupby(DEDUP_KEY)
    label_conflicts = int((groups.Attack.nunique() > 1).sum()) if len(groups) else 0
    if label_conflicts:
        raise ValueError(f'{label_conflicts} duplicate groups disagree on the original label')
    out = frame.loc[~duplicated].copy()
    relabeled = relabel_chunk(out)
    out['target_original'] = out['Attack'].eq('Infilteration').astype('int8')
    out['target_corrected'] = relabeled['binary_target'].astype('int8')
    out['correction_rule'] = relabeled['correction_rule']
    stats = dict(
        dedup_policy=DEDUP_POLICY, dedup_key=DEDUP_KEY, relabel_rule_version=RULE_VERSION,
        rows_in=rows_in, rows_out=len(out), duplicate_rows_removed=int(duplicated.sum()),
        duplicate_groups=int(groups.ngroups) if len(groups) else 0,
        positives_original_in=int(frame['Attack'].eq('Infilteration').sum()),
        positives_original_out=int(out.target_original.sum()),
        positives_corrected_out=int(out.target_corrected.sum()),
        corrected_rules=out.loc[out.target_corrected.eq(1), 'correction_rule'].value_counts().to_dict(),
        start_time_monotonic=bool(pd.to_datetime(out['FLOW_START_TIME']).is_monotonic_increasing),
    )
    return out, stats


def prepare_csv(source: str | Path, destination: str | Path, hash_files: bool = True) -> dict:
    """Read a full NF-v3 day CSV, deduplicate, add targets, write CSV and a JSON sidecar."""
    source, destination = Path(source), Path(destination)
    out, stats = prepare_frame(pd.read_csv(source))
    if not stats['start_time_monotonic']:
        raise ValueError('FLOW_START_TIME is not monotonic; the chunked graph builder requires sorted rows')
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + '.tmp')
    out.to_csv(tmp, index=False)
    tmp.replace(destination)
    stats.update(source=str(source), destination=str(destination))
    if hash_files:
        stats.update(source_sha256=sha256_file(source), destination_sha256=sha256_file(destination))
    sidecar = destination.with_suffix('.json')
    sidecar.write_text(json.dumps(stats, indent=2))
    return stats
