"""Deduplicate NF-v3 day CSVs and add original/corrected targets (see utils/nfv3_dedup.py).

Usage: python3 code/python/scripts/prepare_corrected_inputs.py SOURCE.csv DESTINATION.csv [...pairs]
Each destination gets a JSON sidecar with counts and SHA-256 hashes.
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from utils.nfv3_dedup import prepare_csv  # noqa: E402

args = sys.argv[1:]
if not args or len(args) % 2:
    sys.exit(__doc__)
for source, destination in zip(args[::2], args[1::2]):
    print(json.dumps(prepare_csv(source, destination), indent=2), flush=True)
