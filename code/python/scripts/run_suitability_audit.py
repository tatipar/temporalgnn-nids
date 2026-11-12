"""Run the training-free suitability audit locally and write one JSON per dataset plus summary tables.

Example (the run behind ~/data/dataset_suitability_audit_v1, 2026-11-12):

    ~/data/cic-ids2018/.venv/bin/python -I code/python/scripts/run_suitability_audit.py \
        --cic-wed ~/Downloads/cicids2018v3_wed2802.csv --cic-thu ~/Downloads/cicids2018v3_thu0103.csv \
        --dapt ~/data/dapt2020 --ait ait_nds --unraveled ~/data/unraveled_git/data/network-flows \
        --out ~/data/dataset_suitability_audit_v1

Datasets whose path is omitted are skipped; existing reports are reused unless --force is given.
DAPT2020 and AIT-NDS are downloaded into the given folder if missing. Unraveled must be fetched
beforehand (GitLab rate-limits raw downloads; a sparse git clone works):

    git clone --filter=blob:none --no-checkout --depth 1 https://gitlab.com/asu22/unraveled.git unraveled_git
    cd unraveled_git && git sparse-checkout set --no-cone 'data/network-flows/*/*_Flow_labeled.csv' && git checkout master

Peak memory is about 12.6 GB (Unraveled) and 10 GB (CIC).
"""
import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code/python'))
from utils import suitability_audit as sa  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--cic-wed')
parser.add_argument('--cic-thu')
parser.add_argument('--dapt')
parser.add_argument('--ait')
parser.add_argument('--unraveled')
parser.add_argument('--out', required=True)
parser.add_argument('--force', action='store_true')
args = parser.parse_args()

jobs = {}
if args.cic_wed and args.cic_thu:
    days = {'wed': args.cic_wed, 'thu': args.cic_thu}
    jobs['cic_original'] = lambda: sa.load_cic_nfv3(days, 'original')
    jobs['cic_corrected'] = lambda: sa.load_cic_nfv3(days, 'corrected')
if args.dapt:
    jobs['dapt2020'] = lambda: (sa.download_dapt2020(args.dapt), sa.load_dapt2020(args.dapt))[1]
if args.ait:
    jobs['ait'] = lambda: (sa.download_ait(args.ait), sa.load_ait(args.ait))[1]
if args.unraveled:
    jobs['unraveled'] = lambda: sa.load_unraveled(args.unraveled)

out = Path(args.out).expanduser()
revision = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], text=True).strip()
reports = {}
for key, load in jobs.items():
    path = out / f'{key}.json'
    if path.exists() and not args.force:
        reports[key] = json.loads(path.read_text())
        print(f'{key}: reusing {path}')
        continue
    report = sa.audit(load())
    report['code_revision'] = revision
    sa.save(report, out, name=key)
    reports[key] = report
    gc.collect()

# The tables cover every report in the output folder, including datasets not run this time.
ORDER = ['cic_original', 'cic_corrected', 'dapt2020', 'ait', 'unraveled']
for path in sorted(out.glob('*.json')):
    reports.setdefault(path.stem, json.loads(path.read_text()))
reports = dict(sorted(reports.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else len(ORDER)))
summary = pd.DataFrame([sa.summary_row(r) for r in reports.values()])
stages = pd.DataFrame([row for r in reports.values() for row in sa.stage_rows(r)])
summary.to_csv(out / 'summary.csv', index=False)
stages.to_csv(out / 'stages.csv', index=False)
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 40)
print(summary.set_index('dataset').T.to_string())
print(stages.to_string())
