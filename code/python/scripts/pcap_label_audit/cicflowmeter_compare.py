"""Compare where CIC's own CICFlowMeter CSV (Thursday 01-03) puts Infiltration labels in time with the
documented attack span, next to NF-v3.

CIC's Timestamp column is a 12-hour clock without AM/PM in local time (UTC-4). The capture runs
from about 08:00 to 17:00 local, so hours 1-7 are read as PM. Local time is converted to UTC.

Usage: python3 -I cicflowmeter_compare.py   (run nfv3_label_audit.py thu first for the NF-v3 comparison)
Writes cicflowmeter_thu_summary.json and cicflowmeter_thu_label_rate_by_hour.csv to
<CICIDS_DATA>/analysis/label_audit/thu/.
"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_config import DATA, DAYS, out_dir  # noqa: E402

UTC_OFFSET_H = 4
OUT = out_dir('thu')
c = pd.read_csv(os.path.join(DATA, 'cicflowmeter', 'thu_0103_cicflowmeter.csv'),
                usecols=['Dst Port', 'Timestamp', 'Label'], low_memory=False)
c = c[c.Label != 'Label']                       # the CSV repeats its header inside the file
c['Dst Port'] = c['Dst Port'].astype(int)
clock = pd.to_datetime(c.Timestamp, format='%d/%m/%Y %H:%M:%S')
pm = clock.dt.hour < 8
c['ts_utc'] = clock + pd.to_timedelta(pm * 12 + UTC_OFFSET_H, unit='h')
c['lab'] = c.Label.eq('Infilteration')
lo, hi = (pd.to_datetime(t, unit='s') for t in DAYS['thu']['attack_span'])
a = c[c.lab]
S = dict(rows=len(c), labeled=int(c.lab.sum()), labeled_share=round(float(c.lab.mean()), 3),
         capture_utc=[str(c.ts_utc.min()), str(c.ts_utc.max())],
         labeled_utc=[str(a.ts_utc.min()), str(a.ts_utc.max())],
         documented_attack_span_utc=[str(lo), str(hi)],
         labeled_outside_attack_span_share=round(float((~a.ts_utc.between(lo, hi)).mean()), 4),
         top_labeled_dst_ports=a['Dst Port'].value_counts().head(10).to_dict())
nf = os.path.join(OUT, 'summary.json')
if os.path.exists(nf):
    S['nfv3_unconfirmed_labels_outside_attack_span_share'] = json.load(open(nf))['unconfirmed_labels_outside_attack_span_share']
c.groupby(c.ts_utc.dt.hour).lab.agg(flows='size', label_rate='mean').round(3).to_csv(
    os.path.join(OUT, 'cicflowmeter_thu_label_rate_by_hour.csv'))
json.dump(S, open(os.path.join(OUT, 'cicflowmeter_thu_summary.json'), 'w'), indent=2)
print(json.dumps(S, indent=2))
print(c.groupby(c.ts_utc.dt.hour).lab.agg(flows='size', label_rate='mean').round(3).T.to_string())
