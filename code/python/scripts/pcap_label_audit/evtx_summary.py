"""Summarize Windows .evtx files: event IDs, time range, logon failures by source IP, process creations."""
import collections, re, sys
import Evtx.Evtx as evtx
for path in sys.argv[1:]:
    ids = collections.Counter(); prov = collections.Counter(); fails = collections.Counter(); procs = collections.Counter()
    tmin = tmax = None
    with evtx.Evtx(path) as log:
        for rec in log.records():
            x = rec.xml()
            eid = re.search(r'<EventID[^>]*>(\d+)</EventID>', x); eid = eid and eid[1]
            t = re.search(r'SystemTime="([^"]+)"', x)[1][:19]
            tmin = min(tmin or t, t); tmax = max(tmax or t, t)
            p = re.search(r'Provider Name="([^"]+)"', x)[1]
            ids[(p, eid)] += 1
            if eid in ('4625', '4624'):
                ip = re.search(r'<Data Name="IpAddress">([^<]*)<', x); user = re.search(r'<Data Name="TargetUserName">([^<]*)<', x)
                fails[(eid, ip and ip[1], user and user[1])] += 1
            if eid in ('4688', '7045', '4104', '1'):
                img = re.search(r'<Data Name="(?:NewProcessName|ImagePath|ServiceName|ScriptBlockText|Image)">([^<]{0,120})', x)
                procs[(eid, img and img[1])] += 1
    print(f"== {path.split('/')[-1]}  {tmin} .. {tmax}  records={sum(ids.values())}")
    print("  top events:", ids.most_common(12))
    print("  logons (eid, ip, user):", fails.most_common(10))
    print("  proc/service:", procs.most_common(15))
