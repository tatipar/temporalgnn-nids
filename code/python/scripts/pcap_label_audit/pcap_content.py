"""Extract DNS query names and TLS ClientHello SNI per source host from classic pcap files (pure Python)."""
import struct, sys, collections, json

def packets(path):
    with open(path, 'rb') as fh:
        gh = fh.read(24)
        endian = '<' if gh[:4] == b'\xd4\xc3\xb2\xa1' else '>'
        while True:
            h = fh.read(16)
            if len(h) < 16: return
            ts, _, incl, _ = struct.unpack(endian + 'IIII', h)
            yield ts, fh.read(incl)

def parse_name(buf, off):
    labels = []
    while off < len(buf):
        n = buf[off]
        if n == 0: break
        if n & 0xC0: break
        labels.append(buf[off + 1:off + 1 + n].decode('ascii', 'replace')); off += n + 1
    return '.'.join(labels).lower()

def sni(payload):
    try:
        if payload[0] != 0x16 or payload[5] != 0x01: return None
        p = 9 + 2 + 32
        p += 1 + payload[p]                       # session id
        p += 2 + struct.unpack('>H', payload[p:p + 2])[0]  # cipher suites
        p += 1 + payload[p]                       # compression
        end = p + 2 + struct.unpack('>H', payload[p:p + 2])[0]; p += 2
        while p + 4 <= end:
            t, ln = struct.unpack('>HH', payload[p:p + 4]); p += 4
            if t == 0: return payload[p + 5:p + 5 + struct.unpack('>H', payload[p + 3:p + 5])[0]].decode('ascii', 'replace').lower()
            p += ln
    except (IndexError, struct.error):
        return None

out = {}
for path in sys.argv[1:]:
    dns, tls = collections.Counter(), collections.Counter()
    for ts, pk in packets(path):
        if len(pk) < 34 or pk[12:14] != b'\x08\x00': continue
        ihl = (pk[14] & 0x0F) * 4; proto = pk[23]; ip = pk[14:]; l4 = 14 + ihl
        src = '.'.join(map(str, pk[26:30])); dst = '.'.join(map(str, pk[30:34]))
        if proto == 17 and len(pk) >= l4 + 8:
            dport = struct.unpack('>H', pk[l4 + 2:l4 + 4])[0]
            if dport == 53 and dst == '172.31.0.2':
                d = pk[l4 + 8:]
                if len(d) > 12 and not (d[2] & 0x80): dns[(src, parse_name(d, 12))] += 1
        elif proto == 6 and len(pk) >= l4 + 20:
            dport = struct.unpack('>H', pk[l4 + 2:l4 + 4])[0]
            if dport == 443:
                pl = pk[l4 + (pk[l4 + 12] >> 4) * 4:]
                if len(pl) > 43:
                    s = sni(pl)
                    if s: tls[(src, dst, s)] += 1
    out[path] = dict(dns=[[*k, v] for k, v in dns.items()], tls=[[*k, v] for k, v in tls.items()])
json.dump(out, sys.stdout)
