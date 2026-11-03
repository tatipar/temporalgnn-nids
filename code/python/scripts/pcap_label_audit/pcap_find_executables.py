"""Find packets in classic pcap files that carry a Windows executable (PE) signature; prints endpoints and payload start."""
import struct, sys, socket, datetime
SIG = b"This program cannot be run in DOS mode"
for path in sys.argv[1:]:
    with open(path, 'rb') as fh:
        gh = fh.read(24); e = '<' if gh[:4] == b'\xd4\xc3\xb2\xa1' else '>'
        link = struct.unpack(e+'I', gh[20:24])[0]
        while True:
            h = fh.read(16)
            if len(h) < 16: break
            ts, _, incl, _ = struct.unpack(e+'IIII', h); d = fh.read(incl)
            if SIG not in d: continue
            off = 14 if link == 1 else 0
            ip = d[off:]; ihl = (ip[0] & 15) * 4; proto = ip[9]
            s, t = socket.inet_ntoa(ip[12:16]), socket.inet_ntoa(ip[16:20])
            sp, dp = struct.unpack('>HH', ip[ihl:ihl+4])
            pl = ip[ihl + ((ip[ihl+12] >> 4) * 4):] if proto == 6 else b''
            print(path.split('/')[-1], datetime.datetime.utcfromtimestamp(ts).strftime('%H:%M:%S'),
                  f"{s}:{sp} -> {t}:{dp}", 'proto', proto, '| payload start:', pl[:60])
