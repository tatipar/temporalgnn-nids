"""Minimal pure-Python reader for classic pcap files with Ethernet framing (no third-party parsers)."""
import struct

MAGIC = {b'\xd4\xc3\xb2\xa1': ('<', 1e-6), b'\xa1\xb2\xc3\xd4': ('>', 1e-6),
         b'\x4d\x3c\xb2\xa1': ('<', 1e-9), b'\xa1\xb2\x3c\x4d': ('>', 1e-9)}


def packets(path):
    """Yield (timestamp_seconds, raw_frame) for each record."""
    with open(path, 'rb') as fh:
        gh = fh.read(24)
        endian, unit = MAGIC[gh[:4]]
        linktype = struct.unpack(endian + 'I', gh[20:24])[0]
        if linktype != 1:
            raise ValueError(f'{path}: link type {linktype} is not Ethernet')
        while True:
            h = fh.read(16)
            if len(h) < 16:
                return
            sec, frac, incl, _ = struct.unpack(endian + 'IIII', h)
            yield sec + frac * unit, fh.read(incl)


def ipv4(frame):
    """Parse an Ethernet/IPv4 frame. Returns None for non-IPv4 frames, else a dict with
    src_mac, dst_mac, src, dst, proto, sport, dport, tcp_flags (TCP only) and l4 payload offset."""
    off = 14
    ethertype = frame[12:14]
    if ethertype == b'\x81\x00':            # 802.1Q VLAN tag
        ethertype = frame[16:18]; off = 18
    if ethertype != b'\x08\x00' or len(frame) < off + 20:
        return None
    ihl = (frame[off] & 0x0F) * 4
    l4 = off + ihl
    p = dict(dst_mac=frame[0:6].hex(':'), src_mac=frame[6:12].hex(':'),
             src='.'.join(map(str, frame[off + 12:off + 16])), dst='.'.join(map(str, frame[off + 16:off + 20])),
             proto=frame[off + 9], sport=None, dport=None, tcp_flags=None, payload=None)
    if p['proto'] in (6, 17) and len(frame) >= l4 + 4:
        p['sport'], p['dport'] = struct.unpack('>HH', frame[l4:l4 + 4])
        if p['proto'] == 6 and len(frame) >= l4 + 20:
            p['tcp_flags'] = frame[l4 + 13]
            p['payload'] = l4 + (frame[l4 + 12] >> 4) * 4
        elif p['proto'] == 17:
            p['payload'] = l4 + 8
    return p


SYN, ACK = 0x02, 0x10
