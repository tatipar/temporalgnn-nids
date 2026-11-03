"""Extract selected members of a remote CSE-CIC-IDS2018 pcap.zip via HTTP range requests."""
import os, shutil, sys, urllib.parse, urllib.request, zipfile
class RangeFile:
    CHUNK = 16 << 20
    def __init__(s, url):
        s.url = url; s.pos = 0; s.buf_start = 0; s.buf = b""
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"))
        s.size = int(r.headers["Content-Length"])
    def seekable(s): return True
    def tell(s): return s.pos
    def seek(s, off, wh=0):
        s.pos = off if wh == 0 else s.pos + off if wh == 1 else s.size + off; return s.pos
    def read(s, n=-1):
        if n < 0: n = s.size - s.pos
        out = b""
        while n > 0 and s.pos < s.size:
            if not (s.buf_start <= s.pos < s.buf_start + len(s.buf)):
                end = min(s.pos + max(n, s.CHUNK), s.size) - 1
                req = urllib.request.Request(s.url, headers={"Range": f"bytes={s.pos}-{end}"})
                s.buf = urllib.request.urlopen(req).read(); s.buf_start = s.pos
            i = s.pos - s.buf_start; piece = s.buf[i:i + n]
            out += piece; s.pos += len(piece); n -= len(piece)
        return out
