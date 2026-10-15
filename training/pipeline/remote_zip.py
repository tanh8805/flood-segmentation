"""Seekable read-only file over HTTP Range, so zipfile can extract single members of a huge remote zip."""
import io, urllib.request, time


class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, block=1 << 20):
        self.url, self.block, self.pos, self.cache = url, block, 0, {}
        req = urllib.request.Request(url, headers={"Range": "bytes=0-0", "User-Agent": "curl/8.9.1"})
        with urllib.request.urlopen(req, timeout=60) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])

    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _fetch(self, a, b):  # inclusive
        for attempt in range(5):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}", "User-Agent": "curl/8.9.1"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    return r.read()
            except Exception:
                time.sleep(2 * (attempt + 1))
        raise IOError(f"range fetch failed {a}-{b}")

    def read(self, n=-1):
        if n < 0 or self.pos + n > self.size:
            n = self.size - self.pos
        if n <= 0:
            return b""
        out, end = [], self.pos + n
        # large reads: fetch directly; small: use block cache
        if n > self.block:
            data = self._fetch(self.pos, end - 1)
            self.pos = end
            return data
        while self.pos < end:
            bi = self.pos // self.block
            if bi not in self.cache:
                if len(self.cache) > 64: self.cache.clear()
                self.cache[bi] = self._fetch(bi * self.block, min((bi + 1) * self.block, self.size) - 1)
            blk = self.cache[bi]
            off = self.pos - bi * self.block
            take = blk[off: off + (end - self.pos)]
            out.append(take); self.pos += len(take)
        return b"".join(out)

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)
