"""Stream an adsb.lol day archive straight from GitHub, keeping only the traces that touch a
bounding box. Nothing is written to disk but the result, so a 4 GB day costs no local space."""
from __future__ import annotations

import io
import re
import tarfile
import time
from typing import Callable, Iterable, Iterator

import requests

from .adsb import Trace, in_bbox, parse_trace_bytes

TRACE_RE = re.compile(r"trace_full_([0-9a-f~]+)\.json$")
API = "https://api.github.com/repos/adsblol/globe_history_{year}/releases/tags/{tag}"


def release_asset_urls(date: str, pod: str = "planes-readsb-prod-0") -> list[str]:
    """Download URLs of the (possibly split) tar for one UTC day, date = 'YYYY-MM-DD'."""
    y, m, d = date.split("-")
    tag = f"v{y}.{m}.{d}-{pod}"
    r = requests.get(API.format(year=y, tag=tag), timeout=60)
    r.raise_for_status()
    assets = [a for a in r.json()["assets"] if ".tar" in a["name"]]
    return [a["browser_download_url"] for a in sorted(assets, key=lambda a: a["name"])]


class _ChainedHTTP(io.RawIOBase):
    """A read-only stream over several URLs in sequence, resuming with Range on dropped connections."""

    def __init__(self, urls: list[str], retries: int = 20, chunk: int = 1 << 20):
        self.urls, self.retries, self.chunk = urls, retries, chunk
        self.i, self.pos, self.resp, self.buf = 0, 0, None, b""
        self.total = 0

    def readable(self):
        return True

    def _open(self):
        for k in range(self.retries):
            try:
                h = {"Range": f"bytes={self.pos}-"} if self.pos else {}
                r = requests.get(self.urls[self.i], headers=h, stream=True, timeout=120)
                if r.status_code in (200, 206):
                    if self.pos and r.status_code == 200:
                        raise IOError("server ignored Range; cannot resume")
                    self.resp = r.iter_content(self.chunk)
                    return
                r.close()
            except (requests.RequestException, IOError) as e:
                if k == self.retries - 1:
                    raise
            time.sleep(min(60, 2 ** k))

    def readinto(self, b):
        while not self.buf:
            if self.i >= len(self.urls):
                return 0
            if self.resp is None:
                self._open()
            try:
                self.buf = next(self.resp)
                self.pos += len(self.buf); self.total += len(self.buf)
            except StopIteration:
                self.i += 1; self.pos = 0; self.resp = None
            except requests.RequestException:
                self.resp = None          # reopen with Range at self.pos
        n = min(len(b), len(self.buf))
        b[:n] = self.buf[:n]; self.buf = self.buf[n:]
        return n


def iter_tar_traces(fileobj, bbox, progress: Callable[[int, int], None] | None = None) -> Iterator[Trace]:
    """Yield traces touching bbox from a tar stream (readsb day archive layout traces/xx/trace_full_*.json)."""
    n_seen = 0
    with tarfile.open(fileobj=fileobj, mode="r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            mm = TRACE_RE.search(m.name)
            if not mm:
                continue
            n_seen += 1
            raw = tf.extractfile(m).read()
            try:
                tr = parse_trace_bytes(raw, mm.group(1))
            except Exception:
                continue
            if tr is not None and in_bbox(tr, bbox):
                yield tr
            if progress and n_seen % 2000 == 0:
                progress(n_seen, getattr(fileobj, "total", 0))


def fetch_day(date: str, bbox, pod: str = "planes-readsb-prod-0",
              progress: Callable[[int, int], None] | None = None) -> list[Trace]:
    urls = release_asset_urls(date, pod)
    stream = _ChainedHTTP(urls)
    prog = (lambda n, _: progress(n, stream.total)) if progress else None
    return list(iter_tar_traces(io.BufferedReader(stream, 1 << 22), bbox, prog))
