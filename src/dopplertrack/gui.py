"""dopplertrack-gui: pick a bounding box on a satellite map, drop a recording, geolocate it."""
from __future__ import annotations

import datetime as dt
import io
import queue
import tempfile
import threading
import tkinter as tk
import traceback
from concurrent.futures import ThreadPoolExecutor
import webbrowser
from pathlib import Path
from tkinter import filedialog, ttk

import numpy as np
import requests
import tkintermapview
from matplotlib import colormaps
from matplotlib.colors import to_hex
from PIL import Image, ImageTk
from tkinterdnd2 import DND_FILES, TkinterDnD

from . import pipeline
from .adsb import load_traces
from .fetch import release_asset_urls
from .search import Grid
from .terrain import Terrain
from .tracks import detect_events

TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
PAD_DEG = 0.12  # ~13 km: flights are heard from up to 12 km outside the box
CACHE = Path.home() / ".cache" / "dopplertrack"


def _fetch_tile(url):
    try:
        r = requests.get(url, headers={"User-Agent": "dopplertrack"}, timeout=15)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        return None


class Map(tkintermapview.TkinterMapView):
    """The stock widget loads tiles in threads that also create, hold and drop Tk objects, which
    segfaults Tk 9. Here the threads only download; every Tk object stays on the main thread."""

    def __init__(self, *args, **kw):
        self._pool = ThreadPoolExecutor(8)
        self._inflight = []  # (zoom, x, y, canvas tile, future)
        super().__init__(*args, **kw)

    def pre_cache(self):
        pass

    def load_images_background(self):
        pass

    def update_canvas_tile_images(self):
        while self.image_load_queue_tasks:
            (zoom, x, y), tile = self.image_load_queue_tasks.pop()
            self._inflight.append((zoom, x, y, tile, self._pool.submit(_fetch_tile, self.tile_server.format(x=x, y=y, z=zoom))))
        waiting = []
        for zoom, x, y, tile, fut in self._inflight:
            if not fut.done():
                waiting.append((zoom, x, y, tile, fut)); continue
            img = fut.result()
            if img is not None and zoom == round(self.zoom):
                img = self.tile_image_cache[f"{zoom}{x}{y}"] = ImageTk.PhotoImage(img)
                tile.set_image(img)
        self._inflight = waiting
        if self.running:
            self.after(10, self.update_canvas_tile_images)


def locate(path, date, bbox, spacing, log):
    """Recording + UTC date + (latmin, latmax, lonmin, lonmax) -> dict describing the best location."""
    la0, la1, lo0, lo1 = bbox
    CACHE.mkdir(parents=True, exist_ok=True)
    tr_path = CACHE / f"{date}_{la0:.3f}_{la1:.3f}_{lo0:.3f}_{lo1:.3f}.npz"
    if not tr_path.exists():  # fail fast, before the slow audio steps
        try:
            release_asset_urls(date)
        except requests.HTTPError:
            raise RuntimeError(f"no adsb.lol flight archive for {date}. Enter the UTC date the recording was made "
                               "(archives appear about a day later).") from None
    tmp = Path(tempfile.mkdtemp(prefix="dopplertrack-"))
    log("decoding audio, computing spectrogram")
    sp = pipeline.make_spectrogram(path, tmp / "spec.npz")
    log("detecting flyovers")
    evs = detect_events(sp, 60, 300, top=8, progress=lambda k, n: log(f"  window {k}/{n}"))
    if len(evs) < 2:
        raise RuntimeError("fewer than two Doppler events found in this recording")
    pad_lon = PAD_DEG / max(np.cos(np.radians((la0 + la1) / 2)), 0.1)
    if tr_path.exists():
        traces = load_traces(tr_path)
    else:  # streams a ~4 GB archive; slow, so cache the filtered result
        traces = pipeline.fetch_region(date, (la0 - PAD_DEG, la1 + PAD_DEG, lo0 - pad_lon, lo1 + pad_lon), tr_path)
    elev = Terrain().elevation
    grid = Grid.make(la0, la1, lo0, lo1, spacing, elev)
    # the two strongest flyovers (as validated in the README); scores of different event pairs
    # are not comparable, so trying several pairs and keeping the top score picks wrong ones
    e1, e2 = sorted(evs[:2], key=lambda e: e.t_cpa)
    log(f"searching flyovers at {e1.t_cpa:.0f}s and {e2.t_cpa:.0f}s")
    # auto-detected fits can be far off in speed/distance (Oyster Bay event 2: 79 m/s vs 141 true),
    # and these only gate candidates; the real-trajectory score ranks them, so gate loosely
    res, segs = pipeline.run_search(sp, e1, e2, traces, grid, v_tol=0.9, d_tol=(0.25, 4.0), progress=False)
    if not len(res):
        raise RuntimeError("no flight pair is consistent with the recording inside this box and date")
    pairs = res.drop_duplicates(["icao1", "icao2"]).reset_index(drop=True)
    row = res.iloc[0].copy(); row["lat0"] = grid.lat0; row["lon0"] = grid.lon0
    log(f"refining around {row.lat:.5f}, {row.lon:.5f}")
    # shortcut: no --extra flyovers; auto-detected extras included junk fits and moved Oyster Bay
    # from 460 m to 660 m off. Upgrade by filtering extras (edge-of-grid fits, reused flights) first.
    fine = pipeline.refine(sp, e1, e2, segs, row, elevation=elev)
    b = fine.iloc[0]
    near = fine[fine.score >= 0.95 * b.score]
    m_per_deg = 111320.0
    radius = float(np.hypot((near.lat - b.lat) * m_per_deg, (near.lon - b.lon) * m_per_deg * np.cos(np.radians(b.lat))).max()) + 25
    margin = (pairs.score[0] - pairs.score[1]) / pairs.score[1] if len(pairs) > 1 else None
    return dict(lat=float(b.lat), lon=float(b.lon), score=float(b.score), radius=radius, margin=margin,
                flights=(row.icao1, row.icao2), n_events=len(evs), n_pairs=len(pairs),
                candidates=[(float(r.lat), float(r.lon), float(r.score)) for r in pairs.head(5).itertuples()],
                # heat map: best score per coarse grid point, and the fine grid's high-scoring part
                coarse=[(float(r.lat), float(r.lon), float(r.score)) for r in res.drop_duplicates("gidx").head(400).itertuples()],
                coarse_m=spacing, fine=[(float(r.lat), float(r.lon), float(r.score)) for r in fine[fine.score >= 0.6 * b.score].itertuples()],
                fine_m=50.0)


def _square(lat, lon, half_m):
    dla = half_m / 111320; dlo = half_m / (111320 * np.cos(np.radians(lat)))
    return [(lat - dla, lon - dlo), (lat - dla, lon + dlo), (lat + dla, lon + dlo), (lat + dla, lon - dlo)]


def _circle(lat, lon, r_m):
    a = np.linspace(0, 2 * np.pi, 48)
    return list(zip(lat + r_m * np.sin(a) / 111320, lon + r_m * np.cos(a) / (111320 * np.cos(np.radians(lat)))))


class App:
    def __init__(self):
        self.root = TkinterDnD.Tk()
        self.root.title("dopplertrack")
        self.root.geometry("1200x780")
        self.q = queue.Queue()
        self.file = tk.StringVar(); self.date = tk.StringVar(); self.spacing = tk.StringVar(value="200")
        self.bbox = None; self.overlays = []; self.box_poly = None; self.result = None

        side = ttk.Frame(self.root, padding=8); side.pack(side="left", fill="y")
        self.map = Map(self.root, corner_radius=0)
        self.map.pack(side="right", fill="both", expand=True)
        self.map.set_tile_server(TILES, max_zoom=19)
        self.map.set_position(40.8, -73.5); self.map.set_zoom(8)
        self._bind_box_drawing()

        ttk.Label(side, text="1. Bounding box", font=("", 11, "bold")).pack(anchor="w")
        ttk.Label(side, text="Shift+drag on the map, or:").pack(anchor="w")
        ttk.Button(side, text="Use current map view", command=self.box_from_view).pack(fill="x")
        self.box_lbl = ttk.Label(side, text="no box selected", wraplength=260); self.box_lbl.pack(anchor="w", pady=(0, 10))

        ttk.Label(side, text="2. Recording", font=("", 11, "bold")).pack(anchor="w")
        ttk.Button(side, text="Choose audio/video file…", command=self.pick_file).pack(fill="x")
        ttk.Label(side, textvariable=self.file, wraplength=260).pack(anchor="w")
        ttk.Label(side, text="(or drop a file on this window)").pack(anchor="w")
        ttk.Label(side, text="Recording date (UTC, YYYY-MM-DD)").pack(anchor="w", pady=(6, 0))
        ttk.Entry(side, textvariable=self.date).pack(fill="x")
        ttk.Label(side, text="Grid spacing, m (smaller = slower)").pack(anchor="w", pady=(6, 0))
        ttk.Entry(side, textvariable=self.spacing).pack(fill="x", pady=(0, 10))

        ttk.Label(side, text="3. Geolocate", font=("", 11, "bold")).pack(anchor="w")
        self.go = ttk.Button(side, text="Geolocalize", command=self.start); self.go.pack(fill="x")
        self.log_box = tk.Text(side, width=36, height=10, wrap="word"); self.log_box.pack(fill="x", pady=6)

        self.out = ttk.Label(side, text="", wraplength=260, justify="left"); self.out.pack(anchor="w")
        self.coords = tk.StringVar()
        ttk.Entry(side, textvariable=self.coords, state="readonly").pack(fill="x")
        row = ttk.Frame(side); row.pack(fill="x", pady=4)
        ttk.Button(row, text="Copy", command=self.copy).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Open in Google Maps", command=self.gmaps).pack(side="left", expand=True, fill="x")

        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<Drop>>", lambda e: self.set_file(self.root.tk.splitlist(e.data)[0]))
        self.root.after(200, self.poll)

    # --- bounding box
    def _bind_box_drawing(self):
        c = self.map.canvas; st = {}
        def down(e): st["p0"] = (e.x, e.y); st["id"] = c.create_rectangle(e.x, e.y, e.x, e.y, outline="red", width=2)
        def move(e): c.coords(st["id"], *st["p0"], e.x, e.y)
        def up(e):
            c.delete(st["id"])
            (x0, y0), (x1, y1) = st["p0"], (e.x, e.y)
            if abs(x1 - x0) > 5 and abs(y1 - y0) > 5:
                a = self.map.convert_canvas_coords_to_decimal_coords(x0, y0); b = self.map.convert_canvas_coords_to_decimal_coords(x1, y1)
                self.set_box(min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1]))
        c.bind("<Shift-ButtonPress-1>", down); c.bind("<Shift-B1-Motion>", move); c.bind("<Shift-ButtonRelease-1>", up)

    def box_from_view(self):
        (n, w), (s, e) = self.map.convert_canvas_coords_to_decimal_coords(0, 0), \
            self.map.convert_canvas_coords_to_decimal_coords(self.map.winfo_width(), self.map.winfo_height())
        self.set_box(s, n, w, e)

    def set_box(self, la0, la1, lo0, lo1):
        self.bbox = (la0, la1, lo0, lo1)
        if self.box_poly:
            self.box_poly.delete()
        self.box_poly = self.map.set_polygon([(la0, lo0), (la0, lo1), (la1, lo1), (la1, lo0)], fill_color=None, outline_color="red", border_width=2)
        self.box_lbl.config(text=f"lat {la0:.3f}…{la1:.3f}, lon {lo0:.3f}…{lo1:.3f}\n≈ {(la1 - la0) * 111:.0f} × {(lo1 - lo0) * 111 * np.cos(np.radians((la0 + la1) / 2)):.0f} km")

    # --- file
    def pick_file(self):
        p = filedialog.askopenfilename()
        if p:
            self.set_file(p)

    def set_file(self, p):
        self.file.set(p)

    # --- run
    def say(self, msg):
        self.q.put(("log", msg))

    def start(self):
        try:
            dt.date.fromisoformat(self.date.get()); spacing = float(self.spacing.get())
            assert self.bbox and self.file.get()
        except Exception:
            self.out.config(text="Select a box, a file, a valid date (YYYY-MM-DD) and a numeric spacing."); return
        self.go.config(state="disabled"); self.log_box.delete("1.0", "end"); self.out.config(text="working…")
        def work():
            try:
                self.q.put(("done", locate(self.file.get(), self.date.get(), self.bbox, spacing, self.say)))
            except Exception as ex:
                self.q.put(("error", f"{ex}\n{traceback.format_exc(limit=3)}"))
        threading.Thread(target=work, daemon=True).start()

    def poll(self):
        while not self.q.empty():
            kind, v = self.q.get()
            if kind == "log":
                self.log_box.insert("end", v + "\n"); self.log_box.see("end")
            elif kind == "error":
                self.go.config(state="normal"); self.out.config(text=f"Failed: {v}")
            else:
                self.go.config(state="normal"); self.show(v)
        self.root.after(200, self.poll)

    # --- result
    def show(self, r):
        self.result = r
        for o in self.overlays:
            o.delete()
        lat, lon, rad = r["lat"], r["lon"], r["radius"]
        self.overlays = []
        for pts, size in ((r["coarse"], r["coarse_m"]), (r["fine"], r["fine_m"])):
            if not pts:
                continue
            sc = np.array([p[2] for p in pts]); lo_, hi_ = sc.min(), max(sc.max(), sc.min() + 1e-9)
            for la, lo, s in sorted(pts, key=lambda p: p[2]):  # brightest drawn last, on top
                col = to_hex(colormaps["turbo"]((s - lo_) / (hi_ - lo_)))
                self.overlays.append(self.map.set_polygon(_square(la, lo, size / 2), fill_color=col, outline_color=col, border_width=1))
        self.overlays.append(self.map.set_polygon(_circle(lat, lon, rad), fill_color=None, outline_color="yellow", border_width=3))
        for i, (la, lo, _) in enumerate(r["candidates"][1:], 2):
            self.overlays.append(self.map.set_marker(la, lo, text=f"#{i}", text_color="white", marker_color_circle="gray", marker_color_outside="white"))
        self.overlays.append(self.map.set_marker(lat, lon, text=f"{lat:.5f}, {lon:.5f}", text_color="white", marker_color_circle="red", marker_color_outside="yellow"))
        self.map.set_position(lat, lon); self.map.set_zoom(14 if rad < 3000 else 12)
        m = r["margin"]
        self.out.config(text=f"Best location (uncertainty ±{rad:.0f} m)\nscore {r['score']:.1f}"
                        + (f", {m * 100:.1f}% ahead of next flight pair" if m is not None else "")
                        + f"\nflights {r['flights'][0]} / {r['flights'][1]}\n{r['n_events']} flyovers detected, {r['n_pairs']} flight pairs ranked"
                        + f"\n\nHeat map: big squares = {r['coarse_m']:.0f} m search grid (top 400), small = 50 m refinement."
                        " Colour = score, blue low → red high. Yellow circle = uncertainty. Gray pins = next-best flight pairs.")
        self.coords.set(f"{lat:.6f}, {lon:.6f}")

    def copy(self):
        self.root.clipboard_clear(); self.root.clipboard_append(self.coords.get())

    def gmaps(self):
        if self.result:
            webbrowser.open(f"https://www.google.com/maps/search/?api=1&query={self.result['lat']:.6f},{self.result['lon']:.6f}")


def main():
    App().root.mainloop()


if __name__ == "__main__":
    main()
