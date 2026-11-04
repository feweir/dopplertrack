"""dopplertrack-gui: pick a bounding box on a satellite map, drop a recording, geolocate it."""
from __future__ import annotations

import dataclasses
import datetime as dt
import io
import json
import queue
import threading
import time
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
from PIL import Image, ImageDraw, ImageTk
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
SESSIONS = Path.home() / "dopplertrack-sessions"


def _fetch_tile(url):
    try:
        r = requests.get(url, headers={"User-Agent": "dopplertrack"}, timeout=15)
        r.raise_for_status()
        return Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception:
        return None


class Map(tkintermapview.TkinterMapView):
    """The stock widget loads tiles in threads that also create, hold and drop Tk objects, which
    segfaults Tk 9. Here the threads only download; every Tk object stays on the main thread."""

    def __init__(self, *args, **kw):
        self._pool = ThreadPoolExecutor(8)
        self._inflight = []  # (zoom, x, y, canvas tile, future)
        self._raw = {}       # (zoom, x, y) -> downloaded PIL tile, so overlays redraw without refetching
        # overlays are drawn into the tiles (Tk canvas items cannot be translucent):
        # name -> dict(items=[...], alpha=0..1, on=bool); items are ("rect", x0, y0, x1, y1, colour)
        # or ("line", (N, 2) array, colour, width) in Web Mercator units (0..1)
        self.layers = {}
        super().__init__(*args, **kw)

    def set_layer(self, name, **kw):
        self.layers.setdefault(name, dict(items=[], alpha=1.0, on=True)).update(kw)
        self.refresh_tiles()

    def _compose(self, zoom, x, y):
        img = self._raw[(zoom, x, y)]
        n = 256 * 2 ** zoom
        for layer in self.layers.values():
            if not (layer["on"] and layer["items"] and layer["alpha"] > 0):
                continue
            ov = Image.new("RGBA", img.size); d = ImageDraw.Draw(ov)
            for it in layer["items"]:
                if it[0] == "rect":
                    x0, y0, x1, y1 = (it[1] * n - x * 256, it[2] * n - y * 256, it[3] * n - x * 256, it[4] * n - y * 256)
                    if x1 >= 0 and y1 >= 0 and x0 <= 256 and y0 <= 256:
                        d.rectangle((x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)), fill=it[5])
                else:
                    xy = it[1] * n - (x * 256, y * 256)
                    if xy[:, 0].max() >= 0 and xy[:, 1].max() >= 0 and xy[:, 0].min() <= 256 and xy[:, 1].min() <= 256:
                        d.line([tuple(p) for p in xy], fill=it[2], width=it[3])
            ov.putalpha(ov.getchannel("A").point(lambda v, a=layer["alpha"]: int(v * a)))
            img = Image.alpha_composite(img, ov)
        return ImageTk.PhotoImage(img)

    def _show_tile(self, zoom, x, y, tile):
        img = self.tile_image_cache[f"{zoom}{x}{y}"] = self._compose(zoom, x, y)
        tile.set_image(img)

    def refresh_tiles(self):
        self.tile_image_cache.clear()
        z = round(self.zoom)
        for col in self.canvas_tile_array:
            for tile in col:
                x, y = (int(v) for v in tile.tile_name_position)
                if (z, x, y) in self._raw:
                    self._show_tile(z, x, y, tile)

    def pre_cache(self):
        pass

    def load_images_background(self):
        pass

    def update_canvas_tile_images(self):
        while self.image_load_queue_tasks:
            (zoom, x, y), tile = self.image_load_queue_tasks.pop()
            if (zoom, x, y) in self._raw:
                self._show_tile(zoom, x, y, tile)
            else:
                self._inflight.append((zoom, x, y, tile, self._pool.submit(_fetch_tile, self.tile_server.format(x=x, y=y, z=zoom))))
        waiting = []
        for zoom, x, y, tile, fut in self._inflight:
            if not fut.done():
                waiting.append((zoom, x, y, tile, fut)); continue
            img = fut.result()
            if img is not None:
                if len(self._raw) > 1500:  # ~300 MB; drop the oldest half
                    for k in list(self._raw)[:750]:
                        del self._raw[k]
                self._raw[(zoom, x, y)] = img
                if zoom == round(self.zoom):
                    self._show_tile(zoom, x, y, tile)
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
    pipeline.log = log  # route the pipeline's progress lines (scored k/n, ...) into the window
    out = SESSIONS / f"{time.strftime('%Y%m%d-%H%M%S')}_{Path(path).stem}"
    (out / "plots").mkdir(parents=True)
    log(f"session folder: {out}")
    log("decoding audio, computing spectrogram")
    sp = pipeline.make_spectrogram(path, out / "spectrogram.npz")
    log("detecting flyovers")
    evs = detect_events(sp, 60, 300, top=8, progress=lambda k, n: log(f"  window {k}/{n}"))
    (out / "detected_events.json").write_text(json.dumps([e.to_dict() for e in evs], indent=1))
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
    # Auto-detected fits are rough (Oyster Bay event 2: 79 m/s and 140 Hz vs 141 m/s and 132 Hz
    # by hand), and they only seed the search; the real-trajectory score ranks. So score over a
    # 110 s window centred on the pass, sweep f0 ±10 % and allow ±15 s on the event spacing (its
    # timing shifts with the bad fit). On Oyster Bay this moves the true point from below a wrong
    # one 78 km away to 20 % above it. Loosening the speed/distance gates too was 30x more candidates.
    e1, e2 = (dataclasses.replace(e, t_start=e.t_cpa - 55, t_end=e.t_cpa + 55) for e in sorted(evs[:2], key=lambda e: e.t_cpa))
    log(f"searching flyovers at {e1.t_cpa:.0f}s and {e2.t_cpa:.0f}s")
    res, segs = pipeline.run_search(sp, e1, e2, traces, grid, tol_s=15, f0_tol=0.10)
    for k, e in ((1, e1), (2, e2)):
        (out / f"event{k}.json").write_text(json.dumps(e.to_dict(), indent=1))
    res.to_csv(out / "search_results.csv", index=False)
    if not len(res):
        raise RuntimeError("no flight pair is consistent with the recording inside this box and date")
    pairs = res.drop_duplicates(["icao1", "icao2"]).reset_index(drop=True)
    row = res.iloc[0].copy(); row["lat0"] = grid.lat0; row["lon0"] = grid.lon0
    log(f"refining around {row.lat:.5f}, {row.lon:.5f}")
    # shortcut: no --extra flyovers; auto-detected extras included junk fits and moved Oyster Bay
    # from 460 m to 660 m off. Upgrade by filtering extras (edge-of-grid fits, reused flights) first.
    fine = pipeline.refine(sp, e1, e2, segs, row, elevation=elev, f0_tol=0.10)
    b = fine.iloc[0]
    near = fine[fine.score >= 0.95 * b.score]
    m_per_deg = 111320.0
    radius = float(np.hypot((near.lat - b.lat) * m_per_deg, (near.lon - b.lon) * m_per_deg * np.cos(np.radians(b.lat))).max()) + 25
    coarse = res.drop_duplicates("gidx").head(400)
    coarse = coarse[~(coarse.lat.between(fine.lat.min(), fine.lat.max()) & coarse.lon.between(fine.lon.min(), fine.lon.max()))]
    # flight tracks while the recording ran (audio time = unix time + t0_map)
    t0_map = e1.t_cpa - float(row.t_heard1)
    t_a, t_b = sp.t[0] - t0_map - 120, sp.t[-1] - t0_map + 120
    tracks = []
    for tr in traces:
        m = (tr.t >= t_a) & (tr.t <= t_b) & ~tr.on_ground
        if m.sum() >= 2:
            tracks.append((tr.icao, tr.lat[m], tr.lon[m]))
    margin = (pairs.score[0] - pairs.score[1]) / pairs.score[1] if len(pairs) > 1 else None
    r = dict(lat=float(b.lat), lon=float(b.lon), score=float(b.score), radius=radius, margin=None if margin is None else float(margin),
                flights=(row.icao1, row.icao2), n_events=len(evs), n_pairs=len(pairs),
                candidates=[(float(r.lat), float(r.lon), float(r.score)) for r in pairs.head(5).itertuples()],
                # heat map: best score per coarse grid point outside the fine grid, and the whole fine grid
                coarse=[(float(r.lat), float(r.lon), float(r.score)) for r in coarse.itertuples()], coarse_m=spacing,
                fine=[(float(r.lat), float(r.lon), float(r.score)) for r in fine.itertuples()], fine_m=50.0,
                tracks=[(icao, la.tolist(), lo.tolist()) for icao, la, lo in tracks],
                audio=str(path), date=date, bbox=list(bbox), session=str(out))
    fine.to_csv(out / "refined_50m.csv", index=False)
    (out / "result.json").write_text(json.dumps(r))
    log("drawing plots")
    _save_plots(out / "plots", sp, evs, e1, e2, res, fine, segs, row, elev, r)
    log(f"saved to {out}")
    return r



def _save_plots(d, sp, evs, e1, e2, res, fine, segs, row, elev, r):
    """PNG diagnostics for a session: the recording, both event fits, the curves the winning
    location predicts over the audio, and both score maps."""
    import matplotlib.pyplot as plt
    from .doppler import speed_of_sound
    from .geo import lla_to_enu
    from .plots import _show, plot_event_fit, plot_modelled, plot_score_map
    from .score import EventWindow, modelled_curve

    def save(fig, name):
        fig.tight_layout(); fig.savefig(d / name, dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 5))
    _show(ax, sp, vmax=8)
    for i, e in enumerate(evs, 1):
        used = e.t_cpa in (e1.t_cpa, e2.t_cpa)
        ax.axvline(e.t_cpa, color="cyan" if used else "white", lw=1.2 if used else 0.6, ls="-" if used else "--")
        ax.text(e.t_cpa, sp.f[-1], f" {i}", color="cyan" if used else "white", va="top")
    ax.set_xlabel("recording time, s"); ax.set_title("recording (whitened 40-450 Hz); detected flyovers, cyan = the two used")
    save(fig, "recording_spectrogram.png")

    c = speed_of_sound()
    obs = lla_to_enu(r["lat"], r["lon"], float(elev(np.array([r["lat"]]), np.array([r["lon"]]))[0]), row.lat0, row.lon0)
    seg1, seg2 = segs[int(row.seg1)], segs[int(row.seg2)]
    dd = np.linalg.norm(seg1.enu - obs, axis=1); j = int(np.argmin(dd))
    t0_map = e1.t_cpa - (seg1.t[j] + dd[j] / c)  # audio time = unix time + t0_map, at the refined point
    shift = float(fine.iloc[0]["shift"]); scales = np.arange(0.9, 1.1001, 0.005)
    for k, (e, seg, icao) in enumerate(((e1, seg1, row.icao1), (e2, seg2, row.icao2)), 1):
        fig, ax = plt.subplots(figsize=(8, 4.5)); plot_event_fit(sp, e, ax=ax); save(fig, f"event{k}_fit.png")
        to, rat = modelled_curve(seg, obs, c)
        f0s = scales[int(np.argmax(EventWindow.from_fit(sp, e).sweep_scores(to + t0_map, rat, [shift], scales)[0]))]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_modelled(sp, e, seg, obs, t0_map + shift, f0_scale=f0s, ax=ax,
                      label=f"event {k}: curve heard from flight {icao} at the result point")
        save(fig, f"event{k}_modelled.png")

    fig, ax = plt.subplots(figsize=(8, 7))
    plot_score_map(res, ax=ax); ax.plot(r["lon"], r["lat"], "r+", ms=16, mew=2)
    ax.set_title("search: best score per grid point"); save(fig, "search_score_map.png")
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(fine.lon, fine.lat, c=fine.score, cmap="turbo", marker="s", s=40)
    plt.colorbar(sc, ax=ax, label="score"); ax.plot(r["lon"], r["lat"], "r+", ms=18, mew=2)
    ax.set_aspect(1 / np.cos(np.radians(r["lat"]))); ax.set_xlabel("lon"); ax.set_ylabel("lat")
    ax.set_title(f"refined 50 m grid; peak {r['lat']:.6f}, {r['lon']:.6f}"); save(fig, "refined_score_map.png")


def _merc(lat, lon):
    """lat/lon (degrees) -> Web Mercator x, y in 0..1, the frame map tiles are cut in."""
    lat = np.clip(np.asarray(lat, float), -85, 85)
    return (np.asarray(lon, float) + 180) / 360, (1 - np.log(np.tan(np.radians(lat)) + 1 / np.cos(np.radians(lat))) / np.pi) / 2


def _heat_items(pts, size_m):
    """Gapped squares coloured by score (blue low -> red high), like the README's score maps."""
    if not pts:
        return []
    a = np.array(pts); lo_, hi_ = a[:, 2].min(), max(a[:, 2].max(), a[:, 2].min() + 1e-9)
    half = 0.35 * size_m  # 70 % of the grid step, leaving gaps
    dla = half / 111320; dlo = half / (111320 * np.cos(np.radians(a[:, 0])))
    x0, y1 = _merc(a[:, 0] - dla, a[:, 1] - dlo); x1, y0 = _merc(a[:, 0] + dla, a[:, 1] + dlo)
    order = np.argsort(a[:, 2])  # brightest drawn last, on top
    return [("rect", x0[i], y0[i], x1[i], y1[i], to_hex(colormaps["turbo"]((a[i, 2] - lo_) / (hi_ - lo_)))) for i in order]


def _cross(size, colour):
    im = Image.new("RGBA", (size, size)); d = ImageDraw.Draw(im); c = size // 2
    for w, col in ((5, "black"), (3, colour)):  # dark outline keeps it visible on any imagery
        d.line((c, 1, c, size - 2), fill=col, width=w); d.line((1, c, size - 2, c), fill=col, width=w)
    return ImageTk.PhotoImage(im)


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

        # sidebar inside a scrolling canvas, so the result buttons stay reachable on short screens
        outer = ttk.Frame(self.root); outer.pack(side="left", fill="y")
        cv = tk.Canvas(outer, highlightthickness=0); sb = ttk.Scrollbar(outer, orient="vertical", command=cv.yview)
        cv.configure(yscrollcommand=sb.set); sb.pack(side="right", fill="y"); cv.pack(side="left", fill="y")
        side = ttk.Frame(cv, padding=8); cv.create_window((0, 0), window=side, anchor="nw")
        side.bind("<Configure>", lambda e: cv.configure(scrollregion=cv.bbox("all"), width=side.winfo_reqwidth()))
        def wheel(e):  # only over the sidebar (the map zooms on wheel), and not inside text boxes
            if str(e.widget).startswith(str(cv)) and not isinstance(e.widget, tk.Text):
                cv.yview_scroll(-1 if e.delta > 0 else 1, "units")
        self.root.bind_all("<MouseWheel>", wheel, add="+")
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
        ttk.Button(side, text="Open saved session…", command=self.open_session).pack(fill="x", pady=(4, 0))

        # result actions sit right under the run buttons so they are visible without scrolling
        ttk.Label(side, text="Result (lat, lon)").pack(anchor="w", pady=(8, 0))
        self.coords = tk.StringVar()
        ttk.Entry(side, textvariable=self.coords, state="readonly").pack(fill="x")
        row = ttk.Frame(side); row.pack(fill="x", pady=4)
        ttk.Button(row, text="Copy", command=self.copy).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Open in Google Maps", command=self.gmaps).pack(side="left", expand=True, fill="x")
        self.log_box = tk.Text(side, width=36, height=10, wrap="word"); self.log_box.pack(fill="x", pady=6)

        self.show_tracks = tk.BooleanVar(value=True); self.track_alpha = tk.DoubleVar(value=0.6)
        ttk.Checkbutton(side, text="Show flight tracks", variable=self.show_tracks, command=self.tracks_changed).pack(anchor="w")
        ttk.Label(side, text="Track opacity").pack(anchor="w")
        ttk.Scale(side, from_=0.05, to=1.0, variable=self.track_alpha, command=lambda _: self.tracks_changed(delay=150)).pack(fill="x")
        self._track_job = None

        self.out = ttk.Label(side, text="", wraplength=260, justify="left"); self.out.pack(anchor="w")

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
        self.map.set_layer("heat", items=_heat_items(r["coarse"], r["coarse_m"]) + _heat_items(r["fine"], r["fine_m"]), alpha=0.85)
        tracks = []
        for icao, la, lo in r["tracks"]:
            x, y = _merc(la, lo); hit = icao in r["flights"]
            tracks.append(("line", np.column_stack([x, y]), "#00e5ff" if hit else "#ffffff", 3 if hit else 1))
        self.map.set_layer("tracks", items=tracks, alpha=self.track_alpha.get(), on=self.show_tracks.get())
        self.icons = [_cross(23, "red"), _cross(13, "white")]  # keep references, Tk drops unreferenced images
        self.overlays = [self.map.set_polygon(_circle(lat, lon, rad), fill_color=None, outline_color="yellow", border_width=2)]
        for i, (la, lo, _) in enumerate(r["candidates"][1:], 2):
            self.overlays.append(self.map.set_marker(la, lo, text=f"#{i}", text_color="white", icon=self.icons[1]))
        self.overlays.append(self.map.set_marker(lat, lon, icon=self.icons[0]))
        self.map.set_position(lat, lon); self.map.set_zoom(14 if rad < 3000 else 12)
        m = r["margin"]
        self.out.config(text=f"Best location (uncertainty ±{rad:.0f} m)\nscore {r['score']:.1f}"
                        + (f", {m * 100:.1f}% ahead of next flight pair" if m is not None else "")
                        + f"\nflights {r['flights'][0]} / {r['flights'][1]}\n{r['n_events']} flyovers detected, {r['n_pairs']} flight pairs ranked"
                        + f"\n\nHeat map: big squares = {r['coarse_m']:.0f} m search grid (top 400), small = 50 m refinement."
                        " Colour = score, blue low → red high. Red cross = best location, yellow circle = uncertainty,"
                        " white crosses = next-best flight pairs. Tracks: cyan = the two matched flights.")
        self.coords.set(f"{lat:.6f}, {lon:.6f}")

    def open_session(self):
        d = filedialog.askdirectory(initialdir=SESSIONS if SESSIONS.exists() else Path.home(), title="Session folder")
        if not d:
            return
        try:
            r = json.loads((Path(d) / "result.json").read_text())
        except (OSError, ValueError) as ex:
            self.out.config(text=f"Not a session folder ({ex})"); return
        self.file.set(r["audio"]); self.date.set(r["date"]); self.spacing.set(str(r["coarse_m"]))
        self.set_box(*r["bbox"])
        self.log_box.delete("1.0", "end"); self.log_box.insert("end", f"opened {d}\n")
        self.show(r)

    def tracks_changed(self, delay=0):
        # the slider fires on every pixel of movement; redraw tiles once it settles
        if self._track_job:
            self.root.after_cancel(self._track_job)
        self._track_job = self.root.after(delay, lambda: self.map.set_layer("tracks", on=self.show_tracks.get(), alpha=self.track_alpha.get()))

    def copy(self):
        self.root.clipboard_clear(); self.root.clipboard_append(self.coords.get())

    def gmaps(self):
        if self.result:
            webbrowser.open(f"https://www.google.com/maps/search/?api=1&query={self.result['lat']:.6f},{self.result['lon']:.6f}")


def main():
    App().root.mainloop()


if __name__ == "__main__":
    main()
