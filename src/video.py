"""Audio-reactive music video renderer (pure numpy + Pillow + ffmpeg, CPU only).

Every visual is a pure function of (frame index, precomputed audio features), so
frames render in parallel across all runner cores and stream straight to ffmpeg.
"""
from __future__ import annotations

import glob
import math
import os
import subprocess
from multiprocessing import Pool

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G: dict = {}  # per-process state (set by _init)


# ─── fonts ─────────────────────────────────────────────────────────────────────
def _font_path(kind: str) -> str:
    wanted = {"title": ["Anton*.ttf", "*Black*.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf"],
              "body": ["Inter*.ttf", "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "*.ttf"]}[kind]
    dirs = [os.path.join(ROOT, "assets", "fonts"), "/usr/share/fonts", "/usr/local/share/fonts"]
    for pat in wanted:
        for d in dirs:
            hits = sorted(glob.glob(os.path.join(d, "**", pat), recursive=True))
            if hits:
                return hits[0]
    raise FileNotFoundError("No TTF font found — add one to assets/fonts/")


_FONTS: dict = {}


def font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    key = (kind, size)
    if key not in _FONTS:
        f = ImageFont.truetype(_font_path(kind), size)
        try:
            f.set_variation_by_axes([700])  # bold weight for variable fonts like Inter
        except Exception:  # noqa: BLE001
            pass
        _FONTS[key] = f
    return _FONTS[key]


# ─── audio analysis ────────────────────────────────────────────────────────────
def analyze(path: str, fps: int) -> dict:
    sr, x = wavfile.read(path)
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    x /= np.abs(x).max() + 1e-9
    hop, win = sr / fps, 4096
    n = int(len(x) / hop)
    pad = np.pad(x, (win // 2, win))
    idx = (np.arange(n) * hop).astype(int)
    frames = np.stack([pad[i:i + win] for i in idx]) * np.hanning(win)
    mag = np.abs(np.fft.rfft(frames, axis=1))
    freqs = np.fft.rfftfreq(win, 1 / sr)

    def band(lo, hi):
        v = mag[:, (freqs >= lo) & (freqs < hi)].mean(axis=1)
        return np.clip(v / (np.percentile(v, 97) + 1e-9), 0, 1.2)

    def smooth(v, rel=0.85):
        out = np.empty_like(v)
        cur = 0.0
        for i, s in enumerate(v):
            cur = s if s > cur else cur * rel + s * (1 - rel)
            out[i] = cur
        return out

    bass, mid, high = smooth(band(25, 160)), smooth(band(160, 2000), 0.8), smooth(band(2000, 12000), 0.7)
    rms = np.sqrt((frames ** 2).mean(axis=1))
    energy = smooth(np.clip(rms / (np.percentile(rms, 97) + 1e-9), 0, 1.2), 0.93)

    # beat pulses from bass onsets
    raw_bass = band(25, 160)
    flux = np.maximum(0, raw_bass - np.convolve(raw_bass, np.ones(8) / 8, mode="same"))
    thr = np.percentile(flux, 90)
    beat, last, cur = np.zeros(n), -99, 0.0
    for i in range(n):
        cur *= 0.82
        if flux[i] > thr and i - last > fps * 0.22:
            cur, last = 1.0, i
        beat[i] = cur

    edges = np.searchsorted(freqs, np.geomspace(40, 12000, 65))
    spec = np.stack([mag[:, edges[k]:max(edges[k + 1], edges[k] + 1)].mean(axis=1) + 1e-9
                     for k in range(64)], axis=1)
    spec = np.log1p(spec / np.percentile(spec, 95, axis=0)) / math.log(2.5)
    for k in range(1, n):  # fall-off smoothing
        spec[k] = np.maximum(spec[k], spec[k - 1] * 0.86)
    travel = np.cumsum((0.35 + bass * 0.9 + beat * 0.4) / fps)
    return dict(n=n, fps=fps, bass=bass, mid=mid, high=high, energy=energy, beat=beat,
                spec=np.clip(spec, 0, 1.2), travel=travel, duration=len(x) / sr)


# ─── colour ────────────────────────────────────────────────────────────────────
def make_lut(palette: list[str]) -> np.ndarray:
    cols = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)] for c in palette], dtype=np.float32) / 255
    cols = np.vstack([cols, np.minimum(1, cols[-1] * 0.4 + 0.65)])  # hot white-ish top end
    pos = np.linspace(0, 1, len(cols))
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, pos, cols[:, c]) for c in range(3)], axis=1)


def shade(field: np.ndarray) -> np.ndarray:
    return G["lut"][(np.clip(field, 0, 1) * 255).astype(np.uint8)]


# ─── backgrounds (low-res, float RGB 0..1) ─────────────────────────────────────
def bg_tunnel(i, f):
    X, Y, R, A = G["grid"]
    t, bass, beat = f["travel"][i], f["bass"][i], f["beat"][i]
    depth = 0.35 / (R + 0.04)
    u = depth + t * 1.6
    v = A / math.pi + t * 0.05
    rings = 0.5 + 0.5 * np.sin(u * 6.0)
    spokes = 0.5 + 0.5 * np.sin(v * 12 * math.pi + depth * 0.8)
    field = (rings * 0.7 + spokes * 0.3) * np.clip(R * 1.8, 0, 1) ** 1.2
    field = field * (0.55 + 0.45 * f["energy"][i]) + 0.12 * beat * np.exp(-R * 2)
    return shade(field + bass * 0.08)


def bg_plasma(i, f):
    X, Y, R, A = G["grid"]
    t = f["travel"][i] * 0.6
    s = 2.2 + f["bass"][i] * 0.8
    v = (np.sin(X * s + t) + np.sin(Y * s * 1.3 - t * 1.2) + np.sin((X + Y) * s * 0.7 + t * 0.7)
         + np.sin(np.sqrt((X * 1.5 + math.sin(t * 0.3)) ** 2 + (Y + math.cos(t * 0.4)) ** 2) * s * 1.6))
    field = 0.5 + 0.5 * np.sin(v * 1.2 + t)
    return shade(field * (0.5 + 0.5 * f["energy"][i]) + 0.15 * f["beat"][i])


def bg_kaleido(i, f):
    X, Y, R, A = G["grid"]
    t = f["travel"][i]
    seg = 2 * math.pi / 8
    a = np.abs(((A + t * 0.15) % seg) - seg / 2)
    kx, ky = R * np.cos(a), R * np.sin(a)
    s = 5 + f["bass"][i] * 2
    v = np.sin(kx * s - t * 1.5) * np.cos(ky * s * 1.4 + t) + np.sin(R * 7 - t * 2.5)
    field = (0.5 + 0.35 * v) * np.clip(1.3 - R * 0.6, 0, 1)
    return shade(field * (0.55 + 0.5 * f["energy"][i]) + 0.12 * f["beat"][i])


def bg_starfield(i, f):
    h, w = G["lh"], G["lw"]
    sx, sy, sz = G["stars"]
    z = (sz - f["travel"][i] * 0.9) % 1.0 + 0.02
    px = (sx / z * 0.5 * h + w / 2).astype(int)
    py = (sy / z * 0.5 * h + h / 2).astype(int)
    ok = (px >= 0) & (px < w) & (py >= 0) & (py < h)
    img = np.zeros((h, w), np.float32)
    np.add.at(img, (py[ok], px[ok]), np.clip(0.15 / z[ok], 0, 1.5))
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, 0.7 + f["bass"][i] * 0.6) * 9
    X, Y, R, A = G["grid"]
    neb = 0.18 * (0.5 + 0.5 * np.sin(X * 1.7 + f["travel"][i] * 0.2) * np.cos(Y * 2.1)) * (0.6 + f["mid"][i])
    return shade(np.clip(neb + img, 0, 1) + 0.1 * f["beat"][i] * np.exp(-R * 3))


def bg_synthgrid(i, f):
    X, Y, R, A = G["grid"]
    hz = 0.12
    out = np.zeros(X.shape + (3,), np.float32)
    sky = Y < hz
    out[sky] = shade(np.clip(0.25 + (Y[sky] + 1) * 0.35, 0, 1))
    # sun with stripe cutouts
    sr = 0.55 + 0.04 * f["bass"][i]
    sd = np.sqrt(X ** 2 + (Y + 0.25) ** 2)
    stripes = (np.sin((Y + 0.25) * 40 - f["travel"][i] * 2) > (Y + 0.25) * 3) | (Y < -0.05)
    sun = (sd < sr) & sky & stripes
    out[sun] = shade(0.75 + 0.25 * (1 - (Y[sun] + 0.8) / 1.0))
    glow = np.exp(-np.maximum(sd - sr, 0) * 4) * 0.35 * sky
    out += shade(np.full_like(X, 0.8)) * glow[..., None]
    # floor grid
    fl = ~sky
    z = 0.35 / (Y[fl] - hz + 0.02)
    gx = X[fl] * z
    gz = z + f["travel"][i] * 1.5
    lx =np.abs(((gx * 2) % 1) - 0.5) < 0.04 * z ** 0.6
    lz = np.abs((gz % 1) - 0.5) < 0.05 * z ** 0.6
    inten = (lx | lz).astype(np.float32) * np.clip(1.3 - z * 0.12, 0, 1) * np.clip((Y[fl] - hz) * 9, 0, 1) * (0.7 + 0.5 * f["bass"][i])
    out[fl] = shade(0.08 + inten * 0.85)
    return np.clip(out * (0.9 + 0.2 * f["beat"][i]), 0, 1)


def bg_waves(i, f):
    X, Y, R, A = G["grid"]
    t = f["travel"][i]
    acc = np.full(X.shape, 0.06, np.float32)
    for k in range(6):
        amp = (0.12 + 0.05 * k) * (0.6 + f["bass"][i] * 0.8 if k % 2 == 0 else 0.6 + f["mid"][i] * 0.8)
        y = np.sin(X * (1.4 + k * 0.35) + t * (1 + 0.3 * k) + k) * amp + (k - 2.5) * 0.12
        acc += np.exp(-np.abs(Y - y) * (28 - k * 2)) * (0.35 + 0.1 * k)
    return shade(np.clip(acc * (0.6 + 0.5 * f["energy"][i]) + 0.1 * f["beat"][i], 0, 1))


BACKGROUNDS = dict(tunnel=bg_tunnel, plasma=bg_plasma, kaleido=bg_kaleido, starfield=bg_starfield,
                   synthgrid=bg_synthgrid, waves=bg_waves)


# ─── per-process setup ─────────────────────────────────────────────────────────
def _init(state):
    G.update(state)
    W, H, ds = G["W"], G["H"], G["ds"]
    lw, lh = W // ds, H // ds
    ys, xs = np.mgrid[0:lh, 0:lw].astype(np.float32)
    X = (xs - lw / 2) / (lh / 2)
    Y = (ys - lh / 2) / (lh / 2)
    G.update(lw=lw, lh=lh, grid=(X, Y, np.sqrt(X ** 2 + Y ** 2), np.arctan2(Y, X)))
    G["lut"] = make_lut(G["palette"])
    rng = np.random.default_rng(7)
    G["stars"] = (rng.uniform(-1.8, 1.8, 2500), rng.uniform(-1, 1, 2500), rng.uniform(0, 1, 2500))
    vig = np.clip(1.25 - (X ** 2 * 0.25 + Y ** 2 * 0.45), 0.25, 1)
    G["vignette"] = vig[..., None].astype(np.float32)
    hi = np.array([int(G["palette"][-1][k:k + 2], 16) for k in (1, 3, 5)])
    G["accent"] = tuple(int(c) for c in np.minimum(255, hi * 0.6 + 110))


def _ring(draw, cx, cy, r0, spec, scale, color, width):
    n = len(spec) * 2
    for k in range(n):
        v = spec[k if k < len(spec) else n - 1 - k]
        ang = k / n * 2 * math.pi - math.pi / 2
        r1 = r0 + 4 * scale + v * 120 * scale
        c, s = math.cos(ang), math.sin(ang)
        draw.line([(cx + c * r0, cy + s * r0), (cx + c * r1, cy + s * r1)], fill=color, width=width)


def _bars(draw, W, H, spec, scale, color):
    n = len(spec)
    bw = W * 0.6 / n
    x0 = W * 0.2
    base = H * 0.70
    for k, v in enumerate(spec):
        h = 6 * scale + v * 160 * scale
        draw.rectangle([x0 + k * bw + 1, base - h, x0 + (k + 1) * bw - 2, base], fill=color)


def render_frame(i: int) -> bytes:
    f, W, H, ds = G["f"], G["W"], G["H"], G["ds"]
    sc = H / 1080
    t = i / f["fps"]
    bg = BACKGROUNDS[G["style"]](i, f) * G["vignette"]

    # spectrum glow layer at low res
    glow = Image.new("L", (G["lw"], G["lh"]))
    gd = ImageDraw.Draw(glow)
    ring_style = G["style"] not in ("synthgrid", "waves")
    r0 = (H * 0.17 * (1 + 0.07 * f["bass"][i] + 0.05 * f["beat"][i])) / ds
    if ring_style:
        _ring(gd, G["lw"] / 2, G["lh"] / 2, r0, f["spec"][i], sc / ds, 255, max(1, int(5 * sc / ds)))
    else:
        _bars(gd, G["lw"], G["lh"], f["spec"][i], sc / ds, 255)
    glow = np.asarray(glow.filter(ImageFilter.GaussianBlur(3)), np.float32)[..., None] / 255
    bg = np.clip(bg + glow * np.array(G["accent"]) / 255 * 0.9, 0, 1)

    img = Image.fromarray((bg * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    acc = G["accent"]
    if ring_style:
        _ring(d, W / 2, H / 2, r0 * ds, f["spec"][i], sc, acc, max(2, int(4 * sc)))
        rr = r0 * ds - 6 * sc
        d.ellipse([W / 2 - rr, H / 2 - rr, W / 2 + rr, H / 2 + rr], fill=(8, 8, 14), outline=acc,
                  width=max(2, int(3 * sc)))
        _center_text(d, W / 2, H / 2 - 18 * sc, G["artist"], font("title", int(52 * sc)), (255, 255, 255))
        _center_text(d, W / 2, H / 2 + 34 * sc, G["genre"].upper(), font("body", int(22 * sc)), acc)
    else:
        _bars(d, W, H, f["spec"][i], sc, acc)

    if G.get("thumb"):
        return img.tobytes()

    # header
    pad = int(48 * sc)
    d.text((pad, pad), G["title"], font=font("title", int(54 * sc)), fill=(255, 255, 255),
           stroke_width=int(3 * sc), stroke_fill=(0, 0, 0))
    d.text((pad, pad + int(70 * sc)), f"{G['artist']}  •  {G['genre']}", font=font("body", int(26 * sc)),
           fill=acc, stroke_width=int(2 * sc), stroke_fill=(0, 0, 0))

    # intro title card
    if t < 5:
        a = min(1, t / 0.8) * min(1, (5 - t) / 0.8)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, int(150 * a)))
        od = ImageDraw.Draw(ov)
        _center_text(od, W / 2, H / 2 - 40 * sc, G["title"], font("title", int(120 * sc)),
                     (255, 255, 255, int(255 * a)), W * 0.9)
        _center_text(od, W / 2, H / 2 + 70 * sc, f"{G['artist']} — {G['genre']}", font("body", int(36 * sc)),
                     acc + (int(255 * a),))
        img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
        d = ImageDraw.Draw(img)

    # karaoke lyrics
    if G["timings"]:
        _lyrics(d, img, t, W, H, sc, acc)

    # progress bar
    prog = t / f["duration"]
    d.rectangle([0, H - int(6 * sc), int(W * prog), H], fill=acc)
    return img.tobytes()


def _center_text(d, cx, cy, text, fnt, fill, max_w=None):
    if max_w:
        while d.textlength(text, font=fnt) > max_w and fnt.size > 20:
            fnt = ImageFont.truetype(fnt.path, int(fnt.size * 0.9))
    d.text((cx, cy), text, font=fnt, fill=fill, anchor="mm", stroke_width=max(2, fnt.size // 22),
           stroke_fill=(0, 0, 0))


def _lyrics(d, img, t, W, H, sc, acc):
    tl = G["timings"]
    cur = None
    for k, line in enumerate(tl):
        nxt_start = tl[k + 1]["start"] if k + 1 < len(tl) else 1e9
        if line["start"] - 0.25 <= t < min(nxt_start - 0.05, line["end"] + 2.0):
            cur = k
            break
    if cur is None:
        return
    line = tl[cur]
    words = line["text"].split()
    size = int(64 * sc)
    fnt = font("title", size)
    while d.textlength(line["text"], font=fnt) > W * 0.88 and size > 24:
        size = int(size * 0.92)
        fnt = font("title", size)
    y = H * (0.86 if G["style"] in ("synthgrid", "waves") else 0.80)
    total_w = d.textlength(line["text"], font=fnt)
    x = W / 2 - total_w / 2
    prog = (t - line["start"]) / max(0.3, line["end"] - line["start"])
    lit = int(len(words) * min(1, max(0, prog)) + 0.999) if prog > 0 else 0
    for wi, w in enumerate(words):
        fill = (255, 255, 255) if wi < lit else (170, 170, 185)
        d.text((x, y), w, font=fnt, fill=fill, anchor="ls", stroke_width=max(2, int(4 * sc)),
               stroke_fill=(0, 0, 0))
        if wi < lit:
            ww = d.textlength(w, font=fnt)
            d.rectangle([x, y + 8 * sc, x + ww, y + 12 * sc], fill=acc)
        x += d.textlength(w + " ", font=fnt)
    if cur + 1 < len(tl):
        _center_text(d, W / 2, y + 52 * sc, tl[cur + 1]["text"], font("body", int(30 * sc)), (200, 200, 210),
                     W * 0.85)


# ─── public API ────────────────────────────────────────────────────────────────
def _state(spec, cfg, audio, timings):
    v = cfg["video"]
    return dict(W=v["width"], H=v["height"], ds=3 if v["height"] >= 1000 else 2,
                f=analyze(audio, v["fps"]), palette=spec["palette"], style=spec["visual_style"],
                title=spec["title"], artist=cfg["channel"]["artist"], genre=spec["genre"],
                timings=timings if v.get("lyrics_captions", True) else None)


def render(spec: dict, cfg: dict, audio: str, timings, out_path: str, workers: int | None = None) -> str:
    st = _state(spec, cfg, audio, timings)
    W, H, fps, n = st["W"], st["H"], cfg["video"]["fps"], st["f"]["n"]
    print(f"🎬 rendering {n} frames ({st['f']['duration']:.0f}s) style={st['style']} {W}x{H}")
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-i", audio, "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-preset", "fast", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "256k", "-shortest", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE)
    with Pool(workers or os.cpu_count(), initializer=_init, initargs=(st,)) as pool:
        for k, frame in enumerate(pool.imap(render_frame, range(n), chunksize=6)):
            ff.stdin.write(frame)
            if k % (fps * 15) == 0:
                print(f"   {k / n:5.1%}", flush=True)
    ff.stdin.close()
    if ff.wait() != 0:
        raise RuntimeError("ffmpeg encoding failed")
    return out_path


def thumbnail(spec: dict, cfg: dict, audio: str, out_path: str) -> str:
    st = _state(spec, cfg, audio, None)
    st.update(W=1280, H=720, ds=2, thumb=True)
    _init(st)
    f = st["f"]
    i = int(np.argmax(np.convolve(f["energy"], np.ones(30), mode="same")[int(6 * f["fps"]):])) + int(6 * f["fps"])
    img = Image.frombytes("RGB", (1280, 720), render_frame(i))
    # repaint header area with big title
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    od.rectangle([0, 520, 1280, 720], fill=(0, 0, 0, 170))
    img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    d = ImageDraw.Draw(img)
    _center_text(d, 640, 600, spec["title"].upper(), font("title", 110), (255, 255, 255), 1200)
    _center_text(d, 640, 680, f"{cfg['channel']['artist']} • {spec['genre'].upper()}", font("body", 30),
                 G["accent"])
    img.save(out_path, quality=92)
    return out_path
