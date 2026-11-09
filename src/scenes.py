"""Lyric-driven animated scenes layered over the music-video background.

Each sung line is mapped to a scene from what it's about ("rain", "aag", "shehar", "dil"…);
scenes crossfade at line boundaries and every scene still reacts to the audio.
All scenes are pure functions of (time, audio features) so frames render in parallel.
"""
from __future__ import annotations

import math
import re

import numpy as np
from scipy.ndimage import gaussian_filter

# keyword → scene. Roman-script Hindi/Hinglish words included (lyrics are displayed in Roman).
KEYWORDS = {
    "storm": ["storm", "thunder", "lightning", "toofan", "toofaan", "tufaan", "bijli", "aandhi", "chaos", "war"],
    "rain": ["rain", "raining", "tears", "cry", "crying", "baarish", "barish", "baarishein", "aansu", "aansoo",
             "monsoon", "boondein", "badal", "clouds", "sawan", "saawan"],
    "fire": ["fire", "flame", "flames", "burn", "burning", "blaze", "aag", "jalta", "jale", "jalna", "jal",
             "sholay", "angaar", "ember", "embers", "heat", "lava", "chingari"],
    "city": ["city", "street", "streets", "town", "concrete", "building", "skyline", "shehar", "sheher", "gali",
             "galiyon", "sadak", "sadkon", "mumbai", "delhi", "bazaar", "traffic", "metro", "neon"],
    "stars": ["star", "stars", "sky", "moon", "night", "galaxy", "space", "sitara", "sitaare", "taare", "tara",
              "chaand", "chand", "aasmaan", "asmaan", "raat", "falak", "midnight", "satellite"],
    "hearts": ["heart", "hearts", "love", "kiss", "darling", "baby", "dil", "pyaar", "pyar", "ishq", "mohabbat",
               "jaan", "sanam", "dhadkan", "dhadke", "jaana", "mehboob"],
    "petals": ["flower", "flowers", "rose", "roses", "bloom", "spring", "garden", "phool", "gulab", "kaliyan",
               "bahaar", "khushboo", "petal", "petals"],
    "snow": ["snow", "winter", "cold", "ice", "frozen", "december", "barf", "sardi", "thand", "freeze"],
    "ocean": ["ocean", "sea", "wave", "waves", "tide", "river", "water", "samundar", "samandar", "sagar", "lehren",
              "lehron", "dariya", "nadi", "paani", "beach", "shore", "kinara"],
    "sunrise": ["sun", "sunrise", "morning", "dawn", "light", "shine", "golden", "bright", "suraj", "subah",
                "roshni", "ujala", "savera", "dhoop", "kiran", "noor"],
}
SCENES = list(KEYWORDS) + ["burst"]


def scene_for_line(text: str, section: str) -> str | None:
    words = set(re.findall(r"[a-z']+", text.lower()))
    for scene, keys in KEYWORDS.items():  # dict order = priority (storm before rain, etc.)
        if words & set(keys):
            return scene
    if section in ("chorus", "hook", "drop"):
        return "burst"
    return None


def build_timeline(timings, duration: float, default: str) -> list[tuple[float, float, str]]:
    """[(start, end, scene)] covering the song. Lines without a keyword keep the previous scene,
    so the picture doesn't flicker; choruses burst."""
    if not timings:
        return [(0.0, duration, default)]
    tl, cur = [], default
    for k, line in enumerate(timings):
        s = scene_for_line(line["text"], line.get("section", "verse"))
        if s:
            cur = s
        elif line.get("section") not in ("chorus", "hook") and cur == "burst":
            cur = default
        start = max(0.0, line["start"] - 0.3)
        end = timings[k + 1]["start"] - 0.3 if k + 1 < len(timings) else duration
        if tl and tl[-1][2] == cur:
            tl[-1] = (tl[-1][0], end, cur)
        else:
            tl.append((start if tl else 0.0, end, cur))
    tl[-1] = (tl[-1][0], duration, tl[-1][2])
    return tl


def default_scene(spec: dict) -> str:
    text = " ".join(l for s in spec["sections"] for l in s["lines"]) + " " + spec.get("theme", "")
    counts = {sc: len(set(re.findall(r"[a-z']+", text.lower())) & set(k)) for sc, k in KEYWORDS.items()}
    best = max(counts, key=counts.get)
    return best if counts[best] else "stars"


def scene_weights(timeline, t: float, fade: float = 0.6) -> list[tuple[str, float]]:
    """Active scene(s) at time t with crossfade weights."""
    for k, (s, e, name) in enumerate(timeline):
        if s <= t < e or k == len(timeline) - 1:
            a = min(1.0, (t - s) / fade) if k > 0 else 1.0
            if a < 1.0:
                return [(timeline[k - 1][2], 1 - a), (name, a)]
            return [(name, 1.0)]
    return [(timeline[-1][2], 1.0)]


# ─── scene renderers ───────────────────────────────────────────────────────────
# Each returns (add_rgb, dim): light added on top (float, 0..~1) and a 0..1 multiplier for the
# background underneath (e.g. city silhouettes darken it).
_RNG = np.random.default_rng(11)
P = {k: _RNG.random((600, 4)) for k in ("rain", "fire", "stars", "snow", "petals", "spark")}
HEARTS = _RNG.random((14, 3))
BUILDINGS = None


def _dots(h, w, xs, ys, vals, sigma=1.0):
    img = np.zeros((h, w), np.float32)
    xi, yi = xs.astype(int), ys.astype(int)
    ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
    np.add.at(img, (yi[ok], xi[ok]), vals[ok] if np.ndim(vals) else vals)
    return gaussian_filter(img, sigma) if sigma else img


def _tint(mask, rgb):
    return mask[..., None] * np.array(rgb, np.float32)[None, None]


def sc_rain(h, w, t, f, i, heavy=1.0):
    p = P["rain"][: int(420 * heavy)]
    speed = 1.6 * h
    xs = (p[:, 0] * w * 1.2 - t * 0.12 * h) % w
    ys = (p[:, 1] * h + t * speed * (0.7 + 0.6 * p[:, 2])) % h
    img = np.zeros((h, w), np.float32)
    for k in range(24):  # streaks
        img += _dots(h, w, xs + k * 0.15, ys - k * h * 0.006, 0.6 * (1 - k / 24), sigma=0)
    img = gaussian_filter(img, 0.7) * (1.6 + 0.8 * f["high"][i])
    return _tint(img, (0.75, 0.85, 1.0)), 0.6


def sc_storm(h, w, t, f, i):
    add, _ = sc_rain(h, w, t, f, i, heavy=1.4)
    flash = 0.9 if f["beat"][i] > 0.95 and (int(t * 7) % 3 == 0) else 0.0
    if flash:
        X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(0, 1, h))
        bolt_x = math.sin(t * 13.7) * 0.6
        path = bolt_x + 0.15 * np.sin(Y * 23 + t * 40) * Y
        bolt = np.exp(-np.abs(X - path) * 90) * (Y < 0.85)
        add = add + _tint(bolt * 1.5 + flash * 0.25, (0.85, 0.9, 1.0))
    return add, 0.45


def sc_fire(h, w, t, f, i):
    p = P["fire"][:320]
    life = (p[:, 1] + t * (0.25 + 0.3 * p[:, 2])) % 1.0
    xs = p[:, 0] * w + np.sin(t * 2 + p[:, 3] * 20) * w * 0.03 * life
    ys = h * (1.02 - life * 0.95)
    emb = _dots(h, w, xs, ys, (1 - life) * 4.5 * (0.7 + 0.6 * f["bass"][i]), sigma=1.5)
    emb += gaussian_filter(emb, 5) * 2.5
    Y = np.linspace(0, 1, h)[:, None]
    glow = np.clip((Y - 0.45) / 0.55, 0, 1) ** 2 * (0.6 + 0.45 * f["bass"][i]) * np.ones((1, w))
    flick = 0.85 + 0.15 * np.sin(np.linspace(0, 30, w)[None, :] + t * 9)
    return _tint(emb, (1.0, 0.55, 0.15)) + _tint(glow * flick, (1.0, 0.3, 0.05)), 0.65


def _skyline(h, w):
    global BUILDINGS
    if BUILDINGS is None or BUILDINGS[0].shape != (h, w):
        rng = np.random.default_rng(5)
        heights = np.zeros(w)
        x = 0
        while x < w:
            bw = int(rng.integers(w // 40, w // 14))
            heights[x:x + bw] = rng.uniform(0.18, 0.55)
            x += bw + int(rng.integers(0, 3))
        Y = np.linspace(1, 0, h)[:, None]
        mask = (Y < heights[None, :]).astype(np.float32)
        win = ((np.arange(h)[:, None] % 6 < 2) & (np.arange(w)[None, :] % 5 < 2)).astype(np.float32) * mask
        win *= (Y < heights[None, :] - 0.02)
        seed = rng.random((h, w)).astype(np.float32)
        BUILDINGS = (mask, win, seed)
    return BUILDINGS


def sc_city(h, w, t, f, i):
    mask, win, seed = _skyline(h, w)
    lit = (seed > 0.55 - 0.25 * f["mid"][i]) & ((seed * 100 + int(t * 2)) % 7 > 1)
    windows = gaussian_filter(win * lit, 0.5) * 1.4
    dim = 1 - 0.88 * mask
    return _tint(windows, (1.0, 0.85, 0.45)), dim


def sc_stars(h, w, t, f, i):
    p = P["stars"][:380]
    tw = 0.5 + 0.5 * np.sin(t * (1 + 3 * p[:, 2]) + p[:, 3] * 40)
    img = _dots(h, w, p[:, 0] * w, p[:, 1] * h * 0.9, tw * (1.5 + 1.5 * f["high"][i]) * (p[:, 2] > 0.3), 0.9)
    img += gaussian_filter(img, 3) * 1.5
    period = 3.0
    ph = (t % period) / period
    if ph < 0.35:  # shooting star
        k = int(t // period)
        sx, sy = (0.2 + 0.6 * ((k * 0.37) % 1)) * w, (0.1 + 0.2 * ((k * 0.61) % 1)) * h
        tt = ph / 0.35
        xs = sx + np.linspace(0, 1, 30) * w * 0.25 * tt
        ys = sy + np.linspace(0, 1, 30) * h * 0.12 * tt
        img += _dots(h, w, xs, ys, np.linspace(0.1, 1.6, 30) * (1 - tt), 0.8)
    return _tint(img * 2.2, (1.0, 0.97, 0.9)), 0.55


def _heart_sprite(n):
    y, x = np.mgrid[-1.2:1.2:n * 1j, -1.2:1.2:n * 1j]
    y = -y
    return (((x ** 2 + y ** 2 - 1) ** 3 - x ** 2 * y ** 3) < 0).astype(np.float32)


def sc_hearts(h, w, t, f, i):
    img = np.zeros((h, w), np.float32)
    for k, (a, b, c) in enumerate(HEARTS):
        size = int(h * (0.05 + 0.07 * c) * (1 + 0.15 * f["beat"][i]))
        spr = _heart_sprite(max(size, 6))
        life = (b + t * (0.08 + 0.06 * c)) % 1.0
        x = int(a * w + math.sin(t * 1.3 + k) * w * 0.03)
        y = int(h * (1.1 - life * 1.25))
        y0, x0 = y - size // 2, x - size // 2
        ys, xs = slice(max(0, y0), min(h, y0 + size)), slice(max(0, x0), min(w, x0 + size))
        sy, sx = slice(ys.start - y0, ys.stop - y0), slice(xs.start - x0, xs.stop - x0)
        if ys.stop > ys.start and xs.stop > xs.start:
            img[ys, xs] += spr[sy, sx] * (0.55 * math.sin(life * math.pi))
    img = gaussian_filter(img, 1.2) + gaussian_filter(img, 6) * 0.8
    return _tint(img, (1.0, 0.3, 0.5)), 0.9


def _drift(h, w, t, p, fall, sway, size_v):
    xs = (p[:, 0] * w + np.sin(t * (0.6 + p[:, 2]) + p[:, 3] * 9) * w * sway + t * w * 0.02) % w
    ys = (p[:, 1] * h + t * h * fall * (0.5 + p[:, 2])) % h
    return xs, ys, size_v


def sc_snow(h, w, t, f, i):
    p = P["snow"][:350]
    xs, ys, _ = _drift(h, w, t, p, 0.09, 0.03, 1)
    img = _dots(h, w, xs, ys, 1.2 + 1.5 * p[:, 2], sigma=1.4 + 0.6 * f["bass"][i])
    return _tint(img * 3.5, (0.95, 0.97, 1.0)), 0.6


def sc_petals(h, w, t, f, i):
    p = P["petals"][:200]
    xs, ys, _ = _drift(h, w, t, p, 0.12, 0.06, 1)
    img = _dots(h, w, xs, ys, 1.5 + 1.5 * p[:, 2], sigma=2.0)
    img2 = _dots(h, w, xs + 2.5, ys + 1.2, 1.2, sigma=1.8)
    return _tint(img * 3.2, (1.0, 0.45, 0.65)) + _tint(img2 * 2.5, (1.0, 0.8, 0.85)), 0.65


def sc_ocean(h, w, t, f, i):
    X = np.linspace(0, 1, w)[None, :]
    Y = np.linspace(0, 1, h)[:, None]
    add = np.zeros((h, w, 3), np.float32)
    dim = np.ones((h, w), np.float32)
    for k in range(4):
        base = 0.62 + k * 0.1
        amp = 0.025 * (1 + k * 0.4) * (1 + 0.8 * f["bass"][i])
        crest = base + amp * np.sin(X * (9 - k * 1.5) * math.pi + t * (1.2 + k * 0.4) + k)
        below = (Y > crest).astype(np.float32)
        foam = np.exp(-np.abs(Y - crest) * h * 0.15)
        dim *= 1 - below * 0.22
        add += _tint(foam * (0.5 + 0.2 * k), (0.6, 0.9, 1.0)) + _tint(below * 0.05, (0.1, 0.4, 0.8))
    return add, dim


def sc_sunrise(h, w, t, f, i):
    X, Y = np.meshgrid(np.linspace(-w / h, w / h, w), np.linspace(-1, 1, h))
    cy = 1.05
    ang = np.arctan2(Y - cy, X)
    r = np.sqrt(X ** 2 + (Y - cy) ** 2)
    rays = (0.5 + 0.5 * np.cos(ang * 18 + t * 0.4)) ** 6 * np.exp(-r * 0.6)
    glow = np.exp(-r * 1.6) * (0.8 + 0.4 * f["bass"][i])
    return _tint(rays * 0.6 + glow * 0.7, (1.0, 0.75, 0.35)), 0.95


def sc_burst(h, w, t, f, i):
    X, Y = np.meshgrid(np.linspace(-w / h, w / h, w), np.linspace(-1, 1, h))
    ang = np.arctan2(Y, X)
    r = np.sqrt(X ** 2 + Y ** 2)
    beat = f["beat"][i]
    rays = (0.5 + 0.5 * np.cos(ang * 12 - t * 1.5)) ** 8 * np.clip(r - 0.25, 0, 1) * (0.35 + 0.65 * beat)
    p = P["spark"][:260]
    life = (p[:, 0] + t * (0.5 + 0.5 * p[:, 1])) % 1.0
    a = p[:, 2] * 2 * math.pi
    dist = life * 1.6
    xs = (np.cos(a) * dist * h / 2) + w / 2
    ys = (np.sin(a) * dist * h / 2) + h / 2
    sparks = _dots(h, w, xs, ys, (1 - life) * (0.8 + beat), sigma=1.0)
    return _tint(rays * 0.55 + sparks * 1.6, (1.0, 0.95, 0.85)), 1.0


RENDER = dict(rain=sc_rain, storm=sc_storm, fire=sc_fire, city=sc_city, stars=sc_stars, hearts=sc_hearts,
              snow=sc_snow, petals=sc_petals, ocean=sc_ocean, sunrise=sc_sunrise, burst=sc_burst)


def apply(bg: np.ndarray, timeline, t: float, f: dict, i: int, accent=None, strength: float = 1.0) -> np.ndarray:
    """Composite the lyric scene(s) active at time t onto a low-res float RGB background."""
    h, w = bg.shape[:2]
    out_add = np.zeros_like(bg)
    out_dim = np.ones((h, w), np.float32)
    for name, wt in scene_weights(timeline, t):
        add, dim = RENDER[name](h, w, t, f, i)
        out_add += add * wt
        dim = np.asarray(dim, np.float32)
        out_dim *= 1 - (1 - dim) * wt
    if accent is not None:  # nudge scene colours toward the song's palette so it stays coherent
        out_add = out_add * 0.8 + out_add.mean(axis=2, keepdims=True) * np.array(accent) / 255 * 0.25
    dim = out_dim[..., None] if np.ndim(out_dim) == 2 else out_dim
    return np.clip(bg * (1 - (1 - dim) * strength) + out_add * strength, 0, 1)
