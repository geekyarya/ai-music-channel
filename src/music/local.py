"""Zero-dependency fallback: synthesizes a full beat on the CPU and lays TTS vocals
(rap-style, locked to the bar grid) over it. Always works on a GitHub runner.

Quality is below ACE-Step (the voice raps/speaks rather than truly sings), but it
guarantees a daily upload even when free GPU quota is gone — and because we place
every line ourselves, it returns exact lyric timings for karaoke captions.
"""
from __future__ import annotations

import asyncio
import math
import os
import random
import re
import subprocess
import tempfile

import numpy as np
from scipy.signal import butter, fftconvolve, sosfilt

SR = 44100
NOTE = {"c": 0, "c#": 1, "db": 1, "d": 2, "d#": 3, "eb": 3, "e": 4, "f": 5, "f#": 6, "gb": 6,
        "g": 7, "g#": 8, "ab": 8, "a": 9, "a#": 10, "bb": 10, "b": 11}

VOICES = {
    ("en", "male"): ["en-US-GuyNeural", "en-US-ChristopherNeural", "en-US-AndrewNeural", "en-US-EricNeural"],
    ("en", "female"): ["en-US-AriaNeural", "en-US-JennyNeural", "en-US-AvaNeural", "en-US-MichelleNeural"],
    ("hi", "male"): ["hi-IN-MadhurNeural"],
    ("hi", "female"): ["hi-IN-SwaraNeural"],
}
ESPEAK = {("en", "male"): "en-us+m3", ("en", "female"): "en-us+f3",
          ("hi", "male"): "hi+m3", ("hi", "female"): "hi+f3"}
DESI = ("desi", "bollywood", "sufi", "gully", "indian")


# ─── genre → groove ────────────────────────────────────────────────────────────
def family(genre: str) -> str:
    g = genre.lower()
    if "romantic" in g or "sufi" in g:
        return "lofi" if "romantic" in g else "rock"
    if "gully" in g:
        return "hiphop"
    if any(k in g for k in ("trap", "drill")):
        return "trap"
    if any(k in g for k in ("lo-fi", "lofi", "r&b", "soul", "jazz")):
        return "lofi"
    if any(k in g for k in ("hip", "rap", "boom")):
        return "hiphop"
    if any(k in g for k in ("rock", "punk", "metal", "country")):
        return "rock"
    return "four"  # pop, edm, synthwave, afrobeats, house…


PATTERNS = {  # 16 steps per bar
    "hiphop": dict(kick=[0, 7, 10], snare=[4, 12], hat=list(range(0, 16, 2)), ohat=[14], swing=0.18),
    "lofi":   dict(kick=[0, 9], snare=[4, 12], hat=list(range(0, 16, 2)), ohat=[], swing=0.28),
    "trap":   dict(kick=[0, 11], snare=[8], hat=list(range(16)), ohat=[], swing=0.0),
    "rock":   dict(kick=[0, 6, 8, 10], snare=[4, 12], hat=list(range(0, 16, 2)), ohat=[], swing=0.0),
    "four":   dict(kick=[0, 4, 8, 12], snare=[4, 12], hat=[], ohat=[2, 6, 10, 14], swing=0.0),
}
RAP_FAMILIES = {"hiphop", "trap"}


# ─── synth helpers ─────────────────────────────────────────────────────────────
def t_(dur):
    return np.arange(int(dur * SR)) / SR


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def lp(x, hz, order=2):
    return sosfilt(butter(order, min(hz, SR / 2 - 100), "low", fs=SR, output="sos"), x)


def hp(x, hz, order=2):
    return sosfilt(butter(order, hz, "high", fs=SR, output="sos"), x)


def bp(x, lo, hi):
    return sosfilt(butter(2, [lo, hi], "band", fs=SR, output="sos"), x)


def kick():
    t = t_(0.45)
    f = 45 + 110 * np.exp(-t * 28)
    return np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t * 7) * 1.0


def snare():
    t = t_(0.25)
    n = bp(np.random.randn(len(t)), 1200, 8000) * np.exp(-t * 18)
    return 0.55 * n + 0.35 * np.sin(2 * np.pi * 185 * t) * np.exp(-t * 25)


def clap():
    t = t_(0.3)
    env = sum(np.exp(-np.clip(t - d, 0, None) * 60) * (t >= d) for d in (0, 0.011, 0.022)) / 3
    return bp(np.random.randn(len(t)), 900, 5000) * (env + 0.4 * np.exp(-t * 12)) * 0.8


def tabla(pitch=1.0, open_=True):
    """Tabla-ish stroke: pitched membrane with a fast upward bend (open 'na') or a dry slap."""
    t = t_(0.35 if open_ else 0.12)
    f0 = (330 if open_ else 210) * pitch
    f = f0 * (1 + 0.25 * np.exp(-t * 30))
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR) + 0.35 * np.sin(2 * np.pi * np.cumsum(f * 2.1) / SR)
    click = hp(np.random.randn(len(t)), 2500) * np.exp(-t * 120) * 0.3
    return (tone * np.exp(-t * (9 if open_ else 30)) + click) * 0.45


def hat(open_=False):
    t = t_(0.3 if open_ else 0.06)
    return hp(np.random.randn(len(t)), 7000, 4) * np.exp(-t * (12 if open_ else 70)) * 0.35


def sub808(freq, dur, glide_from=None):
    t = t_(dur)
    f = np.full_like(t, freq)
    if glide_from:
        f = freq + (glide_from - freq) * np.exp(-t * 18)
    x = np.sin(2 * np.pi * np.cumsum(f) / SR)
    return np.tanh(2.2 * x) * np.minimum(1, t * 200) * np.exp(-t * 1.1) * 0.7


def saw(freq, dur, detune=0.0):
    t = t_(dur)
    return 2 * ((t * freq * (1 + detune)) % 1) - 1


def adsr(n, a=0.01, r=0.1):
    env = np.ones(n)
    na, nr = int(a * SR), int(r * SR)
    if na:
        env[:na] = np.linspace(0, 1, na)
    if nr and nr < n:
        env[-nr:] *= np.linspace(1, 0, nr)
    return env


def pad_chord(notes, dur, bright=1800):
    x = sum(saw(midi_hz(m), dur, d) for m in notes for d in (-0.004, 0.004)) / (2 * len(notes))
    return lp(x, bright) * adsr(len(x), 0.25, 0.4)


def keys_chord(notes, dur):  # electric-piano-ish for lo-fi / r&b
    t = t_(dur)
    x = sum(np.sin(2 * np.pi * midi_hz(m) * t) + 0.3 * np.sin(4 * np.pi * midi_hz(m) * t) for m in notes)
    return x / len(notes) * np.exp(-t * 1.2) * (1 + 0.15 * np.sin(2 * np.pi * 5 * t)) * adsr(len(t), 0.005, 0.2)


def pluck(m, dur):
    t = t_(dur)
    return (np.sign(np.sin(2 * np.pi * midi_hz(m) * t)) * 0.4 + np.sin(2 * np.pi * midi_hz(m) * t)) \
        * np.exp(-t * 9) * 0.5


def power_chord(root, dur):
    x = sum(saw(midi_hz(m), dur, d) for m in (root, root + 7, root + 12) for d in (-0.003, 0.003))
    return np.tanh(3 * lp(x, 3500)) * adsr(int(dur * SR), 0.005, 0.05) * 0.5


def reverb(x, seconds=1.6, mix=0.25):
    t = t_(seconds)
    ir_l = np.random.randn(len(t)) * np.exp(-t * 4.5 / seconds)
    ir_r = np.random.randn(len(t)) * np.exp(-t * 4.5 / seconds)
    ir_l, ir_r = lp(ir_l, 6000), lp(ir_r, 6000)
    ir_l /= np.abs(ir_l).sum() ** 0.5 * 6
    ir_r /= np.abs(ir_r).sum() ** 0.5 * 6
    mono = x.mean(axis=1) if x.ndim == 2 else x
    wet = np.stack([fftconvolve(mono, ir_l)[: len(mono)], fftconvolve(mono, ir_r)[: len(mono)]], axis=1)
    dry = x if x.ndim == 2 else np.stack([x, x], axis=1)
    return dry * (1 - mix) + wet * mix


def place(buf, sig, start_s, gain=1.0, pan=0.0):
    i = int(start_s * SR)
    if i >= len(buf):
        return
    sig = sig[: len(buf) - i]
    l, r = math.cos((pan + 1) * math.pi / 4), math.sin((pan + 1) * math.pi / 4)
    buf[i:i + len(sig), 0] += sig * gain * l * 1.414
    buf[i:i + len(sig), 1] += sig * gain * r * 1.414


# ─── harmony ───────────────────────────────────────────────────────────────────
def progression(key: str):
    m = re.match(r"\s*([A-Ga-g][#b]?)\s*(\w*)", key or "A minor")
    root = NOTE.get(m.group(1).lower(), 9) if m else 9
    minor = not (m and m.group(2).lower().startswith("maj"))
    base = 48 + root  # C3-ish
    if minor:  # i – VI – III – VII
        degs = [(0, "m"), (8, "M"), (3, "M"), (10, "M")]
    else:      # I – V – vi – IV
        degs = [(0, "M"), (7, "M"), (9, "m"), (5, "M")]
    chords = []
    for d, q in degs:
        r = base + d
        r = r - 12 if r > 57 else r
        chords.append([r, r + (3 if q == "m" else 4), r + 7])
    return chords


# ─── vocals ────────────────────────────────────────────────────────────────────
def _decode(path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1",
                          "-ar", str(SR), "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32).copy()
    # trim silence
    nz = np.where(np.abs(x) > 0.02)[0]
    return x[max(0, nz[0] - 200): nz[-1] + 2000] if len(nz) else x


def _tts_edge(lines, voice, rate, tmp):
    import edge_tts

    async def one(i, text, sem):
        async with sem:
            p = os.path.join(tmp, f"v{i}.mp3")
            for attempt in range(4):
                try:
                    await edge_tts.Communicate(text, voice, rate=rate).save(p)
                    return p
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(2 + attempt * 3)
            raise RuntimeError(f"edge-tts failed for line {i}")

    async def run():
        sem = asyncio.Semaphore(4)
        return await asyncio.gather(*(one(i, t, sem) for i, t in enumerate(lines)))

    return asyncio.run(run())


def _tts_espeak(lines, vocal, tmp, lang="en"):
    paths = []
    v = ESPEAK[(lang, vocal)]
    for i, text in enumerate(lines):
        p = os.path.join(tmp, f"v{i}.wav")
        subprocess.run(["espeak-ng", "-v", v, "-s", "165", "-p", "45", "-w", p, text], check=True)
        paths.append(p)
    return paths


def synth_vocals(lines, vocal, fam, tts_pref, tmp, lang="en"):
    rate = "+12%" if fam in RAP_FAMILIES else "-4%"
    paths, engine = None, None
    if tts_pref == "edge":
        try:
            voice = random.choice(VOICES[(lang, vocal)])
            paths, engine = _tts_edge(lines, voice, rate, tmp), f"edge-tts {voice}"
        except Exception as e:  # noqa: BLE001
            print(f"  ! edge-tts unavailable ({type(e).__name__}: {str(e)[:120]}), using espeak-ng")
    if paths is None:
        paths, engine = _tts_espeak(lines, vocal, tmp, lang), "espeak-ng"
    return [_decode(p) for p in paths], engine


def stretch(x, factor):
    """Speed up/slow down speech without changing pitch (ffmpeg atempo)."""
    if abs(factor - 1) < 0.02:
        return x
    p = subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-f", "f32le", "-ar", str(SR), "-ac", "1", "-i", "-",
         "-filter:a", f"atempo={factor:.3f}", "-f", "f32le", "-"],
        input=x.astype(np.float32).tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


# ─── main entry ────────────────────────────────────────────────────────────────
def generate(spec: dict, cfg: dict, out_dir: str) -> dict:
    fam = family(spec["genre"])
    pat = PATTERNS[fam]
    bpm = spec["bpm"]
    if fam == "trap" and bpm < 120:
        bpm *= 2  # trap is written double-time with a half-time snare
    if fam != "trap" and bpm > 135:
        bpm //= 2
    beat = 60 / bpm
    bar = 4 * beat
    step = bar / 16

    # 1) vocals first — the arrangement is laid out around them
    lang = "hi" if spec.get("language") in ("hi", "hinglish") else "en"
    lines, spoken, owners = [], [], []
    for si, s in enumerate(spec["sections"]):
        for k, l in enumerate(s["lines"]):
            lines.append(l)
            spoken.append(s.get("native", s["lines"])[k] if lang == "hi" else l)  # Devanagari → correct accent
            owners.append(si)
    with tempfile.TemporaryDirectory() as tmp:
        vox, tts_engine = synth_vocals(spoken, spec["vocal"], fam, cfg["music"]["local"].get("tts", "edge"),
                                       tmp, lang)

    rap = fam in RAP_FAMILIES
    base_bars = 1 if (rap or bar > 3.2) else 2
    timeline, cur_bar = [], 4  # 4-bar intro
    section_bars = []  # (start_bar, end_bar, type)
    prev_sec = None
    for i, (x, si) in enumerate(zip(vox, owners)):
        if si != prev_sec:
            if prev_sec is not None:
                cur_bar += 1  # breath between sections
            section_bars.append([cur_bar, None, spec["sections"][si]["type"]])
            prev_sec = si
        dur = len(x) / SR
        nbars = base_bars
        slot = nbars * bar * 0.94
        if dur > slot:
            f = min(dur / slot, 1.3)
            x = stretch(x, f)
            dur = len(x) / SR
            while dur > nbars * bar * 0.97:
                nbars += 1
        lead = 0 if rap else beat * 0.5
        timeline.append({"start": cur_bar * bar + lead, "audio": x, "section": spec["sections"][si]["type"],
                         "text": lines[i], "nbars": nbars})
        cur_bar += nbars
        section_bars[-1][1] = cur_bar
    total_bars = cur_bar + 4  # outro
    total = total_bars * bar + 2.5
    n = int(total * SR)

    chords = progression(spec.get("key", "A minor"))
    drums = np.zeros((n, 2))
    music = np.zeros((n, 2))
    vbus = np.zeros((n, 2))
    K, S, C, H, OH = kick(), snare(), clap(), hat(), hat(True)
    desi = any(k in spec["genre"].lower() for k in DESI)
    TA, TB, TC = tabla(1.0), tabla(1.33), tabla(1.0, open_=False)
    tabla_pat = {0: TC, 3: TA, 6: TB, 8: TC, 10: TA, 11: TA, 14: TB}
    sec_of_bar = {}
    for a, b, typ in section_bars:
        for k in range(a, b):
            sec_of_bar[k] = typ

    for b in range(total_bars):
        t0 = b * bar
        typ = sec_of_bar.get(b, "intro" if b < 4 else ("outro" if b >= total_bars - 4 else "gap"))
        hot = typ in ("chorus", "drop")
        chord = chords[b % 4]
        intro_outro = typ in ("intro", "outro")

        # drums
        if not intro_outro or (typ == "intro" and b >= 2):
            for s16 in range(16):
                ts = t0 + s16 * step + (pat["swing"] * step if s16 % 2 else 0)
                if s16 in pat["kick"] and not (typ == "intro"):
                    place(drums, K, ts, 0.95)
                if s16 in pat["snare"] and not intro_outro:
                    place(drums, S if fam in ("hiphop", "lofi", "rock") else C, ts, 0.6)
                if s16 in pat["hat"]:
                    g = 0.5 if s16 % 4 == 0 else 0.32
                    place(drums, H, ts, g * random.uniform(0.8, 1.1), pan=0.25)
                    if fam == "trap" and hot and s16 in (13, 14, 15) and b % 2:  # hat rolls
                        place(drums, H, ts + step / 2, 0.25, pan=0.25)
                if s16 in pat["ohat"]:
                    place(drums, OH, ts, 0.35, pan=-0.2)
                if desi and s16 in tabla_pat and not (typ == "intro" and b < 2):
                    place(drums, tabla_pat[s16], ts, 0.55 if hot else 0.4, pan=-0.35)
                if fam == "four" and hot and s16 % 2 == 0 and s16 not in pat["ohat"]:
                    place(drums, H, ts, 0.18, pan=0.3)

        # bass
        root = chord[0] - 12
        if fam == "trap":
            place(music, sub808(midi_hz(root - 12), bar * 0.7), t0, 0.9)
            place(music, sub808(midi_hz(root - 12), bar * 0.3, glide_from=midi_hz(root)), t0 + 11 * step, 0.7)
        elif not intro_outro:
            hits = [0, 6, 10] if fam in ("hiphop", "lofi") else list(range(0, 16, 2))
            for h in hits:
                note = sub808(midi_hz(root), step * (3 if fam in ("hiphop", "lofi") else 1.8))
                place(music, lp(note, 400), t0 + h * step, 0.7)

        # harmony
        if fam == "lofi":
            place(music, keys_chord([m + 12 for m in chord], bar), t0, 0.35, pan=-0.1)
        elif fam == "rock" and not intro_outro:
            for h in range(0, 16, 2):
                place(music, power_chord(chord[0], step * 1.9), t0 + h * step, 0.22 if hot else 0.16, pan=-0.3)
        else:
            place(music, pad_chord([m + 12 for m in chord], bar, 2600 if hot else 1500), t0, 0.32)

        # arp / melody hook (chorus & intro)
        if hot or typ == "intro":
            arp = [chord[0] + 24, chord[1] + 24, chord[2] + 24, chord[1] + 24]
            for k in range(8):
                place(music, pluck(arp[k % 4], step * 2), t0 + k * 2 * step, 0.16, pan=0.35 if k % 2 else -0.35)

    # vocals (chorus doubled for width)
    for ev in timeline:
        x = hp(ev["audio"], 120)
        place(vbus, x, ev["start"], 1.0)
        if ev["section"] == "chorus":
            place(vbus, x, ev["start"] + 0.018, 0.45, pan=-0.6)
            place(vbus, x, ev["start"] + 0.031, 0.45, pan=0.6)
    vbus = reverb(vbus, 1.2 if rap else 2.0, 0.15 if rap else 0.28)
    music = reverb(music, 2.2, 0.3)

    # duck music under vocals a little; sidechain pad to kick for "four"
    venv = np.convolve(np.abs(vbus.mean(axis=1)), np.ones(4410) / 4410, mode="same")
    duck = 1 - 0.35 * np.clip(venv / (venv.max() + 1e-9) * 4, 0, 1)
    mix = drums * 0.8 + music * duck[:, None] * 0.8 + vbus * 1.25
    if fam == "lofi":
        mix = np.stack([lp(mix[:, 0], 7000), lp(mix[:, 1], 7000)], axis=1)
        crackle = (np.random.rand(n) > 0.9993) * np.random.randn(n) * 0.15 + np.random.randn(n) * 0.004
        mix += crackle[:, None]
    # master: soft clip + normalise, fade out
    mix = np.tanh(mix / (np.percentile(np.abs(mix), 99.5) + 1e-9) * 0.9)
    fade = int(3 * SR)
    mix[-fade:] *= np.linspace(1, 0, fade)[:, None]
    mix = mix / np.abs(mix).max() * 0.95

    out = os.path.join(out_dir, "song.wav")
    _write_wav(out, mix)
    timings = [{"start": round(e["start"], 3), "end": round(e["start"] + len(e["audio"]) / SR, 3),
                "text": e["text"], "section": e["section"]} for e in timeline]
    return {"audio": out, "timings": timings, "engine": f"local synth ({fam}, {bpm} BPM) + {tts_engine}"}


def _write_wav(path, x):
    from scipy.io import wavfile
    wavfile.write(path, SR, (np.clip(x, -1, 1) * 32767).astype(np.int16))
