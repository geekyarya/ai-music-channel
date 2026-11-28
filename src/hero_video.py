"""Assemble a Captain Gamchha episode: Hindi voices (free edge-tts) + shot clips/stills + captions + music bed
→ 1080x1920 MP4 for YouTube Shorts / Instagram Reels."""
from __future__ import annotations

import asyncio
import json
import os
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from music import local as synth
from video import font

W, H, FPS, SR = 1080, 1920, 30, 44100

# Two free Hindi neural voices, varied with pitch/rate per character.
VOICES = {
    "narrator":        ("hi-IN-MadhurNeural", "+6%", "-4Hz"),
    "hero":            ("hi-IN-MadhurNeural", "-4%", "-14Hz"),
    "villain":         ("hi-IN-MadhurNeural", "-8%", "+4Hz"),
    "villager_male":   ("hi-IN-MadhurNeural", "+12%", "+12Hz"),
    "villager_female": ("hi-IN-SwaraNeural", "+6%", "+0Hz"),
    "kid":             ("hi-IN-SwaraNeural", "+12%", "+45Hz"),
}
LABEL = {"hero": "CAPTAIN GAMCHHA", "villain": "LALA LAALCHI"}


def _run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{cmd[0]} failed: {p.stderr[-800:]}")
    return p.stdout


def duration(path: str) -> float:
    return float(_run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path]).strip())


# ─── voices ───────────────────────────────────────────────────────────────────
def voices(shots: list[dict], tmp: str) -> list[str | None]:
    import edge_tts

    async def one(i, s, sem):
        if not s["line"]:
            return None
        voice, rate, pitch = VOICES.get(s["speaker"], VOICES["narrator"])
        p = os.path.join(tmp, f"voice_{i}.mp3")
        async with sem:
            for attempt in range(4):
                try:
                    await edge_tts.Communicate(s["line"], voice, rate=rate, pitch=pitch).save(p)
                    return p
                except Exception:  # noqa: BLE001
                    await asyncio.sleep(2 + 3 * attempt)
        print(f"   edge-tts failed for shot {i}; using espeak-ng")
        p = os.path.join(tmp, f"voice_{i}.wav")
        subprocess.run(["espeak-ng", "-v", "hi", "-s", "150", "-w", p, s["line"]], check=True)
        return p

    async def go():
        sem = asyncio.Semaphore(3)
        return await asyncio.gather(*(one(i, s, sem) for i, s in enumerate(shots)))

    return asyncio.run(go())


def _decode(path: str) -> np.ndarray:
    raw = _run_bytes(["ffmpeg", "-loglevel", "error", "-i", path, "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"])
    return np.frombuffer(raw, dtype=np.float32).copy()


def _run_bytes(cmd) -> bytes:
    return subprocess.run(cmd, capture_output=True, check=True).stdout


# ─── captions ─────────────────────────────────────────────────────────────────
def _text(d: ImageDraw.ImageDraw, cx: int, y: int, text: str, size: int, fill, kind="title", stroke=8):
    f = font(kind, size)
    w = d.textlength(text, font=f)
    d.text((cx - w / 2, y), text, font=f, fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0))
    return f.size


def _wrap(d, text: str, size: int, max_w: int = 960, kind="title") -> list[str]:
    f = font(kind, size)
    lines, cur = [], ""
    for w in text.split():
        nxt = (cur + " " + w).strip()
        if cur and d.textlength(nxt, font=f) > max_w:
            lines.append(cur)
            cur = w
        else:
            cur = nxt
    return lines + ([cur] if cur else [])


def overlay(shot: dict, i: int, n: int, spec: dict, cfg: dict, path: str) -> str:
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    if i == 0:  # series banner + episode title
        _text(d, W // 2, 120, cfg["hero"]["name"].upper(), 96, (255, 214, 0))
        y = 235
        for ln in _wrap(d, spec["title"], 64)[:2]:
            _text(d, W // 2, y, ln, 64, (255, 255, 255), stroke=6)
            y += 78
    if shot.get("line_roman"):
        lines = _wrap(d, shot["line_roman"], 70)[:3]
        y = int(H * 0.70) - 40 * (len(lines) - 1)
        label = LABEL.get(shot["speaker"])
        if label:
            _text(d, W // 2, y - 70, label, 46, (255, 80, 60), stroke=5)
        for ln in lines:
            _text(d, W // 2, y, ln, 70, (255, 255, 255), stroke=7)
            y += 84
    if i == n - 1:
        _text(d, W // 2, H - 260, "FOLLOW FOR NEXT EPISODE", 58, (255, 214, 0), stroke=6)
    img.save(path)
    return path


# ─── shot segments ───────────────────────────────────────────────────────────
def segment(i: int, src_dir: str, dur: float, ov: str, out: str) -> str:
    clip = os.path.join(src_dir, f"shot_{i}.mp4")
    still = os.path.join(src_dir, f"shot_{i}.png")
    cover = f"scale={W}:{H}:force_original_aspect_ratio=increase:flags=lanczos,crop={W}:{H},setsar=1"
    if os.path.exists(clip):
        L = duration(clip)
        slow = min(max(dur / L, 1.0), 1.6)  # stretch a little, then hold the last frame
        vf = f"[0:v]setpts={slow:.3f}*PTS,{cover},fps={FPS},tpad=stop_mode=clone:stop_duration={dur:.2f}[v]"
        inp = ["-i", clip]
    else:  # still image → slow cinematic push-in (alternating direction)
        frames = int(dur * FPS) + 1
        z = "min(zoom+0.0009,1.15)" if i % 2 == 0 else "if(eq(on,0),1.15,max(zoom-0.0009,1.0))"
        vf = (f"[0:v]scale={W * 2}:{H * 2}:force_original_aspect_ratio=increase,crop={W * 2}:{H * 2},"
              f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':d={frames}:s={W}x{H}:fps={FPS},"
              f"setsar=1[v]")
        inp = ["-loop", "1", "-i", still]
    _run(["ffmpeg", "-y", "-loglevel", "error", *inp, "-i", ov, "-filter_complex",
          f"{vf};[v][1:v]overlay=0:0,format=yuv420p[o]", "-map", "[o]", "-t", f"{dur:.3f}", "-r", str(FPS),
          "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-an", out])
    return out


# ─── music bed ────────────────────────────────────────────────────────────────
def music_bed(total: float, cuts: list[float], seed: int) -> np.ndarray:
    """A heroic desi groove (dhol-style kick + tabla + drone) plus a whoosh on every cut. Pure numpy, free."""
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    n = int(total * SR) + SR
    buf = np.zeros((n, 2), dtype=np.float64)
    bpm = 100 + int(rng.integers(0, 20))
    beat = 60 / bpm
    t = 0.0
    k, ta, tb = synth.kick(), synth.tabla(1.0, True), synth.tabla(1.3, False)
    while t < total:
        for step in range(8):  # one bar of 8th notes
            at = t + step * beat / 2
            if step in (0, 3, 6):
                synth.place(buf, k, at, 0.55)
            if step in (2, 5, 7):
                synth.place(buf, ta, at, 0.30, pan=-0.3)
            if step in (1, 4):
                synth.place(buf, tb, at, 0.22, pan=0.3)
        t += 4 * beat
    root = 45 + int(rng.integers(0, 5))
    tt = np.arange(n) / SR
    drone = sum(np.sin(2 * np.pi * synth.midi_hz(m) * tt) for m in (root, root + 7, root + 12)) / 3
    drone *= 0.10 * (0.8 + 0.2 * np.sin(2 * np.pi * 0.25 * tt))
    buf += np.stack([drone, drone], axis=1)
    for c in cuts[1:]:  # whoosh into each new shot
        w = int(0.45 * SR)
        noise = synth.bp(np.random.randn(w), 600, 6000) * np.hanning(w) * 0.35
        synth.place(buf, noise, max(0, c - 0.3), 1.0)
    return buf[: int(total * SR)]


def assemble(spec: dict, cfg: dict, src_dir: str, out_dir: str) -> str:
    tmp = os.path.join(out_dir, "parts")
    os.makedirs(tmp, exist_ok=True)
    shots = spec["shots"]
    vs = voices(shots, tmp)
    lead = 0.25
    durs, starts, t = [], [], 0.0
    for v in vs:
        d = (duration(v) if v else 0) + lead + 0.45
        d = min(max(d, 2.6), 8.0)
        starts.append(t)
        durs.append(d)
        t += d
    total = t
    segs = []
    for i, s in enumerate(shots):
        ov = overlay(s, i, len(shots), spec, cfg, os.path.join(tmp, f"ov_{i}.png"))
        segs.append(segment(i, src_dir, durs[i], ov, os.path.join(tmp, f"seg_{i}.mp4")))
        print(f"   shot {i + 1}/{len(shots)}: {durs[i]:.1f}s {'clip' if os.path.exists(os.path.join(src_dir, f'shot_{i}.mp4')) else 'still'}")
    lst = os.path.join(tmp, "list.txt")
    with open(lst, "w") as fh:
        fh.writelines(f"file '{os.path.abspath(p)}'\n" for p in segs)
    silent = os.path.join(tmp, "video_silent.mp4")
    _run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent])

    # audio: music bed + voices (music ducks while someone speaks)
    bed = music_bed(total, starts, int(spec.get("episode", 1)) * 7919)
    voice = np.zeros(len(bed))
    for v, st in zip(vs, starts):
        if v:
            x = _decode(v)
            i0 = int((st + lead) * SR)
            x = x[: max(0, len(voice) - i0)]
            voice[i0:i0 + len(x)] += x
    env = np.convolve(np.abs(voice) > 0.01, np.ones(int(0.25 * SR)) / int(0.25 * SR), mode="same")
    duck = 1.0 - 0.6 * np.clip(env * 3, 0, 1)
    mix = bed * duck[:, None] * 0.9 + np.stack([voice, voice], axis=1) * 1.25
    mix /= max(1.0, np.abs(mix).max() / 0.95)
    wav = os.path.join(tmp, "audio.wav")
    import scipy.io.wavfile as wf
    wf.write(wav, SR, (mix * 32767).astype(np.int16))
    out = os.path.join(out_dir, "episode.mp4")
    _run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-map", "0:v", "-map", "1:a",
          "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart", out])
    with open(os.path.join(out_dir, "timeline.json"), "w") as fh:
        json.dump({"starts": starts, "durations": durs, "total": total}, fh, indent=1)
    print(f"   episode: {total:.1f}s → {out}")
    return out
