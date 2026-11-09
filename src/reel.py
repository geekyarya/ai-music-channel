"""Vertical 9:16 clip of the best part of the song, for Instagram Reels and YouTube Shorts."""
from __future__ import annotations

import copy
import os
import subprocess

import numpy as np
from scipy.io import wavfile

import video


def pick_window(audio: str, timings, length: float) -> float:
    """Start time: just before the first chorus if we know it, else the loudest stretch."""
    sr, x = wavfile.read(audio)
    dur = len(x) / sr
    if dur <= length:
        return 0.0
    if timings:
        chorus = [t for t in timings if t.get("section") in ("chorus", "hook")]
        if chorus:
            return float(min(max(0.0, chorus[0]["start"] - 1.5), dur - length))
    x = x.astype(np.float32)
    if x.ndim == 2:
        x = x.mean(axis=1)
    hop = sr // 2  # half-second energy blocks
    e = np.array([np.sqrt(np.mean(x[i:i + hop] ** 2)) for i in range(0, len(x) - hop, hop)])
    win = int(length * 2)
    sums = np.convolve(e, np.ones(win), mode="valid")
    return float(min(np.argmax(sums) / 2, dur - length))


def make_reel(spec: dict, cfg: dict, audio: str, timings, out_dir: str) -> str:
    r = cfg.get("reels", {})
    length = float(r.get("seconds", 45))
    start = pick_window(audio, timings, length)
    clip = os.path.join(out_dir, "reel_audio.wav")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-t", f"{length:.2f}",
                    "-i", audio, "-af", f"afade=t=in:d=0.3,afade=t=out:st={length - 1.5:.2f}:d=1.5", clip],
                   check=True)
    shifted = None
    if timings:
        shifted = [{**t, "start": t["start"] - start, "end": t["end"] - start}
                   for t in timings if t["end"] > start and t["start"] < start + length]
    rcfg = copy.deepcopy(cfg)
    rcfg["video"].update(width=1080, height=1920, intro_seconds=2.5, lyrics_y=0.64)
    out = os.path.join(out_dir, "reel.mp4")
    print(f"📱 reel: {length:.0f}s from {start:.1f}s")
    return video.render(spec, rcfg, clip, shifted, out)
