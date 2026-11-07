"""Best-effort lyric timing for AI-sung songs.

We know the lyrics, but not when the model sang each line. Whisper (CPU, int8)
transcribes the song with word timestamps, then each lyric line is fuzzy-matched
to the transcript in order. If too little matches, captions are skipped rather
than shown out of sync.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9' ]", "", s.lower()).strip()


def align_lyrics(audio_path: str, spec: dict, min_coverage: float = 0.45):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("  (faster-whisper not installed — no captions)")
        return None
    print("  → aligning lyrics with Whisper…")
    import subprocess

    import numpy as np
    pcm = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", audio_path, "-f", "f32le", "-ac", "1",
                          "-ar", "16000", "-"], capture_output=True, check=True).stdout
    audio = np.frombuffer(pcm, dtype=np.float32)
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(audio, word_timestamps=True, condition_on_previous_text=False,
                               vad_filter=False, language="en", beam_size=3)
    words = [(_norm(w.word), w.start, w.end) for s in segs for w in (s.words or []) if _norm(w.word)]
    if not words:
        return None

    lines = [(l, s["type"]) for s in spec["sections"] for l in s["lines"]]
    timings, p = [], 0
    for text, typ in lines:
        target = _norm(text)
        n = len(target.split())
        best = (0.0, None, None)
        for i in range(p, min(len(words), p + 80)):
            for ln in (n - 2, n - 1, n, n + 1, n + 2):
                if ln < 1 or i + ln > len(words):
                    continue
                cand = " ".join(w[0] for w in words[i:i + ln])
                r = SequenceMatcher(None, target, cand).ratio()
                if r > best[0]:
                    best = (r, i, i + ln)
        if best[0] >= 0.55:
            _, i, j = best
            timings.append({"start": round(words[i][1], 3), "end": round(words[j - 1][2], 3),
                            "text": text, "section": typ})
            p = j
    coverage = len(timings) / max(1, len(lines))
    print(f"  aligned {len(timings)}/{len(lines)} lines ({coverage:.0%})")
    return timings if coverage >= min_coverage else None
