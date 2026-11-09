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
    """Lowercase and drop punctuation/symbols, keeping letters of any script (incl. Devanagari matras)."""
    import unicodedata
    s = "".join(c for c in s.lower() if not unicodedata.category(c).startswith(("P", "S")))
    return re.sub(r"\s+", " ", s).strip()


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
    hindi = spec.get("language") in ("hi", "hinglish")
    model = WhisperModel("small" if hindi else "small.en", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(audio, word_timestamps=True, condition_on_previous_text=False,
                               vad_filter=False, language="hi" if hindi else "en", beam_size=3)
    words = [(_norm(w.word), w.start, w.end) for s in segs for w in (s.words or []) if _norm(w.word)]
    if not words:
        return None

    # Whisper writes Hindi in Devanagari, so match against the Devanagari lyrics, display Roman.
    lines = [(l, s.get("native", s["lines"])[k] if hindi else l, s["type"])
             for s in spec["sections"] for k, l in enumerate(s["lines"])]
    timings, p = [], 0
    for text, match_text, typ in lines:
        target = _norm(match_text)
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
    if hindi:
        min_coverage = min(min_coverage, 0.35)  # Whisper is weaker on sung Hindi; partial captions beat none
    print(f"  aligned {len(timings)}/{len(lines)} lines ({coverage:.0%})")
    return timings if coverage >= min_coverage else None
