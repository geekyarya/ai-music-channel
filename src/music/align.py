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
    found, p, misses = {}, 0, 0  # found: line index → (start, end)
    for li, (text, match_text, typ) in enumerate(lines):
        target = _norm(match_text)
        n = len(target.split())
        best = (0.0, None, None)
        # Look only a few lines ahead, so a repeated chorus can't make us jump to its last occurrence;
        # widen after several misses in a row to recover if Whisper dropped a whole section.
        window = 80 if misses >= 4 else 4 * n + 6
        for i in range(p, min(len(words), p + window)):
            for ln in (n - 2, n - 1, n, n + 1, n + 2):
                if ln < 1 or i + ln > len(words):
                    continue
                cand_words = [w[0] for w in words[i:i + ln]]
                r = SequenceMatcher(None, target, " ".join(cand_words)).ratio()
                if hindi:  # Devanagari lines share many characters; also require whole-word overlap
                    r = (r + SequenceMatcher(None, target.split(), cand_words).ratio()) / 2
                if r > best[0]:
                    best = (r, i, i + ln)
        if best[0] >= (0.5 if hindi else 0.55):
            _, i, j = best
            found[li] = (words[i][1], words[j - 1][2])
            p, misses = j, 0
        else:
            misses += 1
    coverage = len(found) / max(1, len(lines))
    print(f"  aligned {len(found)}/{len(lines)} lines ({coverage:.0%})")
    if hindi:
        min_coverage = min(min_coverage, 0.15)  # Whisper is weaker on sung Hindi; we interpolate the rest
    if coverage < min_coverage:
        return None

    # Fill unmatched lines that sit between two matched ones by spreading them evenly in the gap.
    idx = sorted(found)
    for a, b in zip(idx, idx[1:]):
        gap = b - a - 1
        if gap <= 0:
            continue
        t0, t1 = found[a][1] + 0.15, found[b][0] - 0.15
        if t1 - t0 < gap * 0.8:  # not enough room — those lines probably weren't sung
            continue
        step = (t1 - t0) / gap
        for k in range(gap):
            found[a + 1 + k] = (t0 + k * step, t0 + (k + 1) * step - 0.1)
    filled = len(found) - len(idx)
    if filled:
        print(f"  interpolated {filled} more lines")
    return [{"start": round(found[i][0], 3), "end": round(found[i][1], 3), "text": lines[i][0],
             "section": lines[i][2]} for i in sorted(found)]
