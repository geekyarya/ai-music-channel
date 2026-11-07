"""Turns 'today' into a complete song spec: concept, lyrics, sound, and visual vibe."""
from __future__ import annotations

import random
import re

from llm import chat_json

VISUAL_STYLES = {
    "tunnel": "hypnotic neon tunnel rushing toward the viewer",
    "synthgrid": "retro sunset with a scrolling perspective grid",
    "starfield": "warp-speed starfield flying through space",
    "plasma": "flowing liquid plasma / lava-lamp color fields",
    "kaleido": "rotating kaleidoscope mandala",
    "waves": "layered glowing waveforms like ocean light",
}

SYSTEM = """You are a hit songwriter and creative director for an AI music YouTube channel.
You write ORIGINAL songs. Never copy or closely imitate existing lyrics, and never name or
impersonate real artists. Keep lyrics radio-safe (no slurs, no explicit sexual content).
Write vivid, specific, emotionally real lines with strong rhythm and rhyme. Hooks must be catchy.
Respond with a single JSON object only."""

USER_TMPL = """Write today's song.

Genre: {genre}
Sound palette: {style}
Language: {language}
Avoid these themes (used recently): {recent_themes}
Creative seed (use loosely for inspiration): {seed}

Return JSON with exactly these keys:
{{
  "title": "short catchy song title",
  "theme": "one-line summary of what the song is about",
  "mood": "2-4 mood words",
  "bpm": integer tempo appropriate for the genre,
  "key": "e.g. A minor",
  "vocal": "male" or "female",
  "music_tags": "comma separated music-generation tags: genre, instruments, vocal type, mood, production style (max 25 words, no artist names)",
  "sections": [
    {{"type": "verse" | "chorus" | "bridge" | "pre-chorus" | "outro", "lines": ["...", "..."]}}
  ],
  "visual_style": one of {styles},
  "palette": ["#rrggbb", "#rrggbb", "#rrggbb", "#rrggbb"]  (4 colors matching the vibe, dark→bright),
  "description": "2-3 sentence YouTube description hook (no hashtags)",
  "hashtags": ["#tag", ...] (5-8)
}}

Structure: verse (8 lines), chorus (4-6 lines), verse (8 lines), chorus, bridge (4 lines), chorus.
Each line 5-12 words. Rap genres: dense internal rhymes and multisyllabic flow."""

SEEDS = [
    "a late-night drive", "an old photograph", "winning against the odds", "a city that never sleeps",
    "first heartbreak", "summer that ended too soon", "a robot learning to dream", "leaving home",
    "the last train", "rain on a tin roof", "a friendship that survived", "running from the past",
    "a secret rooftop", "neon in puddles", "becoming who you were meant to be", "a letter never sent",
    "a festival at sunrise", "the grind before success", "ghost of a memory", "falling for a stranger",
    "an ocean you've never seen", "the quiet after a storm", "a mixtape from someone", "gravity",
]


def pick_genre(genres: list[dict], history: list[dict], avoid_recent: int, override: str | None = None) -> dict:
    if override:
        for g in genres:
            if g["name"].lower() == override.lower():
                return g
        return {"name": override, "style": override}
    recent = {h.get("genre") for h in history[-avoid_recent:]}
    choices = [g for g in genres if g["name"] not in recent] or genres
    return random.choice(choices)


def write_song(cfg: dict, history: list[dict], genre_override: str | None = None, offline: bool = False) -> dict:
    genre = pick_genre(cfg["genres"], history, cfg.get("avoid_recent", 5), genre_override)
    if offline:
        spec = _offline_song(genre)
    else:
        recent_themes = "; ".join(h.get("theme", "") for h in history[-15:]) or "none"
        spec = chat_json(
            cfg["llm"]["providers"],
            SYSTEM,
            USER_TMPL.format(
                genre=genre["name"], style=genre["style"], language=cfg["channel"]["language"],
                recent_themes=recent_themes, seed=random.choice(SEEDS),
                styles=" | ".join(f'"{s}"' for s in VISUAL_STYLES),
            ),
        )
    return _normalize(spec, genre)


def _normalize(spec: dict, genre: dict) -> dict:
    """Make the LLM output safe to use downstream, whatever it returned."""
    spec["genre"] = genre["name"]
    spec["genre_style"] = genre["style"]
    spec["title"] = re.sub(r'[<>"]', "", str(spec.get("title") or "Untitled")).strip()[:80]
    try:
        spec["bpm"] = int(max(60, min(180, int(spec.get("bpm", 100)))))
    except (TypeError, ValueError):
        spec["bpm"] = 100
    spec["vocal"] = "female" if str(spec.get("vocal", "")).lower().startswith("f") else "male"
    if spec.get("visual_style") not in VISUAL_STYLES:
        spec["visual_style"] = random.choice(list(VISUAL_STYLES))
    pal = [c for c in spec.get("palette", []) if isinstance(c, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", c)]
    spec["palette"] = (pal + ["#0b0221", "#3a0ca3", "#f72585", "#4cc9f0"])[:4] if len(pal) < 4 else pal[:4]
    sections = []
    for s in spec.get("sections", []):
        lines = [str(l).strip() for l in s.get("lines", []) if str(l).strip()]
        if lines:
            sections.append({"type": str(s.get("type", "verse")).lower(), "lines": lines})
    if not sections:
        raise ValueError("Song has no lyrics")
    spec["sections"] = sections
    spec["hashtags"] = [h if h.startswith("#") else f"#{h}" for h in spec.get("hashtags", [])][:8]
    spec.setdefault("music_tags", genre["style"])
    spec.setdefault("mood", "energetic")
    spec.setdefault("theme", spec["title"])
    spec.setdefault("description", "")
    return spec


def lyrics_text(spec: dict) -> str:
    """Lyrics in the [section] format music models expect."""
    return "\n\n".join(f"[{s['type']}]\n" + "\n".join(s["lines"]) for s in spec["sections"])


# ─── Offline fallback (no LLM) — used for testing and as a last resort ──────────
def _offline_song(genre: dict) -> dict:
    a = ["midnight", "neon", "golden", "broken", "electric", "silent", "burning", "endless"]
    n = ["city", "signal", "highway", "heartbeat", "skyline", "echo", "satellite", "river"]
    title = f"{random.choice(a).title()} {random.choice(n).title()}"
    verse1 = [
        "Woke up with the static humming in my head",
        "Every word I never said is dancing on the thread",
        "Streetlights flicker like they're counting down my time",
        "I been chasing every shadow trying to make it rhyme",
        "Pocket full of promises and none of them are free",
        "Mirror keep on asking who I'm really trying to be",
        "Concrete in my sneakers but the sky is in my lungs",
        "Running on the rhythm of the songs I haven't sung",
    ]
    verse2 = [
        "Phone lit up at three a.m. another empty call",
        "Built a little kingdom out of nothing, watched it fall",
        "Picked up all the pieces made a staircase out of stone",
        "Every step I'm climbing now I'm climbing on my own",
        "They said slow it down but I was born to break the speed",
        "Planted every heartbreak and it grew into a seed",
        "Now the night is mine and every window starts to glow",
        "Turn the volume up and let the whole world know",
    ]
    chorus = [f"We're the {title.lower()} burning through the dark",
              "Hearts like thunder, every beat a spark",
              "Hands up high, we never coming down",
              f"{title}, we own this town"]
    bridge = ["If the world goes quiet, I'll be loud",
              "If the sky goes grey, I'll split the cloud",
              "Every scar's a story that I wear",
              "Every dream I had is waiting there"]
    return {
        "title": title, "theme": "rising above the noise of the city at night", "mood": "defiant, hopeful",
        "bpm": random.choice([88, 92, 96, 140]), "key": "A minor", "vocal": "male",
        "music_tags": genre["style"],
        "sections": [{"type": "verse", "lines": verse1}, {"type": "chorus", "lines": chorus},
                     {"type": "verse", "lines": verse2}, {"type": "chorus", "lines": chorus},
                     {"type": "bridge", "lines": bridge}, {"type": "chorus", "lines": chorus}],
        "visual_style": random.choice(list(VISUAL_STYLES)),
        "palette": random.choice([["#0b0221", "#3a0ca3", "#f72585", "#4cc9f0"],
                                  ["#03071e", "#6a040f", "#e85d04", "#ffba08"],
                                  ["#011627", "#2ec4b6", "#e71d36", "#fdfffc"]]),
        "description": f"{title} is about refusing to fade into the noise of the city.",
        "hashtags": ["#aimusic", "#newmusic", f"#{genre['name'].replace(' ', '')}"],
    }
