"""Write one Captain Gamchha mini-episode (shot list + Hindi dialogue) with the free LLM chain."""
from __future__ import annotations

import random
import re

from llm import chat_json

SYSTEM = ("You are the head writer of a hit Indian comedy short-video series about an ORIGINAL desi village "
          "superhero. You write punchy, family-friendly, laugh-out-loud 35-second episodes in everyday Hindi "
          "(Hinglish words welcome). Big visual gags, a fast hook in the first 2 seconds, a twist or punchline "
          "at the end. Never use existing superheroes or brands (no Batman, Superman, Marvel, DC, etc.). "
          "No politics, religion jokes, caste, violence with blood, or insults to real people. Reply with JSON only.")

USER_TMPL = """Write episode #{episode} of "{name}".

Hero: {look}
Other characters: {cast}
Setting: {setting}
Hero's catchphrase (use it once, at the heroic moment): "{catchphrase}"
Recent episode plots to AVOID repeating: {recent}
Inspiration seed (use loosely): {seed}

Make exactly {shots} shots, each about 5 seconds. Return JSON:
{{
  "title": "short funny Hinglish title in Roman script (max 60 chars)",
  "plot": "one-line English summary",
  "shots": [
    {{
      "visual": "English image prompt for ONE frame: who is in it (name them), what they are doing, where, camera angle (wide/close-up/low angle...). Describe the action vividly. Max 45 words.",
      "motion": "English: what moves in this 5-second shot (e.g. 'buffalo gallops toward camera, cape flutters, dust flies'). Max 20 words.",
      "speaker": "narrator" | "hero" | "villain" | "villager_male" | "villager_female" | "kid",
      "line": "ONE short spoken line in Hindi written in Devanagari (max 14 words, natural spoken Hindi)",
      "line_roman": "the same line in Roman (Hinglish) spelling for on-screen captions"
    }}
  ],
  "description": "2 sentence YouTube/Instagram description in Hinglish, ending with a question to make people comment",
  "hashtags": ["#tag", ...] (5-7, no #ai)
}}
Shot 1 must hook instantly (problem or crazy situation). The last shot ends with a punchline.
Visual rules (the video AI is simple): at most 2 characters per shot, name every character in "visual"
(e.g. "Captain Gamchha", "Toofan the buffalo", "Chhotu the goat"); show Captain Gamchha in at least 4 shots.
"motion" must be ONE simple, slow movement (walking, turning the head, smiling, cape fluttering, camera slowly
pushing in). NO crowds, dust storms, explosions, water sprays, fast spinning or flying objects."""

SEEDS = [
    "the village hand pump stops working before a wedding", "a monkey steals Dadi's spectacles",
    "Lala Laalchi rigs the cricket match", "the tractor won't start on harvest day",
    "a power cut during the India match", "a rooster that won't stop crowing at midnight",
    "Toofan the buffalo gets stage fright at the village fair", "the chai stall runs out of sugar",
    "a kite gets stuck on the electricity pole", "Chhotu the goat eats the panchayat's official papers",
    "a fake baba comes to the village", "the school bus gets stuck in mud", "the mela's giant wheel gets stuck",
    "Lala Laalchi tries to sell 'magic' fertiliser", "the village gets its first drone",
    "a thief steals all the mangoes from the orchard", "Captain Gamchha has to learn to use a smartphone",
    "the village wrestling dangal", "a snake in the grain store", "the bride's mobile goes missing",
    "Holi colours go missing", "a speeding SUV races through the village lanes", "the well rope breaks",
    "Lala Laalchi kidnaps the village's prize cow", "the WiFi tower comes to Rampur",
]

SPEAKERS = {"narrator", "hero", "villain", "villager_male", "villager_female", "kid"}


def write_episode(cfg: dict, history: list[dict]) -> dict:
    h = cfg["hero"]
    episode = 1 + sum(1 for e in history if e.get("kind") == "hero")
    recent = "; ".join(e.get("plot", "") for e in history if e.get("kind") == "hero")[-1500:] or "none"
    spec = chat_json(cfg["llm"]["providers"], SYSTEM, USER_TMPL.format(
        episode=episode, name=h["name"], look=h["look"], cast="; ".join(h["cast"]), setting=h["setting"],
        catchphrase=h["catchphrase"], recent=recent, seed=random.choice(SEEDS), shots=h.get("shots", 7)))
    return normalize(spec, cfg, episode)


def normalize(spec: dict, cfg: dict, episode: int) -> dict:
    h = cfg["hero"]
    shots = []
    for s in spec.get("shots", [])[: int(h.get("shots", 7)) + 1]:
        visual = str(s.get("visual") or "").strip()
        line = str(s.get("line") or "").strip()
        if not visual:
            continue
        sp = str(s.get("speaker") or "narrator").lower()
        shots.append({"visual": visual, "motion": str(s.get("motion") or "subtle camera push in").strip(),
                      "speaker": sp if sp in SPEAKERS else "narrator", "line": line,
                      "line_roman": str(s.get("line_roman") or line).strip()})
    if len(shots) < 3:
        raise ValueError(f"LLM returned only {len(shots)} usable shots")
    spec["shots"] = shots
    spec["episode"] = episode
    spec["title"] = re.sub(r'[<>"]', "", str(spec.get("title") or f"{h['name']} Episode {episode}")).strip()[:70]
    spec["plot"] = str(spec.get("plot") or spec["title"])
    tags = [t if t.startswith("#") else "#" + t for t in spec.get("hashtags", []) if isinstance(t, str)]
    spec["hashtags"] = [t for t in dict.fromkeys(tags + h.get("hashtags", [])) if "ai" != t.lower().lstrip("#")]
    spec["description"] = str(spec.get("description") or "")
    return spec


def _characters(shot: dict, cfg: dict) -> list[str]:
    h = cfg["hero"]
    v = shot["visual"]
    out = []
    if re.search(r"gamchha|captain|hero|\bhe\b|his", v, re.I):
        out.append(h["look_short"])
    for c in h["cast"]:
        name = c.split(":")[0].strip()
        if name.split()[0].lower() in v.lower():
            out.append(c)
    return out


def image_prompts(shot: dict, cfg: dict) -> tuple[str, str]:
    """SDXL reads 77 tokens per text encoder, so the shot goes to one encoder and the character sheet to the
    other (prompt / prompt_2). Both end with the style, so every shot looks like the same show."""
    h = cfg["hero"]
    style = h["style"]
    chars = _characters(shot, cfg)
    return f"{shot['visual']}, {style}", ", ".join(chars + [h["setting_short"], style])


def has_hero(shot: dict) -> bool:
    return bool(re.search(r"gamchha|captain", shot["visual"], re.I))


def video_prompt(shot: dict, cfg: dict) -> str:
    """LTX-Video reads long prompts (T5), and works best with a detailed, literal description."""
    h = cfg["hero"]
    return " ".join([shot["motion"].rstrip(".") + ".", shot["visual"].rstrip(".") + ".",
                     *[c.rstrip(".") + "." for c in _characters(shot, cfg)],
                     "3D animated film, cinematic lighting, smooth natural motion."])
