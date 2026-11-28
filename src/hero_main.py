"""Captain Gamchha pipeline: episode script → Kaggle GPU (SDXL stills + LTX-Video clips) → voices, captions,
music → vertical MP4 → YouTube Short + Instagram Reel + Facebook Page reel."""
from __future__ import annotations

import argparse
import base64
import copy
import datetime as dt
import json
import os
import random
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import hero_story  # noqa: E402
import hero_video  # noqa: E402
import kaggle_run  # noqa: E402
import youtube  # noqa: E402
from main import HISTORY, ROOT, _since, load_history  # noqa: E402

REF = os.path.join(ROOT, "assets", "hero_ref.png")
FACE = os.path.join(ROOT, "assets", "hero_face.png")  # head crop: IP-Adapter face copies the face, not the pose

TEST_EPISODE = {
    "title": "Toofan Express Ki Pehli Udaan",
    "plot": "Captain Gamchha races his buffalo Toofan to fix the village hand pump before a wedding",
    "description": "Shaadi se pehle hand pump band! Kya Captain Gamchha time pe pahunch paayega? Comment karo 👇",
    "hashtags": ["#CaptainGamchha", "#desicomedy"],
    "shots": [
        {"visual": "Wide shot of a village wedding tent, worried villagers crowd around a broken hand pump, "
                   "Dadi with a walking stick shouting", "motion": "villagers wave arms, wedding lights flicker",
         "speaker": "villager_female", "line": "अरे! शादी है और हैंडपंप ही बंद हो गया!",
         "line_roman": "Arey! Shaadi hai aur handpump hi band ho gaya!"},
        {"visual": "Low angle hero shot of Captain Gamchha standing on a haystack at sunrise, red cape flying",
         "motion": "cape flutters in the wind, camera tilts up", "speaker": "hero",
         "line": "गाँव की रक्षा, गमछे की कसम!", "line_roman": "Gaon ki raksha, gamchhe ki kasam!"},
        {"visual": "Captain Gamchha riding Toofan the huge black buffalo at full speed through a mustard field",
         "motion": "buffalo gallops toward camera, dust and yellow flowers fly", "speaker": "narrator",
         "line": "और निकल पड़ी तूफ़ान एक्सप्रेस!", "line_roman": "Aur nikal padi Toofan Express!"},
        {"visual": "Close-up of Captain Gamchha pumping the hand pump with superhuman strength, villagers amazed",
         "motion": "arm pumps fast, water bursts out", "speaker": "narrator",
         "line": "एक, दो, तीन... और पानी ही पानी!", "line_roman": "Ek, do, teen... aur paani hi paani!"},
        {"visual": "Water fountain sprays over the whole wedding party, the groom drenched, everyone laughing",
         "motion": "water splashes everywhere, people laugh", "speaker": "villager_male",
         "line": "भैया, पानी चाहिए था, बाढ़ नहीं!", "line_roman": "Bhaiya, paani chahiye tha, baadh nahi!"},
    ],
}


def uploads_today(history: list[dict]) -> int:
    from zoneinfo import ZoneInfo
    now = dt.datetime.now(dt.timezone.utc).astimezone(ZoneInfo("America/Los_Angeles"))
    start = now.replace(hour=0, minute=0, second=0, microsecond=0).astimezone(dt.timezone.utc)
    return sum(bool(h.get("video_id")) + bool(h.get("short_id")) for h in _since(history, start))


def make_visuals(spec: dict, cfg: dict, out: str) -> dict:
    h = cfg["hero"]
    shots = []
    for s in spec["shots"]:
        p1, p2 = hero_story.image_prompts(s, cfg)
        shots.append({"p1": p1, "p2": p2, "vp": hero_story.video_prompt(s, cfg), "hero": hero_story.has_hero(s),
                      "seed": random.randint(1, 2**31 - 1)})
    params = {"shots": shots, "negative": h["negative"], "video": bool(h.get("video_model", True)),
              "image_size": [768, 1344], "video_size": [480, 832], "num_frames": 97,
              "video_steps": int(h.get("video_steps", 30)), "video_budget_minutes": h.get("video_budget_minutes", 85)}
    ref_path = FACE if os.path.exists(FACE) else REF
    if os.path.exists(ref_path):
        # Kaggle rejects big scripts, so embed a small JPEG (IP-Adapter only sees 224x224 anyway)
        import io
        from PIL import Image
        im = Image.open(ref_path).convert("RGB")
        im.thumbnail((448, 768))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=88)
        params["ref_b64"] = base64.b64encode(buf.getvalue()).decode()
        params["ref_scale"] = h.get("ref_scale", 0.6)
        params["ref_kind"] = "face" if ref_path == FACE else "plus"
    src = os.path.join(out, "gpu")
    kaggle_run.run(os.path.join(os.path.dirname(__file__), "hero_kernel.py"), params, h.get("kaggle_slug", "ngr-hero"),
                   src, timeout_minutes=h.get("kaggle_timeout_minutes", 150))
    res = json.load(open(os.path.join(src, "result.json"))) if os.path.exists(os.path.join(src, "result.json")) else {}
    missing = [i for i in range(len(spec["shots"])) if not os.path.exists(os.path.join(src, f"shot_{i}.png"))]
    if missing:
        raise RuntimeError(f"Kaggle didn't return images for shots {missing}: {res.get('notes')}")
    print(f"   GPU: {res.get('gpu')} — {len(res.get('images', []))} images, {len(res.get('clips', []))} clips, "
          f"{res.get('seconds')}s; notes: {res.get('notes')}")
    return {"dir": src, "result": res}


def fake_visuals(spec: dict, out: str) -> dict:
    """Local test without a GPU: coloured placeholder frames."""
    from PIL import Image, ImageDraw
    src = os.path.join(out, "gpu")
    os.makedirs(src, exist_ok=True)
    for i, s in enumerate(spec["shots"]):
        img = Image.new("RGB", (768, 1344), (40 + 30 * i % 200, 60, 120))
        ImageDraw.Draw(img).text((40, 600), s["visual"][:60], fill=(255, 255, 255))
        img.save(os.path.join(src, f"shot_{i}.png"))
    return {"dir": src, "result": {}}


def make_references(cfg: dict, out: str, n: int = 6) -> None:
    """Draw candidate model sheets for the cast; the chosen hero picture becomes assets/hero_ref.png."""
    h = cfg["hero"]
    style = h["style"]
    sheets = [("hero", f"full body character design of {h['look_short']}, standing heroically with hands on hips, "
                       f"front view, centered, whole body visible, simple light village background, {style}",
               h["look"])] * n
    sheets += [("toofan", "full body character design of Toofan, a huge friendly black water buffalo with golden "
                          f"painted horns, red saddle cloth and a brass bell, side view, centered, {style}",
                h["cast"][0])] * 2
    sheets += [("chhotu", "full body character design of Chhotu, a small cute white baby goat with tiny horns and "
                          f"a little beard wearing tiny round goggles, centered, {style}", h["cast"][1])] * 2
    shots = [{"p1": p1, "p2": f"{p2}, {style}", "vp": "", "hero": False, "seed": random.randint(1, 2**31 - 1)}
             for _, p1, p2 in sheets]
    src = os.path.join(out, "gpu")
    kaggle_run.run(os.path.join(os.path.dirname(__file__), "hero_kernel.py"),
                   {"mode": "ref", "shots": shots, "negative": h["negative"] + ", cropped, cut off",
                    "image_size": [896, 1152], "image_steps": 35}, h.get("kaggle_slug", "ngr-hero"), src,
                   timeout_minutes=90)
    from PIL import Image, ImageDraw
    from video import font
    tiles = []
    for i, (kind, _, _) in enumerate(sheets):
        p = os.path.join(src, f"shot_{i}.png")
        if os.path.exists(p):
            im = Image.open(p).convert("RGB").resize((448, 576))
            ImageDraw.Draw(im).text((14, 8), f"{kind} {i + 1}", font=font("title", 44), fill=(255, 230, 0),
                                    stroke_width=4, stroke_fill=(0, 0, 0))
            tiles.append(im)
    cols = 5
    sheet = Image.new("RGB", (448 * cols, 576 * ((len(tiles) + cols - 1) // cols)), (20, 20, 20))
    for k, im in enumerate(tiles):
        sheet.paste(im, ((k % cols) * 448, (k // cols) * 576))
    sheet.save(os.path.join(out, "reference_sheet.jpg"), quality=90)
    print(f"   reference sheet: {len(tiles)} pictures → {out}/reference_sheet.jpg")


def metadata(spec: dict, cfg: dict) -> tuple[str, str, list[str], str]:
    h = cfg["hero"]
    title = f"{spec['title']} | {h['name']} Ep {spec['episode']} #shorts"[:100]
    tags_line = " ".join(spec["hashtags"][:8])
    desc = "\n\n".join(filter(None, [spec["description"], f"{h['name']} — Rampur ka apna superhero 🦸‍♂️ "
                                     "Naya episode roz! Subscribe karo 🔔", tags_line]))
    tags = h.get("default_tags", []) + [t.lstrip("#") for t in spec["hashtags"]] + [spec["title"]]
    caption = "\n\n".join(filter(None, [f"{spec['title']} 😂", spec["description"],
                                        f"{h['name']} Ep {spec['episode']} — follow for the next episode!",
                                        " ".join(spec["hashtags"][:20])]))
    return title, desc, tags, caption


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--offline", action="store_true", help="built-in test episode, no LLM")
    ap.add_argument("--no-gpu", action="store_true", help="placeholder images instead of Kaggle (local test)")
    ap.add_argument("--reference", action="store_true", help="draw candidate character pictures, nothing else")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    if args.reference:
        out = os.path.join(ROOT, "out", "reference")
        make_references(cfg, out)
        return
    history = load_history()

    if not args.no_upload and uploads_today(history) >= int(cfg["youtube"].get("daily_upload_limit", 6)):
        print("::notice::Skipping: YouTube's daily upload quota is used")
        return

    print("✍️  writing episode…")
    if args.offline:
        spec = hero_story.normalize(copy.deepcopy(TEST_EPISODE), cfg,
                                    1 + sum(1 for e in history if e.get("kind") == "hero"))
    else:
        spec = hero_story.write_episode(cfg, history)
    print(f"   Ep {spec['episode']}: “{spec['title']}” — {spec['plot']} ({len(spec['shots'])} shots)")
    for i, s in enumerate(spec["shots"]):
        print(f"     {i + 1}. [{s['speaker']}] {s['line_roman']}  |  {s['visual'][:90]}")

    slug = re.sub(r"[^a-z0-9]+", "-", spec["title"].lower()).strip("-")[:40]
    out = os.path.join(ROOT, "out", f"{dt.date.today():%Y-%m-%d}-hero-{slug}")
    os.makedirs(out, exist_ok=True)
    json.dump(spec, open(os.path.join(out, "episode.json"), "w"), indent=2, ensure_ascii=False)

    print("🎨 drawing + animating shots on Kaggle…" if not args.no_gpu else "🎨 placeholder shots…")
    vis = fake_visuals(spec, out) if args.no_gpu else make_visuals(spec, cfg, out)
    print("🎬 assembling…")
    mp4 = hero_video.assemble(spec, cfg, vis["dir"], out)

    title, desc, tags, caption = metadata(spec, cfg)
    errors, short_id, ig_link = [], None, None
    if args.no_upload:
        print("⏭  uploads skipped (--no-upload)")
    else:
        ycfg = copy.deepcopy(cfg)
        ycfg["youtube"]["category_id"] = cfg["hero"].get("youtube_category_id", "23")
        if youtube.have_credentials():
            try:
                print("📤 uploading YouTube Short…")
                short_id = youtube.upload(mp4, None, title, desc, tags, ycfg, language="hi")
            except Exception as e:  # noqa: BLE001
                errors.append(f"YouTube: {e}")
        import instagram
        if cfg.get("instagram", {}).get("enabled", True) and instagram.have_credentials():
            try:
                print("📤 posting Instagram reel…")
                ig_link = instagram.post_reel(mp4, caption, cfg)
            except Exception as e:  # noqa: BLE001
                errors.append(f"Instagram: {e}")
        import facebook
        if cfg.get("facebook", {}).get("enabled", True) and facebook.have_credentials():
            try:
                print("📤 posting Facebook Page reel…")
                facebook.post_reel(mp4, caption, cfg)
            except Exception as e:  # noqa: BLE001
                errors.append(f"Facebook: {e}")
        history.append({"kind": "hero", "date": dt.date.today().isoformat(),
                        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                        "title": spec["title"], "episode": spec["episode"], "plot": spec["plot"],
                        "theme": spec["plot"], "short_id": short_id, "instagram": ig_link})
        os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
        json.dump(history, open(HISTORY, "w"), indent=2)
    for e in errors:
        print(f"::error::{e}")
    print(f"🎉 done → {out}")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
