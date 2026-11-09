"""Post (or re-post) an already-made song to any of: Instagram Reels, Facebook Page Reels,
YouTube (full video) and YouTube Shorts.

    python scripts/post_reel.py <song folder> [--instagram] [--facebook] [--youtube-video] [--youtube-short]

The folder must contain song.wav and spec.json (as saved by a daily run / its GitHub artifact).
The vertical reel is rendered if missing (songs made before reels existed).
Titles/descriptions are rebuilt with today's rules (e.g. no AI wording while disclose_ai is false).
"""
import argparse
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import facebook  # noqa: E402
import instagram  # noqa: E402
import reel  # noqa: E402
import youtube  # noqa: E402
from main import build_metadata, normalize_old_spec  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--instagram", action="store_true")
ap.add_argument("--facebook", action="store_true")
ap.add_argument("--youtube-video", action="store_true")
ap.add_argument("--youtube-short", action="store_true")
args = ap.parse_args()

cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
spec = normalize_old_spec(json.load(open(os.path.join(args.folder, "spec.json"))), cfg)
engine = spec.get("engine", "")
if engine and not engine.startswith("ACE-Step"):
    sys.exit(f"::error::'{spec['title']}' was made by the fallback engine ({engine}); not posting it.")
print(f"🎵 {spec['title']} ({spec['language_name']} {spec['genre']})")
reel_path = os.path.join(args.folder, "reel.mp4")
if (args.instagram or args.facebook or args.youtube_short) and not os.path.exists(reel_path):
    reel_path = reel.make_reel(spec, cfg, os.path.join(args.folder, "song.wav"), spec.get("timings"), args.folder)

errors = []
lang = "hi" if spec["language"] != "en" else "en"
title, desc, tags = build_metadata(spec, cfg, engine)


def attempt(name, fn):
    try:
        fn()
    except Exception as e:  # noqa: BLE001
        errors.append(f"{name}: {e}")


if args.youtube_video:
    video_path = os.path.join(args.folder, "video.mp4")
    thumb = os.path.join(args.folder, "thumbnail.jpg")
    attempt("YouTube video", lambda: youtube.upload(video_path, thumb if os.path.exists(thumb) else None,
                                                    title, desc, tags, cfg, language=lang))
if args.youtube_short:
    short_desc = "\n".join(filter(None, [spec["description"], "Full song on the channel 🎧",
                                         " ".join(spec["hashtags"][:5] + ["#shorts"])]))
    attempt("YouTube Short", lambda: youtube.upload(reel_path, None, f"{spec['title']} 🎧 #shorts"[:100],
                                                    short_desc, tags + ["shorts"], cfg, language=lang))
if args.instagram:
    if not instagram.have_credentials():
        errors.append("Instagram: no token secret set")
    else:
        attempt("Instagram", lambda: instagram.post_reel(reel_path, instagram.build_caption(spec, cfg), cfg))
if args.facebook:
    if not facebook.have_credentials():
        errors.append("Facebook: no Meta (Facebook-login) token secret set")
    else:
        attempt("Facebook", lambda: facebook.post_reel(reel_path, instagram.build_caption(spec, cfg), cfg))

for e in errors:
    print(f"::error::{e}")
sys.exit(1 if errors else 0)
