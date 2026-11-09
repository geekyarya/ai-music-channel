"""Post (or re-post) the vertical reel for an already-made song.

    python scripts/post_reel.py <song folder> [--youtube-short]

The folder must contain song.wav and spec.json (as saved by the daily run / its GitHub artifact).
If reel.mp4 is missing (songs made before reels existed) it is rendered first.
"""
import argparse
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import instagram  # noqa: E402
import reel  # noqa: E402
import youtube  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--youtube-short", action="store_true", help="also upload the reel as a YouTube Short")
args = ap.parse_args()

cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
spec = json.load(open(os.path.join(args.folder, "spec.json")))
path = os.path.join(args.folder, "reel.mp4")
if not os.path.exists(path):
    path = reel.make_reel(spec, cfg, os.path.join(args.folder, "song.wav"), spec.get("timings"), args.folder)

ok = True
if not instagram.have_credentials():
    print("::error::No Instagram token secret (IG_ACCESS_TOKEN or INSTAGRAM_ACCESS_TOKEN) is set")
    ok = False
else:
    try:
        instagram.post_reel(path, instagram.build_caption(spec, cfg), cfg)
    except Exception as e:  # noqa: BLE001
        print(f"::error::Instagram: {e}")
        ok = False
if args.youtube_short and youtube.have_credentials():
    tags = cfg["youtube"]["default_tags"] + [spec["genre"], spec["title"], "shorts"]
    youtube.upload(path, None, f"{spec['title']} 🎧 #shorts"[:100], spec.get("description", ""), tags, cfg,
                   language="hi" if spec.get("language", "en") != "en" else "en")
sys.exit(0 if ok else 1)
