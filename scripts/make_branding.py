"""Generate YouTube channel art from the same visual engine as the videos.

    python scripts/make_branding.py        → branding/avatar.png, branding/banner.png

Re-run after changing channel.name / channel.artist in config.yaml.
"""
import os
import sys

import numpy as np
import yaml
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import video  # noqa: E402

PALETTE = ["#0b0221", "#3a0ca3", "#f72585", "#4cc9f0"]


def fake_features(n=1):
    """A frozen 'loud moment' so the visuals look energetic."""
    spec = np.clip(0.55 + 0.35 * np.sin(np.linspace(0, 9, 64)) ** 2, 0, 1)[None].repeat(n, 0)
    one = np.ones(n)
    return dict(n=n, fps=30, bass=one * 0.9, mid=one * 0.7, high=one * 0.5, energy=one, beat=one * 0.6,
                spec=spec, travel=one * 3.7, duration=60.0)


def background(style, w, h, ds=2):
    st = dict(W=w, H=h, ds=ds, f=fake_features(), palette=PALETTE, style=style, title="", artist="",
              genre="", timings=None, thumb=True)
    video._init(st)
    bg = video.BACKGROUNDS[style](0, video.G["f"]) * video.G["vignette"]
    return Image.fromarray((np.clip(bg, 0, 1) * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), \
        video.G["accent"]


def main():
    cfg = yaml.safe_load(open(os.path.join(ROOT, "config.yaml")))
    name, artist = cfg["channel"]["name"], cfg["channel"]["artist"]
    out = os.path.join(ROOT, "branding")
    os.makedirs(out, exist_ok=True)

    # avatar 800x800 — YouTube crops it to a circle
    img, acc = background("kaleido", 800, 800)
    d = ImageDraw.Draw(img)
    video._ring(d, 400, 400, 205, fake_features()["spec"][0], 0.9, acc, 5)
    d.ellipse([200, 200, 600, 600], fill=(8, 8, 14), outline=acc, width=8)
    video._center_text(d, 400, 385, artist, video.font("title", 150), (255, 255, 255), 360)
    video._center_text(d, 400, 488, "♪ AI MUSIC", video.font("body", 34), acc)
    img.save(os.path.join(out, "avatar.png"))

    # banner 2560x1440 — only the centre 1546x423 is visible on every device
    img, acc = background("synthgrid", 2560, 1440, ds=4)
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    od.rounded_rectangle([507, 520, 2053, 920], radius=40, fill=(0, 0, 0, 120))
    img = Image.alpha_composite(img.convert("RGBA"), ov).convert("RGB")
    d = ImageDraw.Draw(img)
    video._center_text(d, 1280, 660, name.upper(), video.font("title", 150), (255, 255, 255), 1450)
    video._center_text(d, 1280, 790, "A new AI song every day  •  English  •  Hindi  •  Hinglish",
                       video.font("body", 46), acc, 1450)
    video._center_text(d, 1280, 865, "rap  •  pop  •  desi hip hop  •  bollywood  •  lo-fi  •  rock",
                       video.font("body", 34), (220, 220, 230), 1450)
    img.save(os.path.join(out, "banner.png"))
    print("wrote", out)


if __name__ == "__main__":
    main()
