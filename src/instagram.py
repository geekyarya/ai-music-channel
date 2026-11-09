"""Publish a Reel with Meta's free Instagram Graph API (Instagram professional account + Facebook Page).

Uses the resumable upload flow, so the video doesn't need to be hosted at a public URL:
  1. create a REELS container (upload_type=resumable)
  2. send the file bytes to rupload.facebook.com
  3. wait until Instagram finishes processing
  4. publish the container
"""
from __future__ import annotations

import os
import time

import requests

GRAPH = "https://graph.facebook.com"


def have_credentials() -> bool:
    return bool(os.environ.get("IG_USER_ID", "").strip() and os.environ.get("IG_ACCESS_TOKEN", "").strip())


def _env(name: str) -> str:
    v = os.environ[name].strip().strip('"\'')
    if "=" in v and v.split("=", 1)[0].strip().upper() == name:
        v = v.split("=", 1)[1].strip()
    return v


def _check(r: requests.Response) -> dict:
    try:
        data = r.json()
    except ValueError:
        r.raise_for_status()
        raise
    if r.status_code >= 400 or "error" in data:
        raise RuntimeError(f"Instagram API error {r.status_code}: {data.get('error', data)}")
    return data


def build_caption(spec: dict, cfg: dict) -> str:
    tags = list(dict.fromkeys(spec.get("hashtags", []) + cfg.get("instagram", {}).get("hashtags", [])))[:25]
    return "\n\n".join(filter(None, [
        f"{spec['title']} 🎧", spec.get("description", ""), cfg.get("youtube", {}).get("engagement_line"),
        f"Full song on YouTube: {cfg['channel']['name']} 🔗 link in bio", " ".join(tags)]))


def post_reel(video_path: str, caption: str, cfg: dict) -> str:
    ver = cfg.get("instagram", {}).get("graph_version", "v23.0")
    ig_user, token = _env("IG_USER_ID"), _env("IG_ACCESS_TOKEN")

    # 1. container
    c = _check(requests.post(f"{GRAPH}/{ver}/{ig_user}/media", data={
        "media_type": "REELS", "upload_type": "resumable", "caption": caption[:2200],
        "share_to_feed": "true", "access_token": token}, timeout=60))
    cid = c["id"]
    upload_uri = c.get("uri") or f"https://rupload.facebook.com/ig-api-upload/{ver}/{cid}"

    # 2. bytes
    size = os.path.getsize(video_path)
    with open(video_path, "rb") as fh:
        _check(requests.post(upload_uri, data=fh, timeout=600, headers={
            "Authorization": f"OAuth {token}", "offset": "0", "file_size": str(size)}))

    # 3. wait for processing
    for _ in range(60):
        st = _check(requests.get(f"{GRAPH}/{ver}/{cid}", params={
            "fields": "status_code,status", "access_token": token}, timeout=30))
        code = st.get("status_code")
        if code == "FINISHED":
            break
        if code in ("ERROR", "EXPIRED"):
            raise RuntimeError(f"Instagram processing failed: {st}")
        time.sleep(10)
    else:
        raise RuntimeError("Instagram processing timed out")

    # 4. publish
    pub = _check(requests.post(f"{GRAPH}/{ver}/{ig_user}/media_publish", data={
        "creation_id": cid, "access_token": token}, timeout=60))
    mid = pub["id"]
    try:
        link = _check(requests.get(f"{GRAPH}/{ver}/{mid}", params={
            "fields": "permalink", "access_token": token}, timeout=30)).get("permalink", mid)
    except Exception:  # noqa: BLE001
        link = mid
    print(f"✅ Instagram reel: {link}")
    return link
