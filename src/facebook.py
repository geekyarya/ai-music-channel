"""Publish a Reel to the Facebook Page with Meta's free Page Reels API.

Uses the same Meta token as Instagram (IG_ACCESS_TOKEN / INSTAGRAM_ACCESS_TOKEN). The token needs the
`pages_manage_posts` permission (plus pages_show_list, pages_read_engagement) to post on the Page.
Flow: start an upload session → send the bytes to rupload.facebook.com → finish + publish.
"""
from __future__ import annotations

import os
import time

import requests

from instagram import GRAPH, _check, _token


def have_credentials() -> bool:
    t = _token()
    return bool(t) and not t.startswith("IG")  # Instagram-login tokens can't post to Facebook


def _page(ver: str, token: str, cfg: dict) -> tuple[str, str, str]:
    """(page_id, page_token, name) for the Page named in facebook.page_name. Never guesses: one token
    can manage several Pages, and we must never post to the wrong one."""
    want = (cfg.get("facebook", {}).get("page_name") or "").strip().lower()
    if not want:
        raise RuntimeError("Set facebook.page_name in config.yaml so we never post to the wrong Page")
    me = requests.get(f"{GRAPH}/{ver}/me", params={"fields": "id,name,instagram_business_account",
                                                    "access_token": token}, timeout=30).json()
    if isinstance(me, dict) and "instagram_business_account" in me:  # already a Page token
        if me.get("name", "").strip().lower() == want:
            return me["id"], token, me["name"]
        raise RuntimeError(f"Token is for Page '{me.get('name')}', not '{cfg['facebook']['page_name']}'")
    pages = _check(requests.get(f"{GRAPH}/{ver}/me/accounts", params={
        "fields": "id,name,access_token", "access_token": token, "limit": 100}, timeout=30)).get("data", [])
    for p in pages:
        if p.get("name", "").strip().lower() == want:
            return p["id"], p["access_token"], p["name"]
    raise RuntimeError(f"Token can't reach Page '{cfg['facebook']['page_name']}' "
                       f"(it can reach: {', '.join(p.get('name', '?') for p in pages) or 'none'})")


def post_reel(video_path: str, description: str, cfg: dict) -> str:
    ver = cfg.get("instagram", {}).get("graph_version", "v23.0")
    page_id, ptoken, name = _page(ver, _token(), cfg)

    start = _check(requests.post(f"{GRAPH}/{ver}/{page_id}/video_reels", data={
        "upload_phase": "start", "access_token": ptoken}, timeout=60))
    vid = start["video_id"]
    url = start.get("upload_url") or f"https://rupload.facebook.com/video-upload/{ver}/{vid}"

    size = os.path.getsize(video_path)
    with open(video_path, "rb") as fh:
        _check(requests.post(url, data=fh, timeout=600, headers={
            "Authorization": f"OAuth {ptoken}", "offset": "0", "file_size": str(size)}))

    _check(requests.post(f"{GRAPH}/{ver}/{page_id}/video_reels", data={
        "upload_phase": "finish", "video_id": vid, "video_state": "PUBLISHED",
        "description": description[:2200], "access_token": ptoken}, timeout=60))

    for _ in range(40):  # wait until Facebook has processed and published it
        st = _check(requests.get(f"{GRAPH}/{ver}/{vid}", params={"fields": "status",
                                                                  "access_token": ptoken}, timeout=30))
        status = st.get("status", {})
        if status.get("video_status") == "ready" or status.get("publishing_phase", {}).get("status") == "complete":
            break
        if status.get("video_status") == "error":
            raise RuntimeError(f"Facebook processing failed: {status}")
        time.sleep(10)
    link = f"https://www.facebook.com/reel/{vid}"
    print(f"✅ Facebook reel on {name}: {link}")
    return link
