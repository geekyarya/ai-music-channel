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


TOKEN_NAMES = ("IG_ACCESS_TOKEN", "INSTAGRAM_ACCESS_TOKEN")


def _env(name: str) -> str:
    v = os.environ.get(name, "").strip().strip('"\'')
    if "=" in v and v.split("=", 1)[0].strip().upper() == name:
        v = v.split("=", 1)[1].strip()
    return v


def _token() -> str:
    for n in TOKEN_NAMES:
        if _env(n):
            return _env(n)
    return ""


def have_credentials() -> bool:
    return bool(_token())


def _base(token: str) -> str:
    # Tokens from "Instagram API with Instagram Login" start with IG; Facebook-login tokens use graph.facebook.com
    return "https://graph.instagram.com" if token.startswith("IG") else GRAPH


def resolve_user(ver: str, token: str) -> str:
    """IG_USER_ID if given, else discover the Instagram professional account from the token."""
    if _env("IG_USER_ID"):
        return _env("IG_USER_ID")
    base = _base(token)
    if base != GRAPH:
        me = _check(requests.get(f"{base}/{ver}/me", params={"fields": "user_id,username",
                                                              "access_token": token}, timeout=30))
        print(f"   Instagram account: @{me.get('username')}")
        return str(me.get("user_id") or me["id"])
    # Page token: the Page itself has the linked account
    r = requests.get(f"{GRAPH}/{ver}/me", params={"fields": "name,instagram_business_account{id,username}",
                                                   "access_token": token}, timeout=30).json()
    if isinstance(r, dict) and r.get("instagram_business_account"):
        ig = r["instagram_business_account"]
        print(f"   Instagram account: @{ig.get('username')} (via Page {r.get('name')})")
        return ig["id"]
    # User token: look through the user's Pages
    pages = _check(requests.get(f"{GRAPH}/{ver}/me/accounts", params={
        "fields": "name,instagram_business_account{id,username}", "access_token": token}, timeout=30))
    for p in pages.get("data", []):
        if p.get("instagram_business_account"):
            ig = p["instagram_business_account"]
            print(f"   Instagram account: @{ig.get('username')} (via Page {p.get('name')})")
            return ig["id"]
    raise RuntimeError("No Instagram professional account linked to a Facebook Page was found for this token")


def _warn_expiry(ver: str, token: str) -> None:
    if _base(token) != GRAPH:
        return
    try:
        d = requests.get(f"{GRAPH}/{ver}/debug_token", params={"input_token": token, "access_token": token},
                         timeout=30).json().get("data", {})
        exp = d.get("expires_at") or 0
        if exp and exp - time.time() < 7 * 86400:
            print(f"::warning::Instagram token expires in {(exp - time.time()) / 3600:.1f} h. Run "
                  "scripts/get_instagram_token.py to get a permanent Page token and save it as IG_ACCESS_TOKEN.")
    except Exception:  # noqa: BLE001
        pass


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
    token = _token()
    _warn_expiry(ver, token)
    ig_user = resolve_user(ver, token)
    G = _base(token)

    # 1. container
    c = _check(requests.post(f"{G}/{ver}/{ig_user}/media", data={
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
        st = _check(requests.get(f"{G}/{ver}/{cid}", params={
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
    pub = _check(requests.post(f"{G}/{ver}/{ig_user}/media_publish", data={
        "creation_id": cid, "access_token": token}, timeout=60))
    mid = pub["id"]
    try:
        link = _check(requests.get(f"{G}/{ver}/{mid}", params={
            "fields": "permalink", "access_token": token}, timeout=30)).get("permalink", mid)
    except Exception:  # noqa: BLE001
        link = mid
    print(f"✅ Instagram reel: {link}")
    return link
