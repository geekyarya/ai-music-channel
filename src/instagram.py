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


def _want(cfg: dict) -> str:
    u = (cfg.get("instagram", {}).get("username") or "").lstrip("@").lower()
    if not u:
        raise RuntimeError("Set instagram.username in config.yaml so we never post to the wrong account")
    return u


def resolve_user(ver: str, token: str, cfg: dict) -> str:
    """The Instagram account to post to. It MUST match instagram.username in config.yaml: one Meta token
    can manage several Pages/accounts, and we must never post to the wrong one."""
    want = _want(cfg)
    base = _base(token)
    found = []
    if base != GRAPH:
        me = _check(requests.get(f"{base}/{ver}/me", params={"fields": "user_id,username",
                                                              "access_token": token}, timeout=30))
        found.append(me.get("username"))
        if (me.get("username") or "").lower() == want:
            print(f"   Instagram account: @{me['username']}")
            return str(me.get("user_id") or me["id"])
    else:
        r = requests.get(f"{GRAPH}/{ver}/me", params={"fields": "name,instagram_business_account{id,username}",
                                                       "access_token": token}, timeout=30).json()
        cands = [r] if isinstance(r, dict) and "instagram_business_account" in r else []
        if not cands:  # user token: every Page it manages
            cands = _check(requests.get(f"{GRAPH}/{ver}/me/accounts", params={
                "fields": "name,instagram_business_account{id,username}", "access_token": token,
                "limit": 100}, timeout=30)).get("data", [])
        for p in cands:
            ig = p.get("instagram_business_account") or {}
            found.append(ig.get("username"))
            if (ig.get("username") or "").lower() == want:
                print(f"   Instagram account: @{ig['username']} (via Page {p.get('name')})")
                if _env("IG_USER_ID") and _env("IG_USER_ID") != ig["id"]:
                    raise RuntimeError("IG_USER_ID doesn't match @" + want)
                return ig["id"]
    raise RuntimeError(f"This token has no access to @{want} (it can reach: "
                       f"{', '.join('@' + f for f in found if f) or 'no Instagram accounts'}). "
                       "Regenerate the token and tick the Neon Ghost Radio Page + Instagram account.")


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


def ig_safe_copy(video_path: str) -> str:
    """Re-encode with Instagram's recommended Reels settings (H.264 High, steady ~4.5 Mbps, 2 s keyframes,
    AAC 48 kHz). Used when Instagram fails to process the original encode."""
    import subprocess
    out = os.path.splitext(video_path)[0] + "_ig.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video_path,
                    "-c:v", "libx264", "-profile:v", "high", "-level", "4.1", "-pix_fmt", "yuv420p",
                    "-r", "30", "-g", "60", "-keyint_min", "60", "-sc_threshold", "0",
                    "-b:v", "4500k", "-maxrate", "5000k", "-bufsize", "10000k",
                    "-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2",
                    "-movflags", "+faststart", out], check=True)
    return out


def _processing_failed(e: Exception) -> bool:
    return "ProcessingFailed" in str(e) or "processing failed" in str(e).lower()


def post_reel(video_path: str, caption: str, cfg: dict) -> str:
    """Direct (resumable) upload first; if Instagram can't process it, re-encode; if that fails too,
    host the file briefly as a public GitHub release asset and let Instagram fetch it by URL."""
    try:
        return _post_reel(video_path, caption, cfg)
    except RuntimeError as e:
        if not _processing_failed(e):
            raise
    print("   Instagram couldn't process the upload; retrying with Instagram's recommended settings…")
    safe = ig_safe_copy(video_path)
    time.sleep(20)
    try:
        return _post_reel(safe, caption, cfg)
    except RuntimeError as e:
        if not _processing_failed(e) or not os.environ.get("GITHUB_REPOSITORY"):
            raise
    print("   Direct upload still failing; posting via a temporary public link instead…")
    with _hosted(safe) as url:
        return _post_reel(safe, caption, cfg, video_url=url)


class _hosted:
    """Upload a file as a temporary GitHub release asset (public repo) and delete it afterwards."""

    def __init__(self, path: str):
        self.path = path
        self.tag = f"reel-tmp-{os.environ.get('GITHUB_RUN_ID', int(time.time()))}-{int(time.time())}"

    def _gh(self, *args):
        import subprocess
        env = dict(os.environ, GH_TOKEN=os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN", ""))
        return subprocess.run(["gh", *args, "--repo", os.environ["GITHUB_REPOSITORY"]], env=env,
                               capture_output=True, text=True)

    def __enter__(self) -> str:
        name = f"{self.tag}.mp4"
        import shutil
        tmp = os.path.join(os.path.dirname(self.path), name)
        shutil.copy(self.path, tmp)
        p = self._gh("release", "create", self.tag, tmp, "--prerelease", "--title", "temp reel",
                     "--notes", "Temporary file for Instagram; deleted automatically.")
        if p.returncode != 0:
            raise RuntimeError(f"couldn't host the reel on GitHub: {p.stderr.strip()[-300:]}")
        return f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/releases/download/{self.tag}/{name}"

    def __exit__(self, *exc):
        self._gh("release", "delete", self.tag, "--cleanup-tag", "--yes")
        return False


def _post_reel(video_path: str, caption: str, cfg: dict, video_url: str | None = None) -> str:
    ver = cfg.get("instagram", {}).get("graph_version", "v23.0")
    token = _token()
    _warn_expiry(ver, token)
    ig_user = resolve_user(ver, token, cfg)
    G = _base(token)

    # 1. container (+ 2. bytes, unless Instagram fetches the file from a public URL)
    data = {"media_type": "REELS", "caption": caption[:2200], "share_to_feed": "true", "access_token": token}
    if video_url:
        data["video_url"] = video_url
    else:
        data["upload_type"] = "resumable"
    c = _check(requests.post(f"{G}/{ver}/{ig_user}/media", data=data, timeout=60))
    cid = c["id"]
    if not video_url:
        upload_uri = c.get("uri") or f"https://rupload.facebook.com/ig-api-upload/{ver}/{cid}"
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
