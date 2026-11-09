"""Upload to YouTube with the free Data API v3 (OAuth refresh token stored as a secret)."""
from __future__ import annotations

import os
import time

SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]


def have_credentials() -> bool:
    return all(os.environ.get(k) for k in ("YT_CLIENT_ID", "YT_CLIENT_SECRET", "YT_REFRESH_TOKEN"))


def _service():
    from google.oauth2.credentials import Credentials
    from googleapiclient.discovery import build

    def secret(name: str) -> str:
        # tolerate copy-paste slips: whitespace/newlines, quotes, or a pasted "NAME = value" line
        v = os.environ[name].strip().strip('"\'')
        if "=" in v and v.split("=", 1)[0].strip().upper() == name:
            v = v.split("=", 1)[1].strip()
        return v

    cid = secret("YT_CLIENT_ID")
    if not cid.endswith(".apps.googleusercontent.com"):
        print("   ! YT_CLIENT_ID doesn't look like an OAuth client ID (should end in .apps.googleusercontent.com)")
    creds = Credentials(
        None,
        refresh_token=secret("YT_REFRESH_TOKEN"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=cid,
        client_secret=secret("YT_CLIENT_SECRET"),
        scopes=SCOPES,
    )
    return build("youtube", "v3", credentials=creds, cache_discovery=False)


def _clean(s: str, n: int) -> str:
    return s.replace("<", "").replace(">", "")[:n]


def upload(video: str, thumb: str | None, title: str, description: str, tags: list[str], cfg: dict,
           language: str = "en") -> str:
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    yt = _service()
    y = cfg["youtube"]
    # YouTube caps total tag length at 500 chars
    kept, total = [], 0
    for t in tags:
        t = _clean(t.lstrip("#"), 60)
        if t and total + len(t) + 1 <= 480:
            kept.append(t)
            total += len(t) + 1
    body = {
        "snippet": {
            "title": _clean(title, 100),
            "description": _clean(description, 4900),
            "tags": kept,
            "categoryId": y.get("category_id", "10"),
            "defaultLanguage": language,
            "defaultAudioLanguage": language,
        },
        "status": {
            "privacyStatus": y.get("privacy", "public"),
            "selfDeclaredMadeForKids": bool(y.get("made_for_kids", False)),
            "containsSyntheticMedia": True,  # required AI disclosure — keep this True
            "embeddable": True,
        },
    }
    req = yt.videos().insert(part="snippet,status", body=body,
                             media_body=MediaFileUpload(video, chunksize=8 * 1024 * 1024, resumable=True,
                                                        mimetype="video/mp4"))
    resp, retries = None, 0
    while resp is None:
        try:
            status, resp = req.next_chunk()
            if status:
                print(f"   upload {status.progress():.0%}")
        except HttpError as e:
            if e.resp.status in (500, 502, 503, 504) and retries < 6:
                retries += 1
                time.sleep(2 ** retries)
                continue
            raise
    vid = resp["id"]
    print(f"✅ uploaded: https://youtu.be/{vid}")
    if thumb:
        try:
            yt.thumbnails().set(videoId=vid, media_body=MediaFileUpload(thumb, mimetype="image/jpeg")).execute()
        except HttpError as e:  # custom thumbnails need a phone-verified channel
            print(f"   (thumbnail not set: {e.reason if hasattr(e, 'reason') else e})")
    return vid
