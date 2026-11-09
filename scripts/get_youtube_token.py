"""Run ONCE on your own computer to get a YouTube refresh token.

    pip install google-auth-oauthlib
    python scripts/get_youtube_token.py path/to/client_secret.json

A browser opens → log in with the Google account that owns the channel → allow.
Copy the three printed values into GitHub → Settings → Secrets and variables → Actions.
"""
import json
import sys

from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]

import glob
import os

arg = sys.argv[1] if len(sys.argv) > 1 else "client_secret*.json"
# Windows cmd doesn't expand ~ or wildcards, so do it here.
matches = sorted(glob.glob(os.path.expandvars(os.path.expanduser(arg))))
if not matches:
    sys.exit(f"No file matches {arg!r}. Pass the full path to the downloaded client_secret JSON.")
path = matches[-1]
print(f"Using {path}")
flow = InstalledAppFlow.from_client_secrets_file(path, SCOPES)
creds = flow.run_local_server(port=0, access_type="offline", prompt="consent")
info = json.load(open(path))
app = info.get("installed") or info.get("web")
print("\nAdd these GitHub Actions secrets:\n")
print(f"YT_CLIENT_ID      = {app['client_id']}")
print(f"YT_CLIENT_SECRET  = {app['client_secret']}")
print(f"YT_REFRESH_TOKEN  = {creds.refresh_token}")
