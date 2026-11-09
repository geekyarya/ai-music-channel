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
values = [("YT_CLIENT_ID", app["client_id"]), ("YT_CLIENT_SECRET", app["client_secret"]),
          ("YT_REFRESH_TOKEN", creds.refresh_token)]
print("\nAdd these GitHub Actions secrets:\n")
for name, value in values:
    print(f"{name:<18}= {value}")

# Also write them to a text file and open it, so they're easy to copy (Command Prompt makes copying awkward).
out = os.path.join(os.path.dirname(os.path.abspath(path)), "youtube_secrets_DELETE_AFTER_USE.txt")
with open(out, "w") as fh:
    fh.write("Add each value below as a GitHub secret at:\n"
             "https://github.com/geekyarya/ai-music-channel/settings/secrets/actions/new\n"
             "Name = the line in CAPITALS, Secret = the line under it. Then DELETE this file.\n\n")
    for name, value in values:
        fh.write(f"{name}\n{value}\n\n")
print(f"\nAlso saved to: {out}")
try:
    os.startfile(out)  # opens in Notepad on Windows
except AttributeError:
    pass
