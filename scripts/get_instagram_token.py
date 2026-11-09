"""Run ONCE on your own computer to get the two Instagram secrets (see branding/INSTAGRAM_SETUP.md).

    python get_instagram_token.py

It asks for your Meta App ID, App Secret and a short-lived token from the Graph API Explorer,
then finds your Facebook Page + linked Instagram account and produces a Page token that does not expire.
Values are typed at prompts (not on the command line) so they don't end up in your shell history.
"""
import os
import sys
from getpass import getpass

import requests

V = "v23.0"
G = f"https://graph.facebook.com/{V}"


def get(url, **params):
    r = requests.get(url, params=params, timeout=30)
    data = r.json()
    if "error" in data:
        sys.exit(f"\nMeta API error: {data['error'].get('message')}")
    return data


app_id = input("Meta App ID: ").strip()
app_secret = getpass("Meta App Secret (hidden): ").strip()
short = getpass("Short-lived user token from Graph API Explorer (hidden): ").strip()

long_user = get(f"{G}/oauth/access_token", grant_type="fb_exchange_token", client_id=app_id,
                client_secret=app_secret, fb_exchange_token=short)["access_token"]

pages = get(f"{G}/me/accounts", access_token=long_user, fields="id,name,access_token").get("data", [])
if not pages:
    sys.exit("No Facebook Pages found. Did you tick your Page when generating the token?")
for k, p in enumerate(pages):
    print(f"  [{k}] {p['name']}")
page = pages[int(input("Which Page is linked to the Instagram account? number: ") or 0)] if len(pages) > 1 else pages[0]

ig = get(f"{G}/{page['id']}", fields="instagram_business_account{id,username}",
         access_token=page["access_token"]).get("instagram_business_account")
if not ig:
    sys.exit(f"Page '{page['name']}' has no linked Instagram professional account. Link it first (guide step 2).")

info = get(f"{G}/debug_token", input_token=page["access_token"], access_token=f"{app_id}|{app_secret}")["data"]
expires = info.get("expires_at", 0)
print(f"\nInstagram account: @{ig.get('username')}  (id {ig['id']})")
print("Token expiry:", "never" if not expires else expires)

values = [("IG_USER_ID", ig["id"]), ("IG_ACCESS_TOKEN", page["access_token"])]
out = os.path.join(os.path.expanduser("~"), "Downloads", "instagram_secrets_DELETE_AFTER_USE.txt")
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as fh:
    fh.write("Add each value as a GitHub secret at:\n"
             "https://github.com/geekyarya/ai-music-channel/settings/secrets/actions/new\n"
             "Name = the line in CAPITALS, Secret = the line under it. Then DELETE this file.\n\n")
    for name, value in values:
        fh.write(f"{name}\n{value}\n\n")
print(f"\nSaved to: {out}")
try:
    os.startfile(out)  # opens in Notepad on Windows
except AttributeError:
    pass
