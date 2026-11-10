"""Report what the saved Meta token can do, without posting anything."""
import os
import sys
import time

import requests

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from instagram import _token  # noqa: E402

G = "https://graph.facebook.com/v23.0"
tok = _token()
if not tok:
    sys.exit("No Instagram token secret is set.")
print("Token type:", "Instagram-login" if tok.startswith("IG") else "Facebook-login")

d = requests.get(f"{G}/debug_token", params={"input_token": tok, "access_token": tok}, timeout=30).json()
info = d.get("data", {})
if "error" in d or not info:
    print("debug_token:", d.get("error", d))
else:
    exp = info.get("expires_at", 0)
    print("Valid:", info.get("is_valid"))
    print("Kind:", info.get("type"))
    print("Expires:", "never" if not exp else f"in {(exp - time.time()) / 3600:.1f} hours")
    print("Permissions:", ", ".join(sorted(info.get("scopes", []))))
    need = {"pages_show_list", "pages_read_engagement", "pages_manage_posts", "instagram_basic",
            "instagram_content_publish"}
    missing = need - set(info.get("scopes", []))
    print("Missing permissions:", ", ".join(sorted(missing)) or "none")

me = requests.get(f"{G}/me", params={"fields": "id,name,instagram_business_account{username}",
                                      "access_token": tok}, timeout=30).json()
if "instagram_business_account" in me:
    print(f"This is a PAGE token for: {me.get('name')} → @{me['instagram_business_account'].get('username')}")
else:
    print("This is a USER token for:", me.get("name", me))
    acc = requests.get(f"{G}/me/accounts", params={"fields": "name,instagram_business_account{username}",
                                                    "access_token": tok, "limit": 100}, timeout=30).json()
    for p in acc.get("data", []):
        ig = (p.get("instagram_business_account") or {}).get("username")
        print(f"  Page: {p.get('name')}  →  Instagram: {'@' + ig if ig else '(none linked)'}")
    if "error" in acc:
        print("  /me/accounts error:", acc["error"].get("message"))
