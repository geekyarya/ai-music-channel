# Instagram Reels setup (free)

Every day the pipeline cuts a 45-second vertical clip (starting just before the first chorus) and posts it
as an Instagram Reel through Meta's free Instagram Graph API. You need an Instagram **professional**
account linked to a **Facebook Page**, and a Meta developer app. All free, no card.

## 1. Instagram professional account
1. Create the Instagram account (e.g. `@neonghostradio`). Use `branding/avatar.png` as the profile picture.
2. Instagram → **Settings and activity → Account type and tools → Switch to professional account**
   → **Creator** → category **Musician/band** (or Artist).
3. Bio idea: `New song every day 🌙 English • Hindi • Hinglish` and the YouTube channel link.

## 2. Facebook Page linked to it
1. <https://www.facebook.com/pages/create> → name **Neon Ghost Radio**, category Musician/Band.
2. Link Instagram: on the Page → **Settings → Linked accounts → Instagram → Connect**
   (or in Instagram: **Accounts Center → Accounts → Add accounts**).

## 3. Meta developer app
1. <https://developers.facebook.com/apps> → **Create app**.
   - Use case: **Other** → app type **Business** → name **NGR Uploader** → create.
   - Leave it in **Development** mode. That's fine for posting to your own accounts; no app review needed.
2. In the app dashboard → **Add product** → **Instagram** → choose **API setup with Facebook login**.
3. **App settings → Basic**: note the **App ID** and **App Secret**. Set the Privacy Policy URL to
   `https://geekyarya.github.io/ai-music-channel/privacy.html` and save.

## 4. Get a token in the Graph API Explorer
1. Open <https://developers.facebook.com/tools/explorer>.
2. Top right: **Meta App** = NGR Uploader. **User or Page** = User Token.
3. **Permissions**, add: `pages_show_list`, `pages_read_engagement`, `instagram_basic`,
   `instagram_content_publish`, `business_management`.
4. **Generate Access Token** → in the popup choose the **Neon Ghost Radio** Page and Instagram account → allow.
5. Copy the token (it's short-lived, about 1 hour, so do step 5 straight away).

## 5. Turn it into permanent secrets
On your computer:
```
curl -L -o get_instagram_token.py https://github.com/geekyarya/ai-music-channel/raw/main/scripts/get_instagram_token.py
pip install requests
python get_instagram_token.py
```
Paste the App ID, App Secret and token when asked. It finds your Page and Instagram account and opens a
Notepad file with two values (the Page token it creates **doesn't expire**). Add them at
<https://github.com/geekyarya/ai-music-channel/settings/secrets/actions/new>:
- `IG_USER_ID`
- `IG_ACCESS_TOKEN`

Then delete the Notepad file.

## Notes
- Turn Reels off any time: `instagram.enabled: false` in `config.yaml`. YouTube Shorts: `reels.youtube_shorts`.
- Instagram allows up to 50 API posts per day; we post 1.
- If you ever change your Facebook password or remove the app, generate new secrets with step 4–5.
- Meta asks creators to label realistic AI-generated audio/video ("AI info"). The Graph API doesn't
  appear to offer that label, so check the posted Reel in the Instagram app and add the label there if
  Meta's policy applies to your content.
