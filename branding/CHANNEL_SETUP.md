# YouTube channel setup

Do these once, in order. Steps 1–3 happen on YouTube; steps 4–8 connect the
GitHub pipeline so it can upload.

## 1. Create the channel
1. Sign in to <https://www.youtube.com> with the Google account you want to own the channel.
2. Go to <https://www.youtube.com/account> → **Add or manage your channel(s)** → **Create a channel**.
   This makes a *Brand Account* channel with its own name, separate from your personal name.
3. Name: **Neon Ghost Radio** (or your own — then update `channel.name` / `channel.artist` in `config.yaml` and re-run `python scripts/make_branding.py`).
4. Handle: e.g. `@NeonGhostRadio`.

## 2. Branding (YouTube Studio → Customization → Branding)
- **Picture**: `branding/avatar.png`
- **Banner image**: `branding/banner.png`
- **Video watermark** (optional): `branding/avatar.png`

## 3. Basic info (YouTube Studio → Customization → Basic info)
**Description** (paste):

> A brand-new AI-made song every single day, in English, Hindi and Hinglish.
> Rap, pop, desi hip hop, Bollywood, lo-fi, rock, synthwave and more, each with its own audio-reactive music video and karaoke lyrics.
>
> 🤖 Everything here is created by AI: the lyrics, the vocals, the music and the visuals. Every song is original.
>
> Har din ek naya AI gaana. Subscribe karo aur daily drop miss mat karo! 🔔

**Links**: add your GitHub repo if you want to show how it's made.

**Studio → Settings → Channel → Advanced settings**:
- Audience: **No, not made for kids**
- Country: India (or your own)
- Keywords: `ai music, ai song, hindi song, hindi rap, desi hip hop, bollywood, new song, ai generated music, hinglish`

**Studio → Settings → Upload defaults**:
- Category: **Music**
- Video language: leave blank (the pipeline sets English or Hindi on each upload)

## 4. Verify the channel (needed for custom thumbnails)
<https://www.youtube.com/verify> → phone number → SMS code. Without this, the API cannot set thumbnails.

## 5. Google Cloud project + YouTube API
1. <https://console.cloud.google.com/projectcreate> → name it `ai-music-channel` → Create.
2. <https://console.cloud.google.com/apis/library/youtube.googleapis.com> → **Enable**.
3. **OAuth consent screen** (<https://console.cloud.google.com/auth/overview>):
   - User type: **External**, app name `AI Music Uploader`, your email for support and developer contact.
   - **Audience** → **Publish app** → status **In production**.
     ⚠️ If it stays in "Testing", the login expires after 7 days and uploads stop.
4. **Clients** (<https://console.cloud.google.com/auth/clients>) → **Create client** → type **Desktop app** → Create → **Download JSON**.

## 6. Get the refresh token (on your own computer)
```bash
pip install google-auth-oauthlib
```
```bash
python scripts/get_youtube_token.py path/to/client_secret_XXXX.json
```
A browser opens. Pick the **channel's** account (choose the Neon Ghost Radio brand account if asked).
On "Google hasn't verified this app", click **Advanced → Go to AI Music Uploader (unsafe)**, then **Allow**. It's your own app.
It prints three values.

## 7. Add the GitHub secrets
<https://github.com/geekyarya/ai-music-channel/settings/secrets/actions/new>, one each:
- `YT_CLIENT_ID`
- `YT_CLIENT_SECRET`
- `YT_REFRESH_TOKEN`

## 8. Request the API audit (so videos can be public)
Until this is approved, API uploads from a new Cloud project are **locked to private**.
Form: <https://support.google.com/youtube/contact/yt_api_form>, choosing "I would like to request an API audit".
Describe it as: *"Personal automation that uploads one AI-generated music video per day to my own channel. Only my own channel; no other users; uses videos.insert and thumbnails.set."*
It usually takes a few days to a few weeks. Uploads keep working meanwhile (as private), and you can review them in Studio.
