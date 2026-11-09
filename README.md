# 🎵 AI Music Channel: fully automated, 100% free

Three times a day (10:07 AM, 3:07 PM, 8:07 PM India time), GitHub Actions:

1. **Writes an original song in English, Hindi or Hinglish, sometimes rap, sometimes a sung song.** An LLM picks a concept and writes lyrics, hook, tempo, key, mood, a visual style and a color palette. Genres rotate: rap, trap, drill, lo-fi, pop, rock, EDM, R&B, synthwave, afrobeats, country, and so on.
2. **Makes the music with real AI vocals** using [ACE-Step](https://github.com/ace-step/ACE-Step), an open-source text-to-song model, on a free Hugging Face GPU.
3. **Renders a 1080p music video.** It's audio-reactive (tunnel, plasma, kaleidoscope, starfield, synthwave grid, or light waves), uses the song's palette, pulses with the bass, and shows karaoke lyrics synced with Whisper.
4. **Makes a thumbnail and uploads to YouTube** with title, lyrics, tags, and the AI-content disclosure.
5. **Cuts a 45-second vertical reel** (from just before the first chorus) and posts it as a **YouTube Short** and an **Instagram Reel** (setup: [branding/INSTAGRAM_SETUP.md](branding/INSTAGRAM_SETUP.md)).
6. **Logs it** in `data/history.json` so it doesn't repeat genres or themes.

| Piece | Free service | Cost |
|---|---|---|
| Lyrics / ideas | Google Gemini free tier (`GEMINI_API_KEY`), with Groq or GitHub Models as fallbacks | $0 |
| Singing + music | ACE-Step on Hugging Face ZeroGPU (free account) | $0 |
| Fallback music | Beat synthesized on the runner + Edge neural TTS rap (espeak-ng as backup) | $0 |
| Lyric timing | faster-whisper on CPU | $0 |
| Video | numpy + Pillow + ffmpeg on the runner | $0 |
| Hosting / scheduling | GitHub Actions (unlimited minutes on public repos) | $0 |
| Upload | YouTube Data API v3 (10k quota/day; one upload ≈ 1.6k) | $0 |

If the free GPU quota runs out, the pipeline switches to the local engine, so **a video still goes out every day.**
That fallback raps or speaks the lyrics over a synthesized beat. It sounds noticeably worse than ACE-Step.

---

## Setup (about 20 minutes, once)

> 📺 Creating the YouTube channel itself (name, avatar, banner, description) is covered step by step in **[branding/CHANNEL_SETUP.md](branding/CHANNEL_SETUP.md)**. Channel art is in `branding/`.

### 1. Create the repo
Push this folder to a **public** GitHub repo. Public repos get unlimited Actions minutes. A private repo gets 2,000 minutes a month, which also works but is tighter.

### 2. Hugging Face token (for the AI singing)
1. Sign up at <https://huggingface.co> (free).
2. Go to Settings → Access Tokens → **New token** (type: *Read*).
3. In GitHub, go to repo → Settings → Secrets and variables → Actions → **New repository secret**: `HF_TOKEN`.

### 3. YouTube upload access
1. Create the channel on YouTube.
2. At <https://console.cloud.google.com>, create a project → **APIs & Services → Library** → enable **YouTube Data API v3**.
3. **OAuth consent screen**: External, add yourself as a test user, then **Publish app → In production**.
   ⚠️ If you skip publishing, the token **expires after 7 days** and uploads stop.
4. **Credentials → Create credentials → OAuth client ID → Desktop app**, then download the JSON.
5. On your own computer, run:
   ```bash
   pip install google-auth-oauthlib
   ```
   ```bash
   python scripts/get_youtube_token.py client_secret_XXXX.json
   ```
   Log in with the channel's Google account and click through the "unverified app" warning (it's your own app).
6. Add the three printed values as secrets: `YT_CLIENT_ID`, `YT_CLIENT_SECRET`, `YT_REFRESH_TOKEN`.

> ⚠️ **Important YouTube rule:** videos uploaded through the API from an *unaudited* Google Cloud project are **locked to private**.
> To post publicly, submit the free **[YouTube API Services audit form](https://support.google.com/youtube/contact/yt_api_form)**
> and describe it as "personal automation uploading my own AI-generated music videos to my own channel."
> Approval usually takes a few days to a few weeks. Until then, the videos upload as private, and you can still review them in YouTube Studio.

### 4. Lyrics AI key (free)
Create a key at <https://aistudio.google.com/apikey> and add it as the `GEMINI_API_KEY` secret. `GROQ_API_KEY` (<https://console.groq.com/keys>) is an optional backup.
Without either, the channel still posts daily but reuses built-in lyrics.

### 5. Test it
Go to **Actions → Daily AI Song → Run workflow**, untick *Upload*, and run it.
When it finishes, download the `song-N` artifact to watch the video. Once you're happy with it, leave it alone and it posts daily.

---

## Customize
Everything is in **`config.yaml`**:
- `channel.disclose_ai`: `false` keeps "AI" out of titles, descriptions, tags and channel art so viewers can guess; `true` is reveal mode. YouTube's synthetic-content flag is always set either way, as YouTube's policy requires.
- `channel.name` / `artist`: your channel's branding
- `genres`: add or remove genres and their sound descriptions
- `song.duration_seconds`: song length (ACE-Step)
- `video`: resolution, fps, captions
- `youtube.privacy`, tags, category
- Posting times: the `cron:` line in `.github/workflows/daily-song.yml` (UTC). 3 runs × (song + Short) = 6 uploads, YouTube's default daily API limit.

Drop any `.ttf` into `assets/fonts/` to change typography.

## Run locally
```bash
pip install -r requirements.txt
```
```bash
python src/main.py --offline --backend local --no-upload --seconds 45
```
- `--offline`: no LLM; uses built-in test lyrics
- `--backend ace_step|local`: forces a music engine
- `--genre trap`: forces a genre
- `--seconds N`: renders a short preview

Output goes to `out/<date>-<title>/`: `song.wav`, `video.mp4`, `thumbnail.jpg`, `spec.json`.

## Layout
```
src/main.py          orchestrator
src/songwriter.py    concept + lyrics + visual direction (LLM)
src/llm.py           free LLM provider chain
src/music/ace_step.py   AI singing via Hugging Face Space
src/music/local.py      CPU beat synth + TTS rap fallback
src/music/align.py      Whisper lyric → timestamp alignment
src/video.py         audio-reactive visuals, karaoke, thumbnail
src/youtube.py       uploader
```

## Good to know
- **AI disclosure:** uploads set `containsSyntheticMedia: true`, as YouTube requires for realistic AI content. Keep it on.
- **Monetization:** YouTube's YPP policy on *inauthentic / mass-produced content* can deny monetization to channels of templated AI uploads. Variety (the genre rotation, unique visuals) helps. Adding your own touch, like curation, commentary, or a recurring "AI artist" story, helps more.
- **Copyright:** lyrics are prompted to be original and never imitate real artists. ACE-Step is Apache-2.0 licensed.
- **Free GPU quota** is shared and can change. If ACE-Step fails, the local engine covers that day. You can add more Spaces to `music.ace_step.spaces`.
- GitHub pauses scheduled workflows on repos with no activity for 60 days. The daily history commit prevents that.
