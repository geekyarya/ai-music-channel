"""Full AI songs with real sung/rapped vocals via the open-source ACE-Step model,
running on a free Hugging Face ZeroGPU Space.

Needs an HF_TOKEN (free account): anonymous ZeroGPU quota is shared and almost
always empty, but a logged-in free account gets a daily GPU allowance that
easily covers one song per day.
"""
from __future__ import annotations

import os
import random
import shutil
import subprocess

from songwriter import lyrics_text


def generate(spec: dict, cfg: dict, out_dir: str) -> dict:
    from gradio_client import Client

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN not set — ACE-Step needs a (free) Hugging Face token")

    a = cfg["music"]["ace_step"]
    vocal = f"{spec['vocal']} vocals"
    prompt = f"{spec['music_tags']}, {vocal}, {spec['bpm']} BPM, {spec['mood']}"
    errors = []
    for space in a["spaces"]:
        try:
            print(f"  → ACE-Step on {space}: {prompt}")
            client = Client(space, token=token, verbose=False)
            result = client.predict(
                audio_duration=float(cfg["song"]["duration_seconds"]),
                prompt=prompt,
                lyrics=lyrics_text(spec),
                infer_step=int(a.get("infer_steps", 60)),
                guidance_scale=float(a.get("guidance_scale", 15.0)),
                scheduler_type="euler",
                cfg_type="apg",
                omega_scale=10.0,
                manual_seeds=str(random.randint(1, 2**31 - 1)),
                guidance_interval=0.5,
                guidance_interval_decay=0.0,
                min_guidance_scale=3.0,
                use_erg_tag=True,
                use_erg_lyric=False,
                use_erg_diffusion=True,
                oss_steps=None,
                guidance_scale_text=0.0,
                guidance_scale_lyric=0.0,
                audio2audio_enable=False,
                ref_audio_strength=0.5,
                ref_audio_input=None,
                lora_name_or_path="none",
                api_name="/__call__",
            )
            src = result[0] if isinstance(result, (list, tuple)) else result
            out = os.path.join(out_dir, "song.wav")
            # normalise whatever came back (wav/mp3/flac) to 44.1k stereo wav
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src,
                            "-ar", "44100", "-ac", "2", out], check=True)
            shutil.rmtree(os.path.dirname(src), ignore_errors=True)
            return {"audio": out, "timings": None, "engine": f"ACE-Step ({space})"}
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            errors.append(f"{space}: {type(e).__name__}: {e!r}")
    raise RuntimeError("ACE-Step failed: " + " | ".join(errors))
