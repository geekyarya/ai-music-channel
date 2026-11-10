# Runs ON KAGGLE (free GPU), pushed by src/music/kaggle_gpu.py. Generates one song with ACE-Step.
# The line below is replaced with the song's parameters (base64 JSON) before each push.
PARAMS_B64 = "__PARAMS_B64__"

import base64
import json
import os
import subprocess
import sys
import time

t0 = time.time()
P = json.loads(base64.b64decode(PARAMS_B64).decode("utf-8"))
OUT = "/kaggle/working"
print("params:", {k: (v if k != "lyrics" else f"{len(v)} chars") for k, v in P.items()}, flush=True)


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


log("installing ACE-Step…")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input",
                "git+https://github.com/ace-step/ACE-Step.git"], check=True)

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
major = torch.cuda.get_device_capability(0)[0] if torch.cuda.is_available() else 0
log(f"device: {gpu} (compute capability {major}.x)")

# bfloat16 needs Ampere (8.x)+. Kaggle's T4 (7.5) / P100 (6.0): try float16, fall back to float32 + offload.
if major >= 8:
    attempts = [("bfloat16", False)]
else:
    attempts = [("float16", False), ("float32", True)]


def good_audio(path):
    try:
        x, sr = sf.read(path)
    except Exception as e:  # noqa: BLE001
        log(f"can't read output: {e}")
        return False
    if x.size == 0 or not np.isfinite(x).all():
        log("output has NaN/inf or is empty")
        return False
    rms = float(np.sqrt(np.mean(np.square(x))))
    log(f"output: {len(x) / sr:.1f}s, rms={rms:.4f}")
    return rms > 0.005  # not silent


from acestep.pipeline_ace_step import ACEStepPipeline  # noqa: E402

done = False
for dtype, offload in attempts:
    os.environ["ACE_PIPELINE_DTYPE"] = dtype
    log(f"loading pipeline dtype={dtype} cpu_offload={offload}")
    try:
        pipe = ACEStepPipeline(dtype="bfloat16" if dtype == "bfloat16" else "float32",
                               cpu_offload=offload, torch_compile=False)
        save = os.path.join(OUT, "song_raw.wav")
        pipe(
            format="wav",
            audio_duration=float(P["duration"]),
            prompt=P["prompt"],
            lyrics=P["lyrics"],
            infer_step=int(P["infer_steps"]),
            guidance_scale=float(P["guidance_scale"]),
            scheduler_type="euler",
            cfg_type="apg",
            omega_scale=10.0,
            manual_seeds=str(P["seed"]),
            guidance_interval=0.5,
            guidance_interval_decay=0.0,
            min_guidance_scale=3.0,
            use_erg_tag=True,
            use_erg_lyric=False,
            use_erg_diffusion=True,
            oss_steps=None,
            guidance_scale_text=0.0,
            guidance_scale_lyric=0.0,
            save_path=save,
        )
        if good_audio(save):
            os.replace(save, os.path.join(OUT, "song.wav"))
            done = True
            break
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        log(f"attempt {dtype} failed: {e}")
    finally:
        try:
            del pipe
        except NameError:
            pass
        torch.cuda.empty_cache()

with open(os.path.join(OUT, "result.json"), "w") as fh:
    json.dump({"ok": done, "gpu": gpu, "seconds": round(time.time() - t0)}, fh)
log("DONE" if done else "FAILED")
if not done:
    sys.exit(1)
