# Runs ON KAGGLE (free T4 GPU), pushed by src/kaggle_run.py for hero_main.py.
# 1) draws every shot of the episode with SDXL (same style + character sheet → consistent cast)
# 2) animates each still into a ~4 s clip with LTX-Video image-to-video
# Anything that fails degrades gracefully: a shot without a clip is animated later with camera moves.
PARAMS_B64 = "__PARAMS_B64__"

import base64
import gc
import io
import json
import os
import subprocess
import sys
import time

t0 = time.time()
P = json.loads(base64.b64decode(PARAMS_B64).decode("utf-8"))
OUT = "/kaggle/working"
RESULT = {"images": [], "clips": [], "notes": []}


def log(msg):
    print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)


def save_result():
    RESULT["seconds"] = round(time.time() - t0)
    json.dump(RESULT, open(os.path.join(OUT, "result.json"), "w"), indent=1)


def pip(*args):
    p = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-input", *args],
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-3000:], flush=True)
        raise SystemExit(f"pip install failed: {args}")


log("installing…")
pip("-U", "diffusers>=0.34", "accelerate>=1.6", "sentencepiece", "imageio", "imageio-ffmpeg", "protobuf")
log("installed")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
RESULT["gpu"] = gpu
log(f"device: {gpu}")
shots = P["shots"]


def free():
    gc.collect()
    torch.cuda.empty_cache()


# ─── 1. stills with SDXL ──────────────────────────────────────────────────────
from diffusers import AutoencoderKL, StableDiffusionXLPipeline  # noqa: E402

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
ref, enc = None, None
if P.get("ref_b64"):
    from transformers import CLIPVisionModelWithProjection
    enc = CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter", subfolder="models/image_encoder",
                                                        torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", vae=vae,
                                                 image_encoder=enc, torch_dtype=torch.float16, variant="fp16",
                                                 use_safetensors=True).to("cuda")
if P.get("ref_b64"):
    try:
        # IP-Adapter Plus copies the reference's look (face, shades, costume) into shots where the hero appears.
        pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models",
                             weight_name="ip-adapter-plus_sdxl_vit-h.safetensors")
        ref = Image.open(io.BytesIO(base64.b64decode(P["ref_b64"]))).convert("RGB")
        log("using the hero reference image (IP-Adapter)")
    except Exception as e:  # noqa: BLE001
        RESULT["notes"].append(f"ip-adapter failed: {e}")
        log(f"IP-Adapter unavailable ({e}); continuing with text only")
W, H = P.get("image_size", [768, 1344])
for i, s in enumerate(shots):
    kw = {}
    if ref is not None:
        pipe.set_ip_adapter_scale(float(P.get("ref_scale", 0.6)) if s.get("hero") else 0.0)
        kw = {"ip_adapter_image": ref}
    img = pipe(prompt=s["p1"], prompt_2=s["p2"], negative_prompt=P["negative"], width=W, height=H,
               num_inference_steps=int(P.get("image_steps", 30)), guidance_scale=6.5,
               generator=torch.Generator("cuda").manual_seed(int(s["seed"])), **kw).images[0]
    img.save(os.path.join(OUT, f"shot_{i}.png"))
    RESULT["images"].append(i)
    log(f"image {i + 1}/{len(shots)} done")
    save_result()
del pipe, vae
free()

# ─── 2. motion with LTX-Video ─────────────────────────────────────────────────
budget = float(P.get("video_budget_minutes", 80)) * 60
if P.get("mode") == "ref":
    P["video"] = False
if P.get("video", True):
    try:
        from diffusers import LTXImageToVideoPipeline
        from diffusers.utils import export_to_video
        from transformers import T5EncoderModel

        repo = P.get("ltx_repo", "Lightricks/LTX-Video")
        # T5-XXL overflows in float16 (T4 has no fast bfloat16), so encode all prompts on the CPU in bfloat16.
        log("encoding video prompts…")
        te = T5EncoderModel.from_pretrained(repo, subfolder="text_encoder", torch_dtype=torch.bfloat16)
        vpipe = LTXImageToVideoPipeline.from_pretrained(repo, text_encoder=te, torch_dtype=torch.float16)
        cpu = torch.device("cpu")
        embeds = []
        with torch.no_grad():
            for s in shots:
                pe, pm, ne, nm = vpipe.encode_prompt(
                    s["vp"], negative_prompt="worst quality, inconsistent motion, blurry, jittery, distorted",
                    do_classifier_free_guidance=True, device=cpu, dtype=torch.bfloat16, max_sequence_length=256)
                embeds.append((pe, pm, ne, nm))
        vpipe.text_encoder = None
        del te
        free()

        def load(dtype):
            vpipe.transformer.to(dtype)
            vpipe.vae.to(dtype)
            vpipe.to("cuda")
            return dtype

        dtype = load(torch.float16)
        log("LTX-Video ready (float16)")
        VW, VH = P.get("video_size", [480, 832])
        i = 0
        while i < len(shots):
            if time.time() - t0 > budget:
                RESULT["notes"].append(f"video time budget reached at shot {i}")
                log("time budget reached; remaining shots stay still images")
                break
            img = Image.open(os.path.join(OUT, f"shot_{i}.png")).convert("RGB").resize((VW, VH), Image.LANCZOS)
            pe, pm, ne, nm = (x.to("cuda", dtype) if x.is_floating_point() else x.to("cuda") for x in embeds[i])
            with torch.no_grad():
                frames = vpipe(image=img, prompt_embeds=pe, prompt_attention_mask=pm, negative_prompt_embeds=ne,
                               negative_prompt_attention_mask=nm, width=VW, height=VH,
                               num_frames=int(P.get("num_frames", 97)),
                               num_inference_steps=int(P.get("video_steps", 30)), guidance_scale=3.0,
                               decode_timestep=0.03, decode_noise_scale=0.025,
                               generator=torch.Generator("cuda").manual_seed(int(shots[i]["seed"]))).frames[0]
            arr = np.stack([np.asarray(f, dtype=np.float32) for f in frames])
            ok = np.isfinite(arr).all() and arr.std() > 4 and arr.mean() > 8
            if ok:  # reject clips that melt into mush: sharpness must stay near the first frame's
                def sharp(a):
                    g = a.mean(axis=2)
                    return float(np.var(g[1:-1, 1:-1] * 4 - g[:-2, 1:-1] - g[2:, 1:-1] - g[1:-1, :-2] - g[1:-1, 2:]))
                s0 = sharp(arr[0]) + 1e-6
                worst = min(sharp(arr[k]) for k in range(len(arr) // 4, len(arr), max(1, len(arr) // 8))) / s0
                log(f"clip {i + 1}: sharpness kept {worst:.0%}")
                if worst < float(P.get("min_sharpness", 0.45)):
                    RESULT["notes"].append(f"shot {i}: clip went blurry ({worst:.0%}), using the still")
                    i += 1
                    continue
            if not ok:
                if dtype == torch.float16:
                    log("float16 produced a bad clip; switching LTX to float32")
                    RESULT["notes"].append("ltx float16 bad → float32")
                    dtype = load(torch.float32)
                    continue  # retry this shot
                RESULT["notes"].append(f"shot {i}: bad clip, skipped")
                i += 1
                continue
            export_to_video(frames, os.path.join(OUT, f"shot_{i}.mp4"), fps=24)
            RESULT["clips"].append(i)
            RESULT["ltx_dtype"] = str(dtype)
            log(f"clip {i + 1}/{len(shots)} done")
            save_result()
            i += 1
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        RESULT["notes"].append(f"video model failed: {type(e).__name__}: {e}"[:500])
save_result()
log(f"finished: {len(RESULT['images'])} images, {len(RESULT['clips'])} clips")
