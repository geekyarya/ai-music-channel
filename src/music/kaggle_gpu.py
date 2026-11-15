"""Generate the song with ACE-Step on a free Kaggle GPU (about 30 GPU-hours/week per account).

Flow: write a private Kaggle script kernel with this song's parameters → `kaggle kernels push`
(which runs it on a GPU) → poll its status → `kaggle kernels output` to download song.wav.
Needs KAGGLE_USERNAME and KAGGLE_KEY (from kaggle.com → Settings → API → Create New Token), and a
phone-verified Kaggle account (required for GPU + internet in kernels).
"""
from __future__ import annotations

import base64
import glob
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import time

from music.ace_step import build_inputs

HERE = os.path.dirname(os.path.abspath(__file__))


def have_credentials() -> bool:
    return bool(os.environ.get("KAGGLE_USERNAME", "").strip() and os.environ.get("KAGGLE_KEY", "").strip())


def _kaggle(*args, check=True) -> str:
    env = dict(os.environ, KAGGLE_USERNAME=os.environ["KAGGLE_USERNAME"].strip(),
               KAGGLE_KEY=os.environ["KAGGLE_KEY"].strip())
    p = subprocess.run(["kaggle", *args], capture_output=True, text=True, env=env)
    out = (p.stdout or "") + (p.stderr or "")
    if check and p.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args[:2])} failed: {out.strip()[-600:]}")
    return out


def generate(spec: dict, cfg: dict, out_dir: str) -> dict:
    if not have_credentials():
        raise RuntimeError("KAGGLE_USERNAME / KAGGLE_KEY not set")
    k = cfg["music"].get("kaggle", {})
    a = cfg["music"]["ace_step"]
    user = os.environ["KAGGLE_USERNAME"].strip()
    slug = k.get("kernel_slug", "ngr-ace-step")
    kid = f"{user}/{slug}"

    prompt, lyrics = build_inputs(spec, cfg)
    params = {"prompt": prompt, "lyrics": lyrics, "duration": float(cfg["song"]["duration_seconds"]),
              "infer_steps": int(a.get("infer_steps", 60)), "guidance_scale": float(a.get("guidance_scale", 15.0)),
              "seed": random.randint(1, 2**31 - 1)}
    code = open(os.path.join(HERE, "kaggle_kernel.py")).read().replace(
        "__PARAMS_B64__", base64.b64encode(json.dumps(params).encode()).decode())

    work = tempfile.mkdtemp(prefix="kaggle_")
    try:
        with open(os.path.join(work, "run.py"), "w") as fh:
            fh.write(code)
        with open(os.path.join(work, "kernel-metadata.json"), "w") as fh:
            json.dump({"id": kid, "title": slug, "code_file": "run.py", "language": "python",
                       "kernel_type": "script", "is_private": True, "enable_gpu": True,
                       "machine_shape": k.get("accelerator", "NvidiaTeslaT4"),
                       "enable_internet": True, "dataset_sources": [], "competition_sources": [],
                       "kernel_sources": []}, fh)
        print(f"  → ACE-Step on Kaggle GPU ({kid}): {prompt}")
        quota = _kaggle("quota", check=False).strip()
        print("   Kaggle quota: " + " | ".join(l.strip() for l in quota.splitlines() if l.strip())[:300])
        timeout_s = int(float(k.get("timeout_minutes", 80)) * 60)

        def status_now() -> str:
            out = _kaggle("kernels", "status", kid, check=False)
            m = re.search(r'status "([^"]+)"', out)
            return (m.group(1) if m else out.strip()).lower()

        def clear():
            """Delete the notebook: this ends any queued/running sessions it has (the CLI has no cancel).
            Kaggle allows only 2 batch GPU sessions, so stale queued ones block new runs."""
            print("   clearing old Kaggle sessions (deleting the notebook)…")
            _kaggle("kernels", "delete", kid, "-y", check=False)
            time.sleep(15)

        def push(retry_on_limit=True):
            out = _kaggle("kernels", "push", "-p", work, "--accelerator", k.get("accelerator", "NvidiaTeslaT4"),
                          "-t", str(timeout_s)).strip()
            print("   " + " | ".join(out.splitlines()[-3:]))
            if "error" in out.lower():
                if retry_on_limit and "session count" in out.lower():
                    clear()
                    return push(retry_on_limit=False)
                raise RuntimeError(f"Kaggle rejected the notebook: {out[-400:]}. Check that your Kaggle phone "
                                   "number is verified (kaggle.com → Settings → Account) and see 'Check Kaggle'.")

        # Runs never overlap (workflow concurrency group), so an active session now is a stale leftover.
        if any(x in status_now() for x in ("queued", "running")):
            clear()
        push()

        deadline = time.time() + timeout_s
        requeue_after = float(k.get("requeue_after_minutes", 20)) * 60
        queued_since, repushed = time.time(), 0
        status = ""
        time.sleep(30)
        while time.time() < deadline:
            out = _kaggle("kernels", "status", kid, check=False)
            m = re.search(r'status "([^"]+)"', out)
            status = (m.group(1) if m else out.strip()).lower()
            if any(s in status for s in ("complete", "error", "cancel")):
                break
            if any(s in status for s in ("denied", "not found", "401", "403")):
                raise RuntimeError(f"Kaggle refused the status check: {out.strip()[-300:]}")
            if "queued" in status:
                if time.time() - queued_since > requeue_after and repushed < 2:
                    repushed += 1
                    print(f"   still queued after {requeue_after / 60:.0f} min; clearing and pushing again "
                          f"({repushed}/2)…")
                    clear()
                    push()
                    queued_since = time.time()
            else:
                queued_since = time.time()  # running: reset the queue timer
            time.sleep(30)
        print(f"   Kaggle status: {status or 'timeout'}")

        dl = os.path.join(work, "out")
        os.makedirs(dl, exist_ok=True)
        _kaggle("kernels", "output", kid, "-p", dl, check=False)
        song = os.path.join(dl, "song.wav")
        if "complete" not in status or not os.path.exists(song):
            logs = sorted(glob.glob(os.path.join(dl, "*.log")))
            tail = ""
            raw = open(logs[0], errors="replace").read() if logs else _kaggle("kernels", "logs", kid, check=False)
            if raw:
                try:  # Kaggle logs are a JSON list of {stream_name, data}
                    raw = "".join(e.get("data", "") for e in json.loads(raw))
                except Exception:  # noqa: BLE001
                    pass
                tail = raw[-1500:]
            raise RuntimeError(f"Kaggle run did not produce a song (status={status or 'timeout'}).\n{tail}")
        res = {}
        if os.path.exists(os.path.join(dl, "result.json")):
            res = json.load(open(os.path.join(dl, "result.json")))
        out = os.path.join(out_dir, "song.wav")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", song, "-ar", "44100", "-ac", "2", out],
                       check=True)
        print(f"   song ready ({res.get('gpu', 'GPU')}, {res.get('seconds', '?')}s on Kaggle)")
        return {"audio": out, "timings": None, "engine": "ACE-Step (Kaggle GPU)"}
    finally:
        shutil.rmtree(work, ignore_errors=True)
