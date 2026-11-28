"""Run a script on a free Kaggle GPU and download its outputs (generic version of music/kaggle_gpu.py).

push a private script kernel → poll its status → `kaggle kernels output`. Needs KAGGLE_USERNAME + KAGGLE_KEY.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
import time


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


def run(script_path: str, params: dict, slug: str, out_dir: str, accelerator: str = "NvidiaTeslaT4",
        timeout_minutes: float = 150, requeue_after_minutes: float = 45, max_requeues: int = 1,
        fallback_accelerator: str | None = "NvidiaTeslaP100") -> str:
    """Returns the directory holding the kernel's output files. Raises if the kernel never completed."""
    if not have_credentials():
        raise RuntimeError("KAGGLE_USERNAME / KAGGLE_KEY not set")
    user = os.environ["KAGGLE_USERNAME"].strip()
    kid = f"{user}/{slug}"
    code = open(script_path).read().replace("__PARAMS_B64__", base64.b64encode(json.dumps(params).encode()).decode())
    work = tempfile.mkdtemp(prefix="kaggle_")
    with open(os.path.join(work, "run.py"), "w") as fh:
        fh.write(code)

    def write_meta(acc):
        with open(os.path.join(work, "kernel-metadata.json"), "w") as fh:
            json.dump({"id": kid, "title": slug, "code_file": "run.py", "language": "python", "kernel_type": "script",
                       "is_private": True, "enable_gpu": True, "machine_shape": acc, "enable_internet": True,
                       "dataset_sources": [], "competition_sources": [], "kernel_sources": []}, fh)

    write_meta(accelerator)
    quota = _kaggle("quota", check=False).strip()
    print("   Kaggle quota: " + " | ".join(l.strip() for l in quota.splitlines() if l.strip())[:300])
    timeout_s = int(timeout_minutes * 60)

    def status_now() -> str:
        out = _kaggle("kernels", "status", kid, check=False)
        m = re.search(r'status "([^"]+)"', out)
        return (m.group(1) if m else out.strip()).lower()

    def clear():
        print("   clearing old Kaggle sessions (deleting the notebook)…")
        _kaggle("kernels", "delete", kid, "-y", check=False)
        time.sleep(15)

    def push(retry_on_limit=True):
        out = _kaggle("kernels", "push", "-p", work, "--accelerator", accelerator, "-t", str(timeout_s)).strip()
        print("   " + " | ".join(out.splitlines()[-3:]))
        if "error" in out.lower():
            if retry_on_limit and "session count" in out.lower():
                clear()
                return push(retry_on_limit=False)
            raise RuntimeError(f"Kaggle rejected the notebook: {out[-400:]}")

    if any(x in status_now() for x in ("queued", "running")):
        clear()
    push()
    deadline = time.time() + timeout_s
    queued_since, repushed, status, last = time.time(), 0, "", ""
    time.sleep(30)
    while time.time() < deadline:
        status = status_now()
        if status != last:
            print(f"   [{time.strftime('%H:%M:%S')}] Kaggle: {status}")
            last = status
        if any(s in status for s in ("complete", "error", "cancel")):
            break
        if any(s in status for s in ("denied", "not found", "401", "403")):
            raise RuntimeError(f"Kaggle refused the status check: {status[-300:]}")
        if "queued" in status:
            if time.time() - queued_since > requeue_after_minutes * 60 and repushed < max_requeues:
                repushed += 1
                if fallback_accelerator and fallback_accelerator != accelerator:
                    accelerator = fallback_accelerator  # the T4 queue is jammed; try the other free GPU
                    write_meta(accelerator)
                print(f"   still queued after {requeue_after_minutes:.0f} min; pushing again on {accelerator}")
                clear()
                push()
                queued_since = time.time()
        else:
            queued_since = time.time()
        time.sleep(30)
    print(f"   Kaggle status: {status or 'timeout'}")
    os.makedirs(out_dir, exist_ok=True)
    _kaggle("kernels", "output", kid, "-p", out_dir, check=False)
    log = ""
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".log"):
            raw = open(os.path.join(out_dir, f), errors="replace").read()
            try:
                raw = "".join(e.get("data", "") for e in json.loads(raw))
            except Exception:  # noqa: BLE001
                pass
            log = raw
    if log:
        print("   ── Kaggle log (tail) ──\n" + "\n".join("   " + l for l in log[-2500:].splitlines()))
    if "complete" not in status:
        raise RuntimeError(f"Kaggle run did not complete (status={status or 'timeout'})")
    return out_dir
