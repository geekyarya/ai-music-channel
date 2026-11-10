"""Tiny OpenAI-compatible chat client that walks a list of free providers."""
from __future__ import annotations

import json
import os
import re
import time

import requests

ENDPOINTS = {
    # Hugging Face Inference Providers router: OpenAI-compatible, reuses the HF_TOKEN
    # the singing step already needs (free monthly credits cover ~1 song/day easily).
    "hf": ("https://router.huggingface.co/v1/chat/completions", "HF_TOKEN"),
    # GitHub Models: free, authenticated by the GITHUB_TOKEN every Actions run gets
    # (the workflow needs `permissions: models: read`).
    "github": ("https://models.github.ai/inference/chat/completions", "GITHUB_TOKEN"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
}


class LLMError(RuntimeError):
    pass


# When a configured model is retired (404), pick the best one the provider currently offers.
PREFERRED = {
    "groq": ["openai/gpt-oss-120b", "moonshotai/kimi-k2-instruct", "llama-3.3-70b-versatile",
             "meta-llama/llama-4-maverick-17b-128e-instruct", "qwen/qwen3-32b", "openai/gpt-oss-20b",
             "llama-3.1-8b-instant"],
}


def _pick_model(provider: str, key: str) -> str | None:
    url = ENDPOINTS[provider][0].replace("/chat/completions", "/models")
    try:
        ids = [m["id"] for m in requests.get(url, headers={"Authorization": f"Bearer {key}"}, timeout=30)
               .json().get("data", [])]
    except Exception:  # noqa: BLE001
        return None
    for m in PREFERRED.get(provider, []):
        if m in ids:
            return m
    chat = [i for i in ids if not re.search(r"whisper|guard|tts|embed|vision|audio|playai", i, re.I)]
    return chat[0] if chat else None


def chat_json(providers: list[dict], system: str, user: str, temperature: float = 1.0) -> dict:
    """Ask each provider in turn for a JSON object; return the first valid one."""
    errors = []
    for p in providers:
        url, key_env = ENDPOINTS[p["name"]]
        key = os.environ.get(key_env)
        if not key:
            errors.append(f"{p['name']}: {key_env} not set")
            continue
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if p["name"] == "github":
            headers.update({"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"})
        for attempt in range(3):
            body = {
                "model": p["model"],
                "temperature": temperature,
                "max_tokens": 4096,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            }
            if attempt == 0:  # some endpoints reject/mishandle response_format — drop it on retries
                body["response_format"] = {"type": "json_object"}
            if p["name"] == "gemini":  # thinking models: keep reasoning short so the reply comes fast
                body["reasoning_effort"] = "low"
            r = None
            try:
                r = requests.post(url, headers=headers, json=body, timeout=240, allow_redirects=False)
                if r.status_code >= 400:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:900]!r}")
                text = r.json()["choices"][0]["message"]["content"] or ""
                if not text.strip():
                    raise LLMError(f"empty completion: {r.text[:300]!r}")
                return _parse_json(text)
            except Exception as e:  # noqa: BLE001 — any failure → retry / next provider
                detail = "" if r is None or "HTTP" in str(e) or "empty" in str(e) else \
                    f" [HTTP {r.status_code}, url={r.url}, redirects={[h.status_code for h in r.history]}, " \
                    f"content-type={r.headers.get('content-type')}, body={r.text[:300]!r}]"
                msg = f"{p['name']}/{p['model']} try {attempt + 1}: {type(e).__name__}: {e}{detail}"
                print("  ! " + msg)
                errors.append(msg)
                if r is not None and r.status_code == 404 and "model" in r.text.lower() \
                        and not p.get("_auto"):
                    alt = _pick_model(p["name"], key)
                    if alt and alt != p["model"]:
                        print(f"    model {p['model']} is gone; switching to {alt}")
                        p = {**p, "model": alt, "_auto": True}
                        continue
                if r is not None and r.status_code in (401, 403, 404):
                    break  # auth/model problem: retrying won't help
                if r is not None and r.status_code == 429:
                    if re.search(r"per ?day|PerDay|retry in \d+h|retry in [1-9]\d*m", r.text, re.I):
                        print("    daily quota used up for this model — moving on")
                        break  # retrying only burns more of tomorrow's quota
                    # per-minute limit: wait as long as the API asks (max 90 s)
                    m = re.search(r'retry(?:Delay|-after)["\':\s]*"?(\d+)', r.text + str(r.headers), re.I)
                    wait = min(90, int(m.group(1)) + 2) if m else 45
                    print(f"    rate-limited, waiting {wait}s")
                    time.sleep(wait)
                    continue
                if isinstance(e, requests.Timeout):
                    break  # a timed-out request still counts against quota; don't hammer it
                time.sleep(3 * (attempt + 1))
    raise LLMError("All LLM providers failed:\n  " + "\n  ".join(errors))


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))
