"""Tiny OpenAI-compatible chat client that walks a list of free providers."""
from __future__ import annotations

import json
import os
import re
import time

import requests

ENDPOINTS = {
    # GitHub Models: free, authenticated by the GITHUB_TOKEN every Actions run gets
    # (the workflow needs `permissions: models: read`).
    "github": ("https://models.github.ai/inference/chat/completions", "GITHUB_TOKEN"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai/chat/completions", "GEMINI_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
}


class LLMError(RuntimeError):
    pass


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
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
            }
            if attempt == 0:  # some endpoints reject/mishandle response_format — drop it on retries
                body["response_format"] = {"type": "json_object"}
            r = None
            try:
                r = requests.post(url, headers=headers, json=body, timeout=120)
                if r.status_code >= 400:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:300]!r}")
                text = r.json()["choices"][0]["message"]["content"] or ""
                if not text.strip():
                    raise LLMError(f"empty completion: {r.text[:300]!r}")
                return _parse_json(text)
            except Exception as e:  # noqa: BLE001 — any failure → retry / next provider
                detail = "" if r is None or "HTTP" in str(e) or "empty" in str(e) else \
                    f" [HTTP {r.status_code}, content-type={r.headers.get('content-type')}, body={r.text[:300]!r}]"
                msg = f"{p['name']}/{p['model']} try {attempt + 1}: {type(e).__name__}: {e}{detail}"
                print("  ! " + msg)
                errors.append(msg)
                if r is not None and r.status_code in (401, 403, 404):
                    break  # auth/model problem: retrying won't help
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
