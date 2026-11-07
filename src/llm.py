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
        for attempt in range(3):
            try:
                r = requests.post(
                    url,
                    headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                    json={
                        "model": p["model"],
                        "temperature": temperature,
                        "messages": [{"role": "system", "content": system},
                                     {"role": "user", "content": user}],
                        "response_format": {"type": "json_object"},
                    },
                    timeout=120,
                )
                if r.status_code == 429 or r.status_code >= 500:
                    raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"]
                return _parse_json(text)
            except Exception as e:  # noqa: BLE001 — any failure → retry / next provider
                errors.append(f"{p['name']}/{p['model']} try {attempt + 1}: {e}")
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
