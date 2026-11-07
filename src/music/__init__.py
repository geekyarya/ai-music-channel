"""Music engine dispatcher: tries each configured backend until one succeeds."""
from __future__ import annotations

import traceback


def make_song(spec: dict, cfg: dict, out_dir: str, only: str | None = None) -> dict:
    backends = [only] if only else cfg["music"]["backends"]
    errors = []
    for name in backends:
        try:
            print(f"♪ music backend: {name}")
            if name == "ace_step":
                from music import ace_step
                res = ace_step.generate(spec, cfg, out_dir)
                if cfg["video"].get("lyrics_captions", True):
                    from music.align import align_lyrics
                    res["timings"] = align_lyrics(res["audio"], spec)
                return res
            if name == "local":
                from music import local
                return local.generate(spec, cfg, out_dir)
            raise ValueError(f"unknown backend {name}")
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            errors.append(f"{name}: {e}")
            print(f"  ✗ {name} failed, trying next backend")
    raise RuntimeError("All music backends failed: " + " | ".join(errors))
