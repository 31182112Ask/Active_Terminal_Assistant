from __future__ import annotations

from functools import lru_cache
from importlib import resources


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    return resources.files("app.prompts").joinpath(name).read_text(encoding="utf-8")

