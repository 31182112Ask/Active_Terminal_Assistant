from __future__ import annotations

import re
from difflib import SequenceMatcher


def compact_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def similarity_ratio(left: str, right: str) -> float:
    return SequenceMatcher(None, compact_whitespace(left).lower(), compact_whitespace(right).lower()).ratio()

