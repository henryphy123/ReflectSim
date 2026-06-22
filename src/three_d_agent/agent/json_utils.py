"""Shared helper for parsing LLM JSON output that may be wrapped in markdown."""
import json
import re
from typing import Any

_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*\n?(.*?)\n?\s*```\s*$", re.DOTALL)


def parse_llm_json(text: str) -> Any:
    """Parse `text` as JSON. Strips ```...``` markdown fences if present.

    Raises ValueError with the original text on failure.
    """
    candidate = text.strip()
    m = _FENCE_RE.match(candidate)
    if m:
        candidate = m.group(1).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return JSON: {text!r}") from e
