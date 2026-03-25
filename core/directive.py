from dataclasses import dataclass
from typing import Any, Iterable, Optional
import json


@dataclass
class Directive:
    """
    Normalized directive object used by the bot execution pipeline.
    """
    name: str
    reasoning: str = ""
    target_x: Optional[float] = None
    target_y: Optional[float] = None
    raw: Any = None
    error: Optional[str] = None
    fallback_used: bool = False


# --- Internal helpers ---

def _make_fallback(raw: Any, fallback: str, error: str) -> Directive:
    """Return a fallback Directive with the given error message."""
    return Directive(name=fallback, raw=raw, error=error, fallback_used=True)


def _normalize_from_string(raw: str, allowed_set: set, fallback: str) -> Directive:
    """
    Parse a directive from a plain string or a JSON string.
    Accepts exact directive name matches and JSON-encoded dicts.
    No partial matching — if it doesn't match exactly, returns the fallback.
    """
    text = raw.strip()

    # Strip markdown code fences (e.g. ```json ... ``` or ``` ... ```).
    if text.startswith("```"):
        text = text.lstrip("`")
        if text.startswith("json"):
            text = text[4:]
        # Strip the closing fence, if present.
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # If it looks like JSON, try to parse it as a dict/list and recurse.
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return _make_fallback(raw, fallback, "invalid JSON string")
        directive = _normalize_from_dict(parsed, allowed_set, fallback) if isinstance(parsed, dict) \
               else _make_fallback(raw, fallback, "JSON array is not a valid directive")
        directive.raw = raw  # preserve original string as the raw value
        return directive

    # Exact match only.
    if text in allowed_set:
        return Directive(name=text, raw=raw)

    return _make_fallback(raw, fallback, f"unknown directive '{text}'")


def _normalize_from_dict(raw: dict, allowed_set: set, fallback: str) -> Directive:
    """
    Parse a directive from a dict.
    Accepts 'directive' or 'name' as the command key.
    Accepts optional 'target_x' and 'target_y' as float coordinates.
    """
    name = raw.get("directive") or raw.get("name")
    reasoning = raw.get("reasoning", "")

    # Parse optional coordinate target — silently ignore if not valid floats.
    try:
        target_x = float(raw["target_x"]) if raw.get("target_x") is not None else None
        target_y = float(raw["target_y"]) if raw.get("target_y") is not None else None
    except (ValueError, TypeError):
        target_x = None
        target_y = None

    if not isinstance(name, str):
        return _make_fallback(raw, fallback, "missing directive name")

    if name not in allowed_set:
        return _make_fallback(raw, fallback, f"unknown directive '{name}'")

    return Directive(
        name=name,
        reasoning=reasoning or "",
        target_x=target_x,
        target_y=target_y,
        raw=raw,
    )


# --- Public interface ---

def normalize_directive(raw: Any, allowed: Iterable[str], fallback: str) -> Directive:
    """
    Normalize raw LLM output (dict / JSON string / plain string / Directive)
    into a valid Directive. Unknown or malformed input falls back to the
    provided fallback directive name.
    """
    allowed_set = set(allowed)

    if raw is None:
        return _make_fallback(raw, fallback, "empty directive")

    if isinstance(raw, Directive):
        if raw.name in allowed_set:
            return raw
        return _make_fallback(raw, fallback, f"unknown directive '{raw.name}'")

    if isinstance(raw, str):
        return _normalize_from_string(raw, allowed_set, fallback)

    if isinstance(raw, dict):
        return _normalize_from_dict(raw, allowed_set, fallback)

    return _make_fallback(raw, fallback, f"unsupported directive type: {type(raw).__name__}")
