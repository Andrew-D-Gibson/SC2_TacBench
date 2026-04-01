# Run with:
#   python test_directive.py

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from core.directives.directive import normalize_directive  # noqa: E402
from core.directives.execute_directive import get_directive_registry  # noqa: E402


def test_valid_dict():
    allowed = get_directive_registry().keys()
    raw = {"directive": "HOLD_POSITION", "reasoning": "test"}
    d = normalize_directive(raw, allowed=allowed, fallback="RETREAT")
    assert d.name == "HOLD_POSITION"
    assert not d.fallback_used


def test_unknown_directive():
    allowed = get_directive_registry().keys()
    raw = {"directive": "UNKNOWN", "reasoning": "test"}
    d = normalize_directive(raw, allowed=allowed, fallback="RETREAT")
    assert d.name == "RETREAT"
    assert d.fallback_used


def test_missing_keys():
    allowed = get_directive_registry().keys()
    raw = {"reasoning": "missing"}
    d = normalize_directive(raw, allowed=allowed, fallback="RETREAT")
    assert d.name == "RETREAT"
    assert d.fallback_used


def test_valid_json_string():
    allowed = get_directive_registry().keys()
    raw = '{"directive": "HOLD_POSITION", "reasoning": "test"}'
    d = normalize_directive(raw, allowed=allowed, fallback="RETREAT")
    assert d.name == "HOLD_POSITION"
    assert not d.fallback_used


def test_invalid_json_string():
    allowed = get_directive_registry().keys()
    raw = '{"directive": "HOLD_POSITION"'
    d = normalize_directive(raw, allowed=allowed, fallback="RETREAT")
    assert d.name == "RETREAT"
    assert d.fallback_used


if __name__ == "__main__":
    test_valid_dict()
    test_unknown_directive()
    test_missing_keys()
    test_invalid_json_string()
    print("All directive tests passed.")
