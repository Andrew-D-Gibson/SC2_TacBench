"""
meta_reasoner.py — Analyzes SC2 game failure logs and decides what to change.

Calls a local Qwen model via Ollama.  Returns a structured decision dict:
  {"action": "edit_prompt"|"edit_code"|"stop_missing_info"|"noop",
   "reason": "...",
   "changes": [{"file": "...", "instructions": "..."}]}
"""

import json
import os

import requests

from config import (
    EDITABLE_FILES,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    PROJECT_ROOT,
)

# ── System prompt ──────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = f"""You are an expert prompt engineer and Python developer optimizing an LLM agent that plays custom StarCraft II maps.

Your job: analyze failure logs from the LLM player, then decide what change to make to improve performance.

You may only edit files from this whitelist:
{json.dumps(EDITABLE_FILES, indent=2)}

DECISION LOGIC:
- Strongly prefer editing the prompt file. Only propose code changes to formatting functions if you are confident the player lacks critical information that the prompt cannot compensate for.
- If the player clearly lacks access to game state data it needs (e.g., a key unit type or event is never present in the logs passed to it), and no prompt change could fix this, return action "stop_missing_info".
- Otherwise, return action "edit_prompt" or "edit_code".

OUTPUT FORMAT:
You must respond with a single JSON object only. No preamble, no explanation outside the JSON, no markdown code fences.

{{
  "action": "edit_prompt" | "edit_code" | "stop_missing_info",
  "reason": "1-3 sentence explanation of what went wrong and why this change should help",
  "changes": [
    {{
      "file": "relative/path/to/file.txt",
      "instructions": "Detailed plain-English description of what to change in this file. Be specific about what to add, remove, or reword. Do not write the new file content here."
    }}
  ]
}}

For "stop_missing_info", omit the "changes" key and explain in "reason" exactly what data the player cannot see.
For "edit_prompt" or "edit_code", "changes" must contain one entry per file being modified. Only include files from the whitelist.
"""


# ── Ollama interface ───────────────────────────────────────────────────────────

def _call_ollama(user_message: str) -> str:
    """POST to Ollama /api/chat and return the assistant reply text."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        "stream": False,
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def _parse_decision(raw: str) -> dict:
    """Strip markdown fences, parse JSON, validate required keys."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.lstrip("`")
        if text.startswith("json"):
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    parsed = json.loads(text)

    action = parsed.get("action")
    if action not in ("edit_prompt", "edit_code", "stop_missing_info"):
        raise ValueError(f"Unknown action: {action!r}")
    if "reason" not in parsed:
        raise ValueError("Missing 'reason' field")
    if action != "stop_missing_info":
        changes = parsed.get("changes")
        if not isinstance(changes, list) or not changes:
            raise ValueError("'changes' must be a non-empty list for edit actions")

    return parsed


def _read_file(rel_path: str) -> str:
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    with open(abs_path, encoding="utf-8") as f:
        return f.read()


def _build_user_message(
    failed_maps: list[str],
    all_results: dict,
    run_history: dict,
) -> str:
    """
    Assemble the context passed to the meta-reasoner:
      1. Current prompt file
      2. Current formatting file(s)
      3. Full JSONL log for each failed map
      4. Run history summary table
    """
    parts = []

    # 1. Player prompt
    parts.append("=== CURRENT PLAYER PROMPT (prompt.txt) ===")
    parts.append(_read_file("prompt.txt"))

    # 2. Formatting source files (all editable files except the prompt)
    for rel_path in EDITABLE_FILES:
        if rel_path == "prompt.txt":
            continue
        parts.append(f"\n=== CURRENT FORMATTING FILE ({rel_path}) ===")
        parts.append(_read_file(rel_path))

    # 3. Failure log contents
    for map_id in failed_maps:
        result = all_results.get(map_id, {})
        log_path = result.get("log_path")
        best = run_history.get(map_id, {})
        parts.append(f"\n=== FAILURE LOG: {map_id} ===")
        parts.append(f"Current run: total_steps={result.get('total_steps', 0)}, outcome={result.get('outcome', 'LOSS')}")
        parts.append(f"Best ever:   total_steps={best.get('best_steps', 0)}, result={best.get('best_result', 'LOSS')}")
        if log_path and os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                parts.append(f.read())
        else:
            parts.append("(no log file found — map likely crashed)")

    # 4. Run history table
    parts.append("\n=== RUN HISTORY SUMMARY ===")
    parts.append(f"{'Map':<20} {'Current Steps':>14} {'Best Steps':>11} {'Best Result':>12}")
    parts.append("-" * 60)
    for map_id in all_results:
        result = all_results[map_id]
        best = run_history.get(map_id, {})
        parts.append(
            f"{map_id:<20} {result.get('total_steps', 0):>14} "
            f"{best.get('best_steps', 0):>11} {best.get('best_result', 'LOSS'):>12}"
        )

    return "\n".join(parts)


# ── Public interface ───────────────────────────────────────────────────────────

def analyze(
    failed_maps: list[str],
    all_results: dict,
    run_history: dict,
) -> dict:
    """
    Call the meta-reasoner and return a decision dict.
    Retries once on JSON parse failure.
    Falls back to {"action": "noop", ...} if both attempts fail.
    """
    user_msg = _build_user_message(failed_maps, all_results, run_history)

    for attempt in range(2):
        try:
            if attempt == 1:
                user_msg = user_msg + "\n\nIMPORTANT: Return ONLY valid JSON. No preamble, no markdown, no extra text."
            raw = _call_ollama(user_msg)
            decision = _parse_decision(raw)
            return decision
        except json.JSONDecodeError as exc:
            print(f"  [meta_reasoner] JSON parse error (attempt {attempt+1}): {exc}")
        except ValueError as exc:
            print(f"  [meta_reasoner] Validation error (attempt {attempt+1}): {exc}")
        except Exception as exc:
            print(f"  [meta_reasoner] Ollama call failed (attempt {attempt+1}): {exc}")
            # Don't retry on connection errors — Ollama is down
            break

    print("  [meta_reasoner] Both attempts failed — skipping this iteration.")
    return {"action": "noop", "reason": "meta-reasoner parse failed after 2 attempts", "changes": []}
