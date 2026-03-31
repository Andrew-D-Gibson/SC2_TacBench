"""
meta_reasoner.py — Analyzes SC2 game failure logs and decides what to change.

Two-phase approach for better small-model performance:
  Phase 1 (Analysis): Free-form chain-of-thought — summarize each match, diagnose
                      failures, validate conclusions. No JSON pressure.
  Phase 2 (Decision): Given the analysis, output a single structured JSON decision.

Returns a decision dict:
  {"action": "edit_prompt"|"edit_code"|"stop_missing_info"|"noop",
   "reason": "...",
   "changes": [{"file": "...", "instructions": "..."}]}
"""

import json
import os
import re

import requests

import orc_console

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_META_MODEL,
    EDITABLE_FILES,
    META_REASONER_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    PROJECT_ROOT,
)


# ── Phase 1: Analysis system prompt ───────────────────────────────────────────

_ANALYSIS_SYSTEM = """\
You are an expert StarCraft II analyst reviewing the performance of an LLM-controlled bot.

Your task is to deeply analyze the game logs provided and produce a structured written analysis.
Work through each failed map one at a time. For each map, cover these sections:

MATCH SUMMARY
  Briefly describe the arc of the game: what forces were present, how the bot moved,
  what major engagements happened, and how it ended.

FAILURE ANALYSIS
  Identify the specific decisions that led to the loss. Be concrete — reference step
  numbers, unit types, and directive choices from the log. Ask: did the bot move when
  it should have held? Did it ignore a critical enemy unit? Did it issue the wrong
  directive for the matchup?

INSTRUCTION FOLLOWING
  Examine whether the bot correctly interpreted and followed the format and intent of
  its instructions. Check for each of these failure modes:
  - FORMAT FAILURES: Look for log entries where "fallback_used" is true or "llm_error"
    is set. These mean the model returned something unparseable or used an invalid
    directive name. Count how many steps this happened and what fraction of total calls
    it represents.
  - REASONING/ACTION MISMATCH: Look at the "reasoning" field alongside the chosen
    "directive". Does the stated reasoning support the action taken? Flag cases where
    the bot said it was doing one thing but issued a different directive.
  - IGNORED CONTEXT: Compare the battlefield observation ("battlefield" field) with the
    directive chosen. Did the prompt contain urgent information (e.g. a structure under
    attack, a numerical advantage) that the bot failed to act on? Cite specific steps.
  - OUTPUT FORMAT NOTE: If format failures are frequent (>20% of calls), this is likely
    a prompt engineering problem and should be the top recommendation.

ROOT CAUSE
  State the single most important reason the bot lost. This should be something that,
  if fixed, would most plausibly change the outcome.

SELF-CHECK
  Review your own Root Cause above. Play devil's advocate:
  - Is there another explanation you might be missing?
  - Does the Root Cause actually appear multiple times in the log, or was it a one-off?
  - Would fixing it actually require a prompt change, or is the information already there
    and the bot just misread it?
  Revise your Root Cause if warranted, and briefly state your confidence level.

After all maps are analyzed, write a short OVERALL RECOMMENDATION section summarizing
the highest-leverage single change across all failures.

Be specific and analytical. Reference the actual log data.\
"""


# ── Phase 2: Decision system prompt ───────────────────────────────────────────

_DECISION_SYSTEM = f"""\
You are an expert prompt engineer and Python developer optimizing an LLM agent that plays custom StarCraft II maps.

You will be given:
  1. The current player prompt and formatting code.
  2. A detailed analysis of recent match failures (already written).

Your job: based on the analysis, decide on the single best change to make.

You may only edit files from this whitelist:
{json.dumps(EDITABLE_FILES, indent=2)}

DECISION LOGIC:
- Strongly prefer editing the prompt file. Only propose code changes to formatting
  functions if the player clearly lacks access to critical game state data that the
  prompt cannot compensate for.
- If the player lacks access to data that is never present in the logs and no prompt
  change could fix this, return action "stop_missing_info".
- Otherwise, return action "edit_prompt" or "edit_code".

OUTPUT FORMAT:
Respond with a single JSON object only. No preamble, no explanation outside the JSON,
no markdown code fences.

{{
  "action": "edit_prompt" | "edit_code" | "stop_missing_info",
  "reason": "1-3 sentence explanation of what went wrong and why this change should help",
  "changes": [
    {{
      "file": "relative/path/to/file.txt",
      "instructions": "Detailed plain-English description of what to change. Be specific about what to add, remove, or reword. Do not write the new file content here."
    }}
  ]
}}

For "stop_missing_info", omit "changes" and explain in "reason" exactly what data the player cannot see.
For "edit_prompt" or "edit_code", "changes" must contain one entry per file being modified.
Only include files from the whitelist.\
"""


# ── LLM callers ───────────────────────────────────────────────────────────────

def _call_claude(system: str, user_message: str, label: str, max_tokens: int = 4096) -> str:
    """Call the Anthropic Messages API with streaming. Prints tokens live."""
    import anthropic

    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "META_REASONER_BACKEND=claude but ANTHROPIC_API_KEY is not set."
        )

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    orc_console.llm_stream_header(label)
    parts = []
    with client.messages.stream(
        model=CLAUDE_META_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    ) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            parts.append(text)
    print("\n", flush=True)
    return "".join(parts)


def _call_ollama(system: str, user_message: str, label: str, think: bool = True) -> str:
    """POST to Ollama /api/chat with streaming. Prints tokens live with a label prefix."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_message},
        ],
        "stream": True,
        "think": think,
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
        stream=True,
    )
    response.raise_for_status()

    orc_console.llm_stream_header(label)
    parts = []
    for line in response.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        delta = chunk.get("message", {}).get("content", "")
        if delta:
            print(delta, end="", flush=True)
            parts.append(delta)
        if chunk.get("done"):
            break
    print("\n", flush=True)
    return "".join(parts)


def _call_llm(system: str, user_message: str, label: str, think: bool = True) -> str:
    """Route to Claude or Ollama based on META_REASONER_BACKEND."""
    if META_REASONER_BACKEND == "claude":
        # Decision phase uses smaller budget; analysis can run long.
        max_tokens = 1024 if "decision" in label else 4096
        return _call_claude(system, user_message, label, max_tokens=max_tokens)
    return _call_ollama(system, user_message, label, think=think)


# ── JSON parsing ───────────────────────────────────────────────────────────────

def _parse_decision(raw: str) -> dict:
    """Strip markdown fences, parse JSON, validate required keys."""
    text = raw.strip()

    # Strip <think>...</think> blocks emitted by reasoning models (e.g. qwen3)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # Strip ```json ... ``` or ``` ... ``` fences
    if text.startswith("```"):
        text = text.lstrip("`")
        if text.startswith("json"):
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()

    # Some models emit a brief preamble before the JSON object; find the first {
    brace = text.find("{")
    if brace > 0:
        text = text[brace:]

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


# ── Context builders ───────────────────────────────────────────────────────────

def _read_file(rel_path: str) -> str:
    abs_path = os.path.join(PROJECT_ROOT, rel_path)
    with open(abs_path, encoding="utf-8") as f:
        return f.read()


def _build_game_context(
    failed_maps: list[str],
    all_results: dict,
    run_history: dict,
) -> str:
    """
    Assembles raw game context: logs for each failed map + run history table.
    Does NOT include editable file contents (those go in the decision phase).
    """
    parts = []

    for map_id in failed_maps:
        result = all_results.get(map_id, {})
        log_path = result.get("log_path")
        best = run_history.get(map_id, {})
        parts.append(f"=== GAME LOG: {map_id} ===")
        parts.append(f"This run:  total_steps={result.get('total_steps', 0)}, outcome={result.get('outcome', 'LOSS')}")
        parts.append(f"Best ever: total_steps={best.get('best_steps', 0)}, result={best.get('best_result', 'LOSS')}")
        if log_path and os.path.exists(log_path):
            with open(log_path, encoding="utf-8") as f:
                parts.append(f.read())
        else:
            parts.append("(no log file — map likely crashed before writing)")

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


def _build_decision_context(analysis: str) -> str:
    """
    Assembles context for the decision phase: editable file contents + the analysis.
    """
    parts = []

    parts.append("=== CURRENT PLAYER PROMPT (prompt.txt) ===")
    parts.append(_read_file("prompt.txt"))

    for rel_path in EDITABLE_FILES:
        if rel_path == "prompt.txt":
            continue
        parts.append(f"\n=== CURRENT FORMATTING FILE ({rel_path}) ===")
        parts.append(_read_file(rel_path))

    parts.append("\n=== MATCH ANALYSIS (from Phase 1) ===")
    parts.append(analysis)

    return "\n".join(parts)


# ── Public interface ───────────────────────────────────────────────────────────

def analyze(
    failed_maps: list[str],
    all_results: dict,
    run_history: dict,
) -> dict:
    """
    Two-phase meta-reasoning:
      Phase 1 — Free-form analysis: summarize matches, diagnose failures, self-validate.
      Phase 2 — Structured decision: given the analysis, output JSON action.

    Retries the decision phase once on JSON parse failure.
    Falls back to {"action": "noop"} if both attempts fail.
    """
    game_context = _build_game_context(failed_maps, all_results, run_history)

    # ── Phase 1: Analysis ──────────────────────────────────────────────────────
    orc_console.meta_phase(1, "Analyzing match failures", META_REASONER_BACKEND)
    try:
        analysis = _call_llm(_ANALYSIS_SYSTEM, game_context, "meta_reasoner / analysis")
    except Exception as exc:
        orc_console.meta_error(f"Analysis call failed: {exc}")
        return {"action": "noop", "reason": f"analysis phase failed: {exc}", "changes": []}

    # ── Phase 2: Decision ──────────────────────────────────────────────────────
    orc_console.meta_phase(2, "Deciding what to change", META_REASONER_BACKEND)
    decision_context = _build_decision_context(analysis)

    for attempt in range(2):
        try:
            if attempt == 1:
                decision_context += "\n\nIMPORTANT: Output ONLY the JSON object. No preamble, no markdown, no extra text."
            raw = _call_llm(_DECISION_SYSTEM, decision_context, "meta_reasoner / decision", think=False)
            return _parse_decision(raw)
        except json.JSONDecodeError as exc:
            orc_console.meta_error(f"JSON parse error (attempt {attempt+1}): {exc}")
        except ValueError as exc:
            orc_console.meta_error(f"Validation error (attempt {attempt+1}): {exc}")
        except Exception as exc:
            orc_console.meta_error(f"LLM call failed (attempt {attempt+1}): {exc}")
            break

    orc_console.meta_error("Decision phase failed after 2 attempts — skipping iteration.")
    return {"action": "noop", "reason": "decision phase failed after 2 attempts", "changes": []}
