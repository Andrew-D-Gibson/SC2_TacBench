"""
orchestrator.py — Self-improving SC2 prompt optimization loop.

Repeatedly runs SC2 maps with the Qwen/Ollama player, analyzes failure logs
with a local meta-reasoner, proposes edits to prompt.txt / obs_raw_text.py,
applies them via a second LLM call, and keeps/reverts based on total_steps improvement.

Usage:
    python orchestrator.py

Stop with Ctrl-C at any time; the git state will be clean (either committed or stashed).
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import meta_reasoner
import file_editor
from config import (
    EDITABLE_FILES,
    LOGS_DIR,
    MAP_LIST,
    MAX_ITERATIONS,
    MAX_STAGNANT_ITERATIONS,
    ORCHESTRATOR_LOG_PATH,
    PLAYER_ENTRY,
    PROJECT_ROOT,
)


# ── Git helpers ────────────────────────────────────────────────────────────────

def _git(*args) -> str:
    result = subprocess.run(
        ["git"] + list(args),
        capture_output=True, text=True, check=True,
        cwd=PROJECT_ROOT,
    )
    return result.stdout.strip()


def git_baseline():
    """Commit any dirty working tree so we have a clean base before the loop."""
    status = _git("status", "--porcelain")
    if status:
        print("[git] Committing dirty working tree as baseline...")
        _git("add", "-A")
        _git("commit", "-m", "orchestrator: baseline before run")
        print("[git] Baseline committed.")
    else:
        print("[git] Working tree clean — no baseline commit needed.")


def git_stash():
    _git("stash", "push", "-m", "orchestrator: pre-iteration stash")


def git_stash_pop():
    _git("stash", "pop")


def git_stash_drop():
    """Drop the most recent stash (after a successful commit makes it obsolete)."""
    try:
        _git("stash", "drop")
    except subprocess.CalledProcessError:
        pass  # No stash to drop — fine.


def git_commit(message: str):
    _git("add", "-A")
    _git("commit", "-m", message)


# ── Logging ────────────────────────────────────────────────────────────────────

def _log_event(event: dict):
    """Append one JSON line to orchestrator_log.jsonl and print a summary line."""
    event["timestamp"] = datetime.now().isoformat()
    with open(ORCHESTRATOR_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def _print_status(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ── Map runner ─────────────────────────────────────────────────────────────────

def _find_new_log(run_start_time: float) -> Path | None:
    """
    Return the .jsonl file in LOGS_DIR whose mtime is >= run_start_time.
    Using mtime avoids Windows Path set-comparison issues.
    If multiple files qualify (shouldn't happen), returns the newest.
    """
    logs_dir = Path(LOGS_DIR)
    logs_dir.mkdir(parents=True, exist_ok=True)
    candidates = [p for p in logs_dir.glob("*.jsonl") if p.stat().st_mtime >= run_start_time]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _parse_log_summary(log_path: Path) -> dict:
    """Read the first JSON object from a game log and return its summary dict.
    Handles both compact JSONL (one object per line) and pretty-printed JSON."""
    with open(log_path, encoding="utf-8") as f:
        content = f.read()
    decoder = json.JSONDecoder()
    summary, _ = decoder.raw_decode(content.strip())
    assert summary.get("type") == "summary", f"Expected summary entry, got: {summary.get('type')}"
    return summary


def run_one_map(map_id: str) -> dict:
    """
    Run a single map via `python core/main.py` with TACBENCH_MAP injected.
    Returns a result dict with: map_id, won, total_steps, outcome, log_path, error.
    """
    env = os.environ.copy()
    env["TACBENCH_MAP"] = map_id

    _print_status(f"Running map '{map_id}'...")
    start = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, PLAYER_ENTRY],
            env=env,
            cwd=PROJECT_ROOT,
        )
        elapsed = time.time() - start
        _print_status(f"Map '{map_id}' finished in {elapsed:.0f}s (exit code {proc.returncode}).")
    except Exception as exc:
        error_msg = f"subprocess.run raised {type(exc).__name__}: {exc}"
        _print_status(f"ERROR running '{map_id}': {error_msg}")
        return {
            "map_id": map_id, "won": False, "total_steps": 0,
            "outcome": "ERROR", "log_path": None, "error": error_msg,
        }

    log_path = _find_new_log(start)
    if log_path is None:
        return {
            "map_id": map_id, "won": False, "total_steps": 0,
            "outcome": "ERROR", "log_path": None,
            "error": "No new log file found after run — game may have crashed before writing log.",
        }

    try:
        summary = _parse_log_summary(log_path)
    except Exception as exc:
        return {
            "map_id": map_id, "won": False, "total_steps": 0,
            "outcome": "ERROR", "log_path": str(log_path), "error": f"Log parse error: {exc}",
        }

    outcome     = summary.get("outcome", "LOSS")
    total_steps = summary.get("total_steps", 0)
    return {
        "map_id":      map_id,
        "won":         outcome == "WIN",
        "total_steps": total_steps,
        "outcome":     outcome,
        "log_path":    str(log_path),
        "error":       None,
    }


def run_all_maps() -> dict:
    """Run every map in MAP_LIST sequentially. Returns map_id → result dict."""
    results = {}
    for map_id in MAP_LIST:
        results[map_id] = run_one_map(map_id)
    return results


# ── Improvement logic ──────────────────────────────────────────────────────────

def all_won(results: dict) -> bool:
    return all(r["won"] for r in results.values())


def check_improvement(old_results: dict, new_results: dict) -> bool:
    """Return True if performance improved with no WIN→LOSS regressions on any map."""
    improved_any = False
    for map_id, new_r in new_results.items():
        old_r = old_results.get(map_id, {"won": False, "total_steps": 0})
        if old_r["won"] and not new_r["won"]:
            return False  # WIN→LOSS regression: always revert
        if new_r["won"] and not old_r["won"]:
            improved_any = True
        elif not new_r["won"] and not old_r["won"] and new_r["total_steps"] > old_r["total_steps"]:
            improved_any = True
    return improved_any


def _improvement_per_map(old_results: dict, new_results: dict) -> dict:
    delta = {}
    for map_id, new_r in new_results.items():
        old_steps = old_results.get(map_id, {}).get("total_steps", 0)
        delta[map_id] = new_r["total_steps"] - old_steps
    return delta


def _update_run_history(run_history: dict, results: dict):
    for map_id, r in results.items():
        prev = run_history.get(map_id, {"best_steps": 0, "best_result": "LOSS"})
        if r["won"] or r["total_steps"] > prev["best_steps"]:
            run_history[map_id] = {
                "best_steps":  r["total_steps"],
                "best_result": r["outcome"],
            }


def _results_summary(results: dict) -> str:
    parts = []
    for map_id, r in results.items():
        parts.append(f"{map_id}: {r['outcome']} ({r['total_steps']} steps)")
    return " | ".join(parts)


# ── Main loop ──────────────────────────────────────────────────────────────────

def main():
    _print_status("=== SC2 Prompt Optimization Orchestrator starting ===")
    _print_status(f"Maps: {MAP_LIST}")
    _print_status(f"Max iterations: {MAX_ITERATIONS}  |  Max stagnant: {MAX_STAGNANT_ITERATIONS}")

    # Ensure clean git state before we start touching files
    try:
        git_baseline()
    except subprocess.CalledProcessError as exc:
        _print_status(f"[git] WARNING: git_baseline failed: {exc}. Continuing anyway.")

    run_history: dict = {m: {"best_steps": 0, "best_result": "LOSS"} for m in MAP_LIST}
    stagnant_count = 0
    stash_pending  = False

    for iteration in range(1, MAX_ITERATIONS + 1):
        _print_status(f"\n{'='*60}")
        _print_status(f"ITERATION {iteration}/{MAX_ITERATIONS}  (stagnant: {stagnant_count}/{MAX_STAGNANT_ITERATIONS})")
        _print_status(f"{'='*60}")

        # ── Stop: stagnant ─────────────────────────────────────────────────────
        if stagnant_count >= MAX_STAGNANT_ITERATIONS:
            msg = f"No improvement for {MAX_STAGNANT_ITERATIONS} consecutive iterations — stopping."
            _print_status(msg)
            _log_event({"event": "stop", "iteration": iteration, "reason": "stagnant"})
            return

        # ── Run all maps ───────────────────────────────────────────────────────
        results = run_all_maps()
        _print_status(f"Results: {_results_summary(results)}")
        _log_event({"event": "run_complete", "iteration": iteration, "results": {
            m: {"won": r["won"], "total_steps": r["total_steps"], "outcome": r["outcome"],
                "error": r.get("error")}
            for m, r in results.items()
        }})

        # ── Stop: all won ──────────────────────────────────────────────────────
        if all_won(results):
            _print_status("All maps WON — optimization complete!")
            _log_event({"event": "stop", "iteration": iteration, "reason": "all_maps_won"})
            try:
                git_commit(f"orchestrator iter {iteration}: all maps won")
            except subprocess.CalledProcessError:
                pass
            return

        # ── Build meta-reasoner context ────────────────────────────────────────
        failed_maps = [m for m, r in results.items() if not r["won"]]
        _print_status(f"Failed maps: {failed_maps}. Calling meta-reasoner...")

        meta_decision = meta_reasoner.analyze(failed_maps, results, run_history)
        _print_status(f"Meta decision: action={meta_decision['action']} | {meta_decision['reason'][:100]}")
        _log_event({"event": "meta_decision", "iteration": iteration,
                    "action": meta_decision["action"], "reason": meta_decision["reason"],
                    "changes": meta_decision.get("changes", [])})

        # ── Stop: missing info ─────────────────────────────────────────────────
        if meta_decision["action"] == "stop_missing_info":
            _print_status(f"Meta-reasoner says stop (missing info): {meta_decision['reason']}")
            _log_event({"event": "stop", "iteration": iteration, "reason": "missing_info"})
            return

        # ── Noop (parse failure) ───────────────────────────────────────────────
        if meta_decision["action"] == "noop":
            _print_status("Meta-reasoner returned noop — skipping iteration.")
            stagnant_count += 1
            continue

        # ── Stash current state, apply edits ──────────────────────────────────
        try:
            git_stash()
            stash_pending = True
        except subprocess.CalledProcessError as exc:
            _print_status(f"[git] stash failed: {exc} — skipping iteration.")
            stagnant_count += 1
            continue

        edit_ok = file_editor.apply_changes(meta_decision.get("changes", []))

        if not edit_ok:
            _print_status("File edits failed — reverting stash.")
            git_stash_pop()
            stash_pending = False
            _log_event({"event": "iteration_reverted", "iteration": iteration, "reason": "edit_failed"})
            stagnant_count += 1
            continue

        # ── Re-run maps with edited files ──────────────────────────────────────
        _print_status("Edits applied. Re-running maps...")
        new_results = run_all_maps()
        _print_status(f"New results: {_results_summary(new_results)}")
        _log_event({"event": "run_complete", "iteration": iteration, "phase": "post_edit",
                    "results": {m: {"won": r["won"], "total_steps": r["total_steps"]}
                                for m, r in new_results.items()}})

        # ── Stop: all won after edit ───────────────────────────────────────────
        if all_won(new_results):
            commit_msg = (
                f"orchestrator iter {iteration}: {meta_decision['action']} — "
                f"{meta_decision['reason'][:120]} — all maps won"
            )
            git_commit(commit_msg)
            git_stash_drop()
            stash_pending = False
            _print_status("All maps WON after edit — optimization complete!")
            _log_event({"event": "stop", "iteration": iteration, "reason": "all_maps_won"})
            return

        # ── Keep or revert ─────────────────────────────────────────────────────
        improved = check_improvement(results, new_results)
        delta    = _improvement_per_map(results, new_results)

        if improved:
            commit_msg = (
                f"orchestrator iter {iteration}: {meta_decision['action']} — "
                f"{meta_decision['reason'][:120]}"
            )
            git_commit(commit_msg)
            git_stash_drop()
            stash_pending = False
            _update_run_history(run_history, new_results)
            stagnant_count = 0
            _print_status(f"IMPROVEMENT — changes committed. Delta: {delta}")
            _log_event({"event": "iteration_kept", "iteration": iteration,
                        "improvement_per_map": delta})
        else:
            git_stash_pop()
            stash_pending = False
            stagnant_count += 1
            _print_status(f"No improvement (delta: {delta}) — changes reverted.")
            _log_event({"event": "iteration_reverted", "iteration": iteration,
                        "reason": "no_improvement", "delta": delta})

    # ── Stop: max iterations ───────────────────────────────────────────────────
    _print_status(f"Reached MAX_ITERATIONS ({MAX_ITERATIONS}) — stopping.")
    _print_status("Final run history:")
    for map_id, h in run_history.items():
        _print_status(f"  {map_id}: best={h['best_steps']} steps, result={h['best_result']}")
    _log_event({"event": "stop", "iteration": MAX_ITERATIONS, "reason": "max_iterations",
                "run_history": run_history})


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[orchestrator] Interrupted by user. Git state may have a pending stash — run `git stash list` to check.")
        sys.exit(0)
