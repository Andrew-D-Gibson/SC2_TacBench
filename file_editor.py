"""
file_editor.py — Two-step file editing for the orchestrator.

Step 1 (meta_reasoner.py): produces plain-English instructions per file.
Step 2 (here): a second Ollama call turns instructions into a complete edited file,
               which is validated and written to disk.

The whitelist from config.EDITABLE_FILES is enforced as a hard check —
any proposed edit outside that list is silently skipped with a warning.
"""

import os
import py_compile
import tempfile

import requests

from config import (
    EDITABLE_FILES,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OLLAMA_TIMEOUT,
    PROJECT_ROOT,
)


# ── Editor Ollama call ─────────────────────────────────────────────────────────

def _call_editor(instructions: str, current_content: str) -> str:
    """
    Ask Ollama to apply instructions to the file and return the full updated content.
    No system prompt — the editor prompt is self-contained.
    """
    prompt = (
        "You are a precise file editor. Apply the following instructions to the file below "
        "and return the complete updated file content.\n\n"
        f"INSTRUCTIONS:\n{instructions}\n\n"
        f"CURRENT FILE CONTENT:\n{current_content}\n\n"
        "Return ONLY the complete updated file content. "
        "No explanation, no markdown fences, no preamble. "
        "The output will be written directly to disk."
    )
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


# ── Validation ─────────────────────────────────────────────────────────────────

def _validate(rel_path: str, content: str) -> tuple[bool, str]:
    """
    Returns (ok, error_message).
    - .py files: write to a temp file and run py_compile.
    - .txt / .md files: check non-empty (>50 chars).
    """
    if not content or not content.strip():
        return False, "Editor returned empty content"

    ext = os.path.splitext(rel_path)[1].lower()

    if ext == ".py":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            py_compile.compile(tmp_path, doraise=True)
            return True, ""
        except py_compile.PyCompileError as exc:
            return False, f"py_compile error: {exc}"
        finally:
            os.unlink(tmp_path)

    else:  # .txt, .md, etc.
        if len(content.strip()) < 50:
            return False, f"Content suspiciously short ({len(content.strip())} chars)"
        return True, ""


# ── Public interface ───────────────────────────────────────────────────────────

def apply_changes(changes: list[dict]) -> bool:
    """
    Apply a list of file changes produced by the meta-reasoner.

    Each change: {"file": "relative/path", "instructions": "plain English"}

    Returns True if every change was successfully written.
    Returns False if any change failed (whitelist violation, Ollama error, or validation failure).
    Files that fail are not written; files that pass are written immediately.
    """
    if not changes:
        print("  [file_editor] No changes to apply.")
        return False

    all_ok = True

    for change in changes:
        rel_path    = change.get("file", "").strip()
        instructions = change.get("instructions", "").strip()

        # ── Whitelist check ────────────────────────────────────────────────────
        if rel_path not in EDITABLE_FILES:
            print(f"  [file_editor] BLOCKED: '{rel_path}' is not in EDITABLE_FILES whitelist.")
            all_ok = False
            continue

        abs_path = os.path.join(PROJECT_ROOT, rel_path)

        # ── Read current content ───────────────────────────────────────────────
        try:
            with open(abs_path, encoding="utf-8") as f:
                current_content = f.read()
        except OSError as exc:
            print(f"  [file_editor] Cannot read '{rel_path}': {exc}")
            all_ok = False
            continue

        # ── Editor call ────────────────────────────────────────────────────────
        print(f"  [file_editor] Editing '{rel_path}'...")
        try:
            new_content = _call_editor(instructions, current_content)
        except Exception as exc:
            print(f"  [file_editor] Ollama editor call failed for '{rel_path}': {exc}")
            all_ok = False
            continue

        # Strip accidental markdown fences from the editor's output
        stripped = new_content.strip()
        if stripped.startswith("```"):
            stripped = stripped.lstrip("`")
            if stripped.startswith(("python", "txt", "text")):
                stripped = stripped.split("\n", 1)[1] if "\n" in stripped else stripped
            if stripped.endswith("```"):
                stripped = stripped[:-3]
            new_content = stripped.strip()

        # ── Validate ───────────────────────────────────────────────────────────
        ok, err = _validate(rel_path, new_content)
        if not ok:
            print(f"  [file_editor] Validation FAILED for '{rel_path}': {err} — not written.")
            all_ok = False
            continue

        # ── Write ──────────────────────────────────────────────────────────────
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"  [file_editor] '{rel_path}' written successfully.")

    return all_ok
