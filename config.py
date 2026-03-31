# config.py — All tunable constants for the SC2 prompt optimization orchestrator.
#
# Discovered codebase facts:
#   Player prompt:         prompt.txt (project root)
#                          Loaded by each Bot class via _PROMPT_PATH / read_text()
#   Formatting functions:  core/obs_raw_text.py
#                          Contains all obs/unit/cluster/terrain formatters
#   Map runner:            python core/main.py
#                          Reads settings via pydantic-settings from .env
#                          os.environ takes precedence, so TACBENCH_MAP can be injected
#                          per-run without touching .env
#   Map IDs:               tacbench_01, tacbench_02
#                          Correspond to maps/tacbench_01.py, maps/tacbench_02.py
#   Log location:          ./logs/  (JSONL, one file per run)
#   Log summary entry:     first line, type="summary", fields: outcome, total_steps

import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ── Project paths ──────────────────────────────────────────────────────────────

PROJECT_ROOT    = os.path.dirname(os.path.abspath(__file__))
PLAYER_ENTRY    = os.path.join(PROJECT_ROOT, "core", "main.py")
LOGS_DIR        = os.path.join(PROJECT_ROOT, "logs")
ORCHESTRATOR_LOG_PATH = os.path.join(PROJECT_ROOT, "orchestrator_log.jsonl")

# ── Files the meta-reasoner is permitted to edit ───────────────────────────────
# Paths are relative to PROJECT_ROOT.  file_editor.py enforces this whitelist.

EDITABLE_FILES = [
    "prompt.txt",
    "core/obs_raw_text.py",
]

# ── Maps to evaluate each iteration ───────────────────────────────────────────

MAP_LIST = [
    "tacbench_01",
    "tacbench_02",
]

# ── Ollama settings ────────────────────────────────────────────────────────────

OLLAMA_MODEL    = "qwen3:8b"               # meta-reasoner model
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_TIMEOUT  = 600                      # seconds; meta-reasoning is slower than playing

# ── Meta-reasoner backend ──────────────────────────────────────────────────────
# Set META_REASONER_BACKEND="claude" (and ensure ANTHROPIC_API_KEY is in .env
# or the environment) to use the Claude API instead of Ollama for both the
# analysis and decision phases.

META_REASONER_BACKEND = "claude"          # "ollama" | "claude"
FILE_EDITOR_BACKEND   = "claude"          # "ollama" | "claude"
CLAUDE_META_MODEL     = "claude-sonnet-4-6"
ANTHROPIC_API_KEY     = os.environ.get("ANTHROPIC_API_KEY")  # set in .env or shell

# ── Loop limits ────────────────────────────────────────────────────────────────

MAX_ITERATIONS          = 1
MAX_STAGNANT_ITERATIONS = 5

