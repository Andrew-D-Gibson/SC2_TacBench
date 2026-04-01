# TacBench — CLAUDE.md

This file exists to get Claude back up to speed quickly at the start of a new session.

---

## What This Project Is

**TacBench** is a StarCraft II tactical AI benchmark and self-improvement loop. An LLM (Claude or a local Ollama model) controls a small SC2 army by reading a text battlefield summary every K game steps and issuing JSON directives (ATTACK, RETREAT, HOLD_POSITION, etc.).

The project has two top-level systems:

1. **Game loop** (`core/`) — runs inside python-sc2, translates game state into LLM prompts, parses and executes LLM directives, and logs everything to `logs/`.
2. **Orchestrator** (`orchestrator/`) — a self-improvement meta-loop that runs games, analyzes failures with a second LLM call, edits `prompt.txt` and/or `core/observation/obs_raw_text.py`, re-runs, and keeps or git-reverts changes based on measured improvement.

---

## Directory Structure

```
SC2_TacBench/
├── CLAUDE.md                          ← you are here
├── prompt.txt                         ← the system prompt fed to the playing LLM (editable by orchestrator)
├── .env                               ← local settings (not committed); see .env.example
├── requirements.txt                   ← Python >= 3.11, burnysc2, pydantic-settings, anthropic
│
├── orchestrator/                      ← self-improvement loop (prompt engineering meta-system)
│   ├── __init__.py
│   ├── orchestrator.py                ← main entry point: `python orchestrator/orchestrator.py`
│   ├── config.py                      ← all orchestrator constants (paths, backends, limits)
│   ├── meta_reasoner.py               ← two-phase LLM analysis of failures → edit decision JSON
│   ├── file_editor.py                 ← takes edit instructions, generates new file content via LLM
│   └── orc_console.py                 ← rich terminal UI for the orchestrator loop
│
├── core/                              ← SC2 game bot
│   ├── __init__.py
│   ├── main.py                        ← game entry point: `python core/main.py`
│   ├── settings.py                    ← pydantic-settings; all TACBENCH_* .env vars live here
│   ├── console.py                     ← ANSI console output for in-game display
│   ├── map_loader.py                  ← dynamically loads maps/<name>.py at game start
│   │
│   ├── bot/                           ← bot implementations (all inherit BaseSC2Bot)
│   │   ├── __init__.py
│   │   ├── BaseSC2Bot.py              ← game loop, LLM scheduling, directive execution, logging
│   │   ├── ClaudeBot.py               ← calls Anthropic Messages API (async)
│   │   ├── OllamaBot.py               ← calls local Ollama endpoint (streaming)
│   │   └── FairlibBot.py              ← wraps LLM in fairlib ReAct agent stack
│   │
│   ├── tactics/                       ← tactical analysis
│   │   ├── __init__.py
│   │   └── clustering.py              ← spatial unit clustering, threat/ratio labels, ghost tracking
│   │
│   ├── observation/                   ← battlefield observation formatting (primary prompt engineering surface)
│   │   ├── __init__.py
│   │   ├── obs_raw_text.py            ← converts game state → LLM-readable text (editable by orchestrator)
│   │   └── terrain_encoder.py         ← ASCII terrain grid from SC2 height/pathing maps
│   │
│   └── directives/                    ← LLM output parsing and execution
│       ├── __init__.py
│       ├── directive.py               ← parses raw LLM output → Directive dataclass
│       └── execute_directive.py       ← executes directives as SC2 unit commands; pluggable registry
│
├── maps/                              ← map scenario files
│   ├── __init__.py                    ← BaseMapScenario base class
│   ├── tacbench_01.py                 ← Destroy enemy Command Center; lose if army count = 0
│   └── tacbench_02.py                 ← (second scenario, same pattern)
│
├── tests/
│   └── test_directive.py              ← unit tests for directive parsing/normalization
│
├── misc/
│   └── test_pipeline.py               ← integration test for the full bot game loop
│
├── logs/                              ← JSONL episode logs, one file per run (gitignored)
└── logs_important/                    ← manually curated notable runs
```

---

## How to Run Things

### Run a single game
```bash
# From project root. Settings come from .env (TACBENCH_* prefix).
python core/main.py

# Override map without touching .env:
TACBENCH_MAP=tacbench_02 python core/main.py

# Use Claude instead of Ollama:
TACBENCH_BOT_TYPE=claude TACBENCH_MODEL_NAME=claude-sonnet-4-6 python core/main.py
```

### Run the self-improvement orchestrator
```bash
python orchestrator/orchestrator.py
```

### Run tests
```bash
python tests/test_directive.py
```

---

## Import Conventions

**All internal imports use absolute paths from the project root.** Both entry points (`core/main.py` and `orchestrator/orchestrator.py`) add project root to `sys.path` at startup via:
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
```

This means every other file can use the full package path regardless of working directory:
```python
from core.observation.obs_raw_text import obs_raw_text
from core.tactics.clustering import ClusterTracker
from core.directives.directive import Directive
from orchestrator.config import PROJECT_ROOT
```

Never use flat imports like `from clustering import ...` — those were the old pre-reorganization style and will break.

---

## Configuration (.env)

All game settings use the `TACBENCH_` prefix. Orchestrator settings do not. Key values:

| Variable | Default | Notes |
|---|---|---|
| `TACBENCH_MAP` | `tacbench_01` | map ID, must match `maps/<name>.py` |
| `TACBENCH_BOT_TYPE` | `ollama` | `ollama` \| `claude` \| `fairlib` |
| `TACBENCH_MODEL_NAME` | `qwen3:8b` | model name passed to selected bot |
| `TACBENCH_K_STEPS` | `30` | game steps between LLM calls |
| `TACBENCH_MAX_STEPS` | `1000` | episode timeout (steps) |
| `TACBENCH_FALLBACK_DIRECTIVE` | `HOLD_POSITION` | used if LLM output is unparseable |
| `TACBENCH_PLAYER_RACE` | `Terran` | |
| `TACBENCH_OPPONENT_RACE` | `Terran` | |
| `TACBENCH_OPPONENT_DIFFICULTY` | `Easy` | SC2 difficulty enum name |
| `TACBENCH_CLUSTER_RADIUS` | `12.0` | tile radius for spatial unit grouping |
| `TACBENCH_CLUSTER_TRACK_INTERVAL` | `5` | steps between velocity snapshots |
| `TACBENCH_SHOW_TACTICAL_OVERVIEW` | `true` | cluster+matchup section in prompt |
| `TACBENCH_SHOW_TERRAIN` | `false` | ASCII terrain grid (~500–1500 tokens) |
| `TACBENCH_TERRAIN_DOWNSAMPLE` | `4` | higher = fewer tokens |
| `TACBENCH_SHOW_HISTORY` | `false` | include recent decision history |
| `TACBENCH_HISTORY_LENGTH` | `3` | how many past calls to include |
| `TACBENCH_SHOW_LLM_PROMPT` | `false` | print full prompt to console (debug) |
| `ANTHROPIC_API_KEY` | — | no prefix; read directly by pydantic Field alias |

Orchestrator-specific (in `orchestrator/config.py`, also read from `.env`):

| Variable | Default | Notes |
|---|---|---|
| `META_REASONER_BACKEND` | `ollama` | `ollama` \| `claude` |
| `FILE_EDITOR_BACKEND` | `ollama` | `ollama` \| `claude` |
| `CLAUDE_META_MODEL` | `claude-sonnet-4-6` | used when backend = claude |
| `OLLAMA_MODEL` | `qwen3.5:9b` | used when backend = ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | |
| `OLLAMA_TIMEOUT` | `600` | seconds; meta-reasoning is slow |
| `MAX_ITERATIONS` | `1` | orchestrator loop iterations |
| `MAX_STAGNANT_ITERATIONS` | `5` | stop if no improvement after N iters |

---

## Key Architectural Patterns

### Game Loop (BaseSC2Bot.on_step)
Every game step:
1. Check step limit → `TIMEOUT` if exceeded.
2. Run `MapScenario.on_step()`, then check win/loss conditions.
3. Every `CLUSTER_TRACK_INTERVAL` steps: update `ClusterTracker` (velocity snapshots).
4. Every `K_STEPS` steps AND no LLM call in flight: build `obs_raw_text`, prepend briefing + history, fire async LLM task.
5. Every step: check if pending LLM task completed; if so, parse directives and cache them.
6. Every step: execute the most recently cached directive.

LLM calls are **non-blocking** — a `asyncio.Task` is created and polled each step. If a prior call is still running when the next K-step interval arrives, the new call is skipped entirely to avoid HP-delta drift.

### Observation Text Format (obs_raw_text.py)
The battlefield text fed to the LLM is assembled from sections controlled by `.env` toggles:
- `GAME TIME:` — current step and wall time
- `SUPPLY:` — current/max supply
- `LOCATIONS:` — start location, known enemy structures
- `TACTICAL OVERVIEW:` — clusters with matchup analysis (the main tactical payload)
- `YOUR UNITS(N):` / `ENEMY UNITS(N):` — individual unit lines
- `YOUR STRUCTURES:` / `ENEMY STRUCTURES:` — structure listings
- `TERRAIN:` — ASCII grid (optional, expensive)
- `LAST KNOWN ENEMY POSITIONS:` — ghost clusters that left vision

The tactical overview cluster header format:
```
YOUR FORCES:
  GROUP A [GND]: 8u @ (32,45) | 640/800HP [80%] | moving NE (3.8 tiles/call)
    [Marine x6, Medivac x2] | cost:700 | atk+1.0 arm+0.0
    vs CLUSTER 1 [GND] 14u @ (8,23) dist 14.4 rng:6 | gnd-threat: NEARBY | str 14.0 | ratio 8/14.0 [DISADVANTAGED] | stationary

ENEMY FORCES:
  CLUSTER 1 [GND]: 14u @ (8,23) | 950/950HP [100%] | gnd-rng:6 air-rng:0 | stationary
    [Marine x10, Marauder x4] | cost:1000
```

Key fields:
- `dist` — Euclidean distance between cluster centers
- `rng:N` — enemy weapon range relevant to the friendly cluster type (air or ground)
- `gnd-rng` / `air-rng` — max ground/air weapon range of any unit in the enemy cluster
- `cost:N` — total mineral+gas value (minerals + gas, single combined number)
- `str N.N` — HP-weighted count of enemy units that can attack the friendly type
- `ratio X/Y [LABEL]` — friendly count vs enemy strength → ADVANTAGED/EVEN/DISADVANTAGED/CRITICAL
- `gnd-threat:` label — SAFE / NEARBY / THREAT / CONTACT (based on range multipliers vs distance)

### Clustering (clustering.py)
- Ground and air units are **always clustered separately** (so ground threat labels don't bleed into air matchups).
- `cluster_units()` is a greedy radius-based algorithm: seed on first unassigned unit, grow by pulling in units within `cluster_radius` tiles, recompute centroid, repeat.
- `ClusterTracker` stores position snapshots every `CLUSTER_TRACK_INTERVAL` steps to compute velocity vectors.
- Range and strength metrics (`max_ground_range`, `max_air_range`, `anti_ground_strength`, `anti_air_strength`) are computed from the sc2 library's unit properties (`u.ground_range`, `u.air_range`, `u.can_attack_ground`, `u.can_attack_air`).
- Ghost tracking: enemy clusters that go out of vision are stored as `ghost_enemy_positions` (pruned after `_GHOST_MAX_STEPS = 300` steps).

Force ratio thresholds (in `clustering.py`):
```python
_RATIO_ADVANTAGED    = 1.2   # friendly/enemy >= 1.2 → ADVANTAGED
_RATIO_EVEN          = 0.8   # >= 0.8 → EVEN
_RATIO_DISADVANTAGED = 0.5   # >= 0.5 → DISADVANTAGED; below → CRITICAL
```

Threat distance multipliers (relative to enemy weapon range):
```python
_CONTACT_RANGE_FACTOR = 1.5   # dist <= range * 1.5 → CONTACT
_THREAT_RANGE_FACTOR  = 2.0   # dist <= range * 2.0 → THREAT
_NEARBY_RANGE_FACTOR  = 3.0   # dist <= range * 3.0 → NEARBY
```

### Directive System
`directive.py` handles all the messy LLM output: JSON strings, raw dicts, markdown code fences, plain English strings, and arrays of directives. Everything normalizes to `Directive` dataclass instances.

`execute_directive.py` uses a registry pattern:
```python
DIRECTIVE_REGISTRY = {
    "ATTACK": attack,
    "MOVE": move,
    "RETREAT": retreat,
    "HOLD_POSITION": hold_position,
    "SPREAD": spread,
    "FOCUS_FIRE": focus_fire,
}
```
Custom directives can be registered via `register_directive(name, handler)`. The `_resolve_army()` helper filters the friendly unit collection by cluster labels ("A", "B") or individual unit IDs from `directive.units`.

### Episode Logging
Each game writes one JSONL file to `logs/` named `{datetime}__{botname}__{mapname}__log.jsonl`.

The first line is always a `type: "summary"` entry with `outcome` (WIN/LOSS/TIMEOUT/TIE), `total_steps`, `total_llm_calls`, `total_llm_failures`, and config snapshot.

Subsequent lines are `type: "llm_call"` entries (one per LLM interaction) and a final `type: "final_state"` snapshot.

The orchestrator reads only the first line of each log to determine outcome and step count for improvement tracking.

### Map Scenarios
Each map is a Python file in `maps/` that defines a class named exactly `MapScenario(BaseMapScenario)`.

Required interface:
```python
class MapScenario(BaseMapScenario):
    briefing: str = "..."          # prepended to every LLM prompt for this map
    settings_overrides: dict = {}  # reserved for future per-map setting injection

    def on_step(self, bot) -> None: ...   # update internal state
    def check_win(self, bot) -> bool: ... # return True → WIN
    def check_loss(self, bot) -> bool: ...# return True → LOSS
```

`map_loader.py` dynamically imports by normalizing the map name (lowercase, spaces → underscores) and loading `maps/<name>.py`. The `briefing` string is prepended to each LLM prompt before the battlefield observation.

### Orchestrator Safety
The orchestrator uses git to protect against bad edits:
1. `git_baseline()` — commits any dirty state before the loop starts.
2. Before each edit: `git stash` saves the current state.
3. Edits are applied and maps re-run.
4. If improved: `git commit` the edit and `git stash drop`.
5. If not improved: `git stash pop` to revert.
6. On Ctrl-C: stash may be left pending — run `git stash list` to check.

The **file edit whitelist** (in `orchestrator/config.py`) controls what the orchestrator can touch:
```python
EDITABLE_FILES = [
    "prompt.txt",
    "core/observation/obs_raw_text.py",
]
```
All paths are relative to `PROJECT_ROOT`. `file_editor.py` enforces this list as a hard check.

---

## Prompt Engineering Notes

The LLM playing the game receives two things:
1. **System prompt** (`prompt.txt`) — tactical doctrine, directive reference, coordinate system explanation. This is loaded once at bot import time. The orchestrator can edit this file.
2. **User message** — the battlefield observation produced by `obs_raw_text()`. The orchestrator can also edit `obs_raw_text.py` to change what information is presented and how.

The most impactful levers for improving bot performance are:
- Adjusting tactical doctrine in `prompt.txt` (rule priorities, decision heuristics).
- Changing what the tactical overview shows or how it's formatted in `obs_raw_text.py`.
- Tuning `TACBENCH_K_STEPS` (call frequency vs. token cost tradeoff).

---

## Recent Changes Worth Knowing

- **Cluster cost display**: changed from `cost:900m/100g` to `cost:1000` (single mineral+gas total).
- **Enemy cluster header**: now shows `gnd-rng:N air-rng:N` — max ground/air weapon range of any unit in the cluster.
- **Matchup lines (YOUR FORCES)**: the relevant enemy weapon range now appears immediately after distance as `rng:N`, so the LLM can compare distance vs. effective range at a glance. The redundant `range N` field that appeared mid-line was removed.
- **Directory reorganization**: `core/` was split into `core/bot/`, `core/tactics/`, `core/observation/`, `core/directives/`. The orchestrator files moved from project root to `orchestrator/`. All imports use absolute `from core.xxx` style.
