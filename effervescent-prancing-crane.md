# Plan: Economic/Production Gameplay for TacBench

## Context

TacBench currently handles only pure tactical combat — the LLM moves/attacks/retreats existing units, but can't gather resources, construct buildings, or train new units. The user wants to expand into the full SC2 macro loop. `maps/tacbench_03.py` already exists with the objective "Build Barracks then Marines to destroy the enemy Command Center" — this plan makes that map actually work.

The core challenge: existing directive handlers are synchronous, but `bot.build()` (burnysc2's auto-placement build call) is a coroutine and must be awaited. The solution is to make `execute_directive` async and use `asyncio.iscoroutine()` to await economic handlers while leaving the existing sync handlers untouched.

---

## Files to Modify

1. `core/directives/directive.py` — add `unit_type` field
2. `core/directives/execute_directive.py` — add 3 handlers, make async
3. `core/bot/BaseSC2Bot.py` — make `_apply_llm_result` async, add await at call sites
4. `core/observation/obs_raw_text.py` — add `_fmt_economy()` section
5. `core/settings.py` — add `show_economy` toggle
6. `prompt.txt` — document 3 new directives

---

## Step 1 — `core/directives/directive.py`

Add one field to the `Directive` dataclass after `target_unit` (line 16):
```python
unit_type: Optional[str] = None   # e.g. "MARINE", "BARRACKS" for BUILD/TRAIN
```

In `_normalize_from_dict` (line 65), add parsing after the `target_unit` block (after line 100):
```python
unit_type = None
raw_ut = raw.get("unit_type")
if isinstance(raw_ut, str):
    unit_type = raw_ut.strip().upper()
```

Add `unit_type=unit_type` to the `Directive(...)` constructor call at line 108.

---

## Step 2 — `core/directives/execute_directive.py`

**2a.** Add imports near the top (after existing imports):
```python
import asyncio
from sc2.ids.unit_typeid import UnitTypeId
```

**2b.** Add two lookup tables after `_RETREAT_DISTANCE` (line 10):
```python
_BUILDABLE_STRUCTURES: dict[str, UnitTypeId] = {
    "BARRACKS": UnitTypeId.BARRACKS, "SUPPLY_DEPOT": UnitTypeId.SUPPLYDEPOT,
    "FACTORY": UnitTypeId.FACTORY, "STARPORT": UnitTypeId.STARPORT,
    "ENGINEERING_BAY": UnitTypeId.ENGINEERINGBAY, "ARMORY": UnitTypeId.ARMORY,
    "BUNKER": UnitTypeId.BUNKER, "MISSILE_TURRET": UnitTypeId.MISSILETURRET,
    "REFINERY": UnitTypeId.REFINERY, "COMMAND_CENTER": UnitTypeId.COMMANDCENTER,
}

# (unit_type_name) → (UnitTypeId to train, UnitTypeId of producing structure)
_TRAINABLE_UNITS: dict[str, tuple[UnitTypeId, UnitTypeId]] = {
    "MARINE":    (UnitTypeId.MARINE,    UnitTypeId.BARRACKS),
    "MARAUDER":  (UnitTypeId.MARAUDER,  UnitTypeId.BARRACKS),
    "REAPER":    (UnitTypeId.REAPER,    UnitTypeId.BARRACKS),
    "GHOST":     (UnitTypeId.GHOST,     UnitTypeId.BARRACKS),
    "HELLION":   (UnitTypeId.HELLION,   UnitTypeId.FACTORY),
    "SIEGETANK": (UnitTypeId.SIEGETANK, UnitTypeId.FACTORY),
    "THOR":      (UnitTypeId.THOR,      UnitTypeId.FACTORY),
    "VIKING":    (UnitTypeId.VIKINGFIGHTER, UnitTypeId.STARPORT),
    "MEDIVAC":   (UnitTypeId.MEDIVAC,   UnitTypeId.STARPORT),
    "BANSHEE":   (UnitTypeId.BANSHEE,   UnitTypeId.STARPORT),
    "SCV":       (UnitTypeId.SCV,       UnitTypeId.COMMANDCENTER),
}
```

**2c.** Add three new handlers before `DIRECTIVE_REGISTRY` (after `retreat`, line 138):
```python
async def build(bot: BotAI, army, enemies, directive: Directive):
    """Build a structure using auto worker selection and auto placement."""
    if not directive.unit_type:
        console.warn("BUILD directive missing unit_type — no action taken.")
        return
    structure_id = _BUILDABLE_STRUCTURES.get(directive.unit_type)
    if structure_id is None:
        console.warn(f"BUILD: unknown unit_type '{directive.unit_type}' — no action taken.")
        return
    if directive.target_x is not None and directive.target_y is not None:
        near = Point2((directive.target_x, directive.target_y))
    else:
        near = bot.start_location
    await bot.build(structure_id, near=near)


async def train(bot: BotAI, army, enemies, directive: Directive):
    """Train a unit from the first idle structure capable of producing it."""
    if not directive.unit_type:
        console.warn("TRAIN directive missing unit_type — no action taken.")
        return
    entry = _TRAINABLE_UNITS.get(directive.unit_type)
    if entry is None:
        console.warn(f"TRAIN: unknown unit_type '{directive.unit_type}' — no action taken.")
        return
    unit_type_id, structure_type_id = entry
    producers = bot.structures(structure_type_id).ready.idle
    if not producers:
        console.warn(f"TRAIN: no idle {structure_type_id.name} — no action taken.")
        return
    producers.first.train(unit_type_id)


def gather(bot: BotAI, army, enemies, directive: Directive):
    """Send idle workers to nearest mineral field."""
    workers = bot.workers.idle
    if not workers:
        return
    if not bot.mineral_field:
        console.warn("GATHER: no mineral fields visible — no action taken.")
        return
    for worker in workers:
        worker.gather(bot.mineral_field.closest_to(worker))
```

**2d.** Add to `DIRECTIVE_REGISTRY` (lines 141-148):
```python
    "BUILD":         build,
    "TRAIN":         train,
    "GATHER":        gather,
```

**2e.** Make `execute_directive` async and handle mixed sync/async. Replace line 189-218:
```python
async def execute_directive(bot: BotAI, directive, fallback: str = "HOLD_POSITION") -> None:
    """
    Translate a named directive into python-sc2 unit actions.
    Supports both sync and async handlers — async handlers are awaited automatically.
    Called once when a new LLM response arrives — not repeated each game step.
    """
    if isinstance(directive, Directive):
        directive_name = directive.name
    else:
        directive_name = str(directive)
        directive = Directive(name=directive_name)

    all_army = bot.units
    enemies = bot.enemy_units.visible

    # Economic directives work with zero army units; tactical ones need an army.
    if not all_army and directive_name not in ("BUILD", "TRAIN", "GATHER"):
        return

    army = _resolve_army(bot, all_army, directive)

    handler = DIRECTIVE_REGISTRY.get(directive_name)
    if handler is not None:
        result = handler(bot, army, enemies, directive)
        if asyncio.iscoroutine(result):
            await result
        return

    fallback_handler = DIRECTIVE_REGISTRY.get(fallback)
    if fallback_handler is not None:
        console.warn(f"directive '{directive_name}' not in registry — falling back to '{fallback}'.")
        result = fallback_handler(bot, army, enemies, directive)
        if asyncio.iscoroutine(result):
            await result
    else:
        console.warn(f"directive '{directive_name}' not in registry and fallback '{fallback}' also not registered.")
```

---

## Step 3 — `core/bot/BaseSC2Bot.py`

**3a.** Change `_apply_llm_result` signature (line 370) to `async def _apply_llm_result(...)`.

**3b.** Update the execute loop (line 404-406) to await:
```python
for directive in directives:
    await execute_directive(self, directive, fallback=self.FALLBACK_DIRECTIVE)
```

**3c.** Add `unit_type` to the log dict (lines 412-424):
```python
{
    "directive":     d.name,
    "units":         d.units,
    "target_x":      d.target_x,
    "target_y":      d.target_y,
    "target_unit":   d.target_unit,
    "unit_type":     d.unit_type,      # new
    "reasoning":     d.reasoning,
    "fallback_used": d.fallback_used,
    "error":         d.error,
}
```

**3d.** In `_check_llm_task` (line 367), add `await`:
```python
await self._apply_llm_result(result)
```

**3e.** In `on_step` non-realtime path, add `await` to the `_apply_llm_result` call (wherever it exists in the non-realtime branch — search for `self._apply_llm_result` in `on_step`).

> **Important:** Steps 2e and 3 must be applied in the same edit session — after `execute_directive` becomes async, any non-awaited call silently discards the coroutine.

---

## Step 4 — `core/observation/obs_raw_text.py`

Add a new formatter after `_fmt_supply` (around line 128):
```python
def _fmt_economy(bot: BotAI) -> str:
    """Worker counts, structures under construction, and production queues."""
    lines = ["ECONOMY:"]

    total_workers = len(bot.workers)
    idle_workers  = len(bot.workers.idle)
    lines.append(f"  Workers: {total_workers} total, {idle_workers} idle")

    pending = [s for s in bot.structures if not s.is_ready]
    for s in pending:
        pct = int(s.build_progress * 100)
        lines.append(f"  [BUILDING] {s.name} {pct}% @ ({s.position.x:.0f},{s.position.y:.0f})")

    producing = [s for s in bot.structures.ready if s.orders]
    for s in producing:
        order_names = [str(o.ability).split(".")[-1] for o in s.orders]
        lines.append(f"  {s.name} @ ({s.position.x:.0f},{s.position.y:.0f}): training {', '.join(order_names)}")

    return "\n".join(lines)
```

In `obs_raw_text()`, add the call after the supply section:
```python
if cfg.show_supply:
    sections.append(_fmt_supply(bot))
if cfg.show_economy:          # new
    sections.append(_fmt_economy(bot))
```

Also add economic examples to the inline `directives_format_reminder` string at the bottom of `obs_raw_text()`:
```
Economic examples:
{"reasoning": "...", "directive": "BUILD", "unit_type": "BARRACKS"}
{"reasoning": "...", "directive": "TRAIN", "unit_type": "MARINE"}
{"reasoning": "...", "directive": "GATHER"}
```

---

## Step 5 — `core/settings.py`

Add after `show_supply` (line 27):
```python
show_economy: bool = False   # worker counts, build queue, structures under construction
```

Default is `False` — existing combat maps (tacbench_01, tacbench_02) don't need this noise. Enable with `TACBENCH_SHOW_ECONOMY=true` in `.env` for tacbench_03.

---

## Step 6 — `prompt.txt`

Add three new entries to the DIRECTIVE REFERENCE section:

```
BUILD  ← construct a structure
  Order an SCV to build the specified structure. Auto-selects a worker and
  finds a valid placement near the target or your start location.
  REQUIRES unit_type. target_x/target_y are optional.
  Available: BARRACKS, SUPPLY_DEPOT, FACTORY, STARPORT, ENGINEERING_BAY,
             ARMORY, BUNKER, MISSILE_TURRET, REFINERY, COMMAND_CENTER
  Example: {"reasoning": "...", "directive": "BUILD", "unit_type": "BARRACKS"}

TRAIN  ← produce a unit from a structure
  Starts training the specified unit in the first idle structure of the correct type.
  REQUIRES unit_type.
  Barracks: MARINE, MARAUDER, REAPER, GHOST
  Factory:  HELLION, SIEGETANK, THOR
  Starport: VIKING, MEDIVAC, BANSHEE
  Command Center: SCV
  Example: {"reasoning": "...", "directive": "TRAIN", "unit_type": "MARINE"}

GATHER  ← send idle workers to mine
  Sends all currently idle SCVs to the nearest mineral field.
  Use after a build order completes to restore income.
  Example: {"reasoning": "...", "directive": "GATHER"}
```

---

## Verification

1. Run `python tests/test_directive.py` — all existing tests should pass. Add tests for BUILD/TRAIN/GATHER parsing (verify `unit_type` is parsed and uppercased).

2. Add to `.env`:
   ```
   TACBENCH_MAP=tacbench_03
   TACBENCH_SHOW_ECONOMY=true
   TACBENCH_BOT_TYPE=claude
   TACBENCH_MODEL_NAME=claude-sonnet-4-6
   ```

3. Run `python core/main.py` — confirm:
   - Economy section appears in the console/log observations
   - The LLM issues BUILD/TRAIN/GATHER directives
   - Barracks appears on the map after a BUILD directive
   - Marines are trained after TRAIN directives
   - Idle SCVs return to mining after GATHER

4. Confirm existing maps still work: `TACBENCH_MAP=tacbench_01 python core/main.py`
