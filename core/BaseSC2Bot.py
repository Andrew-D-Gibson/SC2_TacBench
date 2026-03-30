from obs_raw_text import obs_raw_text
from execute_directive import execute_directive, get_directive_registry
from directive import Directive, normalize_directives
from settings import get_settings
from map_loader import load_map_scenario
from clustering import ClusterTracker
import console

from sc2.bot_ai import BotAI
from sc2.data import Result

import json
import asyncio
import time
from pathlib import Path
from datetime import datetime


class BaseSC2Bot(BotAI):
    """
    Base class for bots running in the SC2 TacBench software.
    Inherits from python-sc2's / burnysc2's BotAI class.

    Configuration (K_STEPS, MAX_STEPS, etc.) is loaded from .env via settings.py.
    See TacBenchSettings for all available options and their defaults.
    """

    # The name of the SC2_TacBench bot being implemented.
    name: str = "BaseSC2Bot"


    def __init__(self):
        super().__init__()
        self.step_count = 0
        self.episode_log = []
        self.llm_failures = 0
        self._pending_llm_task = None
        self._map_scenario = None
        self._episode_log_written = False

        # Settings are loaded from .env — see settings.py and TacBenchSettings.
        self._apply_settings()

        # Cluster tracking — updated every cluster_track_interval game steps.
        # Stores (friendly_ground, friendly_air, enemy_ground, enemy_air).
        self._cluster_tracker = ClusterTracker()
        self._cluster_state: tuple = ([], [], [], [])

        # Persistent unit identifiers: tag → short sequential ID (for LLM targeting).
        self._unit_id_map: dict[int, int] = {}
        self._next_unit_id: int = 1

        # HP history for delta reporting: tag → HP% at last LLM observation.
        self._unit_hp_history: dict[int, int] = {}


    def _apply_settings(self):
        """
        Load and apply configuration from .env / environment variables.
        All frequently-used settings are cached here so on_step doesn't
        need to call get_settings() on every game tick.
        """
        settings = get_settings()
        self.K_STEPS               = settings.k_steps
        self.MAX_STEPS             = settings.max_steps
        self.FALLBACK_DIRECTIVE    = settings.fallback_directive
        self.MODEL_NAME            = settings.model_name
        self.CLUSTER_TRACK_INTERVAL = settings.cluster_track_interval
        self.CLUSTER_RADIUS        = settings.cluster_radius
        self.SHOW_LLM_PROMPT       = settings.show_llm_prompt
        self.SHOW_HISTORY          = settings.show_history
        self.HISTORY_LENGTH        = settings.history_length

    # --- Unit ID and HP delta helpers (used by obs_raw_text) ---

    def get_unit_id(self, unit) -> int:
        """Return a persistent short integer ID for a unit, assigning one if new."""
        if unit.tag not in self._unit_id_map:
            self._unit_id_map[unit.tag] = self._next_unit_id
            self._next_unit_id += 1
        return self._unit_id_map[unit.tag]

    def get_hp_delta(self, unit) -> int | None:
        """
        Return the HP% change since the last time this unit was observed.
        Updates the stored HP% as a side effect — call once per unit per LLM step.
        Returns None on the unit's first appearance.
        """
        tag = unit.tag
        current = int(100 * unit.health / max(unit.health_max, 1))
        delta = current - self._unit_hp_history[tag] if tag in self._unit_hp_history else None
        self._unit_hp_history[tag] = current
        return delta


    # --- Overridable hooks (subclasses should focus here) ---

    def get_new_directive(self, current_battlefield_obs: str, step: int = 0):
        """
        Sync directive generator. Override in subclasses if you prefer a sync API.
        Should return a dict with at least a 'directive' key, e.g.:
            {"directive": "FOCUS_FIRE", "reasoning": "Enemy is weak."}
        For directives that require a target position, include 'target_x' and 'target_y':
            {"directive": "ATTACK", "target_x": 32.0, "target_y": 64.0, "reasoning": "..."}
        """
        return {
            "directive": "FOCUS_FIRE",
            "reasoning": "[FAKE LLM] Default stub — override get_new_directive in your subclass.",
            "target_x": None,
            "target_y": None,
        }


    async def get_new_directive_async(self, current_battlefield_obs: str, step: int = 0):
        """
        Async directive generator. Override this in subclasses that call an async LLM.
        """
        return self.get_new_directive(current_battlefield_obs, step=step)


    # --- BotAI lifecycle methods ---

    async def on_start(self):
        """
        Called once at the beginning of a game.
        Loads the map scenario file matching the current map name.
        """
        console.print_game_start(self.game_info.map_name, self.race) #self.start_location

        settings = get_settings()
        self._map_scenario = load_map_scenario(settings.map)


    async def on_step(self, iteration: int):
        """
        Called every game step.
        Pipeline:
        1) Enforce step limit.
        2) Update map scenario state and check win/loss conditions.
        3) Consume completed LLM tasks (if any).
        4) Every K steps, schedule a new LLM call (non-blocking).
        5) Execute cached directive every step.
        """
        self.step_count += 1

        # Auto-end the scenario if we've reached the maximum allowed time.
        if self.step_count >= self.MAX_STEPS:
            console.warn("Step limit reached — ending episode.")
            self._write_episode_log("TIMEOUT")
            await self.client.leave()
            return

        # Update map scenario state, then check win/loss.
        if self._map_scenario:
            self._map_scenario.on_step(self)

            if self._map_scenario.check_win(self):
                console.warn("Win condition met — ending episode.")
                self._write_episode_log("WIN")
                await self.client.leave()
                return

            if self._map_scenario.check_loss(self):
                console.warn("Loss condition met — ending episode.")
                self._write_episode_log("LOSS")
                await self.client.leave()
                return

        # Update cluster state (velocity tracking) more often than LLM calls.
        # Also update on LLM call steps to guarantee clusters are populated
        # before the first prompt (handles K_STEPS < cluster_track_interval).
        if self.step_count % self.CLUSTER_TRACK_INTERVAL == 0 or self.step_count % self.K_STEPS == 0:
            self._cluster_state = self._cluster_tracker.update(
                self, self.step_count, self.CLUSTER_RADIUS
            )

        # Check whether a prior LLM task finished and update cached directive.
        await self._check_llm_task()

        # Only build the battlefield observation (and snapshot HP history for
        # delta tracking) when we are actually going to send it to the LLM.
        # If the previous call is still in-flight we skip entirely — otherwise
        # get_hp_delta would advance _unit_hp_history without the LLM ever
        # seeing those values, causing the structure alerts to drift out of sync.
        if self.step_count % self.K_STEPS == 0:
            if not (self._pending_llm_task and not self._pending_llm_task.done()):
                battlefield = obs_raw_text(self, self.step_count)
                if self._map_scenario and self._map_scenario.briefing:
                    battlefield = self._map_scenario.briefing + "\n\n" + battlefield
                if self.SHOW_HISTORY and self.episode_log:
                    battlefield += "\n\n" + self._build_history_section(self.HISTORY_LENGTH)

                self._schedule_llm_call(self.step_count, battlefield)



    async def on_end(self, game_result: Result):
        """
        Called once at the end of a game.
        Skips entirely if win/loss was already handled in on_step — this is
        the normal path in realtime mode, where client.leave() causes SC2
        to report a Defeat regardless of the actual outcome.
        """
        if self._episode_log_written:
            return

        outcome = "WIN" if game_result == Result.Victory else \
                  "LOSS" if game_result == Result.Defeat else \
                  "TIE"

        console.print_game_over(outcome, self.step_count, len(self.episode_log))
        self._write_episode_log(outcome)


    # --- Logging and LLM scheduling helpers ---

    def _write_episode_log(self, outcome: str):
        """
        Persist a JSONL log for this run. Idempotent — safe to call from
        both on_step (win/loss detection) and on_end (engine callback).
        Appends a final_state snapshot of the battlefield at the moment of ending.
        """
        if self._episode_log_written:
            return

        # Snapshot the battlefield at end-of-game before writing.
        # Cluster state may be stale if the game ended mid-cycle, so refresh it.
        self._cluster_state = self._cluster_tracker.update(
            self, self.step_count, self.CLUSTER_RADIUS
        )
        final_obs = obs_raw_text(self, self.step_count)
        self.episode_log.append({
            "type": "final_state",
            "step": self.step_count,
            "outcome": outcome,
            "battlefield": final_obs.splitlines(),
        })

        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d__%H_%M_%S")

        Path("./logs/").mkdir(parents=True, exist_ok=True)

        log_path = Path(f"./logs/{formatted_now}__{self.name}__{self.game_info.map_name}__log.jsonl")
        with open(log_path, "w", encoding="utf-8") as f:
            # Count only llm_call entries for the summary (excludes the final_state snapshot).
            llm_calls = sum(1 for e in self.episode_log if e.get("type") == "llm_call")
            summary = {
                "type": "summary",
                "outcome": outcome,
                "total_steps": self.step_count,
                "total_llm_calls": llm_calls,
                "total_llm_failures": self.llm_failures,
                "config": {
                    "MODEL_NAME": self.MODEL_NAME,
                    "K_STEPS": self.K_STEPS,
                    "MAX_STEPS": self.MAX_STEPS,
                    "FALLBACK_DIRECTIVE": self.FALLBACK_DIRECTIVE,
                },
            }
            f.write(json.dumps(summary, indent=4) + "\n")
            for entry in self.episode_log:
                f.write(json.dumps(entry, indent=4) + "\n")

        self._episode_log_written = True
        console.print_log_saved(str(log_path.resolve()))


    def _build_history_section(self, length: int) -> str:
        """
        Build a compact history string from the most recent LLM call log entries.
        Each entry shows the step, unit counts at observation time, directive
        chosen, and the model's reasoning — enough for the LLM to spot trends
        without the token cost of full battlefield dumps.

        Format example:
            RECENT HISTORY (last 3 decisions):
            [Step  60] YOUR 8 | ENEMY 6  →  FOCUS_FIRE  "targeting weakest unit"
            [Step  90] YOUR 6 | ENEMY 5  →  FOCUS_FIRE  "continuing pressure"
            [Step 120] YOUR 4 | ENEMY 5  →  HOLD_POSITION  "taking heavy losses"
        """
        entries = self.episode_log[-length:]
        lines = [f"RECENT HISTORY (last {len(entries)} decision{'s' if len(entries) != 1 else ''}):"]
        for e in entries:
            # Extract unit counts from stored battlefield lines.
            your_count = 0
            enemy_count = 0
            in_your = in_enemy = False
            for line in (e.get("battlefield") or []):
                if line == "YOUR FORCES:":
                    in_your, in_enemy = True, False
                elif line == "ENEMY FORCES:":
                    in_your, in_enemy = False, True
                elif line.startswith("YOUR UNITS("):
                    your_count = int(line.split("(")[1].split(")")[0])
                elif line.startswith("ENEMY UNITS("):
                    enemy_count = int(line.split("(")[1].split(")")[0])
                elif in_your and "]:" in line and line.strip().startswith("GROUP "):
                    try:
                        your_count += int(line.split("]:")[1].strip().split("u")[0])
                    except (IndexError, ValueError):
                        pass
                elif in_enemy and "]:" in line and line.strip().startswith("CLUSTER "):
                    try:
                        enemy_count += int(line.split("]:")[1].strip().split("u")[0])
                    except (IndexError, ValueError):
                        pass
            your_str  = str(your_count)  if your_count  else "?"
            enemy_str = str(enemy_count) if enemy_count else "?"

            # Build compact directive summary — handles both new list format and legacy single-directive.
            raw_directives = e.get("directives")
            if raw_directives is not None:
                cmd_parts = []
                for d in raw_directives:
                    name = d.get("directive", "?")
                    units = d.get("units")
                    units_str = f'[{",".join(str(u) for u in units)}]' if units else ""
                    tx, ty = d.get("target_x"), d.get("target_y")
                    coord_str = f"→({tx:.0f},{ty:.0f})" if tx is not None and ty is not None else ""
                    reasoning = d.get("reasoning") or ""
                    r_str = f' "{reasoning}"' if reasoning else ""
                    cmd_parts.append(f"{name}{units_str}{coord_str}{r_str}")
                cmd_str = "  |  ".join(cmd_parts) if cmd_parts else "(no commands)"
            else:
                # Legacy log entry
                tx, ty = e.get("target_x"), e.get("target_y")
                cmd_str = f"{e.get('directive', '?')}@({tx}, {ty})"

            lines.append(
                f"[Step {e['step']:>4}] YOUR UNITS {your_str},  ENEMY UNITS {enemy_str},  {cmd_str}"
            )
        return "\n".join(lines)


    def _schedule_llm_call(self, step: int, battlefield: str):
        """
        Schedule a single in-flight LLM call if none is already running.
        Prevents overlapping calls and keeps the game loop non-blocking.
        """
        if self._pending_llm_task and not self._pending_llm_task.done():
            return

        self._pending_llm_task = asyncio.create_task(
            self._run_llm_call(step, battlefield)
        )


    async def _check_llm_task(self):
        """
        If an LLM task has finished, update the cached directive and emit a log entry.
        """
        if not self._pending_llm_task or not self._pending_llm_task.done():
            return

        result = await self._pending_llm_task
        self._pending_llm_task = None
        self._apply_llm_result(result)


    def _apply_llm_result(self, result: dict):
        """
        Parse the LLM response into a list of directives, execute each one immediately,
        and emit a log entry.  Directives are issued once — not repeated each step.
        """
        raw       = result["raw"]
        llm_error = result["error"]

        if llm_error:
            # The LLM call itself failed (network error, timeout, etc.) — no commands issued.
            self.llm_failures += 1
            self.episode_log.append({
                "type": "llm_call",
                "step": result["step"],
                "battlefield": result["battlefield"].splitlines() if result["battlefield"] else [],
                "directives": [],
                "raw": raw,
                "llm_latency_ms": result["latency_ms"],
                "llm_error": llm_error,
                "fallback_used": True,
            })
            console.warn(f"LLM call failed at step {result['step']}: {llm_error}")
            return

        directives = normalize_directives(
            raw,
            allowed=get_directive_registry().keys(),
            fallback=self.FALLBACK_DIRECTIVE,
        )

        any_fallback = any(d.fallback_used for d in directives)
        if any_fallback:
            self.llm_failures += 1

        # Execute each directive exactly once.
        for directive in directives:
            execute_directive(self, directive, fallback=self.FALLBACK_DIRECTIVE)

        self.episode_log.append({
            "type": "llm_call",
            "step": result["step"],
            "battlefield": result["battlefield"].splitlines() if result["battlefield"] else [],
            "directives": [
                {
                    "directive":     d.name,
                    "units":         d.units,
                    "target_x":      d.target_x,
                    "target_y":      d.target_y,
                    "target_unit":   d.target_unit,
                    "reasoning":     d.reasoning,
                    "fallback_used": d.fallback_used,
                    "error":         d.error,
                }
                for d in directives
            ],
            "raw": raw,
            "llm_latency_ms": result["latency_ms"],
            "llm_error": None,
            "fallback_used": any_fallback,
        })

        console.print_directives(
            step=result["step"],
            directives=directives,
            friendly=len(self.units),
            enemy=len(self.enemy_units.visible),
            latency_ms=result["latency_ms"],
        )


    async def _run_llm_call(self, step: int, battlefield: str) -> dict:
        """
        Execute a single LLM call, returning raw output plus metadata.
        No timeout is applied — the call always runs to completion so that
        slow models cannot cascade into repeated failures. The scheduling
        guard in _schedule_llm_call prevents overlapping calls.
        """
        if self.SHOW_LLM_PROMPT:
            console.print_llm_prompt(step, self.MODEL_NAME, battlefield)

        start = time.perf_counter()
        error = None
        raw = None
        try:
            raw = await self.get_new_directive_async(battlefield, step=step)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        latency_ms = int((time.perf_counter() - start) * 1000)

        return {
            "step": step,
            "battlefield": battlefield,
            "raw": raw,
            "error": error,
            "latency_ms": latency_ms,
        }
