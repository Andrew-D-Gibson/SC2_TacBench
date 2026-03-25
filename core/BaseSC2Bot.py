from obs_raw_text import obs_raw_text
from execute_directive import execute_directive, get_directive_registry
from directive import Directive, normalize_directive
from settings import get_settings
from map_loader import load_map_scenario

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

        # Initialise to fallback directive until the first LLM call completes.
        self.current_directive = Directive(name=self.FALLBACK_DIRECTIVE, reasoning="startup default")


    def _apply_settings(self):
        """
        Load and apply configuration from .env / environment variables.
        """
        settings = get_settings()
        self.K_STEPS = settings.k_steps
        self.MAX_STEPS = settings.max_steps
        self.FALLBACK_DIRECTIVE = settings.fallback_directive
        self.MODEL_NAME = settings.model_name


    # --- Overridable hooks (subclasses should focus here) ---

    def get_new_directive(self, current_battlefield_obs: str):
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


    async def get_new_directive_async(self, current_battlefield_obs: str):
        """
        Async directive generator. Override this in subclasses that call an async LLM.
        """
        return self.get_new_directive(current_battlefield_obs)


    # --- BotAI lifecycle methods ---

    async def on_start(self):
        """
        Called once at the beginning of a game.
        Loads the map scenario file matching the current map name.
        """
        print(f"\n[TacBench] Game started with bot {self.name}")
        print(f"[TacBench] Map: {self.game_info.map_name}")
        print(f"[TacBench] My race: {self.race}")
        print(f"[TacBench] Start location: {self.start_location}\n")

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
            print("[TacBench] Step limit reached - ending episode.")
            self._write_episode_log("TIMEOUT")
            await self.client.leave()
            return

        # Update map scenario state, then check win/loss.
        if self._map_scenario:
            self._map_scenario.on_step(self)

            if self._map_scenario.check_win(self):
                print("[TacBench] Win condition met - ending episode.")
                self._write_episode_log("WIN")
                await self.client.leave()
                return

            if self._map_scenario.check_loss(self):
                print("[TacBench] Loss condition met - ending episode.")
                self._write_episode_log("LOSS")
                await self.client.leave()
                return

        # Check whether a prior LLM task finished and update cached directive.
        await self._check_llm_task()

        if self.step_count % self.K_STEPS == 0:
            battlefield = obs_raw_text(self, self.step_count)
            if self._map_scenario and self._map_scenario.briefing:
                battlefield = self._map_scenario.briefing + "\n\n" + battlefield
            cfg = get_settings()
            if cfg.show_history and self.episode_log:
                battlefield += "\n\n" + self._build_history_section(cfg.history_length)

            self._schedule_llm_call(self.step_count, battlefield)

        # Every step: execute current cached directive.
        await execute_directive(self, self.current_directive, fallback=self.FALLBACK_DIRECTIVE)


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

        print(f"\n[TacBench] Game over - {outcome}")
        print(f"[TacBench] Total steps: {self.step_count}")
        print(f"[TacBench] Total LLM calls: {len(self.episode_log)}")
        self._write_episode_log(outcome)


    # --- Logging and LLM scheduling helpers ---

    def _write_episode_log(self, outcome: str):
        """
        Persist a JSONL log for this run. Idempotent — safe to call from
        both on_step (win/loss detection) and on_end (engine callback).
        """
        if self._episode_log_written:
            return

        now = datetime.now()
        formatted_now = now.strftime("%Y-%m-%d__%H_%M_%S")

        Path("./logs/").mkdir(parents=True, exist_ok=True)

        log_path = Path(f"./logs/{formatted_now}__{self.name}__{self.game_info.map_name}__log.jsonl")
        with open(log_path, "w", encoding="utf-8") as f:
            summary = {
                "type": "summary",
                "outcome": outcome,
                "total_steps": self.step_count,
                "total_llm_calls": len(self.episode_log),
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
        print(f"[TacBench] Log saved to: {log_path.resolve()}")


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
            # Extract unit counts from stored battlefield lines where available.
            your_count = "?"
            enemy_count = "?"
            for line in (e.get("battlefield") or []):
                if line.startswith("YOUR UNITS("):
                    your_count = line.split("(")[1].split(")")[0]
                elif line.startswith("ENEMY UNITS("):
                    enemy_count = line.split("(")[1].split(")")[0]
            reasoning = e.get("reasoning") or ""
            reasoning_str = f'  "{reasoning}"' if reasoning else ""
            lines.append(
                f"[Step {e['step']:>4}] YOUR UNITS {your_count} | ENEMY UNITS {enemy_count}"
                f"  →  {e['directive']}{reasoning_str}"
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
        Update the cached directive and emit a log entry from a completed LLM result dict.
        If the LLM call itself failed (timeout / exception), skip normalization and
        use the fallback directive directly with the real error message attached.
        """
        if result["error"]:
            # LLM call failed before returning output — skip normalization.
            directive = Directive(
                name=self.FALLBACK_DIRECTIVE,
                raw=result["raw"],
                error=result["error"],
                fallback_used=True,
            )
        else:
            directive = normalize_directive(
                result["raw"],
                allowed=get_directive_registry().keys(),
                fallback=self.FALLBACK_DIRECTIVE,
            )

        self.current_directive = directive

        if directive.fallback_used:
            self.llm_failures += 1

        self.episode_log.append({
            "type": "llm_call",
            "step": result["step"],
            "battlefield": result["battlefield"].splitlines() if result["battlefield"] else [],
            "directive": directive.name,
            "reasoning": directive.reasoning,
            "target_x": directive.target_x,
            "target_y": directive.target_y,
            "raw": result["raw"],
            "llm_latency_ms": result["latency_ms"],
            "llm_error": directive.error,
            "fallback_used": directive.fallback_used,
        })

        print(f"[Step {result['step']:>5}] Directive: {directive.name}")
        if directive.reasoning:
            print(f"           Reasoning: {directive.reasoning}")
        if directive.error:
            print(f"           LLM error: {directive.error}")
        print(f"           Friendly: {len(self.units)} | Enemy visible: {len(self.enemy_units.visible)}\n")


    async def _run_llm_call(self, step: int, battlefield: str) -> dict:
        """
        Execute a single LLM call, returning raw output plus metadata.
        No timeout is applied — the call always runs to completion so that
        slow models cannot cascade into repeated failures. The scheduling
        guard in _schedule_llm_call prevents overlapping calls.
        """
        start = time.perf_counter()
        error = None
        raw = None
        try:
            raw = await self.get_new_directive_async(battlefield)
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
