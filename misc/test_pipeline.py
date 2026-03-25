# test_pipeline.py
# Validates:
#   1. python-sc2 bot runs a full game loop
#   2. Raw unit observations are accessible (same data PySC2 would give you)
#   3. A fake "LLM directive" can be injected and acted on mid-game
#   4. Win/loss signal is received at episode end
#
# Run with:
#   python test_pipeline.py

import asyncio
import random
import json
from enum import Enum
from pathlib import Path


from sc2 import maps
from sc2.bot_ai import BotAI
from sc2.data import Difficulty, Race, Result
from sc2.ids.unit_typeid import UnitTypeId
from sc2.main import run_game
from sc2.player import Bot, Computer



MAX_STEPS = 5000


DIRECTIVES = [
    "ADVANCE_AND_ATTACK",
    "FOCUS_FIRE",
    "RETREAT",
    "HOLD_POSITION",
    "SPREAD"
]


# ── Fake LLM: returns a random directive every K steps ──────────────────────
def fake_llm_call(battlefield_text: str) -> dict:
    """
    Stand-in for the real Ollama call.
    Returns a valid directive dict without hitting the LLM.
    Swap this out for your real LLMCommander.query() later.
    """
    directive = DIRECTIVES[0] #random.choice(DIRECTIVES) 
    return {
        "reasoning": f"[FAKE LLM] Chose {directive} based on battlefield state.",
        "directive": directive,
        "target":    None,
        "priority":  "objective",
    }


# ── Battlefield Summarizer (minimal version) ────────────────────────────────
UNIT_NAMES = {
    48:  "Marine",
    51:  "Marauder",
    54:  "Medivac",
    45:  "SCV",
    105: "Zergling",
    107: "Roach",
    110: "Hydralisk",
    9:   "Baneling",
}

def obs_to_text(bot: BotAI, step: int) -> str:
    """
    Convert python-sc2 bot state to a human-readable battlefield summary.
    This is the observation layer your real LLM will consume.
    """
    friendly = bot.units  # your units
    enemy    = bot.enemy_units

    def fmt_units(units, label):
        if not units:
            return f"{label}: none visible"
        lines = []
        for u in units[:8]:  # cap at 8 to keep prompt size sane
            name   = UNIT_NAMES.get(u.type_id.value, f"unit_{u.type_id.value}")
            hp_pct = int(100 * u.health / max(u.health_max, 1))
            lines.append(f"{name} HP:{hp_pct}% @ ({u.position.x:.0f},{u.position.y:.0f})")
        if len(units) > 8:
            lines.append(f"... and {len(units)-8} more")
        return f"{label} ({len(units)} units): " + "; ".join(lines)

    minerals = bot.minerals
    supply   = f"{bot.supply_used}/{bot.supply_cap}"

    return (
        f"=== BATTLEFIELD REPORT [Step {step}] ===\n"
        f"{fmt_units(friendly, 'FRIENDLY')}\n"
        f"{fmt_units(enemy,    'ENEMY (visible)')}\n"
        f"RESOURCES: {minerals} minerals | Supply: {supply}\n"
        f"========================================="
    )


# ── Scripted Executor ────────────────────────────────────────────────────────
async def execute_directive(bot: BotAI, directive: str, target=None):
    """
    Translate a named directive into python-sc2 actions.
    This runs every step using the cached last directive.
    """
    army = bot.units.of_type([
        UnitTypeId.MARINE,
        UnitTypeId.MARAUDER,
        UnitTypeId.ZERGLING,   # in case you test as Zerg
    ])

    if not army:
        return

    enemies = bot.enemy_units.visible

    if directive == "ADVANCE_AND_ATTACK":
        if enemies:
            target_pos = enemies.center
            for unit in army:
                unit.attack(target_pos)
        elif bot.enemy_structures:
            for unit in army:
                unit.attack(bot.enemy_structures.random.position)

    elif directive == "FOCUS_FIRE":
        if enemies:
            weakest = min(enemies, key=lambda u: u.health)
            for unit in army:
                unit.attack(weakest)

    elif directive == "RETREAT":
        rally = bot.start_location
        for unit in army:
            unit.move(rally)

    elif directive == "HOLD_POSITION":
        if enemies:
            for unit in army:
                closest = enemies.closest_to(unit)
                if unit.distance_to(closest) < 7:
                    unit.attack(closest)

    elif directive == "SPREAD":
        center = army.center
        for i, unit in enumerate(army):
            offset_x = (i % 3 - 1) * 3.0
            offset_y = (i // 3 - 1) * 3.0
            unit.move(center + (offset_x, offset_y))


# ── Main Bot ─────────────────────────────────────────────────────────────────
class TacBenchTestBot(BotAI):

    K_STEPS = 30  # how often to call the LLM

    def __init__(self):
        super().__init__()
        self.step_count       = 0
        self.current_directive = "HOLD_POSITION"
        self.current_reasoning = ""
        self.episode_log      = []   # list of dicts, one per LLM call
        self.last_outcome     = None


    async def on_start(self):
        print("\n[TacBench] Game started.")
        print(f"[TacBench] Map: {self.game_info.map_name}")
        print(f"[TacBench] My race: {self.race}")
        print(f"[TacBench] Start location: {self.start_location}\n")


    async def on_step(self, iteration: int):
        self.step_count += 1

        if self.step_count >= MAX_STEPS:
            print("[TacBench] Step limit reached — ending episode.")
            await self.client.leave()  # cleanly triggers on_end
            return

        # ── Every K steps: summarize + call (fake) LLM ──────────────────────
        if self.step_count % self.K_STEPS == 0:
            battlefield = obs_to_text(self, self.step_count)
            result      = fake_llm_call(battlefield)

            self.current_directive = result["directive"]
            self.current_reasoning = result["reasoning"]

            # Log this call
            self.episode_log.append({
                "step":       self.step_count,
                "battlefield": battlefield,
                "directive":  self.current_directive,
                "reasoning":  self.current_reasoning,
            })

            # Print to console so you can watch it work
            print(f"[Step {self.step_count:>5}] Directive: {self.current_directive}")
            print(f"           Reasoning: {self.current_reasoning}")
            print(f"           Friendly: {len(self.units)} | Enemy visible: {len(self.enemy_units.visible)}\n")

        # ── Every step: execute current cached directive ─────────────────────
        await execute_directive(self, self.current_directive)


    async def on_end(self, game_result: Result):
        outcome = "WIN" if game_result == Result.Victory else \
                  "LOSS" if game_result == Result.Defeat else \
                  "TIE"

        self.last_outcome = outcome
        print(f"\n[TacBench] Game over — {outcome}")
        print(f"[TacBench] Total steps: {self.step_count}")
        print(f"[TacBench] Total LLM calls: {len(self.episode_log)}")
        self._write_episode_log(outcome)

        # Print directive distribution
        from collections import Counter
        dist = Counter(e["directive"] for e in self.episode_log)
        print(f"[TacBench] Directive distribution: {dict(dist)}")


    def _write_episode_log(self, outcome: str):
        log_path = Path("test_episode_log.jsonl")
        with open(log_path, "w") as f:
            summary = {
                "type": "summary",
                "outcome": outcome,
                "total_steps": self.step_count,
                "total_llm_calls": len(self.episode_log),
            }
            f.write(json.dumps(summary) + "\n")
            for entry in self.episode_log:
                f.write(json.dumps(entry) + "\n")
        print(f"[TacBench] Log saved to: {log_path.resolve()}")



# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    bot = TacBenchTestBot()
    # NOTE: realtime=False is more robust for custom maps that end via triggers.
    # realtime=True can request an out-of-range game_loop after the game ends.

    try:
        run_game(
            maps.get("tacbench_01"),
            [
                Bot(Race.Terran, bot),
                Computer(Race.Zerg, Difficulty.Easy),  # built-in SC2 AI as opponent
            ],
            realtime=False,   # step as fast as possible
        )
    except ValueError as e:
        if "4294967296" in str(e) or "out of range" in str(e).lower():
            print("[TacBench] Game ended (loop counter overflow — normal for custom maps).")
        else:
            raise

