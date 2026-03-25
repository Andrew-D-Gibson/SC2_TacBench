from BaseSC2Bot import BaseSC2Bot
from FairlibBot import FairlibBot
from OllamaBot import OllamaBot
from settings import get_settings

from sc2 import maps
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer

import asyncio


# This value (2^32) is the unsigned 32-bit integer overflow point for SC2's
# internal game_loop counter. When a custom map ends via a trigger while
# running in realtime=True mode, python-sc2 can receive an out-of-range loop
# counter value and raise a ValueError. Catching it here is intentional and
# normal — it does not indicate a bug in your code.
_SC2_LOOP_OVERFLOW = "4294967296"


if __name__ == "__main__":
    settings = get_settings()

    # Validate race and difficulty settings early so misconfigured .env files
    # produce a clear error message instead of a cryptic AttributeError.
    try:
        player_race = getattr(Race, settings.player_race)
    except AttributeError:
        valid = [r.name for r in Race if r.name not in ("Norace", "Random")]
        raise ValueError(
            f"Invalid player_race '{settings.player_race}' in settings. "
            f"Valid options: {', '.join(valid)}"
        )

    try:
        opponent_race = getattr(Race, settings.opponent_race)
    except AttributeError:
        valid = [r.name for r in Race if r.name not in ("Norace", "Random")]
        raise ValueError(
            f"Invalid opponent_race '{settings.opponent_race}' in settings. "
            f"Valid options: {', '.join(valid)}"
        )

    try:
        opponent_difficulty = getattr(Difficulty, settings.opponent_difficulty)
    except AttributeError:
        valid = [d.name for d in Difficulty]
        raise ValueError(
            f"Invalid opponent_difficulty '{settings.opponent_difficulty}' in settings. "
            f"Valid options: {', '.join(valid)}"
        )

    bot = OllamaBot()
    result = asyncio.run(bot.warmup())

    try:
        run_game(
            maps.get(settings.map),
            [
                Bot(player_race, bot),
                Computer(opponent_race, opponent_difficulty),
            ],
            realtime=settings.realtime,
        )
    except ValueError as e:
        if _SC2_LOOP_OVERFLOW in str(e) or "out of range" in str(e).lower():
            print("[TacBench] Game ended (loop counter overflow - normal for custom maps).")
        else:
            raise
