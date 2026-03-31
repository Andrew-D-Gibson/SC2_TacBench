from BaseSC2Bot import BaseSC2Bot
from ClaudeBot import ClaudeBot
from FairlibBot import FairlibBot
from OllamaBot import OllamaBot
from settings import get_settings
import console

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


def _resolve_enum(value: str, enum_cls, field_name: str):
    """
    Look up a named member on an enum class.
    Raises a clear ValueError (instead of a cryptic AttributeError) if the
    name isn't found, listing all valid options.
    """
    try:
        return getattr(enum_cls, value)
    except AttributeError:
        valid = [m.name for m in enum_cls if m.name not in ("Norace", "Random")]
        raise ValueError(
            f"Invalid {field_name} '{value}' in settings. "
            f"Valid options: {', '.join(valid)}"
        )


if __name__ == "__main__":
    console.init()
    settings = get_settings()

    player_race         = _resolve_enum(settings.player_race,       Race,       "player_race")
    opponent_race       = _resolve_enum(settings.opponent_race,      Race,       "opponent_race")
    opponent_difficulty = _resolve_enum(settings.opponent_difficulty, Difficulty, "opponent_difficulty")

    bot_type = settings.bot_type.lower()
    if bot_type == "claude":
        bot = ClaudeBot()
    elif bot_type == "fairlib":
        bot = FairlibBot()
    else:
        bot = OllamaBot()

    console.print_startup_banner(bot.name, settings.model_name)

    if isinstance(bot, OllamaBot):
        asyncio.run(bot.warmup())

    try:
        run_game(
            maps.get(settings.map),
            [
                Bot(player_race, bot),
                Computer(opponent_race, opponent_difficulty),
            ],
            realtime=True,
        )
    except ValueError as e:
        if _SC2_LOOP_OVERFLOW in str(e) or "out of range" in str(e).lower():
            console.warn("Game ended (loop counter overflow — normal for custom maps).")
        else:
            raise
