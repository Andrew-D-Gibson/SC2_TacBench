from sc2.ids.unit_typeid import UnitTypeId

from maps import BaseMapScenario

_CC_TYPES = frozenset([
    UnitTypeId.COMMANDCENTER,
    UnitTypeId.ORBITALCOMMAND,
    UnitTypeId.PLANETARYFORTRESS,
])

_BRIEFING = """\
OBJECTIVE: Defend your Command Center from enemy attack waves.
You win when all enemy units are defeated.
You lose if your Command Center is destroyed.\
"""


class MapScenario(BaseMapScenario):
    """
    tacbench_02 — defend your Command Center from attack waves.

    Win:  All enemy units are defeated.
    Loss: Your Command Center is destroyed.
    """

    briefing = _BRIEFING
    settings_overrides = {}

    def __init__(self):
        self._seen_enemies = False   # True once any enemy unit is spotted
        self._had_own_cc   = False   # True once bot's CC is confirmed present

    def on_step(self, bot) -> None:
        if bot.enemy_units:
            self._seen_enemies = True
        if bot.structures.of_type(_CC_TYPES):
            self._had_own_cc = True

    def check_win(self, bot) -> bool:
        """Win when all enemy units have been destroyed."""
        return self._seen_enemies and not bot.enemy_units

    def check_loss(self, bot) -> bool:
        """Lose when the bot's Command Center is destroyed."""
        return self._had_own_cc and not bot.structures.of_type(_CC_TYPES)
