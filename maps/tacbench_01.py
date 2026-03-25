from sc2.ids.unit_typeid import UnitTypeId

from maps import BaseMapScenario

_CC_TYPES = frozenset([
    UnitTypeId.COMMANDCENTER,
    UnitTypeId.ORBITALCOMMAND,
    UnitTypeId.PLANETARYFORTRESS,
])

_BRIEFING = """\
OBJECTIVE: Locate and destroy the enemy Command Center.
You win when the enemy Command Center is destroyed.
You lose if all your combat units are eliminated.\
"""


class MapScenario(BaseMapScenario):
    """
    tacbench_01 — head-to-head assault.

    Win:  the enemy Command Center (or upgraded variant) has been seen and
          is no longer visible (i.e. destroyed). Tracks sighting state so
          the condition is immune to fog-of-war false positives.
    Loss: bot army count drops to zero.
    """

    briefing = _BRIEFING
    settings_overrides = {}

    def __init__(self):
        self._seen_enemy_cc = False

    def on_step(self, bot) -> None:
        if not self._seen_enemy_cc and bot.enemy_structures.of_type(_CC_TYPES):
            self._seen_enemy_cc = True
            print("[TacBench] Enemy Command Center located.")

    def check_win(self, bot) -> bool:
        return self._seen_enemy_cc and not bot.enemy_structures.of_type(_CC_TYPES)

    def check_loss(self, bot) -> bool:
        return bot.army_count == 0
