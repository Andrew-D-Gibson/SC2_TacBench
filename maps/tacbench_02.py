from sc2.ids.unit_typeid import UnitTypeId

from maps import BaseMapScenario

_BRIEFING = """\
OBJECTIVE: Defend all of your Supply Depots from enemy attack waves.
You win when all enemy units are defeated.
You lose immediately if ANY Supply Depot is destroyed — losing even one ends the mission.

CRITICAL RULES:
- If you see a "!! STRUCTURE ALERTS !!" section, a Supply Depot is actively taking damage RIGHT NOW.
  You MUST redirect your nearest units to defend it immediately — this is your highest priority.
- HP changes labeled "(lost N% HP since last report)" mean that unit or structure took damage
  between this report and the previous one.  Negative HP on a Supply Depot is an emergency.\
"""


class MapScenario(BaseMapScenario):
    """
    tacbench_02 — defend all Supply Depots from attack waves.

    Win:  All enemy units are defeated.
    Loss: Any Supply Depot is destroyed.
    """

    briefing = _BRIEFING
    settings_overrides = {}

    def __init__(self):
        self._seen_enemies = False   # True once any enemy unit is spotted
        self._depot_count  = None    # Supply Depot count once first observed

    def on_step(self, bot) -> None:
        if bot.enemy_units:
            self._seen_enemies = True
        depots = bot.structures.of_type(UnitTypeId.SUPPLYDEPOT)
        if self._depot_count is None and depots:
            self._depot_count = len(depots)

    def check_win(self, bot) -> bool:
        """Win when all enemy units have been destroyed."""
        return self._seen_enemies and not bot.enemy_units

    def check_loss(self, bot) -> bool:
        """Lose when any Supply Depot is destroyed."""
        if self._depot_count is None:
            return False
        return len(bot.structures.of_type(UnitTypeId.SUPPLYDEPOT)) < self._depot_count
