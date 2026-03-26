from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

from directive import Directive
import console


# --- Directive handler functions ---
# Each handler takes (bot, army, enemies, directive).
# 'army'     — filtered Units collection (Marines + Marauders)
# 'enemies'  — currently visible enemy units
# 'directive'— the full Directive object, including optional target_x/target_y

def move(bot: BotAI, army, enemies, directive: Directive):
    """Move all army units to the target position (target_x, target_y required)."""
    if directive.target_x is None or directive.target_y is None:
        console.warn("MOVE directive is missing target_x/target_y — no action taken.")
        return
    target = Point2((directive.target_x, directive.target_y))
    for unit in army:
        unit.move(target)


def attack(bot: BotAI, army, enemies, directive: Directive):
    """Attack-move all army units toward the target position (target_x, target_y required)."""
    if directive.target_x is None or directive.target_y is None:
        console.warn("ATTACK directive is missing target_x/target_y — no action taken.")
        return
    target = Point2((directive.target_x, directive.target_y))
    for unit in army:
        unit.attack(target)


def focus_fire(bot: BotAI, army, enemies, directive: Directive):
    """Focus all army units on the lowest-health visible enemy."""
    if enemies:
        weakest = min(enemies, key=lambda u: u.health)
        for unit in army:
            unit.attack(weakest)


def hold_position(bot: BotAI, army, enemies, directive: Directive):
    """Attack nearby enemies (within range 7) but don't advance."""
    if enemies:
        for unit in army:
            closest = enemies.closest_to(unit)
            if unit.distance_to(closest) < 7:
                unit.attack(closest)


def spread(bot: BotAI, army, enemies, directive: Directive):
    """Spread army units into a grid formation around their center."""
    center = army.center
    for i, unit in enumerate(army):
        offset_x = (i % 3 - 1) * 3.0
        offset_y = (i // 3 - 1) * 3.0
        unit.move(center + (offset_x, offset_y))


DIRECTIVE_REGISTRY = {
    "MOVE":         move,
    "ATTACK":       attack,
    "FOCUS_FIRE":   focus_fire,
    "HOLD_POSITION": hold_position,
    "SPREAD":       spread,
}


def get_directive_registry() -> dict:
    """
    Return a copy of the current directive registry.
    """
    return dict(DIRECTIVE_REGISTRY)


def register_directive(name: str, handler) -> None:
    """
    Register (or override) a directive handler by name.

    This lets you add new tactical behaviors from outside this file without
    editing the registry directly — useful when building custom bot subclasses.

    The handler must have this signature:
        handler(bot: BotAI, army, enemies, directive: Directive) -> None

    Example — adding a "KITE" directive that retreats while attacking:

        from execute_directive import register_directive
        from sc2.bot_ai import BotAI
        from directive import Directive

        def kite(bot: BotAI, army, enemies, directive: Directive):
            for unit in army:
                if enemies:
                    closest = enemies.closest_to(unit)
                    flee_pos = unit.position.towards(closest.position, -5)
                    unit.move(flee_pos)

        register_directive("KITE", kite)

    After calling register_directive, "KITE" is a valid directive name that
    the LLM can return and the bot will execute each step.
    """
    DIRECTIVE_REGISTRY[name] = handler


async def execute_directive(bot: BotAI, directive, fallback: str = "HOLD_POSITION"):
    """
    Translate a named directive into python-sc2 unit actions.
    Runs every game step using the most recently cached directive.
    """
    army = bot.units.of_type([
        UnitTypeId.MARINE,
        UnitTypeId.MARAUDER,
    ])

    if not army:
        return

    enemies = bot.enemy_units.visible

    # Accept either a Directive object or a raw string name.
    if isinstance(directive, Directive):
        directive_name = directive.name
    else:
        directive_name = str(directive)
        directive = Directive(name=directive_name)

    handler = DIRECTIVE_REGISTRY.get(directive_name)
    if handler is not None:
        handler(bot, army, enemies, directive)
        return

    # Directive not found — warn and attempt the fallback.
    fallback_handler = DIRECTIVE_REGISTRY.get(fallback)
    if fallback_handler is not None:
        console.warn(f"directive '{directive_name}' not in registry — falling back to '{fallback}'.")
        fallback_handler(bot, army, enemies, directive)
    else:
        console.warn(f"directive '{directive_name}' not in registry and fallback '{fallback}' is also not registered. No action taken.")
