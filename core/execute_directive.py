from sc2.bot_ai import BotAI
from sc2.position import Point2

from directive import Directive
import console

# ── Tactical constants ────────────────────────────────────────────────────────

_HOLD_POSITION_RANGE = 7     # tiles within which to attack an enemy while holding position
_RETREAT_DISTANCE    = 15.0  # tiles to move away from the enemy centroid when retreating


# --- Army resolver ---


def _resolve_army(bot: BotAI, all_army, directive: Directive):
    """
    Resolve directive.units (cluster labels like "A","B" or unit IDs like 1,5,12)
    to a filtered friendly army collection.  Falls back to all_army if:
      - directive.units is None / empty
      - no requested labels/IDs match any current unit
    """
    if not directive.units:
        return all_army

    fg, fa, eg, ea = getattr(bot, "_cluster_state", ([], [], [], []))
    cluster_map = {c.label: c for c in fg + fa}

    # Build reverse ID map: short_id -> tag (from all known friendly units)
    id_to_tag = {v: k for k, v in bot._unit_id_map.items()}

    wanted_tags: set = set()
    for item in directive.units:
        if isinstance(item, str):
            cluster = cluster_map.get(item)
            if cluster:
                for u in cluster.units:
                    wanted_tags.add(u.tag)
        elif isinstance(item, int):
            tag = id_to_tag.get(item)
            if tag is not None:
                wanted_tags.add(tag)

    if not wanted_tags:
        return all_army

    filtered = all_army.filter(lambda u: u.tag in wanted_tags)
    return filtered if filtered else all_army


# --- Directive handler functions ---
# Each handler takes (bot, army, enemies, directive).
# 'army'     — filtered Units collection, already resolved
#              from directive.units if provided
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
    """Focus army units on a specific enemy unit by target_unit ID. No-op if target_unit is missing or not found."""
    if not enemies:
        return
    if directive.target_unit is None:
        console.warn("FOCUS_FIRE directive is missing target_unit — no action taken.")
        return
    id_to_tag = {v: k for k, v in bot._unit_id_map.items()}
    tag = id_to_tag.get(directive.target_unit)
    target = enemies.find_by_tag(tag) if tag is not None else None
    if target is None:
        console.warn(f"FOCUS_FIRE target_unit {directive.target_unit} not found in visible enemies — no action taken.")
        return
    for unit in army:
        unit.attack(target)


def hold_position(bot: BotAI, army, enemies, directive: Directive):
    """Attack nearby enemies (within _HOLD_POSITION_RANGE tiles) but don't advance."""
    if enemies:
        for unit in army:
            closest = enemies.closest_to(unit)
            if unit.distance_to(closest) < _HOLD_POSITION_RANGE:
                unit.attack(closest)


def spread(bot: BotAI, army, enemies, directive: Directive):
    """Spread army units into a grid formation around their center."""
    center = army.center
    for i, unit in enumerate(army):
        offset_x = (i % 3 - 1) * 3.0
        offset_y = (i // 3 - 1) * 3.0
        unit.move(center + (offset_x, offset_y))


def retreat(bot: BotAI, army, enemies, directive: Directive):
    """Move all army units directly away from the centroid of visible enemies."""
    if not enemies:
        return  # Nothing to retreat from — no-op.

    enemy_center = enemies.center
    army_center  = army.center

    dx = army_center.x - enemy_center.x
    dy = army_center.y - enemy_center.y
    dist = (dx ** 2 + dy ** 2) ** 0.5

    if dist < 0.001:  # Exactly overlapping — pick an arbitrary escape direction.
        dx, dy, dist = 1.0, 0.0, 1.0

    tx = army_center.x + (dx / dist) * _RETREAT_DISTANCE
    ty = army_center.y + (dy / dist) * _RETREAT_DISTANCE

    # Clamp to map bounds so units don't walk off the edge.
    map_w, map_h = bot.game_info.map_size
    tx = max(0.0, min(float(map_w), tx))
    ty = max(0.0, min(float(map_h), ty))

    target = Point2((tx, ty))
    for unit in army:
        unit.move(target)


DIRECTIVE_REGISTRY = {
    "MOVE":          move,
    "ATTACK":        attack,
    "FOCUS_FIRE":    focus_fire,
    "HOLD_POSITION": hold_position,
    "SPREAD":        spread,
    "RETREAT":       retreat,
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


def execute_directive(bot: BotAI, directive, fallback: str = "HOLD_POSITION") -> None:
    """
    Translate a named directive into python-sc2 unit actions.
    Called once when a new LLM response arrives — not repeated each game step.
    """
    all_army = bot.units
    if not all_army:
        return

    enemies = bot.enemy_units.visible

    if isinstance(directive, Directive):
        directive_name = directive.name
    else:
        directive_name = str(directive)
        directive = Directive(name=directive_name)

    army = _resolve_army(bot, all_army, directive)

    handler = DIRECTIVE_REGISTRY.get(directive_name)
    if handler is not None:
        handler(bot, army, enemies, directive)
        return

    fallback_handler = DIRECTIVE_REGISTRY.get(fallback)
    if fallback_handler is not None:
        console.warn(f"directive '{directive_name}' not in registry — falling back to '{fallback}'.")
        fallback_handler(bot, army, enemies, directive)
    else:
        console.warn(f"directive '{directive_name}' not in registry and fallback '{fallback}' also not registered.")
