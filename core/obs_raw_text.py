from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId

from settings import get_settings
from terrain_encoder import terrain_encoder

# Terrain is static — cache it after the first computation.
_terrain_cache: dict[str, str] = {}


def _fmt_unit_entry(u) -> str:
    hp_pct = int(100 * u.health / max(u.health_max, 1))
    return f"{u.name} {hp_pct}%@({u.position.x:.0f},{u.position.y:.0f})"


def _fmt_units(units, label: str, cap: int = 64) -> str:
    if not units:
        return f"{label}: none"
    entries = [_fmt_unit_entry(u) for u in units[:cap]]
    suffix = f" +{len(units) - cap} more" if len(units) > cap else ""
    return f"{label}({len(units)}): " + "; ".join(entries) + suffix


def _fmt_structures(structures, label: str) -> str:
    if not structures:
        return None
    entries = [_fmt_unit_entry(s) for s in structures]
    return f"{label}({len(structures)}): " + "; ".join(entries)


def _fmt_supply(bot: BotAI) -> str:
    return f"Supply: {bot.supply_used}/{bot.supply_cap} | Army: {bot.army_count}"


def _fmt_time(bot: BotAI, step: int) -> str:
    return f"Time: {bot.time_formatted} (step {step})"


def _fmt_terrain(bot: BotAI, downsample: int) -> str:
    key = (bot.game_info.map_name, downsample)
    if key not in _terrain_cache:
        gi = bot.game_info
        _terrain_cache[key] = terrain_encoder(
            gi.terrain_height.data_numpy,
            gi.pathing_grid.data_numpy,
            gi.placement_grid.data_numpy,
            downsample_factor=downsample,
        )
    return _terrain_cache[key]


def _fmt_locations(bot: BotAI) -> str:
    own = f"({bot.start_location.x:.0f},{bot.start_location.y:.0f})"
    enemy_locs = "; ".join(
        f"({p.x:.0f},{p.y:.0f})" for p in bot.enemy_start_locations
    )
    return f"My base: {own} | Known enemy bases: {enemy_locs}"


# --- Public interface ---

def obs_raw_text(bot: BotAI, step: int) -> str:
    """
    Convert python-sc2 bot state into a compact text battlefield summary
    for LLM consumption. Toggle sections via TACBENCH_SHOW_* in .env.
    """
    cfg = get_settings()
    sections = []

    if cfg.show_game_time:
        sections.append(_fmt_time(bot, step))
    if cfg.show_supply:
        sections.append(_fmt_supply(bot))
    if cfg.show_locations:
        sections.append(_fmt_locations(bot))
    if cfg.show_terrain:
        sections.append(_fmt_terrain(bot, cfg.terrain_downsample))
    if cfg.show_your_units:
        sections.append(_fmt_units(bot.units, "YOUR UNITS"))
    if cfg.show_your_structures:
        result = _fmt_structures(bot.structures, "YOUR STRUCTURES")
        if result:
            sections.append(result)
    if cfg.show_enemy_units:
        sections.append(_fmt_units(bot.enemy_units, "ENEMY UNITS"))
    if cfg.show_enemy_structures:
        result = _fmt_structures(bot.enemy_structures, "ENEMY STRUCTURES")
        if result:
            sections.append(result)

    return "\n".join(sections)
