from sc2.bot_ai import BotAI

from settings import get_settings
from terrain_encoder import build_terrain_grid, format_terrain_grid
from clustering import build_tactical_clusters, range_label, ratio_label, UnitCluster

# Cache the downsampled terrain grid (expensive to compute; static per map).
# Stores (ds_grid, orig_h, orig_w) keyed by (map_name, downsample_factor).
_terrain_cache: dict[tuple, tuple] = {}


# ── Unit / structure formatters ───────────────────────────────────────────────

def _fmt_unit_entry(u) -> str:
    hp_pct = int(100 * u.health / max(u.health_max, 1))
    return f"{u.name} {hp_pct}%@({u.position.x:.0f},{u.position.y:.0f})"


def _fmt_units(units, label: str, cap: int = 64) -> str:
    if not units:
        return f"{label}: none"
    entries = [_fmt_unit_entry(u) for u in units[:cap]]
    suffix  = f" +{len(units) - cap} more" if len(units) > cap else ""
    return f"{label}({len(units)}): " + "; ".join(entries) + suffix


def _fmt_structures(structures, label: str) -> str:
    if not structures:
        return None
    entries = [_fmt_unit_entry(s) for s in structures]
    return f"{label}({len(structures)}): " + "; ".join(entries)


# ── Simple stats ──────────────────────────────────────────────────────────────

def _fmt_supply(bot: BotAI) -> str:
    return f"Supply: {bot.supply_used}/{bot.supply_cap} | Army: {bot.army_count}"


def _fmt_time(bot: BotAI, step: int) -> str:
    return f"Time: {bot.time_formatted} (step {step})"


def _fmt_locations(bot: BotAI) -> str:
    own        = f"({bot.start_location.x:.0f},{bot.start_location.y:.0f})"
    enemy_locs = "; ".join(f"({p.x:.0f},{p.y:.0f})" for p in bot.enemy_start_locations)
    return f"My base: {own} | Known enemy bases: {enemy_locs}"


# ── Terrain ───────────────────────────────────────────────────────────────────

def _get_ds_grid(bot: BotAI, downsample: int) -> tuple:
    """Return cached (ds_grid, orig_h, orig_w), building it on first call."""
    key = (bot.game_info.map_name, downsample)
    if key not in _terrain_cache:
        gi    = bot.game_info
        h_arr = gi.terrain_height.data_numpy
        p_arr = gi.pathing_grid.data_numpy
        pl_arr = gi.placement_grid.data_numpy
        ds    = build_terrain_grid(h_arr, p_arr, pl_arr, downsample)
        _terrain_cache[key] = (ds, len(h_arr), len(h_arr[0]))
    return _terrain_cache[key]


def _fmt_terrain(
    bot: BotAI,
    downsample: int,
    friendly: list[UnitCluster],
    enemy: list[UnitCluster],
) -> str:
    """
    Format the terrain map with cluster labels overlaid.
    Friendly groups are labelled A, B, C…; enemy clusters 1, 2, 3…
    """
    ds, orig_h, orig_w = _get_ds_grid(bot, downsample)

    # Build overlay list: (label_char, game_x, game_y)
    # Friendly written first so enemy labels win on collision (higher urgency).
    overlays: list[tuple[str, float, float]] = []
    for fc in friendly:
        overlays.append((fc.label, fc.center.x, fc.center.y))
    for ec in enemy:
        overlays.append((ec.label, ec.center.x, ec.center.y))

    return format_terrain_grid(
        ds,
        downsample,
        flip_y=True,
        overlays=overlays if overlays else None,
        orig_h=orig_h,
        orig_w=orig_w,
    )


# ── Tactical overview ─────────────────────────────────────────────────────────

def _fmt_tactical_overview(
    friendly: list[UnitCluster],
    enemy: list[UnitCluster],
) -> str:
    """
    Per-group matchup table.  Clusters are pre-computed by the caller so the
    same objects are shared with the terrain overlay.

    Example:
        TACTICAL OVERVIEW:
          YOUR GROUP A: 8 units @ (32, 45) | 680/800 HP [85%]
            vs ENEMY CLUSTER 1 (10 units @ (55, 60)) — dist  8.2 [THREAT]  | ratio 0.80 [EVEN]
            vs ENEMY CLUSTER 2 ( 3 units @ (40, 35)) — dist 25.3 [NEARBY]  | ratio 2.67 [ADVANTAGED]
    """
    if not friendly and not enemy:
        return "TACTICAL OVERVIEW: no combatants visible"

    lines = ["TACTICAL OVERVIEW:"]

    if not friendly:
        lines.append("  YOUR FORCES: none")
    else:
        for fc in friendly:
            hp_str = f"{int(fc.hp_current)}/{int(fc.hp_max)}"
            lines.append(
                f"  YOUR GROUP {fc.label}: {fc.count} units "
                f"@ ({fc.center.x:.0f}, {fc.center.y:.0f}) "
                f"| {hp_str} HP [{fc.hp_pct}%]"
            )
            if not enemy:
                lines.append("    vs ENEMY: none visible")
            else:
                for ec in sorted(enemy, key=lambda e: e.distance_to(fc)):
                    dist = ec.distance_to(fc)
                    rlbl = range_label(dist)
                    rlat = ratio_label(fc.count, ec.count)
                    lines.append(
                        f"    vs ENEMY CLUSTER {ec.label} "
                        f"({ec.count} units @ ({ec.center.x:.0f}, {ec.center.y:.0f})) "
                        f"- dist {dist:5.1f} [{rlbl:<7}] "
                        f"| local ratio {fc.count}/{ec.count} [{rlat}]"
                    )

    return "\n".join(lines)


# ── Public interface ──────────────────────────────────────────────────────────

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

    # Compute clusters once — shared by both terrain overlay and tactical overview.
    friendly, enemy = [], []
    if cfg.show_tactical_overview or cfg.show_terrain:
        friendly, enemy = build_tactical_clusters(bot, radius=cfg.cluster_radius)

    if cfg.show_terrain:
        sections.append(_fmt_terrain(bot, cfg.terrain_downsample, friendly, enemy))

    if cfg.show_tactical_overview:
        sections.append(_fmt_tactical_overview(friendly, enemy))

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
