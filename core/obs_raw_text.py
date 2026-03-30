import math

from sc2.bot_ai import BotAI

from settings import get_settings
from terrain_encoder import build_terrain_grid, format_terrain_grid
from clustering import UnitCluster, compute_threat, ratio_label, velocity_toward_label

# Cache the downsampled terrain grid (expensive; static per map).
# Stores (ds_grid, orig_h, orig_w) keyed by (map_name, downsample_factor).
_terrain_cache: dict[tuple, tuple] = {}


# ── Unit entry formatters ─────────────────────────────────────────────────────

def _hp_delta_str(delta: int | None) -> str:
    """Human-readable HP delta label, intentionally verbose so the LLM acts on it."""
    if delta is None or delta == 0:
        return ""
    if delta < 0:
        return f" (lost {abs(delta)}% HP since last report)"
    return f" (healed {delta}% HP since last report)"


def _fmt_unit(unit, bot) -> str:
    """Format any unit or structure: persistent ID, current HP%, HP delta since last call."""
    uid    = bot.get_unit_id(unit)
    hp_pct = int(100 * unit.health / max(unit.health_max, 1))
    delta  = bot.get_hp_delta(unit)
    return f"{unit.name} #{uid} {hp_pct}%HP{_hp_delta_str(delta)} @ ({unit.position.x:.0f},{unit.position.y:.0f})"


def _fmt_units(units, label: str, bot, cap: int = 64) -> str:
    if not units:
        return f"{label}: none"
    entries = [_fmt_unit(u, bot) for u in units[:cap]]
    suffix  = f" +{len(units) - cap} more" if len(units) > cap else ""
    return f"{label}({len(units)}): " + "; ".join(entries) + suffix


def _fmt_structures(structures, label: str, bot) -> str | None:
    if not structures:
        return None
    entries = [_fmt_unit(s, bot) for s in structures]
    return f"{label}({len(structures)}): " + "; ".join(entries)


def _fmt_structure_alerts(bot) -> str | None:
    """
    Emit urgent warnings for own structures taking damage since the last report.

    IMPORTANT: call this BEFORE any formatter that calls get_hp_delta, so that
    _unit_hp_history still holds the previous-report snapshot.  The delta shown
    here is therefore genuinely "since the LLM last saw this structure."
    """
    alerts = []
    for s in sorted(bot.structures, key=lambda s: s.tag):
        prev = bot._unit_hp_history.get(s.tag)
        if prev is None:
            continue  # first observation — no baseline yet
        current = int(100 * s.health / max(s.health_max, 1))
        lost = prev - current
        if lost > 0:
            uid = bot.get_unit_id(s)
            alerts.append(
                f"  !! {s.name} #{uid} is UNDER ATTACK — lost {lost}% HP "
                f"(now at {current}%) @ ({s.position.x:.0f},{s.position.y:.0f})"
            )
    if not alerts:
        return None
    return (
        "!! STRUCTURE ALERTS — IMMEDIATE ACTION REQUIRED !!\n"
        + "\n".join(alerts)
        + "\n!! Send units to defend these structures NOW !!"
    )


# ── Simple stats ──────────────────────────────────────────────────────────────

def _fmt_supply(bot: BotAI) -> str:
    return f"Supply: {bot.supply_used}/{bot.supply_cap} | Army: {bot.army_count}"


def _fmt_time(bot: BotAI, step: int) -> str:
    cfg = get_settings()
    steps_left = cfg.max_steps - step
    return f"Time: {bot.time_formatted} (step {step} | {steps_left} steps remaining)"


def _fmt_locations(bot: BotAI) -> str:
    own        = f"({bot.start_location.x:.0f},{bot.start_location.y:.0f})"
    enemy_locs = "; ".join(f"({p.x:.0f},{p.y:.0f})" for p in bot.enemy_start_locations)
    return f"My base: {own} | Known enemy bases: {enemy_locs}"


# ── Terrain ───────────────────────────────────────────────────────────────────

def _get_ds_grid(bot: BotAI, downsample: int) -> tuple:
    key = (bot.game_info.map_name, downsample)
    if key not in _terrain_cache:
        gi    = bot.game_info
        ds    = build_terrain_grid(
            gi.terrain_height.data_numpy,
            gi.pathing_grid.data_numpy,
            gi.placement_grid.data_numpy,
            downsample,
        )
        _terrain_cache[key] = (ds, len(gi.terrain_height.data_numpy), len(gi.terrain_height.data_numpy[0]))
    return _terrain_cache[key]


def _fmt_terrain(
    bot: BotAI,
    downsample: int,
    friendly: list[UnitCluster],
    enemy: list[UnitCluster],
) -> str:
    ds, orig_h, orig_w = _get_ds_grid(bot, downsample)
    overlays = [(c.label, c.center.x, c.center.y) for c in friendly + enemy]
    return format_terrain_grid(
        ds, downsample,
        flip_y=True,
        overlays=overlays or None,
        orig_h=orig_h,
        orig_w=orig_w,
    )


# ── Tactical overview ─────────────────────────────────────────────────────────

def _cluster_type_tag(c: UnitCluster) -> str:
    return "AIR" if c.is_air else "GND"


def _fmt_cluster_velocity(c: UnitCluster, k_steps: int) -> str:
    if c.is_stationary():
        return "stationary"
    speed = c.speed() * k_steps
    # atan2(vy, vx): 0°=east, 90°=north — matches SC2 coordinate orientation
    deg  = math.degrees(math.atan2(c.velocity_y, c.velocity_x)) % 360
    dirs = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"]
    idx  = int((deg + 22.5) / 45) % 8
    return f"moving {dirs[idx]} ({speed:.1f} tiles/call)"


def _fmt_tactical_overview(
    friendly_ground: list[UnitCluster],
    friendly_air:   list[UnitCluster],
    enemy_ground:   list[UnitCluster],
    enemy_air:      list[UnitCluster],
    k_steps: int = 30,
) -> str:
    """
    Per-group matchup table.

    For each friendly cluster (ground and air), shows every visible enemy
    cluster with:
      • distance
      • dynamic threat label (SAFE / DISTANT / NEARBY / THREAT / CONTACT),
        scaled by the enemy's actual weapon range and strength vs this cluster type
      • enemy range and strength (relevant to this cluster's type)
      • local force ratio (friendly count vs enemy relevant strength)
      • enemy velocity relative to this cluster

    Example line:
      GROUP A [GND]: 8u @ (32,45) | 680/800HP [85%] | moving NE (3.8 tiles/call)
        vs CLUSTER 1 [GND] 10u @ (55,60) dist 8.2 | gnd-threat: CONTACT | range 6 str 7.2 | ratio 8/7.2 [EVEN] | approaching (4.0 tiles/call)
        vs CLUSTER 2 [AIR]  3u @ (40,35) dist 25.3 | gnd-threat: SAFE (no anti-ground)
    """
    all_friendly = friendly_ground + friendly_air
    all_enemy    = enemy_ground    + enemy_air

    if not all_friendly and not all_enemy:
        return "TACTICAL OVERVIEW: no combatants visible"

    lines = ["TACTICAL OVERVIEW:"]

    if not all_friendly:
        lines.append("  YOUR FORCES: none")
    else:
        for fc in sorted(all_friendly, key=lambda c: c.label):
            hp_str  = f"{int(fc.hp_current)}/{int(fc.hp_max)}"
            vel_str = _fmt_cluster_velocity(fc, k_steps)
            lines.append(
                f"  GROUP {fc.label} [{_cluster_type_tag(fc)}]: {fc.count}u "
                f"@ ({fc.center.x:.0f},{fc.center.y:.0f}) "
                f"| {hp_str}HP [{fc.hp_pct}%] | {vel_str}"
            )

            if not all_enemy:
                lines.append("    vs ENEMY: none visible")
                continue

            for ec in sorted(all_enemy, key=lambda e: e.distance_to(fc)):
                dist     = ec.distance_to(fc)
                threat   = compute_threat(fc, ec, dist)
                vel_lbl  = velocity_toward_label(ec, fc, k_steps)

                # Relevant range + strength for this matchup
                if fc.is_air:
                    e_range = ec.max_air_range
                    e_str   = ec.anti_air_strength
                    threat_tag = "air-threat"
                else:
                    e_range = ec.max_ground_range
                    e_str   = ec.anti_ground_strength
                    threat_tag = "gnd-threat"

                rlat = ratio_label(fc.count, e_str)

                if threat == "SAFE":
                    capability = f"no anti-{'air' if fc.is_air else 'ground'}"
                    lines.append(
                        f"    vs CLUSTER {ec.label} [{_cluster_type_tag(ec)}] "
                        f"{ec.count}u @ ({ec.center.x:.0f},{ec.center.y:.0f}) "
                        f"dist {dist:.1f} | {threat_tag}: SAFE ({capability}) | {vel_lbl}"
                    )
                else:
                    lines.append(
                        f"    vs CLUSTER {ec.label} [{_cluster_type_tag(ec)}] "
                        f"{ec.count}u @ ({ec.center.x:.0f},{ec.center.y:.0f}) "
                        f"dist {dist:.1f} | {threat_tag}: {threat} "
                        f"| range {e_range:.0f} str {e_str:.1f} "
                        f"| ratio {fc.count}/{e_str:.1f} [{rlat}] "
                        f"| {vel_lbl}"
                    )

    return "\n".join(lines)


# ── Public interface ──────────────────────────────────────────────────────────

def obs_raw_text(bot, step: int) -> str:
    """
    Convert bot state into a compact text battlefield summary for LLM consumption.
    Toggle sections via TACBENCH_SHOW_* in .env.
    """
    cfg      = get_settings()
    sections = []

    if cfg.show_game_time:
        sections.append(_fmt_time(bot, step))
    if cfg.show_supply:
        sections.append(_fmt_supply(bot))
    if cfg.show_locations:
        sections.append(_fmt_locations(bot))

    # Use pre-computed cluster state from BaseSC2Bot's tracker.
    fg, fa, eg, ea = getattr(bot, "_cluster_state", ([], [], [], []))
    all_friendly = fg + fa
    all_enemy    = eg + ea

    if cfg.show_terrain:
        sections.append(_fmt_terrain(bot, cfg.terrain_downsample, all_friendly, all_enemy))

    if cfg.show_tactical_overview:
        sections.append(_fmt_tactical_overview(fg, fa, eg, ea, k_steps=cfg.k_steps))

    # Structure alerts must come before any formatter that calls get_hp_delta,
    # so the HP history still reflects the previous report when we diff against it.
    structure_alerts = _fmt_structure_alerts(bot)
    if structure_alerts:
        sections.append(structure_alerts)

    if cfg.show_your_units:
        sections.append(_fmt_units(bot.units, "YOUR UNITS", bot))
    if cfg.show_your_structures:
        result = _fmt_structures(bot.structures, "YOUR STRUCTURES", bot)
        if result:
            sections.append(result)
    if cfg.show_enemy_units:
        sections.append(_fmt_units(bot.enemy_units, "ENEMY UNITS", bot))
    if cfg.show_enemy_structures:
        result = _fmt_structures(bot.enemy_structures, "ENEMY STRUCTURES", bot)
        if result:
            sections.append(result)

    return "\n".join(sections)
