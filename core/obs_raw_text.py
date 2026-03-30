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


# Stim pack buff IDs (BuffId.STIMPACK=27, BuffId.STIMPACKMARAUDER=28)
_STIM_BUFF_IDS = frozenset({27, 28})


def _fmt_unit_full(unit, bot) -> str:
    """
    Format a mobile unit for the combined cluster view.
    Includes HP, energy (if any), special status tags, and position.
    Status tags: STIMMED, CLOAKED, BURROWED, CONSTRUCTING (SCV only).
    """
    uid    = bot.get_unit_id(unit)
    hp_pct = int(100 * unit.health / max(unit.health_max, 1))
    delta  = bot.get_hp_delta(unit)
    parts  = [f"{unit.name} #{uid} {hp_pct}%HP{_hp_delta_str(delta)}"]

    if unit.energy_max > 0:
        parts.append(f"{unit.energy:.0f}en")

    tags = []
    if unit.is_cloaked:
        tags.append("CLOAKED")
    if unit.is_burrowed:
        tags.append("BURROWED")
    if _STIM_BUFF_IDS & unit.buffs:
        tags.append("STIMMED")
    if unit.name == "SCV" and unit.orders and "BUILD" in str(unit.orders[0].ability).upper():
        tags.append("CONSTRUCTING")
    if tags:
        parts.append(f"[{', '.join(tags)}]")

    parts.append(f"@ ({unit.position.x:.0f},{unit.position.y:.0f})")
    return " ".join(parts)


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
    return f"{label}({len(structures)}): " + "\n".join(entries)


def _fmt_structure_alerts(bot) -> str | None:
    """
    Report own structures that have taken damage since the last observation.
    Advisory only — the LLM should weigh this against other objectives.

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
                f"  {s.name} #{uid} took {lost}% damage "
                f"(now at {current}%) @ ({s.position.x:.0f},{s.position.y:.0f})"
            )
    if not alerts:
        return None
    return "STRUCTURE DAMAGE REPORT:\n" + "\n".join(alerts)


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


def _fmt_matchup_lines(fc: UnitCluster, all_enemy: list[UnitCluster], k_steps: int) -> list[str]:
    """Matchup lines for one friendly cluster against all visible enemy clusters."""
    lines = []
    if not all_enemy:
        lines.append("    vs ENEMY: none visible")
        return lines
    for ec in sorted(all_enemy, key=lambda e: e.distance_to(fc)):
        dist    = ec.distance_to(fc)
        threat  = compute_threat(fc, ec, dist)
        vel_lbl = velocity_toward_label(ec, fc, k_steps)
        if fc.is_air:
            e_range, e_str, threat_tag = ec.max_air_range, ec.anti_air_strength, "air-threat"
        else:
            e_range, e_str, threat_tag = ec.max_ground_range, ec.anti_ground_strength, "gnd-threat"
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
    return lines


def _fmt_forces(
    friendly_ground: list[UnitCluster],
    friendly_air:    list[UnitCluster],
    enemy_ground:    list[UnitCluster],
    enemy_air:       list[UnitCluster],
    bot,
    k_steps: int = 30,
) -> str:
    """
    Combined cluster + matchup + unit listing.

    For each friendly cluster: cluster header, per-enemy matchup lines, then
    each individual unit on its own line with HP, energy, status tags, position.
    Enemy clusters follow with their own units listed underneath.

    Example:
      YOUR FORCES:
        GROUP A [GND]: 3u @ (32,45) | 240/300HP [80%] | moving NE (3.8 tiles/call)
          vs CLUSTER 1 [GND] 4u @ (55,60) dist 8.2 | gnd-threat: CONTACT | ...
          Marine #1 95%HP @ (31,44)
          Marine #2 80%HP (lost 15% HP since last report) [STIMMED] @ (33,46)
          Medivac #3 100%HP 125en @ (32,45)
      ENEMY FORCES:
        CLUSTER 1 [GND]: 4u @ (55,60) | 360/400HP [90%] | approaching (4.0 tiles/call)
          Marine #5 90%HP @ (54,59)
    """
    all_friendly = friendly_ground + friendly_air
    all_enemy    = enemy_ground    + enemy_air

    if not all_friendly and not all_enemy:
        return "FORCES: no combatants visible"

    lines = []

    lines.append("YOUR FORCES:")
    if not all_friendly:
        lines.append("  none")
    else:
        for fc in sorted(all_friendly, key=lambda c: c.label):
            hp_str  = f"{int(fc.hp_current)}/{int(fc.hp_max)}"
            vel_str = _fmt_cluster_velocity(fc, k_steps)
            lines.append(
                f"  GROUP {fc.label} [{_cluster_type_tag(fc)}]: {fc.count}u "
                f"@ ({fc.center.x:.0f},{fc.center.y:.0f}) "
                f"| {hp_str}HP [{fc.hp_pct}%] | {vel_str}"
            )
            for ml in _fmt_matchup_lines(fc, all_enemy, k_steps):
                lines.append(ml)
            for u in fc.units:
                lines.append(f"    {_fmt_unit_full(u, bot)}")

    lines.append("ENEMY FORCES:")
    if not all_enemy:
        lines.append("  none visible")
    else:
        for ec in sorted(all_enemy, key=lambda c: c.label):
            hp_str  = f"{int(ec.hp_current)}/{int(ec.hp_max)}"
            vel_str = _fmt_cluster_velocity(ec, k_steps)
            lines.append(
                f"  CLUSTER {ec.label} [{_cluster_type_tag(ec)}]: {ec.count}u "
                f"@ ({ec.center.x:.0f},{ec.center.y:.0f}) "
                f"| {hp_str}HP [{ec.hp_pct}%] | {vel_str}"
            )
            for u in ec.units:
                lines.append(f"    {_fmt_unit_full(u, bot)}")

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
        # Combined view: cluster headers + matchups + individual units
        sections.append(_fmt_forces(fg, fa, eg, ea, bot, k_steps=cfg.k_steps))
    else:
        # Fallback flat lists when tactical overview is disabled
        if cfg.show_your_units:
            sections.append(_fmt_units(bot.units, "YOUR UNITS", bot))
        if cfg.show_enemy_units:
            sections.append(_fmt_units(bot.enemy_units, "ENEMY UNITS", bot))

    if cfg.show_your_structures:
        result = _fmt_structures(bot.structures, "YOUR STRUCTURES", bot)
        if result:
            sections.append(result)
    if cfg.show_enemy_structures:
        result = _fmt_structures(bot.enemy_structures, "ENEMY STRUCTURES", bot)
        if result:
            sections.append(result)

    return "\n".join(sections)
