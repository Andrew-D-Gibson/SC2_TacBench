"""
warnings.py — Trajectory-based engagement warnings for TacBench observations.

Computes intercept times using relative-motion quadratic equations and
generates plain-English alerts for:

  1. Friendly clusters on course to enter a specific enemy cluster's weapon range
     (tagged HEADS-UP / CAUTION / WARNING based on force ratio).
  2. Enemy clusters on course to reach weapon range of a friendly cluster or a
     friendly structure (tagged ALERT).

The intercept time is found by solving:
    |dp + dv·t|² = R²
    |dv|²·t² + 2(dp·dv)·t + (|dp|² − R²) = 0
where dp = posA − posB, dv = velA − velB, and R is the distance threshold
(typically enemy_weapon_range × CONTACT_RANGE_FACTOR).

This correctly handles both objects in motion simultaneously.
"""

from __future__ import annotations

import math

from core.tactics.clustering import UnitCluster, ratio_label, _CONTACT_RANGE_FACTOR


# ── Intercept math ─────────────────────────────────────────────────────────────

def _time_to_range(
    ax: float, ay: float, vax: float, vay: float,
    bx: float, by: float, vbx: float, vby: float,
    threshold: float,
) -> float | None:
    """
    Return the number of game steps until the distance between A and B first
    equals `threshold`, accounting for the independent motion of both objects.

    Returns the smallest positive t, or None if:
      - they are already within threshold (current-state threat levels cover this)
      - there is no relative motion between them
      - they will never converge to that distance (discriminant < 0)
      - both roots are in the past
    """
    dpx = ax - bx
    dpy = ay - by
    if math.sqrt(dpx ** 2 + dpy ** 2) <= threshold:
        return None  # already within range; handled by existing threat labels

    dvx = vax - vbx
    dvy = vay - vby
    a = dvx ** 2 + dvy ** 2
    if a < 1e-9:
        return None  # no relative motion — distance won't change

    b = 2.0 * (dpx * dvx + dpy * dvy)
    c = dpx ** 2 + dpy ** 2 - threshold ** 2

    disc = b ** 2 - 4.0 * a * c
    if disc < 0:
        return None  # trajectories never converge

    sqrt_d = math.sqrt(disc)
    candidates = [
        t for t in ((-b - sqrt_d) / (2.0 * a), (-b + sqrt_d) / (2.0 * a))
        if t > 0.0
    ]
    return min(candidates) if candidates else None


# ── Warning generators ─────────────────────────────────────────────────────────

def _friendly_engagement_warnings(
    all_friendly: list[UnitCluster],
    all_enemy:    list[UnitCluster],
    k_steps: int,
    lookahead_steps: int,
) -> list[tuple[float, str]]:
    """
    For each friendly cluster, check whether its trajectory (combined with the
    enemy's trajectory) will bring it within the enemy's weapon range within
    `lookahead_steps` game steps.  Classify the intercept by force ratio.

    Returns a list of (calls_until_intercept, warning_text) tuples.
    """
    entries: list[tuple[float, str]] = []
    for fc in sorted(all_friendly, key=lambda c: c.label):
        for ec in sorted(all_enemy, key=lambda c: c.label):
            e_range = ec.max_air_range if fc.is_air else ec.max_ground_range
            e_str   = ec.anti_air_strength if fc.is_air else ec.anti_ground_strength
            if e_range <= 0 or e_str <= 0:
                continue  # this enemy cluster cannot threaten this unit type

            t = _time_to_range(
                fc.center.x, fc.center.y, fc.velocity_x, fc.velocity_y,
                ec.center.x, ec.center.y, ec.velocity_x, ec.velocity_y,
                threshold=e_range * _CONTACT_RANGE_FACTOR,
            )
            if t is None or t > lookahead_steps:
                continue

            calls = t / max(k_steps, 1)
            rlat  = ratio_label(fc.count, e_str)

            if rlat in ("ADVANTAGED", "EVEN"):
                tag    = "[HEADS-UP]"
                advice = "you appear to have the advantage"
            elif rlat == "DISADVANTAGED":
                tag    = "[CAUTION]"
                advice = "enemy is stronger — weigh your options carefully"
            else:  # CRITICAL
                tag    = "[WARNING]"
                advice = "you are heavily outmatched — strongly consider retreating"

            entries.append((
                calls,
                f"{tag} Group {fc.label} will enter weapon range of Enemy Cluster {ec.label}"
                f" in ~{calls:.1f} calls [{rlat}] — {advice}.",
            ))
    return entries


def _enemy_approach_warnings(
    all_enemy:    list[UnitCluster],
    all_friendly: list[UnitCluster],
    bot,
    k_steps: int,
    lookahead_steps: int,
) -> list[tuple[float, str]]:
    """
    For each enemy cluster, check whether its trajectory will bring it within
    weapon range of a friendly cluster or a friendly structure within
    `lookahead_steps` steps.

    Returns a list of (calls_until_intercept, warning_text) tuples.
    """
    entries: list[tuple[float, str]] = []
    for ec in sorted(all_enemy, key=lambda c: c.label):
        # ── vs friendly clusters ──────────────────────────────────────────────
        for fc in sorted(all_friendly, key=lambda c: c.label):
            e_range = ec.max_air_range if fc.is_air else ec.max_ground_range
            if e_range <= 0:
                continue

            t = _time_to_range(
                ec.center.x, ec.center.y, ec.velocity_x, ec.velocity_y,
                fc.center.x, fc.center.y, fc.velocity_x, fc.velocity_y,
                threshold=e_range * _CONTACT_RANGE_FACTOR,
            )
            if t is None or t > lookahead_steps:
                continue

            calls = t / max(k_steps, 1)
            entries.append((
                calls,
                f"[ALERT] Enemy Cluster {ec.label} will reach weapon range of"
                f" your Group {fc.label} in ~{calls:.1f} calls.",
            ))

        # ── vs friendly structures ────────────────────────────────────────────
        gnd_range = ec.max_ground_range
        if gnd_range <= 0:
            continue  # enemy can't threaten ground structures

        threshold = gnd_range * _CONTACT_RANGE_FACTOR
        for s in sorted(getattr(bot, "structures", []), key=lambda s: s.tag):
            t = _time_to_range(
                ec.center.x, ec.center.y, ec.velocity_x, ec.velocity_y,
                s.position.x, s.position.y, 0.0, 0.0,
                threshold=threshold,
            )
            if t is None or t > lookahead_steps:
                continue

            calls = t / max(k_steps, 1)
            uid   = bot.get_unit_id(s)
            entries.append((
                calls,
                f"[ALERT] Enemy Cluster {ec.label} will reach weapon range of"
                f" your {s.name} #{uid} @({s.position.x},{s.position.y}) in ~{calls:.1f} calls.",
            ))

    return entries


# ── Public interface ───────────────────────────────────────────────────────────

def fmt_trajectory_warnings(
    friendly_ground: list[UnitCluster],
    friendly_air:    list[UnitCluster],
    enemy_ground:    list[UnitCluster],
    enemy_air:       list[UnitCluster],
    bot,
    k_steps: int,
    lookahead_calls: int = 8,
) -> str:
    """
    Generate a TRAJECTORY WARNINGS block for the LLM observation.

    Warnings are sorted by time-to-intercept (soonest first).  Anything
    occurring within the next LLM call is additionally tagged [TIME SENSITIVE].

    Returns an empty string if no intercepts are predicted within the window.

    Tag legend:
      [HEADS-UP]       friendly approaching enemy, ratio ADVANTAGED or EVEN
      [CAUTION]        friendly approaching enemy, ratio DISADVANTAGED
      [WARNING]        friendly approaching enemy, ratio CRITICAL
      [ALERT]          enemy approaching a friendly cluster or structure
      [TIME SENSITIVE] intercept within the next LLM call (~k_steps steps)
    """
    all_friendly    = friendly_ground + friendly_air
    all_enemy       = enemy_ground    + enemy_air
    lookahead_steps = lookahead_calls * k_steps

    entries: list[tuple[float, str]] = []
    entries.extend(_friendly_engagement_warnings(all_friendly, all_enemy, k_steps, lookahead_steps))
    entries.extend(_enemy_approach_warnings(all_enemy, all_friendly, bot, k_steps, lookahead_steps))

    if not entries:
        return ""

    entries.sort(key=lambda e: e[0])

    lines = ["TRAJECTORY WARNINGS:"]
    for calls, text in entries:
        prefix = "  [TIME SENSITIVE] " if calls < 1.0 else "  "
        lines.append(f"{prefix}{text}")

    return "\n".join(lines)
