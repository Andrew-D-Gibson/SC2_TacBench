"""
clustering.py — Spatial unit clustering for TacBench tactical observations.

Groups nearby units into clusters so the LLM can reason about *local* force
matchups rather than the full-army aggregate.

Ground and air units are always clustered separately, ensuring that e.g. a
ground cluster with no anti-air weapons never inflates the threat shown to an
air group.

Public API
----------
ClusterTracker          — stateful tracker; call update() every N game steps
ratio_label(f, e)       — ADVANTAGED / EVEN / DISADVANTAGED / CRITICAL
compute_threat(fc, ec, dist) — dynamic threat label based on range + strength
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List

from sc2.bot_ai import BotAI
from sc2.position import Point2

# ── Force ratio thresholds ─────────────────────────────────────────────────────

_RATIO_ADVANTAGED    = 1.2   # friendly-to-enemy ratio above which we are ADVANTAGED
_RATIO_EVEN          = 0.8   # ratio above which forces are considered EVEN
_RATIO_DISADVANTAGED = 0.5   # ratio above which forces are DISADVANTAGED (below → CRITICAL)

# ── Threat level distance multipliers (relative to enemy weapon range) ─────────

_CONTACT_RANGE_FACTOR = 1.5  # enemy weapon just about reaches us → CONTACT
_THREAT_RANGE_FACTOR  = 2.0  # closing fast → THREAT
_NEARBY_RANGE_FACTOR  = 3.0  # visible but not imminent → NEARBY

# Strength ratio thresholds for bumping threat level up or down one step
_HIGH_THREAT_STR_RATIO = 2.0  # enemy much stronger than our group → bump threat up
_LOW_THREAT_STR_RATIO  = 0.5  # enemy much weaker than our group  → bump threat down

# Ghost enemy positions older than this many steps are pruned
_GHOST_MAX_STEPS = 300

# ── Unit mineral/gas build costs (used for cluster cost metric) ────────────────

_UNIT_COSTS: dict[str, tuple[int, int]] = {
    # Terran
    "SCV": (50, 0), "Marine": (50, 0), "Reaper": (50, 50),
    "Marauder": (100, 25), "Ghost": (200, 100),
    "Hellion": (100, 0), "Hellbat": (100, 0),
    "WidowMine": (75, 25), "WidowMineBurrowed": (75, 25),
    "SiegeTank": (150, 125), "SiegeTankSieged": (150, 125),
    "Cyclone": (150, 100), "Thor": (300, 200), "ThorHighImpactMode": (300, 200),
    "Liberator": (150, 150), "LiberatorAG": (150, 150),
    "Viking": (150, 75), "VikingAssault": (150, 75), "VikingFighter": (150, 75),
    "Medivac": (100, 100), "Raven": (100, 200),
    "Banshee": (150, 100), "Battlecruiser": (400, 300),
    # Zerg
    "Drone": (50, 0), "Overlord": (100, 0), "Overseer": (50, 50),
    "Zergling": (25, 0), "Baneling": (50, 25),
    "Roach": (75, 25), "RoachBurrowed": (75, 25), "Ravager": (75, 75),
    "Hydralisk": (100, 50), "LurkerMP": (50, 100), "LurkerMPBurrowed": (50, 100),
    "Infestor": (100, 150), "InfestorBurrowed": (100, 150),
    "SwarmHostMP": (200, 100), "Ultralisk": (300, 200),
    "BroodLord": (150, 150), "Mutalisk": (100, 100),
    "Corruptor": (150, 100), "Viper": (100, 200), "Queen": (150, 0),
    # Protoss
    "Probe": (50, 0), "Zealot": (100, 0), "Stalker": (125, 50),
    "Sentry": (50, 100), "Adept": (100, 25),
    "HighTemplar": (50, 150), "DarkTemplar": (125, 125), "Archon": (100, 300),
    "Immortal": (275, 100), "Colossus": (300, 200), "Disruptor": (150, 150),
    "Phoenix": (150, 100), "VoidRay": (250, 150), "Oracle": (150, 150),
    "Carrier": (350, 250), "Tempest": (250, 175),
    "Observer": (25, 75), "WarpPrism": (200, 0), "Mothership": (400, 400),
}


# ── Force ratio labels ─────────────────────────────────────────────────────────

def ratio_label(friendly: float, enemy: float) -> str:
    """
    Map a friendly / enemy strength ratio to a qualitative label.
    Both arguments can be unit counts or HP-weighted strength values.
    """
    if enemy <= 0:
        return "ADVANTAGED"
    if friendly <= 0:
        return "CRITICAL"
    r = friendly / enemy
    if r >= _RATIO_ADVANTAGED:
        return "ADVANTAGED"
    if r >= _RATIO_EVEN:
        return "EVEN"
    if r >= _RATIO_DISADVANTAGED:
        return "DISADVANTAGED"
    return "CRITICAL"


# ── UnitCluster ────────────────────────────────────────────────────────────────

@dataclass
class UnitCluster:
    label: str = ""
    units: list = field(default_factory=list)
    center: Point2 = field(default_factory=lambda: Point2((0, 0)))
    count: int = 0
    hp_current: float = 0.0
    hp_max: float = 0.0
    is_air: bool = False

    # Enemy-side metrics (populated by _compute_cluster_metrics)
    max_ground_range: float = 0.0   # max range of ground weapons in this cluster
    max_air_range: float = 0.0      # max range of air weapons in this cluster
    anti_ground_strength: float = 0.0  # HP-weighted count of units that can attack ground
    anti_air_strength: float = 0.0    # HP-weighted count of units that can attack air

    # Common metrics (populated by _compute_common_metrics for all clusters)
    composition: str = ""           # e.g. "Marine x3, Marauder"
    total_cost_minerals: int = 0
    total_cost_gas: int = 0
    avg_attack_upgrade: float = 0.0
    avg_armor_upgrade: float = 0.0

    # Velocity — tiles per game step; set by ClusterTracker
    velocity_x: float = 0.0
    velocity_y: float = 0.0

    @property
    def hp_pct(self) -> int:
        return int(100 * self.hp_current / max(self.hp_max, 1))

    def distance_to(self, other: "UnitCluster") -> float:
        return self.center.distance_to(other.center)

    def speed(self) -> float:
        return math.sqrt(self.velocity_x ** 2 + self.velocity_y ** 2)

    def is_stationary(self, threshold: float = 0.05) -> bool:
        return self.speed() < threshold

    def projected_position(self, k_steps: int) -> Point2:
        """Estimated position after k_steps game steps at current velocity."""
        return Point2((
            self.center.x + self.velocity_x * k_steps,
            self.center.y + self.velocity_y * k_steps,
        ))


# ── Threat computation ─────────────────────────────────────────────────────────

def compute_threat(fc: UnitCluster, ec: UnitCluster, distance: float) -> str:
    """
    Dynamic threat label for a friendly cluster (fc) facing an enemy cluster (ec).

    Selects the enemy's relevant range and strength based on whether fc is air or
    ground.  Scales threat thresholds off the enemy's actual weapon range rather
    than fixed tile constants.  A strength modifier bumps the threat up or down
    by one level when the strength ratio is strongly asymmetric.

    Returns one of: SAFE | DISTANT | NEARBY | THREAT | CONTACT
    """
    if fc.is_air:
        enemy_range  = ec.max_air_range
        enemy_str    = ec.anti_air_strength
    else:
        enemy_range  = ec.max_ground_range
        enemy_str    = ec.anti_ground_strength

    if enemy_str <= 0.0 or enemy_range <= 0.0:
        return "SAFE"

    # Base level from distance (scaled by actual weapon range)
    if distance <= enemy_range * _CONTACT_RANGE_FACTOR:
        level = 4   # CONTACT — in range right now
    elif distance <= enemy_range * _THREAT_RANGE_FACTOR:
        level = 3   # THREAT  — will be in range very soon
    elif distance <= enemy_range * _NEARBY_RANGE_FACTOR:
        level = 2   # NEARBY  — visible, monitor
    else:
        level = 1   # DISTANT — far away

    # Strength modifier: enemy effective strength vs friendly unit count
    str_ratio = enemy_str / max(fc.count, 1)
    if str_ratio >= _HIGH_THREAT_STR_RATIO:
        level = min(4, level + 1)
    elif str_ratio <= _LOW_THREAT_STR_RATIO:
        level = max(1, level - 1)

    return {1: "DISTANT", 2: "NEARBY", 3: "THREAT", 4: "CONTACT"}[level]


def velocity_toward_label(
    ec: UnitCluster,
    fc: UnitCluster,
    k_steps: int = 30,
) -> str:
    """
    Describe enemy cluster movement relative to a specific friendly cluster.
    Speed is reported in tiles per LLM call (velocity_per_step * k_steps).
    """
    speed = ec.speed()
    if speed < 0.05:
        return "stationary"

    # Unit vector from enemy to friendly
    dx = fc.center.x - ec.center.x
    dy = fc.center.y - ec.center.y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist < 0.001:
        return "stationary"

    dot = (ec.velocity_x * dx + ec.velocity_y * dy) / (speed * dist)
    tiles_per_call = speed * k_steps

    if dot > 0.5:
        return f"approaching ({tiles_per_call:.1f} tiles/call)"
    if dot < -0.5:
        return f"retreating ({tiles_per_call:.1f} tiles/call)"
    return f"moving perpendicular ({tiles_per_call:.1f} tiles/call)"


# ── Core clustering algorithm ──────────────────────────────────────────────────

def cluster_units(units: list, radius: float = 12.0) -> list[UnitCluster]:
    """
    Greedy radius-based clustering — O(n²), fine for SC2 unit counts.

    1. Pop the first unassigned unit as a seed.
    2. Grow the cluster by pulling in any unassigned unit within `radius` of the
       current centroid, recomputing the centroid after each addition.
    3. Seal the cluster and repeat until no unassigned units remain.

    Returns clusters sorted largest-first (ties broken by x position).
    """
    remaining = list(units)
    clusters: list[UnitCluster] = []

    while remaining:
        members = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            cx = sum(u.position.x for u in members) / len(members)
            cy = sum(u.position.y for u in members) / len(members)
            still_out: list = []
            for u in remaining:
                dx = u.position.x - cx
                dy = u.position.y - cy
                if math.sqrt(dx * dx + dy * dy) <= radius:
                    members.append(u)
                    changed = True
                else:
                    still_out.append(u)
            remaining = still_out

        cx = sum(u.position.x for u in members) / len(members)
        cy = sum(u.position.y for u in members) / len(members)

        clusters.append(UnitCluster(
            units=members,
            center=Point2((cx, cy)),
            count=len(members),
            hp_current=sum(u.health     for u in members),
            hp_max    =sum(u.health_max for u in members),
        ))

    clusters.sort(key=lambda c: (-c.count, c.center.x))
    return clusters


# ── Cluster metric computation ─────────────────────────────────────────────────

def _compute_common_metrics(cluster: UnitCluster) -> None:
    """
    Populate composition, cost, and upgrade fields from the cluster's units.
    Called for ALL clusters (friendly and enemy).
    """
    comp = Counter(u.name for u in cluster.units)
    cluster.composition = ", ".join(
        f"{name} x{count}" if count > 1 else name
        for name, count in comp.most_common()
    )

    total_m = total_g = 0
    for u in cluster.units:
        m, g = _UNIT_COSTS.get(u.name, (0, 0))
        total_m += m
        total_g += g
    cluster.total_cost_minerals = total_m
    cluster.total_cost_gas      = total_g

    n = len(cluster.units)
    cluster.avg_attack_upgrade = sum(u.attack_upgrade_level for u in cluster.units) / n
    cluster.avg_armor_upgrade  = sum(u.armor_upgrade_level  for u in cluster.units) / n


def _compute_cluster_metrics(cluster: UnitCluster) -> None:
    """Populate range and strength fields from the cluster's unit composition."""
    gnd_ranges = [u.ground_range for u in cluster.units if u.can_attack_ground]
    air_ranges = [u.air_range    for u in cluster.units if u.can_attack_air]
    cluster.max_ground_range = max(gnd_ranges) if gnd_ranges else 0.0
    cluster.max_air_range    = max(air_ranges) if air_ranges else 0.0
    cluster.anti_ground_strength = sum(
        u.health / max(u.health_max, 1)
        for u in cluster.units if u.can_attack_ground
    )
    cluster.anti_air_strength = sum(
        u.health / max(u.health_max, 1)
        for u in cluster.units if u.can_attack_air
    )


# ── Velocity tracking snapshot ─────────────────────────────────────────────────

@dataclass
class _Snap:
    center: Point2
    step: int = 0


def _apply_velocities(
    clusters: list[UnitCluster],
    snaps: list[_Snap],
    elapsed_steps: int,
    match_max: float = 30.0,
) -> None:
    """
    Match each cluster to the nearest previous snapshot (within match_max tiles)
    and compute per-step velocity.  Unmatched clusters keep velocity (0, 0).
    """
    if not snaps:
        return
    for c in clusters:
        best = min(snaps, key=lambda s: s.center.distance_to(c.center))
        if best.center.distance_to(c.center) <= match_max:
            c.velocity_x = (c.center.x - best.center.x) / elapsed_steps
            c.velocity_y = (c.center.y - best.center.y) / elapsed_steps


def _find_unmatched_snaps(
    clusters: list[UnitCluster],
    snaps: list[_Snap],
    match_max: float = 30.0,
) -> list[_Snap]:
    """
    Return snaps from the previous observation that were not matched to any
    current cluster.  These represent enemy groups that have left vision.
    """
    if not snaps:
        return []
    if not clusters:
        return list(snaps)
    matched_ids: set[int] = set()
    for c in clusters:
        best = min(snaps, key=lambda s: s.center.distance_to(c.center))
        if best.center.distance_to(c.center) <= match_max:
            matched_ids.add(id(best))
    return [s for s in snaps if id(s) not in matched_ids]


# ── Raw cluster builder (internal) ────────────────────────────────────────────

def _build_clusters_raw(
    bot: BotAI,
    radius: float,
) -> tuple[list[UnitCluster], list[UnitCluster], list[UnitCluster], list[UnitCluster]]:
    """
    Split both sides into ground/air, cluster each independently, populate
    metrics for enemy clusters, and assign labels.

    Returns: (friendly_ground, friendly_air, enemy_ground, enemy_air)

    Labels:
      Friendly: A, B, C… across both ground and air, sorted by count desc
      Enemy:    1, 2, 3… across both ground and air, sorted by distance to
                nearest friendly cluster
    """
    all_friendly = list(bot.units)
    all_enemy    = list(bot.enemy_units.visible)

    fg_units = [u for u in all_friendly if not u.is_flying]
    fa_units = [u for u in all_friendly if     u.is_flying]
    eg_units = [u for u in all_enemy    if not u.is_flying]
    ea_units = [u for u in all_enemy    if     u.is_flying]

    fg = cluster_units(fg_units, radius)
    fa = cluster_units(fa_units, radius)
    eg = cluster_units(eg_units, radius)
    ea = cluster_units(ea_units, radius)

    for c in fa + ea:
        c.is_air = True

    # Common metrics (composition, cost, upgrades) for all clusters.
    for c in fg + fa + eg + ea:
        _compute_common_metrics(c)

    # Enemy-only metrics (weapon ranges and anti-ground/air strength).
    for c in eg + ea:
        _compute_cluster_metrics(c)

    # ── Label friendly: A, B, C… across ground + air by count ──────────────
    all_friendly_clusters = fg + fa
    all_friendly_clusters.sort(key=lambda c: (-c.count, c.center.x))
    for i, c in enumerate(all_friendly_clusters):
        c.label = chr(ord("A") + min(i, 25))  # cap at Z

    # ── Label enemy: 1, 2, 3… closest to any friendly first ────────────────
    all_enemy_clusters = eg + ea
    if all_friendly_clusters and all_enemy_clusters:
        def _min_dist(ec: UnitCluster) -> float:
            return min(ec.distance_to(fc) for fc in all_friendly_clusters)
        all_enemy_clusters.sort(key=_min_dist)
    for i, c in enumerate(all_enemy_clusters):
        c.label = str(i + 1)

    return fg, fa, eg, ea


# ── ClusterTracker — stateful, runs every N game steps ────────────────────────

class ClusterTracker:
    """
    Maintains cluster state and velocity estimates across game steps.

    Usage in BaseSC2Bot.on_step:
        if iteration % settings.cluster_track_interval == 0:
            self._cluster_state = self._cluster_tracker.update(self, self.step_count, radius)

    The four-tuple (friendly_ground, friendly_air, enemy_ground, enemy_air) is
    consumed by obs_raw_text via bot._cluster_state.

    ghost_enemy_positions: list[_Snap]
        Positions of enemy clusters that were visible in a prior observation but
        are no longer visible.  Each entry carries the step when it was last seen.
        Entries older than _GHOST_MAX_STEPS steps are automatically pruned.
    """

    def __init__(self) -> None:
        self._prev_fg: list[_Snap] = []
        self._prev_fa: list[_Snap] = []
        self._prev_eg: list[_Snap] = []
        self._prev_ea: list[_Snap] = []
        self._prev_step: int = 0
        self.ghost_enemy_positions: list[_Snap] = []

    def update(
        self,
        bot: BotAI,
        step: int,
        radius: float,
    ) -> tuple[list[UnitCluster], list[UnitCluster], list[UnitCluster], list[UnitCluster]]:
        """
        Build fresh clusters, apply velocity from previous snapshot, store new
        snapshot, and return (friendly_ground, friendly_air, enemy_ground, enemy_air).
        """
        fg, fa, eg, ea = _build_clusters_raw(bot, radius)
        elapsed = max(1, step - self._prev_step)

        _apply_velocities(fg, self._prev_fg, elapsed)
        _apply_velocities(fa, self._prev_fa, elapsed)
        _apply_velocities(eg, self._prev_eg, elapsed)
        _apply_velocities(ea, self._prev_ea, elapsed)

        # Ghost tracking: find enemy snaps not matched to any current cluster.
        # These represent groups that have left our field of vision.
        unmatched = (
            _find_unmatched_snaps(eg, self._prev_eg) +
            _find_unmatched_snaps(ea, self._prev_ea)
        )
        self.ghost_enemy_positions = [
            g for g in (self.ghost_enemy_positions + unmatched)
            if step - g.step <= _GHOST_MAX_STEPS
        ]

        self._prev_fg = [_Snap(c.center, step) for c in fg]
        self._prev_fa = [_Snap(c.center, step) for c in fa]
        self._prev_eg = [_Snap(c.center, step) for c in eg]
        self._prev_ea = [_Snap(c.center, step) for c in ea]
        self._prev_step = step

        return fg, fa, eg, ea
