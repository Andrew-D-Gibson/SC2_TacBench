"""
clustering.py — Spatial unit clustering for TacBench tactical observations.

Groups nearby units into clusters so the LLM can reason about *local* force
matchups rather than the full-army aggregate.  Designed to scale from 1 cluster
(current scenarios) to N clusters per side as scenarios grow.

Public API:
    cluster_units(units, radius) -> list[UnitCluster]   # cluster any unit collection
    build_tactical_clusters(bot, radius)                 # returns (friendly, enemy) cluster lists
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from sc2.bot_ai import BotAI
from sc2.ids.unit_typeid import UnitTypeId
from sc2.position import Point2

_ARMY_TYPES = [UnitTypeId.MARINE, UnitTypeId.MARAUDER]

# ── Range category thresholds (tiles) ─────────────────────────────────────────

RANGE_CONTACT = 6.0
RANGE_THREAT  = 10.0
RANGE_NEARBY  = 12.0


def range_label(distance: float) -> str:
    if distance < RANGE_CONTACT:
        return "CONTACT"
    if distance < RANGE_THREAT:
        return "THREAT"
    if distance < RANGE_NEARBY:
        return "NEARBY"
    return "DISTANT"


# ── Force ratio labels ─────────────────────────────────────────────────────────

def ratio_label(friendly_count: int, enemy_count: int) -> str:
    if enemy_count == 0:
        return "ADVANTAGED"
    if friendly_count == 0:
        return "CRITICAL"
    ratio = friendly_count / enemy_count
    if ratio >= 1.2:
        return "ADVANTAGED"
    if ratio >= 0.8:
        return "EVEN"
    if ratio >= 0.5:
        return "DISADVANTAGED"
    return "CRITICAL"


# ── UnitCluster dataclass ──────────────────────────────────────────────────────

@dataclass
class UnitCluster:
    label: str              # "A","B"... for friendly; "1","2"... for enemy
    units: list             # raw python-sc2 unit objects
    center: Point2
    count: int
    hp_current: float
    hp_max: float

    @property
    def hp_pct(self) -> int:
        return int(100 * self.hp_current / max(self.hp_max, 1))

    def distance_to(self, other: "UnitCluster") -> float:
        return self.center.distance_to(other.center)

    def distance_to_point(self, point: Point2) -> float:
        return self.center.distance_to(point)


# ── Clustering algorithm ───────────────────────────────────────────────────────

def cluster_units(units, radius: float = 12.0) -> List[UnitCluster]:
    """
    Greedy radius-based clustering.

    Algorithm (O(n²), fine for SC2 unit counts of 5–30):
    1. Pick the first unassigned unit as a cluster seed.
    2. Repeatedly pull in any unassigned unit within `radius` of the current
       cluster centroid, recomputing the centroid after each addition.
    3. When no more units can be absorbed, seal the cluster and repeat from step 1.

    Returns clusters sorted by unit count descending (largest cluster first).
    """
    remaining = list(units)
    clusters: List[UnitCluster] = []

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
        hp_cur = sum(u.health     for u in members)
        hp_max = sum(u.health_max for u in members)

        clusters.append(UnitCluster(
            label="",   # assigned later
            units=members,
            center=Point2((cx, cy)),
            count=len(members),
            hp_current=hp_cur,
            hp_max=hp_max,
        ))

    # Sort largest-first, break ties by x position for determinism
    clusters.sort(key=lambda c: (-c.count, c.center.x))
    return clusters


def _assign_labels(clusters: List[UnitCluster], use_letters: bool) -> None:
    """Stamp A/B/C... or 1/2/3... labels in-place."""
    for i, c in enumerate(clusters):
        c.label = chr(ord("A") + i) if use_letters else str(i + 1)


# ── Public entry point ─────────────────────────────────────────────────────────

def build_tactical_clusters(
    bot: BotAI,
    radius: float = 12.0,
) -> tuple[List[UnitCluster], List[UnitCluster]]:
    """
    Cluster both sides and return (friendly_clusters, enemy_clusters).

    Friendly clusters  → sorted largest-first, labeled A, B, C...
    Enemy clusters     → sorted by distance to nearest friendly cluster
                         (closest threat first), labeled 1, 2, 3...
    """
    friendly_units = list(bot.units.of_type(_ARMY_TYPES))
    enemy_units    = list(bot.enemy_units.visible)

    friendly = cluster_units(friendly_units, radius)
    enemy    = cluster_units(enemy_units,    radius)

    _assign_labels(friendly, use_letters=True)

    # Sort enemy clusters by their closest distance to any friendly cluster
    if friendly and enemy:
        def _min_dist_to_friendly(ec: UnitCluster) -> float:
            return min(ec.distance_to(fc) for fc in friendly)
        enemy.sort(key=_min_dist_to_friendly)

    _assign_labels(enemy, use_letters=False)

    return friendly, enemy
