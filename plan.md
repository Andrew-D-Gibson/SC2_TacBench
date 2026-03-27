Movement/Velocity — How Big a Gap?

  Moderate, and scenario-dependent. With K=30 steps between calls, a slow
  infantry engagement probably doesn't move far enough to matter much. But
  for faster scenarios — an enemy army charging across the map, or a unit
  trying to flank — the LLM is always reacting to where things were, not
  where they're going. This leads to ghost-chasing (attacking empty ground)
   and being surprised by contact.

  How to fill it: Track previous cluster positions between LLM calls and
  add a velocity line to the tactical overview:
  vs ENEMY CLUSTER 1 (8 units @ (55, 60)) — dist 14.2 [NEARBY] | moving
  TOWARD you (~4 tiles/call)
  vs ENEMY CLUSTER 2 (3 units @ (40, 35)) — dist 8.2 [THREAT]  | stationary
  This requires storing cluster centroids from the previous call in
  BaseSC2Bot and doing a simple delta. Cheap to implement, high value for
  dynamic scenarios.

  ---
  Other Capability Gaps

  Situational awareness

  - Attack range is invisible. The LLM has no idea whether an enemy at
  distance 8 can actually shoot it. A Marine has range 5, a Siege Tank has
  range 13. Your THREAT/NEARBY thresholds are fixed but real danger zones
  vary by unit composition. The LLM can't tell if it's already taking fire
  or safely out of range.
  - Fog of war is silent. "No enemies visible" looks identical to "enemies
  are hidden nearby." The LLM can't distinguish absence from concealment.
  Showing a last-known-position for recently seen clusters ("ENEMY CLUSTER
  1: last seen at (55, 60), 12 steps ago — current position unknown") would
   let it reason about hidden threats.
  - Cluster spread is invisible. The LLM only sees a centroid. Ten units
  packed into a 2-tile ball fight completely differently from ten units
  spread over 20 tiles, but both look the same in the report. Adding a
  spread radius or bounding box would help.
  - High ground advantage. Terrain shows height symbols but the LLM has no
  knowledge that attacking uphill has a miss chance penalty. It might
  confidently ATTACK into a high-ground fortification with no idea it's at
  a disadvantage.

  Temporal / dynamic

  - HP delta between calls. Knowing you lost 120 HP since the last
  observation is more actionable than just current HP. "Your group took 180
   damage in the last 30 steps" signals active engagement even if no enemy
  is currently CONTACT.
  - New/lost enemy clusters. The LLM has no way to detect "ENEMY CLUSTER 2
  appeared since last call" or "ENEMY CLUSTER 1 was destroyed." The
  observation is a snapshot with no change markers.
  - Directive staleness. The current directive might be 60 steps stale
  because the LLM is busy. A note like "current directive: ATTACK (set 60
  steps ago)" would help it understand why units might be mid-march when
  the situation has already changed.

  Unit capabilities

  - No ability awareness. Marauder has a slow field, Marines can stim. None
   of this is exposed. The LLM can't make decisions that involve trading HP
   for speed (stim) or controlling enemy kiting (slow).
  - Unit attack ranges not shown. Related to above — the LLM issues
  FOCUS_FIRE without knowing whether your units are even in range to shoot.

  Strategic / mission

  - No explicit objective progress. The briefing tells the LLM what to do,
  but there's no "you are 15 tiles from the objective" or "objective is 40%
   complete." It has to infer progress from unit positions and the scenario
   briefing, which works but is fragile.
  - Steps remaining unknown. The observation shows step 450 but not step
  450/1000 (45% of time used). Given that urgency was important enough to
  add to the doctrine, showing the remaining budget explicitly seems
  worthwhile.
  - No outcome feedback in history. The history section shows past
  directives but not whether things got better or worse afterward. "RETREAT
   → friendly HP recovered from 320 to 480" is more useful than just
  "RETREAT."

  Control granularity

  - One command for all units. This is the biggest structural gap. You
  can't flank, you can't sacrifice a unit to buy time, you can't have one
  group engage while another retreats. Everything happens in lockstep. This
   is the per-group directive work you already identified as future work.
  - No target selection for FOCUS_FIRE. The handler picks lowest-HP
  automatically, which is usually right but removes LLM agency. Sometimes
  killing the highest-threat unit (Siege Tank, Medivac) matters more than
  the lowest-HP Zergling.

  ---
  Priority Order for a New Mission

  If I were ranking which of these to address before the next mission:

  1. Steps remaining — trivial to add, directly relevant to time-pressure
  scenarios
  2. HP delta / damage taken — one extra field, high signal value
  3. Movement velocity — moderate effort, high value for dynamic scenarios
  4. Fog of war / last-known position — important if the mission involves
  concealment
  5. Attack range awareness — would need unit type → range table, but would
   dramatically improve engagement decisions