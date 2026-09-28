# ABBA/BAAB exploratory context replication

{
  "planned_sessions": 16,
  "attempted_sessions": 16,
  "complete_sessions": 16,
  "not_attempted_sessions": 0,
  "planned_prediction_slots": 672,
  "valid_prediction_slots": 672,
  "observed_total_tokens": 1010051,
  "unknown_usage_sessions": 0,
  "usable_correct_slots": 652,
  "valid_focal_slots": 336,
  "scheduled_focal_slots": 336,
  "complete_blocks": 4
}

| Block | Sequence | Valid/4 | Stable different | Stable same | Any unstable | Incomplete |
|---|---|---:|---:|---:|---:|---:|
| 1 | ABBA | 4/4 | 0 | 19 | 2 | 0 |
| 2 | ABBA | 4/4 | 0 | 19 | 2 | 0 |
| 3 | BAAB | 4/4 | 0 | 21 | 0 | 0 |
| 4 | BAAB | 4/4 | 0 | 13 | 8 | 0 |

Exploratory replication selected after known outcomes; no new source units.
Ordinal-position balance does not equal clock-time balance or eliminate service drift.
Starts are client wrapper dispatch timestamps; ends are client metadata updates. Midpoints summarize this client-side request envelope, not server inference timestamps.
Within-block pairs reuse the same 21 focal cases and calls; no independent-pair inference.
Two-repeat stable differences are finite observations, not causal lower bounds; zero does not prove no effect.
Failed and unattempted requests retain full planned denominators; incomplete blocks are not filled from another block.
