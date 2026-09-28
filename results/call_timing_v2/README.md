# Frozen TEP local call timing: post hoc audit

This offline audit verifies 66 source metadata hashes for seven mapping
conditions and four fixed-size context conditions, two services, and three
replicates. It reads no prediction labels and performs no inference or scoring.
Run: `python src/analysis/audit_call_timing_v2.py` from the project root.

`calls.csv` lists every included call separately, with its preassigned replicate
identifier, schedule position, recorded local start, metadata update, subprocess
duration and source hash. `contrasts.csv` covers clean versus each of the six
mapping interventions plus four fixed-size background/position comparisons for
each service. `timing.json` records source hashes, definitions and caveats.

The runner records `started_at` after saving the request, before preparing and
launching the CLI subprocess. This is a client wall-clock timestamp, not an exact
dispatch time or a server inference-start timestamp. `updated_at` is a later local
metadata write; `elapsed_seconds` includes client/transport/service waiting.
No server inference timing is inferred from these fields.

R1/R2/R3 are shuffled replicate identifiers, not chronological or simultaneous
time blocks. A complete separation of local start ranges is explicitly flagged.
An overlap of ranges does not establish adequate temporal balance. Neither fact
alone identifies server state or establishes a causal input effect.

## Example: Sol B-back precedes B-front in all local start records

All B-back local starts lie between 2026-09-27T14:10:30.851261Z and
2026-09-27T14:21:49.402239Z. All B-front starts lie between
2026-09-27T14:26:33.579748Z and 2026-09-27T14:35:16.522218Z.
The gap between the latest B-back start and the earliest B-front start is
284.177509 seconds. This is a
separation of start timestamps, not a demonstrated gap in server computation.

Manuscript wording: “For Sol, all three B-back requests had earlier recorded
client start times than all three B-front requests. Consequently, agreement
within each set of three calls does not isolate a position effect from temporal
service variation. Replicate identifiers were not synchronized time blocks.”

All results are descriptive. No p-values, variability subtraction, independent
sample inflation or causal attribution is performed. The frozen study remains
unchanged, and this new audit is not retrospectively part of its protocol.
