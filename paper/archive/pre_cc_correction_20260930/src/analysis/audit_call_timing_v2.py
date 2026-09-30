"""Read-only source audit of local client call timing in the frozen TEP study.

No model calls, response scoring, or changes to frozen artifacts. Outputs are a
new descriptive audit under results/call_timing_v2. The original runner records
started_at before preparing/launching its CLI subprocess; this is not a server
inference timestamp. R1/R2/R3 labels are replicate identifiers, not time blocks.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/independent_validation"
OUT = ROOT / "results/call_timing_v2"
MAPPING = ("clean", "rotation", "random_1", "random_2", "random_3", "same_unit_swap", "pair_shuffle")
CONTEXT = ("a_front", "a_back", "b_front", "b_back")
CONTRASTS = tuple(("clean", c, "mapping") for c in MAPPING if c != "clean") + (
    ("a_front", "b_front", "background_at_front"),
    ("a_back", "b_back", "background_at_back"),
    ("a_front", "a_back", "position_with_A"),
    ("b_front", "b_back", "position_with_B"),
)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise ValueError(message)


def parsed_time(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    need(parsed.tzinfo is not None, "Naive timestamp cannot establish UTC timing")
    return parsed.astimezone(timezone.utc)


def utc(value):
    return parsed_time(value).isoformat(timespec="microseconds").replace("+00:00", "Z")


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    protocol = load(BASE / "protocol.json")
    frozen = load(BASE / "predictions/frozen_manifest.json")
    schedule = load(BASE / "session_schedule.json")
    need(sha(BASE / "protocol.json") == frozen["protocol_sha256"], "Protocol/frozen manifest mismatch")
    need(sha(BASE / "protocol.json") == (BASE / "protocol.sha256").read_text().strip(), "Protocol SHA mismatch")
    need(sha(BASE / "session_schedule.json") == protocol["schedule_sha256"], "Schedule hash mismatch")
    runner = ROOT / "src/experiments/run_revision_v3.py"
    need(sha(runner) == protocol["code_sha256"][str(runner.relative_to(ROOT))], "Runner differs from frozen version")
    locations = {(j["model"], j["group"]): i + 1 for i, j in enumerate(schedule)}
    source_hashes = {str(p.relative_to(ROOT)): sha(p) for p in (
        BASE / "protocol.json", BASE / "protocol.sha256", BASE / "session_schedule.json",
        BASE / "predictions/frozen_manifest.json", runner)}
    calls = []
    for model in protocol["models"]:
        for condition in MAPPING + CONTEXT:
            for repeat in (1, 2, 3):
                group = f"{condition}_r{repeat}"
                relative = f"{model}/{group}_metadata.json"
                path = BASE / "predictions" / relative
                digest = sha(path)
                need(digest == frozen["files"][relative], f"Frozen metadata hash mismatch: {relative}")
                meta = load(path)
                need(meta["condition"] == condition and meta["repeat"] == repeat and
                     meta["model_requested"] == model and meta["dataset"] == "tep", "Metadata association mismatch")
                need(meta["protocol_sha256"] == frozen["protocol_sha256"] and meta["runner_sha256"] == sha(runner),
                     "Metadata runner/protocol mismatch")
                need(len(meta["attempts"]) == 1, "Multiple attempts require a separate timing audit")
                source_hashes[str(path.relative_to(ROOT))] = digest
                start, updated = utc(meta["started_at"]), utc(meta["updated_at"])
                need(parsed_time(updated) >= parsed_time(start), "Metadata update precedes runner start")
                calls.append({
                    "model_requested": model, "condition": condition, "repeat_label": repeat,
                    "group": group, "request_n": meta["n_samples"], "status": meta["status"],
                    "scheduled_position_1based": locations[(model, group)],
                    "runner_recorded_start_utc": start, "metadata_updated_at_utc": updated,
                    "client_subprocess_elapsed_seconds": meta["attempts"][0]["elapsed_seconds"],
                    "server_inference_start_utc": "not_recorded",
                    "server_resolved_model": meta.get("server_resolved_model") or "not_recorded",
                    "temperature": meta.get("temperature") if meta.get("temperature") is not None else "not_recorded",
                    "metadata_path": str(path.relative_to(ROOT)), "metadata_sha256": digest,
                })
    need(len(calls) == 66, "Expected 2 models × 11 conditions × 3 calls")
    calls.sort(key=lambda row: (row["model_requested"], row["runner_recorded_start_utc"]))
    timing_by_condition = {}
    for model in protocol["models"]:
        for condition in MAPPING + CONTEXT:
            rows = [r for r in calls if r["model_requested"] == model and r["condition"] == condition]
            timing_by_condition[f"{model}/{condition}"] = {
                "start_min_utc": rows[0]["runner_recorded_start_utc"],
                "start_max_utc": rows[-1]["runner_recorded_start_utc"],
                "replicate_labels_in_local_start_order": [r["repeat_label"] for r in rows],
                "calls_in_local_start_order": [r["group"] for r in rows],
            }
    comparisons = []
    for model in protocol["models"]:
        for left, right, kind in CONTRASTS:
            a, b = timing_by_condition[f"{model}/{left}"], timing_by_condition[f"{model}/{right}"]
            amin, amax = parsed_time(a["start_min_utc"]), parsed_time(a["start_max_utc"])
            bmin, bmax = parsed_time(b["start_min_utc"]), parsed_time(b["start_max_utc"])
            if amax < bmin:
                relation, gap = "all_left_starts_before_all_right_starts", (bmin - amax).total_seconds()
            elif bmax < amin:
                relation, gap = "all_right_starts_before_all_left_starts", (amin - bmax).total_seconds()
            else:
                relation, gap = "start_ranges_overlap_or_touch", ""
            comparisons.append({
                "model_requested": model, "left_condition": left, "right_condition": right, "contrast_kind": kind,
                "n_left_calls": 3, "n_right_calls": 3,
                "left_start_min_utc": a["start_min_utc"], "left_start_max_utc": a["start_max_utc"],
                "right_start_min_utc": b["start_min_utc"], "right_start_max_utc": b["start_max_utc"],
                "local_start_relation": relation, "positive_gap_between_start_ranges_seconds": gap,
                "interpretation": "Local start order only; overlapping start ranges do not establish temporal balance, and separated starts do not by themselves prove server inference intervals were disjoint.",
            })
    separated = [r for r in comparisons if r["local_start_relation"] != "start_ranges_overlap_or_touch"]
    example = next(r for r in comparisons if r["model_requested"] == "gpt-5.6-sol" and r["left_condition"] == "b_front" and r["right_condition"] == "b_back")
    outcome = {
        "schema": "call-timing-audit-v2", "status": "descriptive_audit_complete",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Seven TEP mapping conditions and four fixed-size contexts, both services, all three replicates; excludes early, focal-only and FD002",
        "calls_audited": len(calls), "contrasts_audited": len(comparisons),
        "all_metadata_hashes_verified": True, "metadata_files_verified": len(calls),
        "input_source_hashes": source_hashes,
        "analysis_code_sha256": sha(Path(__file__)),
        "timestamp_semantics": {
            "started_at": "Local UTC wall-clock value assigned by the frozen runner after saving the request and before constructing/launching the Codex CLI subprocess. It is not a scheduler enqueue timestamp, exact network dispatch timestamp, or server inference start.",
            "updated_at": "Local UTC metadata-write time after subprocess completion and event parsing; not a server inference completion timestamp.",
            "elapsed_seconds": "Rounded local monotonic subprocess-duration measurement, including client/transport/service waiting; not isolated server compute time.",
            "server_inference_timestamps": "Unavailable in the audited metadata; not inferred from client values.",
        },
        "replicate_index_semantics": "R1/R2/R3 are preassigned replicate identifiers shuffled in the session schedule, not synchronous blocks or chronological labels.",
        "condition_timing": timing_by_condition,
        "completely_separated_local_start_contrasts": separated,
        "sol_b_front_vs_b_back": example,
        "boundaries": [
            "Start-range overlap is not evidence of adequate balance or absence of temporal confounding.",
            "A stable difference across three calls per condition cannot isolate an input effect from unobserved service state or chance.",
            "Model aliases do not establish immutable server weights; temperature was not exposed/set by the runner.",
            "Calls sharing a batch induce case dependence; repeated labels and cross-pair comparisons are not new independent cases.",
            "This is a post hoc audit of existing logs; no new inference, response scoring, protocol changes or outcome selection.",
        ],
    }
    # Recheck source hashes immediately before writing only the new audit outputs.
    need(all(sha(ROOT / p) == value for p, value in source_hashes.items()), "Source changed during audit")
    OUT.mkdir(parents=True, exist_ok=True)
    save_csv(OUT / "calls.csv", calls)
    save_csv(OUT / "contrasts.csv", comparisons)
    (OUT / "timing.json").write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf-8")
    readme = f"""# Frozen TEP local call timing: post hoc audit

This offline audit verifies {len(calls)} source metadata hashes for seven mapping
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

All B-back local starts lie between {example['right_start_min_utc']} and
{example['right_start_max_utc']}. All B-front starts lie between
{example['left_start_min_utc']} and {example['left_start_max_utc']}.
The gap between the latest B-back start and the earliest B-front start is
{example['positive_gap_between_start_ranges_seconds']} seconds. This is a
separation of start timestamps, not a demonstrated gap in server computation.

Manuscript wording: “For Sol, all three B-back requests had earlier recorded
client start times than all three B-front requests. Consequently, agreement
within each set of three calls does not isolate a position effect from temporal
service variation. Replicate identifiers were not synchronized time blocks.”

All results are descriptive. No p-values, variability subtraction, independent
sample inflation or causal attribution is performed. The frozen study remains
unchanged, and this new audit is not retrospectively part of its protocol.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")
    print(json.dumps({"calls": len(calls), "contrasts": len(comparisons),
                      "separated_start_contrasts": len(separated), "sol_example": example}, indent=2))


if __name__ == "__main__":
    main()
