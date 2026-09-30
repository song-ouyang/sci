"""Post hoc true-class accounting of the completed hydraulic policy study.

Offline only. The sole study inputs are the already generated hydraulic policy
CSV and its separate truth file. No raw predictions, other studies, network,
models, policy changes, subgroup selection, or significance tests are accessed.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = ROOT / "results/policy_hydraulic_v1/analysis/policy_cases.csv"
TRUTH = ROOT / "results/policy_hydraulic_v1/answers/truth.json"
OUT = ROOT / "results/policy_error_classes_v1/audit.json"
LABELS = ["No_leakage", "Weak_leakage", "Severe_leakage"]
POLICIES = ["Single_F", "Single_R", "Same", "View", "Majority_F",
            "Majority_R", "Unanimity_F", "Unanimity_R"]
RETAINING = ["Same", "View", "Unanimity_F", "Unanimity_R"]
FIELDS = ["block", "panel", "round", "policy", "id", "truth",
          "base_prediction", "base_correct", "output", "reason", "action"]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    need(path.is_file() and not path.is_symlink(), f"Missing/symlinked file: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evidence(row):
    return {key: row[key] for key in ["id", "panel", "round", "block", "truth",
                                     "base_prediction", "output"]}


def case_key(row):
    return row["panel"], row["round"], row["id"]


def by_truth(rows):
    counts = Counter(row["truth"] for row in rows)
    return {label: counts[label] for label in LABELS}


def event_summary(rows):
    return {"count": len(rows), "unique_cycles": len({r["id"] for r in rows}),
            "by_truth": by_truth(rows),
            "cases": [evidence(r) for r in sorted(rows, key=case_key)]}


def audit():
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in [CASES, TRUTH]}
    truth_artifact = json.loads(TRUTH.read_text(encoding="utf-8"))
    need(set(truth_artifact) == {"labels"}, "Unexpected truth schema")
    truth = truth_artifact["labels"]
    need(len(truth) == 54 and set(truth.values()) == set(LABELS), "Unexpected truth inventory")
    with CASES.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        need(reader.fieldnames == FIELDS, "Unexpected policy-case CSV schema")
        rows = list(reader)
    need(len(rows) == 432 and {r["policy"] for r in rows} == set(POLICIES),
         "Expected the completed 8-policy x54-decision hydraulic accounting")
    for row in rows:
        row["round"], row["block"] = int(row["round"]), int(row["block"])
        need(row["round"] in [1, 2] and 0 <= row["block"] < 6, "Unknown round/block")
        need(row["truth"] == truth[row["id"]], "CSV/truth mismatch")
        need(row["base_prediction"] in LABELS and row["base_correct"] in ["True", "False"],
             "This bounded audit requires the completed valid initial calls")
        row["base_correct"] = row["base_correct"] == "True"
        need(row["base_correct"] == (row["base_prediction"] == row["truth"]),
             "Incorrect baseline correctness flag")
        need(row["output"] in ["", *LABELS], "Invalid policy output")
        expected = ("accepted_correct" if row["output"] == row["truth"] else "accepted_error") if row["output"] else "rejected"
        need(row["action"] == expected, "Action/output mismatch")
        need(row["reason"] in ["accepted", "disagreement", "tie"],
             "Unexpected missingness: this audit concerns fully completed hydraulic calls")
    grouped = {policy: [r for r in rows if r["policy"] == policy] for policy in POLICIES}
    reference = {case_key(r): r for r in grouped["Single_F"]}
    need(len(reference) == 54, "Duplicate or missing initial decisions")
    need(len({r["id"] for r in reference.values()}) == 27, "Expected 27 repeated focal cycles")
    for policy, rr in grouped.items():
        need(len(rr) == 54 and len({case_key(r) for r in rr}) == 54
             and {case_key(r) for r in rr} == set(reference), "Policy decision inventory differs")
        need(by_truth(rr) == {label: 18 for label in LABELS}, "Unexpected full-panel class denominator")
        baseline_policy = "Single_R" if policy.endswith("_R") else "Single_F"
        baseline = {case_key(r): r for r in grouped[baseline_policy]}
        for row in rr:
            need(row["base_prediction"] == baseline[case_key(row)]["output"],
                 "Policy does not use its designated initial prediction")
            if policy in RETAINING and row["output"]:
                need(row["output"] == row["base_prediction"], "Rejection-only gate changed a label")

    retained = {}
    for policy, rr in grouped.items():
        errors = [r for r in rr if r["action"] == "accepted_error"]
        transitions = []
        for true in LABELS:
            for predicted in LABELS:
                matching = [r for r in errors if r["truth"] == true and r["output"] == predicted]
                if matching:
                    transitions.append({"truth": true, "wrong_prediction": predicted,
                                        **event_summary(matching)})
        retained[policy] = {**event_summary(errors), "true_to_wrong_label": transitions}

    lookup = {policy: {case_key(r): r for r in grouped[policy]} for policy in RETAINING}
    persistent = []
    for source_id in sorted({r["id"] for r in reference.values()}):
        appearances = sorted([r for r in reference.values() if r["id"] == source_id],
                             key=lambda r: r["round"])
        need(len(appearances) == 2 and [r["round"] for r in appearances] == [1, 2],
             "Every focal cycle must occur in both planned rounds")
        records = [lookup[p][case_key(r)] for r in appearances for p in RETAINING]
        if not (all(r["action"] == "accepted_error" for r in records)
                and len({r["output"] for r in records}) == 1):
            continue
        persistent.append({"id": source_id, "panel": appearances[0]["panel"],
                           "truth": appearances[0]["truth"], "wrong_prediction": records[0]["output"],
                           "rounds": [1, 2],
                           "decisions": [{"round": r["round"], "block": r["block"],
                                          "retained_by": RETAINING.copy()} for r in appearances]})

    voting = {}
    for policy in ["Majority_F", "Majority_R"]:
        rr = grouped[policy]
        corrected = [r for r in rr if not r["base_correct"] and r["output"] == r["truth"]]
        harmed = [r for r in rr if r["base_correct"] and r["output"] and r["output"] != r["truth"]]
        rejected_correct = [r for r in rr if r["base_correct"] and not r["output"]]
        rejected_error = [r for r in rr if not r["base_correct"] and not r["output"]]
        voting[policy] = {
            "initial_policy": "Single_F" if policy == "Majority_F" else "Single_R",
            "planned_decisions": 54,
            "corrected_base_error": event_summary(corrected),
            "harmed_base_correct": event_summary(harmed),
            "rejected_base_correct": event_summary(rejected_correct),
            "rejected_base_error": event_summary(rejected_error),
        }
    return {
        "schema": "policy-error-classes-v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_type": "post_hoc_descriptive_true_class_accounting",
        "post_hoc": True,
        "source_study": "policy_hydraulic_v1",
        "script_sha256": sha(Path(__file__).resolve()),
        "source_files_sha256": source_hashes,
        "scope": {"policy_case_rows": 432, "policies": 8,
                  "planned_decisions_per_policy": 54, "distinct_focal_cycles": 27,
                  "rounds": 2, "physical_rigs": 1,
                  "decisions_per_true_class_per_policy": {label: 18 for label in LABELS}},
        "retained_errors_by_policy": retained,
        "persistent_retained_errors": {
            "criterion": "Same wrong label retained by Same, View, Unanimity_F and Unanimity_R in each of both recorded rounds. Count a source-round decision once, not once per policy.",
            "policies": RETAINING.copy(),
            "unique_cycles": len(persistent),
            "repeated_decisions": sum(len(r["decisions"]) for r in persistent),
            "records": persistent,
        },
        "voting_changes": voting,
        "limits": [
            "Post hoc interpretation after viewing aggregate policy results; not a preregistered class endpoint",
            "All three true classes are counted; no thresholds, panels, policies, features or calls are selected or changed",
            "Repeated decisions do not increase physical source count; class counts are descriptive, without subgroup significance tests or future-rate claims",
            "Finite repeated wrong labels do not establish a model mechanism or universal stability",
            "Input CSV and truth hashes bind this audit; it does not itself re-audit raw service outputs or the freeze process",
            "No Paderborn data or predictions, network access, new inference, raw signal inspection or new policy scoring",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    result = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    need(not args.output.is_symlink(), "Refusing symlinked output")
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    need(not temporary.is_symlink(), "Refusing symlinked temporary output")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"output": str(args.output),
                      "post_hoc": True,
                      "persistent_unique_cycles": result["persistent_retained_errors"]["unique_cycles"],
                      "persistent_repeated_decisions": result["persistent_retained_errors"]["repeated_decisions"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
