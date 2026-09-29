"""Offline, result-informed decision-loss accounting of frozen policy studies.

Writes only the new decision_value_audit directory. No network, inference,
training, policy tuning, new labels, or imports of execution/old scoring code.
The optional bearing bootstrap is a conditional reweighting sensitivity analysis,
not a population confidence interval: shared batch responses couple specimens.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/decision_value_audit_2026-09-29"
STUDIES = {"hydraulic": "policy_hydraulic_v1", "paderborn": "policy_paderborn_v1"}
POLICIES = ("Accept_all_F1", "Same", "View", "Reject_all")
SEED = 20260929
BOOTSTRAP_REPLICATES = 10000
COST_GRID = np.linspace(0, 1, 101)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    need(path.is_file() and not path.is_symlink(), f"Missing/symlinked input: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def csv_read(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def summarize(rows):
    """All denominators concern the same initial F1 decisions, not devices."""
    n = len(rows)
    e = sum(r["action"] == "accepted_error" for r in rows)
    a = sum(r["action"] == "accepted_correct" for r in rows)
    rc = sum(r["action"] == "rejected" and r["base_correct"] for r in rows)
    re = sum(r["action"] == "rejected" and not r["base_correct"] for r in rows)
    need(n == a + e + rc + re and n > 0, "Non-partitioned or empty decisions")
    return {"N": n, "accepted_correct": a, "accepted_error": e,
            "rejected_base_correct": rc, "rejected_base_error": re,
            "rejected": rc + re, "base_correct": a + rc, "base_errors": e + re,
            "coverage": (a + e) / n, "correct_yield": a / n,
            "error_release_fraction": e / n, "selective_error": fraction(e, a + e),
            "false_rejection_fraction": fraction(rc, a + rc),
            "error_capture_fraction": fraction(re, e + re),
            "loss_intercept": e / n, "loss_slope": (rc + re) / n}


def intersection(left, right):
    """Exact c where normalized affine losses tie, including outside [0,1]."""
    delta_e = Fraction(left["accepted_error"], left["N"]) - Fraction(right["accepted_error"], right["N"])
    delta_r = Fraction(left["rejected"], left["N"]) - Fraction(right["rejected"], right["N"])
    if not delta_r:
        return {"kind": "coincident" if not delta_e else "parallel",
                "c_exact": None, "c": None, "in_unit_interval": False}
    c = -delta_e / delta_r
    return {"kind": "single_intersection", "c_exact": str(c), "c": float(c),
            "in_unit_interval": 0 <= c <= 1,
            "left_lower_below": delta_r > 0,
            "left_lower_above": delta_r < 0}


def policy_actions(base_prediction, f2, r2, truth):
    """A gate retains F1 on agreement; it never repairs/replaces its label."""
    outputs = {"Accept_all_F1": base_prediction,
               "Same": base_prediction if base_prediction == f2 else "",
               "View": base_prediction if base_prediction == r2 else "", "Reject_all": ""}
    return {p: {"output": output, "action": "rejected" if not output else
                ("accepted_correct" if output == truth else "accepted_error")}
            for p, output in outputs.items()}


def load_study(root, dataset):
    base = root / "results" / STUDIES[dataset]
    paths = [base / f for f in (
        "protocol.json", "protocol.sha256", "source_audit.json", "answers/truth.json",
        "inference/dataset.json", "predictions/frozen_manifest.json",
        "analysis/policy_cases.csv", "analysis/predictions.csv", "analysis/requests.csv",
        "analysis/policy_costs.csv", "analysis/scores.json")]
    hashes = {str(p.relative_to(root)): sha(p) for p in paths}
    protocol, frozen = read(base / "protocol.json"), read(base / "predictions/frozen_manifest.json")
    need(sha(base / "protocol.json") == (base / "protocol.sha256").read_text().strip(), "Protocol checksum mismatch")
    need(frozen["protocol_sha256"] == sha(base / "protocol.json"), "Frozen protocol mismatch")
    for relative, expected in frozen["files"].items():
        target = base / "predictions" / relative
        need(target.resolve().is_relative_to((base / "predictions").resolve()), "Frozen path escape")
        need(sha(target) == expected, f"Frozen response mismatch: {relative}")
    for field, relative in (("answers_sha256", "answers/truth.json"),
                            ("source_audit_sha256", "source_audit.json"),
                            ("dataset_sha256", "inference/dataset.json")):
        need(protocol[field] == sha(base / relative), f"Protocol-linked source changed: {relative}")
    truth = read(base / "answers/truth.json")["labels"]
    data, source = read(base / "inference/dataset.json"), read(base / "source_audit.json")
    labels = set(data["allowed_labels"])
    focal = {i: panel["panel_id"] for panel in data["panels"] for i in panel["focal_ids"]}
    need(len(focal) == (27 if dataset == "hydraulic" else 36), "Focal inventory mismatch")
    src = source["selected_records"] if dataset == "hydraulic" else source["records"]
    metadata = {x["id"]: x for x in src if x["role"] == "focal"}
    need(set(metadata) == set(focal), "Source metadata/focal mismatch")
    requests = csv_read(base / "analysis/requests.csv")
    need(len(requests) == (36 if dataset == "hydraulic" else 12), "Request count mismatch")
    need(all(r["status"] == "complete" for r in requests), "Audit requires complete requests")
    request_index = {r["group"]: r for r in requests}
    need(len(request_index) == len(requests), "Duplicate request")
    raw = {}
    for group in request_index:
        relative = f"{protocol['model']}/{group}_results.json"
        need(relative in frozen["files"], "Parsed results not in frozen manifest")
        result = read(base / "predictions" / relative)
        raw[group] = {r["id"]: r["predicted"] for r in result}
        need(len(raw[group]) == len(result), "Duplicate frozen result id")
        need(set(raw[group].values()) <= labels, "Illegal frozen label")
    predictions = csv_read(base / "analysis/predictions.csv")
    need(len(predictions) == sum(len(x) for x in raw.values()), "Prediction export count differs")
    observed_keys, by_call = set(), {}
    for r in predictions:
        need((r["group"], r["id"]) not in observed_keys, "Duplicate exported prediction")
        observed_keys.add((r["group"], r["id"]))
        need(raw[r["group"]][r["id"]] == r["predicted"] and truth[r["id"]] == r["truth"], "Frozen prediction/export/truth mismatch")
        need(r["correct"] == str(r["predicted"] == r["truth"]), "Incorrect exported correctness")
        req = request_index[r["group"]]
        need(all(r[k] == req[k] for k in ("panel", "round", "condition", "repeat")), "Request index mismatch")
        is_focal = r["id"] in focal
        need(r["focal"] == str(is_focal), "Incorrect focal indicator")
        if is_focal:
            key = (r["panel"], int(r["round"]), r["id"], r["condition"], int(r["repeat"]))
            need(key not in by_call and focal[r["id"]] == r["panel"], "Duplicate/misassigned focal prediction")
            by_call[key] = r["predicted"]
    old_rows = csv_read(base / "analysis/policy_cases.csv")
    old_index = {(r["panel"], int(r["round"]), r["id"], r["policy"]): r for r in old_rows}
    need(len(old_index) == len(old_rows) == len(focal) * 2 * 8, "Policy case inventory mismatch")
    rows = []
    for source_id, panel in sorted(focal.items()):
        meta = metadata[source_id]
        true = truth[source_id]
        need(true == meta["true_label" if dataset == "hydraulic" else "label"], "Source label mismatch")
        unit = f"joint_block_{meta['block_id']}" if dataset == "hydraulic" else meta["bearing"]
        for round_no in (1, 2):
            key = (panel, round_no, source_id)
            f1, f2, r2 = (by_call[key + condition] for condition in (("F", 1), ("F", 2), ("R", 2)))
            for policy, value in policy_actions(f1, f2, r2, true).items():
                row = {"dataset": dataset, "panel": panel, "round": round_no, "id": source_id,
                       "source_unit": unit, "source_unit_kind": "joint_condition_block" if dataset == "hydraulic" else "physical_bearing",
                       "regime_or_cooler": str(meta["profile"][0]) if dataset == "hydraulic" else meta["regime"],
                       "truth": true, "policy": policy, "base_prediction": f1,
                       "base_correct": f1 == true, **value}
                if policy != "Reject_all":
                    old = old_index[key + (("Single_F" if policy == "Accept_all_F1" else policy),)]
                    need(old["output"] == value["output"] and old["action"] == value["action"]
                         and old["base_correct"] == str(f1 == true) and old["base_prediction"] == f1
                         and old["truth"] == true, "Independent gate recomputation differs")
                rows.append(row)
    totals = {p: summarize([r for r in rows if r["policy"] == p]) for p in POLICIES}
    previous = read(base / "analysis/scores.json")
    need(previous["protocol_sha256"] == sha(base / "protocol.json") and
         previous["frozen_manifest_sha256"] == sha(base / "predictions/frozen_manifest.json"), "Score linkage differs")
    for p in POLICIES[:-1]:
        old = previous["policies"]["Single_F" if p == "Accept_all_F1" else p]
        for field in ("accepted_correct", "accepted_error", "rejected_base_correct", "rejected_base_error", "coverage", "selective_error"):
            need(totals[p][field] == old[field], f"Old summary differs: {p}/{field}")
    costs = csv_read(base / "analysis/policy_costs.csv")
    budget = {}
    for p in ("Same", "View"):
        selected = [r for r in costs if r["policy"] == p]
        need(len(selected) == len(data["panels"]) * 2, "Policy-cost rows missing")
        need(all(int(r["planned_requests"]) == int(r["attempted_requests"]) == 2 and int(r["unknown_token_calls"]) == 0 for r in selected), "Unequal/unknown call budget")
        budget[p] = {field: sum(float(r[field]) if field == "client_service_seconds_sum" else int(r[field]) for r in selected)
                     for field in ("planned_requests", "attempted_requests", "known_tokens", "client_service_seconds_sum")}
    need(budget["Same"]["planned_requests"] == budget["View"]["planned_requests"], "Unequal request counts")
    n_units = len({r["source_unit"] for r in rows})
    need(n_units == (27 if dataset == "hydraulic" else 9), "Source cluster count changed")
    return rows, {"dataset": dataset, "physical_rigs": 1, "distinct_focal_records": len(focal),
                  "source_units": n_units, "source_unit_kind": rows[0]["source_unit_kind"],
                  "rounds": 2, "totals": totals, "same_view_budget": budget,
                  "source_hashes": hashes, "frozen_response_files_verified": len(frozen["files"])}


def stratified_cluster_totals(rows, replicates=BOOTSTRAP_REPLICATES, seed=SEED):
    """Paired empirical reweighting of bearings; all rounds/regimes stay intact."""
    units = sorted({r["source_unit"] for r in rows})
    strata = defaultdict(list)
    matrix = np.zeros((len(units), len(POLICIES), 4), dtype=int)
    for j, unit in enumerate(units):
        rr = [r for r in rows if r["source_unit"] == unit]
        truth = {r["truth"] for r in rr}
        need(len(truth) == 1, "Bearing has inconsistent truth")
        strata[next(iter(truth))].append(j)
        for k, p in enumerate(POLICIES):
            stats = summarize([r for r in rr if r["policy"] == p])
            matrix[j, k] = [stats[x] for x in ("N", "accepted_error", "rejected", "accepted_correct")]
    rng = np.random.default_rng(seed)
    totals = np.zeros((replicates, len(POLICIES), 4), dtype=int)
    for label in sorted(strata):
        indices = np.array(strata[label])
        sampled = rng.choice(indices, size=(replicates, len(indices)), replace=True)
        totals += matrix[sampled].sum(axis=1)
    return totals, {label: len(v) for label, v in sorted(strata.items())}


def bootstrap_sensitivity(rows):
    draws, strata = stratified_cluster_totals(rows)
    need(strata == {"healthy": 3, "inner_race_fault": 3, "outer_race_fault": 3}, "Unexpected bearing strata")
    need(np.all(draws[:, :, 0] == 72), "Bootstrap broke source-round inventory")
    result = []
    # The same sampled specimens enter both policies, preserving their pairing.
    for c in COST_GRID:
        loss = (draws[:, :, 1] + c * draws[:, :, 2]) / draws[:, :, 0]
        for k, p in enumerate(POLICIES):
            low, high = np.quantile(loss[:, k], [0.025, 0.975])
            result.append({"dataset": "paderborn", "quantity": f"loss_{p}", "c": round(float(c), 2),
                           "q025": float(low), "q975": float(high)})
        low, high = np.quantile(loss[:, 2] - loss[:, 1], [0.025, 0.975])
        result.append({"dataset": "paderborn", "quantity": "loss_View_minus_Same", "c": round(float(c), 2),
                       "q025": float(low), "q975": float(high)})
    return result, {"replicates": BOOTSTRAP_REPLICATES, "seed": SEED, "strata": strata,
                    "resampling_unit": "physical_bearing_with_all_four_regimes_and_two_rounds",
                    "interpretation": "95% empirical reweighting sensitivity intervals conditional on observed joint responses, fixed backgrounds, class composition and service period; NOT population confidence intervals",
                    "limitations": ["Only three bearings per class and nine total", "Cross-bearing dependence remains because outputs share requests and backgrounds", "No resampling of unseen rigs, factories, service periods or request contexts", "Percentiles do not quantify uncertainty in deployment costs or establish strategy superiority"]}


def group_summaries(rows):
    result = []
    for fields in (("round",), ("panel",), ("source_unit",), ("source_unit", "round"),
                   ("regime_or_cooler",), ("truth",)):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[f] for f in fields) + (row["policy"],)].append(row)
        for key, group in sorted(groups.items()):
            result.append({"dataset": rows[0]["dataset"], "grouping": "+".join(fields),
                           "group": "|".join(map(str, key[:-1])), "policy": key[-1], **summarize(group)})
    return result


def leave_one_group_out(rows):
    result = []
    # Deterministic sensitivity: units, whole request panels, and whole rounds.
    for field in ("source_unit", "panel", "round"):
        groups = sorted({r[field] for r in rows})
        if len(groups) < 2:
            continue
        for excluded in groups:
            retained = [r for r in rows if r[field] != excluded]
            same = summarize([r for r in retained if r["policy"] == "Same"])
            view = summarize([r for r in retained if r["policy"] == "View"])
            result.append({"dataset": rows[0]["dataset"], "excluded_dimension": field,
                           "excluded_group": str(excluded), "retained_N": same["N"],
                           "delta_error_View_minus_Same": view["accepted_error"] - same["accepted_error"],
                           "delta_rejected_View_minus_Same": view["rejected"] - same["rejected"],
                           "delta_correct_View_minus_Same": view["accepted_correct"] - same["accepted_correct"],
                           **intersection(view, same)})
    return result


def write_csv(path, rows):
    need(bool(rows), "Empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report_text(studies):
    lines = ["# Descriptive decision-value audit (2026-09-29)", "",
             "This result-informed offline analysis adds no model calls and changes no policy or frozen outcome. Same retains F1 when F1=F2; View retains F1 when F1=R2. Both use two requests per panel-round, although observed token use and duration differ. Rejecting a base-correct answer is a lost correct decision; rejecting a base-error only withholds it and does not establish that anyone repaired it.", "",
             "With a correct accepted diagnosis assigned loss 0, an incorrect accepted diagnosis loss 1, and every rejected diagnosis loss c, the empirical loss is L(c)=(E+cR)/N for 0<=c<=1. E is accepted errors, R all rejections, and N all planned focal decisions. Calls, delay, fault severity and downstream repair are excluded. c is an uncalibrated sensitivity parameter, not a clinical or factory cost estimate. Accept-all uses the same initial F1 predictions (one request per panel-round); reject-all needs no diagnostic calls. Thus only Same/View are equal-call-budget comparisons.", "",
             "|Dataset|Policy|N|Accepted correct|Accepted errors|Rejected base-correct|Rejected base-error|Coverage|Correct yield|Empirical loss|",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for dataset, study in studies.items():
        for p, s in study["totals"].items():
            lines.append(f"|{dataset}|{p}|{s['N']}|{s['accepted_correct']}|{s['accepted_error']}|{s['rejected_base_correct']}|{s['rejected_base_error']}|{s['coverage']:.3f}|{s['correct_yield']:.3f}|({s['accepted_error']}+{s['rejected']}c)/{s['N']}|")
    lines += ["", "## English text for manuscript integration", "",
              "A result-informed loss sensitivity analysis compared the fixed two-request checks without selecting a new deployment rule. In the hydraulic study, Same and View had empirical losses (4+17c)/54 and (5+10c)/54, respectively. The difference, (1-7c)/54, favors View only when c>1/7: its six additional accepted correct diagnoses accompany one additional accepted error. Compared with accepting every F1 output, the empirical break-even rejection costs are 6/17 for Same and 1/2 for View. Neither check improves every endpoint.", "",
              "In the bearing study, the corresponding losses were (58+6c)/72 and (21+42c)/72. View has lower observed loss than Same throughout 0<=c<=1, but this comparison alone does not establish useful deployment: rejecting every diagnosis has lower observed loss than View when c<21/30, and accepting every F1 diagnosis has lower observed loss when c>20/21. Only between these two empirical intersections does View beat both reference actions under this particular loss specification. The retained error fraction remains 21/30 (70.0%). These are descriptive sample-dependent tradeoffs, not optimized, validated or calibrated industrial decisions.", "",
              "The decision counts contain two rounds on 27 hydraulic cycles and 36 bearing recordings, not 54 or 72 independent devices. Source and round breakdowns retain the dependence structure. The hydraulic data concern one rig; its 27 condition blocks are not 27 physical rigs, and no rig-level confidence interval is estimated. For Paderborn, a class-stratified paired resampling analysis keeps all four regimes and both rounds together for each of nine physical bearings. Its 2.5th–97.5th percentiles describe sensitivity to empirical specimen weights conditional on the observed requests and fixed backgrounds. Shared batch responses leave dependence across bearings, so these bands are not population confidence intervals or evidence of cross-rig generalization.", "",
              "## Reproduction and boundaries", "",
              "Run `.venv/bin/python src/analysis/audit_decision_value.py`. Frozen response member hashes, protocol-linked source/truth/dataset hashes, exported predictions against frozen parsed responses, and independent F1/F2/R2 gate reconstruction are checked before writing new outputs. Existing policy counts must match exactly. The analysis reads but never executes old inference or scoring code. No files in the source studies are changed.", "",
              "`decision_cases.csv` has one row per initial decision and action; `summary.csv` reports the partition and normalized loss coefficients; `source_round_breakdown.csv` covers all sources, rounds, panels, true labels and operating groups. `loss_curves.csv` evaluates c from 0 to 1 in steps of 0.01. `intersections.csv` includes all exact pairwise intersections, including those outside the stipulated cost range; these are not fitted thresholds. `leave_one_group_out.csv` is deterministic deletion sensitivity. `bearing_resampling_sensitivity.csv` contains conditional percentiles with the limitations above. `audit.json` binds input/output hashes and resampling settings.", "",
              "This audit was designed after aggregate outcomes were known. No subgroup, record, round, cost value or policy is dropped or selected for favorable performance. No hypothesis tests or causal effect claims are added. Two rounds cannot establish service stability. Equal request counts do not imply equal monetary cost or latency, and no token-to-currency conversion is assumed.", ""]
    return "\n".join(lines)


def run(root=ROOT, output=OUT):
    root, output = root.resolve(), output.resolve()
    need(output.name == OUT.name and output.parent == root / "results", "Output restricted to the new audit directory")
    output.mkdir(parents=True, exist_ok=True)
    all_rows, studies = [], {}
    for dataset in STUDIES:
        rows, info = load_study(root, dataset)
        all_rows.extend(rows)
        studies[dataset] = info
    summaries, curves, crossings, groups, deleted = [], [], [], [], []
    for dataset, study in studies.items():
        for policy, stats in study["totals"].items():
            summaries.append({"dataset": dataset, "policy": policy, **stats})
            curves.extend({"dataset": dataset, "policy": policy, "c": round(float(c), 2),
                           "loss": (stats["accepted_error"] + float(c) * stats["rejected"]) / stats["N"]} for c in COST_GRID)
        for left, right in itertools.combinations(POLICIES, 2):
            cross = intersection(study["totals"][left], study["totals"][right])
            cross.setdefault("left_lower_below", None)
            cross.setdefault("left_lower_above", None)
            crossings.append({"dataset": dataset, "left": left, "right": right, **cross})
        rr = [r for r in all_rows if r["dataset"] == dataset]
        groups.extend(group_summaries(rr))
        deleted.extend(leave_one_group_out(rr))
    boot, boot_meta = bootstrap_sensitivity([r for r in all_rows if r["dataset"] == "paderborn"])
    for filename, rows in (("decision_cases.csv", all_rows), ("summary.csv", summaries),
                           ("loss_curves.csv", curves), ("intersections.csv", crossings),
                           ("source_round_breakdown.csv", groups), ("leave_one_group_out.csv", deleted),
                           ("bearing_resampling_sensitivity.csv", boot)):
        for row in rows:
            if filename == "leave_one_group_out.csv":
                row.setdefault("left_lower_below", None)
                row.setdefault("left_lower_above", None)
        write_csv(output / filename, rows)
    (output / "REPORT.md").write_text(report_text(studies))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=True, constrained_layout=True)
    styles = {"Accept_all_F1": ("#555555", "--"), "Same": ("#0072B2", "-"),
              "View": ("#D55E00", "-"), "Reject_all": ("#888888", ":")}
    for ax, (dataset, study) in zip(axes, studies.items()):
        for p, stats in study["totals"].items():
            color, linestyle = styles[p]
            ax.plot(COST_GRID, stats["loss_intercept"] + COST_GRID * stats["loss_slope"], label=p.replace("_", " "), color=color, linestyle=linestyle)
        ax.set(title=dataset.capitalize(), xlabel="Rejection loss c (accepted error = 1)", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Empirical loss per planned decision")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(output / "decision_loss.png", dpi=180)
    fig.savefig(output / "decision_loss.svg", metadata={"Date": None})
    plt.close(fig)
    # End-of-run hashes confirm all inspected original artifacts are unchanged.
    for study in studies.values():
        for relative, expected in study["source_hashes"].items():
            need(sha(root / relative) == expected, "Input changed during audit")
    audit = {"schema": "decision-value-audit-2026-09-29", "analysis_type": "result_informed_descriptive_loss_sensitivity",
             "new_model_calls": 0, "network_requests": 0, "policy_selection": False,
             "script_sha256": sha(Path(__file__).resolve()), "studies": studies,
             "loss_definition": "L(c)=(accepted_error+c*all_rejections)/planned_decisions; correct acceptance costs 0; no cost for calls/delay/repair; c in [0,1] is uncalibrated",
             "paired_bearing_resampling": boot_meta,
             "hydraulic_interval": "Not estimated: one physical rig; condition blocks do not provide independently sampled rigs",
             "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "audit.json"},
             "checks": {"frozen_members_valid": True, "protocol_sources_valid": True,
                        "predictions_match_frozen_responses": True, "gates_independently_recomputed": True,
                        "old_summary_counts_match": True, "same_view_equal_request_counts": True,
                        "input_hashes_unchanged": True}}
    (output / "audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = run(args.root, args.root / "results" / OUT.name)
    print(json.dumps({"output": str(args.root / "results" / OUT.name), "checks": result["checks"],
                      "decisions_per_policy": {k: v["totals"]["Same"]["N"] for k, v in result["studies"].items()}}, indent=2))
