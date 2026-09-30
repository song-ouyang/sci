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
COMPLETED_DATE = "2026-09-30"


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


def empirical_loss(stats, c, total_request_loss=0):
    """Evaluate a specified, uncalibrated total referral/request-loss scenario."""
    need(np.isfinite(c) and c >= 0, "Referral loss must be finite and nonnegative")
    need(np.isfinite(total_request_loss) and total_request_loss >= 0,
         "Total request loss must be finite and nonnegative")
    return (stats["accepted_error"] + c * stats["rejected"] + total_request_loss) / stats["N"]


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
    for p in POLICIES[:-1]:
        old_policy = "Single_F" if p == "Accept_all_F1" else p
        selected = [r for r in costs if r["policy"] == old_policy]
        need(len(selected) == len(data["panels"]) * 2, "Policy-cost rows missing")
        calls = 1 if p == "Accept_all_F1" else 2
        need(all(int(r["planned_requests"]) == int(r["attempted_requests"]) == calls and int(r["unknown_token_calls"]) == 0 for r in selected), "Unequal/unknown call budget")
        for cost in selected:
            selected_requests = [r for r in requests if r["panel"] == cost["panel"] and r["round"] == cost["round"]
                                 and ((r["condition"], int(r["repeat"])) in
                                      (("F", 1),) + (() if p == "Accept_all_F1" else (("F" if p == "Same" else "R", 2),)))]
            need(len(selected_requests) == calls, "Required-call cost inventory differs")
            need(sum(int(r["tokens"]) for r in selected_requests) == int(cost["known_tokens"]), "Token-cost export differs")
            need(abs(sum(float(r["elapsed_seconds"]) for r in selected_requests) - float(cost["client_service_seconds_sum"])) < 1e-9,
                 "Duration-cost export differs")
        budget[p] = {field: sum(float(r[field]) if field == "client_service_seconds_sum" else int(r[field]) for r in selected)
                     for field in ("planned_requests", "attempted_requests", "known_tokens", "client_service_seconds_sum")}
    budget["Reject_all"] = {field: 0 for field in budget["Accept_all_F1"]}
    need(budget["Same"]["planned_requests"] == budget["View"]["planned_requests"], "Unequal request counts")
    n_units = len({r["source_unit"] for r in rows})
    need(n_units == (27 if dataset == "hydraulic" else 9), "Source cluster count changed")
    return rows, {"dataset": dataset, "physical_rigs": 1, "distinct_focal_records": len(focal),
                  "source_units": n_units, "source_unit_kind": rows[0]["source_unit_kind"],
                  "rounds": 2, "totals": totals,
                  "same_view_budget": {p: budget[p] for p in ("Same", "View")},
                  "policy_resource_totals": budget,
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


def contrast_summary(rows):
    """Exact endpoint checks of the affine View-minus-Same loss contrast."""
    result = []
    for dataset in STUDIES:
        selected = [r for r in rows if r["dataset"] == dataset]
        for label, subset in [("all", selected)] + [
                (f"round_{rnd}", [r for r in selected if r["round"] == rnd]) for rnd in (1, 2)]:
            same, view = (summarize([r for r in subset if r["policy"] == p]) for p in ("Same", "View"))
            delta_e = view["accepted_error"] - same["accepted_error"]
            delta_r = view["rejected"] - same["rejected"]
            result.append({"dataset": dataset, "subset": label, "N": same["N"],
                           "delta_errors": delta_e, "delta_rejections": delta_r,
                           "delta_correct": view["accepted_correct"] - same["accepted_correct"],
                           "loss_difference_c0": delta_e / same["N"],
                           "loss_difference_c1": (delta_e + delta_r) / same["N"],
                           "View_strictly_lower_all_c_0_to_1": max(delta_e, delta_e + delta_r) < 0,
                           **intersection(view, same)})
    return result


def write_csv(path, rows):
    need(bool(rows), "Empty CSV")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def manuscript_text(studies):
    h, b = (studies[d]["totals"] for d in ("hydraulic", "paderborn"))
    hs, hv, bs, bv = h["Same"], h["View"], b["Same"], b["View"]
    # Bind each numerical assertion used in the prose to the independent counts.
    need((hs["N"], hs["accepted_error"], hs["rejected"], hv["accepted_error"], hv["rejected"],
          hv["accepted_correct"] - hs["accepted_correct"]) == (54, 4, 17, 5, 10, 6), "Hydraulic prose/count mismatch")
    need((bs["N"], bs["accepted_error"], bs["rejected"], bv["accepted_error"], bv["rejected"],
          bv["accepted_correct"]) == (72, 58, 6, 21, 42, 9), "Bearing prose/count mismatch")
    h_cross = intersection(hs, hv)["c_exact"]
    h_same_base = intersection(hs, h["Accept_all_F1"])["c_exact"]
    h_view_base = intersection(hv, h["Accept_all_F1"])["c_exact"]
    b_cross = intersection(bs, bv)
    b_reject = intersection(bv, b["Reject_all"])["c_exact"]
    b_accept = intersection(bv, b["Accept_all_F1"])["c_exact"]
    need(h_cross == "1/7" and h_same_base == "6/17" and h_view_base == "1/2", "Hydraulic prose/intersection mismatch")
    need(b_cross["c_exact"] == "37/36" and b_reject == "7/10" and b_accept == "20/21", "Bearing prose/intersection mismatch")
    return [
        f"A result-informed loss sensitivity analysis compared the fixed two-request checks without selecting a new deployment rule. When total request loss is omitted (K_p=0), hydraulic Same and View have empirical losses (4+17c)/54 and (5+10c)/54. Their difference, View minus Same, is (1-7c)/54 and favors View only when c>{h_cross}: its six additional accepted correct diagnoses accompany one additional accepted error. Compared with accepting every F1 output, the empirical break-even referral losses are {h_same_base} for Same and {h_view_base} for View. Neither check improves every endpoint.",
        f"In the bearing study, the corresponding losses are (58+6c)/72 and (21+42c)/72. View has lower observed loss than Same throughout 0<=c<=1, but this comparison alone does not establish useful deployment: rejecting every diagnosis has lower observed loss than View when c<{b_reject}, and accepting every F1 diagnosis has lower observed loss when c>{b_accept}. Only between these empirical intersections does View beat both reference actions under this loss specification with K_p=0. The retained error fraction remains {bv['accepted_error']}/{bv['accepted_error']+bv['accepted_correct']} ({100*bv['selective_error']:.1f}%). These are descriptive sample-dependent tradeoffs, not optimized, validated or calibrated industrial decisions.",
        "The decision counts contain two rounds on 27 hydraulic cycles and 36 bearing recordings, not 54 or 72 independent devices. Source and round breakdowns retain the dependence structure. The hydraulic data concern one rig; its 27 condition blocks are not 27 physical rigs, and no rig-level confidence interval is estimated. For Paderborn, class-stratified paired resampling keeps all four regimes and both rounds together for each of nine physical bearings. Its 2.5th--97.5th percentiles describe sensitivity to empirical specimen weights conditional on the observed requests and fixed backgrounds. Shared batch responses leave dependence across bearings, so these bands are not population confidence intervals or evidence of cross-rig generalization."]


def sensitivity_summary(contrasts, deleted, boot):
    summaries = []
    for dataset in STUDIES:
        for dimension in ("source_unit", "panel", "round"):
            selected = [r for r in deleted if r["dataset"] == dataset and r["excluded_dimension"] == dimension]
            if not selected:
                continue
            summaries.append({"dataset": dataset, "excluded_dimension": dimension, "deletions": len(selected),
                **{field + "_range": [min(r[field] for r in selected), max(r[field] for r in selected)]
                   for field in ("delta_error_View_minus_Same", "delta_rejected_View_minus_Same", "delta_correct_View_minus_Same")},
                "View_strictly_lower_all_c_0_to_1_count": sum(max(r["delta_error_View_minus_Same"], r["delta_error_View_minus_Same"] +
                     r["delta_rejected_View_minus_Same"]) < 0 for r in selected)})
    intervals = [r for r in boot if r["quantity"] == "loss_View_minus_Same" and r["c"] in (0, .5, 1)]
    return {"deletion_summaries": summaries, "bearing_selected_conditional_percentiles": intervals,
            "rounds": contrasts, "request_loss_K": 0,
            "interpretation": "Deterministic deletion and empirical source reweighting conditional on observed requests; no new devices, environments or service periods"}


def sensitivity_text(contrasts, deleted, boot):
    c = {(r["dataset"], r["subset"]): r for r in contrasts}
    h1, h2 = (c["hydraulic", f"round_{rnd}"] for rnd in (1, 2))
    b1, b2 = (c["paderborn", f"round_{rnd}"] for rnd in (1, 2))
    need(h1["kind"] == "coincident" and (h2["delta_correct"], h2["delta_errors"]) == (6, 1), "Hydraulic round prose differs")
    need((b1["delta_errors"], b2["delta_errors"], b1["delta_correct"], b2["delta_correct"]) == (-20, -17, -1, 2), "Bearing round prose differs")
    bearings = [r for r in deleted if r["dataset"] == "paderborn" and r["excluded_dimension"] == "source_unit"]
    reversal = [r for r in bearings if max(r["delta_error_View_minus_Same"], r["delta_error_View_minus_Same"] + r["delta_rejected_View_minus_Same"]) >= 0]
    need(len(bearings) == 9 and len(reversal) == 1 and reversal[0]["excluded_group"] == "KI04", "Bearing deletion prose differs")
    panels = [r for r in deleted if r["dataset"] == "hydraulic" and r["excluded_dimension"] == "panel"]
    intervals = {r["c"]: r for r in boot if r["quantity"] == "loss_View_minus_Same"}
    return [
        f"Round and deletion sensitivity limits any stability claim. Hydraulic Same and View have equal aggregate counts in round 1; the entire six-correct/one-error increment occurs in round 2. Deleting one hydraulic panel moves the View--Same crossing among {', '.join(sorted({r['c_exact'] for r in panels}))}; therefore the pooled crossing is not a stable engineering constant.",
        f"For bearings, View reduces accepted errors in both rounds (by {-b1['delta_errors']} and {-b2['delta_errors']}) but retains {b1['delta_correct']} and +{b2['delta_correct']} correct diagnoses relative to Same. Its round-1 loss becomes higher than Same for c>{b1['c_exact']}; round 2 favors View over the entire displayed range. Deleting each physical bearing still reduces accepted errors (by {min(-r['delta_error_View_minus_Same'] for r in bearings)}--{max(-r['delta_error_View_minus_Same'] for r in bearings)}), but deleting {reversal[0]['excluded_group']} reverses the high-cost comparison for c>{reversal[0]['c_exact']}. Thus error-release reduction survives these particular deletions, whereas a uniformly lower loss does not.",
        f"At c=1 the conditional bearing reweighting percentiles for View-minus-Same loss are [{intervals[1]['q025']:.4f}, {intervals[1]['q975']:.4f}], spanning both signs. This is a sensitivity to the weights assigned to the nine observed bearings, not a population confidence interval or a validation on additional specimens."]


def report_text(studies, contrasts, deleted, boot):
    lines = ["# Descriptive decision-value audit (2026-09-29 schema; completed 2026-09-30)", "",
             "This result-informed offline analysis adds no model calls and changes no policy or frozen outcome. Same retains F1 when F1=F2; View retains F1 when F1=R2. Both use two requests per panel-round, although observed token use and duration differ. Rejecting a base-correct answer is a lost correct decision; rejecting a base-error only withholds it and does not establish that anyone repaired it.", "",
             "Assign loss 0 to an accepted correct diagnosis, loss 1 to an accepted error, and loss c to every rejection/referral. Here c is the TOTAL expected downstream loss of withholding/referring a diagnosis, which may include handling, delay and residual errors after referral; it does not assume that a human perfectly repairs rejected errors. A common c for all rejections and a common unit loss for all accepted errors are simplifying sensitivity assumptions, not measured factory costs. Heterogeneous severity, class-dependent referral outcomes and referral capacity are not modeled. The inspected range 0<=c<=1 is illustrative, not a claim that real referral losses cannot exceed 1.", "",
             "For policy p, L_p(c,K_p)=(E_p+cR_p+K_p)/N, where E_p is accepted errors, R_p all rejections, N the planned focal decisions, and K_p the TOTAL loss attributable to that policy's prescribed diagnostic requests in the same normalized units. A hypothetical per-request loss kappa would give K_p=kappa*Q_p; alternatively one may supply an independently justified K_p using tokens, elapsed time or other resources without counting downstream referral loss twice. The function empirical_loss(stats,c,total_request_loss) supports this scenario input. No monetary calibration is inferred here. All reported curves, crossings and sensitivity summaries set K_p=0; consequently they are decision-loss comparisons that omit request costs, NOT cost-effectiveness estimates. They do not include downstream referral outcomes beyond the assumed c.", "",
             "Accept-all uses the same initial F1 predictions (one request per panel-round); reject-all needs no diagnostic requests but incurs c for every decision. Only Same/View have equal request counts; their measured tokens and summed client service durations differ, so their actual K_p need not be equal. For unequal K_p the View-minus-Same numerator gains K_View-K_Same and the displayed crossings move. All policy resource totals, including accept-all and reject-all, are retained in audit.json.", "",
             "|Dataset|Policy|N|Accepted correct|Accepted errors|Rejected base-correct|Rejected base-error|Coverage|Correct yield|Empirical loss|",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for dataset, study in studies.items():
        for p, s in study["totals"].items():
            lines.append(f"|{dataset}|{p}|{s['N']}|{s['accepted_correct']}|{s['accepted_error']}|{s['rejected_base_correct']}|{s['rejected_base_error']}|{s['coverage']:.3f}|{s['correct_yield']:.3f}|({s['accepted_error']}+{s['rejected']}c)/{s['N']}|")
    lines += ["", "## English text for manuscript integration", "", *[s + "\n" for s in manuscript_text(studies)],
              "## Round and source sensitivity (request costs omitted)", "", *[s + "\n" for s in sensitivity_text(contrasts, deleted, boot)],
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
                           "total_request_loss_K": 0,
                           "loss": empirical_loss(stats, float(c))} for c in COST_GRID)
        for left, right in itertools.combinations(POLICIES, 2):
            cross = intersection(study["totals"][left], study["totals"][right])
            cross.setdefault("left_lower_below", None)
            cross.setdefault("left_lower_above", None)
            crossings.append({"dataset": dataset, "left": left, "right": right, **cross})
        rr = [r for r in all_rows if r["dataset"] == dataset]
        groups.extend(group_summaries(rr))
        deleted.extend(leave_one_group_out(rr))
    boot, boot_meta = bootstrap_sensitivity([r for r in all_rows if r["dataset"] == "paderborn"])
    contrasts = contrast_summary(all_rows)
    for filename, rows in (("decision_cases.csv", all_rows), ("summary.csv", summaries),
                           ("loss_curves.csv", curves), ("intersections.csv", crossings),
                           ("source_round_breakdown.csv", groups), ("leave_one_group_out.csv", deleted),
                           ("round_loss_contrasts.csv", contrasts),
                           ("bearing_resampling_sensitivity.csv", boot)):
        for row in rows:
            if filename == "leave_one_group_out.csv":
                row.setdefault("left_lower_below", None)
                row.setdefault("left_lower_above", None)
        write_csv(output / filename, rows)
    (output / "REPORT.md").write_text(report_text(studies, contrasts, deleted, boot))
    # Compact prose in TeX-safe form; the manuscript builder may select passages.
    (output / "manuscript_fragment.tex").write_text("% Offline descriptive follow-up, completed 2026-09-30.\n" +
        "\n\n".join(s.replace("K_p", "$K_p$").replace("<=", r"$\leq$").replace("%", r"\%")
                      for s in manuscript_text(studies) + sensitivity_text(contrasts, deleted, boot)) + "\n")
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
        ax.set(title=dataset.capitalize(), xlabel="Total referral loss c (accepted error = 1)", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Empirical decision loss (request cost omitted)")
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(output / "decision_loss.png", dpi=180)
    fig.savefig(output / "decision_loss.svg", metadata={"Date": None})
    fig.savefig(output / "decision_loss.pdf", metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)
    # End-of-run hashes confirm all inspected original artifacts are unchanged.
    for study in studies.values():
        for relative, expected in study["source_hashes"].items():
            need(sha(root / relative) == expected, "Input changed during audit")
    audit = {"schema": "decision-value-audit-2026-09-29", "analysis_type": "result_informed_descriptive_loss_sensitivity",
             "completed_date": COMPLETED_DATE, "completed_date_timezone": "Asia/Shanghai",
             "new_model_calls": 0, "network_requests": 0, "policy_selection": False,
             "script_sha256": sha(Path(__file__).resolve()), "studies": studies,
             "loss_definition": {"general": "L_p(c,K_p)=(accepted_error+c*all_rejections+K_p)/planned_decisions",
                                 "plotted": "K_p=0 for every policy; c in [0,1] is an uncalibrated sensitivity range",
                                 "c": "Total expected downstream rejection/referral loss, potentially including handling, delay and residual errors; no perfect repair assumption",
                                 "K_p": "Total diagnostic-request loss for this policy; if constant hypothetical per-call loss kappa is used, K_p=kappa*Q_p; no conversion from tokens to currency assumed",
                                 "unit": "Accepted correct diagnosis = 0; accepted error = 1; equal error severity and constant referral loss assumed",
                                 "limits": "No measured referral outcomes, capacity effects or actual deployment-cost validation; all empirical crossings are sample-dependent"},
             "round_loss_contrasts": contrasts,
             "sensitivity_summary": sensitivity_summary(contrasts, deleted, boot),
             "leave_one_group_out": deleted,
             "all_pairwise_intersections_K0": crossings,
             "paired_bearing_resampling": boot_meta,
             "hydraulic_interval": "Not estimated: one physical rig; condition blocks do not provide independently sampled rigs",
             "output_sha256": {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "audit.json"},
             "checks": {"frozen_members_valid": True, "protocol_sources_valid": True,
                        "predictions_match_frozen_responses": True, "gates_independently_recomputed": True,
                        "old_summary_counts_match": True, "same_view_equal_request_counts": True,
                        "policy_resource_totals_match_required_requests": True, "manuscript_numbers_bound_to_counts": True,
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
