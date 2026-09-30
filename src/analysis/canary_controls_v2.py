"""Read-only reconstruction of frozen canary calls plus fixed collapse controls.

No inference, training, downloads, resampling or edits of source-study artifacts.
The v1 control is explicitly post hoc; the same implementation can be frozen
before the separately authorized v2 stability calls.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from fractions import Fraction
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.run_hydraulic_v1 import audit_outputs, token_count

THRESHOLDS = (Fraction(9, 10), Fraction(85, 100), Fraction(95, 100), Fraction(1))
POLICIES = ("Single", "Known_answer", "Collapse_90", "Vote_first_tie", "Vote_abstain_tie")

def need(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def collapse_pass(predictions, threshold=Fraction(9, 10)):
    """Needs only returned label values, never true labels or role membership."""
    need(bool(predictions), "Empty valid predictions")
    return Fraction(max(Counter(predictions).values()), len(predictions)) < threshold

def known_answer_pass(predictions, truth, canary):
    return predictions is not None and sum(predictions[i] == truth[i] for i in canary) >= 8

def load_calls(base):
    # The existing independent response audit checks raw/final/parsed agreement
    # and immutable code/input hashes BEFORE this function opens the answer key.
    protocol, groups, audited, _ = audit_outputs(base, require_frozen=True)
    need(sha(base / "answers/records.json") == protocol["answers_sha256"], "Answer hash changed")
    answers = read(base / "answers/records.json")["records"]
    truth = {r["id"]: r["true_label"] for r in answers}
    roles = {r["id"]: r["role"] for r in answers}
    need(len(truth) == len(answers) == 27, "Expected 27 unique records")
    focal = [r["id"] for r in answers if r["role"] == "focal"]
    canary = [r["id"] for r in answers if r["role"] == "canary"]
    need(len(focal) == 18 and len(canary) == 9, "Role partition differs")
    rows, calls = [], []
    by_name = {r["group"]: r for r in audited}
    schedule = read(base / "schedule.json")["blocks"]
    for block, names in enumerate(schedule, 1):
        for slot, name in enumerate(names, 1):
            rec = by_name[name]
            need({s["id"] for s in groups[name]["samples"]} == set(truth), "Input membership changed")
            preds = {r["id"]: r["predicted"] for r in rec["predictions"]} if rec["predictions"] else None
            if preds is not None:
                need(set(preds) == set(truth), "Prediction membership differs")
            known = known_answer_pass(preds, truth, canary)
            collapse = preds is not None and collapse_pass(list(preds.values()))
            row = {"group": name, "arm": groups[name]["condition"], "block": block, "slot": slot,
                   "status": rec["status"], "valid": preds is not None,
                   "modal_all": max(Counter(preds.values()).values()) if preds else None,
                   "n_all": 27, "n_focal": 18, "n_canary": 9,
                   "focal_correct": sum(preds[i] == truth[i] for i in focal) if preds else 0,
                   "canary_correct": sum(preds[i] == truth[i] for i in canary) if preds else None,
                   "known_pass": known, "collapse_90_pass": collapse,
                   "tokens": token_count(rec["usage"]), "elapsed_seconds": rec.get("elapsed_seconds"),
                   "started_at": rec.get("started_at"), "completed_at": rec.get("completed_at")}
            rows.append(row)
            calls.append({**row, "predictions": preds, "focal": focal, "truth": truth,
                          "role": roles, "source_units": {r["id"]: r["block_id"] for r in answers}})
    need(len(calls) == protocol["calls_scheduled"], "Planned call count differs")
    return calls, rows, {"protocol_sha256": sha(base / "protocol.json"),
                        "frozen_manifest_sha256": sha(base / "predictions/frozen_manifest.json"),
                        "answer_sha256": sha(base / "answers/records.json"),
                        "source_focal_blocks": len({r["block_id"] for r in answers if r["role"] == "focal"}),
                        "distinct_focal_cycles": 18, "canary_cycles": 9, "physical_rigs": 1}

def make_windows(calls):
    groups = defaultdict(list)
    for call in calls:
        groups[(call["arm"], call["block"])].append(call)
    result = {}
    for key, group in sorted(groups.items()):
        seq = sorted(group, key=lambda r: r["slot"])
        need(len(seq) == 3 and len({r["slot"] for r in seq}) == 3, "Each window needs three distinct slots")
        need(all(r["focal"] == seq[0]["focal"] and r["truth"] == seq[0]["truth"] for r in seq), "Window data changed")
        result[key] = seq
    return result

def decide(window, policy, threshold=Fraction(9, 10)):
    first = window[0]
    focal = first["focal"]
    out = {i: None for i in focal}
    selected, required = None, window
    if policy == "Single":
        required, selected = window[:1], first if first["valid"] else None
    elif policy in {"Known_answer", "Collapse"}:
        selected = next((r for r in window if r["valid"] and
                         (r["known_pass"] if policy == "Known_answer" else collapse_pass(list(r["predictions"].values()), threshold))), None)
        required = window[:window.index(selected) + 1] if selected else window
    elif policy in {"Vote_first_tie", "Vote_abstain_tie"}:
        if all(r["valid"] for r in window):
            for i in focal:
                counts = Counter(r["predictions"][i] for r in window).most_common()
                tied = len(counts) > 1 and counts[0][1] == counts[1][1]
                out[i] = (first["predictions"][i] if policy == "Vote_first_tie" else None) if tied else counts[0][0]
    else:
        raise ValueError(policy)
    if selected:
        out = {i: selected["predictions"][i] for i in focal}
    return out, required, selected["group"] if selected else ""

def policy_summary(rows, windows):
    n = len(rows)
    ac = sum(r["action"] == "accepted_correct" for r in rows)
    ae = sum(r["action"] == "accepted_error" for r in rows)
    reject = n - ac - ae
    q = sum(r["planned_requests"] for r in windows)
    return {"decisions": n, "windows": len(windows), "accepted_correct": ac, "accepted_error": ae,
            "rejected": reject, "coverage": (ac + ae) / n, "correct_yield": ac / n,
            "accepted_error_rate": ae / (ac + ae) if ac + ae else None,
            "requests": q, "calls_per_window": q / len(windows),
            "known_tokens": sum(r["known_tokens"] for r in windows),
            "unknown_token_calls": sum(r["unknown_token_calls"] for r in windows),
            "client_seconds_sum": sum(r["client_seconds_sum"] for r in windows)}

def evaluate(calls):
    windows = make_windows(calls)
    case_rows, window_rows, cross, sensitivity = [], [], [], []
    definitions = [(p, "Collapse" if p == "Collapse_90" else p, Fraction(9, 10)) for p in POLICIES]
    definitions += [(f"Collapse_{int(t * 100)}", "Collapse", t) for t in THRESHOLDS[1:]]
    for (arm, block), seq in windows.items():
        for label, policy, threshold in definitions:
            outputs, required, selected = decide(seq, policy, threshold)
            wr = {"arm": arm, "block": block, "policy": label, "selected_call": selected,
                  "required_groups": "|".join(r["group"] for r in required),
                  "planned_requests": len(required),
                  "known_tokens": sum(r["tokens"] or 0 for r in required),
                  "unknown_token_calls": sum(r["tokens"] is None and r["status"] != "not_attempted" for r in required),
                  "client_seconds_sum": sum(r["elapsed_seconds"] or 0 for r in required)}
            window_rows.append(wr)
            for i, predicted in outputs.items():
                action = "rejected" if predicted is None else ("accepted_correct" if predicted == seq[0]["truth"][i] else "accepted_error")
                case_rows.append({"arm": arm, "block": block, "id": i, "source_block": seq[0]["source_units"][i],
                                  "policy": label, "truth": seq[0]["truth"][i], "prediction": predicted or "",
                                  "action": action, "selected_call": selected})
    summaries = {label: policy_summary([r for r in case_rows if r["policy"] == label],
                                      [r for r in window_rows if r["policy"] == label]) for label, _, _ in definitions}
    for threshold in THRESHOLDS:
        for known in (False, True):
            for collapse in (False, True):
                subset = [r for r in calls if r["valid"] and r["known_pass"] == known and
                          collapse_pass(list(r["predictions"].values()), threshold) == collapse]
                cross.append({"collapse_threshold": str(threshold), "known_pass": known, "collapse_pass": collapse,
                              "calls": len(subset), "focal_outputs": sum(r["n_focal"] for r in subset),
                              "focal_correct": sum(r["focal_correct"] for r in subset),
                              "focal_errors": sum(r["n_focal"] - r["focal_correct"] for r in subset)})
    for threshold in THRESHOLDS:
        name = f"Collapse_{int(threshold * 100)}"
        sensitivity.append({"policy": name, "rejection_threshold": str(threshold),
                            "minimum_modal_count_to_reject_of_27": (27 * threshold.numerator + threshold.denominator - 1) // threshold.denominator,
                            **summaries[name]})
    by_arm = {arm: {p: policy_summary([r for r in case_rows if r["policy"] == p and r["arm"] == arm],
                                     [r for r in window_rows if r["policy"] == p and r["arm"] == arm])
                   for p in POLICIES} for arm in sorted({r["arm"] for r in calls})}
    by_block = {str(block): {p: policy_summary([r for r in case_rows if r["policy"] == p and r["block"] == block],
                                             [r for r in window_rows if r["policy"] == p and r["block"] == block])
                            for p in POLICIES} for block in sorted({r["block"] for r in calls})}
    return {"policies": summaries, "by_arm": by_arm, "cases": case_rows, "windows": window_rows,
            "by_block": by_block, "cross_tab": cross, "sensitivity": sensitivity}

def csv_write(path, rows):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

def render_tex(result, label="posthoc"):
    m = result["policies"]
    lines = [r"\begin{table}[htbp]\centering\small",
             r"\caption{" + ("Post hoc" if label == "posthoc" else "Prospectively fixed") + r" three-call-window policy comparison. The collapse-only gate rejects when at least 90\% of all returned labels are identical; it does not access known labels. Vote uses the fixed first-call tie rule, with tie abstention shown separately. Calls/window are offline required-call counts; all source requests were collected regardless of gate decisions.}",
             r"\label{tab:canary_controls_" + label + "}",
             r"\begin{tabular}{lrrrrrr}\toprule", r"Policy & Correct & Error & Rejected & Coverage (\%) & Yield (\%) & Calls/window \\\midrule"]
    for p in POLICIES:
        s = m[p]
        lines.append(f"{p.replace('_',' ')} & {s['accepted_correct']} & {s['accepted_error']} & {s['rejected']} & {100*s['coverage']:.1f} & {100*s['correct_yield']:.1f} & {s['calls_per_window']:.2f}" + r" \\")
    lines += [r"\bottomrule\end{tabular}\end{table}",
              "These policies use the same returned payloads and maximum three-call windows. Consequently the control isolates the use of reference labels at the decision stage, not the effect of removing canary records from the prompt. Counts reuse the same focal cycles across arms and blocks. They do not establish independent device replication or a deployment guarantee."]
    return "\n".join(lines) + "\n"

def run(base, out):
    base, out = Path(base).resolve(), Path(out).resolve()
    need(base.parent == ROOT / "results" and out.is_relative_to(ROOT / "results"), "Paths outside results")
    need(out != base and not out.is_relative_to(base / "predictions"), "Cannot overwrite source predictions")
    if base.name == "canary_hydraulic_v1":
        need((out / "PLAN.md").exists(), "Post hoc plan must precede scoring")
    calls, call_rows, provenance = load_calls(base)
    result = evaluate(calls)
    out.mkdir(parents=True, exist_ok=True)
    for filename, rows in [("calls.csv", call_rows), ("policy_cases.csv", result["cases"]),
                           ("policy_windows.csv", result["windows"]), ("gate_cross_tab.csv", result["cross_tab"]),
                           ("collapse_threshold_sensitivity.csv", result["sensitivity"]),
                           ("summary.csv", [{"policy": k, **v} for k, v in result["policies"].items()])]:
        csv_write(out / filename, rows)
    interpretation = "posthoc_existing_27_calls" if base.name == "canary_hydraulic_v1" else "prospectively_fixed_new_source_time_followup"
    summary = {"schema": "canary-controls-v2", "interpretation": interpretation, "new_model_calls_by_analysis": 0,
               "source_study": str(base.relative_to(ROOT)), "script_sha256": sha(__file__),
               "provenance": provenance, "scheduled_calls": len(calls), "valid_calls": sum(r["valid"] for r in calls),
               "primary_collapse_threshold": "9/10", "sensitivity_thresholds": [str(t) for t in THRESHOLDS],
               "policies": result["policies"], "by_arm": result["by_arm"], "by_block": result["by_block"], "gate_cross_tab": result["cross_tab"],
               "limitations": ["Fixed full payload; label-free gate does not remove canary records", "Repeated focal records share requests and are not independent devices",
                               "No p-value or population confidence interval", "Response screening does not establish actual maintenance or monetary benefit"]}
    if (out / "PLAN.md").exists():
        summary["analysis_plan_sha256"] = sha(out / "PLAN.md")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "manuscript_fragment.tex").write_text(render_tex(result, "posthoc" if base.name == "canary_hydraulic_v1" else "stability"))
    (out / "REPORT.md").write_text("# Fixed canary/collapse control\n\n" + interpretation + "\n\n" +
        "All tables reconstruct frozen outputs in integer schedule order. No model calls are made by this analysis.\n\n" +
        "\n".join(f"- {p}: AC={s['accepted_correct']}, AE={s['accepted_error']}, withheld={s['rejected']}, calls/window={s['calls_per_window']:.3f}." for p,s in summary["policies"].items()) +
        "\n\nThe full acceptance cross-tab and threshold sensitivity are retained. This control is not evidence that canaries can be removed from prompts without changing model behavior.\n")
    return summary

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/canary_hydraulic_v1")
    parser.add_argument("--out", type=Path, default=ROOT / "results/canary_controls_v2")
    args = parser.parse_args()
    result = run(args.root, args.out)
    print(json.dumps({k: result[k] for k in ("interpretation", "scheduled_calls", "valid_calls", "policies")}, indent=2))
