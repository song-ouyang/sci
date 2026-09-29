"""Decision policies over windows of identical calls (offline, frozen outputs).

A window is a set of byte-identical calls that an operator could have issued for
one decision batch. For every window the same four policies are applied:

  single    labels of the first call in schedule order;
  vote      majority over all calls in the window (three-way ties abstain);
  canary    first call whose canary accuracy reaches tau (screen-and-retry),
            otherwise abstain;
  gated     majority over the calls whose canaries pass (ties -> earliest
            passing call), otherwise abstain.

Coverage, accepted error and usable correct yield are reported per study.
tau = 0.8 is the operating target fixed before these analyses; canaries are
labelled records sharing the request and are never scored as focal cases.
"""
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.error_structure_v3 import abba_calls, hydraulic_calls, policy_calls, tep_calls

OUT = ROOT / "results/error_structure_v3"
TAU = 0.8


def prospective_calls():
    base = ROOT / "results/canary_hydraulic_v1"
    if not (base / "predictions/frozen_manifest.json").exists():
        return []
    answers = json.loads((base / "answers/records.json").read_text())["records"]
    truth = {r["id"]: r["true_label"] for r in answers}
    role = {r["id"]: r["role"] for r in answers}
    schedule = json.loads((base / "schedule.json").read_text())["blocks"]
    out = []
    for b, block in enumerate(schedule, 1):
        for pos, name in enumerate(block):
            meta = json.loads((base / f"predictions/gpt-5.6-sol/{name}_metadata.json").read_text())
            if meta["status"] != "complete":
                continue
            preds = json.loads((base / f"predictions/gpt-5.6-sol/{name}_results.json").read_text())
            out.append({"study": "Hydraulic (prospective)", "service": "gpt-5.6-sol",
                        "group": f"{name.split('_')[0]}_b{b}", "order": (b, pos), "block": b,
                        "records": [(p["id"], truth[p["id"]], p["predicted"], role[p["id"]] == "focal") for p in preds]})
    return out


def windows(calls):
    """Identical-input windows that contain canaries."""
    out = defaultdict(list)
    for c in calls:
        if c["service"] != "gpt-5.6-sol" or all(f for *_, f in c["records"]):
            continue
        key = (c["study"], c["group"])
        if c["study"] in {"Hydraulic (new blocks)", "Paderborn bearing"}:
            key = (c["study"], c["group"], c["block"])  # panel x condition x round
        out[key].append(c)
    return {k: sorted(v, key=lambda c: str(c["order"])) for k, v in out.items()}


def canary_ok(call):
    canary = [(t, p) for _, t, p, f in call["records"] if not f]
    return sum(t == p for t, p in canary) / len(canary) >= TAU - 1e-12


def evaluate(window):
    focal = [i for i, _, _, f in window[0]["records"] if f]
    truth = {i: t for i, t, _, _ in window[0]["records"]}
    preds = [{i: p for i, _, p, _ in c["records"]} for c in window]
    passing = [p for c, p in zip(window, preds) if canary_ok(c)]

    def vote(pool, tie_first):
        out = {}
        for i in focal:
            top = Counter(p[i] for p in pool).most_common()
            if len(top) > 1 and top[0][1] == top[1][1]:
                out[i] = pool[0][i] if tie_first else None
            else:
                out[i] = top[0][0]
        return out

    decisions = {"single": preds[0], "vote": vote(preds, False),
                 "canary": passing[0] if passing else {i: None for i in focal},
                 "gated": vote(passing, True) if passing else {i: None for i in focal}}
    calls_used = {"single": 1, "vote": len(window),
                  "canary": next((k + 1 for k, c in enumerate(window) if canary_ok(c)), len(window)),
                  "gated": len(window)}
    return {name: {"correct": sum(d[i] == truth[i] for i in focal if d[i] is not None),
                   "error": sum(d[i] != truth[i] for i in focal if d[i] is not None),
                   "abstain": sum(d[i] is None for i in focal), "calls": calls_used[name]}
            for name, d in decisions.items()}


def main():
    calls = (tep_calls() + abba_calls() + hydraulic_calls() + policy_calls("policy_hydraulic_v1", "Hydraulic (new blocks)")
             + policy_calls("policy_paderborn_v1", "Paderborn bearing") + prospective_calls())
    totals = defaultdict(lambda: defaultdict(Counter))
    n_windows = Counter()
    for key, window in windows(calls).items():
        n_windows[key[0]] += 1
        for policy, r in evaluate(window).items():
            totals[key[0]][policy].update(r)
    rows = []
    for study, policies in totals.items():
        for policy, r in policies.items():
            n = r["correct"] + r["error"] + r["abstain"]
            accepted = r["correct"] + r["error"]
            rows.append({"study": study, "policy": policy, "windows": n_windows[study], "decisions": n,
                         "accepted_correct": r["correct"], "accepted_error": r["error"], "abstained": r["abstain"],
                         "coverage": accepted / n, "accepted_error_rate": r["error"] / accepted if accepted else None,
                         "usable_correct_yield": r["correct"] / n, "calls_per_window": r["calls"] / n_windows[study]})
    with (OUT / "decision_policies.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        e = r["accepted_error_rate"]
        print(f"{r['study']:<24} {r['policy']:<7} windows={r['windows']:>2} n={r['decisions']:>3} AC={r['accepted_correct']:>3} "
              f"AE={r['accepted_error']:>3} abst={r['abstained']:>3} cov={100*r['coverage']:5.1f} "
              f"err={'  -- ' if e is None else f'{100*e:5.1f}'} yield={100*r['usable_correct_yield']:5.1f} calls/w={r['calls_per_window']:.1f}")


if __name__ == "__main__":
    main()
