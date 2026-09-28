"""Service-period trend and repeated-call voting on the frozen TEP outputs (offline).

* Time trend: per-call accuracy of every Sol/Astra request on the source-disjoint
  panel against its client start time. Conditions whose correct descriptor-value
  associations are intact are pooled to show call-level drift independent of
  mapping faults.
* Voting: majority vote over the three repeated calls of the same input
  (ties resolved by the first call), compared with the mean single-call
  accuracy and the three-fold request cost.
"""
from collections import Counter, defaultdict
import csv
from datetime import datetime
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/statistics_v2"
INTACT = {"clean", "pair_shuffle", "focal_only", "a_front", "a_back", "b_front", "b_back"}


def load_cases():
    table, truth = defaultdict(dict), {}
    with open(ROOT / "results/independent_validation/analysis/cases.csv", newline="") as handle:
        for r in csv.DictReader(handle):
            table[(r["model"], r["condition"], int(r["repeat"]))][r["id"]] = r["predicted"]
            truth[r["id"]] = r["true_label"]
    return table, truth


def main():
    table, truth = load_cases()
    focal = sorted(table[("gpt-5.6-sol", "focal_only", 1)])
    starts = {}
    with open(ROOT / "results/call_timing_v2/calls.csv", newline="") as handle:
        for r in csv.DictReader(handle):
            starts[(r["model_requested"], r["condition"], int(r["repeat_label"]))] = r["runner_recorded_start_utc"]
    rows = []
    for (model, cond, rep), preds in sorted(table.items()):
        if cond in {"early", "fd002"} or (model, cond, rep) not in starts:
            continue
        ids = focal if cond in {"a_front", "a_back", "b_front", "b_back"} else sorted(preds)
        acc = float(np.mean([preds[i] == truth[i] for i in ids]))
        rows.append({"model": model, "condition": cond, "repeat": rep, "n": len(ids), "accuracy": acc,
                     "intact_mapping": cond in INTACT,
                     "start_utc": starts[(model, cond, rep)].replace("Z", "+00:00")})
    origin = min(datetime.fromisoformat(r["start_utc"]) for r in rows)
    for r in rows:
        r["start_minute"] = (datetime.fromisoformat(r["start_utc"]) - origin).total_seconds() / 60
    trend = {}
    for model in ["gpt-5.6-sol", "gpt-6-astra"]:
        sub = [r for r in rows if r["model"] == model and r["intact_mapping"]]
        rho, p = spearmanr([r["start_minute"] for r in sub], [r["accuracy"] for r in sub])
        first, second = [r["accuracy"] for r in sub if r["start_minute"] < 12], [r["accuracy"] for r in sub if r["start_minute"] >= 12]
        trend[model] = {"n_calls": len(sub), "spearman_rho": None if np.isnan(rho) else float(rho),
                        "spearman_p": None if np.isnan(p) else float(p),
                        "mean_acc_first_12_min": float(np.mean(first)), "n_first": len(first),
                        "mean_acc_after_12_min": float(np.mean(second)), "n_after": len(second)}
    votes = []
    for model in ["gpt-5.6-sol", "gpt-6-astra"]:
        for cond in ["clean", "rotation", "random_1", "random_2", "random_3", "same_unit_swap", "pair_shuffle",
                     "focal_only", "a_front", "a_back", "b_front", "b_back", "early", "fd002"]:
            calls = [table[(model, cond, k)] for k in (1, 2, 3)]
            ids = focal if cond in {"a_front", "a_back", "b_front", "b_back"} else sorted(calls[0])
            single = [np.mean([c[i] == truth[i] for i in ids]) for c in calls]
            voted = []
            for i in ids:
                counts = Counter(c[i] for c in calls)
                top = max(counts.values())
                winner = next(c[i] for c in calls if counts[c[i]] == top)
                voted.append(winner == truth[i])
            votes.append({"model": model, "condition": cond, "n": len(ids),
                          "single_call_mean": float(np.mean(single)), "single_call_min": float(np.min(single)),
                          "single_call_max": float(np.max(single)), "majority_of_3": float(np.mean(voted)),
                          "request_cost_factor": 3})
    for name, data in [("call_time_trend.csv", rows), ("voting.csv", votes)]:
        with (OUT / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (OUT / "time_trend.json").write_text(json.dumps(trend, indent=2) + "\n")
    print(json.dumps(trend, indent=2))
    for v in votes:
        print(f"{v['model']:<12} {v['condition']:<15} single={100*v['single_call_mean']:5.1f} "
              f"[{100*v['single_call_min']:5.1f},{100*v['single_call_max']:5.1f}] vote={100*v['majority_of_3']:5.1f}")
    for r in sorted((r for r in rows if r["model"] == "gpt-5.6-sol"), key=lambda r: r["start_minute"]):
        print(f"{r['start_minute']:6.1f} {r['condition']:<15} r{r['repeat']} {100*r['accuracy']:5.1f} intact={r['intact_mapping']}")


if __name__ == "__main__":
    main()
