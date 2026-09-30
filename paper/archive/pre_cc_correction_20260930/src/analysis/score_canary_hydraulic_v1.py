"""Frozen scoring rules for the prospective canary-screening study (offline).

Written and hashed into the protocol before any inference. Reads the frozen
predictions and, only after the freeze check, the answer key.

Prespecified endpoints (tau = 8/9 canary labels correct):
  E1  Spearman correlation between canary accuracy and focal accuracy over all
      valid calls, with an exact-permutation one-sided p-value (10,000 random
      permutations of focal accuracies across calls, seed 20260929).
  E2  Focal accuracy of accepted versus all valid calls, and of accepted versus
      rejected calls; accepted-call coverage.
  E3  Screen-and-retry: in each arm x block window, the first call (in frozen
      schedule order) whose canaries pass delivers the focal labels; windows with
      no passing call abstain. Reported: delivered accuracy, abstentions, calls
      used, compared with the window's first call alone and with a three-call
      majority vote (ties -> first call's label).
  E4  Hidden value swap (arm H) versus canonical (arm C): focal accuracy and
      canary rejection rate, with a within-block exact randomization test that
      exchanges whole calls between the two arms.
All calls, failures and abstentions are reported; no threshold is tuned.
"""
from collections import Counter
import itertools
import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/canary_hydraulic_v1"
TAU = 8 / 9
SEED = 20260929


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    manifest = BASE / "predictions/frozen_manifest.json"
    if not manifest.exists():
        sys.exit("Outputs are not frozen; refusing to open the answer key.")
    protocol = load(BASE / "protocol.json")
    schedule = load(BASE / "schedule.json")["blocks"]
    groups = load(BASE / "inference/batches.json")["groups"]
    answers = load(BASE / "answers/records.json")
    truth = {r["id"]: r["true_label"] for r in answers["records"]}
    role = {r["id"]: r["role"] for r in answers["records"]}
    folder = BASE / "predictions" / protocol["model"]
    calls = []
    for b, block in enumerate(schedule, 1):
        for pos, name in enumerate(block):
            meta = load(folder / f"{name}_metadata.json")
            row = {"group": name, "arm": groups[name]["condition"], "block": b, "position": pos,
                   "status": meta["status"]}
            if meta["status"] == "complete":
                preds = {p["id"]: p["predicted"] for p in load(folder / f"{name}_results.json")}
                focal = [i for i in preds if role[i] == "focal"]
                canary = [i for i in preds if role[i] == "canary"]
                row.update(focal_correct=sum(preds[i] == truth[i] for i in focal), n_focal=len(focal),
                           canary_correct=sum(preds[i] == truth[i] for i in canary), n_canary=len(canary),
                           focal_modal=Counter(preds[i] for i in focal).most_common(1)[0][1],
                           labels={i: preds[i] for i in focal})
                row["focal_accuracy"] = row["focal_correct"] / row["n_focal"]
                row["canary_accuracy"] = row["canary_correct"] / row["n_canary"]
                row["accepted"] = row["canary_accuracy"] >= TAU - 1e-12
            calls.append(row)
    valid = [c for c in calls if c["status"] == "complete"]
    fa = np.array([c["focal_accuracy"] for c in valid])
    ca = np.array([c["canary_accuracy"] for c in valid])
    rng = np.random.default_rng(SEED)
    rho = spearmanr(ca, fa)[0] if len(set(ca)) > 1 and len(set(fa)) > 1 else float("nan")
    null = [spearmanr(ca, rng.permutation(fa))[0] for _ in range(10000)] if not np.isnan(rho) else []
    p_rho = (1 + sum(x >= rho - 1e-12 for x in null)) / (1 + len(null)) if null else None
    acc = [c for c in valid if c["accepted"]]
    rej = [c for c in valid if not c["accepted"]]
    e2 = {"valid_calls": len(valid), "planned_calls": len(calls), "accepted_calls": len(acc),
          "focal_accuracy_all_valid": float(fa.mean()),
          "focal_accuracy_accepted": float(np.mean([c["focal_accuracy"] for c in acc])) if acc else None,
          "focal_accuracy_rejected": float(np.mean([c["focal_accuracy"] for c in rej])) if rej else None}
    # E3: windows = arm x block, frozen schedule order
    windows = {}
    for c in calls:
        windows.setdefault((c["arm"], c["block"]), []).append(c)
    e3 = []
    for (arm, block), seq in sorted(windows.items()):
        seq = sorted(seq, key=lambda c: c["position"])
        first = seq[0]
        chosen = next((c for c in seq if c["status"] == "complete" and c["accepted"]), None)
        used = (seq.index(chosen) + 1) if chosen else len(seq)
        ok = [c for c in seq if c["status"] == "complete"]
        vote = None
        if len(ok) == len(seq):
            ids = list(ok[0]["labels"])
            voted = {}
            for i in ids:
                counts = Counter(c["labels"][i] for c in ok)
                top = counts.most_common()
                voted[i] = ok[0]["labels"][i] if len(top) > 1 and top[0][1] == top[1][1] else top[0][0]
            vote = sum(voted[i] == truth[i] for i in ids) / len(ids)
        e3.append({"arm": arm, "block": block,
                   "first_call_accuracy": first.get("focal_accuracy"),
                   "screen_retry_accuracy": chosen["focal_accuracy"] if chosen else None,
                   "screen_retry_calls": used, "abstained": chosen is None, "majority_accuracy": vote})
    # E4: H versus C, exchanging calls within blocks
    def arm_calls(arm):
        return [c for c in valid if c["arm"] == arm]
    e4 = {}
    if arm_calls("H") and arm_calls("C"):
        by_block = {}
        for c in valid:
            if c["arm"] in {"H", "C"}:
                by_block.setdefault(c["block"], []).append(c)
        obs = np.mean([c["focal_accuracy"] for c in arm_calls("C")]) - np.mean([c["focal_accuracy"] for c in arm_calls("H")])
        layouts = []
        for b, cs in by_block.items():
            n_c = sum(c["arm"] == "C" for c in cs)
            layouts.append([set(x) for x in itertools.combinations(range(len(cs)), n_c)])
        blocks = list(by_block.values())
        stats = []
        for combo in itertools.product(*layouts):
            left, right = [], []
            for cs, chosen_idx in zip(blocks, combo):
                for k, c in enumerate(cs):
                    (left if k in chosen_idx else right).append(c["focal_accuracy"])
            stats.append(np.mean(left) - np.mean(right))
        stats = np.array(stats)
        e4 = {"accuracy_C": float(np.mean([c["focal_accuracy"] for c in arm_calls("C")])),
              "accuracy_H": float(np.mean([c["focal_accuracy"] for c in arm_calls("H")])),
              "difference_C_minus_H": float(obs),
              "randomization_p_two_sided": float(np.mean(np.abs(stats) >= abs(obs) - 1e-12)),
              "reassignments": len(stats),
              "rejection_rate_C": float(np.mean([not c["accepted"] for c in arm_calls("C")])),
              "rejection_rate_H": float(np.mean([not c["accepted"] for c in arm_calls("H")]))}
    result = {"tau": TAU, "E1": {"spearman_rho": None if np.isnan(rho) else float(rho), "p_one_sided": p_rho},
              "E2": e2, "E3": e3, "E4": e4,
              "calls": [{k: v for k, v in c.items() if k != "labels"} for c in calls]}
    out = BASE / "analysis"
    out.mkdir(exist_ok=True)
    (out / "scores.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ["E1", "E2", "E4"]}, indent=2))
    for w in e3:
        print(w)


if __name__ == "__main__":
    main()
