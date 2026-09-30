"""Error structure of LLM diagnostic calls and known-answer (canary) screening.

Offline analysis of frozen outputs from every study; no model calls.

1. Error-variance decomposition. For a set of k byte-identical calls on n focal
   cases, the n x k correctness matrix is split into a call component (whole
   responses better or worse than others), a stable case component (cases that
   are right or wrong in every call) and a residual, unstable component. The
   shares indicate which check can help: known-answer screening for call-level
   failures, repeat voting for unstable errors, and neither for stable errors.
2. Error stability. Each focal case is classified as stable correct, stable
   error (wrong in every call) or unstable. Unanimity screening can never remove
   stable errors, so their share bounds any consistency check.
3. Canary screening. Records with known labels that share the request with the
   focal records (the background panels) act as canaries. A call is accepted
   only if its canary accuracy reaches a threshold fixed in advance (0.8, the
   operating accuracy target). Screen-and-retry takes the first accepted call of
   each identical-input group in call order, else abstains.
"""
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/error_structure_v3"
TAU = 0.8  # canary accuracy threshold fixed before this analysis (operating target)


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- loaders
# Every loader returns call records:
#   {study, service, group, order, block, records: [(id, truth, pred, focal)]}
# ``group`` identifies byte-identical payloads; ``order`` is the call sequence
# used for screen-and-retry; ``block`` is the execution block (or None).
def tep_calls():
    rows = read_csv(ROOT / "results/independent_validation/analysis/cases.csv")
    focal = {r["id"] for r in rows if r["condition"] == "focal_only" and r["model"] == "gpt-5.6-sol" and r["repeat"] == "1"}
    calls = defaultdict(list)
    for r in rows:
        calls[(r["model"], r["condition"], int(r["repeat"]))].append(r)
    out = []
    for (model, cond, rep), rs in calls.items():
        has_background = cond in {"a_front", "a_back", "b_front", "b_back"}
        out.append({"study": "TEP", "service": model, "group": cond, "order": rep, "block": None,
                    "records": [(r["id"], r["true_label"], r["predicted"], (r["id"] in focal) or not has_background)
                                for r in rs]})
    return out


def abba_calls():
    base = ROOT / "results/context_blocked_v1"
    answers = load_json(base / "answers/records.json")
    truth = {r["id"]: r["label"] for r in answers}
    focal = {r["id"] for r in answers if r["focal"]}
    out = []
    for b, block in enumerate(load_json(base / "schedule.json")["blocks"], 1):
        for pos, name in enumerate(block, 1):
            preds = load_json(base / f"predictions/gpt-5.6-sol/{name}_results.json")
            out.append({"study": "TEP ABBA", "service": "gpt-5.6-sol", "group": name[-1], "order": 2 * b + pos,
                        "block": b, "records": [(p["id"], truth[p["id"]], p["predicted"], p["id"] in focal) for p in preds]})
    return out


def hydraulic_calls():
    v1 = ROOT / "results/hydraulic_validation_v1"
    comp = ROOT / "results/hydraulic_validation_v1_completion"
    panels = load_json(v1 / "answers/records.json")["panels"]
    truth = {r["id"]: r["true_label"] for p in panels.values() for r in p}
    focal = {r["id"] for r in panels["focal"]}
    out = []
    for rep, base in [(1, v1), (2, v1), (3, comp), (4, comp), (5, comp)]:
        for cond in ["clean", "clean_repeat", "pair_reverse", "descriptor_rotation", "a_front", "a_front_repeat",
                     "a_back", "b_front", "b_back"]:
            preds = load_json(base / f"predictions/gpt-5.6-sol/{cond}_r{rep}_results.json")
            prompt = (base / f"inference/prompts/{cond}_r{rep}.txt").read_bytes()
            out.append({"study": "Hydraulic (5 blocks)", "service": "gpt-5.6-sol",
                        "group": cond.replace("_repeat", ""), "payload": hashlib.sha256(prompt).hexdigest(),
                        "order": 2 * rep + cond.endswith("_repeat"), "block": rep,
                        "records": [(p["id"], truth[p["id"]], p["predicted"], p["id"] in focal) for p in preds]})
    return out


def policy_calls(study, label):
    rows = read_csv(ROOT / f"results/{study}/analysis/predictions.csv")
    reqs = {r["group"]: r for r in read_csv(ROOT / f"results/{study}/analysis/requests.csv")}
    calls = defaultdict(list)
    for r in rows:
        calls[r["group"]].append(r)
    out = []
    for group, rs in calls.items():
        first = rs[0]
        out.append({"study": label, "service": "gpt-5.6-sol", "group": f"{first['panel']}_{first['condition']}",
                    "order": (int(first["round"]), reqs[group]["started_at"]), "block": int(first["round"]),
                    "records": [(r["id"], r["truth"], r["predicted"], r["focal"] == "True") for r in rs]})
    return out


# --------------------------------------------------------------------------- analyses
def decompose(matrix):
    """Two-way decomposition of a cases x calls correctness matrix (sums of squares shares)."""
    y = np.asarray(matrix, dtype=float)
    n, k = y.shape
    m = y.mean()
    r, c = y.mean(axis=1), y.mean(axis=0)
    ss_case = k * np.sum((r - m) ** 2)
    ss_call = n * np.sum((c - m) ** 2)
    ss_res = np.sum((y - r[:, None] - c[None, :] + m) ** 2)
    total = ss_case + ss_call + ss_res
    if total == 0:
        return {"call": 0.0, "case": 0.0, "unstable": 0.0}
    return {"call": ss_call / total, "case": ss_case / total, "unstable": ss_res / total}


def stability(labels, truth):
    """Classify focal cases over repeated calls: stable correct / stable error / unstable."""
    counts = Counter()
    for case_labels, y in zip(labels, truth):
        correct = [l == y for l in case_labels]
        if all(correct):
            counts["stable_correct"] += 1
        elif not any(correct) and len(set(case_labels)) == 1:
            counts["stable_error_same"] += 1
        elif not any(correct):
            counts["stable_error_varied"] += 1
        else:
            counts["unstable"] += 1
    return counts


def group_summary(calls):
    """Per identical-input group: accuracy, decomposition, stability, voting and unanimity."""
    groups = defaultdict(list)
    for c in calls:
        groups[(c["study"], c["service"], c["group"])].append(c)
    rows = []
    for (study, service, group), cs in groups.items():
        if len(cs) < 2:
            continue
        ids = sorted({i for i, _, _, f in cs[0]["records"] if f})
        truth = {i: t for i, t, _, _ in cs[0]["records"]}
        preds = [{i: p for i, _, p, _ in c["records"]} for c in cs]
        labels = [[p[i] for p in preds] for i in ids]
        y = [truth[i] for i in ids]
        correct = np.array([[l == t for l in ls] for ls, t in zip(labels, y)], dtype=int)
        stab = stability(labels, y)
        n = len(ids)
        errors = n * len(cs) - correct.sum()
        vote = []
        for ls, t in zip(labels, y):
            top = Counter(ls).most_common()
            vote.append(top[0][0] == t if len(top) == 1 or top[0][1] > top[1][1] else None)
        unanimous = [len(set(ls)) == 1 for ls in labels]
        acc_unan = [ls[0] == t for ls, t, u in zip(labels, y, unanimous) if u]
        rows.append({"study": study, "service": service, "group": group, "n_cases": n, "calls": len(cs),
                     "accuracy": float(correct.mean()), "error_rate": float(1 - correct.mean()),
                     **{f"share_{k}": v for k, v in decompose(correct).items()},
                     "stable_correct": stab["stable_correct"], "stable_error": stab["stable_error_same"] + stab["stable_error_varied"],
                     "stable_error_same_label": stab["stable_error_same"], "unstable": stab["unstable"],
                     "majority_accuracy": float(np.mean([v is True for v in vote])),
                     "majority_ties": sum(v is None for v in vote),
                     "unanimity_coverage": float(np.mean(unanimous)),
                     "unanimity_accepted_error": float(1 - np.mean(acc_unan)) if acc_unan else None,
                     "total_errors": int(errors)})
    return rows


def canary_screen(calls, tau=TAU):
    """Per-call canary accuracy, screening and screen-and-retry for calls that contain canaries."""
    per_call, groups = [], defaultdict(list)
    for c in calls:
        focal = [(t, p) for _, t, p, f in c["records"] if f]
        canary = [(t, p) for _, t, p, f in c["records"] if not f]
        if not canary:
            continue
        modal = Counter(p for _, p in focal).most_common(1)[0][1] / len(focal)
        row = {"study": c["study"], "service": c["service"], "group": c["group"], "block": c["block"],
               "order": str(c["order"]), "n_focal": len(focal), "n_canary": len(canary),
               "focal_accuracy": float(np.mean([t == p for t, p in focal])),
               "canary_accuracy": float(np.mean([t == p for t, p in canary])),
               "focal_modal_share": modal}
        row["accepted"] = row["canary_accuracy"] >= tau
        per_call.append(row)
        groups[(c["study"], c["service"], c["group"])].append((c["order"], row))
    summary = defaultdict(lambda: defaultdict(float))
    by_study = defaultdict(list)
    for r in per_call:
        by_study[(r["study"], r["service"])].append(r)
    out = []
    for (study, service), rs in by_study.items():
        fa = np.array([r["focal_accuracy"] for r in rs])
        ca = np.array([r["canary_accuracy"] for r in rs])
        acc = [r for r in rs if r["accepted"]]
        rho, p = spearmanr(ca, fa) if len(set(ca)) > 1 and len(set(fa)) > 1 else (np.nan, np.nan)
        # screen-and-retry: first accepted call per identical-input group, up to three tries
        retry_acc, retry_calls, retry_abstain = [], [], 0
        for (s, sv, g), seq in groups.items():
            if (s, sv) != (study, service):
                continue
            seq = [r for _, r in sorted(seq, key=lambda x: str(x[0]))]
            for start in range(0, len(seq), 3):  # consecutive windows of up to three calls
                window = seq[start:start + 3]
                chosen = next((k for k, r in enumerate(window) if r["accepted"]), None)
                if chosen is None:
                    retry_abstain += 1
                    retry_calls.append(len(window))
                else:
                    retry_acc.append(window[chosen]["focal_accuracy"])
                    retry_calls.append(chosen + 1)
        out.append({"study": study, "service": service, "calls": len(rs),
                    "focal_accuracy_all_calls": float(fa.mean()),
                    "spearman_canary_focal": None if np.isnan(rho) else float(rho),
                    "spearman_p": None if np.isnan(p) else float(p),
                    "accepted_calls": len(acc),
                    "focal_accuracy_accepted": float(np.mean([r["focal_accuracy"] for r in acc])) if acc else None,
                    "focal_accuracy_rejected": float(np.mean([r["focal_accuracy"] for r in rs if not r["accepted"]])) if len(acc) < len(rs) else None,
                    "retry_decisions": len(retry_acc) + retry_abstain, "retry_abstentions": retry_abstain,
                    "retry_accuracy_of_delivered": float(np.mean(retry_acc)) if retry_acc else None,
                    "retry_mean_calls": float(np.mean(retry_calls)) if retry_calls else None})
    return per_call, out


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    calls = (tep_calls() + abba_calls() + hydraulic_calls() + policy_calls("policy_hydraulic_v1", "Hydraulic (new blocks)")
             + policy_calls("policy_paderborn_v1", "Paderborn bearing"))
    groups = group_summary(calls)
    per_call, canary = canary_screen(calls)
    write_csv(OUT / "groups.csv", groups)
    write_csv(OUT / "canary_calls.csv", per_call)
    write_csv(OUT / "canary_summary.csv", canary)
    (OUT / "summary.json").write_text(json.dumps({"tau": TAU, "groups": groups, "canary": canary}, indent=2) + "\n")
    for g in groups:
        print(f"{g['study'][:20]:<20} {g['service'][4:12]:<8} {g['group'][:16]:<16} n={g['n_cases']:>2} k={g['calls']} "
              f"acc={100*g['accuracy']:5.1f} call={g['share_call']:.2f} case={g['share_case']:.2f} unst={g['share_unstable']:.2f} "
              f"SC={g['stable_correct']:>2} SE={g['stable_error']:>2} U={g['unstable']:>2} vote={100*g['majority_accuracy']:5.1f}")
    print()
    for c in canary:
        print(json.dumps(c))


if __name__ == "__main__":
    main()
