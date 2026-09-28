#!/usr/bin/env python3
"""Score claude_conversation_exploratory_v1 results after all sessions frozen."""
import json, pathlib, sys
from collections import defaultdict

ROOT = pathlib.Path(__file__).parents[2]
EXP  = ROOT / "results/claude_conversation_exploratory_v1"
BATCHES = ROOT / "results/revision_v3/inference/batches.json"
DATASETS = ROOT / "results/revision_v3/data/datasets.json"

def load_gt():
    d = json.loads(DATASETS.read_text())
    gt = {}
    for ds in d["datasets"].values():
        for s in ds["test"]:
            gt[s["id"]] = s["label"]
    return gt

def macro_f1(y_true, y_pred, classes):
    from collections import Counter
    tp = defaultdict(int); fp = defaultdict(int); fn = defaultdict(int)
    for t, p in zip(y_true, y_pred):
        if t == p: tp[t] += 1
        else:      fp[p] += 1; fn[t] += 1
    f1s = []
    for c in classes:
        prec = tp[c]/(tp[c]+fp[c]) if (tp[c]+fp[c]) else 0
        rec  = tp[c]/(tp[c]+fn[c]) if (tp[c]+fn[c]) else 0
        f1s.append(2*prec*rec/(prec+rec) if (prec+rec) else 0)
    return sum(f1s)/len(f1s) if f1s else 0.0

def score_session(group_name, predictions_file, gt, batches):
    grp = batches["groups"][group_name]
    ids = [s["id"] for s in grp["samples"]]
    valid = grp["valid_labels"]

    try:
        preds = json.loads(predictions_file.read_text())
    except Exception as e:
        return {"status": "failure", "reason": f"cannot parse: {e}"}

    if len(preds) != len(ids):
        return {"status": "failure", "reason": f"length mismatch: got {len(preds)}, expected {len(ids)}"}

    pred_map = {}
    for p in preds:
        if not isinstance(p, dict) or "id" not in p or "predicted" not in p:
            return {"status": "failure", "reason": "bad prediction object"}
        if p["predicted"] not in valid:
            return {"status": "failure", "reason": f"invalid label: {p['predicted']}"}
        pred_map[p["id"]] = p["predicted"]

    for i in ids:
        if i not in pred_map:
            return {"status": "failure", "reason": f"missing id: {i}"}

    y_true = [gt[i] for i in ids]
    y_pred = [pred_map[i] for i in ids]
    acc = sum(t==p for t,p in zip(y_true,y_pred)) / len(ids)
    mf1 = macro_f1(y_true, y_pred, valid)
    return {"status": "success", "accuracy": round(acc*100,1), "macro_f1": round(mf1*100,1),
            "n": len(ids), "correct": sum(t==p for t,p in zip(y_true,y_pred))}

def main():
    gt = load_gt()
    batches = json.loads(BATCHES.read_text())

    results = {}
    for model in ["sonnet", "opus"]:
        pred_dir = EXP / "predictions" / model
        for cond in ["cmapss_clean", "tep_clean", "tep_misnamed"]:
            for r in [1, 2, 3]:
                key = f"{model}/{cond}_r{r}"
                fname = pred_dir / f"{cond}_r{r}_predictions.json"
                grp_key = f"{cond}_r{r}"
                if fname.exists():
                    res = score_session(grp_key, fname, gt, batches)
                else:
                    res = {"status": "not_run"}
                results[key] = res

    # Print summary
    print(f"{'Session':<30} {'Status':<10} {'Acc %':<8} {'F1 %':<8}")
    print("-"*58)
    for k, v in results.items():
        acc = f"{v['accuracy']:.1f}" if v.get("accuracy") is not None else "N/A"
        f1  = f"{v['macro_f1']:.1f}"  if v.get("macro_f1")  is not None else "N/A"
        print(f"{k:<30} {v['status']:<10} {acc:<8} {f1:<8}")

    # Per model/condition summary
    print("\n--- Mean ± sample SD (ddof=1) ---")
    for model in ["sonnet", "opus"]:
        for cond in ["cmapss_clean", "tep_clean", "tep_misnamed"]:
            accs = []
            f1s  = []
            for r in [1,2,3]:
                key = f"{model}/{cond}_r{r}"
                v = results[key]
                if v.get("accuracy") is not None:
                    accs.append(v["accuracy"]); f1s.append(v["macro_f1"])
            if len(accs) == 3:
                import statistics
                print(f"{model:7} {cond:15} Acc {statistics.mean(accs):.1f}±{statistics.stdev(accs):.1f}  F1 {statistics.mean(f1s):.1f}±{statistics.stdev(f1s):.1f}")
            else:
                print(f"{model:7} {cond:15} only {len(accs)}/3 sessions complete")

    # Save JSON
    out = EXP / "scores.json"
    out.write_text(json.dumps({"sessions": results}, indent=2))
    print(f"\nSaved: {out}")

if __name__ == "__main__":
    main()
