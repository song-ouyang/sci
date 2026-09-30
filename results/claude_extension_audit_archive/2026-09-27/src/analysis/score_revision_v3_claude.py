"""Score all frozen Claude-extension revision-v3 sessions.

Run ONLY after all sessions are frozen (all _results.json and _metadata.json
files present, no session still in progress). Opens answer keys exactly once,
after prediction files are validated.

Usage:
  python score_revision_v3_claude.py [--validate-only]
"""
import argparse
import csv
import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EXT_DIR = ROOT / "results/revision_v3_claude_extension"
DATA_PATH = ROOT / "results/revision_v3/data/datasets.json"
BATCHES_PATH = ROOT / "results/revision_v3/inference/batches.json"

MODELS = ["claude-sonnet-4-6", "claude-opus-5-5"]
LABELS = {
    "cmapss": ["Healthy", "Degrading", "Critical"],
    "tep": ["Normal", "Fault_1", "Fault_2", "Fault_4", "Fault_6", "Fault_7", "Fault_13"],
}
GROUP_NAMES = [
    f"{ds}_{cond}_r{r}"
    for ds, conds in [("cmapss", ["clean"]), ("tep", ["clean", "misnamed"])]
    for cond in conds
    for r in [1, 2, 3]
]
BOOTSTRAP_SEED = 20260926
BOOTSTRAP_N = 10_000


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metric_values(truth, predicted, labels):
    truth = np.array(truth)
    predicted = np.array(predicted)
    n = len(truth)
    correct = int(np.sum(truth == predicted))
    accuracy = correct / n
    per_class = {}
    for label in labels:
        tp = int(np.sum((truth == label) & (predicted == label)))
        fp = int(np.sum((truth != label) & (predicted == label)))
        fn = int(np.sum((truth == label) & (predicted != label)))
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[label] = {"support": int(np.sum(truth == label)), "tp": tp, "fp": fp, "fn": fn,
                            "precision": p, "recall": r, "f1": f1}
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(labels)
    confusion = {label: {other: int(np.sum((truth == label) & (predicted == other)))
                         for other in labels} for label in labels}
    return {
        "n": n, "correct": correct, "accuracy": accuracy,
        "macro_f1": macro_f1, "per_class": per_class, "confusion": confusion,
    }


def bootstrap_ci(truth, predicted, labels, n=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    rng = np.random.default_rng(seed)
    truth = np.array(truth)
    predicted = np.array(predicted)
    idx = rng.integers(0, len(truth), size=(n, len(truth)))
    acc = np.mean(truth[idx] == predicted[idx], axis=1)
    macro_f1 = np.zeros(n)
    for label in labels:
        t_l = (truth == label)[idx]
        p_l = (predicted == label)[idx]
        tp = np.sum(t_l & p_l, axis=1).astype(float)
        denom = 2 * tp + np.sum(~t_l & p_l, axis=1) + np.sum(t_l & ~p_l, axis=1)
        macro_f1 += np.where(denom > 0, 2 * tp / denom, 0.0) / len(labels)
    return {
        "accuracy_95ci": [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))],
        "macro_f1_95ci": [float(np.percentile(macro_f1, 2.5)), float(np.percentile(macro_f1, 97.5))],
    }


def load_truth_by_id(dataset_name: str, datasets: dict) -> dict:
    """Returns {sample_id: true_label} for the given dataset."""
    ds_entry = datasets[dataset_name]
    rows = ds_entry["test"] if isinstance(ds_entry, dict) else ds_entry
    return {row["id"]: row["label"] for row in rows}


def validate_predictions(preds_dir: Path, batches: dict) -> dict:
    """Pre-label validation: checks all expected result files exist and are internally valid."""
    issues = {}
    for model in MODELS:
        for gname in GROUP_NAMES:
            results_path = preds_dir / model / f"{gname}_results.json"
            meta_path = preds_dir / model / f"{gname}_metadata.json"
            if not results_path.exists():
                meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
                issues[f"{model}/{gname}"] = meta.get("validation_error", "results file missing")
                continue
            preds = json.loads(results_path.read_text())
            group = batches["groups"][gname]
            if len(preds) != len(group["samples"]):
                issues[f"{model}/{gname}"] = f"count mismatch: {len(preds)} vs {len(group['samples'])}"
            for p, s in zip(preds, group["samples"]):
                if p["id"] != s["id"]:
                    issues[f"{model}/{gname}"] = f"ID mismatch: {p['id']} vs {s['id']}"
                    break
    return issues


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-only", action="store_true",
                        help="Check prediction files without opening answer keys")
    args = parser.parse_args()

    preds_dir = EXT_DIR / "predictions"
    batches = json.loads(BATCHES_PATH.read_text())

    print("=== Pre-label validation ===")
    issues = validate_predictions(preds_dir, batches)
    if issues:
        print(f"ISSUES ({len(issues)}):")
        for k, v in issues.items():
            print(f"  {k}: {v}")
    else:
        print("All prediction files valid.")

    if args.validate_only:
        print("\n--validate-only: stopping before opening answer keys.")
        return

    if issues:
        print(f"\n{len(issues)} sessions have issues; they will be recorded as N/A in scores.")

    print("\n=== Opening answer keys ===")
    datasets = json.loads(DATA_PATH.read_text())
    data_hash = sha256_file(DATA_PATH)
    print(f"data/datasets.json sha256: {data_hash}")

    datasets_inner = datasets.get("datasets", datasets)  # handle nested structure
    truth_by_dataset = {}
    for ds in ["cmapss", "tep"]:
        truth_by_dataset[ds] = load_truth_by_id(ds, datasets_inner)
        print(f"  {ds}: {len(truth_by_dataset[ds])} labelled samples")

    print("\n=== Scoring ===")
    out_dir = EXT_DIR / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_scores = {}
    for model in MODELS:
        all_scores[model] = {}
        for gname in GROUP_NAMES:
            group = batches["groups"][gname]
            dataset = group["dataset"]
            results_path = preds_dir / model / f"{gname}_results.json"
            if gname in {k.split("/")[1] for k in issues if k.startswith(model + "/")} \
                    or not results_path.exists():
                all_scores[model][gname] = {"status": "N/A"}
                continue
            preds = json.loads(results_path.read_text())
            truth_map = truth_by_dataset[dataset]
            truth = [truth_map[p["id"]] for p in preds]
            predicted = [p["predicted"] for p in preds]
            labels = LABELS[dataset]
            m = metric_values(truth, predicted, labels)
            ci = bootstrap_ci(truth, predicted, labels)
            all_scores[model][gname] = {"status": "complete", **m, **ci}
            print(f"  {model}/{gname}: acc={m['accuracy']:.3f} macro_f1={m['macro_f1']:.3f}")

    scores_path = out_dir / "scores.json"
    scores_path.write_text(json.dumps({
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_n": BOOTSTRAP_N,
        "data_sha256": data_hash,
        "scores": all_scores,
    }, indent=2) + "\n")
    print(f"\nSaved: {scores_path}")

    write_summary(out_dir, all_scores)
    write_csv(out_dir, all_scores)


def write_summary(out_dir: Path, all_scores: dict):
    lines = ["# Claude Extension Revision-v3 Scores\n"]
    for model in MODELS:
        alias = "Sonnet" if "sonnet" in model else "Opus"
        lines.append(f"## {alias} ({model})\n")
        lines.append(f"| Group | Accuracy | Macro F1 | 95% CI (acc) |\n")
        lines.append(f"|-------|----------|----------|--------------|\n")
        for gname in GROUP_NAMES:
            s = all_scores[model].get(gname, {})
            if s.get("status") == "N/A":
                lines.append(f"| {gname} | N/A | N/A | N/A |\n")
            else:
                ci = s.get("accuracy_95ci", [None, None])
                lines.append(
                    f"| {gname} | {s['accuracy']:.3f} | {s['macro_f1']:.3f} | "
                    f"[{ci[0]:.3f}, {ci[1]:.3f}] |\n"
                )
        lines.append("\n")
    summary_path = out_dir / "summary.md"
    summary_path.write_text("".join(lines))
    print(f"Saved: {summary_path}")


def write_csv(out_dir: Path, all_scores: dict):
    rows = []
    for model in MODELS:
        for gname in GROUP_NAMES:
            s = all_scores[model].get(gname, {})
            group_parts = gname.rsplit("_r", 1)
            condition = group_parts[0]
            repeat = group_parts[1] if len(group_parts) > 1 else "?"
            rows.append({
                "model": model,
                "group": gname,
                "condition": condition,
                "repeat": repeat,
                "accuracy": s.get("accuracy", ""),
                "macro_f1": s.get("macro_f1", ""),
                "acc_ci_lo": s.get("accuracy_95ci", ["", ""])[0],
                "acc_ci_hi": s.get("accuracy_95ci", ["", ""])[1],
                "status": s.get("status", "complete"),
            })
    csv_path = out_dir / "scores.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
