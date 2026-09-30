"""
Compute metrics for v2 experiments with bootstrap confidence intervals
and supervised baselines (SVM, Random Forest, 1D-CNN placeholder).
Generates LaTeX tables and updated paper figures.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

ROOT = Path(__file__).parent.parent.parent
V2_DIR = ROOT / "results" / "batches_v2"
CLEAN_V2 = ROOT / "results" / "batches_clean_v2"
TABLES = ROOT / "results" / "tables_v2"
TABLES.mkdir(parents=True, exist_ok=True)

STRATEGIES = ["zero_shot", "few_shot", "cot", "domain"]
STRATEGY_LABELS = ["ZS", "FS", "CoT", "DK"]
MODELS = {
    "haiku":  "Haiku 4.5",
    "sonnet": "Sonnet 5",
    "opus":   "Opus 4.6",
}

DATASETS = {
    "cwru":   {"n": 100, "classes": 10, "random_baseline": 10.0},
    "cmapss": {"n": 60,  "classes": 3,  "random_baseline": 33.3},
    "tep":    {"n": 42,  "classes": 7,  "random_baseline": 14.3},
}


# ── bootstrap CI ─────────────────────────────────────────────────────────────

def bootstrap_ci(correct: np.ndarray, n_boot: int = 2000, alpha: float = 0.95) -> tuple[float, float]:
    """Return (lower, upper) percentile bootstrap CI for accuracy."""
    rng = np.random.default_rng(42)
    acc = correct.mean()
    boot = np.array([rng.choice(correct, size=len(correct), replace=True).mean()
                     for _ in range(n_boot)])
    lo = np.percentile(boot, 100 * (1 - alpha) / 2)
    hi = np.percentile(boot, 100 * (1 + alpha) / 2)
    return float(lo * 100), float(hi * 100)


# ── load results ─────────────────────────────────────────────────────────────

def load_results(dataset: str, strategy: str, n: int, model_key: str) -> list[dict] | None:
    path = V2_DIR / f"{dataset}_{strategy}_v2_{n}_{model_key}.json"
    if not path.exists():
        return None
    data = json.load(open(path))
    return [d for d in data if d is not None]


def score(records: list[dict]) -> dict:
    correct = np.array([int(r["predicted"] == r["true_label"]) for r in records])
    acc = float(correct.mean() * 100)
    lo, hi = bootstrap_ci(correct)
    preds = [r["predicted"] for r in records]
    trues = [r["true_label"] for r in records]
    labels = sorted(set(trues))
    f1 = float(f1_score(trues, preds, labels=labels, average="macro",
                        zero_division=0) * 100)
    return {"acc": acc, "f1": f1, "ci_lo": lo, "ci_hi": hi, "n": len(records)}


# ── supervised baselines ─────────────────────────────────────────────────────

def _get_features_labels(dataset: str) -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    if dataset == "cwru":
        df = pd.read_csv(ROOT / "data" / "cwru" / "cwru_processed.csv")
        label_col = "label"
        feature_cols = [c for c in df.columns if c.startswith("feature_") or
                        c in ("rms", "kurtosis", "peak", "crest_factor", "skewness",
                              "std", "mean", "spectral_centroid", "spectral_spread",
                              "peak_freq_1", "peak_freq_2", "peak_freq_3")]
        if not feature_cols:
            feature_cols = [c for c in df.columns if df[c].dtype in (float, int)
                            and c not in ("fault_id",) and "label" not in c.lower()
                            and "source" not in c.lower()]
    elif dataset == "cmapss":
        df = pd.read_csv(ROOT / "data" / "cmapss" / "cmapss_processed.csv")
        label_col = "health_stage"
        feature_cols = [c for c in df.columns if df[c].dtype in (float, int)
                        and "label" not in c.lower() and "health" not in c.lower()
                        and "source" not in c.lower() and "rul" not in c.lower()]
    elif dataset == "tep":
        df = pd.read_csv(ROOT / "data" / "tep" / "tep_processed.csv")
        label_col = "label"
        feature_cols = [c for c in df.columns if c.startswith("var_")]
    else:
        raise ValueError(dataset)

    le = LabelEncoder()
    y = le.fit_transform(df[label_col])
    X = df[feature_cols].fillna(0).values
    return X, y, le


def run_supervised_baselines(dataset: str) -> dict:
    """Run SVM + RF on full data with 5-fold cross-validation."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler

    X, y, le = _get_features_labels(dataset)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {"SVM": [], "Random Forest": []}

    for train_idx, test_idx in cv.split(X, y):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        Xtr_s = scaler.fit_transform(Xtr)
        Xte_s = scaler.transform(Xte)

        svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
        svm.fit(Xtr_s, ytr)
        results["SVM"].append(accuracy_score(yte, svm.predict(Xte_s)) * 100)

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(Xtr, ytr)
        results["Random Forest"].append(accuracy_score(yte, rf.predict(Xte)) * 100)

    return {k: {"acc": float(np.mean(v)), "std": float(np.std(v))}
            for k, v in results.items()}


# ── latex table ──────────────────────────────────────────────────────────────

def bold_max(values: list[float]) -> list[str]:
    mx = max(values)
    return [f"\\textbf{{{v:.1f}}}" if v == mx else f"{v:.1f}" for v in values]


def write_latex_table(path: Path, rows: list[tuple]) -> None:
    """rows: (model_name, [acc_per_strategy], mean_acc, ci_lo, ci_hi)"""
    lines = []
    for name, accs, mean, lo, hi in rows:
        b = bold_max(accs)
        lines.append(
            f"{name} & {' & '.join(b)} & {mean:.1f} & [{lo:.1f},{hi:.1f}] \\\\"
        )
    path.write_text("\n".join(lines))


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    all_scores = {}

    print("Computing v2 metrics …\n")
    for dataset, info in DATASETS.items():
        n = info["n"]
        all_scores[dataset] = {}
        print(f"  {dataset.upper()} (n={n}, {info['classes']}-class)")

        for model_key, model_name in MODELS.items():
            per_strat = []
            last_records = None
            for strategy in STRATEGIES:
                recs = load_results(dataset, strategy, n, model_key)
                if recs is None:
                    per_strat.append(None)
                else:
                    s = score(recs)
                    per_strat.append(s)
                    last_records = recs
            all_scores[dataset][model_key] = per_strat

            valid = [s for s in per_strat if s is not None]
            if valid:
                mean = np.mean([s["acc"] for s in valid])
                # Combined bootstrap CI using last strategy's records as proxy
                if last_records:
                    correct = np.array([int(r["predicted"] == r["true_label"])
                                        for r in last_records])
                    lo, hi = bootstrap_ci(correct)
                else:
                    lo, hi = mean - 5, mean + 5
                accs = [s["acc"] if s else 0.0 for s in per_strat]
                print(f"    {model_name}: {' / '.join(f'{a:.1f}' for a in accs)} → avg {mean:.1f}% CI [{lo:.1f},{hi:.1f}]")

    # Supervised baselines
    print("\n  Supervised baselines (5-fold CV) …")
    baselines = {}
    for dataset in DATASETS:
        try:
            bl = run_supervised_baselines(dataset)
            baselines[dataset] = bl
            for name, res in bl.items():
                print(f"    {dataset} {name}: {res['acc']:.1f}% ± {res['std']:.1f}%")
        except Exception as e:
            print(f"    {dataset}: baseline error — {e}")
            baselines[dataset] = {}

    # Write tables
    for dataset, info in DATASETS.items():
        n = info["n"]
        rows = []
        for model_key, model_name in MODELS.items():
            per_strat = all_scores[dataset][model_key]
            accs = [s["acc"] if s else 0.0 for s in per_strat]
            mean = np.mean([a for a in accs if a > 0]) if any(a > 0 for a in accs) else 0.0
            # Use average of per-strategy CIs
            los = [s["ci_lo"] for s in per_strat if s]
            his = [s["ci_hi"] for s in per_strat if s]
            lo = float(np.mean(los)) if los else mean - 5
            hi = float(np.mean(his)) if his else mean + 5
            if any(a > 0 for a in accs):
                rows.append((model_name, accs, mean, lo, hi))

        for bl_name, bl_res in baselines.get(dataset, {}).items():
            acc = bl_res["acc"]
            std = bl_res["std"]
            rows.append((bl_name, [acc] * 4, acc, acc - std, acc + std))

        table_path = TABLES / f"{dataset}_v2.tex"
        write_latex_table(table_path, rows)
        print(f"  ✓ Wrote {table_path.name}")

    # Summary CSV
    rows_csv = []
    for dataset in DATASETS:
        for model_key, model_name in MODELS.items():
            for i, strategy in enumerate(STRATEGIES):
                s = all_scores[dataset][model_key][i]
                if s:
                    rows_csv.append({
                        "dataset": dataset, "model": model_name,
                        "strategy": strategy, "acc": s["acc"], "f1": s["f1"],
                        "ci_lo": s["ci_lo"], "ci_hi": s["ci_hi"],
                    })
    pd.DataFrame(rows_csv).to_csv(TABLES / "all_v2_metrics.csv", index=False)
    print(f"\n  ✓ Summary CSV: {TABLES / 'all_v2_metrics.csv'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
