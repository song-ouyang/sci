"""
Compute paper metrics from Claude stratified batches and frozen GPT results.
Generates figures, CSV metrics, and reusable LaTeX tables for the paper.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict
if __package__:
    from .score_gpt_models import load_batches, validate_model, check_manifest
else:
    from score_gpt_models import load_batches, validate_model, check_manifest
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

BATCHES = Path(__file__).parent.parent.parent / "results" / "batches"
CLEAN = Path(__file__).parent.parent.parent / "results" / "batches_clean"
FIGURES = Path(__file__).parent.parent.parent / "results" / "figures"
TABLES = Path(__file__).parent.parent.parent / "results" / "tables"
ROOT = Path(__file__).resolve().parents[2]
GPT_RESULTS = ROOT / "results" / "gpt_models"

STRATEGIES = ["zero_shot", "few_shot", "cot", "domain"]
STRATEGY_LABELS = ["Zero-shot", "Few-shot", "CoT", "Domain"]

CWRU_CLASSES = ["Normal", "IR007", "IR014", "IR021", "B007", "B014", "B021", "OR007", "OR014", "OR021"]
CWRU_COARSE = {"Normal": "Normal", "IR007": "IR", "IR014": "IR", "IR021": "IR",
               "B007": "B", "B014": "B", "B021": "B", "OR007": "OR", "OR014": "OR", "OR021": "OR"}
CWRU_COARSE_CLASSES = ["Normal", "IR", "B", "OR"]

MODELS_CWRU = {
    "Opus 4.6": "_opus_s.json",
    "Sonnet 5": "_sonnet_s.json",
    "Haiku 4.5": "_haiku_s.json",
}

MODELS_CROSS = {
    "Opus 4.6": "_opus_s.json",
    "Sonnet 5": "_sonnet_s.json",
    "Haiku 4.5": "_haiku_s.json",
    "GPT-5.5": "gpt-5.5",
    "GPT-5.6 Sol": "gpt-5.6-sol",
    "GPT-6 Astra": "gpt-6-astra",
}


def load_and_merge(dataset, strategy, suffix):
    if suffix.startswith("gpt-"):
        results_file = GPT_RESULTS / suffix / f"{dataset}_{strategy}_results.json"
    else:
        results_file = BATCHES / f"{dataset}_{strategy}_results{suffix}"
    key_name = "strat50" if dataset == "cwru" else "strat30" if dataset == "cmapss" else "strat21"
    key_file = CLEAN / f"{dataset}_{strategy}_{key_name}_key.json"

    if not results_file.exists():
        return None
    if not key_file.exists():
        return None

    results = json.load(open(results_file))
    key = json.load(open(key_file))
    key_dict = {k["id"]: k["true_label"] for k in key}

    merged = []
    for r in results:
        true = key_dict.get(r["id"], None)
        if true is None:
            continue
        pred = r.get("predicted", "UNPARSEABLE")
        merged.append({"true_label": true, "predicted": pred})

    return merged


def compute_metrics(data):
    true = [d["true_label"] for d in data]
    pred = [d["predicted"] for d in data]
    valid = [(t, p) for t, p in zip(true, pred) if p != "UNPARSEABLE"]
    if not valid:
        return None
    t, p = zip(*valid)
    return {
        "accuracy": accuracy_score(t, p),
        "f1_macro": f1_score(t, p, average="macro", zero_division=0),
        "f1_weighted": f1_score(t, p, average="weighted", zero_division=0),
        "precision_macro": precision_score(t, p, average="macro", zero_division=0),
        "recall_macro": recall_score(t, p, average="macro", zero_division=0),
        "n_samples": len(data),
        "n_valid": len(valid),
        "n_correct": sum(1 for a, b in zip(t, p) if a == b),
    }


def generate_summary():
    batches, input_digest = load_batches(ROOT / "gpt_experiment_batches.json")
    for model_id in (value for value in MODELS_CROSS.values() if value.startswith("gpt-")):
        _, snapshot = validate_model(model_id, GPT_RESULTS, batches, input_digest)
        check_manifest(model_id, GPT_RESULTS, snapshot)
    rows = []
    for model, suffix in MODELS_CWRU.items():
        for st in STRATEGIES:
            data = load_and_merge("cwru", st, suffix)
            if data:
                m = compute_metrics(data)
                if m:
                    rows.append({"Dataset": "CWRU", "Model": model, "Strategy": st, **m})
                    print(f"  CWRU {model:12s} {st:12s}: {m['n_correct']}/{m['n_samples']} = {m['accuracy']*100:.1f}%")

    # Coarse-grained CWRU (4-class: Normal/IR/B/OR)
    for model, suffix in MODELS_CWRU.items():
        for st in STRATEGIES:
            data = load_and_merge("cwru", st, suffix)
            if data:
                coarse = [{"true_label": CWRU_COARSE.get(d["true_label"], d["true_label"]),
                           "predicted": CWRU_COARSE.get(d["predicted"], d["predicted"])} for d in data]
                m = compute_metrics(coarse)
                if m:
                    rows.append({"Dataset": "CWRU-4", "Model": model, "Strategy": st, **m})
                    print(f"  CWRU-4 {model:12s} {st:12s}: {m['n_correct']}/{m['n_samples']} = {m['accuracy']*100:.1f}%")

    for ds in ["cmapss", "tep"]:
        for model, suffix in MODELS_CROSS.items():
            for st in STRATEGIES:
                data = load_and_merge(ds, st, suffix)
                if data:
                    m = compute_metrics(data)
                    if m:
                        rows.append({"Dataset": ds.upper(), "Model": model, "Strategy": st, **m})
                        print(f"  {ds.upper():6s} {model:12s} {st:12s}: {m['n_correct']}/{m['n_samples']} = {m['accuracy']*100:.1f}%")

    df = pd.DataFrame(rows)
    if len(df) > 0:
        TABLES.mkdir(parents=True, exist_ok=True)
        df.to_csv(TABLES / "all_metrics_final.csv", index=False)
    return df


def plot_cwru_heatmap(df):
    FIGURES.mkdir(parents=True, exist_ok=True)
    cwru = df[df["Dataset"] == "CWRU"]
    if cwru.empty:
        return
    pivot = cwru.pivot_table(index="Model", columns="Strategy", values="accuracy")
    pivot = pivot.reindex(columns=STRATEGIES)
    pivot.columns = STRATEGY_LABELS

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="YlOrRd",
                vmin=0, vmax=100, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Accuracy (%)"})
    ax.set_title("CWRU 10-Class Bearing Fault Diagnosis Accuracy (%)\nModel × Prompt Strategy", fontsize=12)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(FIGURES / "cwru_heatmap_final.png", dpi=300)
    plt.close()
    print("  Heatmap saved")


def plot_cwru_bar_chart(df):
    FIGURES.mkdir(parents=True, exist_ok=True)
    cwru = df[df["Dataset"] == "CWRU"]
    if cwru.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(STRATEGIES))
    width = 0.25
    models = list(MODELS_CWRU.keys())
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, model in enumerate(models):
        md = cwru[cwru["Model"] == model]
        accs = []
        for st in STRATEGIES:
            row = md[md["Strategy"] == st]
            accs.append(row["accuracy"].values[0] * 100 if len(row) > 0 else 0)
        bars = ax.bar(x + i * width, accs, width, label=model, color=colors[i], edgecolor="white")
        for bar, acc in zip(bars, accs):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{acc:.0f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xlabel("Prompt Strategy", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("CWRU 10-Class Bearing Fault Diagnosis:\nModel Capability × Prompt Strategy", fontsize=13)
    ax.set_xticks(x + width)
    ax.set_xticklabels(STRATEGY_LABELS)
    ax.set_ylim(0, 115)
    ax.legend(title="Model", loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "cwru_bar_chart_final.png", dpi=300)
    plt.close()
    print("  Bar chart saved")


def plot_cwru_coarse_heatmap(df):
    FIGURES.mkdir(parents=True, exist_ok=True)
    cwru4 = df[df["Dataset"] == "CWRU-4"]
    if cwru4.empty:
        return
    pivot = cwru4.pivot_table(index="Model", columns="Strategy", values="accuracy")
    pivot = pivot.reindex(columns=STRATEGIES)
    pivot.columns = STRATEGY_LABELS

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="YlOrRd",
                vmin=0, vmax=100, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Accuracy (%)"})
    ax.set_title("CWRU 4-Class Fault Type Accuracy (%)\n(Normal / Inner Race / Ball / Outer Race)", fontsize=12)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(FIGURES / "cwru_coarse_heatmap.png", dpi=300)
    plt.close()
    print("  Coarse heatmap saved")


def plot_confusion_matrices():
    FIGURES.mkdir(parents=True, exist_ok=True)
    interesting = [
        ("Haiku 4.5", "zero_shot", "_haiku_s.json"),
        ("Haiku 4.5", "few_shot", "_haiku_s.json"),
        ("Haiku 4.5", "domain", "_haiku_s.json"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (model, st, suffix) in enumerate(interesting):
        data = load_and_merge("cwru", st, suffix)
        if not data:
            continue
        true = [d["true_label"] for d in data]
        pred = [d["predicted"] for d in data]
        cm = confusion_matrix(true, pred, labels=CWRU_CLASSES)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=CWRU_CLASSES, yticklabels=CWRU_CLASSES)
        acc = accuracy_score(true, pred)
        axes[idx].set_title(f"{model} - {STRATEGY_LABELS[STRATEGIES.index(st)]}\nAcc: {acc*100:.0f}%", fontsize=11)
        axes[idx].set_ylabel("True Label" if idx == 0 else "")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].tick_params(axis='y', rotation=0)

    plt.suptitle("Confusion Matrices: Haiku 4.5 on CWRU 10-Class", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "confusion_matrices_final.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("  Confusion matrices saved")


def plot_cross_dataset(df):
    FIGURES.mkdir(parents=True, exist_ok=True)
    datasets_plot = [
        ("CMAPSS", "C-MAPSS (3)*\nChance: 33.3%"),
        ("TEP", "TEP (7)*\nChance: 14.3%"),
        ("CWRU-4", "CWRU (4)\nChance: 25.0%"),
        ("CWRU", "CWRU (10)\nChance: 10.0%"),
    ]
    models = list(MODELS_CROSS.keys())
    values = df.pivot_table(index="Model", columns="Dataset", values="accuracy")
    values = values.reindex(index=models, columns=[key for key, _ in datasets_plot]) * 100
    fig, ax = plt.subplots(figsize=(9, 6.1))
    ax.set_facecolor("#eeeeee")
    sns.heatmap(values, annot=True, fmt=".1f", mask=values.isna(),
                cmap="Blues", vmin=0, vmax=100, linewidths=1, linecolor="white",
                cbar_kws={"label": "Mean accuracy (%)"}, ax=ax)
    for row in range(len(models)):
        for col in range(len(datasets_plot)):
            if pd.isna(values.iloc[row, col]):
                ax.text(col + 0.5, row + 0.5, "Not evaluated",
                        ha="center", va="center", fontsize=9, color="#555555")
    ax.hlines(3, *ax.get_xlim(), colors="white", linewidth=3)
    ax.set_xticklabels([label for _, label in datasets_plot], rotation=0)
    ax.set_yticklabels(models, rotation=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("Archived prompt benchmark: mean accuracy", fontsize=13, pad=14)
    fig.text(0.02, 0.035, "* TEP inputs contain metadata and shifted sensor labels.\n"
             "  GPT inputs for both datasets expose class-correlated IDs; these are not clean sensor-only scores.",
             fontsize=8, color="#555555", va="bottom")
    plt.tight_layout(rect=(0, 0.095, 1, 1))
    fig.savefig(FIGURES / "cross_dataset_final.png", dpi=300)
    plt.close()
    print("  Cross-dataset chart saved")


def write_paper_tables(df):
    """Write table bodies from unrounded per-experiment metrics."""
    TABLES.mkdir(parents=True, exist_ok=True)
    for dataset, filename in [("CMAPSS", "cmapss_comparison.tex"), ("TEP", "tep_comparison.tex")]:
        lines = ["% Generated by src/analysis/compute_metrics_final.py; do not edit values manually."]
        for index, model in enumerate(MODELS_CROSS):
            if index == 3:
                lines.append(r"\midrule")
            rows = df[(df["Dataset"] == dataset) & (df["Model"] == model)].set_index("Strategy")
            accs = [float(rows.loc[strategy, "accuracy"]) * 100 for strategy in STRATEGIES]
            best = max(accs)
            cells = [r"\textbf{" + f"{value:.1f}" + "}" if np.isclose(value, best) else f"{value:.1f}" for value in accs]
            lines.append(model + " & " + " & ".join(cells) + f" & {np.mean(accs):.1f}" + r" \\")
        (TABLES / filename).write_text("\n".join(lines) + "\n")
    summary = []
    for index, model in enumerate(MODELS_CROSS):
        if index == 3:
            summary.append(r"\midrule")
        cells = []
        for dataset in ["CMAPSS", "TEP", "CWRU-4", "CWRU"]:
            rows = df[(df["Dataset"] == dataset) & (df["Model"] == model)]
            cells.append(f"{rows['accuracy'].mean() * 100:.1f}" if len(rows) else "--")
        summary.append(model + " & " + " & ".join(cells) + r" \\")
    (TABLES / "cross_domain_comparison.tex").write_text("\n".join(summary) + "\n")
    f1 = []
    for model in MODELS_CROSS:
        if not model.startswith("GPT-"):
            continue
        cells = []
        for dataset in ["CMAPSS", "TEP"]:
            rows = df[(df["Dataset"] == dataset) & (df["Model"] == model)]
            cells += [f"{rows[metric].mean() * 100:.1f}" for metric in ["accuracy", "f1_macro"]]
        f1.append(model + " & " + " & ".join(cells) + r" \\")
    (TABLES / "gpt_macro_f1.tex").write_text("\n".join(f1) + "\n")


def print_latex_table(df):
    for ds_label, ds_key in [("CWRU 10-Class", "CWRU"), ("CWRU 4-Class (Fault Type)", "CWRU-4")]:
        cwru = df[df["Dataset"] == ds_key]
        if cwru.empty:
            continue
        print(f"\n  LaTeX Table — {ds_label}:")
        print("  \\begin{tabular}{lcccc|c}")
        print("  \\hline")
        print("  Model & Zero-shot & Few-shot & CoT & Domain & Average \\\\")
        print("  \\hline")
        for model in MODELS_CWRU.keys():
            m = cwru[cwru["Model"] == model]
            accs = []
            parts = []
            for st in STRATEGIES:
                row = m[m["Strategy"] == st]
                if len(row) > 0:
                    a = row["accuracy"].values[0] * 100
                    accs.append(a)
                    parts.append(f"{a:.1f}")
                else:
                    parts.append("--")
            avg = np.mean(accs) if accs else 0
            print(f"  {model} & {' & '.join(parts)} & {avg:.1f} \\\\")
        print("  \\hline")
        print("  \\end{tabular}")

    # Cross-domain tables: C-MAPSS and TEP, all six evaluated models
    for ds_label, ds_key, n_cls in [("C-MAPSS", "CMAPSS", 3), ("TEP", "TEP", 7)]:
        cross = df[df["Dataset"] == ds_key]
        if cross.empty:
            continue
        print(f"\n  LaTeX Table — {ds_label} ({n_cls}-class), All Models:")
        print("  \\begin{tabular}{lcccc|c}")
        print("  \\hline")
        print("  Model & Zero-shot & Few-shot & CoT & Domain & Average \\\\")
        print("  \\hline")
        for model in MODELS_CROSS.keys():
            m = cross[cross["Model"] == model]
            accs = []
            parts = []
            for st in STRATEGIES:
                row = m[m["Strategy"] == st]
                if len(row) > 0:
                    a = row["accuracy"].values[0] * 100
                    accs.append(a)
                    parts.append(f"{a:.1f}")
                else:
                    parts.append("--")
            avg = np.mean(accs) if accs else 0
            print(f"  {model} & {' & '.join(parts)} & {avg:.1f} \\\\")
        print("  \\hline")
        print("  \\end{tabular}")


if __name__ == "__main__":
    print("=== Generating Summary ===")
    df = generate_summary()
    if df.empty:
        print("No results found! Check that *_s.json files exist.")
    else:
        print(f"\n=== Plotting ({len(df)} results) ===")
        plot_cwru_heatmap(df)
        plot_cwru_coarse_heatmap(df)
        plot_cwru_bar_chart(df)
        plot_confusion_matrices()
        if len(df[df["Dataset"] != "CWRU"]) > 0:
            plot_cross_dataset(df)
        write_paper_tables(df)
        print_latex_table(df)
        print("\nDone!")
