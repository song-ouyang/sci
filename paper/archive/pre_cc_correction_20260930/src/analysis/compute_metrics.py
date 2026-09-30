"""
Compute metrics and generate figures for the paper.
3 models × 4 strategies × 3 datasets
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

BATCHES = Path(__file__).parent.parent.parent / "results" / "batches"
FIGURES = Path(__file__).parent.parent.parent / "results" / "figures"
TABLES = Path(__file__).parent.parent.parent / "results" / "tables"

STRATEGIES = ["zero_shot", "few_shot", "cot", "domain"]
STRATEGY_LABELS = ["Zero-shot", "Few-shot", "CoT", "Domain"]

MODELS_CWRU = {
    "Opus 4.6": "_results_v2.json",
    "Sonnet": "_results_sonnet.json",
    "Haiku 4.5": "_results_haiku.json",
}

MODELS_OTHER = {
    "Opus 4.6": "_results_v2.json",
}


def load_results(dataset, strategy, suffix):
    f = BATCHES / f"{dataset}_{strategy}{suffix}"
    if not f.exists():
        return None
    return json.load(open(f))


def compute_all_metrics(data):
    true = [d["true_label"] for d in data]
    pred = [d["predicted"] for d in data]
    return {
        "accuracy": accuracy_score(true, pred),
        "f1_macro": f1_score(true, pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(true, pred, average="weighted", zero_division=0),
        "precision_macro": precision_score(true, pred, average="macro", zero_division=0),
        "recall_macro": recall_score(true, pred, average="macro", zero_division=0),
        "n_samples": len(data),
        "n_correct": sum(1 for t, p in zip(true, pred) if t == p),
    }


def generate_summary_table():
    rows = []
    # CWRU with all models
    for model, suffix in MODELS_CWRU.items():
        for st in STRATEGIES:
            data = load_results("cwru", st, suffix)
            if data:
                m = compute_all_metrics(data)
                rows.append({"Dataset": "CWRU", "Model": model, "Strategy": st, **m})

    # CMAPSS and TEP with Opus only
    for ds in ["cmapss", "tep"]:
        for model, suffix in MODELS_OTHER.items():
            for st in STRATEGIES:
                data = load_results(ds, st, suffix)
                if data:
                    m = compute_all_metrics(data)
                    rows.append({"Dataset": ds.upper(), "Model": model, "Strategy": st, **m})

    df = pd.DataFrame(rows)
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / "all_metrics.csv", index=False)
    print("Summary table saved to results/tables/all_metrics.csv")
    return df


def plot_cwru_heatmap(df):
    FIGURES.mkdir(parents=True, exist_ok=True)
    cwru = df[df["Dataset"] == "CWRU"]
    pivot = cwru.pivot_table(index="Model", columns="Strategy", values="accuracy")
    pivot = pivot[STRATEGIES]
    pivot.columns = STRATEGY_LABELS

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(pivot * 100, annot=True, fmt=".1f", cmap="YlOrRd",
                vmin=20, vmax=100, ax=ax, linewidths=0.5,
                cbar_kws={"label": "Accuracy (%)"})
    ax.set_title("CWRU 10-Class Bearing Fault Diagnosis Accuracy (%)\nModel × Prompt Strategy", fontsize=12)
    ax.set_ylabel("")
    ax.set_xlabel("")
    plt.tight_layout()
    fig.savefig(FIGURES / "cwru_heatmap.png", dpi=300)
    plt.close()
    print("Heatmap saved to results/figures/cwru_heatmap.png")


def plot_cwru_bar_chart(df):
    FIGURES.mkdir(parents=True, exist_ok=True)
    cwru = df[df["Dataset"] == "CWRU"]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(STRATEGIES))
    width = 0.25
    models = list(MODELS_CWRU.keys())
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, model in enumerate(models):
        model_data = cwru[cwru["Model"] == model]
        accs = []
        for st in STRATEGIES:
            row = model_data[model_data["Strategy"] == st]
            accs.append(row["accuracy"].values[0] * 100 if len(row) > 0 else 0)
        bars = ax.bar(x + i * width, accs, width, label=model, color=colors[i], edgecolor="white")
        for bar, acc in zip(bars, accs):
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
    fig.savefig(FIGURES / "cwru_bar_chart.png", dpi=300)
    plt.close()
    print("Bar chart saved to results/figures/cwru_bar_chart.png")


def plot_confusion_matrices():
    FIGURES.mkdir(parents=True, exist_ok=True)
    classes_cwru = ["Normal", "IR007", "IR014", "IR021", "B007", "B014", "B021", "OR007", "OR014", "OR021"]

    # Sonnet confusion matrices for the most interesting strategies
    interesting = [
        ("Sonnet", "zero_shot", "_results_sonnet.json"),
        ("Sonnet", "few_shot", "_results_sonnet.json"),
        ("Sonnet", "domain", "_results_sonnet.json"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for idx, (model, st, suffix) in enumerate(interesting):
        data = load_results("cwru", st, suffix)
        if not data:
            continue
        true = [d["true_label"] for d in data]
        pred = [d["predicted"] for d in data]
        all_labels = sorted(set(true) | set(pred))
        cm = confusion_matrix(true, pred, labels=classes_cwru)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[idx],
                    xticklabels=classes_cwru, yticklabels=classes_cwru)
        acc = accuracy_score(true, pred)
        axes[idx].set_title(f"Sonnet - {STRATEGY_LABELS[STRATEGIES.index(st)]}\nAcc: {acc*100:.0f}%", fontsize=11)
        axes[idx].set_ylabel("True Label" if idx == 0 else "")
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].tick_params(axis='y', rotation=0)

    plt.suptitle("Confusion Matrices: Sonnet on CWRU 10-Class", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "sonnet_confusion_matrices.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Confusion matrices saved to results/figures/sonnet_confusion_matrices.png")


def plot_cross_dataset():
    FIGURES.mkdir(parents=True, exist_ok=True)
    datasets = ["CWRU", "CMAPSS", "TEP"]
    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(datasets))
    width = 0.2
    colors = ["#E53935", "#FB8C00", "#43A047", "#1E88E5"]

    for i, (st, label) in enumerate(zip(STRATEGIES, STRATEGY_LABELS)):
        accs = []
        for ds in ["cwru", "cmapss", "tep"]:
            data = load_results(ds, st, "_results_v2.json")
            if data:
                true = [d["true_label"] for d in data]
                pred = [d["predicted"] for d in data]
                accs.append(accuracy_score(true, pred) * 100)
            else:
                accs.append(0)
        bars = ax.bar(x + i * width, accs, width, label=label, color=colors[i])

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title("Opus 4.6: Cross-Domain Diagnosis Accuracy", fontsize=13)
    ax.set_xticks(x + 1.5 * width)
    ax.set_xticklabels(["CWRU\n(Bearing, 10-class)", "C-MAPSS\n(Engine, 3-class)", "TEP\n(Chemical, 7-class)"])
    ax.set_ylim(0, 115)
    ax.legend(title="Strategy")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "cross_dataset_opus.png", dpi=300)
    plt.close()
    print("Cross-dataset chart saved to results/figures/cross_dataset_opus.png")


def print_latex_table(df):
    print("\n" + "=" * 70)
    print("LaTeX Table: CWRU Results")
    print("=" * 70)
    cwru = df[df["Dataset"] == "CWRU"]
    print("\\begin{table}[htbp]")
    print("\\centering")
    print("\\caption{Classification accuracy (\\%) on CWRU bearing dataset (10-class) across models and prompt strategies}")
    print("\\label{tab:cwru_results}")
    print("\\begin{tabular}{lcccc|c}")
    print("\\hline")
    print("Model & Zero-shot & Few-shot & CoT & Domain & Average \\\\")
    print("\\hline")
    for model in ["Opus 4.6", "Sonnet", "Haiku 4.5"]:
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
        print(f"{model} & {' & '.join(parts)} & {avg:.1f} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


if __name__ == "__main__":
    df = generate_summary_table()
    print("\n--- Full Metrics ---")
    print(df.to_string(index=False))

    plot_cwru_heatmap(df)
    plot_cwru_bar_chart(df)
    plot_confusion_matrices()
    plot_cross_dataset()
    print_latex_table(df)
