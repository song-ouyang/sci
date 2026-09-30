"""Render auditable, publication-sized diagnostics from frozen revision-v3 scores.

Run from any directory with the project environment::

    .venv/bin/python3 src/analysis/plot_revision_v3_diagnostics.py

This script never calls a model or alters evaluation inputs. Percentages and
percentage-point differences are converted from the scorer's fraction units.
Missing primary responses remain missing; later sessions are not substituted.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/revision_v3"
MODELS = ("gpt-5.5", "gpt-5.6-sol", "gpt-6-astra")
METHODS = (*MODELS, "RF", "SVM")
NAMES = {
    "gpt-5.5": "GPT-5.5",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gpt-6-astra": "GPT-6 Astra",
    "RF": "Random Forest",
    "SVM": "RBF SVM",
}
COLORS = {"LLM": "#1b526b", "baseline": "#758369"}
REPEAT_COLORS = ("#1b526b", "#c06926", "#806298")
MARKERS = ("o", "s", "^")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(score_path: Path, dataset_path: Path):
    report = json.loads(score_path.read_text())
    if report["experiment_version"] != "revision-v3" or report["audit"]["status"] != "verified":
        raise ValueError("Only verified revision-v3 scores are supported")
    if report["primary_repeat"] != 1:
        raise ValueError("The figure protocol requires primary repeat 1")
    if sha(dataset_path) != report["evaluation_provenance"]["datasets_sha256"]:
        raise ValueError("Frozen datasets do not match the scored dataset hash")
    datasets = json.loads(dataset_path.read_text())["datasets"]
    metrics = {}
    for row in report["metrics"]:
        key = (row["model"], row["dataset"], row["condition"], row["repeat"])
        if key in metrics:
            raise ValueError(f"Duplicate metric: {key}")
        metrics[key] = row
        ds = datasets[row["dataset"]]
        support = Counter(x["label"] for x in ds["test"])
        n = len(ds["test"])
        if len({x["group"] for x in ds["test"]}) != n:
            raise ValueError("Figures assume exactly one observation per test group")
        if row["n_expected"] != n:
            raise ValueError(f"Score/data sample-count mismatch: {key}")
        if not row["available"]:
            if row["accuracy"] is not None or row["per_class"] is not None:
                raise ValueError("Unavailable metrics must not contain classification scores")
            continue
        if row["n_scored"] != n or not np.isclose(row["accuracy"], row["correct"] / n):
            raise ValueError(f"Inconsistent accuracy counts: {key}")
        if set(row["per_class"]) != set(ds["valid_labels"]):
            raise ValueError(f"Inconsistent valid labels: {key}")
        for label, values in row["per_class"].items():
            if values["support"] != support[label] or not np.isclose(values["recall"], values["tp"] / support[label]):
                raise ValueError(f"Inconsistent class recall: {key}, {label}")
    pairs = {}
    for row in report["paired_contrasts"]:
        if row["contrast"] != "clean_minus_descriptor_mapping_fault":
            continue
        model = row["a"].split("/")[0]
        repeat = row["repeat"]
        key = (model, repeat)
        if key in pairs or row["dataset"] != "tep" or row["difference_direction"] != "a minus b":
            raise ValueError(f"Unexpected paired contrast: {key}")
        clean = metrics[(model, "tep", "clean", repeat)]
        shifted = metrics[(model, "tep", "misnamed", repeat)]
        if row["a"] != clean["id"] or row["b"] != shifted["id"]:
            raise ValueError(f"Unmatched paired metric identifiers: {key}")
        if not np.isclose(row["accuracy_difference"], clean["accuracy"] - shifted["accuracy"]):
            raise ValueError(f"Inconsistent paired effect: {key}")
        if row["n_groups"] != len(datasets["tep"]["test"]) or not np.isclose(row["flip_rate"], row["flip_count"] / row["n_groups"]):
            raise ValueError(f"Inconsistent paired flip count: {key}")
        pairs[key] = row
    for dataset in ("cmapss", "tep"):
        for model in METHODS:
            metrics[(model, dataset, "clean", 1 if model in MODELS else None)]
    for model in MODELS:
        for repeat in (1, 2, 3):
            pairs[(model, repeat)]
    return report, datasets, metrics, pairs


def metric(metrics, model, dataset, condition="clean"):
    return metrics[(model, dataset, condition, 1 if model in MODELS else None)]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, path: Path) -> None:
    # Keep the physical figure width at exactly 6.5 inches; do not tight-crop.
    fig.savefig(path.with_suffix(".pdf"), metadata={"Creator": "plot_revision_v3_diagnostics.py", "CreationDate": None})
    fig.savefig(path.with_suffix(".png"), dpi=300)
    plt.close(fig)


def primary_accuracy(metrics, datasets, figure_dir, csv_dir):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.3), sharey=True)
    fig.subplots_adjust(left=.205, right=.985, top=.82, bottom=.29, wspace=.13)
    rows = []
    for ax, dataset, title in zip(axes, ("cmapss", "tep"), ("(a) C-MAPSS", "(b) TEP")):
        for y, model in enumerate(METHODS):
            row = metric(metrics, model, dataset)
            available = row["available"]
            low, high = row["accuracy_wilson95"]
            rows.append({
                "metric_id": row["id"], "dataset": dataset, "model": model,
                "condition": "clean", "repeat": row["repeat"], "primary": True,
                "available": available, "status": row["status"],
                "failure_reason": row.get("failure_reason"),
                "n_expected": row["n_expected"], "n_scored": row["n_scored"],
                "correct": row["correct"],
                "accuracy_percent": 100 * row["accuracy"] if available else None,
                "wilson95_low_percent": 100 * low if available else None,
                "wilson95_high_percent": 100 * high if available else None,
            })
            if not available:
                ax.axhspan(y - .36, y + .36, color="#eeeeee", zorder=0)
                ax.text(51, y, "N/A: invalid batch", va="center", ha="center", fontsize=8, color="#666666")
                continue
            point = 100 * row["accuracy"]
            color = COLORS["LLM" if model in MODELS else "baseline"]
            error = np.maximum(0, [[point - 100 * low], [100 * high - point]])
            ax.errorbar(point, y, xerr=error, fmt="o" if model in MODELS else "s",
                        color=color, markersize=5.5, linewidth=1.5, capsize=3, zorder=3)
            ax.text(128, y, f"{point:.1f}", va="center", ha="right", fontsize=8.5)
        n = len(datasets[dataset]["test"])
        ax.set_title(f"{title}\n{n} {'engines' if dataset == 'cmapss' else 'run IDs'}", loc="left", fontsize=10, pad=9)
        ax.set_xlim(-3, 131)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Accuracy (%)")
        ax.set_yticks(range(len(METHODS)), [NAMES[m] for m in METHODS])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_bounds(0, 100)
        ax.grid(axis="x", alpha=.16, linewidth=.6)
        ax.set_axisbelow(True)
    axes[0].set_ylim(len(METHODS) - .45, -.65)
    fig.text(.205, .035, "Primary session (R1); baselines fitted once. Bars: descriptive 95% Wilson intervals.\n"
             "Intervals condition on these selected groups and observed outputs; N/A is not zero.", fontsize=7.6, va="bottom", color="#444444")
    save(fig, figure_dir / "primary_accuracy")
    write_csv(csv_dir / "primary_accuracy.csv", rows)


def mapping_sensitivity(pairs, datasets, figure_dir, csv_dir):
    fig, axes = plt.subplots(1, 2, figsize=(6.5, 3.5), sharey=True,
                             gridspec_kw={"width_ratios": [1.65, 1]})
    fig.subplots_adjust(left=.205, right=.98, top=.78, bottom=.26, wspace=.22)
    effect_ax, flip_ax = axes
    rows = []
    all_limits = []
    for y, model in enumerate(MODELS):
        for repeat, offset, color, marker in zip((1, 2, 3), (-.15, 0, .15), REPEAT_COLORS, MARKERS):
            row = pairs[(model, repeat)]
            effect = 100 * row["accuracy_difference"]
            low, high = [100 * x for x in row["accuracy_difference_group_bootstrap95"]]
            all_limits.extend((low, high))
            flip_low, flip_high = row["flip_rate_wilson95"]
            rows.append({
                "clean_metric_id": row["a"], "misnamed_metric_id": row["b"],
                "dataset": "tep", "model": model, "repeat": repeat, "primary": repeat == 1,
                "n_groups": row["n_groups"], "accuracy_difference_pp": effect,
                "paired_bootstrap95_low_pp": low, "paired_bootstrap95_high_pp": high,
                "flip_count": row["flip_count"], "flip_rate_percent": 100 * row["flip_rate"],
                "flip_wilson95_low_percent": 100 * flip_low,
                "flip_wilson95_high_percent": 100 * flip_high,
            })
            if repeat == 1:
                error = np.maximum(0, [[effect - low], [high - effect]])
                effect_ax.errorbar(effect, y + offset, xerr=error, color=color,
                                   fmt="none", linewidth=1.5, capsize=3, zorder=3)
            for ax, value in ((effect_ax, effect), (flip_ax, 100 * row["flip_rate"])):
                ax.scatter(value, y + offset, marker=marker, s=31, edgecolor=color,
                           facecolor=color if repeat == 1 else "white", linewidth=1.2, zorder=4)
        if all(pairs[(model, r)]["flip_count"] == 0 for r in (1, 2, 3)):
            effect_ax.text(6, y, "0 pp in all repeats", ha="left", va="center", fontsize=8)
            flip_ax.text(7, y, f"0/{len(datasets['tep']['test'])}\nin all repeats", ha="left", va="center", fontsize=8)
    effect_ax.set_title("(a) Accuracy effect", loc="left", fontsize=10, pad=8)
    flip_ax.set_title("(b) Prediction flips", loc="left", fontsize=10, pad=8)
    effect_ax.set_xlim(min(-5, min(all_limits) - 5), max(95, max(all_limits) + 5))
    effect_ax.set_xticks([0, 20, 40, 60, 80])
    effect_ax.set_xlabel("Clean minus misnamed (pp)")
    effect_ax.axvline(0, color="#999999", linestyle="--", linewidth=.8)
    flip_ax.set_xlim(-5, 105)
    flip_ax.set_xticks([0, 50, 100])
    flip_ax.set_xlabel("Flip rate (%)")
    for ax in axes:
        ax.set_yticks(range(len(MODELS)), [NAMES[m] for m in MODELS])
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", alpha=.16, linewidth=.6)
        ax.set_axisbelow(True)
    effect_ax.set_ylim(2.5, -.5)
    handles = [Line2D([], [], marker=marker, color=color, linestyle="-" if repeat == 1 else "none",
                      markerfacecolor=color if repeat == 1 else "white", markersize=5,
                      label="R1 (95% CI in a)" if repeat == 1 else f"R{repeat}")
               for repeat, color, marker in zip((1, 2, 3), REPEAT_COLORS, MARKERS)]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.60, .995), ncol=3,
               frameon=False, fontsize=8, handlelength=1.5, columnspacing=1.3)
    fig.text(.205, .028, "Same 70 run IDs in every pair; sessions are not pooled. CI: paired group bootstrap.\n"
             "Astra's zero observed flips and [0, 0] effect CI do not establish equivalence or robustness.",
             fontsize=7.3, va="bottom", color="#444444")
    save(fig, figure_dir / "mapping_sensitivity")
    write_csv(csv_dir / "mapping_sensitivity.csv", rows)


def class_recall(metrics, datasets, figure_dir, csv_dir):
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 5.7), gridspec_kw={"height_ratios": [5, 6]})
    fig.subplots_adjust(left=.31, right=.97, top=.895, bottom=.235, hspace=.51)
    cmap = matplotlib.colormaps["Blues"].copy()
    cmap.set_bad("#e1e1e1")
    rows = []
    panels = [
        ("cmapss", [(m, "clean") for m in METHODS], "(a) C-MAPSS: clean inputs"),
        ("tep", [(m, c) for m in MODELS for c in ("clean", "misnamed")],
         "(b) TEP: descriptor intervention"),
    ]
    for ax, (dataset, combinations, title) in zip(axes, panels):
        labels = datasets[dataset]["valid_labels"]
        support = Counter(x["label"] for x in datasets[dataset]["test"])
        values = np.full((len(combinations), len(labels)), np.nan)
        for y, (model, condition) in enumerate(combinations):
            row = metric(metrics, model, dataset, condition)
            for x, label in enumerate(labels):
                details = row["per_class"][label] if row["available"] else None
                value = 100 * details["recall"] if details else None
                values[y, x] = value if value is not None else np.nan
                rows.append({
                    "metric_id": row["id"], "dataset": dataset, "model": model,
                    "condition": condition, "repeat": row["repeat"], "primary": True,
                    "available": row["available"], "status": row["status"],
                    "class": label, "test_class_support": support[label],
                    "true_positives": details["tp"] if details else None,
                    "recall_percent": value,
                    "recall_wilson95_low_percent": 100 * details["recall_wilson95"][0] if details else None,
                    "recall_wilson95_high_percent": 100 * details["recall_wilson95"][1] if details else None,
                })
        img = ax.imshow(np.ma.masked_invalid(values), cmap=cmap, vmin=0, vmax=100,
                        interpolation="nearest", aspect="auto")
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                value = values[y, x]
                ax.text(x, y, "N/A" if np.isnan(value) else f"{value:.0f}", ha="center", va="center",
                        fontsize=9, color="white" if value >= 65 else "#222222")
        ax.set_xticks(range(len(labels)), [x.replace("_", " ") for x in labels], fontsize=8.5)
        ax.set_yticks(range(len(combinations)), [NAMES[m] + (f" · {c}" if dataset == "tep" else "")
                                                 for m, c in combinations], fontsize=8.5)
        ax.tick_params(length=0)
        ax.set_xticks(np.arange(len(labels) + 1) - .5, minor=True)
        ax.set_yticks(np.arange(len(combinations) + 1) - .5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1.4)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        if dataset == "tep":
            for y in (1.5, 3.5):
                ax.axhline(y, color="#999999", linewidth=.8)
        sizes = ", ".join(str(n) for n in sorted(set(support.values())))
        ax.set_title(f"{title}\n{sizes} observations per class", loc="left", fontsize=10, pad=8)
    color_ax = fig.add_axes([.40, .14, .45, .022])
    bar = fig.colorbar(img, cax=color_ax, orientation="horizontal", ticks=[0, 25, 50, 75, 100])
    bar.outline.set_visible(False)
    bar.ax.tick_params(length=2, labelsize=8)
    bar.set_label("Class recall (%)", fontsize=9, labelpad=3)
    fig.text(.31, .016, "LLMs: primary session (R1). Baselines: one fit. Gray = invalid batch (N/A).\n"
             "Misnamed = fixed cyclic shift of all 52 TEP descriptors; numeric values are unchanged.",
             fontsize=7.1, va="bottom", color="#444444")
    save(fig, figure_dir / "class_recall")
    write_csv(csv_dir / "class_recall.csv", rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores", type=Path, default=BASE / "analysis/scores.json")
    parser.add_argument("--datasets", type=Path, default=BASE / "data/datasets.json")
    parser.add_argument("--figure-dir", type=Path, default=ROOT / "paper/figures/revision_v3")
    parser.add_argument("--csv-dir", type=Path, default=BASE / "analysis/figure_data")
    args = parser.parse_args()
    input_hashes = {"scores_sha256": sha(args.scores), "datasets_sha256": sha(args.datasets)}
    report, datasets, metrics, pairs = load_inputs(args.scores, args.datasets)
    args.figure_dir.mkdir(parents=True, exist_ok=True)
    args.csv_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 9,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .7, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.facecolor": "white",
    })
    primary_accuracy(metrics, datasets, args.figure_dir, args.csv_dir)
    mapping_sensitivity(pairs, datasets, args.figure_dir, args.csv_dir)
    class_recall(metrics, datasets, args.figure_dir, args.csv_dir)
    if input_hashes != {"scores_sha256": sha(args.scores), "datasets_sha256": sha(args.datasets)}:
        raise RuntimeError("An evaluation input changed during figure generation")
    names = ("primary_accuracy", "mapping_sensitivity", "class_recall")
    provenance = {
        "schema_version": 1, "experiment_version": "revision-v3", **input_hashes,
        "script": str(Path(__file__).resolve().relative_to(ROOT)),
        "script_sha256": sha(Path(__file__)), "primary_repeat": 1,
        "rates_unit": "percent", "differences_unit": "percentage points",
        "missing_values": "Empty CSV numeric cells and visible N/A; never replaced by zero or later sessions.",
        "statistics": report["statistics"],
        "dependencies": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "artifacts": {str(path): sha(path) for name in names for path in
                      (args.csv_dir / f"{name}.csv", args.figure_dir / f"{name}.pdf", args.figure_dir / f"{name}.png")},
    }
    (args.csv_dir / "diagnostics_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (args.csv_dir / "README.md").write_text(
        "# Revision-v3 diagnostic figure data\n\n"
        "Generate from the project root with:\n\n"
        "```sh\n.venv/bin/python3 src/analysis/plot_revision_v3_diagnostics.py\n```\n\n"
        "Only Python, NumPy, and Matplotlib are required. The `--scores`, `--datasets`, "
        "`--figure-dir`, and `--csv-dir` options allow reuse. The script verifies the dataset hash, "
        "sample counts, class supports, metric identities, and paired contrast arithmetic before plotting. "
        "It never runs inference or edits source scores, prompts, or protocols. "
        "`diagnostics_provenance.json` records source/artifact hashes and statistical scope.\n\n"
        "CSV rates and CI limits are percentages; effect differences are percentage points (pp). "
        "Blank numeric cells represent unavailable results, not zero. Primary means LLM repeat 1; "
        "baseline repeat is blank because it was fitted once. All class supports describe the frozen "
        "test set, including the unscored primary batch.\n\n"
        "## Suggested captions and interpretation limits\n\n"
        "**primary_accuracy:** Clean-input accuracy for the prespecified primary session and the "
        "matched RF/SVM baselines, evaluated on 60 C-MAPSS engines and 70 TEP run IDs. "
        "Bars show two-sided descriptive 95% Wilson intervals. The invalid GPT-5.5 C-MAPSS batch "
        "remains N/A; no later repeat replaces it. LLMs use zero-shot grouped prompts whereas "
        "baselines use labeled training data. These are distinct information regimes.\n\n"
        "**mapping_sensitivity:** Paired TEP accuracy difference (clean minus misnamed descriptors, "
        "pp) and prediction flip rate (%), using the same 70 run IDs in each pair. The filled R1 "
        "points have 95% paired group-bootstrap effect intervals (10,000 draws); open R2/R3 "
        "points show session variation without pooling sessions as additional test observations. "
        "The intervention cyclically shifts all 52 complete descriptors while preserving numeric "
        "values and their order. Astra has zero observed effect and zero flips in all three pairs; "
        "its degenerate [0, 0] empirical effect interval does not establish equivalence or general "
        "robustness. A per-pair Wilson interval for zero flips is retained in the CSV.\n\n"
        "**class_recall:** Primary-session class recall (%) for clean C-MAPSS inputs (20 engines "
        "per class) and both TEP descriptor conditions (10 run IDs per class). Each heatmap cell "
        "uses its true-class denominator. Gray N/A cells denote the invalid primary batch. "
        "Per-class Wilson intervals are retained in the CSV; the heatmap is descriptive and "
        "does not support class-wise superiority claims.\n\n"
        "All intervals condition on the selected devices/runs and observed grouped outputs. "
        "They do not capture shared-batch dependence, model/session variability, selection bias, "
        "or benchmark contamination. They are unadjusted descriptive intervals, not confirmatory "
        "tests. Zero observed errors on the selected late-stage TEP subset do not establish "
        "industrial safety or early detection performance.\n"
    )
    print(json.dumps({"figures": [str(args.figure_dir / f"{name}.pdf") for name in names],
                      "csv_directory": str(args.csv_dir), "source_hashes": input_hashes}, indent=2))


if __name__ == "__main__":
    main()
