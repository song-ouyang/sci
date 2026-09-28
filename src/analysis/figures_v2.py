"""Figures for the revised manuscript (offline; reads results/statistics_v2 and frozen outputs)."""
import csv
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STATS = ROOT / "results/statistics_v2"
OUT = ROOT / "results/industrial_interface_focused_overleaf/figures/v2"

# Validated categorical slots (light print surface) plus neutral inks.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "pdf.fonttype": 42,
})


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


# ------------------------------------------------------------------ Fig. 1
def framework():
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    def box(x, y, w, h, title, lines, edge=INK2, size=6.8):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=1.2",
                                    linewidth=0.9, edgecolor=edge, facecolor="white"))
        ax.text(x + 1.0, y + h - 1.4, title, fontsize=7.6, fontweight="bold", color=INK, va="top")
        for k, line in enumerate(lines):
            ax.text(x + 1.0, y + h - 5.4 - 3.3 * k, line, fontsize=size, color=INK2, va="top")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.8, shrinkA=0, shrinkB=0))

    top, h = 22, 29
    box(0.5, top, 22.5, h, "1  Serialized record",
        ["one text field per channel:", "ID | name | unit | value", "", "TEP: 52 variables",
         "C-MAPSS: 60 features", "Hydraulic rig: 28 features"])
    box(25.8, top, 22.5, h, "2  Input changes",
        ["Mapping faults:", "  cyclic rotation", "  random derangement", "  same-unit swap",
         "Meaning-preserving:", "  intact-pair reorder", "Context: background A/B,", "  focal first / last"],
        edge=ORANGE)
    box(51.1, top, 22.5, h, "3  Repeated calls",
        ["k identical calls per input", "fresh session per call", "", "Time-blocked schedules:",
         "  ABBA/BAAB (4 blocks)", "  hydraulic rig (5 blocks)", "", "outputs frozen before scoring"],
        edge=BLUE)
    box(76.4, top, 23, h, "4  Inference",
        ["single-call McNemar", "crossed case x call", "  bootstrap interval", "call-level exact",
         "  randomization test", "stable-difference cases", "Fleiss' kappa (repeats)"], edge=AQUA)
    box(18, 1, 64, 16, "5  Engineering decision",
        ["Effect exceeds call-to-call variation   ->  interface defect: add trusted-channel check",
         "Effect within call-to-call variation     ->  not attributable to the input: balance timing, vote",
         "Invalid or incomplete output             ->  counts against availability, not conditional accuracy"],
        size=6.8)
    for x0, x1 in [(23.3, 25.5), (48.6, 50.8), (73.9, 76.1)]:
        arrow(x0, top + h / 2, x1, top + h / 2)
    arrow(88, top - 0.4, 88, 9)
    arrow(88, 9, 82.3, 9)
    fig.savefig(OUT / "framework.pdf", bbox_inches="tight")
    plt.close(fig)



# ------------------------------------------------------------------ Fig. 2
def repeats():
    table, truth = {}, {}
    for r in read_csv(ROOT / "results/independent_validation/analysis/cases.csv"):
        table.setdefault((r["model"], r["condition"], int(r["repeat"])), {})[r["id"]] = r["predicted"]
        truth[r["id"]] = r["true_label"]
    focal = sorted(table[("gpt-5.6-sol", "focal_only", 1)])
    panels = [("(a) Mapping conditions, 70 TEP runs",
               [("clean", "Clean"), ("rotation", "Rotation"), ("random_1", "Random\n1"), ("random_2", "Random\n2"),
                ("random_3", "Random\n3"), ("same_unit_swap", "Same-unit\nswap"), ("pair_shuffle", "Pair\nreorder")], None),
              ("(b) Same 21 focal runs in six contexts",
               [("clean", "Within 70"), ("focal_only", "Focal\nonly"), ("a_front", "Before A"), ("a_back", "After A"),
                ("b_front", "Before B"), ("b_back", "After B")], focal)]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7), gridspec_kw={"width_ratios": [7, 6]})
    markers = ["o", "s", "^"]
    for ax, (title, conds, ids) in zip(axes, panels):
        for mi, (model, color, dx) in enumerate([("gpt-5.6-sol", BLUE, -0.16), ("gpt-6-astra", ORANGE, 0.16)]):
            for k in (1, 2, 3):
                ys = []
                for cond, _ in conds:
                    pred = table[(model, cond, k)]
                    use = ids or sorted(pred)
                    ys.append(100 * np.mean([pred[i] == truth[i] for i in use]))
                ax.plot(np.arange(len(conds)) + dx + (k - 2) * 0.07, ys, markers[k - 1], color=color, ms=5,
                        mec="white", mew=0.7, ls="none")
        ax.set_xticks(np.arange(len(conds)), [n for _, n in conds], fontsize=7)
        ax.set_ylim(0, 105)
        ax.set_title(title, loc="left")
        grid(ax)
    axes[0].set_ylabel("Accuracy (%)")
    handles = [plt.Line2D([], [], ls="none", marker="o", color=BLUE, ms=6, label="GPT-5.6 Sol"),
               plt.Line2D([], [], ls="none", marker="o", color=ORANGE, ms=6, label="GPT-6 Astra")]
    handles += [plt.Line2D([], [], ls="none", marker=m, color=INK2, ms=5, label=f"Call {k}") for k, m in zip((1, 2, 3), markers)]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.06))
    fig.tight_layout()
    fig.savefig(OUT / "repeats.pdf", bbox_inches="tight")
    plt.close(fig)

# ------------------------------------------------------------------ Fig. 3
def forest():
    tep = [r for r in read_csv(STATS / "tep_contrasts.csv") if r["model"] == "gpt-5.6-sol"]
    abba = read_csv(STATS / "abba_contrast.csv")
    mapping = [r for r in tep if r["family"].endswith("mapping")]
    context = [r for r in tep if r["family"].endswith("context") and not r["contrast"].startswith("Focal only")]
    rows = [("Mapping (70 TEP runs)", mapping),
            ("Context (21 focal cases)", context + [dict(abba[0], contrast="Front vs back, time-balanced")])]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharex=True)
    for ax, (title, data) in zip(axes, rows):
        ys = np.arange(len(data))[::-1]
        for y, r in zip(ys, data):
            d, lo, hi = (100 * float(r[k]) for k in ["acc_diff", "acc_diff_ci_low", "acc_diff_ci_high"])
            robust = lo > 0 or hi < 0
            color = ORANGE if robust else BLUE
            ax.plot([lo, hi], [y, y], color=color, lw=2, solid_capstyle="round")
            ax.plot(d, y, "o", ms=6, color=color, mec="white", mew=1.2, zorder=3)
            if float(r["primary_mcnemar_p_holm"]) < 0.05:
                ax.text(hi + 3, y, "R1 p<0.05", va="center", fontsize=6.5, color=INK2)
        ax.axvline(0, color=INK2, lw=0.6)
        ax.set_yticks(ys, [r["contrast"] for r in data])
        ax.set_title(title, loc="left", color=INK)
        ax.set_xlabel("Accuracy difference (pp), 95% crossed bootstrap CI")
        grid(ax, "x")
        ax.set_xlim(-45, 100)
    handles = [plt.Line2D([], [], color=ORANGE, marker="o", lw=2, label="CI excludes 0"),
               plt.Line2D([], [], color=BLUE, marker="o", lw=2, label="CI includes 0")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.55, -0.08))
    fig.tight_layout()
    fig.savefig(OUT / "forest.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 4
def timing():
    rows = [r for r in read_csv(STATS / "call_time_trend.csv") if r["model"] == "gpt-5.6-sol"]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = axes[0]
    groups = [("Intact mapping", lambda r: r["intact_mapping"] == "True" and r["condition"] != "b_back", BLUE, "o"),
              ("Intact mapping, focal after B", lambda r: r["condition"] == "b_back", ORANGE, "s"),
              ("Descriptor fault", lambda r: r["intact_mapping"] == "False", MUTED, "^")]
    for label, keep, color, marker in groups:
        sub = [r for r in rows if keep(r)]
        ax.plot([float(r["start_minute"]) for r in sub], [100 * float(r["accuracy"]) for r in sub], marker,
                color=color, ms=6, mec="white", mew=0.8, ls="none", label=label)
    ax.set_xlabel("Client start time (min from first call)")
    ax.set_ylabel("Call accuracy (%)")
    ax.set_ylim(15, 105)
    ax.set_title("(a) Source-disjoint schedule, GPT-5.6 Sol", loc="left", pad=22)
    grid(ax)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=6.8,
              handletextpad=0.2, columnspacing=0.8)
    ax = axes[1]
    blocks = json.loads((STATS / "summary.json").read_text())["abba_blocks"]
    base = ROOT / "results/context_blocked_v1/analysis/focal_metrics.csv"
    metrics = read_csv(base)
    for cond, color, marker, label in [("A", BLUE, "o", "Focal first"), ("B", ORANGE, "s", "Focal last")]:
        sub = [m for m in metrics if m["condition"] == cond]
        ax.plot([int(m["block"]) + (-0.08 if cond == "A" else 0.08) for m in sub],
                [100 * float(m["accuracy"]) for m in sub], marker, color=color, ms=6, mec="white", mew=0.8,
                ls="none", label=label)
    ax.set_xticks([1, 2, 3, 4], ["1\nABBA", "2\nABBA", "3\nBAAB", "4\nBAAB"])
    ax.set_ylim(15, 105)
    ax.set_xlabel("Execution block")
    ax.set_title("(b) Time-balanced replication", loc="left", pad=22)
    grid(ax)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=6.8)
    fig.tight_layout()
    fig.savefig(OUT / "timing.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 5
def voting():
    rows = [r for r in read_csv(STATS / "voting.csv") if r["model"] == "gpt-5.6-sol"
            and r["condition"] not in {"early", "fd002"}]
    names = {"clean": "Clean", "rotation": "Rotation", "random_1": "Random 1", "random_2": "Random 2",
             "random_3": "Random 3", "same_unit_swap": "Same-unit swap", "pair_shuffle": "Pair reorder",
             "focal_only": "Focal only", "a_front": "Before A", "a_back": "After A", "b_front": "Before B",
             "b_back": "After B"}
    fig, ax = plt.subplots(figsize=(7.0, 2.4))
    x = np.arange(len(rows))
    for i, r in enumerate(rows):
        lo, hi = 100 * float(r["single_call_min"]), 100 * float(r["single_call_max"])
        ax.plot([i - 0.12, i - 0.12], [lo, hi], color=BLUE, lw=2, solid_capstyle="round")
    ax.plot(x - 0.12, [100 * float(r["single_call_mean"]) for r in rows], "o", color=BLUE, ms=6, mec="white",
            mew=1, label="Single call: mean and range of 3")
    ax.plot(x + 0.12, [100 * float(r["majority_of_3"]) for r in rows], "D", color=ORANGE, ms=5.5, mec="white",
            mew=1, label="Majority of 3 calls")
    ax.axvline(6.5, color=GRID, lw=1)
    ax.text(3, 8, "70 TEP runs", ha="center", color=INK2, fontsize=7)
    ax.text(9, 8, "21 focal cases", ha="center", color=INK2, fontsize=7)
    ax.set_xticks(x, [names[r["condition"]] for r in rows], rotation=30, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    grid(ax)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.0), ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "voting.pdf", bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ Fig. 6
def transfer():
    path = ROOT / "results/baselines_v2/scores.json"
    if not path.exists():
        print("baselines_v2 not available; skipping transfer figure")
        return
    import transfer_figure  # noqa: F401  (written once the baseline outputs are fixed)


# ------------------------------------------------------------------ Fig. 7
def hydraulic():
    info = json.loads((STATS / "summary.json").read_text())["hydraulic_info"]
    calls = info["collapse_calls"]
    conds = [("clean", "Canonical"), ("clean_repeat", "Canonical\n(duplicate)"), ("pair_reverse", "Intact-pair\nreversal"),
             ("descriptor_rotation", "Descriptor\nrotation"), ("a_front", "Before A"), ("a_front_repeat", "Before A\n(duplicate)"),
             ("a_back", "After A"), ("b_front", "Before B"), ("b_back", "After B")]
    fig, ax = plt.subplots(figsize=(7.0, 2.7))
    for x, (cond, _) in enumerate(conds):
        for c in [c for c in calls if c["condition"] == cond]:
            off = (c["block"] - 3) * 0.09
            y = 100 * c["correct"] / 18
            if c["collapsed"]:
                ax.plot(x + off, y, "o", ms=6, mfc="white", mec=ORANGE, mew=1.4, zorder=3)
            else:
                ax.plot(x + off, y, "o", ms=6, color=BLUE, mec="white", mew=0.8, zorder=3)
    ax.axhline(100 / 3, color=MUTED, lw=0.8, ls="--")
    ax.text(len(conds) - 0.55, 100 / 3 - 7, "chance", ha="right", fontsize=6.5, color=INK2)
    ax.axvline(3.5, color=GRID, lw=1)
    ax.text(1.5, 103, "18 focal cycles per request", ha="center", fontsize=6.8, color=INK2)
    ax.text(6.0, 103, "18 focal + 18 background cycles", ha="center", fontsize=6.8, color=INK2)
    ax.set_xticks(np.arange(len(conds)), [n for _, n in conds], fontsize=6.8)
    ax.set_ylabel("Focal accuracy (%)")
    ax.set_ylim(0, 110)
    grid(ax)
    handles = [plt.Line2D([], [], ls="none", marker="o", color=BLUE, mec="white", ms=6, label="Discriminating response"),
               plt.Line2D([], [], ls="none", marker="o", mfc="white", mec=ORANGE, mew=1.4, ms=6,
                          label="Collapsed response (one label for $\\geq$16 of 18 cycles)")]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.04))
    fig.tight_layout()
    fig.savefig(OUT / "hydraulic.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    targets = sys.argv[1:] or ["framework", "forest", "timing", "voting", "hydraulic"]
    for name in targets:
        globals()[name]()
        print("wrote", name)
