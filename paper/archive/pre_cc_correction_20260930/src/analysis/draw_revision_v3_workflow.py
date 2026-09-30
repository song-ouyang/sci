"""Draw the study workflow and symbolic descriptor rotation; no model calls."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(6.5, 4.75), facecolor="white")
    ax = fig.add_axes([0.015, 0.02, 0.97, 0.96])
    ax.set(xlim=(0, 10), ylim=(0, 10)); ax.axis("off")
    navy, orange, ink = "#1b526b", "#c06926", "#263842"

    def box(x, y, heading, body):
        ax.add_patch(FancyBboxPatch((x, y), 4.45, 1.67,
                                   boxstyle="round,pad=0.08,rounding_size=0.10",
                                   facecolor="#f2f6f7", edgecolor="#a7bac2", lw=.8))
        ax.text(x+.18, y+1.34, heading, fontsize=9.3, fontweight="bold", color=navy)
        ax.text(x+.18, y+.98, body, fontsize=8.2, va="top", linespacing=1.5, color=ink)

    def arrow(start, end, color=navy):
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                    mutation_scale=12, lw=1.2, color=color))

    ax.text(.1, 9.65, "A  From disjoint source groups to frozen results",
            fontsize=10, fontweight="bold", color=ink)
    box(.12, 7.65, "1  Separate source groups",
        "C-MAPSS: 100 train / 60 test engines\nTEP: 100 train / 70 test run IDs")
    box(5.38, 7.65, "2  Fix shared numerical inputs",
        "C-MAPSS: 60 past-window features\nTEP: 52 named process variables")
    box(5.38, 5.35, "3  Evaluate fixed interfaces",
        "RF / SVM / majority: fit on training data\n3 LLMs x 3 conditions x 3 fresh sessions")
    box(.12, 5.35, "4  Validate, freeze, then score",
        "26/27 complete LLM batches; 1 batch N/A\nRepeat 1 primary; repeats 2/3 separate")
    arrow((4.68, 8.49), (5.28, 8.49))
    arrow((7.61, 7.53), (7.61, 7.11))
    arrow((5.26, 6.18), (4.7, 6.18))
    ax.text(5, 4.89, "Three conditions: C-MAPSS clean, TEP clean, TEP descriptor rotation",
            ha="center", fontsize=8.1, color=ink)

    ax.plot([.08, 9.94], [4.52, 4.52], color="#d9e1e5", lw=.8)
    ax.text(.1, 4.12, "B  Paired TEP intervention: same values, shifted descriptors",
            fontsize=10, fontweight="bold", color=ink)
    ax.text(2.3, 3.52, "Clean rows", ha="center", color=navy, fontweight="bold")
    ax.text(7.68, 3.52, "Mapping-fault rows", ha="center", color=orange, fontweight="bold")
    for y, clean, shifted in [(2.94, r"$(d_1,\ v_1)$", r"$(d_2,\ v_1)$"),
                              (2.37, r"$(d_2,\ v_2)$", r"$(d_3,\ v_2)$"),
                              (1.86, r"$\vdots$", r"$\vdots$"),
                              (1.3, r"$(d_{52},\ v_{52})$", r"$(d_1,\ v_{52})$")]:
        ax.text(2.3, y, clean, ha="center", va="center", color=navy, fontsize=11)
        ax.text(7.68, y, shifted, ha="center", va="center", color=orange, fontsize=11)
    arrow((3.55, 2.48), (6.33, 2.48), orange)
    ax.text(4.94, 2.81, "Rotate descriptors +1", ha="center", fontsize=8, color=orange)
    ax.text(4.94, 2.09, "Keep numerical order", ha="center", fontsize=8, color=ink)
    ax.text(5, .54, r"$d_i$: canonical identity + name + unit; $v_i$: rounded numerical string",
            ha="center", fontsize=8.2, color=ink)
    ax.text(5, .11, "Sample IDs, task instructions, numerical strings and batch positions are unchanged.",
            ha="center", fontsize=8.1, color=ink)
    dest = ROOT / "paper/figures/revision_v3"
    dest.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(dest / f"evaluation_workflow.{ext}", dpi=240,
                    metadata={"Creator": "revision_v3 workflow builder"} if ext == "pdf" else None)
    plt.close(fig)
    print("Generated evaluation_workflow.pdf/png (symbolic schematic).")


if __name__ == "__main__":
    main()
