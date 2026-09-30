"""
Generate final LaTeX tables for the paper with bootstrap CIs and supervised baselines.
Uses all_metrics_final.csv as input.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
TABLES = ROOT / "results" / "tables"
TABLES.mkdir(parents=True, exist_ok=True)

# Bootstrap CI from accuracy + n_correct + n_total
def bootstrap_ci(n_correct: int, n_total: int, n_boot: int = 2000, alpha: float = 0.95) -> tuple[float, float]:
    rng = np.random.default_rng(42)
    arr = np.array([1] * n_correct + [0] * (n_total - n_correct))
    boot = np.array([rng.choice(arr, size=n_total, replace=True).mean() for _ in range(n_boot)])
    lo = np.percentile(boot, 100 * (1 - alpha) / 2)
    hi = np.percentile(boot, 100 * (1 + alpha) / 2)
    return float(lo * 100), float(hi * 100)


def bold_max(values: list[float]) -> list[str]:
    mx = max(v for v in values if v is not None)
    out = []
    for v in values:
        if v is None:
            out.append("---")
        elif v == mx:
            out.append(f"\\textbf{{{v:.1f}}}")
        else:
            out.append(f"{v:.1f}")
    return out


def main():
    df = pd.read_csv(ROOT / "results" / "tables" / "all_metrics_final.csv")
    baselines = json.load(open(ROOT / "results" / "baselines_supervised.json"))

    # ── Map dataset names ─────────────────────────────────────────────────────
    dataset_map = {
        "CWRU": "cwru", "CWRU-4": "cwru4",
        "CMAPSS": "cmapss", "TEP": "tep",
    }
    strategy_map = {
        "zero_shot": "ZS", "few_shot": "FS",
        "cot": "CoT", "domain": "DK",
    }

    strategies = ["zero_shot", "few_shot", "cot", "domain"]
    strategy_labels = [strategy_map[s] for s in strategies]

    model_order = [
        "Haiku 4.5", "Sonnet 5", "Opus 4.6",
        "GPT-5.5", "GPT-5.6 Sol", "GPT-6 Astra",
    ]

    # ── Per-dataset tables ────────────────────────────────────────────────────
    for raw_ds, ds_key in dataset_map.items():
        sub = df[df["Dataset"] == raw_ds].copy()
        if sub.empty:
            continue

        rows = []
        prev_family = None
        for model_name in model_order:
            mdf = sub[sub["Model"] == model_name]
            if mdf.empty:
                continue

            # Determine family for midrule
            family = "GPT" if "GPT" in model_name else "Claude"
            if prev_family and family != prev_family:
                rows.append("\\midrule")
            prev_family = family

            accs = []
            for strat in strategies:
                row = mdf[mdf["Strategy"] == strat]
                if row.empty:
                    accs.append(None)
                else:
                    accs.append(float(row["accuracy"].iloc[0]) * 100)

            bold = bold_max([a for a in accs if a is not None])
            # Re-apply to preserve None positions
            bold_all = []
            bi = 0
            for a in accs:
                if a is None:
                    bold_all.append("---")
                else:
                    bold_all.append(bold[bi])
                    bi += 1

            # Mean and CI over strategies
            valid_accs = [a for a in accs if a is not None]
            mean_acc = np.mean(valid_accs) if valid_accs else 0.0

            # Bootstrap CI from last available result for this model
            last_row = None
            for strat in reversed(strategies):
                r = mdf[mdf["Strategy"] == strat]
                if not r.empty:
                    last_row = r.iloc[0]
                    break

            if last_row is not None:
                n_total = int(last_row["n_valid"])
                n_correct = int(last_row["n_correct"])
                lo, hi = bootstrap_ci(n_correct, n_total)
                ci_str = f"[{lo:.1f},{hi:.1f}]"
            else:
                ci_str = "---"

            rows.append(
                f"{model_name} & {' & '.join(bold_all)} & {mean_acc:.1f} & {ci_str} \\\\"
            )

        # Add supervised baselines
        bl_ds = "cwru" if ds_key in ("cwru", "cwru4") else ds_key
        if bl_ds in baselines:
            rows.append("\\midrule")
            for bl_name in ["SVM", "RF"]:
                bl = baselines[bl_ds][bl_name]
                acc = bl["acc"]
                std = bl["std"]
                # SVM/RF have same accuracy for all strategies (supervised, no prompts)
                bold_bl = [f"{acc:.1f}"] * 4
                # Approximate CI from ±std
                ci_str = f"[{acc-std:.1f},{acc+std:.1f}]"
                rows.append(
                    f"{bl_name} (supervised) & {' & '.join(bold_bl)} & {acc:.1f} & {ci_str} \\\\"
                )

        table_content = "\n".join(rows)
        out_path = TABLES / f"{ds_key}_full.tex"
        out_path.write_text(table_content)
        print(f"✓ {out_path.name}")

    # ── CWRU 4-class CoT paradox table ────────────────────────────────────────
    # Special table highlighting CoT behavior
    sub4 = df[df["Dataset"] == "CWRU-4"].copy()
    claude_rows = []
    for model_name in ["Haiku 4.5", "Sonnet 5", "Opus 4.6"]:
        mdf = sub4[sub4["Model"] == model_name]
        if mdf.empty:
            continue
        accs = []
        for strat in strategies:
            row = mdf[mdf["Strategy"] == strat]
            accs.append(float(row["accuracy"].iloc[0]) * 100 if not row.empty else None)
        bold_all = bold_max([a for a in accs if a is not None])
        # CoT column (index 2) is special — mark if CoT < ZS
        zs = accs[0] if accs[0] is not None else 0
        cot = accs[2] if accs[2] is not None else 0
        # Add arrow annotation if CoT drops below ZS
        cot_str = bold_all[2] if len(bold_all) > 2 else "---"
        if cot < zs - 5:
            cot_str = f"{cot_str}\\dn"  # custom command for down arrow

        line_parts = [model_name]
        for i, (a, b) in enumerate(zip(accs, bold_all)):
            if i == 2 and a is not None and a < (accs[0] or 0) - 5:
                line_parts.append(f"{b}$\\downarrow$")
            else:
                line_parts.append(b)
        mean_acc = np.mean([a for a in accs if a is not None])
        line_parts.append(f"{mean_acc:.1f}")
        claude_rows.append(" & ".join(line_parts) + " \\\\")

    (TABLES / "cwru4_cot_paradox.tex").write_text("\n".join(claude_rows))
    print("✓ cwru4_cot_paradox.tex")

    # ── Summary comparison table ──────────────────────────────────────────────
    # Best strategy per model per dataset
    summary_rows = []
    for model_name in model_order:
        row_parts = [model_name]
        for raw_ds in ["CWRU", "CWRU-4", "CMAPSS", "TEP"]:
            mdf = df[(df["Dataset"] == raw_ds) & (df["Model"] == model_name)]
            if mdf.empty:
                row_parts.append("---")
            else:
                best_acc = float(mdf["accuracy"].max()) * 100
                row_parts.append(f"{best_acc:.1f}")
        summary_rows.append(" & ".join(row_parts) + " \\\\")

    # Add supervised baselines
    summary_rows.append("\\midrule")
    for bl_name in ["SVM", "RF"]:
        row_parts = [f"{bl_name} (supervised)"]
        for ds_key in ["cwru", "cwru", "cmapss", "tep"]:
            if ds_key in baselines:
                row_parts.append(f"{baselines[ds_key][bl_name]['acc']:.1f}")
            else:
                row_parts.append("---")
        summary_rows.append(" & ".join(row_parts) + " \\\\")

    (TABLES / "summary_best_per_dataset.tex").write_text("\n".join(summary_rows))
    print("✓ summary_best_per_dataset.tex")

    print("\nAll tables written to", TABLES)


if __name__ == "__main__":
    main()
