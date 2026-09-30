"""Render the supplementary statistics tables (Appendix F) from results/statistics_v2."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATS = ROOT / "results/statistics_v2"
OUT = ROOT / "results/industrial_interface_focused_overleaf/supp_stats_tables.tex"
NAMES = {"gpt-5.6-sol": "Sol", "gpt-6-astra": "Astra"}


def read(name):
    with open(STATS / name, newline="") as handle:
        return list(csv.DictReader(handle))


def pval(x):
    x = float(x)
    if x < 1e-3:
        mantissa, exponent = f"{x:.1e}".split("e")
        return f"${mantissa}\\times10^{{{int(exponent)}}}$"
    return f"{x:.3f}"


def num(x, scale=1.0, digits=1):
    value = float(x) * scale
    text = f"{value:.{digits}f}"
    return f"${text}$" if value < 0 else text


def kappa(x):
    return "--" if x in ("", "None") else f"{float(x):.2f}"


def contrast_table(rows, caption, label):
    lines = [r"\begin{table}[htbp]\centering\scriptsize",
             rf"\caption{{{caption}}}", rf"\label{{{label}}}", r"\setlength{\tabcolsep}{3pt}",
             r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{@{}llrrrrrrrrrr@{}}\toprule",
             r"Service & Contrast & $n$ & Acc.\ L/R & $\Delta$ [95\% CI] & McN $p$ & Holm & Rand.\ $p$ & (min) & $E$ [95\% CI] & SD & $\kappa$ L/R \\\midrule"]
    for r in rows:
        lines.append(" & ".join([
            NAMES.get(r["model"], r["model"]), r["contrast"], r["n_cases"],
            f"{100*float(r['acc_left']):.1f}/{100*float(r['acc_right']):.1f}",
            f"{num(r['acc_diff'],100)} [{num(r['acc_diff_ci_low'],100)}, {num(r['acc_diff_ci_high'],100)}]",
            pval(r["primary_mcnemar_p"]), pval(r["primary_mcnemar_p_holm"]), f"{float(r['call_perm_p_acc']):.3f}",
            f"{float(r['min_attainable_p_acc']):.3f}",
            f"{num(r['excess_disagreement'],1,3)} [{num(r['excess_ci_low'],1,3)}, {num(r['excess_ci_high'],1,3)}]",
            r["stable_different_cases"], f"{kappa(r['kappa_left'])}/{kappa(r['kappa_right'])}"]) + r" \\")
    lines += [r"\bottomrule\end{tabular}}\end{table}", ""]
    return "\n".join(lines)


def main():
    tep = read("tep_contrasts.csv")
    parts = [contrast_table([r for r in tep if r["family"].endswith("mapping")],
                            "Mapping contrasts on 70 source-disjoint TEP runs, three calls per condition. Accuracy intervals cross case and independent call resampling. McN: exact McNemar test on repeat 1; Holm: adjusted within service and contrast family. Rand.: whole-call exchangeability $p$; (min): smallest attainable value. $E$: excess disagreement with case-bootstrap CI conditional on observed calls. SD: stable-difference cases.",
                            "tab:sf_mapping"),
             contrast_table([r for r in tep if r["family"].endswith("context")],
                            "Context contrasts on the 21 focal TEP runs, three calls per condition. The last contrast changes request size and composition together.",
                            "tab:sf_context"),
             contrast_table(read("abba_contrast.csv"),
                            "Time-balanced ABBA/BAAB replication (Sol, background B, four blocks). The exact design test enumerates the six allocations of two ABBA and two BAAB sequences to four blocks. Accuracy CI: post-hoc case-by-matched-block sensitivity, retaining each selected block's calls jointly. $E$ CI resamples cases conditional on observed calls.",
                            "tab:sf_abba"),
             contrast_table(read("hydraulic_contrasts.csv"),
                            "Hydraulic rig, 18 focal cycles and five blocks. Byte-identical duplicates are pooled where available; call-label exchanges are restricted to each block. Accuracy CI: post-hoc case-by-matched-block sensitivity, preserving joint calls within sampled blocks. Few blocks limit population interpretation. $E$ CI is conditional on observed calls.",
                            "tab:sf_hydraulic")]
    sensitivity = read("abba_contrast.csv") + read("hydraulic_contrasts.csv")
    lines = [r"\begin{table}[htbp]\centering\small",
             r"\caption{Accuracy-interval sensitivity to preserving matched execution blocks. Both methods also resample shared case rows. The matched-block analysis was added during offline correction, not prespecified. The ABBA arbitrary within-block exchangeability sensitivity has 1296 assignments and $p=0.417$; its primary frozen-design test has six assignments and $p=0.667$.}",
             r"\label{tab:sf_block_sensitivity}", r"\begin{tabular}{@{}lrr@{}}\toprule",
             r"Contrast & Independent-call CI (pp) & Matched-block CI (pp) \\\midrule"]
    for r in sensitivity:
        lines.append(r["contrast"] + " & " + f"[{num(r['crossed_ci_low'],100)}, {num(r['crossed_ci_high'],100)}]" + " & " +
                     f"[{num(r['matched_block_ci_low'],100)}, {num(r['matched_block_ci_high'],100)}]" + r" \\")
    lines += [r"\bottomrule\end{tabular}\end{table}", ""]
    parts.append("\n".join(lines))
    rep = read("tep_repeatability.csv")
    lines = [r"\begin{table}[htbp]\centering\small",
             r"\caption{Repeatability of identical calls on the source-disjoint TEP panels: Fleiss' $\kappa$ over three calls and the share of cases with the same label in all three calls.}",
             r"\label{tab:sf_kappa}", r"\begin{tabular}{@{}lrrrr@{}}\toprule",
             r"Condition & Sol $\kappa$ & Sol all agree & Astra $\kappa$ & Astra all agree \\\midrule"]
    by = {(r["model"], r["condition"]): r for r in rep}
    for cond in dict.fromkeys(r["condition"] for r in rep):
        s, a = by[("gpt-5.6-sol", cond)], by[("gpt-6-astra", cond)]
        lines.append(f"{cond.replace('_', ' ')} & {float(s['fleiss_kappa']):.2f} & {100*float(s['all_calls_agree']):.1f}\\% & "
                     f"{float(a['fleiss_kappa']):.2f} & {100*float(a['all_calls_agree']):.1f}\\% \\\\")
    lines += [r"\bottomrule\end{tabular}\end{table}", ""]
    parts.append("\n".join(lines))
    OUT.write_text("% Generated by src/analysis/stats_tables_v2.py; do not edit by hand.\n" + "\n".join(parts))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
