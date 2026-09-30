"""Render supplementary tables for the error-structure and screening analyses."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/error_structure_v3"
OUT = ROOT / "results/industrial_interface_focused_overleaf/supp_structure_tables.tex"


def read(name):
    with open(BASE / name, newline="") as handle:
        return list(csv.DictReader(handle))


def esc(text):
    return text.replace("_", r"\_").replace("%", r"\%")


def pct(x, digits=1):
    return "--" if x in ("", "None", None) else f"{100 * float(x):.{digits}f}"


def table(caption, label, spec, header, rows, small=r"\scriptsize"):
    body = "\n".join(" & ".join(r) + r" \\" for r in rows)
    return (rf"\begin{{table}}[htbp]\centering{small}" "\n" rf"\caption{{{caption}}}\label{{{label}}}" "\n"
            rf"\resizebox{{\linewidth}}{{!}}{{\begin{{tabular}}{{{spec}}}\toprule" "\n" + header + r" \\\midrule" "\n"
            + body + "\n" + r"\bottomrule\end{tabular}}\end{table}" "\n")


def main():
    parts = []
    audit = read("single_call_audit.csv")
    parts.append(table(
        "Descriptive criterion concordance for 19 Sol contrasts. McNemar raw/Holm: primary call pair, adjusted within the original family. Call criterion: $p\\le0.10$ plus an accuracy interval excluding zero (matched-block sensitivity where blocked). Failure to meet it does not establish a false positive. The table marks raw/Holm single-call threshold decisions and the call criterion separately.",
        "tab:sf_audit", "@{}llrrrrccc@{}",
        "Family & Contrast & Raw $p$ & Holm $p$ & Call $p$ & Min.\\ $p$ & Raw$<.05$ & Holm$<.05$ & Call criterion",
        [[esc(r["family"].split("/")[-1]), esc(r["contrast"]), f"{float(r['single_call_p']):.2g}",
          f"{float(r['single_call_p_holm']):.2g}", f"{float(r['call_level_p']):.3f}", f"{float(r['min_attainable_p']):.3f}",
          "Yes" if r["single_call_significant"] == "True" else "No",
          "Yes" if r["single_call_holm_significant"] == "True" else "No",
          "Yes" if r["call_level_supported"] == "True" else "No"] for r in audit]))
    groups = [g for g in read("groups.csv") if int(g["total_errors"]) >= 5]
    rows = []
    for g in sorted(groups, key=lambda g: (g["study"], g["service"], g["group"])):
        k = int(g["calls"])
        err = int(g["total_errors"])
        share = int(g["stable_error"]) * k / err
        single = float(g["error_rate"])
        removed = (float(g["plurality_usable_correct_yield"]) - float(g["accuracy"])) / single
        rows.append([esc(g["study"]), "Sol" if "sol" in g["service"] else "Astra", esc(g["group"]), g["n_cases"], str(k),
                     pct(g["accuracy"]), f"{share:.2f}", g["unstable"], g["stable_error_same_label"],
                     g["stable_error_varying_labels"], g["plurality_withheld_ties"], f"{100 * removed:+.0f}"])
    parts.append(table(
        "Descriptive structure of the 28 identical-input groups with at least five errors. $s$: fraction of error slots contributed by cases wrong in all $k$ calls. U: mixed correct/incorrect cases. AW-same/AW-var: always wrong with the same/varying labels; only AW-same necessarily passes unanimity. Plurality uses all observed $k$ calls, withholding ties (T). $G=100(Y_{\\rm plurality}-A_{\\rm single})/(1-A_{\\rm single})$ is normalized usable-correct gain; ties are non-correct outputs, not diagnostic errors. The same-data accounting bound is $G\\le100(1-s)$.",
        "tab:sf_groups", "@{}lllrrrrrrrrr@{}",
        "Study & Service & Group & $n$ & $k$ & Acc.\\ (\\%) & $s$ & U & AW-same & AW-var & T & $G$ (\\%)", rows))
    sens = read("canary_threshold_sensitivity.csv")
    studies = list(dict.fromkeys(r["study"] for r in sens))
    rows = []
    for tau in dict.fromkeys(r["tau"] for r in sens):
        cells = [tau]
        for st in studies:
            r = next(x for x in sens if x["tau"] == tau and x["study"] == st)
            cells.append(f"{r['accepted']}/{r['calls']}, {pct(r['accepted_focal_accuracy'])}")
        rows.append(cells)
    parts.append(table(
        "Retrospective canary-threshold sensitivity (Sol): accepted calls and focal accuracy of accepted calls (\\%). The analysis target is 0.8; the later prospective protocol separately froze acceptance at 8/9 correct canaries. This threshold retains one wrong canary label.",
        "tab:sf_tau", "@{}l" + "r" * len(studies) + "@{}",
        "$\\tau$ & " + " & ".join(esc(s) for s in studies), rows))
    pol = read("decision_policies.csv")
    names = {"single": "First dispatch", "vote": "Plurality (withhold ties)", "vote_tie_first": "Plurality (first on ties)",
             "canary": "First canary pass", "gated": "Gated plurality"}
    parts.append(table(
        "Offline policy replay over identical-input windows ordered by recorded dispatch timestamps. $k$: available calls per window; policy call use is averaged per window/batch, not per focal decision. Rows reuse source records and observed requests, so they are not independent trials or equal-cost policy tests. The prospective frozen tie-first comparator is shown alongside tie-withholding plurality. Gated plurality uses the earliest passing call on a tie.",
        "tab:sf_policies", "@{}lllrrrrrrr@{}",
        "Study & Policy & $k$ & Windows & Decisions & Correct & Errors & Withheld & Err.\\ (\\%) & Calls/window",
        [[esc(r["study"]), names[r["policy"]], r["calls_available_min"] if r["calls_available_min"] == r["calls_available_max"] else r["calls_available_min"] + "--" + r["calls_available_max"],
          r["windows"], r["decisions"], r["accepted_correct"], r["accepted_error"],
          r["abstained"], pct(r["accepted_error_rate"]), f"{float(r['calls_per_window']):.1f}"] for r in pol]))
    OUT.write_text("% Generated by src/analysis/structure_tables_v3.py; do not edit by hand.\n" + "\n".join(parts))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
