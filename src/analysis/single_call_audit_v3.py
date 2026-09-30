"""Descriptive raw/Holm single-call and exploratory call-criterion concordance.

The call criterion is p <= .10 plus an accuracy interval excluding zero.
It is not a truth label: small call counts yield coarse p-values, and failure
to meet this criterion is not proof of no effect or a single-call false positive.
Blocked intervals are post-hoc matched-block sensitivities.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATS = ROOT / "results/statistics_v2"
OUT = ROOT / "results/error_structure_v3"


def rows(name):
    with open(STATS / name, newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    table = []
    for r in rows("tep_contrasts.csv") + rows("abba_contrast.csv") + rows("hydraulic_contrasts.csv"):
        if r["model"] != "gpt-5.6-sol" or r["contrast"].startswith("Identical duplicate"):
            continue
        p_single = float(r["primary_mcnemar_p"])
        p_holm = float(r["primary_mcnemar_p_holm"])
        p_call = float(r["call_perm_p_acc"])
        p_min = float(r["min_attainable_p_acc"])
        ci_excludes = float(r["acc_diff_ci_low"]) > 0 or float(r["acc_diff_ci_high"]) < 0
        call_level = p_call <= 0.10 + 1e-9 and ci_excludes  # 0.10 is the design minimum for 3 vs 3 calls
        single = p_single < 0.05
        single_holm = p_holm < 0.05
        def verdict(passed):
            return ("both criteria met" if passed and call_level else "single-call criterion only" if passed
                    else "call criterion only" if call_level else "neither criterion met")
        table.append({"family": r["family"], "contrast": r["contrast"], "single_call_p": p_single,
                      "single_call_p_holm": p_holm,
                      "call_level_p": p_call, "min_attainable_p": p_min, "pooled_ci_excludes_zero": ci_excludes,
                      "interval_method": r["acc_diff_ci_method"],
                      "single_call_significant": single, "call_level_supported": call_level,
                      "single_call_holm_significant": single_holm,
                      "verdict": verdict(single), "holm_verdict": verdict(single_holm)})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "single_call_audit.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    counts = {}
    for t in table:
        counts[t["verdict"]] = counts.get(t["verdict"], 0) + 1
    holm_counts = {}
    for t in table:
        holm_counts[t["holm_verdict"]] = holm_counts.get(t["holm_verdict"], 0) + 1
    summary = {"counts": counts, "holm_counts": holm_counts, "n": len(table),
               "raw_significant": sum(t["single_call_significant"] for t in table),
               "raw_significant_without_call_criterion": sum(t["single_call_significant"] and not t["call_level_supported"] for t in table),
               "holm_significant": sum(t["single_call_holm_significant"] for t in table),
               "holm_significant_without_call_criterion": sum(t["single_call_holm_significant"] and not t["call_level_supported"] for t in table),
               "interpretation": "Descriptive criterion disagreement, not a false-positive rate. Call criterion is exploratory p<=.10 plus CI excluding zero; low call counts and coarse design p-values restrict sensitivity."}
    (OUT / "single_call_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    for t in table:
        print(f"{t['family']:<18} {t['contrast']:<38} single p={t['single_call_p']:.2g} call p={t['call_level_p']:.3f} "
              f"(min {t['min_attainable_p']:.3f}) CI-excl={t['pooled_ci_excludes_zero']!s:<5} -> {t['verdict']}")
    print(counts)


if __name__ == "__main__":
    main()
