"""How often does a single-call test agree with call-level evidence? (offline)

Every contrast analysed in results/statistics_v2 is classified by its primary
single-call McNemar test (unadjusted p < 0.05) and by call-level evidence:
the exact call-level randomization p is at its design minimum or below 0.05,
and the crossed case-by-call bootstrap interval excludes zero. The ABBA/BAAB
replication is included as a contrast of its own.
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
        p_call = float(r["call_perm_p_acc"])
        p_min = float(r["min_attainable_p_acc"])
        ci_excludes = float(r["acc_diff_ci_low"]) > 0 or float(r["acc_diff_ci_high"]) < 0
        call_level = p_call <= 0.10 + 1e-9 and ci_excludes  # 0.10 is the design minimum for 3 vs 3 calls
        single = p_single < 0.05
        table.append({"family": r["family"], "contrast": r["contrast"], "single_call_p": p_single,
                      "call_level_p": p_call, "min_attainable_p": p_min, "pooled_ci_excludes_zero": ci_excludes,
                      "single_call_significant": single, "call_level_supported": call_level,
                      "verdict": ("agree: effect" if single and call_level else
                                  "single-call only" if single else
                                  "call-level only" if call_level else "agree: none")})
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "single_call_audit.csv").open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    counts = {}
    for t in table:
        counts[t["verdict"]] = counts.get(t["verdict"], 0) + 1
    (OUT / "single_call_audit.json").write_text(json.dumps({"counts": counts, "n": len(table)}, indent=2) + "\n")
    for t in table:
        print(f"{t['family']:<18} {t['contrast']:<38} single p={t['single_call_p']:.2g} call p={t['call_level_p']:.3f} "
              f"(min {t['min_attainable_p']:.3f}) CI-excl={t['pooled_ci_excludes_zero']!s:<5} -> {t['verdict']}")
    print(counts)


if __name__ == "__main__":
    main()
