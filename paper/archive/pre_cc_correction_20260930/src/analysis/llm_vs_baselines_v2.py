"""Paired comparison of LLM calls with the tuned supervised references (offline).

For every panel, the LLM's three calls are summarized and repeat 1 is compared
with each reference by an exact McNemar test on the shared case IDs. References
are grouped into like-for-like (same information as the LLM) and variants that
receive extra target-domain information. Ranges over all six methods are
reported so that no single reference is selected on test scores.
"""
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/statistics_v2"
METHODS = ["RF", "SVM-RBF", "HGB", "MLP", "kNN", "LogReg"]
PANELS = {"tep_late": "TEP late", "tep_early": "TEP early", "fd001_v3": "FD001", "fd002": "FD002"}
EXTRA = {"tep_late": "tep_phase_aware", "tep_early": "tep_phase_aware", "fd002": "fd002_in_domain_regime_norm"}


def mcnemar(a, b):
    n01 = sum(x == 1 and y == 0 for x, y in zip(a, b))
    n10 = sum(x == 0 and y == 1 for x, y in zip(a, b))
    return n01, n10, (1.0 if n01 + n10 == 0 else binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)


def llm_calls():
    calls = defaultdict(dict)  # (model, panel, repeat) -> {id: correct}
    with open(ROOT / "results/independent_validation/analysis/cases.csv", newline="") as handle:
        for r in csv.DictReader(handle):
            panel = {"clean": "tep_late", "early": "tep_early", "fd002": "fd002"}.get(r["condition"])
            if panel:
                calls[(r["model"], panel, int(r["repeat"]))][r["id"]] = int(r["correct"])
    with open(ROOT / "results/revision_v3/analysis/reuse/clean_correctness.csv", newline="") as handle:
        for r in csv.DictReader(handle):
            if r["dataset"] == "cmapss" and r["model"] in {"gpt-5.6-sol", "gpt-6-astra"} and r["available"] == "1":
                calls[(r["model"], "fd001_v3", int(r["repeat"]))][r["id"]] = int(r["correct"])
    return calls


def main():
    refs = defaultdict(dict)  # (method, variant, panel) -> {id: correct}
    with open(ROOT / "results/baselines_v2/cases.csv", newline="") as handle:
        for r in csv.DictReader(handle):
            refs[(r["method"], r["variant"], r["panel"])][r["id"]] = int(r["correct"])
    calls = llm_calls()
    rows = []
    for panel, label in PANELS.items():
        entry = {"panel": label}
        for model in ["gpt-5.6-sol", "gpt-6-astra"]:
            accs = [100 * np.mean(list(calls[(model, panel, k)].values())) if calls.get((model, panel, k)) else None
                    for k in (1, 2, 3)]
            entry[model] = accs
        ids = sorted(calls[("gpt-6-astra", panel, 1)])
        like = {m: 100 * np.mean([refs[(m, "original_train", panel)][i] for i in ids]) for m in METHODS}
        entry["like_range"] = [min(like.values()), max(like.values())]
        entry["like"] = like
        tests = {}
        for model in ["gpt-5.6-sol", "gpt-6-astra"]:
            llm = [calls[(model, panel, 1)][i] for i in ids]
            for m in METHODS:
                for variant in ["original_train", EXTRA.get(panel)]:
                    if variant and (m, variant, panel) in refs:
                        ref = [refs[(m, variant, panel)][i] for i in ids]
                        n01, n10, p = mcnemar(llm, ref)
                        tests[f"{model}|{m}|{variant}"] = {"llm_only": n01, "ref_only": n10, "p": p}
        entry["tests"] = tests
        if panel in EXTRA:
            extra = {m: 100 * np.mean([refs[(m, EXTRA[panel], panel)][i] for i in ids]) for m in METHODS}
            entry["extra_variant"] = EXTRA[panel]
            entry["extra_range"] = [min(extra.values()), max(extra.values())]
            entry["extra"] = extra
        if panel == "fd002":
            transfer = {m: 100 * np.mean([refs[(m, "fd001_regime_norm_transfer", panel)][i] for i in ids]) for m in METHODS}
            entry["fd001_regime_norm_transfer_range"] = [min(transfer.values()), max(transfer.values())]
        rows.append(entry)
    (OUT / "llm_vs_baselines.json").write_text(json.dumps(rows, indent=2) + "\n")
    for e in rows:
        print(e["panel"], "Sol", [round(x, 1) for x in e["gpt-5.6-sol"]], "Astra", [round(x, 1) for x in e["gpt-6-astra"]],
              "like", [round(x, 1) for x in e["like_range"]], "extra", [round(x, 1) for x in e.get("extra_range", [])])
        for key in ["gpt-6-astra|HGB|original_train", "gpt-5.6-sol|HGB|original_train",
                    f"gpt-6-astra|HGB|{EXTRA.get(e['panel'].lower().replace(' ', '_'), '')}"]:
            if key in e["tests"]:
                print("   ", key, e["tests"][key])
        for key, t in e["tests"].items():
            if "HGB" in key and key.split("|")[2] != "original_train":
                print("   ", key, t)


if __name__ == "__main__":
    main()
