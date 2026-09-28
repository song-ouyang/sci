"""Post hoc descriptive delivery accounting from frozen, audited outcomes.

No inference, response repair, primary-score replacement, or independence claim.
Invalid batches contribute scheduled slots but no usable correct predictions.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

GROUPS = [("cmapss", "clean"), ("tep", "clean"), ("tep", "misnamed")]
MODELS = ["gpt-5.5", "gpt-5.6-sol", "gpt-6-astra", "deepseek-v4-pro", "qwen3.8-max"]
NAMES = ["GPT-5.5", "GPT-5.6 Sol", "GPT-6 Astra", "DeepSeek V4 Pro", "Qwen3.8 Max"]


def summarize(rows, model):
    selected = [x for x in rows if x["model"] == model]
    keys = [(x["dataset"], x["condition"], x["repeat"]) for x in selected]
    expected = {(d, c, r) for d, c in GROUPS for r in (1, 2, 3)}
    if len(keys) != 9 or set(keys) != expected:
        raise ValueError("Missing, duplicate, or unexpected scheduled outcome")
    result = []
    for dataset, condition in GROUPS:
        group = [x for x in selected if (x["dataset"], x["condition"]) == (dataset, condition)]
        slots = 60 if dataset == "cmapss" else 70
        valid = correct = 0
        for x in group:
            if x.get("n_expected", x.get("expected_n")) != slots:
                raise ValueError("Unexpected planned batch size")
            available = x.get("available", x["status"] == "valid")
            if not available:
                if x["n_scored"] != 0 or x.get("correct") is not None or x["accuracy"] is not None:
                    raise ValueError("Invalid batch must not contribute partial classification scores")
                continue
            cm = x["confusion_matrix"]
            total = sum(sum(counts.values()) for counts in cm.values())
            diagonal = sum(counts.get(label, 0) for label, counts in cm.items())
            if total != slots or x["n_scored"] != slots or diagonal != x["correct"]:
                raise ValueError("Confusion counts disagree with scored outcome")
            if abs(diagonal / slots - x["accuracy"]) > 1e-12:
                raise ValueError("Accuracy disagrees with confusion counts")
            valid += 1
            correct += diagonal
        planned = slots * 3
        validated = slots * valid
        result.append(dict(model=model, dataset=dataset, condition=condition,
                           scheduled_batches=3, valid_batches=valid,
                           scheduled_slots=planned, valid_slots=validated,
                           usable_correct=correct,
                           conditional_accuracy=correct / validated if validated else None,
                           label_coverage=validated / planned,
                           usable_correct_fraction=correct / planned,
                           unique_source_units=slots))
    return result


def main():
    from src.analysis.score_revision_v3 import verify_frozen_runs
    from src.analysis.audit_domestic_extension import verify
    verify_frozen_runs(ROOT / "results/revision_v3")
    verify(ROOT / "results/domestic_extension_v2")
    paths = ["results/revision_v3/analysis/scores.json", "results/domestic_extension_v2/analysis/scores.json"]
    reports = [json.loads((ROOT / p).read_text()) for p in paths]
    if reports[0]["audit"]["frozen_manifest_sha256"] != hashlib.sha256((ROOT / "results/revision_v3/predictions/frozen_manifest.json").read_bytes()).hexdigest():
        raise ValueError("Original score manifest mismatch")
    if reports[1]["freeze_sha256"] != hashlib.sha256((ROOT / "results/domestic_extension_v2/frozen_manifest.json").read_bytes()).hexdigest():
        raise ValueError("API score manifest mismatch")
    rows = reports[0]["metrics"] + reports[1]["metrics"]
    summary = [x for model in MODELS for x in summarize(rows, model)]
    out = ROOT / "results/cross_provider_revision/usable_predictions"
    out.mkdir(parents=True, exist_ok=True)
    report = dict(analysis_status="post_hoc_descriptive", independent_sample_expansion=False,
                  scope="Original nine sessions per GPT service and revised nine per API service; excludes unequal-schedule representation follow-up and all ineligible Claude records.",
                  invalid_classification_scores_remain_null=True,
                  source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in paths},
                  rows=summary)
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader(); writer.writerows(summary)
    lines = []
    for model, name in zip(MODELS, NAMES):
        for x in [v for v in summary if v["model"] == model]:
            task = "C-MAPSS" if x["dataset"] == "cmapss" else "TEP " + ("clean" if x["condition"] == "clean" else "corrupted")
            acc = "N/A" if x["conditional_accuracy"] is None else f"{100*x['conditional_accuracy']:.1f}"
            lines.append(f"{name} & {task} & {x['valid_batches']}/3 & {x['usable_correct']}/{x['valid_slots']} & {acc} & {100*x['usable_correct_fraction']:.1f} \\\\")
    latex = r'''\subsection{Usable correct outputs across scheduled requests}
Table~\ref{tab:usable} adds descriptive delivery accounting to the original and revised API schedules. Each cell covers three scheduled batches on the same source units. Repeated classifications are counted as delivery opportunities, not independent observations. The representation/context follow-up has a different schedule and is not mixed into this comparison. The table is not an equal-compute ranking.

\begin{table}[htbp]
\centering\small
\caption{Post hoc delivery accounting across the three scheduled repeats. Valid counts whole batches. $C/V$ counts correct labels within valid output; $A_{\mathrm{valid}}$ and $Y$ are percentages. For $Y$, the denominator is 180 scheduled slots for C-MAPSS and 210 for each TEP condition. Invalid batches remain N/A in classification tables and contribute no usable correct outputs here.}
\label{tab:usable}
\begin{tabular}{@{}llrrrr@{}}
\toprule
Service & Task/condition & Valid & $C/V$ & $A_{\mathrm{valid}}$ & $Y$ \\
\midrule
''' + "\n".join(lines) + r'''
\bottomrule
\end{tabular}
\end{table}
'''
    for model in ("deepseek-v4-pro", "qwen3.8-max"):
        x = next(v for v in summary if v["model"] == model and v["condition"] == "misnamed")
        name = NAMES[MODELS.index(model)]
        latex += (f"\n{name} supplies {x['usable_correct']} correct labels in {x['valid_slots']} validated corrupted-TEP slots "
                  f"({100*x['conditional_accuracy']:.1f}\\% conditional accuracy), compared with "
                  f"{x['scheduled_slots']} scheduled slots ({100*x['usable_correct_fraction']:.1f}\\% usable correct output). ")
    latex += "\nThis difference quantifies delivered output under the specified whole-batch contract. It does not impute the unobserved correctness of failed responses or measure fault-alarm utility, and it depends on the chosen validation and retry rules.\n"
    (out / "paper_insert.tex").write_text(latex)
    print(f"Wrote {len(summary)} descriptive groups from 45 audited scheduled requests.")


if __name__ == "__main__":
    main()
