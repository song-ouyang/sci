#!/usr/bin/env python3
"""Export audited, sample-level revision-v3 cases without new model calls.

The complete frozen-outcome audit precedes label access. A semantically failed
batch contributes explicit unavailable rows, never its partial raw predictions.
Only derived files under analysis/reuse are written.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.analysis.score_revision_v3 import (
    BASELINES, MODELS, load_evaluation_data, metric_values, read_json, require,
    verify_frozen_runs,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prompt_hash(prompt):
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def outcome(model, name, predictions, audit, failures):
    rows = predictions[model][name]
    run = audit["runs"][model][name]
    available = rows is not None
    return {
        "available": int(available), "status": run["status"],
        "by_id": {row["id"]: row["predicted"] for row in rows} if available else {},
        "failure_reason": "" if available else failures[(model, name)]["reason"],
        "session_id": run["session_id"],
        "predictions_file": f"predictions/{model}/{name}_results.json" if available else "",
        "metadata_file": f"predictions/{model}/{name}_metadata.json",
        "expected_session_samples": run["expected_n"], "scored_session_samples": run["n_predictions"],
    }


def make_case_rows(groups, predictions, audit, datasets, baselines):
    failures = {(row["model"], row["group"]): row for row in audit["session_failures"]}
    paired, clean = [], []
    for dataset in ["cmapss", "tep"]:
        for model in MODELS:
            for repeat in [1, 2, 3]:
                name = f"{dataset}_clean_r{repeat}"
                state = outcome(model, name, predictions, audit, failures)
                prompts = {row["id"]: row["prompt"] for row in groups[name]["samples"]}
                for sample in datasets[dataset]:
                    prediction = state["by_id"].get(sample["id"])
                    clean.append({
                        "dataset": dataset, "model": model, "model_type": "LLM", "repeat": repeat,
                        "primary": int(repeat == 1), "id": sample["id"], "source_group": sample["group"],
                        "true_label": sample["label"], "available": state["available"], "status": state["status"],
                        "predicted": prediction, "correct": int(prediction == sample["label"]) if state["available"] else None,
                        "failure_reason": state["failure_reason"], "session_id": state["session_id"],
                        "expected_session_samples": state["expected_session_samples"],
                        "scored_session_samples": state["scored_session_samples"],
                        "prompt_sha256": prompt_hash(prompts[sample["id"]]),
                        "predictions_file": state["predictions_file"], "metadata_file": state["metadata_file"],
                    })
        for method in BASELINES:
            for sample, prediction in zip(datasets[dataset], baselines[dataset][method]):
                clean.append({
                    "dataset": dataset, "model": method, "model_type": "supervised_baseline", "repeat": None,
                    "primary": 1, "id": sample["id"], "source_group": sample["group"], "true_label": sample["label"],
                    "available": 1, "status": "complete", "predicted": prediction,
                    "correct": int(prediction == sample["label"]), "failure_reason": "", "session_id": "",
                    "expected_session_samples": len(datasets[dataset]), "scored_session_samples": len(datasets[dataset]),
                    "prompt_sha256": "", "predictions_file": "baselines/predictions.json", "metadata_file": "baselines/scores.json",
                })
    for model in MODELS:
        for repeat in [1, 2, 3]:
            names = {condition: f"tep_{condition}_r{repeat}" for condition in ["clean", "misnamed"]}
            states = {condition: outcome(model, name, predictions, audit, failures) for condition, name in names.items()}
            prompts = {condition: {row["id"]: row["prompt"] for row in groups[name]["samples"]}
                       for condition, name in names.items()}
            for sample in datasets["tep"]:
                available = all(state["available"] for state in states.values())
                guesses = {condition: state["by_id"].get(sample["id"]) for condition, state in states.items()}
                correctness = {condition: int(guesses[condition] == sample["label"]) if state["available"] else None
                               for condition, state in states.items()}
                transition = "unavailable"
                if available:
                    transition = {(1, 1): "both_correct", (1, 0): "correct_to_incorrect",
                                  (0, 1): "incorrect_to_correct", (0, 0): "both_incorrect"}[
                                      (correctness["clean"], correctness["misnamed"])]
                row = {"dataset": "tep", "model": model, "repeat": repeat, "primary": int(repeat == 1),
                       "id": sample["id"], "source_group": sample["group"], "true_label": sample["label"],
                       "pair_available": int(available), "transition_clean_to_misnamed": transition,
                       "prediction_flipped": int(guesses["clean"] != guesses["misnamed"]) if available else None,
                       "correctness_change_clean_minus_misnamed": correctness["clean"] - correctness["misnamed"] if available else None}
                for condition, state in states.items():
                    row.update({f"{condition}_available": state["available"], f"{condition}_status": state["status"],
                                f"{condition}_predicted": guesses[condition], f"{condition}_correct": correctness[condition],
                                f"{condition}_failure_reason": state["failure_reason"], f"{condition}_session_id": state["session_id"],
                                f"{condition}_prompt_sha256": prompt_hash(prompts[condition][sample["id"]]),
                                f"{condition}_predictions_file": state["predictions_file"],
                                f"{condition}_metadata_file": state["metadata_file"]})
                paired.append(row)
    return paired, clean


def verify_export_rows(paired, clean, report, datasets):
    """Check row coverage, nulls and aggregates against the verified score report."""
    require(len(paired) == 3 * 3 * len(datasets["tep"]), "Paired export row count mismatch")
    require(len(clean) == (3 * 3 + 3) * sum(len(rows) for rows in datasets.values()), "Clean export row count mismatch")
    require(len({(row["model"], row["repeat"], row["id"]) for row in paired}) == len(paired), "Duplicate paired row key")
    require(len({(row["dataset"], row["model"], row["repeat"], row["id"]) for row in clean}) == len(clean),
            "Duplicate clean row key")
    for row in report["metrics"]:
        if row["condition"] != "clean":
            continue
        records = [item for item in clean if item["dataset"] == row["dataset"] and item["model"] == row["model"]
                   and item["repeat"] == row["repeat"]]
        require(len(records) == row["n_expected"], f"{row['id']}: source coverage mismatch")
        if not row["available"]:
            require(all(item["available"] == 0 and item["predicted"] is None and item["correct"] is None
                        and item["scored_session_samples"] == 0 for item in records),
                    f"{row['id']}: failed batch leaked partial predictions or zero-filled metrics")
            continue
        require(all(item["available"] == 1 for item in records), f"{row['id']}: missing successful predictions")
        actual = metric_values([item["true_label"] for item in records], [item["predicted"] for item in records],
                               list(row["per_class"]), row["dataset"])
        require(actual["correct"] == row["correct"] and row["n_scored"] == len(records), f"{row['id']}: correct-count mismatch")
        require(all(math.isclose(actual[name], row[name], abs_tol=1e-12) for name in ["accuracy", "macro_f1"]),
                f"{row['id']}: exported classification metrics differ")
    for model in MODELS:
        for repeat in [1, 2, 3]:
            records = [row for row in paired if row["model"] == model and row["repeat"] == repeat]
            a, b = f"{model}/tep_clean_r{repeat}", f"{model}/tep_misnamed_r{repeat}"
            for condition, key in [("clean", a), ("misnamed", b)]:
                scored = next(row for row in report["metrics"] if row["id"] == key)
                if not scored["available"]:
                    require(all(row[f"{condition}_predicted"] is None and row[f"{condition}_correct"] is None
                                for row in records), f"{key}: unavailable condition leaked partial labels")
                    continue
                actual = metric_values([row["true_label"] for row in records], [row[f"{condition}_predicted"] for row in records],
                                       list(scored["per_class"]), "tep")
                require(actual["correct"] == scored["correct"] and scored["n_scored"] == len(records)
                        and all(math.isclose(actual[metric], scored[metric], abs_tol=1e-12) for metric in ["accuracy", "macro_f1"]),
                        f"{key}: paired-case export metrics differ from verified scores")
            if not all(row["pair_available"] for row in records):
                require(any(row["a"] == a and row["b"] == b for row in report["unavailable_contrasts"]),
                        f"{a}/{b}: unreported unavailable contrast")
                require(all(row["prediction_flipped"] is None and row["correctness_change_clean_minus_misnamed"] is None
                            for row in records), f"{a}/{b}: unavailable pair has numeric comparisons")
                continue
            contrast = next((row for row in report["paired_contrasts"] if row["a"] == a and row["b"] == b), None)
            require(contrast is not None, f"{a}/{b}: missing score contrast")
            require(sum(row["prediction_flipped"] for row in records) == contrast["flip_count"], f"{a}/{b}: flip mismatch")
            require(math.isclose(sum(row["correctness_change_clean_minus_misnamed"] for row in records) / len(records),
                                 contrast["accuracy_difference"], abs_tol=1e-12), f"{a}/{b}: accuracy difference mismatch")
    return {"tep_paired_rows": len(paired), "available_tep_pairs": sum(row["pair_available"] for row in paired),
            "clean_rows": len(clean), "llm_clean_rows": sum(row["model_type"] == "LLM" for row in clean),
            "baseline_clean_rows": sum(row["model_type"] == "supervised_baseline" for row in clean),
            "unavailable_clean_rows": sum(not row["available"] for row in clean),
            "unique_source_units": sum(len(rows) for rows in datasets.values())}


def write_csv(path, rows):
    require(rows, f"No rows for {path.name}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)  # None becomes an empty CSV cell, never a fabricated label or zero.
    temporary.replace(path)


def export(base):
    protocol, groups, predictions, audit = verify_frozen_runs(base)
    report, report_hash = read_json(base / "analysis/scores.json")
    require(report.get("audit") == audit, "Score report does not match current frozen-outcome audit")
    datasets, baselines, evaluation = load_evaluation_data(base, protocol, groups)
    require(report.get("evaluation_provenance") == evaluation, "Score/evaluation data provenance mismatch")
    paired, clean = make_case_rows(groups, predictions, audit, datasets, baselines)
    counts = verify_export_rows(paired, clean, report, datasets)
    source_names = ["protocol.json", "failure_reporting_amendment.json", "transport_audit_note.json",
                    "data/datasets.json", "data/manifest.json", "inference/batches.json",
                    "baselines/predictions.json", "baselines/scores.json", "baselines/manifest.json",
                    "predictions/frozen_manifest.json", "analysis/scores.json", "prediction_contract.schema.json"]
    source_names += [f"inference/prompts/{name}.txt" for name in groups]
    source_names += [f"predictions/{name}" for name in audit["files_sha256"]]
    sources = {name: digest(base / name) for name in sorted(set(source_names))}
    require(sources["analysis/scores.json"] == report_hash, "Scores changed during export")
    # Audit once more before publishing derived outputs; this never refreshes the freeze.
    require(verify_frozen_runs(base)[3] == audit, "Frozen outcomes changed during export")
    output = base / "analysis/reuse"
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "cases.csv", paired)
    write_csv(output / "clean_correctness.csv", clean)
    manifest = {
        "schema": "revision-v3-reuse-export-v1", "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "exporter": "src/analysis/export_revision_v3_cases.py", "exporter_sha256": digest(Path(__file__)),
        "scorer_sha256": digest(ROOT / "src/analysis/score_revision_v3.py"),
        "python_version": sys.version.split()[0], "new_model_calls": 0,
        "writes": "Derived files under results/revision_v3/analysis/reuse only; frozen inputs and outcomes unchanged.",
        "primary_repeat": 1, "counts": counts,
        "study_counts": {key: audit[key] for key in ["planned_sessions", "completed_sessions", "failed_sessions",
                                                      "expected_predictions", "total_valid_predictions"]},
        "null_policy": "An unavailable batch has blank predicted/correct/comparison cells for every planned sample. All partial labels are excluded. available/status/failure_reason identify these N/A rows.",
        "pair_direction": "clean minus misnamed; positive correctness change is harm under the descriptor mapping fault",
        "row_keys": {"cases.csv": ["model", "repeat", "id"],
                     "clean_correctness.csv": ["dataset", "model", "repeat", "id"]},
        "id_policy": "Treat id as an eight-character string. Use dataset + id for cross-file joins; retain leading zeros.",
        "independence": "CSV rows repeat the same 130 source units across methods, sessions and conditions. They are not independent observations; repeat 1 remains primary.",
        "sources_relative_to_revision_root": sources,
        "outputs": {name: {"sha256": digest(output / name), "rows": len(rows)}
                    for name, rows in [("cases.csv", paired), ("clean_correctness.csv", clean)]},
    }
    temporary = output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output / "manifest.json")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "results/revision_v3")
    args = parser.parse_args(argv)
    try:
        manifest = export(args.root.resolve())
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(f"Exported {manifest['counts']['tep_paired_rows']} paired TEP rows and {manifest['counts']['clean_rows']} clean rows "
          f"({manifest['counts']['unavailable_clean_rows']} N/A); no new model calls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
