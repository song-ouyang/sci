"""Post hoc descriptive audit of frozen focal-context and repeat disagreements.

No inference, downloads, scoring-rule changes, confidence intervals or p-values.
All read inputs are from the frozen fourth-stage study; outputs stay here.
"""
import csv
import hashlib
import itertools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/independent_validation"
OUT = Path(__file__).resolve().parent
CONDITIONS = ("clean", "focal_only", "a_front", "a_back", "b_front", "b_back")
CONTRASTS = (("a_front", "b_front"), ("a_back", "b_back"),
             ("a_front", "a_back"), ("b_front", "b_back"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (OUT / name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    protocol = json.loads((BASE / "protocol.json").read_text())
    scores = json.loads((BASE / "analysis/scores.json").read_text())
    manifest = json.loads((BASE / "predictions/frozen_manifest.json").read_text())
    assert scores["protocol_sha256"] == sha(BASE / "protocol.json")
    assert scores["frozen_manifest_sha256"] == sha(BASE / "predictions/frozen_manifest.json")
    focal = tuple(protocol["focal_ids"])
    assert len(focal) == len(set(focal)) == 21
    cases = list(csv.DictReader((BASE / "analysis/cases.csv").open()))
    truth = {}
    for row in cases:
        if row["id"] in focal:
            previous = truth.setdefault(row["id"], row["true_label"])
            assert previous == row["true_label"]
    predictions, requests, files = {}, {}, {}
    accuracy_rows = []
    for model, condition, repeat in itertools.product(protocol["models"], CONDITIONS, (1, 2, 3)):
        key = (model, condition, repeat)
        stem = f"{model}/{condition}_r{repeat}"
        for suffix in ("request.json", "results.json", "metadata.json"):
            relative = f"{stem}_{suffix}"
            path = BASE / "predictions" / relative
            digest = sha(path)
            assert digest == manifest["files"][relative], relative
            files[str(path.relative_to(ROOT))] = digest
        metadata = json.loads((BASE / "predictions" / f"{stem}_metadata.json").read_text())
        assert metadata["status"] == "complete"
        records = json.loads((BASE / "predictions" / f"{stem}_results.json").read_text())
        assert len(records) == len({r["id"] for r in records})
        full_prediction = {r["id"]: r["predicted"] for r in records}
        predictions[key] = {i: full_prediction[i] for i in focal}
        requests[key] = (BASE / "predictions" / f"{stem}_request.json").read_bytes()
        correct = sum(predictions[key][i] == truth[i] for i in focal)
        scored = next(r for r in scores["focal"] if
                      (r["model"], r["condition"], r["repeat"]) == key)
        assert scored["n"] == 21 and scored["correct"] == correct
        for row in cases:
            if (row["model"], row["condition"], int(row["repeat"])) == key:
                assert full_prediction[row["id"]] == row["predicted"]
        accuracy_rows.append({"model": model, "condition": condition, "repeat": repeat,
                              "focal_n": 21, "correct": correct,
                              "full_request_n": len(records)})

    repeats, comparisons, transitions = [], [], []
    for model, condition in itertools.product(protocol["models"], CONDITIONS):
        for left, right in itertools.combinations((1, 2, 3), 2):
            left_key, right_key = (model, condition, left), (model, condition, right)
            assert requests[left_key] == requests[right_key], (left_key, right_key)
            flips = sum(predictions[left_key][i] != predictions[right_key][i] for i in focal)
            repeats.append({"model": model, "condition": condition, "left_repeat": left,
                            "right_repeat": right, "paired_n": 21, "flips": flips,
                            "request_bytes_identical": True})
            for i in focal:
                transitions.append({"kind": "identical_input_repeat", "model": model,
                                    "left_condition": condition, "right_condition": condition,
                                    "left_repeat": left, "right_repeat": right, "id": i,
                                    "left_prediction": predictions[left_key][i],
                                    "right_prediction": predictions[right_key][i],
                                    "flipped": int(predictions[left_key][i] != predictions[right_key][i])})
    for model, (left, right), repeat in itertools.product(protocol["models"], CONTRASTS, (1, 2, 3)):
        left_key, right_key = (model, left, repeat), (model, right, repeat)
        flips = sum(predictions[left_key][i] != predictions[right_key][i] for i in focal)
        comparisons.append({"model": model, "left_condition": left, "right_condition": right,
                            "repeat": repeat, "full_request_n": 42, "paired_n": 21, "flips": flips})
        for i in focal:
            transitions.append({"kind": "fixed_size_context", "model": model,
                                "left_condition": left, "right_condition": right,
                                "left_repeat": repeat, "right_repeat": repeat, "id": i,
                                "left_prediction": predictions[left_key][i],
                                "right_prediction": predictions[right_key][i],
                                "flipped": int(predictions[left_key][i] != predictions[right_key][i])})
    write_csv("identical_input_focal_flips.csv", repeats)
    write_csv("fixed_size_context_focal_flips.csv", comparisons)
    write_csv("focal_correct_counts.csv", accuracy_rows)
    write_csv("focal_disagreement_cases.csv", transitions)
    summary = {
        "status": "post_hoc_descriptive_reanalysis",
        "new_inference_requests": 0,
        "new_source_units": 0,
        "focal_units": 21,
        "primary_study_repeat": 1,
        "all_repeat_pairs_in_this_audit": [[1, 2], [1, 3], [2, 3]],
        "inference_boundary": "Reuse of cases and three calls induces dependence. Repeated-call differences contextualize context contrasts; neither subtraction nor significance testing is performed.",
        "source_checks": {"frozen_request_result_metadata_hashes_verified": len(files),
                          "identical_request_pair_checks": len(repeats),
                          "focal_counts_match_scores": len(accuracy_rows)},
        "identical_input_flips": repeats,
        "fixed_size_context_flips": comparisons,
        "focal_counts": accuracy_rows,
        "input_hashes": files | {
            str(p.relative_to(ROOT)): sha(p) for p in (
                BASE / "protocol.json", BASE / "analysis/scores.json",
                BASE / "analysis/cases.csv", BASE / "predictions/frozen_manifest.json")},
        "analysis_code_sha256": sha(Path(__file__)),
    }
    (OUT / "context_repeat_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["source_checks"]))


if __name__ == "__main__":
    main()
