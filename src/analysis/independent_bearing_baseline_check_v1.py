"""Independent read-only validation of the frozen bearing-baseline follow-up.

Imports no project runner, scorer or data-preparation module. The checker may
refit the ten final selected classifiers after evidence freezing, but never
repeats candidate selection, changes source outputs, or invokes an LLM.
"""

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
import math
from pathlib import Path
import re
import time


REPO = Path(__file__).resolve().parents[2]
LABELS = ("healthy", "outer_race_fault", "inner_race_fault")
REGIMES = ("N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04")
OLD = REPO / "data/paderborn_processed_v1"
DATA = REPO / "data/bearing_baseline_v1"
LLM = REPO / "results/policy_paderborn_v1"


def need(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    need(path.is_file() and not path.is_symlink(), "Missing/symlinked file: " + str(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def child(parent, relative):
    need(isinstance(relative, str) and relative, "Invalid relative path")
    part = Path(relative)
    need(not part.is_absolute() and ".." not in part.parts, "Unsafe path: " + relative)
    path = parent / part
    need(path.resolve().is_relative_to(parent.resolve()), "Path escapes evidence root")
    need(not any(parent.joinpath(*part.parts[:n]).is_symlink() for n in range(1, len(part.parts) + 1)), "Symlink in path")
    return path


def verify_map(parent, hashes, evidence):
    need(isinstance(hashes, dict) and hashes, "Empty/missing evidence hash map")
    for relative, expected in hashes.items():
        path = child(parent, relative)
        actual = sha(path)
        need(actual == expected, "Evidence hash mismatch: " + str(path))
        evidence[str(path.resolve())] = actual


def compare(actual, expected, where, mismatches):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            mismatches.append({"at": where, "error": "object keys differ", "actual_keys": sorted(actual) if isinstance(actual, dict) else None, "expected_keys": sorted(expected)})
            return
        for key in expected:
            compare(actual[key], expected[key], where + "/" + str(key), mismatches)
    elif isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            mismatches.append({"at": where, "error": "array length/type differs"})
            return
        for index, value in enumerate(expected):
            compare(actual[index], value, where + "/" + str(index), mismatches)
    elif type(expected) in {int, float}:
        if type(actual) not in {int, float} or not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            mismatches.append({"at": where, "actual": actual, "expected": expected})
    elif type(actual) is not type(expected) or actual != expected:
        mismatches.append({"at": where, "actual": actual, "expected": expected})


def manual_metrics(truth, predictions):
    """No sklearn metric functions: fixed-order confusion and direct fractions."""
    need(len(truth) == len(predictions) and len(truth) > 0, "Metric denominator absent")
    need(set(truth) <= set(LABELS) and set(predictions) <= set(LABELS), "Unknown metric label")
    matrix = [[0] * len(LABELS) for _ in LABELS]
    for actual, predicted in zip(truth, predictions):
        matrix[LABELS.index(actual)][LABELS.index(predicted)] += 1
    n = len(truth)
    correct = sum(matrix[i][i] for i in range(len(LABELS)))
    recalls, f1s = [], []
    for i in range(len(LABELS)):
        tp = matrix[i][i]
        actual_n = sum(matrix[i])
        predicted_n = sum(row[i] for row in matrix)
        recalls.append(tp / actual_n if actual_n else 0.0)
        f1s.append(2 * tp / (actual_n + predicted_n) if actual_n + predicted_n else 0.0)
    return {"n": n, "correct": correct, "accuracy": correct / n,
            "balanced_accuracy": sum(recalls) / len(LABELS), "macro_f1": sum(f1s) / len(LABELS),
            "confusion_matrix": matrix, "per_class_recall": dict(zip(LABELS, recalls))}


def group_metrics(rows, predicted, field):
    need(len(rows) == len(predicted), "Grouped prediction count differs")
    answer = {}
    for group in sorted({row[field] for row in rows}):
        selected = [(row, pred) for row, pred in zip(rows, predicted) if row[field] == group]
        answer[group] = manual_metrics([row["label"] for row, _ in selected], [pred for _, pred in selected])
    return answer


def check_data(evidence):
    """Audit exact frozen quantized features and physical-specimen separation."""
    plan = read(DATA / "plan.json")
    need(sha(DATA / "plan.json") == (DATA / "plan.sha256").read_text().strip(), "Data plan digest differs")
    verify_map(REPO, plan["code_sha256"], evidence)
    verify_map(REPO, plan["original_input_sha256"], evidence)
    data_audit = read(DATA / "audit.json")
    need(data_audit["plan_sha256"] == sha(DATA / "plan.json"), "Data audit/plan mismatch")
    need(data_audit["schema"] == "bearing-baseline-preprocessing-audit-v2", "Expected explicit corrected-extraction audit")
    amendment_path = DATA / "extraction_amendment_v2.json"
    amendment = read(amendment_path)
    need(sha(amendment_path) == (DATA / "extraction_amendment_v2.sha256").read_text().strip() == data_audit["extraction_amendment_sha256"], "Extraction amendment digest differs")
    need(amendment["prior_plan_sha256"] == sha(DATA / "plan.json") and amendment["prior_failure_sha256"] == sha(DATA / "extraction_failure.json"), "Extraction amendment history differs")
    need(amendment["schema"] == "bearing-baseline-extraction-amendment-v2", "Unknown extraction amendment")
    need(data_audit["code_sha256"] == {**plan["code_sha256"], **amendment["code_sha256"]}, "Extraction code lineage incomplete")
    need(data_audit["original_input_sha256"] == plan["original_input_sha256"], "Extraction input lineage differs")
    verify_map(REPO, data_audit["code_sha256"], evidence)
    need(data_audit["original48_recomputed_all28_strings_equal"] is True, "Original feature reconstruction did not match")
    original_checks = data_audit["original48_recomputation"]
    need(len(original_checks) == 48 and len({row["id"] for row in original_checks}) == 48 and all(row["all28_strings_equal"] is True for row in original_checks), "Incomplete original feature reconstruction evidence")
    for path in (amendment_path, DATA / "extraction_amendment_v2.sha256", DATA / "extraction_failure.json", DATA / "plan.sha256"):
        evidence[str(path.resolve())] = sha(path)
    verify_map(DATA, data_audit["output_sha256"], evidence)
    development, test = read(DATA / "development.json"), read(DATA / "test.json")
    old_dataset, old_truth, old_selection = read(OLD / "dataset.json"), read(OLD / "truth.json"), read(OLD / "selection.json")
    dictionary = old_dataset["dictionary"]
    need(len(dictionary) == 28 and development["dictionary"] == test["dictionary"] == dictionary == plan["dictionary"], "28-feature dictionary mismatch")
    need(len({row["name"] for row in dictionary}) == 28, "Duplicate feature names")
    dev, held = development["records"], test["records"]
    need(len(dev) == 120 and len(held) == 36, "Expected 120 development / 36 test records")
    need(len({row["id"] for row in dev + held}) == 156, "Duplicate/leaking record ID")
    for row in dev + held:
        need(set(row) == {"id", "label", "bearing", "regime", "repetition", "member", "values"}, "Unexpected record fields")
        need(isinstance(row["id"], str) and re.fullmatch(r"[0-9a-f]{8}", row["id"]), "Nonanonymous record ID")
        need(type(row["repetition"]) is int and 1 <= row["repetition"] <= 20, "Invalid source repetition")
        need(row["member"] == f'{row["bearing"]}/{row["regime"]}_{row["bearing"]}_{row["repetition"]}.mat', "Record source identity differs")
        need(row["label"] in LABELS and row["regime"] in REGIMES, "Unknown class/regime")
        need(len(row["values"]) == 28 and all(isinstance(v, str) and re.fullmatch(r"-?\d+\.\d{6}", v) and math.isfinite(float(v)) for v in row["values"]), "Not 28 finite six-decimal strings")
    dev_bearings, test_bearings = {row["bearing"] for row in dev}, {row["bearing"] for row in held}
    need(len(dev_bearings) == 6 and len(test_bearings) == 9 and not dev_bearings & test_bearings, "Physical bearing leakage")
    label_map = {}
    for row in dev + held:
        need(row["bearing"] not in label_map or label_map[row["bearing"]] == row["label"], "Inconsistent specimen truth")
        label_map[row["bearing"]] = row["label"]
    need(Counter(label_map[b] for b in dev_bearings) == Counter({label: 2 for label in LABELS}), "Development specimen classes differ")
    need(Counter(label_map[b] for b in test_bearings) == Counter({label: 3 for label in LABELS}), "Test specimen classes differ")
    need(Counter((r["bearing"], r["regime"]) for r in dev) == Counter({(b, regime): 5 for b in dev_bearings for regime in REGIMES}), "Development regime coverage differs")
    need(Counter((r["bearing"], r["regime"]) for r in held) == Counter({(b, regime): 1 for b in test_bearings for regime in REGIMES}), "Test regime coverage differs")
    old_by_id = {row["id"]: row for row in old_selection["records"]}
    old_by_bearing = {row["bearing"]: row for row in old_selection["records"]}
    need({row["id"] for row in original_checks} == set(old_by_id), "Original48 reconstruction IDs differ")
    old_values = {row["id"]: row["values"] for row in old_dataset["panels"][0]["records"]}
    need([row["id"] for row in held] == old_dataset["panels"][0]["focal_ids"] == plan["test_ids"], "Frozen test order changed")
    for row in held:
        source = old_by_id[row["id"]]
        need(source["role"] == "focal" and row["values"] == old_values[row["id"]], "Frozen test values changed")
        for key in ("id", "label", "bearing", "regime", "repetition", "member"):
            need(row[key] == source[key], "Frozen test identity changed: " + key)
        need(row["label"] == old_truth["labels"][row["id"]], "Frozen test truth changed")
    need(dev_bearings == {r["bearing"] for r in old_selection["records"] if r["role"] == "background"}, "Development specimen set changed")
    # Independently recreate the fixed top-five selection rather than import its generator.
    seed = plan["seed"]
    key = lambda *parts: hashlib.sha256("|".join(map(str, (seed,) + parts)).encode()).hexdigest()
    expected_members = set()
    for bearing in dev_bearings:
        for regime in REGIMES:
            repeats = sorted(range(1, 21), key=lambda rep: key("record-choice", bearing, regime, rep))[:5]
            expected_members.update(f"{bearing}/{regime}_{bearing}_{rep}.mat" for rep in repeats)
    need({r["member"] for r in dev} == expected_members, "Top-five record selection changed")
    for row in dev:
        need(row["id"] == key("anonymous-record", row["bearing"], row["regime"], row["repetition"])[:8], "Development ID changed")
        need(row["label"] == old_by_bearing[row["bearing"]]["label"], "Development specimen label differs from original source")
    planned_development = {row["id"]: row for row in plan["development_records"]}
    need(set(planned_development) == {row["id"] for row in dev}, "Development records differ from frozen selection plan")
    for row in dev:
        need(all(row[field] == planned_development[row["id"]][field] for field in ("label", "bearing", "regime", "repetition", "member")), "Development metadata differs from frozen plan")
    need(sum(row["id"] in old_values for row in dev) == 12, "Expected 12 reused background records")
    for row in dev:
        if row["id"] in old_values:
            need(row["values"] == old_values[row["id"]], "Reused six-decimal data changed")
    for path in (DATA / "plan.json", DATA / "audit.json"):
        evidence[str(path.resolve())] = sha(path)
    return dev, held, dictionary, label_map


def expected_validation_groups(development):
    per_class = {label: sorted({r["bearing"] for r in development if r["label"] == label}) for label in LABELS}
    need(all(len(value) == 2 for value in per_class.values()), "Need two development bearings per class")
    return {frozenset(group) for group in itertools.product(*(per_class[label] for label in LABELS))}


def frozen_baseline(base, evidence):
    manifest_path = base / "prediction_manifest.json"
    need(manifest_path.is_file(), "No prediction freeze: refusing uncompleted baseline")
    need(not (base / "runner.lock").exists(), "Active/interrupted runner lock")
    need(sha(manifest_path) == (base / "prediction_manifest.sha256").read_text().strip(), "Prediction manifest changed")
    protocol, manifest = read(base / "protocol.json"), read(manifest_path)
    need(sha(base / "protocol.json") == (base / "protocol.sha256").read_text().strip(), "Protocol changed")
    need(protocol["schema"] == "bearing-baseline-protocol-v1" and manifest["schema"] == "bearing-baseline-prediction-freeze-v1", "Unknown schema")
    need(manifest["protocol_sha256"] == sha(base / "protocol.json"), "Freeze/protocol mismatch")
    need(protocol["labels"] == list(LABELS) and protocol["regimes"] == list(REGIMES), "Frozen class/regime enumeration changed")
    verify_map(REPO, protocol["source_sha256"], evidence)
    verify_map(base, protocol["local_sha256"], evidence)
    expected_files = {"started.json", "cv.json", "selections.json", "structural_checks.json", "predictions.csv", "run.json"}
    need(set(manifest["files_sha256"]) == expected_files, "Incomplete final-fit freeze")
    need({p.name for p in (base / "fit").iterdir()} == expected_files, "Unexpected/missing fit evidence")
    verify_map(base / "fit", manifest["files_sha256"], evidence)
    for path in (manifest_path, base / "prediction_manifest.sha256", base / "protocol.json", base / "protocol.sha256"):
        evidence[str(path.resolve())] = sha(path)
    return protocol, manifest


def scenarios(development, test):
    answer = {"primary": (None, development, test)}
    for regime in REGIMES:
        training = [row for row in development if row["regime"] != regime]
        held = [row for row in test if row["regime"] == regime]
        need(len(training) == 90 and len(held) == 9, "Wrong LORO dimensions")
        answer["loro_" + regime] = (regime, training, held)
    return answer


def expected_folds(training):
    groups = [sorted({row["bearing"] for row in training if row["label"] == label}) for label in LABELS]
    need(all(len(group) == 2 for group in groups), "Class specimen count in folds differs")
    answer = []
    for index, validation_bearings in enumerate(itertools.product(*groups), 1):
        validation = [i for i, row in enumerate(training) if row["bearing"] in validation_bearings]
        fitting = [i for i, row in enumerate(training) if row["bearing"] not in validation_bearings]
        answer.append({"fold": index, "train": fitting, "validation": validation,
                       "train_bearings": sorted({training[i]["bearing"] for i in fitting}),
                       "validation_bearings": list(validation_bearings)})
    need(len(answer) == 8 and {frozenset(x["validation_bearings"]) for x in answer} == expected_validation_groups(training), "Incomplete validation combinations")
    return answer


def check_candidate_table(base, protocol, tasks, mismatches):
    cv = read(base / "fit/cv.json")
    selections = read(base / "fit/selections.json")
    need(set(protocol["grids"]) == {"SVM", "RF"}, "Unexpected model family")
    expected_keys = {task + "/" + model for task in tasks for model in protocol["grids"]}
    need(set(cv) == expected_keys and len(selections) == 10, "Incomplete candidate/final-selection table")
    selected = {row["task"] + "/" + row["model"]: row for row in selections}
    need(len(selected) == len(selections) and set(selected) == expected_keys, "Duplicate/missing final selections")
    need(sum(len(grid) for grid in protocol["grids"].values()) == 10, "Expected ten candidate settings across families")
    checked_folds, chosen = [], {}
    expected_protocol_folds = {}
    for task, (regime, training, test) in tasks.items():
        folds = expected_folds(training)
        expected_protocol_folds[task] = folds
        training_regimes = sorted({row["regime"] for row in training})
        need(regime is None or regime not in training_regimes, "Held-out regime used for fitting/tuning")
        for model, grid in protocol["grids"].items():
            key = task + "/" + model
            candidates = cv[key]
            need(len(candidates) == len(grid), "Missing candidate: " + key)
            means = []
            for candidate_index, (candidate, params) in enumerate(zip(candidates, grid)):
                need(candidate["candidate_index"] == candidate_index and candidate["params"] == params, "Candidate enumeration/parameters changed")
                need(len(candidate["folds"]) == 8, "Missing validation fold")
                accuracies = []
                for found, frozen_fold in zip(candidate["folds"], folds):
                    train_ids = [training[i]["id"] for i in frozen_fold["train"]]
                    valid_ids = [training[i]["id"] for i in frozen_fold["validation"]]
                    for field in ("fold", "train_bearings", "validation_bearings"):
                        need(found[field] == frozen_fold[field], "Wrong fold specimen allocation: " + field)
                    need(found["train_ids"] == train_ids and found["validation_ids"] == valid_ids, "Wrong CV row identities/order")
                    need(not set(found["train_bearings"]) & set(found["validation_bearings"]), "CV specimen leakage")
                    need(not set(train_ids) & set(valid_ids) and set(train_ids + valid_ids) == {r["id"] for r in training}, "CV row overlap/missing data")
                    need(found["training_regimes"] == training_regimes, "CV regime inventory changed")
                    need(len(train_ids) == len(valid_ids) == (60 if regime is None else 45), "Wrong fold record count")
                    predictions = found["validation_predictions"]
                    truth = [training[i]["label"] for i in frozen_fold["validation"]]
                    measured = manual_metrics(truth, predictions)
                    for field, value in (("correct", measured["correct"]), ("total", measured["n"]), ("accuracy", measured["accuracy"])):
                        compare(found[field], value, key + "/candidate" + str(candidate_index) + "/fold" + str(found["fold"]) + "/" + field, mismatches)
                    accuracies.append(measured["accuracy"])
                    checked_folds.append({"task": task, "model": model, "candidate_index": candidate_index,
                        "fold": found["fold"], "correct": measured["correct"], "total": measured["n"],
                        "accuracy": measured["accuracy"], "train_bearings": found["train_bearings"], "validation_bearings": found["validation_bearings"]})
                mean = sum(accuracies) / 8
                compare(candidate["mean_accuracy"], mean, key + "/candidate" + str(candidate_index) + "/mean_accuracy", mismatches)
                means.append(mean)
            winner = min(range(len(means)), key=lambda i: (-means[i], i))
            saved = selected[key]
            expected = {"task": task, "held_out_regime": regime, "model": model, "candidate_index": winner,
                        "params": grid[winner], "cv_mean_accuracy": means[winner], "train_records": len(training),
                        "test_records": len(test), "training_bearings": sorted({row["bearing"] for row in training}),
                        "training_regimes": training_regimes}
            for field, value in expected.items():
                compare(saved.get(field), value, key + "/selection/" + field, mismatches)
            for field in ("final_fit_seconds", "prediction_seconds", "task_model_seconds_including_checks"):
                need(type(saved[field]) in {int, float} and math.isfinite(saved[field]) and saved[field] >= 0, "Invalid timing record")
            chosen[key] = expected
    compare(protocol["folds"], expected_protocol_folds, "protocol/folds", mismatches)
    need(len(checked_folds) == 400, "Expected all 400 candidate-fold evaluations")
    return chosen, checked_folds


def csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        need(reader.fieldnames is not None and len(reader.fieldnames) == len(set(reader.fieldnames)), "Missing/duplicate CSV columns")
    return rows


def published_metrics(rows):
    values = manual_metrics([row["truth"] for row in rows], [row["predicted"] for row in rows])
    by = {}
    for field in ("bearing", "regime"):
        by[field] = []
        for value in sorted({row[field] for row in rows}):
            group = [row for row in rows if row[field] == value]
            by[field].append({field: value, "correct": sum(r["truth"] == r["predicted"] for r in group), "total": len(group)})
    bearing_correct = [row["correct"] for row in by["bearing"]]
    return {"correct": values["correct"], "total": values["n"], "accuracy": values["accuracy"],
            "macro_f1": values["macro_f1"], "labels": list(LABELS), "confusion_matrix_truth_rows": values["confusion_matrix"],
            "per_class_recall": {label: values["per_class_recall"][label] if any(row["truth"] == label for row in rows) else None for label in LABELS},
            "by_bearing": by["bearing"], "by_regime": by["regime"],
            "bearing_correct_count_range": [min(bearing_correct), max(bearing_correct)]}


def final_predictions(base, tasks):
    rows = csv_rows(base / "fit/predictions.csv")
    need(len(rows) == 216, "Expected 216 baseline predictions")
    need(all(set(row) == {"task", "model", "id", "bearing", "regime", "predicted"} for row in rows), "Unknown prediction fields")
    by = {}
    for task, (_, training, testing) in tasks.items():
        expected = {row["id"]: row for row in testing}
        for model in ("SVM", "RF", "Constant"):
            selected = [row for row in rows if row["task"] == task and row["model"] == model]
            need(len(selected) == len(expected) and {r["id"] for r in selected} == set(expected), "Missing/duplicate task predictions")
            scored = []
            for row in selected:
                source = expected[row["id"]]
                need(row["predicted"] in LABELS and row["bearing"] == source["bearing"] and row["regime"] == source["regime"], "Prediction identity/label mismatch")
                if model == "Constant":
                    counts = Counter(r["label"] for r in training)
                    constant = min(LABELS, key=lambda label: (-counts[label], LABELS.index(label)))
                    need(row["predicted"] == constant, "Incorrect training-majority reference")
                scored.append({**row, "truth": source["label"]})
            by[task + "/" + model] = scored
    need(set((row["task"], row["model"]) for row in rows) == set(itertools.product(tasks, ("SVM", "RF", "Constant"))), "Unexpected task/model prediction")
    for model in ("SVM", "RF", "Constant"):
        by["loro_combined/" + model] = [row for regime in REGIMES for row in by["loro_" + regime + "/" + model]]
    return rows, by


def decode_raw(text, expected_ids):
    text = text.strip()
    if text.startswith("```"):
        found = re.fullmatch(r"```(?:json)?\s*\n(.*)\n```", text, re.DOTALL)
        need(found is not None, "Invalid raw JSON fence")
        text = found.group(1)
    values = json.loads(text)
    need(isinstance(values, list) and len(values) == len(expected_ids), "Incomplete LLM raw array")
    for item, identity in zip(values, expected_ids):
        need(isinstance(item, dict) and set(item) == {"id", "predicted"} and item["id"] == identity and item["predicted"] in LABELS, "LLM raw order/label mismatch")
    return {item["id"]: item["predicted"] for item in values}


def old_llm_calls(test, evidence):
    """Reparse the twelve original completed calls; do not trust scored labels."""
    manifest_path = LLM / "predictions/frozen_manifest.json"
    need(manifest_path.is_file() and not (LLM / "runner.lock").exists(), "Old LLM study not frozen/idle")
    manifest, protocol = read(manifest_path), read(LLM / "protocol.json")
    need(sha(LLM / "protocol.json") == (LLM / "protocol.sha256").read_text().strip() == manifest["protocol_sha256"], "Old LLM protocol changed")
    verify_map(REPO, protocol["code_sha256"], evidence)
    verify_map(REPO, protocol["source_files_sha256"], evidence)
    verify_map(LLM / "predictions", manifest["files"], evidence)
    inventory = {p.relative_to(LLM / "predictions").as_posix(): sha(p) for p in (LLM / "predictions").rglob("*") if p.is_file() and p != manifest_path}
    need(inventory == manifest["files"], "Old LLM inventory differs")
    for name, key in (("inference/batches.json", "input_sha256"), ("inference/dataset.json", "dataset_sha256"),
                      ("schedule.json", "schedule_sha256"), ("source_audit.json", "source_audit_sha256"), ("answers/truth.json", "answers_sha256")):
        path = LLM / name
        need(sha(path) == protocol[key], "Old LLM input digest mismatch")
        evidence[str(path.resolve())] = sha(path)
    for name, expected in protocol["prompts_sha256"].items():
        path = LLM / "inference/prompts" / (name + ".txt")
        need(sha(path) == expected, "Old LLM prompt changed")
        evidence[str(path.resolve())] = sha(path)
    groups = read(LLM / "inference/batches.json")["groups"]
    pairs = read(LLM / "schedule.json")["pairs"]
    truth = read(LLM / "answers/truth.json")["labels"]
    metadata = {row["id"]: row for row in test}
    folder = LLM / "predictions" / protocol["model"]
    calls = {}
    for pair in pairs:
        for name, condition in zip(pair["names"], pair["conditions"]):
            meta = read(folder / (name + "_metadata.json"))
            need(meta["status"] == "complete" and len(meta["attempts"]) == 1 and meta["tools_observed"] == 0, "LLM comparator requires twelve valid calls")
            attempt = meta["attempts"][0]
            need(attempt["response_file"] == name + "_attempt1_response.txt", "Unexpected original attempt")
            raw = (folder / attempt["response_file"]).read_text()
            parsed = decode_raw(raw, [row["id"] for row in groups[name]["samples"]])
            saved = read(folder / (name + "_results.json"))
            need({x["id"]: x["predicted"] for x in saved} == parsed, "Old LLM parsed/raw mismatch")
            events = [json.loads(line) for line in (folder / attempt["events_file"]).read_text().splitlines() if line.strip()]
            finals = [e["item"]["text"] for e in events if e.get("type") == "item.completed" and e.get("item", {}).get("type") == "agent_message"]
            need(len(finals) == 1 and finals[0].strip() == raw.strip(), "Old LLM event/raw mismatch")
            rows = []
            for identity, source in metadata.items():
                need(truth[identity] == source["label"], "Old LLM/test truth mismatch")
                rows.append({"id": identity, "bearing": source["bearing"], "regime": source["regime"],
                             "truth": truth[identity], "predicted": parsed[identity]})
            calls[name] = {"condition": condition, "round": pair["round"], "repeat": pair["repeat"], "rows": rows}
    need(len(calls) == 12 and all(len(call["rows"]) == 36 for call in calls.values()), "Missing original LLM focal calls")
    old_csv = csv_rows(LLM / "analysis/predictions.csv")
    focal_csv = [row for row in old_csv if row["focal"].lower() == "true"]
    need(len(focal_csv) == 432 and len({(r["group"], r["id"]) for r in focal_csv}) == 432, "Old scored focal inventory differs")
    for row in focal_csv:
        call = calls[row["group"]]
        expected = next(r for r in call["rows"] if r["id"] == row["id"])
        need(row["truth"] == expected["truth"] and row["predicted"] == expected["predicted"] and row["condition"] == call["condition"] and int(row["round"]) == call["round"] and int(row["repeat"]) == call["repeat"], "Old CSV differs from raw call")
    evidence[str(manifest_path.resolve())] = sha(manifest_path)
    evidence[str((LLM / "protocol.json").resolve())] = sha(LLM / "protocol.json")
    evidence[str((LLM / "analysis/predictions.csv").resolve())] = sha(LLM / "analysis/predictions.csv")
    return calls


def comparisons(by, calls):
    rows = []
    summaries = {}
    for name, call in sorted(calls.items()):
        summaries[name] = {key: call[key] for key in ("condition", "round", "repeat")}
        summaries[name].update(published_metrics(call["rows"]))
        llm = {row["id"]: row for row in call["rows"]}
        for task in ("primary", "loro_combined"):
            for model in ("SVM", "RF", "Constant"):
                reference = by[task + "/" + model]
                rows.append({"call": name, "condition": call["condition"], "round": call["round"], "repeat": call["repeat"],
                    "baseline_task": task, "model": model, "n": 36,
                    "baseline_correct": sum(r["truth"] == r["predicted"] for r in reference),
                    "llm_correct": summaries[name]["correct"],
                    "prediction_disagreements": sum(r["predicted"] != llm[r["id"]]["predicted"] for r in reference),
                    "baseline_only_correct": sum(r["predicted"] == r["truth"] and llm[r["id"]]["predicted"] != r["truth"] for r in reference),
                    "llm_only_correct": sum(r["predicted"] != r["truth"] and llm[r["id"]]["predicted"] == r["truth"] for r in reference)})
    return summaries, rows


def compare_csv(path, expected, mismatches):
    actual = csv_rows(path)
    if len(actual) != len(expected):
        mismatches.append({"at": path.name, "actual_rows": len(actual), "expected_rows": len(expected)})
        return
    for index, (found, wanted) in enumerate(zip(actual, expected)):
        if set(found) != set(wanted):
            mismatches.append({"at": path.name + "/" + str(index), "error": "CSV columns differ"})
            continue
        parsed = {}
        for key, value in wanted.items():
            text = found[key]
            if type(value) is int:
                parsed[key] = int(text) if re.fullmatch(r"-?\d+", text or "") else text
            elif type(value) is float:
                try:
                    parsed[key] = float(text)
                except ValueError:
                    parsed[key] = text
            else:
                parsed[key] = text
        compare(parsed, wanted, path.name + "/" + str(index), mismatches)


def refit_selected(protocol, tasks, chosen, by, original_environment, mismatches):
    # External scientific libraries are used only for the requested final refits,
    # never for selecting candidates or calculating evaluation metrics.
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")}
    need(versions == original_environment["packages"], "Refit package versions differ from frozen run")
    need(protocol["svm_fixed"]["kernel"] == "rbf" and protocol["svm_fixed"]["probability"] is False, "Unexpected SVM settings")
    need(protocol["rf_fixed"] == {"n_estimators": 200, "random_state": protocol["seed"], "n_jobs": 1}, "Unexpected RF settings")
    reports = []
    for task, (_, training, testing) in tasks.items():
        x = np.asarray([row["values"] for row in training], dtype=float)
        xt = np.asarray([row["values"] for row in testing], dtype=float)
        y = np.asarray([row["label"] for row in training])
        for model in ("SVM", "RF"):
            key = task + "/" + model
            params = chosen[key]["params"]
            if model == "SVM":
                fitted = Pipeline([("standardize", StandardScaler()), ("svm", SVC(C=params["C"], gamma=params["gamma"], kernel="rbf", probability=False, random_state=protocol["seed"]))])
            else:
                fitted = RandomForestClassifier(**protocol["rf_fixed"], **params)
            begin = time.perf_counter()
            fitted.fit(x, y)
            predicted = list(map(str, fitted.predict(xt)))
            expected_map = {row["id"]: row["predicted"] for row in by[key]}
            expected = [expected_map[row["id"]] for row in testing]
            compare(predicted, expected, "independent_refit/" + key, mismatches)
            invariants = {
                "reversed_row_order": list(map(str, fitted.predict(xt[::-1])))[::-1] == predicted,
                "single_row_calls": [str(fitted.predict(row.reshape(1, -1))[0]) for row in xt] == predicted,
                "unlabeled_background_appended": list(map(str, fitted.predict(np.concatenate((xt, x)))))[:len(xt)] == predicted,
            }
            need(all(invariants.values()), "Independent refit row-prediction invariant failed")
            reports.append({"task": task, "model": model, "params": params, "train_records": len(training),
                "test_records": len(testing), "predictions": [{"id": row["id"], "predicted": label} for row, label in zip(testing, predicted)],
                "mismatch_count": sum(a != b for a, b in zip(predicted, expected)), "structural_checks": invariants,
                "independent_fit_and_check_seconds": time.perf_counter() - begin})
    need(len(reports) == 10, "Expected exactly ten final refits")
    return reports, versions


def check(base):
    evidence, mismatches = {}, []
    protocol, manifest = frozen_baseline(base, evidence)
    development, test, dictionary, labels = check_data(evidence)
    need(dictionary == protocol["dictionary"], "Protocol/source features differ")
    inputs = read(base / "test_inputs.json")["records"]
    answers = read(base / "test_answers.json")["labels"]
    need(inputs == [{k: v for k, v in row.items() if k != "label"} for row in test], "Label-free test features/identity differ")
    need(answers == {row["id"]: row["label"] for row in test}, "Separated test truth differs")
    tasks = scenarios(development, test)
    chosen, fold_checks = check_candidate_table(base, protocol, tasks, mismatches)
    prediction_rows, by = final_predictions(base, tasks)
    baseline = {name: published_metrics(rows) for name, rows in by.items()}
    calls = old_llm_calls(test, evidence)
    call_summaries, paired_rows = comparisons(by, calls)
    run = read(base / "fit/run.json")
    need(run["cv_fits"] == 400 and run["final_fits"] == 10 and run["truth_parsed_during_fit"] is False and run["new_llm_calls"] == 0, "Unexpected run-accounting flags")
    need(run["protocol_sha256"] == sha(base / "protocol.json") and run["source_sha256"] == protocol["source_sha256"] and run["local_sha256"] == protocol["local_sha256"], "Run evidence not bound to protocol")
    need(datetime.fromisoformat(run["started_at_utc"]) <= datetime.fromisoformat(run["completed_at_utc"]) <= datetime.fromisoformat(manifest["frozen_at_utc"]), "Fit/freeze chronology invalid")
    structural = read(base / "fit/structural_checks.json")
    need(len(structural) == 10 and {r["task"] + "/" + r["model"] for r in structural} == set(chosen), "Missing structural checks")
    need(all(r[k] is True for r in structural for k in ("reversed_row_order", "single_row_calls", "unlabeled_background_appended")), "Recorded structural check failed")
    dimensions = {"development_records": 120, "development_bearings": 6, "test_records": 36, "test_bearings": 9,
                  "physical_rigs": 1, "inner_folds": 8, "loro_tasks": 4, "loro_train_records": 90, "loro_test_records": 9}
    compare(protocol["dimensions"], dimensions, "protocol/dimensions", mismatches)
    scores_path = base / "analysis/scores.json"
    scores = read(scores_path)
    for key, expected in {"baseline": baseline, "llm_calls": call_summaries, "dimensions": dimensions,
                          "protocol_sha256": sha(base / "protocol.json"), "prediction_manifest_sha256": sha(base / "prediction_manifest.json"),
                          "source_sha256": protocol["source_sha256"], "uniform_prior_expected_accuracy": 1 / 3,
                          "uniform_prior_expected_correct_out_of_36": 12}.items():
        compare(scores.get(key), expected, "scores/" + key, mismatches)
    need(scores["schema"] == "bearing-baseline-scores-v1", "Unknown score schema")
    need(datetime.fromisoformat(scores["generated_at_utc"]) >= datetime.fromisoformat(manifest["frozen_at_utc"]), "Scores predate prediction freeze")
    lookup = {row["id"]: row for row in test}
    scored_predictions = [{**row, "truth": lookup[row["id"]]["label"]} for row in prediction_rows]
    compare_csv(base / "analysis/scored_predictions.csv", scored_predictions, mismatches)
    compare_csv(base / "analysis/llm_call_comparisons.csv", paired_rows, mismatches)
    for path in (scores_path, base / "analysis/scored_predictions.csv", base / "analysis/llm_call_comparisons.csv"):
        evidence[str(path.resolve())] = sha(path)
    refits, versions = refit_selected(protocol, tasks, chosen, by, run["environment"], mismatches)
    for filename, expected in evidence.items():
        need(sha(Path(filename)) == expected, "Evidence changed while checking: " + filename)
    return {"schema": "independent-bearing-baseline-check-v1", "status": "PASS" if not mismatches else "FAIL",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(), "root": str(base), "checker_sha256": sha(Path(__file__)),
        "prediction_frozen_at_utc": manifest["frozen_at_utc"], "protocol_sha256": sha(base / "protocol.json"),
        "checked_file_count": len(evidence), "checked_files_sha256": evidence, "mismatch_count": len(mismatches), "mismatches": mismatches,
        "data_checks": {**dimensions, "dictionary_28_columns_exact": True, "all_values_six_decimal_strings": True,
            "original_test_values_order_metadata_truth_exact": True, "development_test_bearings_disjoint": True,
            "top_five_record_selection_independently_recreated": True, "development_original_records_reused": 12,
            "eight_validation_combinations_per_task": True, "loro_excludes_target_regime_from_training_and_validation": True},
        "candidate_fold_count": len(fold_checks), "recomputed_candidate_folds": fold_checks, "independently_selected_configs": chosen,
        "prediction_rows_checked": len(prediction_rows), "llm_call_count": len(calls), "llm_comparison_rows_checked": len(paired_rows),
        "recomputed_baseline_metrics": baseline, "recomputed_llm_call_metrics": call_summaries, "recomputed_llm_comparisons": paired_rows,
        "independent_final_refits": refits, "refit_package_versions": versions,
        "method": "No project-module imports. Raw LLM response parsing; integer confusion/F1 arithmetic; all 400 stored validation predictions checked without refitting candidates; ten final classifiers reconstructed with frozen settings and independently fitted.",
        "limitations": ["This numerical and split audit does not establish factory generalization, calibrated physical units, or freedom from all collection confounds.",
                        "The original supervised study was designed after known LLM outcomes; refit agreement does not reverse that history.",
                        "The same 36 test records and nine test bearings recur across LLM calls and scenarios; no new independence or p value is inferred.",
                        "Raw MAT feature extraction is bound through the existing data audit; this checker verifies serialized input identity, not a new waveform feature implementation."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO / "results/bearing_baseline_v1")
    args = parser.parse_args()
    report = check(args.root.resolve())
    output = REPO / "results/bearing_baseline_independent_check.json"
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "output": str(output), "mismatches": report["mismatch_count"],
                      "candidate_folds": report["candidate_fold_count"], "independent_final_refits": len(report["independent_final_refits"])}))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
