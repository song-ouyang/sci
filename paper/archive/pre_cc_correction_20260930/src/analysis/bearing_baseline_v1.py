"""Offline, outcome-informed supervised bearing benchmark with a frozen split.

prepare-protocol separates known test labels from inputs. fit parses development
labels and label-free test inputs only (other bound files are byte-hash audited).
score opens the answer key only after checking the frozen prediction manifest.
Nothing is fitted on import. Existing artifacts are never replaced by fit.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
import re
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "src/analysis/bearing_baseline_v1.py"
TESTS = ROOT / "tests/test_bearing_baseline_v1.py"
DATA = ROOT / "data/bearing_baseline_v1"
OUT = ROOT / "results/bearing_baseline_v1"
ORIGINAL = ROOT / "data/paderborn_processed_v1/dataset.json"
LLM = ROOT / "results/policy_paderborn_v1/analysis/predictions.csv"
SEED = 20260928
LABELS = ["healthy", "outer_race_fault", "inner_race_fault"]
REGIMES = ["N15_M07_F10", "N09_M07_F10", "N15_M01_F10", "N15_M07_F04"]
VALUE = re.compile(r"^-?\d+\.\d{6}$")
RECORD_KEYS = {"id", "values", "label", "bearing", "regime", "repetition", "member"}
SVM_GRID = [{"C": c, "gamma": g} for c in (0.1, 1, 10) for g in ("scale", 0.1)]
RF_GRID = [{"max_features": f, "min_samples_leaf": n} for f in ("sqrt", 1.0) for n in (1, 3)]
GRIDS = {"SVM": SVM_GRID, "RF": RF_GRID}


def need(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def relative(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def resolve(path):
    return ROOT / path


def write_csv(path, rows):
    need(bool(rows), "Cannot write an empty table")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def validate_rows(document, dictionary, *, labels, counts=None):
    need(document.get("dictionary") == dictionary, "Feature dictionary/order/units mismatch")
    need(len(dictionary) == 28 and len({d["name"] for d in dictionary}) == 28, "Expected 28 unique columns")
    need(all(set(d) == {"name", "unit"} for d in dictionary), "Dictionary schema changed")
    rows = document["records"]
    expected = RECORD_KEYS if labels else RECORD_KEYS - {"label"}
    need(bool(rows), "Empty records")
    for r in rows:
        need(set(r) == expected, "Unexpected record keys, including forbidden test label")
        need(isinstance(r["id"], str) and re.fullmatch(r"[0-9a-f]{8}", r["id"]), "Nonanonymous ID")
        need(r["regime"] in REGIMES and isinstance(r["repetition"], int) and 1 <= r["repetition"] <= 20, "Invalid source metadata")
        need(r["member"] == f'{r["bearing"]}/{r["regime"]}_{r["bearing"]}_{r["repetition"]}.mat', "Source member identity mismatch")
        need(len(r["values"]) == 28 and all(isinstance(v, str) and VALUE.fullmatch(v) for v in r["values"]), "Expected 28 six-decimal strings")
        need(np.isfinite(np.asarray(r["values"], dtype=float)).all(), "Nonfinite feature")
        if labels:
            need(r["label"] in LABELS, "Unknown label")
    need(len({r["id"] for r in rows}) == len(rows), "Duplicate record ID")
    need(len({r["member"] for r in rows}) == len(rows), "Duplicate source member")
    if labels:
        bearing_labels = defaultdict(set)
        for r in rows:
            bearing_labels[r["bearing"]].add(r["label"])
        need(all(len(v) == 1 for v in bearing_labels.values()), "Inconsistent specimen truth")
    if counts:
        bearings, per_cell = counts
        groups = {r["bearing"] for r in rows}
        need(len(groups) == bearings, "Specimen count mismatch")
        need(Counter((r["bearing"], r["regime"]) for r in rows) == Counter({(b, g): per_cell for b in groups for g in REGIMES}), "Specimen/regime record balance mismatch")
        if labels:
            need(Counter(next(iter(v)) for v in bearing_labels.values()) == Counter({c: bearings // 3 for c in LABELS}), "Specimen/class balance mismatch")
    return rows


def validate_split(development, test, dictionary):
    dev = validate_rows(development, dictionary, labels=True, counts=(6, 5))
    tst = validate_rows(test, dictionary, labels=True, counts=(9, 1))
    for key in ("bearing", "member", "id"):
        need(not {r[key] for r in dev} & {r[key] for r in tst}, "Development/test source leakage: " + key)
    return dev, tst


def split_test(document):
    """Changing test labels cannot change this input representation."""
    inputs = {"schema": "bearing-baseline-test-inputs-v1", "dictionary": document["dictionary"],
              "records": [{k: v for k, v in r.items() if k != "label"} for r in document["records"]]}
    answers = {"schema": "bearing-baseline-test-answers-v1", "labels": {r["id"]: r["label"] for r in document["records"]}}
    return inputs, answers


def grouped_folds(rows):
    """Eight exhaustive one-specimen-per-class validation selections."""
    choices = [[b for b in sorted({r["bearing"] for r in rows}) if next(r["label"] for r in rows if r["bearing"] == b) == label] for label in LABELS]
    need(all(len(x) == 2 for x in choices), "Two development specimens per class required")
    folds = []
    for index, valid_bearings in enumerate(itertools.product(*choices), 1):
        valid = [i for i, r in enumerate(rows) if r["bearing"] in valid_bearings]
        train = [i for i, r in enumerate(rows) if r["bearing"] not in valid_bearings]
        need(not {rows[i]["bearing"] for i in train} & {rows[i]["bearing"] for i in valid}, "Inner fold specimen leakage")
        need(not {rows[i]["member"] for i in train} & {rows[i]["member"] for i in valid}, "Inner fold source leakage")
        need({rows[i]["label"] for i in train} == set(LABELS), "Missing training class")
        folds.append({"fold": index, "train": train, "validation": valid,
                      "train_bearings": sorted({rows[i]["bearing"] for i in train}),
                      "validation_bearings": list(valid_bearings)})
    return folds


def tasks(development_rows, test_rows):
    result = [("primary", None, development_rows, test_rows)]
    for regime in REGIMES:
        train = [r for r in development_rows if r["regime"] != regime]
        tst = [r for r in test_rows if r["regime"] == regime]
        need(len(train) == 90 and len(tst) == 9, "LORO dimensions changed")
        need(not any(r["regime"] == regime for r in train), "Held-out regime in LORO fitting")
        result.append(("loro_" + regime, regime, train, tst))
    return result


def array(rows):
    """The only predictor feature extraction: fixed numeric values, no metadata."""
    return np.asarray([r["values"] for r in rows], dtype=float)


def estimator(model, params):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    if model == "SVM":
        return Pipeline([("scale", StandardScaler()), ("classifier", SVC(kernel="rbf", C=params["C"], gamma=params["gamma"], probability=False, random_state=SEED))])
    if model == "RF":
        return RandomForestClassifier(n_estimators=200, max_features=params["max_features"], min_samples_leaf=params["min_samples_leaf"], random_state=SEED, n_jobs=1)
    raise ValueError("Unknown model: " + model)


def choose_candidate(rows, model, candidates, factory=estimator):
    x = array(rows)
    y = np.asarray([r["label"] for r in rows])
    folds = grouped_folds(rows)
    reports = []
    for index, params in enumerate(candidates):
        fold_reports = []
        start = time.perf_counter()
        for f in folds:
            fitted = factory(model, params)
            fitted.fit(x[f["train"]], y[f["train"]])
            pred = list(fitted.predict(x[f["validation"]]))
            correct = sum(p == y[i] for p, i in zip(pred, f["validation"]))
            fold_reports.append({"fold": f["fold"], "train_bearings": f["train_bearings"],
                "validation_bearings": f["validation_bearings"], "train_ids": [rows[i]["id"] for i in f["train"]],
                "validation_ids": [rows[i]["id"] for i in f["validation"]],
                "training_regimes": sorted({rows[i]["regime"] for i in f["train"]}),
                "validation_predictions": pred, "correct": int(correct), "total": len(f["validation"]),
                "accuracy": correct / len(f["validation"])})
        reports.append({"candidate_index": index, "params": params, "mean_accuracy": sum(f["accuracy"] for f in fold_reports) / len(fold_reports),
                        "elapsed_seconds": time.perf_counter() - start, "folds": fold_reports})
    # Python max retains the first entry when scores tie; enumeration is frozen.
    winner = max(range(len(reports)), key=lambda i: reports[i]["mean_accuracy"])
    return winner, reports


def structural_checks(fitted, x, background):
    """Fixed fitted predictor only: no refitting or task accuracy in these checks."""
    pred = np.asarray(fitted.predict(x))
    checks = {"reversed_row_order": np.array_equal(pred, np.asarray(fitted.predict(x[::-1]))[::-1]),
              "single_row_calls": np.array_equal(pred, np.asarray([fitted.predict(x[i:i + 1])[0] for i in range(len(x))])),
              "unlabeled_background_appended": np.array_equal(pred, np.asarray(fitted.predict(np.vstack([x, background])))[:len(x)])}
    need(all(checks.values()), "Fitted predictor row-independence invariant failed")
    return checks


def prepare_protocol(data=DATA, out=OUT, original=ORIGINAL, llm=LLM):
    data, out = Path(data).resolve(), Path(out).resolve()
    need(not out.exists(), "Output already exists; a new version is required")
    dictionary = read(original)["dictionary"]
    dev, tst = read(data / "development.json"), read(data / "test.json")
    dev_rows, test_rows = validate_split(dev, tst, dictionary)
    input_doc, answer_doc = split_test(tst)
    bound = [data / n for n in ("development.json", "test.json", "plan.json", "plan.sha256", "audit.json")]
    bound += [Path(original), Path(llm), CODE, TESTS]
    # Data preparation provenance and its separately frozen implementation.
    plan = read(data / "plan.json")
    need(digest(data / "plan.json") == (data / "plan.sha256").read_text().strip(), "Data plan hash mismatch")
    audit = read(data / "audit.json")
    need(audit["plan_sha256"] == digest(data / "plan.json"), "Audit/data plan mismatch")
    code_maps = [plan["code_sha256"], audit["code_sha256"]]
    if "extraction_amendment_sha256" in audit:
        amendment_path = data / "extraction_amendment_v2.json"
        amendment_sha = data / "extraction_amendment_v2.sha256"
        failure_path = data / "extraction_failure.json"
        need(digest(amendment_path) == audit["extraction_amendment_sha256"] == amendment_sha.read_text().strip(), "Extraction amendment hash mismatch")
        amendment = read(amendment_path)
        need(amendment["schema"] == "bearing-baseline-extraction-amendment-v2", "Unknown extraction amendment")
        need(amendment["prior_plan_sha256"] == digest(data / "plan.json"), "Amendment/prior plan mismatch")
        need(amendment["prior_failure_sha256"] == digest(failure_path), "Amendment/prior failure mismatch")
        code_maps.append(amendment["code_sha256"])
        bound += [amendment_path, amendment_sha, failure_path]
    combined_codes = {}
    for mapping in code_maps:
        for name, expected in mapping.items():
            need(name not in combined_codes or combined_codes[name] == expected, "Conflicting code provenance: " + name)
            combined_codes[name] = expected
    need(all(audit["code_sha256"].get(name) == expected for name, expected in combined_codes.items()), "Audit omits code lineage")
    for name, expected in combined_codes.items():
        need(digest(resolve(name)) == expected, "Data preparation code changed: " + name)
        bound.append(resolve(name))
    for name, expected in audit["output_sha256"].items():
        need(digest(data / name) == expected, "Prepared data audit hash mismatch: " + name)
        bound.append(data / name)
    source_hashes = {relative(p): digest(p) for p in bound}
    write(out / "test_inputs.json", input_doc)
    write(out / "test_answers.json", answer_doc)
    protocol = {"schema": "bearing-baseline-protocol-v1", "created_at_utc": now(), "seed": SEED,
        "labels": LABELS, "regimes": REGIMES, "dictionary": dictionary,
        "development": relative(data / "development.json"), "test_source_hash_only": relative(data / "test.json"),
        "llm_comparison_csv_score_only": relative(llm), "source_sha256": source_hashes,
        "local_sha256": {n: digest(out / n) for n in ("test_inputs.json", "test_answers.json")},
        "grids": GRIDS, "rf_fixed": {"n_estimators": 200, "random_state": SEED, "n_jobs": 1},
        "svm_fixed": {"kernel": "rbf", "preprocessing": "StandardScaler fitted only on current training fold", "probability": False},
        "selection": "Eight one-specimen-per-class validation choices; unweighted mean accuracy; first highest candidate in frozen enumeration; independently retune within each LORO training set.",
        "constant": "Training-majority class, ties follow labels order (healthy first).",
        "uniform_reference": "Expected accuracy 1/3, no sampled randomized predictions or inferred F1.",
        "folds": {name: grouped_folds(train) for name, _, train, _ in tasks(dev_rows, test_rows)},
        "dimensions": {"development_records": 120, "development_bearings": 6, "test_records": 36, "test_bearings": 9, "physical_rigs": 1, "inner_folds": 8, "loro_tasks": 4, "loro_train_records": 90, "loro_test_records": 9},
        "scope": "Outcome-informed supervised representation benchmark after known LLM findings. Test labels excluded from fitting/tuning. Supervised label/sample budget differs from zero-shot LLM; not a fair algorithm ranking or proof of input sufficiency/mechanism. LORO changes both regime support and training size.",
        "inference": "No p values or bootstrap CIs. Records and repeated LLM calls are dependent; report nine test specimens and per-call descriptive comparisons.",
        "fit_policy": "One execution; failures leave partial artifacts for audit and cannot be resumed or replaced in this version. All final predictions frozen before score opens answers."}
    write(out / "protocol.json", protocol)
    (out / "protocol.sha256").write_text(digest(out / "protocol.json") + "\n")
    return {"protocol": str(out / "protocol.json"), "protocol_sha256": digest(out / "protocol.json"), "ready_to_fit": True}


def audit_protocol(out):
    p = read(out / "protocol.json")
    need(digest(out / "protocol.json") == (out / "protocol.sha256").read_text().strip(), "Protocol changed")
    need(p["schema"] == "bearing-baseline-protocol-v1" and p["seed"] == SEED and p["labels"] == LABELS and p["regimes"] == REGIMES and p["grids"] == GRIDS, "Frozen algorithm mismatch")
    for rel, expected in p["source_sha256"].items():
        need(digest(resolve(rel)) == expected, "Frozen source changed: " + rel)
    for name, expected in p["local_sha256"].items():
        need(digest(out / name) == expected, "Frozen local input changed: " + name)
    return p


def load_fit_inputs(out):
    p = audit_protocol(out)
    # Do not parse test_source_hash_only, test_answers, or old LLM predictions here.
    dev = read(resolve(p["development"]))
    tst = read(out / "test_inputs.json")
    dr = validate_rows(dev, p["dictionary"], labels=True, counts=(6, 5))
    tr = validate_rows(tst, p["dictionary"], labels=False, counts=(9, 1))
    for key in ("bearing", "member", "id"):
        need(not {r[key] for r in dr} & {r[key] for r in tr}, "Frozen source leakage: " + key)
    for name, _, train, _ in tasks(dr, tr):
        need(grouped_folds(train) == p["folds"][name], "Frozen folds changed")
    return p, dr, tr


def fit(out=OUT):
    out = Path(out).resolve()
    p, dev, tst = load_fit_inputs(out)
    run = out / "fit"
    need(not run.exists(), "Fit was already started; no rerun/replacement permitted")
    run.mkdir()
    started, tick = now(), time.perf_counter()
    write(run / "started.json", {"started_at_utc": started, "protocol_sha256": digest(out / "protocol.json")})
    predictions, selections, checks, all_cv = [], [], [], {}
    for name, held_out, train, test in tasks(dev, tst):
        x, y, xt = array(train), np.asarray([r["label"] for r in train]), array(test)
        for model, grid in GRIDS.items():
            begin = time.perf_counter()
            selected, reports = choose_candidate(train, model, grid)
            all_cv[name + "/" + model] = reports
            fitted = estimator(model, grid[selected])
            final_begin = time.perf_counter()
            fitted.fit(x, y)
            final_seconds = time.perf_counter() - final_begin
            prediction_begin = time.perf_counter()
            predicted = list(fitted.predict(xt))
            predict_seconds = time.perf_counter() - prediction_begin
            check_begin = time.perf_counter()
            invariants = structural_checks(fitted, xt, x)
            checks.append({"task": name, "model": model, **invariants, "elapsed_seconds": time.perf_counter() - check_begin})
            selections.append({"task": name, "held_out_regime": held_out, "model": model, "candidate_index": selected,
                "params": grid[selected], "cv_mean_accuracy": reports[selected]["mean_accuracy"],
                "train_records": len(train), "test_records": len(test), "training_bearings": sorted({r["bearing"] for r in train}),
                "training_regimes": sorted({r["regime"] for r in train}), "final_fit_seconds": final_seconds,
                "prediction_seconds": predict_seconds, "task_model_seconds_including_checks": time.perf_counter() - begin})
            predictions.extend({"task": name, "model": model, "id": r["id"], "bearing": r["bearing"], "regime": r["regime"], "predicted": str(pred)} for r, pred in zip(test, predicted))
        counts = Counter(y)
        constant = max(LABELS, key=lambda label: counts[label])
        predictions.extend({"task": name, "model": "Constant", "id": r["id"], "bearing": r["bearing"], "regime": r["regime"], "predicted": constant} for r in test)
        # Preserve completed work if a later task fails, but score remains blocked.
        write(run / "cv.json", all_cv)
        write(run / "selections.json", selections)
        write(run / "structural_checks.json", checks)
        write_csv(run / "predictions.csv", predictions)
    audit_protocol(out)
    environment = {"python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "scikit-learn", "joblib", "threadpoolctl")}}
    write(run / "run.json", {"started_at_utc": started, "completed_at_utc": now(), "elapsed_seconds": time.perf_counter() - tick,
        "environment": environment, "seed": SEED, "protocol_sha256": digest(out / "protocol.json"),
        "source_sha256": p["source_sha256"], "local_sha256": p["local_sha256"], "cv_fits": 400, "final_fits": 10,
        "truth_parsed_during_fit": False, "predictor_features": "28 numeric values only", "new_llm_calls": 0})
    names = ["started.json", "cv.json", "selections.json", "structural_checks.json", "predictions.csv", "run.json"]
    manifest = {"schema": "bearing-baseline-prediction-freeze-v1", "frozen_at_utc": now(),
                "protocol_sha256": digest(out / "protocol.json"), "files_sha256": {n: digest(run / n) for n in names}}
    write(out / "prediction_manifest.json", manifest)
    (out / "prediction_manifest.sha256").write_text(digest(out / "prediction_manifest.json") + "\n")
    return {"status": "predictions_frozen", "prediction_rows": len(predictions), "manifest_sha256": digest(out / "prediction_manifest.json")}


def audit_predictions(out):
    p = audit_protocol(out)
    path = out / "prediction_manifest.json"
    need(digest(path) == (out / "prediction_manifest.sha256").read_text().strip(), "Prediction manifest changed")
    manifest = read(path)
    need(manifest["protocol_sha256"] == digest(out / "protocol.json"), "Prediction/protocol mismatch")
    required = {"started.json", "cv.json", "selections.json", "structural_checks.json", "predictions.csv", "run.json"}
    need(set(manifest["files_sha256"]) == required, "Incomplete prediction freeze")
    for name, expected in manifest["files_sha256"].items():
        need(digest(out / "fit" / name) == expected, "Frozen prediction artifact changed: " + name)
    return p, manifest


def metrics(rows):
    from sklearn.metrics import confusion_matrix, f1_score
    truth, predicted = [r["truth"] for r in rows], [r["predicted"] for r in rows]
    correct = sum(a == b for a, b in zip(truth, predicted))
    groups = {}
    for key in ("bearing", "regime"):
        groups[key] = [{key: value, "correct": sum(r["truth"] == r["predicted"] for r in rows if r[key] == value),
                        "total": sum(r[key] == value for r in rows)} for value in sorted({r[key] for r in rows})]
    bearing_counts = [r["correct"] for r in groups["bearing"]]
    matrix = confusion_matrix(truth, predicted, labels=LABELS)
    return {"correct": correct, "total": len(rows), "accuracy": correct / len(rows),
            "macro_f1": float(f1_score(truth, predicted, labels=LABELS, average="macro", zero_division=0)),
            "labels": LABELS, "confusion_matrix_truth_rows": matrix.tolist(),
            "per_class_recall": {label: float(matrix[i, i] / matrix[i].sum()) if matrix[i].sum() else None for i, label in enumerate(LABELS)},
            "by_bearing": groups["bearing"], "by_regime": groups["regime"],
            "bearing_correct_count_range": [min(bearing_counts), max(bearing_counts)]}


def score(out=OUT):
    out = Path(out).resolve()
    p, manifest = audit_predictions(out)
    # First semantic access to test truth in the fit/score workflow.
    truth = read(out / "test_answers.json")["labels"]
    inputs = read(out / "test_inputs.json")["records"]
    metadata = {r["id"]: r for r in inputs}
    need(set(truth) == set(metadata) and set(truth.values()) == set(LABELS), "Answer IDs/classes mismatch")
    with (out / "fit/predictions.csv").open(newline="") as handle:
        pred = list(csv.DictReader(handle))
    grouped = defaultdict(list)
    for r in pred:
        need(r["id"] in metadata and r["predicted"] in LABELS, "Unknown prediction ID/class")
        need(all(r[k] == metadata[r["id"]][k] for k in ("bearing", "regime")), "Prediction metadata changed")
        grouped[(r["task"], r["model"])].append({**r, "truth": truth[r["id"]]})
    task_names = ["primary"] + ["loro_" + g for g in REGIMES]
    need(set(grouped) == set(itertools.product(task_names, ["SVM", "RF", "Constant"])), "Missing/unexpected task-model predictions")
    for (task, _), rows in grouped.items():
        expected = set(metadata) if task == "primary" else {i for i, r in metadata.items() if r["regime"] == task[len("loro_"):]}
        need(len(rows) == len(expected) and {r["id"] for r in rows} == expected, "Duplicate/missing task predictions")
    for model in ("SVM", "RF", "Constant"):
        grouped[("loro_combined", model)] = [r for g in REGIMES for r in grouped[("loro_" + g, model)]]
    summaries = {task + "/" + model: metrics(rows) for (task, model), rows in grouped.items()}
    with resolve(p["llm_comparison_csv_score_only"]).open(newline="") as handle:
        llm = [r for r in csv.DictReader(handle) if r["focal"].lower() == "true"]
    calls = defaultdict(list)
    for r in llm:
        need(r["id"] in metadata and r["truth"] == truth[r["id"]], "LLM comparison truth/ID mismatch")
        need(r["predicted"] in LABELS, "LLM comparison requires the 12 complete calls")
        calls[r["group"]].append({**r, "bearing": metadata[r["id"]]["bearing"], "regime": metadata[r["id"]]["regime"]})
    need(len(calls) == 12 and all(len(rows) == 36 and {r["id"] for r in rows} == set(metadata) for rows in calls.values()), "Expected 12 calls on identical 36 focal IDs")
    comparisons, call_summaries = [], {}
    for call, rows in sorted(calls.items()):
        lm = {r["id"]: r for r in rows}
        call_summaries[call] = {"condition": rows[0]["condition"], "round": int(rows[0]["round"]), "repeat": int(rows[0]["repeat"]), **metrics(rows)}
        for task in ("primary", "loro_combined"):
            for model in ("SVM", "RF", "Constant"):
                baseline = {r["id"]: r for r in grouped[(task, model)]}
                comparisons.append({"call": call, "condition": rows[0]["condition"], "round": int(rows[0]["round"]), "repeat": int(rows[0]["repeat"]),
                    "baseline_task": task, "model": model, "n": 36,
                    "baseline_correct": summaries[task + "/" + model]["correct"], "llm_correct": call_summaries[call]["correct"],
                    "prediction_disagreements": sum(baseline[i]["predicted"] != lm[i]["predicted"] for i in metadata),
                    "baseline_only_correct": sum(baseline[i]["predicted"] == truth[i] and lm[i]["predicted"] != truth[i] for i in metadata),
                    "llm_only_correct": sum(lm[i]["predicted"] == truth[i] and baseline[i]["predicted"] != truth[i] for i in metadata)})
    analysis = out / "analysis"
    analysis.mkdir(exist_ok=True)
    write_csv(analysis / "scored_predictions.csv", [r for (task, _), rows in grouped.items() if task != "loro_combined" for r in rows])
    write_csv(analysis / "llm_call_comparisons.csv", comparisons)
    result = {"schema": "bearing-baseline-scores-v1", "generated_at_utc": now(), "protocol_sha256": digest(out / "protocol.json"),
        "prediction_manifest_sha256": digest(out / "prediction_manifest.json"), "dimensions": p["dimensions"],
        "source_sha256": p["source_sha256"], "baseline": summaries, "llm_calls": call_summaries,
        "uniform_prior_expected_accuracy": 1 / 3, "uniform_prior_expected_correct_out_of_36": 12,
        "limitations": [p["scope"], p["inference"], "Structural prediction checks test fixed row-wise software behavior; they do not identify a causal mechanism or remove supervised-information differences.", "LORO combined reuses the same nine specimens; 12 LLM calls reuse the same 36 records. No aggregated 432-independent-record comparison."]}
    write(analysis / "scores.json", result)
    return {"scores": str(analysis / "scores.json"), "task_model_summaries": len(summaries), "llm_call_comparisons": len(comparisons)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-protocol", "fit", "score", "audit"])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    if args.command == "prepare-protocol":
        result = prepare_protocol(args.data, args.output)
    elif args.command == "fit":
        result = fit(args.output)
    elif args.command == "score":
        result = score(args.output)
    else:
        _, manifest = audit_predictions(args.output)
        result = {"status": "frozen_predictions_verified", "frozen_at_utc": manifest["frozen_at_utc"]}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
