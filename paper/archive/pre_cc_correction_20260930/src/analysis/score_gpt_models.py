"""Validate, freeze, and score complete multi-model prediction files.

Validation and freezing never open answer keys. Score only after every requested
model has a matching frozen manifest. IDs must follow the source sample order;
invalid/missing predictions are errors, never silently dropped from the score.

Example:
    .venv/bin/python3 src/analysis/score_gpt_models.py \
        --models gpt-5.5 gpt-5.6-sol gpt-6-astra --freeze
    .venv/bin/python3 src/analysis/score_gpt_models.py \
        --models gpt-5.5 gpt-5.6-sol gpt-6-astra --score
"""

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
KEY_SUFFIXES = {"cmapss": "strat30", "tep": "strat21"}
MANIFEST = "predictions_manifest.json"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    # A reader sees the complete previous or new file, never a partial JSON.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def safe_name(value):
    require(
        isinstance(value, str)
        and value not in {"", ".", ".."}
        and "/" not in value
        and "\\" not in value,
        f"Invalid model/group name: {value!r}",
    )
    return value


def load_batches(path):
    batches, digest = read_json(path)
    require(isinstance(batches, dict) and batches, "Input must be a nonempty group object")
    for group, data in batches.items():
        safe_name(group)
        require(isinstance(data, dict), f"{group}: expected a group object")
        labels = data.get("valid_labels")
        require(
            isinstance(labels, list)
            and labels
            and all(isinstance(label, str) and label for label in labels),
            f"{group}: valid_labels must be a nonempty list of strings",
        )
        require(len(labels) == len(set(labels)), f"{group}: duplicate valid_labels")
        samples = data.get("samples")
        require(isinstance(samples, list), f"{group}: samples must be a list")
        require(
            type(data.get("n_samples")) is int
            and data["n_samples"] > 0
            and len(samples) == data["n_samples"],
            f"{group}: n_samples must match the nonempty samples list",
        )
        require(
            all(isinstance(sample, dict) and type(sample.get("id")) is int for sample in samples),
            f"{group}: each sample must have an integer id",
        )
        ids = [sample["id"] for sample in samples]
        require(len(ids) == len(set(ids)), f"{group}: duplicate input sample IDs")
    return batches, digest


def validate_model(model, root, batches, input_digest):
    folder = root / safe_name(model)
    expected_files = {f"{group}_results.json" for group in batches}
    actual_files = {path.name for path in folder.glob("*_results.json")}
    require(
        actual_files == expected_files,
        f"{model}: group files mismatch; missing={sorted(expected_files - actual_files)}, "
        f"unexpected={sorted(actual_files - expected_files)}",
    )
    predictions, files = {}, {}
    for group, data in batches.items():
        filename = f"{group}_results.json"
        rows, digest = read_json(folder / filename)
        require(isinstance(rows, list), f"{model}/{group}: expected a JSON array")
        require(len(rows) == data["n_samples"], f"{model}/{group}: incomplete sample count")
        for index, row in enumerate(rows):
            prefix = f"{model}/{group} row {index}"
            require(
                isinstance(row, dict) and set(row) == {"id", "predicted"},
                f"{prefix}: expected exactly id and predicted fields",
            )
            require(type(row["id"]) is int, f"{prefix}: id must be an integer")
            require(
                isinstance(row["predicted"], str) and row["predicted"] in data["valid_labels"],
                f"{prefix}: label {row['predicted']!r} is not an exact valid_labels match",
            )
        expected_ids = [sample["id"] for sample in data["samples"]]
        require(
            [row["id"] for row in rows] == expected_ids,
            f"{model}/{group}: IDs must exactly match source sample IDs and list order",
        )
        predictions[group] = rows
        files[filename] = {"sha256": digest, "n_samples": len(rows)}
    snapshot = {
        "schema_version": 1,
        "model_id": model,
        "input_sha256": input_digest,
        "files": files,
    }
    return predictions, snapshot


def freeze_model(model, root, snapshot):
    path = root / model / MANIFEST
    if path.exists():
        check_manifest(model, root, snapshot)
        return
    write_json(path, {**snapshot, "frozen_at_utc": datetime.now(timezone.utc).isoformat()})


def check_manifest(model, root, snapshot):
    path = root / model / MANIFEST
    require(path.is_file(), f"{model}: missing {MANIFEST}; run --freeze before --score")
    frozen, _ = read_json(path)
    require(isinstance(frozen, dict), f"{model}: invalid prediction manifest")
    require(
        {field: frozen.get(field) for field in snapshot} == snapshot,
        f"{model}: input or predictions changed after freezing; refusing to score",
    )
    require(bool(frozen.get("frozen_at_utc")), f"{model}: missing freeze timestamp")


def load_keys(batches, keys_dir):
    # Called only after ALL requested model predictions and manifests validate.
    keys, provenance = {}, {}
    for group, data in batches.items():
        dataset = group.split("_", 1)[0]
        require(dataset in KEY_SUFFIXES, f"Unknown answer-key naming convention: {dataset}")
        path = keys_dir / f"{group}_{KEY_SUFFIXES[dataset]}_key.json"
        rows, digest = read_json(path)
        require(isinstance(rows, list), f"{path}: answer key must be an array")
        values = {}
        for row in rows:
            require(
                isinstance(row, dict) and type(row.get("id")) is int,
                f"{path}: each key entry must have an integer id",
            )
            require(row["id"] not in values, f"{path}: duplicate key ID {row['id']}")
            require(row.get("true_label") in data["valid_labels"], f"{path}: invalid true_label")
            values[row["id"]] = row["true_label"]
        expected_ids = [sample["id"] for sample in data["samples"]]
        require(all(sid in values for sid in expected_ids), f"{path}: missing answer-key IDs")
        keys[group] = [values[sid] for sid in expected_ids]
        provenance[group] = {"path": str(path.resolve()), "sha256": digest}
    return keys, provenance


def metrics(truth, predicted, labels):
    correct = sum(actual == guess for actual, guess in zip(truth, predicted))
    confusion = {actual: {guess: 0 for guess in labels} for actual in labels}
    for actual, guess in zip(truth, predicted):
        confusion[actual][guess] += 1
    per_class = {}
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        per_class[label] = {
            "support": sum(confusion[label].values()),
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        }
    return {
        "n_samples": len(truth),
        "correct": correct,
        "accuracy_pct": 100.0 * correct / len(truth),
        "macro_f1_pct": 100.0 * sum(row["f1"] for row in per_class.values()) / len(labels),
        "valid_labels": labels,
        "per_class": per_class,
        "confusion_matrix": confusion,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=ROOT / "gpt_experiment_batches.json")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results" / "gpt_models")
    parser.add_argument("--keys-dir", type=Path, default=ROOT / "results" / "batches_clean")
    parser.add_argument("--models", nargs="+", required=True, help="Model directory names; all must be complete")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--validate-only", action="store_true", help="Default; never reads answer keys")
    mode.add_argument("--freeze", action="store_true", help="Validate and freeze prediction hashes; never reads keys")
    mode.add_argument("--score", action="store_true", help="Require frozen predictions, then read answer keys and score")
    args = parser.parse_args(argv)
    try:
        require(len(args.models) == len(set(args.models)), "Duplicate model IDs")
        batches, input_digest = load_batches(args.input)
        predictions, snapshots = {}, {}
        # Complete validation for every model precedes any writes or key access.
        for model in args.models:
            predictions[model], snapshots[model] = validate_model(
                model, args.results_root, batches, input_digest
            )
        if args.freeze:
            # Check existing freezes first; never silently update a frozen run.
            for model in args.models:
                if (args.results_root / model / MANIFEST).exists():
                    check_manifest(model, args.results_root, snapshots[model])
            for model in args.models:
                freeze_model(model, args.results_root, snapshots[model])
        elif args.score:
            for model in args.models:
                check_manifest(model, args.results_root, snapshots[model])
            keys, key_provenance = load_keys(batches, args.keys_dir)
            report = {
                "schema_version": 1,
                "scored_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_path": str(args.input.resolve()),
                "input_sha256": input_digest,
                "metric_definition": {
                    "accuracy_pct": "100 * correct / n_samples; no samples excluded",
                    "macro_f1_pct": "100 * mean(class F1 over every valid_labels entry); zero_division=0",
                    "per_class_units": "precision, recall, and f1 are fractions from 0 to 1",
                    "confusion_matrix": "rows = true labels; columns = predicted labels",
                },
                "answer_keys": key_provenance,
                "models": {},
            }
            csv_rows = []
            for model in args.models:
                groups = {}
                for group, data in batches.items():
                    result = metrics(keys[group], [row["predicted"] for row in predictions[model][group]], data["valid_labels"])
                    groups[group] = result
                    csv_rows.append({"model_id": model, "group": group, **{key: result[key] for key in ("n_samples", "correct", "accuracy_pct", "macro_f1_pct")}})
                report["models"][model] = {
                    "prediction_manifest": snapshots[model],
                    "groups": groups,
                    "aggregate": {
                        "n_predictions": sum(result["n_samples"] for result in groups.values()),
                        "correct": sum(result["correct"] for result in groups.values()),
                        "pooled_accuracy_pct": 100.0 * sum(result["correct"] for result in groups.values()) / sum(result["n_samples"] for result in groups.values()),
                        "mean_group_accuracy_pct": sum(result["accuracy_pct"] for result in groups.values()) / len(groups),
                        "mean_group_macro_f1_pct": sum(result["macro_f1_pct"] for result in groups.values()) / len(groups),
                    },
                }
            json_path = args.results_root / "scores.json"
            csv_path = args.results_root / "scores.csv"
            write_json(json_path, report)
            with csv_path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=["model_id", "group", "n_samples", "correct", "accuracy_pct", "macro_f1_pct"])
                writer.writeheader()
                writer.writerows(csv_rows)
            print(json.dumps({"status": "scored", "scores_json": str(json_path), "scores_csv": str(csv_path)}, ensure_ascii=False))
            return 0
        print(json.dumps({"status": "frozen" if args.freeze else "valid", "models": args.models, "groups_per_model": len(batches), "samples_per_model": sum(data["n_samples"] for data in batches.values())}, ensure_ascii=False))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
