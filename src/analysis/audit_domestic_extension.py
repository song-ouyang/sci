"""Freeze all domestic-model outcomes without labels, then optionally score them."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments import domestic_api as api
from src.experiments import run_domestic_extension as runner


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return api.strict_json(path.read_text(encoding="utf-8"))


def stream_metadata(path):
    """Recover usage/finish metadata even when a truncated batch cannot be scored."""
    info = {"usage": None, "observed_model": None, "finish_reason": None}
    if not path.exists():
        return info
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith("data:") or line[5:].strip() == "[DONE]":
            continue
        try:
            event = json.loads(line[5:])
            if not isinstance(event, dict):
                continue
            if event.get("usage"):
                info["usage"] = event["usage"]
            if event.get("model"):
                info["observed_model"] = event["model"]
            for choice in event.get("choices", []):
                if choice.get("finish_reason"):
                    info["finish_reason"] = choice["finish_reason"]
        except (ValueError, AttributeError, TypeError):
            continue
    return info


def inventory(base):
    paths = [base / "plan.json", base / "plan.sha256"]
    for name in ("batches", "sessions", "source_snapshot"):
        paths.extend((base / name).rglob("*"))
    require(not any(p.is_symlink() for p in paths), "Symlink in experiment artifacts")
    return {str(p.relative_to(base)): runner.sha(p) for p in sorted(paths) if p.is_file()}


def audit(base):
    plan = runner.verify(base)
    expected = {(j["provider"], j["group"]) for j in plan["jobs"]}
    require(len(expected) == plan["expected_sessions"] == 18, "Unexpected session plan")
    actual = {(p.parent.parent.name, p.parent.name) for p in (base / "sessions").glob("*/*/result.json")}
    require(expected == actual, "Not all 18 planned sessions have terminal outcomes")
    outcomes = []
    for job in plan["jobs"]:
        folder = base / "sessions" / job["provider"] / job["group"]
        result = read_json(folder / "result.json")
        name = result["run_folder"]
        require(isinstance(name, str) and Path(name).name == name and name not in {".", ".."}, "Invalid run path")
        run = folder / name
        require({p.name for p in folder.iterdir()} == {"result.json", name}, "Unaccounted session artifacts")
        require(runner.sha(run / "sha256.json") == result["manifest_sha256"], "Session manifest changed")
        files = read_json(run / "sha256.json")
        require(set(files) == {p.name for p in run.iterdir() if p.name != "sha256.json"}, "Session inventory changed")
        for filename, expected_hash in files.items():
            require(Path(filename).name == filename and filename not in {".", ".."}, "Unsafe manifest path")
            require(runner.sha(run / filename) == expected_hash, "Changed artifact: " + filename)
        require(read_json(run / "request.json") == job["request"], "Request differs from frozen plan")
        batch = read_json(base / "batches" / (job["group"] + ".json"))
        require(read_json(run / "batch.json") == batch, "Batch differs from frozen plan")
        metadata = read_json(run / "metadata.json")
        status = metadata["status"]
        require(status == result["status"] and status in {"valid", "invalid_response", "http_error", "transport_error"},
                "Nonterminal or inconsistent status")
        config = plan["providers"][job["provider"]]
        for key in ("provider", "model", "base_url", "key_env"):
            require(metadata[key] == config[key], "Provider provenance mismatch")
        adapter_name = "domestic_api_v2.py" if plan["schema"] == "domestic-extension-v2" else "domestic_api.py"
        require(metadata["adapter_sha256"] == plan["source_sha256"]["src/experiments/" + adapter_name],
                "Adapter provenance mismatch")
        require(metadata["automatic_retries"] == 0 and metadata["finished_at"], "Missing terminal provenance")
        predictions = None
        parsed = None
        if status == "valid":
            parsed = api.parse_stream((run / "response.sse").read_bytes())
            require(read_json(run / "response.json") == parsed, "Parsed response differs from raw stream")
            predictions = api.validate_predictions(parsed["content"], batch)
            require(read_json(run / "predictions.json") == predictions, "Predictions differ from raw response")
            require(metadata["observed_model"] == parsed["model"] and metadata["usage"] == parsed["usage"],
                    "Response metadata mismatch")
        else:
            require(not (run / "predictions.json").exists(), "Failed batch contains scored predictions")
            if status == "invalid_response":
                invalid = False
                try:
                    parsed = api.parse_stream((run / "response.sse").read_bytes())
                    api.validate_predictions(parsed["content"], batch)
                except (ValueError, TypeError, KeyError, AttributeError):
                    invalid = True
                require(invalid, "Marked invalid but raw batch passes validation")
        outcomes.append({"provider": job["provider"], "model": config["model"], "group": job["group"],
                         "dataset": job["dataset"], "condition": job["condition"], "repeat": job["repeat"],
                         "status": status, "expected_n": len(batch["expected_ids"]),
                         "predictions": predictions, "error": metadata.get("error"),
                         **stream_metadata(run / "response.sse"), "elapsed_seconds": metadata["elapsed_seconds"],
                         "run_folder": str(run.relative_to(base))})
    return plan, outcomes


def freeze(base):
    target = base / "frozen_manifest.json"
    if target.exists():
        return verify(base)
    plan, outcomes = audit(base)
    snapshot = base / "source_snapshot"
    snapshot.mkdir(exist_ok=True)
    for name, expected_hash in plan["source_sha256"].items():
        source = ROOT / name
        require(runner.sha(source) == expected_hash, "Source changed before snapshot")
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            require(runner.sha(destination) == expected_hash, "Existing source snapshot changed")
        else:
            destination.write_bytes(source.read_bytes())
    manifest = {"schema": "domestic-outcomes-freeze-v1", "frozen_at": api.utc_now(),
                "answer_keys_read": False, "planned_sessions": len(outcomes),
                "valid_sessions": sum(o["status"] == "valid" for o in outcomes),
                "valid_predictions": sum(len(o["predictions"] or []) for o in outcomes),
                "expected_predictions": sum(o["expected_n"] for o in outcomes),
                "files": inventory(base),
                "outcomes": [{k: v for k, v in o.items() if k != "predictions"} for o in outcomes]}
    with target.open("x", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return verify(base)


def verify(base):
    manifest = read_json(base / "frozen_manifest.json")
    require(manifest["answer_keys_read"] is False, "Invalid freeze declaration")
    require(inventory(base) == manifest["files"], "Frozen artifact inventory changed")
    plan, outcomes = audit(base)
    require(manifest["outcomes"] == [{k: v for k, v in o.items() if k != "predictions"} for o in outcomes],
            "Frozen outcomes changed")
    require(manifest["valid_sessions"] == sum(o["status"] == "valid" for o in outcomes), "Freeze count mismatch")
    return plan, outcomes, manifest


def score(base):
    # This gate must succeed before any truth or baseline file is opened.
    plan, outcomes, manifest = verify(base)
    from src.analysis import score_revision_v3 as original
    original_protocol = read_json(runner.BASE / "protocol.json")
    require(runner.sha(runner.BASE / "protocol.json") == plan["original_protocol_sha256"], "Original protocol changed")
    require(runner.sha(runner.BASE / "inference/batches.json") == plan["original_input_sha256"], "Original inputs changed")
    groups = read_json(runner.BASE / "inference/batches.json")["groups"]
    datasets, baselines, provenance = original.load_evaluation_data(runner.BASE, original_protocol, groups)
    rows, cases, bootstraps = [], [], {}
    draws = {dataset: original.np.random.default_rng(original_protocol["bootstrap"]["seed"]).integers(
        0, len(records), size=(original_protocol["bootstrap"]["replicates"], len(records)))
        for dataset, records in datasets.items()}
    for outcome in outcomes:
        row = {k: v for k, v in outcome.items() if k not in {"predictions", "usage"}}
        row["primary"] = outcome["repeat"] == 1
        row["n_scored"] = 0
        row["accuracy"] = row["macro_f1"] = None
        if outcome["status"] == "valid":
            records = datasets[outcome["dataset"]]
            require([p["id"] for p in outcome["predictions"]] == [r["id"] for r in records], "Answer ID order mismatch")
            values = original.metric_values([r["label"] for r in records],
                                           [p["predicted"] for p in outcome["predictions"]],
                                           original.LABELS[outcome["dataset"]], outcome["dataset"])
            row.update(values, n_scored=len(records))
            sampled = original.bootstrap_values([r["label"] for r in records],
                                                [p["predicted"] for p in outcome["predictions"]],
                                                original.LABELS[outcome["dataset"]], draws[outcome["dataset"]])
            bootstraps[(outcome["provider"], outcome["group"])] = sampled
            row["macro_f1_group_bootstrap95"] = original.percentile95(sampled["macro_f1"])
            for record, prediction in zip(records, outcome["predictions"]):
                cases.append({"provider": outcome["provider"], "model": outcome["model"], "group": outcome["group"],
                              "id": record["id"], "truth": record["label"], "predicted": prediction["predicted"],
                              "correct": record["label"] == prediction["predicted"]})
        rows.append(row)
    paired = []
    for provider in plan["providers"]:
        for repeat in (1, 2, 3):
            clean = next(o for o in outcomes if o["provider"] == provider and o["group"] == f"tep_clean_r{repeat}")
            corrupt = next(o for o in outcomes if o["provider"] == provider and o["group"] == f"tep_misnamed_r{repeat}")
            item = {"provider": provider, "repeat": repeat, "primary": repeat == 1,
                    "accuracy_difference_pp": None, "prediction_disagreement": None}
            if clean["status"] == corrupt["status"] == "valid":
                left = next(r for r in rows if r["provider"] == provider and r["group"] == clean["group"])
                right = next(r for r in rows if r["provider"] == provider and r["group"] == corrupt["group"])
                item["accuracy_difference_pp"] = 100 * (left["accuracy"] - right["accuracy"])
                item["prediction_disagreement"] = sum(a["predicted"] != b["predicted"]
                                                      for a, b in zip(clean["predictions"], corrupt["predictions"])) / clean["expected_n"]
                item["macro_f1_difference_pp"] = 100 * (left["macro_f1"] - right["macro_f1"])
                for metric in ("accuracy", "macro_f1"):
                    item[metric + "_difference_pp_bootstrap95"] = [100 * v for v in original.percentile95(
                        bootstraps[(provider, clean["group"])][metric] - bootstraps[(provider, corrupt["group"])][metric])]
            paired.append(item)
    variability = []
    for provider in plan["providers"]:
        for dataset, condition in (("cmapss", "clean"), ("tep", "clean"), ("tep", "misnamed")):
            relevant = sorted([r for r in rows if r["provider"] == provider and r["dataset"] == dataset
                               and r["condition"] == condition], key=lambda r: r["repeat"])
            item = {"provider": provider, "dataset": dataset, "condition": condition,
                    "available_repeats": [r["repeat"] for r in relevant if r["status"] == "valid"],
                    "missing_repeats": [r["repeat"] for r in relevant if r["status"] != "valid"]}
            for metric in ("accuracy", "macro_f1"):
                values = [r[metric] for r in relevant if r[metric] is not None]
                item[metric] = {"min": min(values) if values else None, "max": max(values) if values else None,
                                "range": max(values) - min(values) if len(values) > 1 else None}
            variability.append(item)
    baseline_rows = []
    for dataset, methods in baselines.items():
        for method, predictions in methods.items():
            baseline_rows.append({"model": method, "dataset": dataset,
                                  **original.metric_values([r["label"] for r in datasets[dataset]], predictions,
                                                           original.LABELS[dataset], dataset)})
    output = base / "analysis"
    output.mkdir(exist_ok=True)
    usage_summary = {}
    for provider in plan["providers"]:
        usage_summary[provider] = {key: sum((o["usage"] or {}).get(key, 0) for o in outcomes if o["provider"] == provider)
                                   for key in ("prompt_tokens", "completion_tokens", "total_tokens")}
        usage_summary[provider]["sessions_with_usage"] = sum(o["usage"] is not None for o in outcomes if o["provider"] == provider)
    report = {"scored_at": api.utc_now(), "freeze_sha256": runner.sha(base / "frozen_manifest.json"),
              "study_plan_schema": plan["schema"], "supersedes": plan.get("supersedes"),
              "budget_rationale": plan.get("budget_rationale"), "execution": plan.get("execution"),
              "scorer_sha256": runner.sha(Path(__file__)),
              "metric_implementation_sha256": runner.sha(Path(original.__file__)),
              "n_unique_source_units": 130, "expected_sessions": len(outcomes),
              "valid_sessions": manifest["valid_sessions"], "valid_predictions": manifest["valid_predictions"],
              "metrics": rows, "paired": paired, "variability": variability,
              "baselines": baseline_rows, "data_provenance": provenance,
              "token_usage": usage_summary,
              "bootstrap": original_protocol["bootstrap"], "interval_scope": original.CI_NOTE,
              "limitations": ["Exploratory addition after original results; API and inference settings differ across providers.",
                              "130 unique simulated source units; repeats do not add independent observations.",
                              "Invalid whole batches are N/A. Repeat 1 remains primary."]}
    api.write_json(output / "scores.json", report)
    for filename, data in (("metrics.csv", [{k: r.get(k) for k in (
            "provider", "model", "group", "dataset", "condition", "repeat", "primary", "status", "finish_reason", "n_scored", "correct", "accuracy", "macro_f1", "error")} for r in rows]),
                           ("paired.csv", paired), ("cases.csv", cases)):
        with (output / filename).open("w", newline="", encoding="utf-8") as f:
            if data:
                writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for row in data for k in row)))
                writer.writeheader()
                writer.writerows(data)
    lines = ["# DeepSeek / Qwen 追加实验结果", "",
             f"完整有效会话：{manifest['valid_sessions']}/18；有效预测：{manifest['valid_predictions']}/{manifest['expected_predictions']}。",
             "复用 130 个独立来源单位；重复调用不增加独立样本。以下均为第 1 次重复（预设主结果）。", "",
             "| 模型 | C-MAPSS 准确率 | TEP 原始准确率 | TEP 错配准确率 |", "|---|---:|---:|---:|"]
    for provider, config in plan["providers"].items():
        cells = []
        for name in ("cmapss_clean_r1", "tep_clean_r1", "tep_misnamed_r1"):
            row = next(r for r in rows if r["provider"] == provider and r["group"] == name)
            cells.append("N/A" if row["accuracy"] is None else f"{100 * row['accuracy']:.1f}%")
        lines.append("| " + config["model"] + " | " + " | ".join(cells) + " |")
    lines += ["", "## 所有重复", "", "| 模型 | 条件 | 重复 | 状态 | 准确率 | Macro-F1 |", "|---|---|---:|---|---:|---:|"]
    for row in sorted(rows, key=lambda r: (r["provider"], r["dataset"], r["condition"], r["repeat"])):
        accuracy = "N/A" if row["accuracy"] is None else f"{100 * row['accuracy']:.1f}%"
        f1 = "N/A" if row["macro_f1"] is None else f"{100 * row['macro_f1']:.1f}%"
        lines.append(f"| {row['model']} | {row['dataset']}/{row['condition']} | {row['repeat']} | {row['status']} | {accuracy} | {f1} |")
    lines += ["", "失败会话保留原始记录且不重跑；主会话无效时不以其他重复替代。",
              "这是观察既有结果后的探索性追加实验，服务接口与推理参数并不完全等价；两套数据均为仿真数据。",
              "完整混淆矩阵与类别指标见 scores.json；逐条预测见 cases.csv。", ""]
    if plan["schema"] == "domestic-extension-v2":
        lines += ["本轮是预算修正版 v2。此前 v1 因预算配置不匹配停止：4 个无效会话、2 个操作中断会话、12 个未执行会话。",
                  "旧记录完整保存在相邻 domestic_extension 目录；没有替换、删除或混入本轮结果。", ""]
    (output / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:9]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "verify", "score"))
    parser.add_argument("--base", type=Path, default=runner.DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.command == "score":
        score(args.base)
    else:
        _, _, manifest = freeze(args.base) if args.command == "freeze" else verify(args.base)
        print(f"{args.command}: {manifest['valid_sessions']}/18 valid sessions; answer keys not read")


if __name__ == "__main__":
    main()
