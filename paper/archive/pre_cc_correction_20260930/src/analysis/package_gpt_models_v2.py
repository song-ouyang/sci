"""Verify all 24 v2 runs, freeze predictions, and package the complete results.

Answer keys are opened only with --score, after all runs pass verification and
their predictions have been frozen. The default creates the prediction package
without reading answer keys. --validate-only writes nothing and reads no keys.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
MODELS = ("gpt-5.5", "gpt-5.6-sol", "gpt-6-astra")
PROTOCOL = "codex-grouped-v2-txt"
ALLOWED_EVENTS = {
    "thread.started", "turn.started", "turn.completed", "turn.failed", "error",
    "item.started", "item.updated", "item.completed",
}
ALLOWED_ITEMS = {"reasoning", "agent_message", "error"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), sha(raw)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def local_file(folder, filename):
    require(isinstance(filename, str) and filename not in {"", ".", ".."}
            and Path(filename).name == filename and "\\" not in filename,
            f"Unsafe or missing artifact filename: {filename!r}")
    path = folder / filename
    require(path.is_file() and not path.is_symlink(), f"Missing or symlinked artifact: {path}")
    return path


def verify_events(path):
    events, diagnostics, sessions, completed = [], [], [], []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        event = json.loads(line)
        require(isinstance(event, dict) and event.get("type") in ALLOWED_EVENTS,
                f"{path}:{number}: unknown event type")
        item = event.get("item")
        if event["type"].startswith("item."):
            require(isinstance(item, dict) and item.get("type") in ALLOWED_ITEMS,
                    f"{path}:{number}: tool use or unknown item type")
        elif item is not None:
            require(isinstance(item, dict) and item.get("type") in ALLOWED_ITEMS,
                    f"{path}:{number}: tool use or unknown nested item type")
        if event["type"] in {"error", "turn.failed"} or (isinstance(item, dict) and item.get("type") == "error"):
            diagnostics.append({"line": number, "event": event})
        if event["type"] == "thread.started":
            require(isinstance(event.get("thread_id"), str) and event["thread_id"],
                    f"{path}:{number}: missing session ID")
            sessions.append(event["thread_id"])
        if event["type"] == "turn.completed":
            completed.append(event)
        events.append(event)
    require(bool(events), f"Empty event log: {path}")
    require(len(sessions) <= 1, f"Multiple sessions in one attempt: {path}")
    return {"sessions": sessions, "completed": completed, "diagnostics": diagnostics,
            "sha256": sha(path.read_bytes()), "n_events": len(events)}


def verify_all(args, generator, scorer, runner):
    batches, input_hash = generator.load_batches(args.input)
    require(tuple(generator.MODELS) == MODELS, "Unexpected generator model set")
    manifest, manifest_hash = read_json(args.prompts_dir / "manifest.json")
    require(manifest.get("input_sha256") == input_hash, "Input/TXT manifest hash mismatch")
    require(manifest.get("models") == list(MODELS), "TXT manifest model set mismatch")
    require(manifest.get("n_groups") == 8 and set(manifest.get("groups", {})) == set(batches),
            "TXT manifest must cover exactly eight input groups")
    expected_total = sum(group["n_samples"] for group in batches.values())
    require(manifest.get("predictions_per_model") == expected_total,
            "TXT manifest prediction count mismatch")
    run_protocol, protocol_hash = read_json(args.results_root / "run_protocol.json")
    protocol_values = {
        "experiment_version": "v2", "protocol": PROTOCOL, "model_ids": list(MODELS),
        "reasoning_effort": "medium", "input_sha256": input_hash,
        "prompt_manifest_sha256": manifest_hash,
        "groups": {name: group["n_samples"] for name, group in batches.items()},
        "groups_per_model": 8, "predictions_per_model": expected_total,
        "expected_total_predictions": expected_total * len(MODELS),
        "independent_session_per_group": True, "independent_session_per_sample": False,
        "tools_enabled": False, "keys_read_before_prediction_freeze": False,
        "developer_instructions": runner.INSTRUCTIONS,
    }
    for field, expected in protocol_values.items():
        require(field in run_protocol and type(run_protocol[field]) is type(expected)
                and run_protocol[field] == expected, f"Run protocol mismatch: {field}")
    payloads, txt_hashes = {}, {}
    for name, group in batches.items():
        entry = manifest["groups"][name]
        require(entry.get("filename") == f"{name}.txt", f"{name}: manifest TXT filename mismatch")
        raw = local_file(args.prompts_dir, entry["filename"]).read_bytes()
        text = raw.decode("utf-8")
        require(sha(raw) == entry.get("txt_sha256"), f"{name}: TXT hash mismatch")
        require(text == generator.render_group(name, group), f"{name}: TXT differs from original prompts")
        require(entry.get("sample_ids") == [sample["id"] for sample in group["samples"]],
                f"{name}: manifest sample IDs mismatch")
        require(entry.get("n_samples") == group["n_samples"]
                and entry.get("valid_labels") == group["valid_labels"]
                and entry.get("utf8_bytes") == len(raw)
                and entry.get("characters") == len(text), f"{name}: TXT manifest metadata mismatch")
        payloads[name], txt_hashes[name] = text, sha(raw)

    predictions, snapshots, verified, all_sessions = {}, {}, {}, set()
    cli_versions, instruction_hashes, runner_hashes = set(), set(), set()
    # No predictions are frozen and no keys are opened until all 24 pass.
    for model in MODELS:
        predictions[model], snapshots[model] = scorer.validate_model(model, args.results_root, batches, input_hash)
        verified[model] = {}
        folder = args.results_root / model
        for name, group in batches.items():
            prefix = f"{model}/{name}"
            meta_path = local_file(folder, f"{name}_metadata.json")
            metadata, metadata_hash = read_json(meta_path)
            required_values = {
                "status": "complete", "protocol": PROTOCOL, "model_requested": model,
                "reasoning_effort": "medium", "transport": "codex-cli-chatgpt-auth-https",
                "input_sha256": input_hash, "prompt_txt_sha256": txt_hashes[name],
                "batch": name, "n_samples": group["n_samples"], "tools_observed": 0,
                "answer_keys_read": False, "independent_session_per_group": True,
                "independent_session_per_sample": False,
                "prediction_sha256": snapshots[model]["files"][f"{name}_results.json"]["sha256"],
            }
            for field, expected in required_values.items():
                require(field in metadata and type(metadata[field]) is type(expected) and metadata[field] == expected,
                        f"{prefix}: required metadata mismatch: {field}")
            require(isinstance(metadata.get("cli_version"), str) and metadata["cli_version"],
                    f"{prefix}: missing CLI version")
            require(isinstance(metadata.get("runner_sha256"), str)
                    and re.fullmatch(r"[0-9a-f]{64}", metadata["runner_sha256"]),
                    f"{prefix}: missing runner hash")
            cli_versions.add(metadata["cli_version"])
            runner_hashes.add(metadata["runner_sha256"])
            request_path = local_file(folder, f"{name}_request.json")
            request, request_hash = read_json(request_path)
            require(request.get("model") == model and request.get("reasoning_effort") == "medium",
                    f"{prefix}: request model/settings mismatch")
            require(request.get("user_message") == payloads[name], f"{prefix}: request differs from exact TXT")
            require(request.get("developer_instructions") == runner.INSTRUCTIONS,
                    f"{prefix}: developer instructions differ")
            instructions_hash = sha(request["developer_instructions"].encode("utf-8"))
            require(metadata.get("base_instructions_sha256") == instructions_hash,
                    f"{prefix}: developer instruction hash mismatch")
            instruction_hashes.add(instructions_hash)
            fingerprint = sha(json.dumps({
                "model": model, "effort": "medium", "payload": payloads[name],
                "instructions": request["developer_instructions"], "protocol": PROTOCOL,
            }, sort_keys=True).encode())
            require(metadata.get("fingerprint") == fingerprint, f"{prefix}: request fingerprint mismatch")
            attempts = metadata.get("attempts")
            require(isinstance(attempts, list) and attempts, f"{prefix}: no attempt provenance")
            archived_names, earlier_attempts = set(), []
            previous = metadata.get("previous_execution_metadata_file")
            while previous:
                require(previous not in archived_names, f"{prefix}: circular previous-execution metadata")
                archived_names.add(previous)
                archived, _ = read_json(local_file(folder, previous))
                require(all(archived.get(field) == metadata[field] for field in
                            ("protocol", "model_requested", "reasoning_effort", "input_sha256", "fingerprint")),
                        f"{prefix}: previous execution protocol mismatch")
                require(isinstance(archived.get("attempts"), list), f"{prefix}: missing archived attempts")
                earlier_attempts = archived["attempts"] + earlier_attempts
                previous = archived.get("previous_execution_metadata_file")
            attempts = earlier_attempts + attempts
            attempt_reports, attempt_ids = [], []
            for attempt in attempts:
                require(isinstance(attempt, dict) and type(attempt.get("attempt")) is int
                        and attempt["attempt"] > 0, f"{prefix}: invalid attempt number")
                number = attempt["attempt"]
                attempt_ids.append(number)
                require(attempt.get("events_file") == f"{name}_attempt{number}_events.jsonl"
                        and attempt.get("response_file") == f"{name}_attempt{number}_response.txt",
                        f"{prefix}: attempt artifact naming mismatch")
                event_path = local_file(folder, attempt["events_file"])
                event_info = verify_events(event_path)
                for session in event_info["sessions"]:
                    require(session not in all_sessions, f"{prefix}: session reused across model/group attempts")
                    all_sessions.add(session)
                require(attempt.get("malformed_event_lines") == [], f"{prefix}: malformed events were recorded")
                attempt_reports.append({"attempt": number, "returncode": attempt.get("returncode"),
                                        "timed_out": attempt.get("timed_out"),
                                        "events_file": attempt["events_file"], **event_info})
            require(attempt_ids == sorted(set(attempt_ids)), f"{prefix}: duplicate or unordered attempts")
            final, final_events = attempts[-1], attempt_reports[-1]
            require(type(final.get("returncode")) is int and final["returncode"] == 0
                    and final.get("timed_out") is False,
                    f"{prefix}: final attempt did not succeed")
            require(len(final_events["completed"]) == 1 and len(final_events["sessions"]) == 1,
                    f"{prefix}: successful attempt must contain one completed turn in one fresh session")
            require(final.get("usage") == final_events["completed"][0].get("usage"),
                    f"{prefix}: usage metadata mismatch")
            response_path = local_file(folder, final["response_file"])
            response_raw = response_path.read_bytes()
            require(runner.parse_raw(response_raw.decode("utf-8"), group) == predictions[model][name],
                    f"{prefix}: raw model response differs from saved predictions")
            verified[model][name] = {
                "n_predictions": group["n_samples"], "metadata_sha256": metadata_hash,
                "request_sha256": request_hash, "raw_response_sha256": sha(response_raw),
                "prediction_sha256": metadata["prediction_sha256"],
                "txt_sha256": txt_hashes[name], "attempts": attempt_reports,
                "final_session_id": final_events["sessions"][0],
                "server_resolved_model": metadata.get("server_resolved_model"),
                "previous_execution_metadata_file": metadata.get("previous_execution_metadata_file"),
            }
    require(len(cli_versions) == 1 and len(instruction_hashes) == 1,
            "Inconsistent CLI versions or developer instructions across runs")
    report = {
        "schema_version": 1, "experiment_version": "v2", "status": "verified",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "models": list(MODELS), "groups_per_model": 8, "completed_runs": 24,
        "predictions_per_model": expected_total, "total_predictions": expected_total * len(MODELS),
        "input_sha256": input_hash, "txt_manifest_sha256": manifest_hash,
        "run_protocol_sha256_before_packaging": protocol_hash,
        "cli_versions": sorted(cli_versions), "runner_sha256_values": sorted(runner_hashes),
        "base_instructions_sha256": next(iter(instruction_hashes)),
        "diagnostic_errors_count": sum(len(attempt["diagnostics"]) for groups in verified.values()
                                       for result in groups.values() for attempt in result["attempts"]),
        "checks": {"complete_model_group_coverage": True, "exact_string_ids_order_and_labels": True,
                   "original_input_and_txt_hashes": True, "request_identical_to_txt": True,
                   "raw_response_matches_predictions": True, "event_stream_tool_use_absent": True,
                   "fresh_session_per_attempt": True, "all_metadata_complete": True},
        "runs": verified,
    }
    return predictions, snapshots, report, run_protocol


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "gpt_experiment_batches_v2.json")
    parser.add_argument("--prompts-dir", type=Path, default=ROOT / "gpt_prompts_v2")
    parser.add_argument("--results-root", type=Path, default=ROOT / "results/gpt_models_v2")
    parser.add_argument("--keys-dir", type=Path, default=ROOT / "results/batches_clean_v2")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--score", action="store_true", help="After complete verification and freeze, score using keys")
    mode.add_argument("--validate-only", action="store_true", help="Verify all 24 runs without writing or reading keys")
    args = parser.parse_args(argv)
    try:
        generator = load_module("gpt_v2_prompt_generator", ROOT / "make_gpt_prompts.py")
        scorer = load_module("gpt_v2_scorer", ROOT / "src/analysis/score_gpt_models_v2.py")
        runner = load_module("gpt_v2_runner", ROOT / "src/experiments/run_gpt_models_v2.py")
        predictions, snapshots, validation, run_protocol = verify_all(args, generator, scorer, runner)
        # Honor existing freezes even in read-only validation mode.
        for model in MODELS:
            if (args.results_root / model / scorer.MANIFEST).exists():
                scorer.check_manifest(model, args.results_root, snapshots[model])
        if args.validate_only:
            print(json.dumps({"status": "verified", "completed_runs": 24,
                              "total_predictions": validation["total_predictions"]}))
            return 0
        scorer_args = ["--input", str(args.input), "--results-root", str(args.results_root),
                       "--keys-dir", str(args.keys_dir), "--models", *MODELS]
        require(scorer.main([*scorer_args, "--freeze"]) == 0, "Prediction freeze failed")
        for model in MODELS:
            scorer.check_manifest(model, args.results_root, snapshots[model])
        if args.score:
            require(scorer.main([*scorer_args, "--score"]) == 0, "Scoring failed")
        validation["predictions_frozen"] = True
        validation["scored_in_this_packaging_run"] = args.score
        validation["answer_keys_read_in_this_packaging_run"] = args.score
        # Preserve the operator's original run protocol and add completion
        # evidence only after every run, freeze, and optional score succeeds.
        run_protocol["status"] = "complete"
        run_protocol.setdefault("completed_at_utc", datetime.now(timezone.utc).isoformat())
        run_protocol["completed_runs"] = 24
        run_protocol["actual_total_predictions"] = validation["total_predictions"]
        run_protocol["predictions_frozen"] = True
        run_protocol["scored"] = args.score or run_protocol.get("scored", False)
        run_protocol["diagnostic_errors_count"] = validation["diagnostic_errors_count"]
        run_protocol["validation_file"] = "validation.json"
        run_protocol["prediction_archive"] = "gpt_models_v2_results.zip"
        scorer.write_json(args.results_root / "predictions.json", predictions)
        for model in MODELS:
            scorer.write_json(args.results_root / model / "predictions.json", predictions[model])
        scorer.write_json(args.results_root / "run_protocol.json", run_protocol)
        validation["run_protocol_sha256_after_packaging"] = sha((args.results_root / "run_protocol.json").read_bytes())
        scorer.write_json(args.results_root / "validation.json", validation)
        # Explicit allowlist: answer keys, unrelated runs, and source prompts
        # are never swept into the result archive by a broad directory glob.
        files = [args.results_root / name for name in ("predictions.json", "validation.json", "run_protocol.json")]
        if args.score:
            files.extend(args.results_root / name for name in ("scores.json", "scores.csv"))
        for model in MODELS:
            folder = args.results_root / model
            files.extend(folder / name for name in ("predictions.json", scorer.MANIFEST))
            for group in predictions[model]:
                files.extend(folder / f"{group}_{suffix}.json" for suffix in ("results", "metadata"))
        archive_path = args.results_root / "gpt_models_v2_results.zip"
        temporary = archive_path.with_suffix(".zip.tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, arcname=path.relative_to(args.results_root))
        temporary.replace(archive_path)
        print(json.dumps({"status": "packaged", "completed_runs": 24,
                          "total_predictions": validation["total_predictions"],
                          "archive": str(archive_path), "scored": args.score}))
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
