"""Audit the existing Claude extension without opening answer keys.

This is an archival integrity audit, not a scorer or a replacement inference
protocol. Hashes describe current bytes and cannot establish a pre-score freeze,
model identity, the actual delivered context, or absence of tool use. The known
Agent-tool extension remains ineligible for the primary controlled comparison.
Only audit/report.json and audit/report.md are written by the CLI (exit code 2).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
EXT_DIR = ROOT / "results/revision_v3_claude_extension"
PARENT_DIR = ROOT / "results/revision_v3"
MODELS = ("claude-sonnet-4-6", "claude-opus-5-5")
GROUPS = tuple(f"{condition}_r{repeat}" for condition in
               ("cmapss_clean", "tep_clean", "tep_misnamed") for repeat in (1, 2, 3))


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def issue(code: str, detail: str) -> dict:
    return {"code": code, "detail": detail}


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def strict_json(text: str):
    return json.loads(text, object_pairs_hook=_no_duplicate_keys,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError(f"Non-JSON constant: {value}")))


def validate_array(predictions, group):
    """Exact v3 records, labels and ordered IDs; never inspect true labels."""
    if not isinstance(predictions, list):
        raise ValueError("Response is not a JSON array")
    samples = group["samples"]
    if len(predictions) != len(samples):
        raise ValueError(f"Prediction count {len(predictions)} != expected {len(samples)}")
    for position, (prediction, sample) in enumerate(zip(predictions, samples)):
        if not isinstance(prediction, dict) or set(prediction) != {"id", "predicted"}:
            raise ValueError(f"Record {position} must have exactly id and predicted")
        identifier = prediction["id"]
        if not isinstance(identifier, str) or re.fullmatch(r"[0-9a-fA-F]{8}", identifier) is None:
            raise ValueError(f"Record {position} ID is not an eight-character hex string")
        if identifier != sample["id"]:
            raise ValueError(f"Record {position} ID differs from ordered input")
        label = prediction["predicted"]
        if not isinstance(label, str) or label not in group["valid_labels"]:
            raise ValueError(f"Record {position} label is outside case-sensitive valid_labels")


def parse_raw_v3(raw: str, group: dict):
    """Use the v3 whole-response JSON rule; never search prose for an array."""
    text = raw.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip() not in {"```", "```json"} or lines[-1].strip() != "```":
            raise ValueError("Malformed JSON code fence")
        text = "\n".join(lines[1:-1]).strip()
    predictions = strict_json(text)
    validate_array(predictions, group)
    return predictions


def _validate_group(group):
    if not isinstance(group, dict):
        raise ValueError("Missing or invalid batch group")
    samples = group.get("samples")
    labels = group.get("valid_labels")
    if not isinstance(samples, list) or not samples:
        raise ValueError("Batch samples must be a nonempty list")
    if type(group.get("n_samples")) is not int or group["n_samples"] != len(samples):
        raise ValueError("Batch n_samples differs from sample count")
    if not isinstance(labels, list) or not labels or any(not isinstance(x, str) for x in labels):
        raise ValueError("Batch valid_labels must be nonempty strings")
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate valid_labels")
    ids = []
    for row in samples:
        if not isinstance(row, dict) or not isinstance(row.get("prompt"), str):
            raise ValueError("Batch sample requires id and prompt")
        identifier = row.get("id")
        if not isinstance(identifier, str) or re.fullmatch(r"[0-9a-fA-F]{8}", identifier) is None:
            raise ValueError("Batch ID must be an eight-character hex string")
        ids.append(identifier)
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate input IDs")


def audit_extension(base: Path = EXT_DIR, parent: Path = PARENT_DIR) -> dict:
    """Read only the fixed, label-free allowlist; return a blocked audit report.

    No path from a supplied manifest is followed. In particular, datasets.json,
    baseline predictions, earlier model outputs, and legacy scores are not read.
    Identical prediction arrays are not used as evidence of contamination.
    """
    base, parent = Path(base), Path(parent)
    snapshot = {}
    global_issues = []
    # Include later submitted model directories without altering the historical protocol.
    # The original two expected models remain audited even if a directory is absent.
    submitted = sorted(p.name for p in (base / "predictions").iterdir()
                       if p.is_dir() and re.fullmatch(r"[a-z0-9][a-z0-9-]*", p.name)) if (base / "predictions").exists() else []
    models = tuple(MODELS) + tuple(name for name in submitted if name not in MODELS)
    additional_models = [name for name in models if name not in MODELS]
    if additional_models:
        global_issues.append(issue("ADDITIONAL_SUBMITTED_MODELS", "Models submitted after the historical two-model plan: " + ", ".join(additional_models) + ". This does not amend or validate that plan."))

    def read_bytes(path, key, problems):
        # Do not follow symlinks into answer keys or other unapproved files.
        if any(part.is_symlink() for part in (path, *path.parents)):
            problems.append(issue("SYMLINK_REJECTED", key))
            return None
        try:
            value = path.read_bytes()
        except (OSError, ValueError) as exc:
            problems.append(issue("FILE_UNREADABLE", f"{key}: {exc}"))
            return None
        snapshot[key] = sha256(value)
        return value

    def read_json(path, key, problems):
        raw = read_bytes(path, key, problems)
        if raw is None:
            return None
        try:
            return strict_json(raw.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            problems.append(issue("JSON_INVALID", f"{key}: {exc}"))
            return None

    def require_hash(declared, actual, field, problems):
        if not isinstance(declared, str) or re.fullmatch(r"[0-9a-f]{64}", declared) is None:
            problems.append(issue("HASH_MISSING_OR_INVALID", field))
        elif actual is None or declared != actual:
            problems.append(issue("HASH_MISMATCH", field))

    def expect(obj, field, expected, problems, context):
        if obj.get(field) != expected:
            problems.append(issue("FIELD_MISMATCH", f"{context}.{field} differs from expected value"))

    protocol = read_json(base / "protocol.json", "extension:protocol.json", global_issues)
    sources = read_json(base / "source_manifest.json", "extension:source_manifest.json", global_issues)
    batches = read_json(parent / "inference/batches.json", "parent:inference/batches.json", global_issues)
    read_json(parent / "protocol.json", "parent:protocol.json", global_issues)
    for value, name in ((protocol, "protocol"), (sources, "source_manifest"), (batches, "batches")):
        if not isinstance(value, dict):
            global_issues.append(issue("DOCUMENT_SHAPE_INVALID", name))
    protocol = protocol if isinstance(protocol, dict) else {}
    sources = sources if isinstance(sources, dict) else {}
    batches = batches if isinstance(batches, dict) else {}
    groups = batches.get("groups", {})
    groups = groups if isinstance(groups, dict) else {}
    require_hash(protocol.get("parent_protocol_sha256"), snapshot.get("parent:protocol.json"),
                 "protocol.parent_protocol_sha256", global_issues)
    require_hash(protocol.get("input_sha256"), snapshot.get("parent:inference/batches.json"),
                 "protocol.input_sha256", global_issues)
    source_input = sources.get("batches_json", {})
    source_input = source_input if isinstance(source_input, dict) else {}
    require_hash(source_input.get("sha256"), snapshot.get("parent:inference/batches.json"),
                 "source_manifest.batches_json.sha256", global_issues)
    expect(source_input, "path", "results/revision_v3/inference/batches.json", global_issues, "source_manifest.batches_json")
    expect(protocol, "models_requested", list(MODELS), global_issues, "protocol")
    expect(protocol, "expected_groups", list(GROUPS), global_issues, "protocol")
    expect(protocol, "expected_sessions", len(MODELS) * len(GROUPS), global_issues, "protocol")

    frozen_paths = (base / "frozen_manifest.json", base / "predictions/frozen_manifest.json")
    if not any(path.exists() for path in frozen_paths):
        global_issues.append(issue("PRE_SCORE_FREEZE_MISSING", "No prediction frozen_manifest exists; the source manifest covers inputs only."))
    else:
        global_issues.append(issue("PRE_SCORE_FREEZE_UNVERIFIABLE", "A current manifest cannot retroactively establish that these existing results were frozen before scoring."))
    global_issues.extend([
        issue("AGENT_TOOL_ISOLATION_UNVERIFIED", "The archived Agent-tool transport allowed tools; instruction-level prohibition does not establish tool or filesystem isolation."),
        issue("NATIVE_EVENT_PROVENANCE_MISSING", "The extension has no captured provider-native request/response and tool-event chain authenticating its delivered context and completion."),
        issue("MODEL_IDENTITY_UNVERIFIED", "Caller-written model names and a MODEL_ID self-report are not provider-returned model identity evidence."),
        issue("ANSWER_KEY_ISOLATION_UNVERIFIED", "Current output hashes and answer_keys_read=false cannot demonstrate pre-score answer-key isolation."),
    ])
    schedule_path = base / "session_schedule.json"
    if schedule_path.exists():
        schedule = read_json(schedule_path, "extension:session_schedule.json", global_issues)
        rows = schedule.get("schedule", []) if isinstance(schedule, dict) else []
        pairs = [(r.get("model_id_requested"), r.get("group_name")) for r in rows if isinstance(r, dict)]
        if len(pairs) != 18 or set(pairs) != {(m, g) for m in MODELS for g in GROUPS}:
            global_issues.append(issue("SCHEDULE_MODEL_MISMATCH", "The session schedule does not enumerate the same 18 model/group identities as the extension protocol and output paths."))

    prompt_bytes = {}
    group_issues = {}
    source_prompts = sources.get("prompt_files", {})
    source_prompts = source_prompts if isinstance(source_prompts, dict) else {}
    protocol_prompts = protocol.get("prompt_sha256", {})
    protocol_prompts = protocol_prompts if isinstance(protocol_prompts, dict) else {}
    for name in GROUPS:
        problems = []
        group_issues[name] = problems
        try:
            _validate_group(groups.get(name))
            dataset, condition_repeat = name.split("_", 1)
            condition, repeat = condition_repeat.rsplit("_r", 1)
            for field, expected in (("dataset", dataset), ("condition", condition), ("repeat", int(repeat))):
                expect(groups[name], field, expected, problems, f"batches.{name}")
        except (ValueError, TypeError, KeyError) as exc:
            problems.append(issue("BATCH_INVALID", f"{name}: {exc}"))
        key = f"parent:inference/prompts/{name}.txt"
        data = read_bytes(parent / f"inference/prompts/{name}.txt", key, problems)
        prompt_bytes[name] = data
        entry = source_prompts.get(name, {})
        entry = entry if isinstance(entry, dict) else {}
        require_hash(entry.get("sha256"), snapshot.get(key), f"source_manifest.prompt_files.{name}.sha256", problems)
        expect(entry, "path", f"results/revision_v3/inference/prompts/{name}.txt", problems, f"source_manifest.prompt_files.{name}")
        if data is not None:
            expect(entry, "bytes", len(data), problems, f"source_manifest.prompt_files.{name}")
        require_hash(protocol_prompts.get(name.rsplit("_r", 1)[0]), snapshot.get(key), f"protocol.prompt_sha256.{name}", problems)

    sessions = []
    for model in models:
        for name in GROUPS:
            problems = list(group_issues[name])
            folder = base / "predictions" / model
            stem = f"extension:predictions/{model}/{name}"
            predictions = read_json(folder / f"{name}_results.json", stem + "_results.json", problems)
            metadata = read_json(folder / f"{name}_metadata.json", stem + "_metadata.json", problems)
            request = read_json(folder / f"{name}_request.json", stem + "_request.json", problems)
            raw = read_bytes(folder / f"{name}_response.txt", stem + "_response.txt", problems)
            group = groups.get(name)
            array_valid = False
            if not any(p["code"] == "BATCH_INVALID" for p in problems):
                try:
                    validate_array(predictions, group)
                    array_valid = True
                except (ValueError, TypeError, KeyError) as exc:
                    problems.append(issue("PREDICTION_ARRAY_INVALID", str(exc)))
                if raw is not None:
                    try:
                        parsed = parse_raw_v3(raw.decode("utf-8"), group)
                        if parsed != predictions:
                            problems.append(issue("RAW_RESULT_MISMATCH", "Strictly parsed response differs from results JSON."))
                    except (ValueError, UnicodeError, TypeError, KeyError) as exc:
                        problems.append(issue("RAW_RESPONSE_INVALID", str(exc)))
            if not isinstance(metadata, dict):
                problems.append(issue("METADATA_INVALID", "Missing or non-object metadata."))
                metadata = {}
            if not isinstance(request, dict):
                problems.append(issue("REQUEST_INVALID", "Missing or non-object request."))
                request = {}
            for field, suffix in (("prediction_sha256", "_results.json"), ("request_sha256", "_request.json"), ("response_text_sha256", "_response.txt")):
                require_hash(metadata.get(field), snapshot.get(stem + suffix), f"metadata.{field}", problems)
            expect(metadata, "model_requested", model, problems, "metadata")
            expect(metadata, "group_name", name, problems, "metadata")
            expect(metadata, "status", "complete", problems, "metadata")
            expect(metadata, "validation_error", None, problems, "metadata")
            expect(metadata, "transport", "claude-code-agent-tool", problems, "metadata")
            expect(metadata, "answer_keys_read", False, problems, "metadata")
            if isinstance(group, dict):
                for field in ("dataset", "condition", "repeat", "n_samples"):
                    expect(metadata, field, group.get(field), problems, "metadata")
                expect(metadata, "n_predictions", len(group.get("samples", [])), problems, "metadata")
                for field in ("n_samples", "valid_labels"):
                    expect(request, field, group.get(field), problems, "request")
            for field, expected in (("model", model), ("group_name", name), ("transport", "claude-code-agent-tool"), ("instructions", protocol.get("instructions"))):
                expect(request, field, expected, problems, "request")
            require_hash(request.get("input_sha256"), snapshot.get("parent:inference/batches.json"), "request.input_sha256", problems)
            prompt_data = prompt_bytes[name]
            if prompt_data is not None:
                require_hash(request.get("prompt_file_sha256"), sha256(prompt_data), "request.prompt_file_sha256", problems)
                try:
                    prompt = prompt_data.decode("utf-8")
                    expect(request, "user_message", prompt, problems, "request")
                    instructions = protocol.get("instructions")
                    if not isinstance(instructions, str):
                        problems.append(issue("INSTRUCTIONS_INVALID", "Protocol instructions are not a string."))
                    else:
                        require_hash(request.get("full_prompt_sha256"), sha256((instructions + "\n\n" + prompt).encode()), "request.full_prompt_sha256", problems)
                except UnicodeError as exc:
                    problems.append(issue("PROMPT_ENCODING_INVALID", str(exc)))
            artifact_chain_consistent = not problems
            invocation_alias = metadata.get("model_alias")
            expected_family = model.removeprefix("claude-").split("-", 1)[0]
            if invocation_alias and invocation_alias != expected_family:
                problems.append(issue("INVOCATION_ALIAS_IDENTITY_UNVERIFIED", f"Claimed {model} uses invocation alias {invocation_alias}; native service identity is not established."))
            contaminated = str(metadata.get("integrity_flag", "")).upper() == "CONTAMINATED"
            contaminated = contaminated or str(metadata.get("data_integrity_flag", "")).upper() == "CONTAMINATED"
            suspect = str(metadata.get("data_integrity_flag", "")).lower() == "suspect" or str(metadata.get("integrity_flag", "")).lower() == "suspect"
            if contaminated:
                problems.append(issue("REPORTED_CONTAMINATED", "Existing metadata explicitly marks this session CONTAMINATED; it cannot be used in accuracy comparisons. This audit does not infer contamination from identical outputs."))
            if suspect:
                problems.append(issue("REPORTED_SUSPECT", "Existing metadata explicitly marks data integrity as suspect."))
            status = ("reported_contaminated" if contaminated else "reported_suspect" if suspect else
                      "provenance_unverified" if artifact_chain_consistent else "artifact_invalid")
            sessions.append({
                "model": model, "group": name, "status": status,
                "eligible_for_primary_comparison": False,
                "artifact_array_valid": array_valid,
                "artifact_chain_consistent": artifact_chain_consistent,
                "reported_contaminated": contaminated,
                "reported_suspect": suspect,
                "recorded_status": metadata.get("status"),
                "n_predictions": len(predictions) if isinstance(predictions, list) else None,
                "issues": problems + list(global_issues),
            })
    sorted_snapshot = dict(sorted(snapshot.items()))
    return {
        "audit_version": "claude-extension-provenance-audit-v2",
        "audited_models": list(models),
        "protocol_declared_sessions": protocol.get("expected_sessions"),
        "audited_at": datetime.now(timezone.utc).isoformat(),
        "status": "blocked",
        "answer_key_files_opened": False,
        "snapshot_kind": "post_hoc_current_bytes_not_pre_score_freeze",
        "source_snapshot_sha256": sorted_snapshot,
        "source_snapshot_manifest_sha256": sha256(json.dumps(sorted_snapshot, sort_keys=True, separators=(",", ":")).encode()),
        "global_issues": global_issues,
        "counts": {
            "planned": len(sessions),  # Legacy key: audited model/group records, not a verified schedule.
            "audited_records": len(sessions),
            "artifact_arrays_valid": sum(s["artifact_array_valid"] for s in sessions),
            "artifact_chains_consistent": sum(s["artifact_chain_consistent"] for s in sessions),
            "eligible_primary": 0,
            "reported_contaminated": sum(s["reported_contaminated"] for s in sessions),
            "reported_suspect": sum(s["reported_suspect"] for s in sessions),
        },
        "sessions": sessions,
        "interpretation": f"All {len(sessions)} audited model/group records are excluded from the primary controlled comparison. Artifact consistency is separate from verified model inference. No answer-key dataset files or legacy scores were read; saved response or metadata content may itself contain prior label disclosures. No historical freeze, model identity or event trace is reconstructed. Matching outputs alone are not evidence of copying.",
    }


def write_reports(report: dict, base: Path = EXT_DIR) -> None:
    """Write only derived audit reports; never repair or freeze source artifacts."""
    output = Path(base) / "audit"
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = report["counts"]
    lines = ["# Claude extension provenance audit", "", "**Status: blocked for primary comparison.**", "",
             f"Audited at: {report['audited_at']}", "",
             f"Audited model/group records: {counts['audited_records']}; historical protocol declares {report['protocol_declared_sessions']}; structurally valid prediction arrays: {counts['artifact_arrays_valid']}; "
             f"consistent current artifact chains: {counts['artifact_chains_consistent']}; eligible primary sessions: {counts['eligible_primary']}; "
             f"sessions explicitly marked contaminated: {counts['reported_contaminated']}.", "",
             report["interpretation"], "", "## Global limitations", ""]
    lines += [f"- `{p['code']}`: {p['detail']}" for p in report["global_issues"]]
    lines += ["", "## Session inventory", "", "| Requested model | Group | Array valid | Status | Eligible |", "|---|---|---|---|---|"]
    for session in report["sessions"]:
        lines.append(f"| {session['model']} | {session['group']} | {session['artifact_array_valid']} | {session['status']} | false |")
    global_findings = {(p["code"], p["detail"]) for p in report["global_issues"]}
    lines += ["", "## Session-specific findings", ""]
    for session in report["sessions"]:
        findings = [p for p in session["issues"] if (p["code"], p["detail"]) not in global_findings]
        if findings:
            lines.append(f"- `{session['model']}/{session['group']}`:")
            lines.extend(f"  - `{p['code']}`: {p['detail']}" for p in findings)
    lines += ["", "## Snapshot meaning", "",
              "The JSON report contains SHA-256 hashes of the current allowlisted input and session artifacts. "
              "These hashes describe the audit-time bytes only; they are not a pre-score freeze. "
              "No result, metadata, protocol, original v3 core file, or historical manifest was changed.", "",
              "This audit intentionally does not compute accuracy or confidence intervals and never reads datasets.json. "
              "A new controlled inference run with preserved provider-native request, response, identity, and tool-isolation evidence "
              "would be a separate experiment; current artifacts cannot be retroactively made eligible.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=EXT_DIR)
    parser.add_argument("--parent", type=Path, default=PARENT_DIR)
    args = parser.parse_args(argv)
    report = audit_extension(args.base, args.parent)
    write_reports(report, args.base)
    print(json.dumps({"status": report["status"], "counts": report["counts"], "report": str(args.base / "audit/report.json")}, indent=2))
    return 2  # A successful audit execution reports blocked scientific eligibility.


if __name__ == "__main__":
    raise SystemExit(main())
