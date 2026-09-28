"""
Save and validate a single Claude extension session result.

Usage:
  python save_claude_session.py <model_id> <group_name> <repeat> <raw_response_text>

Or import and call save_session() directly.
"""
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXT_DIR = ROOT / "results/revision_v3_claude_extension"
BATCHES_PATH = ROOT / "results/revision_v3/inference/batches.json"
PROMPTS_DIR = ROOT / "results/revision_v3/inference/prompts"

INSTRUCTIONS = (
    "This is a text-only controlled classification experiment. Use only the "
    "supplied prompts and your existing knowledge. Do not use tools, inspect "
    "files, browse, delegate, or ask questions. Classify every sample from its "
    "own prompt. Do not infer labels from IDs, sample order, other samples, or "
    "an assumed class distribution. Return only the requested JSON array."
)

VALID_MODEL_IDS = {"claude-sonnet-4-6", "claude-opus-5-5", "claude-fable-5-1"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_predictions(predictions, group):
    if not isinstance(predictions, list):
        raise ValueError("Response is not a JSON array")
    if len(predictions) != len(group["samples"]):
        raise ValueError(
            f"Incomplete prediction count: got {len(predictions)}, expected {len(group['samples'])}"
        )
    for result, sample in zip(predictions, group["samples"]):
        if not isinstance(result, dict) or set(result) != {"id", "predicted"}:
            raise ValueError(
                f"Each prediction must have exactly keys id and predicted; got {set(result)}"
            )
        if not isinstance(result["id"], str) or not re.fullmatch(r"[0-9a-fA-F]{8}", result["id"]):
            raise ValueError(f"Invalid id format: {result['id']!r}")
        if result["id"] != sample["id"]:
            raise ValueError(
                f"ID mismatch at position {group['samples'].index(sample)}: "
                f"got {result['id']!r}, expected {sample['id']!r}"
            )
        if result["predicted"] not in group["valid_labels"]:
            raise ValueError(
                f"Invalid label {result['predicted']!r} not in {group['valid_labels']}"
            )


def parse_raw(raw: str, group: dict) -> list:
    text = raw.strip()
    # Strip markdown code fence if present
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1]).strip()
    # Find the JSON array
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        text = match.group(0)
    predictions = json.loads(text)
    validate_predictions(predictions, group)
    return predictions


def save_session(
    model_id: str,
    group_name: str,
    raw_response: str,
    agent_metadata=None,
    started_at=None,
    completed_at=None,
    attempt: int = 1,
):
    """
    Validate and save a completed session result.

    Returns (status, predictions_or_error_message).
    status is 'complete' or 'failed'.
    """
    assert model_id in VALID_MODEL_IDS, f"Unknown model: {model_id}"

    batches = json.loads(BATCHES_PATH.read_text(encoding="utf-8"))
    group = batches["groups"][group_name]

    dataset = group["dataset"]
    condition = group["condition"]
    repeat = group["repeat"]
    prompt_file = PROMPTS_DIR / f"{group_name}.txt"
    prompt_txt = prompt_file.read_text(encoding="utf-8")
    full_prompt = INSTRUCTIONS + "\n\n" + prompt_txt

    out_dir = EXT_DIR / "predictions" / model_id
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = group_name
    request_path = out_dir / f"{stem}_request.json"
    response_txt_path = out_dir / f"{stem}_response.txt"
    results_path = out_dir / f"{stem}_results.json"
    metadata_path = out_dir / f"{stem}_metadata.json"

    request_obj = {
        "model": model_id,
        "transport": "claude-code-agent-tool",
        "instructions": INSTRUCTIONS,
        "user_message": prompt_txt,
        "full_prompt_sha256": sha256(full_prompt.encode("utf-8")),
        "prompt_file_sha256": sha256(prompt_file.read_bytes()),
        "input_sha256": sha256(BATCHES_PATH.read_bytes()),
        "group_name": group_name,
        "n_samples": group["n_samples"],
        "valid_labels": group["valid_labels"],
    }
    request_path.write_text(json.dumps(request_obj, indent=2) + "\n", encoding="utf-8")

    response_txt_path.write_text(raw_response, encoding="utf-8")

    completed_at = completed_at or now()
    status = "pending"
    validation_error = None
    predictions = None

    try:
        predictions = parse_raw(raw_response, group)
        status = "complete"
    except Exception as exc:
        status = "failed"
        validation_error = str(exc)

    meta = {
        "protocol": "claude-extension-grouped-revision-v3",
        "model_requested": model_id,
        "model_alias": "sonnet" if model_id == "claude-sonnet-4-6" else "opus",
        "transport": "claude-code-agent-tool",
        "dataset": group["dataset"],
        "condition": condition,
        "repeat": repeat,
        "group_name": group_name,
        "n_samples": group["n_samples"],
        "started_at": started_at,
        "completed_at": completed_at,
        "attempt": attempt,
        "status": status,
        "validation_error": validation_error,
        "response_text_sha256": sha256(raw_response.encode("utf-8")),
        "request_sha256": sha256(request_path.read_bytes()),
        "answer_keys_read": False,
        "independent_session_per_group": True,
        "independent_session_per_sample": False,
        "metadata_unavailable": [
            "token_usage",
            "server_returned_model_id",
            "stop_finish_reason",
            "request_session_id",
            "temperature",
        ],
        "metadata_unavailable_reason": (
            "Claude CLI subprocess returns 'OAuth session expired'; "
            "Agent tool interface does not expose these fields."
        ),
        **(agent_metadata or {}),
    }

    if status == "complete":
        results_path.write_text(
            json.dumps(predictions, indent=2) + "\n", encoding="utf-8"
        )
        meta["prediction_sha256"] = sha256(results_path.read_bytes())
        meta["n_predictions"] = len(predictions)

    metadata_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    label_counts = {}
    if predictions:
        for p in predictions:
            label_counts[p["predicted"]] = label_counts.get(p["predicted"], 0) + 1

    print(
        f"{'OK' if status == 'complete' else 'FAIL'} {model_id} {group_name}: "
        f"{status}"
        + (f" — {len(predictions)} preds, dist={label_counts}" if predictions else "")
        + (f" — ERROR: {validation_error}" if validation_error else "")
    )
    return status, predictions if status == "complete" else validation_error


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: save_session.py <model_id> <group_name> <response_text_file>")
        sys.exit(1)
    model_id = sys.argv[1]
    group_name = sys.argv[2]
    response_file = Path(sys.argv[3])
    raw = response_file.read_text(encoding="utf-8")
    status, result = save_session(model_id, group_name, raw)
    sys.exit(0 if status == "complete" else 1)
