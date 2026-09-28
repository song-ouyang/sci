"""
Run Claude experiments on v2 batches (fixed TEP, anonymized IDs, larger samples).
Outputs to results/batches_v2/.
"""
import argparse
import hashlib
import inspect
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic

ROOT = Path(__file__).parent.parent.parent
BATCH_DIR = ROOT / "results" / "batches_clean_v2"
OUT_DIR = ROOT / "results" / "batches_v2"
PARSER_PROTOCOL = "exact-label-or-single-final-v1"
MAX_OUTPUT_TOKENS = 4096
ACCEPTED_STOP_REASONS = {"end_turn", "stop_sequence"}

MODELS = {
    "haiku":  {"model_id": "claude-haiku-4-5-20251001", "suffix": "_haiku.json"},
    "sonnet": {"model_id": "claude-sonnet-5",           "suffix": "_sonnet.json"},
    "opus":   {"model_id": "claude-opus-4-6",           "suffix": "_opus.json"},
}

VALID_LABELS = {
    "cwru":   ["Normal", "IR007", "IR014", "IR021", "B007", "B014", "B021", "OR007", "OR014", "OR021"],
    "cmapss": ["Healthy", "Degrading", "Critical"],
    "tep":    ["Normal", "Fault_1", "Fault_2", "Fault_4", "Fault_6", "Fault_7", "Fault_13"],
}


def parse_label(text: str, valid: list[str], *, stop_reason="end_turn") -> str:
    """Accept an exact label or one explicit, final conclusion; never substrings.

    Reasoning may precede one final tag, but a tag must be the last nonempty
    line. Multiple tagged conclusions are ambiguous even when they agree.
    Label spelling, punctuation and case are not repaired by the parser.
    """
    if stop_reason not in ACCEPTED_STOP_REASONS or not isinstance(text, str):
        return "UNPARSEABLE"
    text = text.strip()
    if text in valid:
        return text
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    tagged = []
    for index, line in enumerate(lines):
        match = re.fullmatch(
            r"(?:Diagnosis|Health stage|Condition|Answer|Final|Final answer|Final diagnosis):\s*(.*)",
            line,
        )
        xml = re.fullmatch(r"<final>(.*)</final>", line)
        if match or xml:
            tagged.append((index, (match or xml).group(1).strip()))
    if len(tagged) == 1 and tagged[0][0] == len(lines) - 1:
        value = tagged[0][1]
        if value in valid:
            return value
    return "UNPARSEABLE"


def extract_response(response: dict) -> dict:
    """Preserve the complete API response and reject incomplete generations."""
    content = response.get("content", [])
    blocks = [block["text"] for block in content
              if isinstance(block, dict) and block.get("type") == "text"
              and isinstance(block.get("text"), str)] if isinstance(content, list) else []
    raw = "\n".join(blocks)
    stop_reason = response.get("stop_reason")
    error = None
    if stop_reason not in ACCEPTED_STOP_REASONS:
        error = f"incomplete_or_unsupported_stop_reason:{stop_reason}"
    elif not blocks:
        error = "no_text_block"
    elif not raw.strip():
        error = "empty_text"
    return {"raw": raw, "stop_reason": stop_reason,
            "response_error": error, "response": response}


def classify_one(client, model_id: str, prompt: str, *,
                 max_tokens: int = MAX_OUTPUT_TOKENS, return_response: bool = False):
    resp = client.messages.create(
        model=model_id, max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    details = extract_response(resp.model_dump(mode="json"))
    if return_response:
        return details
    return "" if details["response_error"] else details["raw"].strip()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cache_provenance(batch_file, batch, model_id, valid, max_tokens):
    parser_source = inspect.getsource(parse_label) + inspect.getsource(extract_response)
    protocol = {
        "protocol": PARSER_PROTOCOL,
        "input_sha256": sha256(batch_file.read_bytes()),
        "prompt_sha256": [sha256(item["prompt"].encode()) for item in batch],
        "sample_ids": [item["id"] for item in batch],
        "model_requested": model_id,
        "valid_labels": valid,
        "max_output_tokens": max_tokens,
        "parser_sha256": sha256(parser_source.encode()),
        "accepted_stop_reasons": sorted(ACCEPTED_STOP_REASONS),
        "answer_keys_read": False,
    }
    return {**protocol, "protocol_sha256": sha256(
        json.dumps(protocol, sort_keys=True, ensure_ascii=False).encode())}


def load_cache(out_file, metadata_file, expected, batch, valid):
    if not metadata_file.exists():
        raise ValueError(f"Legacy cache has no provenance: {out_file}. Use a new output directory.")
    metadata = json.loads(metadata_file.read_text())
    if metadata.get("provenance") != expected or metadata.get("status") != "complete":
        raise ValueError(f"Cache provenance mismatch: {out_file}; refusing reuse or overwrite")
    if metadata.get("results_sha256") != sha256(out_file.read_bytes()):
        raise ValueError(f"Cache results hash mismatch: {out_file}")
    existing = json.loads(out_file.read_text())
    if not isinstance(existing, list) or len(existing) != len(batch):
        raise ValueError(f"Cache coverage mismatch: {out_file}")
    for item, record in zip(batch, existing):
        if not isinstance(record, dict) or record.get("id") != item["id"]:
            raise ValueError(f"Cache sample ID/order mismatch: {out_file}")
        if not isinstance(record.get("response"), dict):
            raise ValueError(f"Cache lacks complete response: {out_file}")
        details = extract_response(record["response"])
        if any(record.get(key) != value for key, value in details.items()):
            raise ValueError(f"Cache response provenance mismatch: {out_file}")
        expected_label = "UNPARSEABLE" if details["response_error"] else parse_label(
            details["raw"], valid, stop_reason=details["stop_reason"])
        if record.get("predicted") != expected_label:
            raise ValueError(f"Cache parsing mismatch: {out_file}")
    return existing


def run_experiment(dataset: str, strategy: str, n: int, model_key: str, workers: int = 5,
                   *, output_dir=None, max_tokens: int = MAX_OUTPUT_TOKENS):
    batch_name = f"{dataset}_{strategy}_v2_{n}"
    batch_file = BATCH_DIR / f"{batch_name}.json"
    if not batch_file.exists():
        print(f"  SKIP: {batch_file.name} not found")
        return

    model = MODELS[model_key]
    out_dir = Path(output_dir) if output_dir is not None else OUT_DIR
    out_file = out_dir / f"{batch_name}{model['suffix']}"
    metadata_file = out_file.with_suffix(".metadata.json")
    batch = json.loads(batch_file.read_text())
    if (not isinstance(batch, list) or len(batch) != n or
            any(not isinstance(item, dict) or not isinstance(item.get("prompt"), str)
                or not item["prompt"].strip() or "id" not in item for item in batch)):
        raise ValueError(f"Invalid input batch: {batch_file}")
    ids = [item["id"] for item in batch]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate input IDs: {batch_file}")
    valid = VALID_LABELS[dataset]
    provenance = cache_provenance(batch_file, batch, model["model_id"], valid, max_tokens)
    if out_file.exists():
        existing = load_cache(out_file, metadata_file, provenance, batch, valid)
        done = sum(1 for d in existing if "predicted" in d and d["predicted"] != "UNPARSEABLE")
        print(f"  CACHED {out_file.name} ({done}/{len(existing)} valid)")
        return existing

    if metadata_file.exists():
        raise ValueError(f"Orphan cache metadata: {metadata_file}; refusing overwrite")
    if workers < 1 or max_tokens < 1:
        raise ValueError("workers and max_tokens must be positive")
    client = anthropic.Anthropic()
    results = [None] * len(batch)

    def process(idx, item):
        details = classify_one(client, model["model_id"], item["prompt"],
                               max_tokens=max_tokens, return_response=True)
        pred = "UNPARSEABLE" if details["response_error"] else parse_label(
            details["raw"], valid, stop_reason=details["stop_reason"])
        return idx, {"id": item["id"], "predicted": pred, **details}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process, i, item): i for i, item in enumerate(batch)}
        done_count = 0
        for fut in as_completed(futs):
            idx, rec = fut.result()
            results[idx] = rec
            done_count += 1
            if done_count % 10 == 0:
                print(f"    {batch_name}/{model_key}: {done_count}/{len(batch)} complete")

    out_dir.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting historical or concurrently created results.
    with out_file.open("x") as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    metadata = {"status": "complete", "provenance": provenance,
                "results_sha256": sha256(out_file.read_bytes())}
    with metadata_file.open("x") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    done = sum(record["predicted"] != "UNPARSEABLE" for record in results)
    print(f"  ✓ {out_file.name}: {done}/{len(results)} parseable; no answer keys read")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="*", default=["haiku", "sonnet", "opus"])
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS)
    args = parser.parse_args()
    models = args.models
    if any(model not in MODELS for model in models):
        parser.error(f"models must be drawn from {list(MODELS)}")

    experiments = [
        ("tep",    42),
        ("cmapss", 60),
        ("cwru",   100),
    ]
    strategies = ["zero_shot", "few_shot", "cot", "domain"]

    for model_key in models:
        print(f"\n{'='*60}\nModel: {model_key}\n{'='*60}")
        for dataset, n in experiments:
            print(f"\n  [{dataset.upper()} n={n}]")
            for strategy in strategies:
                run_experiment(dataset, strategy, n, model_key, workers=args.workers,
                               output_dir=args.output_dir, max_tokens=args.max_output_tokens)

    print("\nDone. Results in", args.output_dir)
