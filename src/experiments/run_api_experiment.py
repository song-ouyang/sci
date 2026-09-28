"""
Run LLM fault diagnosis experiments via the Anthropic API.
Each sample is sent as an independent API call — no batch effects.
"""

import json
import time
import sys
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import anthropic

BATCHES = Path(__file__).parent.parent.parent / "results" / "batches"

MODELS = {
    "haiku": {
        "model_id": "claude-haiku-4-5-20251001",
        "display": "Haiku 4.5",
        "suffix": "_results_haiku_api.json",
    },
    "sonnet": {
        "model_id": "claude-sonnet-5",
        "display": "Sonnet 5",
        "suffix": "_results_sonnet_api.json",
    },
    "opus": {
        "model_id": "claude-opus-4-6",
        "display": "Opus 4.6",
        "suffix": "_results_opus_api.json",
    },
}

CWRU_CLASSES = ["Normal", "IR007", "IR014", "IR021", "B007", "B014", "B021", "OR007", "OR014", "OR021"]
CMAPSS_CLASSES = ["Healthy", "Degrading", "Critical"]

VALID_LABELS = {
    "cwru": CWRU_CLASSES,
    "cmapss": CMAPSS_CLASSES,
    "tep": ["Normal", "Fault_1", "Fault_2", "Fault_4", "Fault_6", "Fault_7", "Fault_13"],
}


def parse_label(response: str, valid: list[str]) -> str:
    text = response.strip()
    for prefix in ["Diagnosis:", "Health stage:", "Condition:"]:
        if prefix in text:
            text = text.split(prefix)[-1].strip().split("\n")[0].strip().rstrip(".")

    for label in valid:
        if label.lower() == text.lower():
            return label
    for label in valid:
        if label.lower() in text.lower():
            return label
    return "UNPARSEABLE"


def classify_one(client, model_id: str, prompt: str) -> str:
    resp = client.messages.create(
        model=model_id,
        max_tokens=256,
        temperature=0.0,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def run_experiment(dataset: str, strategy: str, model_key: str, workers: int = 5):
    batch_file = BATCHES / f"{dataset}_{strategy}.json"
    if not batch_file.exists():
        print(f"  SKIP: {batch_file} not found")
        return None

    model = MODELS[model_key]
    out_file = BATCHES / f"{dataset}_{strategy}{model['suffix']}"

    if out_file.exists():
        existing = json.load(open(out_file))
        if existing and "predicted" in existing[0]:
            n_correct = sum(1 for d in existing if d.get("predicted") == d.get("true_label"))
            print(f"  EXISTS: {out_file.name} ({n_correct}/{len(existing)} correct)")
            return existing

    batch = json.load(open(batch_file))
    client = anthropic.Anthropic()
    valid = VALID_LABELS[dataset]
    results = [None] * len(batch)
    done = 0

    def process(idx, item):
        raw = classify_one(client, model["model_id"], item["prompt"])
        predicted = parse_label(raw, valid)
        return idx, {
            "id": item["id"],
            "true_label": item["true_label"],
            "predicted": predicted,
            "raw_response": raw[:200],
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process, i, item): i for i, item in enumerate(batch)}
        for fut in as_completed(futures):
            idx, result = fut.result()
            results[idx] = result
            done += 1
            if done % 20 == 0 or done == len(batch):
                n_ok = sum(1 for r in results[:done] if r and r["predicted"] == r["true_label"])
                print(f"    [{done}/{len(batch)}] running acc={n_ok}/{done}")

    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    n_correct = sum(1 for r in results if r["predicted"] == r["true_label"])
    n_parse = sum(1 for r in results if r["predicted"] != "UNPARSEABLE")
    print(f"  DONE: {out_file.name} — {n_correct}/{len(results)} correct, {n_parse}/{len(results)} parsed")
    return results


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY first.")
        print("  1. Go to https://console.anthropic.com/")
        print("  2. Create API key")
        print("  3. export ANTHROPIC_API_KEY=sk-ant-...")
        print("\nEstimated cost: ~$5 total (mostly Opus)")
        sys.exit(1)

    strategies = ["zero_shot", "few_shot", "cot", "domain"]

    # CWRU: all 3 models × 4 strategies
    print("\n=== CWRU (3 models × 4 strategies) ===")
    for model_key in ["haiku", "sonnet", "opus"]:
        print(f"\n--- {MODELS[model_key]['display']} ---")
        for st in strategies:
            print(f"  {st}:")
            run_experiment("cwru", st, model_key, workers=5)

    # C-MAPSS & TEP: Opus only
    for ds in ["cmapss", "tep"]:
        print(f"\n=== {ds.upper()} (Opus only) ===")
        for st in strategies:
            print(f"  {st}:")
            run_experiment(ds, st, "opus", workers=5)


if __name__ == "__main__":
    main()
