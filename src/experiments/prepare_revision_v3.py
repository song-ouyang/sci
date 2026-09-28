"""Freeze label-free grouped inference payloads and a paired-study protocol.

This preparation stage can read labels to validate a stratified design. The
separate inference runner cannot read this file's input or any answer key.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "results/revision_v3"
MODELS = ["gpt-5.5", "gpt-5.6-sol", "gpt-6-astra"]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_once(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode()
    if path.exists() and path.read_bytes() != raw:
        raise ValueError(f"Refusing to change frozen preparation artifact: {path}")
    path.write_bytes(raw)


def render(group):
    return (
        "Classify each industrial sensor observation using only its own sensor "
        "values and the task definition. IDs are random bookkeeping identifiers, "
        "not diagnostic evidence. Do not infer a class balance or use neighboring "
        "observations to choose a label. Return only one JSON array in the exact "
        "input order, with one object per observation and exactly the keys id "
        "and predicted. Preserve each ID and use an exact case-sensitive label "
        "from valid_labels. No explanation or other fields.\n\n"
        + json.dumps({"valid_labels": group["valid_labels"],
                      "samples": group["samples"]}, ensure_ascii=False, indent=2)
        + "\n"
    )


def main():
    data_path = BASE / "data/datasets.json"
    data = json.loads(data_path.read_text())["datasets"]
    groups = {}
    for dataset, conditions in [("cmapss", ["clean"]), ("tep", ["clean", "misnamed"])]:
        ds = data[dataset]
        samples = [{"id": x["id"], "prompt": x["prompt"]} for x in ds["test"]]
        assert len({x["id"] for x in samples}) == len(samples)
        for condition in conditions:
            paired = [dict(x) for x in samples]
            if condition == "misnamed":
                names = ds["feature_display_names"]
                assert len(names) == 52 and len(set(names)) == 52
                # Rotate field labels including their stated units by one place.
                # Numeric strings, sample IDs, instructions and order stay fixed.
                replacement = {name: names[(i + 1) % 52] for i, name in enumerate(names)}
                for sample in paired:
                    lines = sample["prompt"].splitlines(keepends=True)
                    count = 0
                    for i, line in enumerate(lines):
                        field, sep, value = line.partition(":")
                        if field in replacement:
                            lines[i] = replacement[field] + sep + value
                            count += 1
                    if count != 52:
                        raise ValueError(f"Expected exactly 52 sensor lines; got {count}")
                    sample["prompt"] = "".join(lines)
            for repeat in range(1, 4):
                name = f"{dataset}_{condition}_r{repeat}"
                groups[name] = {"dataset": dataset, "condition": condition,
                                "repeat": repeat, "valid_labels": ds["valid_labels"],
                                "n_samples": len(paired), "samples": paired}
    manifest = {"schema": "revision-v3-label-free-v1", "groups": groups}
    target = BASE / "inference/batches.json"
    write_once(target, manifest)
    txt_hashes = {}
    for name, group in groups.items():
        raw = render(group).encode()
        path = BASE / "inference/prompts" / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"Frozen prompt changed: {path}")
        path.write_bytes(raw)
        txt_hashes[name] = sha(raw)
    protocol_path = BASE / "protocol.json"
    created = json.loads(protocol_path.read_text())["frozen_at"] if protocol_path.exists() else datetime.now(timezone.utc).isoformat()
    protocol = {
        "version": "revision-v3", "frozen_at": created,
        "study_type": "exploratory fixed-size study; not a prospectively powered confirmatory trial",
        "models": MODELS, "reasoning_effort": "medium", "repeats": 3,
        "expected_groups": list(groups), "expected_sessions": len(groups) * len(MODELS),
        "dataset_sha256": sha(data_path.read_bytes()),
        "input_sha256": sha(target.read_bytes()), "prompt_sha256": txt_hashes,
        "primary_repeat": 1, "primary_metrics": ["macro_f1", "accuracy"],
        "paired_contrast": "TEP clean minus misnamed; fixed cyclic +1 sensor-label/unit rotation; numeric strings unchanged",
        "contrast_scope": "synthetic gross schema-fault stress test, not a reconstruction of historical v1 defects",
        "primary_ci": "95% Wilson for repeat-1 accuracy; group percentile bootstrap for macro F1 and paired differences, conditional on fixed grouped responses",
        "bootstrap": {"replicates": 10000, "seed": 20260926},
        "repeat_reporting": "Each repeat separately plus min/max; no repeated prediction treated as an independent sample",
        "inference_unit": "One fresh session per model/dataset/condition/repeat, with all selected observations in one fixed shuffled batch",
        "independent_sample_calls": False,
        "order": "Fixed label-independent random permutation, identical across paired conditions and repetitions; session schedule deterministically shuffled by model",
        "key_isolation": "Runner reads only label-free batches, prompts and protocol; empty inference working directory and all tools disabled",
        "failure_policy": "At most 2 transport attempts; semantic parse/coverage/label errors fail closed with logs retained; no prediction repair or score-guided rerun",
        "scoring_policy": "Freeze every completed result and supporting log before opening answer labels; score all samples and all predeclared conditions",
        "baselines": "Fixed Random Forest, scaled RBF SVM and training-majority classifier; identical rounded sensor features and held-out observations; no test tuning",
        "sample_size": {name: len(ds["test"]) for name, ds in data.items()},
        "power_note": "Convenience sample chosen before inference for audit feasibility. Around n=60/70, worst-case binomial 95% accuracy half-widths are about 12/11 percentage points; small effects and stable rankings are not resolved.",
        "limitations": ["One public simulated subset per task", "Known benchmarks may occur in pretraining", "Grouped context couples observations", "Only three repeated sessions per condition", "Fixed order and one synthetic mapping defect", "CLI model aliases, hidden base instructions and temperature prevent exact independent replay"],
    }
    write_once(protocol_path, protocol)
    print(f"Frozen {len(groups)} groups / {protocol['expected_sessions']} sessions; "
          f"samples={protocol['sample_size']}; protocol={sha(protocol_path.read_bytes())}")


if __name__ == "__main__":
    main()
