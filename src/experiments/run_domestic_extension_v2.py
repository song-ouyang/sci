"""Prepare and run an additive DeepSeek/Qwen extension of frozen revision-v3 inputs."""

import argparse
import hashlib
import json
from pathlib import Path
import random
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments import domestic_api_v2 as api
from src.experiments.run_revision_v3 import INSTRUCTIONS

BASE = ROOT / "results/revision_v3"
DEFAULT_OUTPUT = ROOT / "results/domestic_extension_v2"
SOURCE_FILES = [Path(__file__), Path(api.__file__), ROOT / "src/experiments/run_revision_v3.py"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(output):
    if output.exists():
        raise ValueError("Output already exists; do not overwrite a frozen extension")
    original_path = BASE / "protocol.json"
    original = json.loads(original_path.read_text())
    input_path = BASE / "inference/batches.json"
    if sha(input_path) != original["input_sha256"]:
        raise ValueError("Original inference input hash changed")
    groups = json.loads(input_path.read_text())["groups"]
    if set(groups) != set(original["expected_groups"]):
        raise ValueError("Unexpected original groups")
    batches = {}
    for name, group in groups.items():
        prompt_path = BASE / "inference/prompts" / (name + ".txt")
        if sha(prompt_path) != original["prompt_sha256"][name]:
            raise ValueError("Original prompt hash changed: " + name)
        batch = {"messages": [{"role": "system", "content": INSTRUCTIONS},
                              {"role": "user", "content": prompt_path.read_text(encoding="utf-8")}],
                 "expected_ids": [s["id"] for s in group["samples"]],
                 "allowed_labels": group["valid_labels"]}
        api.validate_batch(batch)
        batches[name] = batch
    configs = {name: api.provider_config(name) for name in api.PROVIDERS}
    jobs = []
    for provider, config in configs.items():
        names = list(batches)
        random.Random("domestic-extension-v2:" + provider).shuffle(names)
        for name in names:
            jobs.append({"provider": provider, "group": name,
                         "dataset": groups[name]["dataset"], "condition": groups[name]["condition"],
                         "repeat": groups[name]["repeat"],
                         "request": api.build_payload(config, batches[name], True, 65536)})
    output.mkdir(parents=True)
    for name, batch in batches.items():
        (output / "batches").mkdir(exist_ok=True)
        api.write_json(output / "batches" / (name + ".json"), batch)
    for config in configs.values():
        config.pop("key_configured", None)
    api.write_json(output / "plan.json", {
        "schema": "domestic-extension-v2", "prepared_at": api.utc_now(),
        "study_type": "Exploratory budget-corrected run after original results and v1 transport failures; no v1 answers scored",
        "supersedes": "domestic_extension (stopped budget-misconfiguration run, all records retained)",
        "budget_rationale": "Both providers cap combined reasoning plus final output at 65536 tokens. Qwen also caps thinking at 16384; DeepSeek high remains a different provider-defined effort. Do not claim equivalent compute.",
        "execution": "At most 2 concurrent fresh sessions per provider; fixed shuffled submission order; no automatic retry.",
        "original_protocol_sha256": sha(original_path), "original_input_sha256": sha(input_path),
        "primary_repeat": 1, "n_unique_source_units": 130, "expected_sessions": len(jobs),
        "providers": configs, "jobs": jobs,
        "failure_policy": "One transport attempt per job; no automatic retries or semantic repair. Failures are terminal.",
        "interface_difference": "API system role replaces Codex developer instructions; identical instruction text and user prompts. Provider reasoning settings are not equivalent.",
        "scoring_policy": "No answer keys read here. Freeze all outputs before separate scoring. Invalid whole batches are N/A.",
        "source_sha256": {str(p.relative_to(ROOT)): sha(p) for p in SOURCE_FILES},
        "batch_sha256": {name: sha(output / "batches" / (name + ".json")) for name in batches},
    })
    (output / "plan.sha256").write_text(sha(output / "plan.json") + "\n")
    print("Prepared %d sessions; no model calls: %s" % (len(jobs), output))


def verify(output):
    path = output / "plan.json"
    if sha(path) != (output / "plan.sha256").read_text().strip():
        raise ValueError("Frozen plan changed")
    plan = json.loads(path.read_text())
    for name, expected in plan["source_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Runner source changed: " + name)
    for name, expected in plan["batch_sha256"].items():
        if sha(output / "batches" / (name + ".json")) != expected:
            raise ValueError("Frozen batch changed: " + name)
    return plan


def run(output, provider):
    plan = verify(output)
    jobs = [j for j in plan["jobs"] if provider == "all" or j["provider"] == provider]
    for name in {j["provider"] for j in jobs}:
        config = api.provider_config(name)
        if not config["key_configured"]:
            raise ValueError("Missing " + config["key_env"])
        for key in ("model", "base_url"):
            if config[key] != plan["providers"][name][key]:
                raise ValueError("Configuration differs from frozen plan: " + name + "/" + key)
    def run_job(job):
        folder = output / "sessions" / job["provider"] / job["group"]
        result_path = folder / "result.json"
        if folder.exists():
            if not result_path.exists():
                raise ValueError("Interrupted job requires review; refusing automatic rerun: " + str(folder))
            result = json.loads(result_path.read_text())
            run_folder = folder / result["run_folder"]
            if sha(run_folder / "sha256.json") != result["manifest_sha256"]:
                raise ValueError("Existing result manifest changed")
            for name, expected in json.loads((run_folder / "sha256.json").read_text()).items():
                if sha(run_folder / name) != expected:
                    raise ValueError("Existing result changed: " + name)
            status = json.loads((run_folder / "metadata.json").read_text())["status"]
            if status != result["status"]:
                raise ValueError("Existing result status changed")
            print("KEEP %s %s: %s" % (job["provider"], job["group"], status), flush=True)
            return status
        batch = json.loads((output / "batches" / (job["group"] + ".json")).read_text())
        config = api.provider_config(job["provider"])
        if api.build_payload(config, batch, True, 65536) != job["request"]:
            raise ValueError("Request differs from frozen plan")
        folder.mkdir(parents=True)
        print("START %s %s" % (job["provider"], job["group"]), flush=True)
        run_folder, metadata = api.run_batch(config, batch, folder, True, 65536, 120)
        api.write_json(result_path, {"status": metadata["status"], "run_folder": run_folder.name,
                                    "manifest_sha256": sha(run_folder / "sha256.json")})
        print("%s %s: %s" % (job["provider"], job["group"], metadata["status"]), flush=True)
        return metadata["status"]

    failed = False
    # Per-provider pools preserve fixed submission order while bounding load.
    # Credentials/transport failures stop further submissions; already-started calls finish.
    for name in dict.fromkeys(j["provider"] for j in jobs):
        pending_jobs = iter(j for j in jobs if j["provider"] == name)
        blocked = False
        with ThreadPoolExecutor(max_workers=2) as pool:
            pending = set()
            for _ in range(2):
                job = next(pending_jobs, None)
                if job is not None:
                    pending.add(pool.submit(run_job, job))
            while pending:
                completed, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    status = future.result()
                    failed |= status != "valid"
                    blocked |= status in {"http_error", "transport_error", "started"}
                if not blocked:
                    for _ in completed:
                        job = next(pending_jobs, None)
                        if job is not None:
                            pending.add(pool.submit(run_job, job))
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "run"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--provider", choices=("all", "deepseek", "qwen"), default="all")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    args = parser.parse_args()
    api.load_env(args.env_file)
    if args.command == "prepare":
        prepare(args.output)
    elif args.command == "verify":
        plan = verify(args.output)
        print("Verified %d planned sessions; no model calls" % len(plan["jobs"]))
    else:
        return run(args.output, args.provider)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError) as exc:
        print("Error: " + str(exc), file=sys.stderr)
        sys.exit(2)
