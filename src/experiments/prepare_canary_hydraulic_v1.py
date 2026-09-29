"""Prepare the prospective canary-screening study on unused hydraulic blocks.

No inference and no answer access by the runner. Design fixed before any call:

* 18 new focal cycles: two per pump x cooler stratum, from joint-condition blocks
  not used by any earlier study (two of the remaining shared valve/accumulator
  settings, chosen by hash), middle stable cycle, as in v1.
* 9 canary cycles with known historical labels: one per stratum from the v1
  focal panel, chosen by hash. The model is not told which records are canaries.
* Every request holds the 27 records in one fixed, seeded interleaved order and
  uses the v1 serialization (descriptor=value fields).
* Arms: C canonical field order; H upstream hidden value swap (PS1 and PS2 values
  exchanged in every record, descriptors intact, so the trusted-channel gateway
  accepts it); R intact-pair reversal. The swap and reversal apply to canaries
  and focal records alike, as a serializer fault would.
* Three contiguous execution blocks, each with three byte-identical calls per arm
  in seeded random order: 27 calls, 729 classification slots.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.experiments.interface_contract import canonicalize
from src.experiments.prepare_hydraulic_v1 import CHANNELS, HEADER, LABELS, perturb, prompt_sample, registry

BASE = ROOT / "results/canary_hydraulic_v1"
DATA = ROOT / "data/uci_hydraulic_processed"
V1 = ROOT / "results/hydraulic_validation_v1"
POLICY = ROOT / "results/policy_hydraulic_v1"
SEED = 20260929
ARMS = {"C": "clean", "H": "hidden_value_swap", "R": "pair_reverse"}
CODE = ["src/experiments/prepare_canary_hydraulic_v1.py", "src/experiments/run_hydraulic_v1.py",
        "src/experiments/run_revision_v3.py", "src/experiments/interface_contract.py",
        "src/experiments/prepare_hydraulic_v1.py", "src/analysis/score_canary_hydraulic_v1.py"]


def key(text):
    return hashlib.sha256(f"canary-hydraulic-v1|{SEED}|{text}".encode()).hexdigest()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def used_blocks():
    v1 = json.loads((V1 / "answers/records.json").read_text())["panels"]
    used = {r["block_id"] for panel in v1.values() for r in panel}
    policy = json.loads((POLICY / "source_audit.json").read_text())
    used |= {r["block_id"] for r in policy["selected_records"]}
    return used, v1


def prepare():
    assert not BASE.exists(), f"Refusing to overwrite {BASE}"
    blocks = json.loads((DATA / "blocks.json").read_text())["joint_condition_blocks"]
    profile = np.asarray(json.loads((DATA / "profile.json").read_text())["rows"], dtype=int)
    data = np.load(DATA / "measurement_summary.npz", allow_pickle=False)
    assert list(data["channel_names"]) == CHANNELS and data["X"].shape == (2205, 28)
    used, v1 = used_blocks()
    assert len(used) == 108, len(used)
    strata = {(p, c): [b for b in blocks if b["condition_values"][0] == c and b["condition_values"][2] == p
                       and b["n_stable_cycles_flag_0"] >= 5 and b["block_id"] not in used]
              for p in [0, 1, 2] for c in [3, 20, 100]}
    common = set.intersection(*[{(b["condition_values"][1], b["condition_values"][3]) for b in v} for v in strata.values()])
    assert len(common) >= 2, common
    chosen = sorted(common, key=lambda s: key(f"nuisance|{s[0]}|{s[1]}"))[:2]
    focal, audit = [], []
    for (pump, cooler), cands in strata.items():
        audit.append({"pump": pump, "cooler": cooler, "unused_eligible_blocks": len(cands)})
        for nuisance in chosen:
            b = min([x for x in cands if (x["condition_values"][1], x["condition_values"][3]) == nuisance],
                    key=lambda x: key(f"block|{x['start_cycle_1based']}"))
            a, z = b["start_cycle_1based"] - 1, b["end_cycle_1based_inclusive"]
            stable = np.flatnonzero(profile[a:z, 4] == 0) + a
            index = int(stable[len(stable) // 2])
            focal.append({"cycle_1based": index + 1, "block_id": b["block_id"], "profile": profile[index].tolist(),
                          "true_label": LABELS[pump], "role": "focal"})
    canary = []
    for pump in [0, 1, 2]:
        for cooler in [3, 20, 100]:
            pool = [r for r in v1["focal"] if r["profile"][2] == pump and r["profile"][0] == cooler]
            pick = min(pool, key=lambda r: key(f"canary|{r['cycle_1based']}"))
            canary.append({"cycle_1based": pick["cycle_1based"], "block_id": pick["block_id"],
                           "profile": pick["profile"], "true_label": LABELS[pump], "role": "canary"})
    records = focal + canary
    assert len(focal) == 18 and len(canary) == 9
    assert not {r["block_id"] for r in focal} & used
    for r in records:
        r["id"] = key(f"cycle|{r['cycle_1based']}")[:8]
    assert len({r["id"] for r in records}) == 27
    order = records[:]
    random.Random(SEED).shuffle(order)
    reg = registry()
    canonical = {}
    for r in order:
        values = data["X"][r["cycle_1based"] - 1]
        canonical[r["id"]] = canonicalize({"id": r["id"], "readings": [
            dict(field, value=f"{float(v):.6f}") for field, v in zip(reg, values)]}, reg)
    groups, prompts, schedule = {}, {}, []
    rng = random.Random(SEED)
    for block in range(1, 4):
        jobs = []
        for arm, mode in ARMS.items():
            for copy in range(1, 4):
                name = f"{arm}_b{block}_c{copy}"
                samples = [prompt_sample(perturb(canonical[r["id"]], mode)) for r in order]
                groups[name] = {"dataset": "hydraulic", "condition": arm, "repeat": block,
                                "samples": samples, "valid_labels": LABELS}
                prompts[name] = HEADER + "\n\n" + json.dumps({"valid_labels": LABELS, "samples": samples},
                                                             ensure_ascii=False, separators=(",", ":")) + "\n"
                jobs.append(name)
        rng.shuffle(jobs)
        schedule.append(jobs)
    for arm in ARMS:
        assert len({prompts[n] for n in groups if groups[n]["condition"] == arm}) == 1
    write(BASE / "answers/records.json", {"records": [{k: r[k] for k in ["id", "role", "true_label", "cycle_1based",
                                                                          "block_id", "profile"]} for r in order]})
    write(BASE / "inference/batches.json", {"groups": groups})
    for name, payload in prompts.items():
        path = BASE / "inference/prompts" / f"{name}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload)
    write(BASE / "schedule.json", {"blocks": schedule})
    write(BASE / "source_audit.json", {"strata": audit, "chosen_nuisance_settings": chosen,
                                       "excluded_blocks": sorted(used),
                                       "focal_blocks": sorted(r["block_id"] for r in focal),
                                       "canary_source": "v1 focal panel (historical, labelled)",
                                       "one_rig": True})
    protocol = {
        "schema": "canary-hydraulic-v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "design": ("Prospective test of known-answer (canary) screening, designed after the retrospective "
                   "error-structure analysis of earlier studies and frozen before any call. 18 new focal cycles "
                   "from unused blocks and 9 labelled canary cycles share every request; arms C/H/R; three "
                   "execution blocks with three byte-identical calls per arm."),
        "model": "gpt-5.6-sol", "reasoning_effort": "medium",
        "cli_version": subprocess.check_output(["codex", "--version"], text=True).strip(),
        "timeout_seconds": 900, "workers_max": 2, "max_elapsed_seconds": 10800, "max_total_tokens": 1000000,
        "budget_note": "Dispatch threshold on observed input+output tokens; in-flight calls may exceed it.",
        "calls_scheduled": len(groups), "slots_scheduled": sum(len(g["samples"]) for g in groups.values()),
        "seed": SEED, "time_blocks": 3, "primary_repeat": 1, "retries_per_job": 0,
        "canary_threshold": "accept a call if at least 8 of 9 canary labels are correct",
        "endpoints": ["E1 Spearman canary vs focal accuracy over valid calls, one-sided permutation p",
                      "E2 focal accuracy of accepted vs all and vs rejected calls; coverage",
                      "E3 screen-and-retry per arm x block (first passing call in schedule order, else abstain) "
                      "vs first call and vs three-call majority",
                      "E4 hidden swap H vs canonical C: focal accuracy (within-block call randomization) and "
                      "canary rejection rate"],
        "failure_policy": "One invocation per job; failures and unattempted jobs retained; no replacement calls.",
        "inference_access": "Fresh empty-directory CLI session; tools, plugins and project context disabled.",
        "answers_sha256": digest(BASE / "answers/records.json"),
        "input_sha256": digest(BASE / "inference/batches.json"),
        "schedule_sha256": digest(BASE / "schedule.json"),
        "prompts_sha256": {n: digest(BASE / "inference/prompts" / f"{n}.txt") for n in groups},
        "code_sha256": {n: digest(ROOT / n) for n in CODE},
    }
    write(BASE / "protocol.json", protocol)
    (BASE / "protocol.sha256").write_text(digest(BASE / "protocol.json") + "\n")
    print(json.dumps({k: protocol[k] for k in ["calls_scheduled", "slots_scheduled", "created_at"]}))


if __name__ == "__main__":
    prepare()
