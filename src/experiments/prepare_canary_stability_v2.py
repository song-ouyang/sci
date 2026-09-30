"""Prepare one bounded new-source/panel/later-period stability session, no inference.

All selection uses annotations and a fixed hash, not predictions. New focal
cycles share nine blocks, so they are not eighteen independent source units.
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

BASE = ROOT / "results/canary_stability_v2"
DATA = ROOT / "data/uci_hydraulic_processed"
SEED = 20260930
ARMS = {"C": "clean", "R": "pair_reverse"}
CODE = ["src/experiments/prepare_canary_stability_v2.py", "src/experiments/run_hydraulic_v1.py",
        "src/experiments/run_revision_v3.py", "src/experiments/interface_contract.py",
        "src/experiments/prepare_hydraulic_v1.py", "src/analysis/canary_controls_v2.py",
        "tests/test_canary_controls_v2.py", "tests/test_canary_stability_v2.py"]
SOURCES = ["results/hydraulic_validation_v1/answers/records.json",
           "results/policy_hydraulic_v1/source_audit.json", "results/canary_hydraulic_v1/answers/records.json"]

def key(text):
    return hashlib.sha256(f"canary-stability-v2|{SEED}|{text}".encode()).hexdigest()

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False); f.write("\n")

def old_records():
    v1 = read(ROOT / SOURCES[0])["panels"]
    return [r for panel in v1.values() for r in panel] + read(ROOT / SOURCES[1])["selected_records"] + read(ROOT / SOURCES[2])["records"]

def select(blocks, profile, past):
    used_blocks = {r["block_id"] for r in past}
    used_cycles = {r["cycle_1based"] for r in past}
    strata = {(p,c): [b for b in blocks if b["condition_values"][0] == c and b["condition_values"][2] == p
                     and b["n_stable_cycles_flag_0"] >= 5 and b["block_id"] not in used_blocks]
              for p in (0,1,2) for c in (3,20,100)}
    common = set.intersection(*[{(b["condition_values"][1], b["condition_values"][3]) for b in pool} for pool in strata.values()])
    if len(common) < 2:
        raise ValueError("Fewer than two unused common nuisance settings")
    chosen = sorted(common, key=lambda s: key(f"nuisance|{s[0]}|{s[1]}"))[:2]
    records = []
    for (pump,cooler), pool in strata.items():
        for role, nuisance in zip(("focal","canary"), chosen):
            b = min((b for b in pool if (b["condition_values"][1],b["condition_values"][3]) == nuisance),
                    key=lambda b: key(f"block|{b['start_cycle_1based']}"))
            a,z = b["start_cycle_1based"]-1,b["end_cycle_1based_inclusive"]
            stable = np.flatnonzero(profile[a:z,4] == 0) + a
            positions = [len(stable)//3, 2*len(stable)//3] if role == "focal" else [len(stable)//2]
            for position in positions:
                index = int(stable[position])
                records.append({"id": key(f"cycle|{index+1}")[:8], "role":role,"true_label":LABELS[pump],
                                "cycle_1based":index+1,"block_id":b["block_id"],"profile":profile[index].tolist()})
    assert len(records) == len({r["id"] for r in records}) == len({r["cycle_1based"] for r in records}) == 27
    assert not {r["block_id"] for r in records} & used_blocks
    assert not {r["cycle_1based"] for r in records} & used_cycles
    assert len({r["block_id"] for r in records if r["role"] == "focal"}) == 9
    assert len({r["block_id"] for r in records if r["role"] == "canary"}) == 9
    return records, {"excluded_blocks":sorted(used_blocks),"excluded_cycles":sorted(used_cycles),
                     "chosen_focal_nuisance":chosen[0],"chosen_canary_nuisance":chosen[1],
                     "unused_eligible_blocks":{f"pump{p}_cooler{c}":len(pool) for (p,c),pool in strata.items()}}

def prepare():
    assert not BASE.exists(), f"Refusing to overwrite {BASE}"
    blocks = read(DATA / "blocks.json")["joint_condition_blocks"]
    profile = np.asarray(read(DATA / "profile.json")["rows"],dtype=int)
    data = np.load(DATA / "measurement_summary.npz",allow_pickle=False)
    assert list(data["channel_names"]) == CHANNELS and data["X"].shape == (2205,28)
    records,audit = select(blocks,profile,old_records())
    random.Random(SEED).shuffle(records)
    reg = registry()
    canonical = {r["id"]:canonicalize({"id":r["id"],"readings":[dict(field,value=f"{float(v):.6f}")
                 for field,v in zip(reg,data["X"][r["cycle_1based"]-1])]},reg) for r in records}
    audit.update({"selected_records":records,"focal_block_units":9,"focal_cycles":18,"canary_block_units":9,
                  "canary_cycles":9,"physical_rigs":1,"distinct_serialized_vectors":len({prompt_sample(c)["prompt"] for c in canonical.values()}),
                  "selection_note":"Unused blocks excluded from all three previous hydraulic source selections; completion reuses v1. Two spaced focal cycles per block, one middle canary cycle. No outcomes used."})
    groups,prompts,schedule = {},{},[]
    rng = random.Random(SEED)
    for block in range(1,4):
        jobs = []
        for arm,mode in ARMS.items():
            for copy in range(1,4):
                name = f"{arm}_b{block}_c{copy}"
                samples = [prompt_sample(perturb(canonical[r["id"]],mode)) for r in records]
                groups[name] = {"dataset":"hydraulic","condition":arm,"repeat":block,"samples":samples,"valid_labels":LABELS}
                prompts[name] = HEADER+"\n\n"+json.dumps({"valid_labels":LABELS,"samples":samples},ensure_ascii=False,separators=(",",":"))+"\n"
                jobs.append(name)
        rng.shuffle(jobs); schedule.append(jobs)
    assert len(groups) == 18
    assert all(len({prompts[n] for n in groups if groups[n]["condition"] == arm}) == 1 for arm in ARMS)
    write(BASE / "answers/records.json",{"records":records})
    write(BASE / "source_audit.json",audit)
    write(BASE / "inference/batches.json",{"groups":groups})
    write(BASE / "schedule.json",{"blocks":schedule})
    for name,prompt in prompts.items():
        path = BASE / "inference/prompts" / f"{name}.txt"
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(prompt)
    protocol = {
        "schema":"canary-stability-v2","created_at":datetime.now(timezone.utc).isoformat(),
        "design":"One prospectively fixed later-period follow-up after v1 and its post hoc collapse control. C/R x3 chronological blocks x3 byte-identical calls; 18 new focal cycles from nine unused blocks plus nine new canaries from nine other unused blocks on the same rig.",
        "model":"gpt-5.6-sol","reasoning_effort":"medium","cli_version":subprocess.check_output(["codex","--version"],text=True).strip(),
        "previous_cli_version":"codex-cli 0.155.0-alpha.9.2","timeout_seconds":900,"workers_max":2,
        "max_elapsed_seconds":3600,"max_total_tokens":600000,"calls_scheduled":18,"slots_scheduled":486,
        "budget_note":"Existing subscription CLI only, no paid API or purchase. 600000 observed input+output tokens is a dispatch threshold; at most two in-flight calls can exceed it. No new dispatch with less than one full 900-second timeout remaining. 18 jobs maximum, one invocation per job.",
        "resource_estimate":{"basis":"603721 tokens / 27 frozen v1 calls x18","estimated_tokens":402481,"unknown":"Remaining subscription capacity is not assumed; quota failure retained without retry."},
        "seed":SEED,"time_blocks":3,"retries_per_job":0,
        "canary_threshold":"at least 8 correct of 9; first passing call within fixed arm/block schedule, otherwise withhold all 18 focal decisions",
        "collapse_threshold":"Reject if modal share of all 27 returned labels >=9/10 (25 labels). No truth or role access.",
        "collapse_sensitivity_thresholds":["17/20","19/20","1"],
        "vote_rule":"Three calls per arm/block; modal focal label; all-label ties take first call. Separately report tie-abstention sensitivity. Require three valid calls.",
        "endpoints":["All planned six arm/block windows: Single, Known_answer, Collapse_90, Vote_first_tie and Vote_abstain_tie; AC/AE/withheld, accepted-case error, coverage, correct yield, prescribed calls, observed tokens and summed client seconds.",
                     "All valid-call known-answer x collapse pass cross-tabs and focal errors in each cell; all four fixed collapse thresholds.",
                     "Report full results by arm and chronological block; do not select positive blocks or tune thresholds. Compare descriptively with v1, restricting old study to C/R for like-arm summaries."],
        "unit_of_analysis":"Calls share 18 focal cycles in nine source blocks, nine canaries in nine different blocks, three time blocks, one physical rig. No independent 108-decision or 324-output sample-size claim.",
        "limitations":["Source panel, canary panel, later service period and CLI version change jointly; cannot separate causal effects.","No new device or field validation; no calibrated population confidence interval.","Fixed full payload: collapse control tests incremental label knowledge at decision stage, not removal of canary records.","All calls collected regardless of gates. Policy savings are offline required-call counts, not experimentally realized adaptive savings."],
        "failure_policy":"No retries, repair or replacement; failures and unattempted jobs retain planned denominators. Stop dispatch on failure, missing usage, quota, wall-clock or token threshold. Invalid call cannot pass a gate. Single invalid first call and incomplete vote withhold affected outputs.",
        "inference_access":"Fresh empty-directory CLI session; inherited audited runner disables project context, tools, plugins, skills and web access. Answer key opened only after response audit and immutable output freeze.",
        "data_sha256":{str(p.relative_to(ROOT)):digest(p) for p in [DATA/"blocks.json",DATA/"profile.json",DATA/"measurement_summary.npz"]},
        "past_selection_sha256":{p:digest(ROOT/p) for p in SOURCES},
        "source_audit_sha256":digest(BASE/"source_audit.json"),"answers_sha256":digest(BASE/"answers/records.json"),
        "input_sha256":digest(BASE/"inference/batches.json"),"schedule_sha256":digest(BASE/"schedule.json"),
        "prompts_sha256":{n:digest(BASE/"inference/prompts"/f"{n}.txt") for n in groups},
        "code_sha256":{n:digest(ROOT/n) for n in CODE}}
    write(BASE/"protocol.json",protocol)
    (BASE/"protocol.sha256").write_text(digest(BASE/"protocol.json")+"\n")
    print(json.dumps({"protocol_sha256":digest(BASE/"protocol.json"),"calls":18,"focal_blocks":9,"canary_blocks":9,"cli_version":protocol["cli_version"]}))

if __name__ == "__main__":
    prepare()
