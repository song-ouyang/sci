"""Build the completion study for the interrupted hydraulic follow-up.

The v1 run stopped after two transport failures in block R3. This versioned
completion re-executes the three unfinished execution blocks (R3, R4, R5) in
full, with byte-identical prompts and the original within-block order, so that
every block again contains contemporaneous controls. The v1 directory is read
only; its partial R3 outputs stay frozen and are analysed separately.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
V1 = ROOT / "results/hydraulic_validation_v1"
BASE = ROOT / "results/hydraulic_validation_v1_completion"
BLOCKS = [2, 3, 4]  # zero-based indices of R3, R4, R5 in the v1 schedule


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def prepare():
    assert not BASE.exists(), f"Refusing to overwrite {BASE}"
    v1 = json.loads((V1 / "protocol.json").read_text(encoding="utf-8"))
    assert digest(V1 / "protocol.json") == (V1 / "protocol.sha256").read_text().strip()
    for name, expected in v1["code_sha256"].items():
        assert digest(ROOT / name) == expected, f"Frozen v1 code changed: {name}"
    schedule = json.loads((V1 / "schedule.json").read_text(encoding="utf-8"))["blocks"]
    groups_v1 = json.loads((V1 / "inference/batches.json").read_text(encoding="utf-8"))["groups"]
    blocks = [schedule[i] for i in BLOCKS]
    names = [n for block in blocks for n in block]
    groups = {n: groups_v1[n] for n in names}
    for name in names:
        src = V1 / "inference/prompts" / f"{name}.txt"
        dst = BASE / "inference/prompts" / f"{name}.txt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        assert digest(dst) == v1["prompts_sha256"][name]
    write(BASE / "inference/batches.json", {"groups": groups})
    write(BASE / "schedule.json", {"blocks": blocks})
    protocol = dict(v1)
    protocol.update({
        "schema": "hydraulic-validation-v1-completion",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "design": ("Versioned completion of the interrupted v1 schedule. Blocks R3-R5 are executed in full "
                   "with byte-identical prompts and unchanged within-block order. v1 R1/R2 plus these three "
                   "blocks form the five-block analysis set; v1 partial R3 outputs remain frozen and are "
                   "reported as a sensitivity check only."),
        "completion_of": {"protocol_sha256": digest(V1 / "protocol.json"),
                          "v1_stop_reason": "authentication_quota_or_transport_failure",
                          "v1_blocks_reexecuted": ["R3", "R4", "R5"]},
        "calls_scheduled": len(groups),
        "slots_scheduled": sum(len(g["samples"]) for g in groups.values()),
        "time_blocks": len(blocks),
        "block_labels": ["R3", "R4", "R5"],
        "input_sha256": digest(BASE / "inference/batches.json"),
        "schedule_sha256": digest(BASE / "schedule.json"),
        "prompts_sha256": {n: v1["prompts_sha256"][n] for n in names},
        "code_sha256": {**v1["code_sha256"],
                        "src/experiments/prepare_hydraulic_v1_completion.py":
                            digest(ROOT / "src/experiments/prepare_hydraulic_v1_completion.py")},
        "scoring_plan": (v1["scoring_plan"].split(" No p-values")[0] +
                         " Statistical tests treat the five execution blocks as the replication unit."),
    })
    write(BASE / "protocol.json", protocol)
    (BASE / "protocol.sha256").write_text(digest(BASE / "protocol.json") + "\n")
    print(json.dumps({k: protocol[k] for k in ["calls_scheduled", "slots_scheduled", "block_labels", "created_at"]}))


if __name__ == "__main__":
    prepare()
