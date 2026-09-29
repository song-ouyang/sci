"""Prepare new hydraulic policy panels locally; this module never runs inference.

Only dataset.json may be copied to inference. truth.json and audit.json are
separate author-side artifacts. Preparation uses sampling annotations but never
opens predictions, selects on model performance, or modifies an earlier study.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/uci_hydraulic_processed"
PRIOR = ROOT / "results/hydraulic_validation_v1"
OUT = ROOT / "data/policy_hydraulic_v1"
SEED = 20260928
DOMAIN = "policy-hydraulic-v1"
LABELS = ["No_leakage", "Weak_leakage", "Severe_leakage"]
CHANNELS = [f"PS{i}" for i in range(1, 7)] + [
    "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1"
]
KINDS = ([('pressure', 'bar')] * 6 + [('motor power', 'W')]
         + [('volume flow', 'L/min')] * 2 + [('temperature', 'deg C')] * 4
         + [('vibration', 'mm/s')])
TASK_TEXT = (
    "Classify internal pump leakage in each hydraulic test-rig observation. "
    "No_leakage means no internal leakage; Weak_leakage means weak internal "
    "leakage; Severe_leakage means severe internal leakage. Cooler, valve and "
    "accumulator conditions may also vary. Each feature is a mean or population "
    "standard deviation over one complete 60-second operating cycle (SD denotes "
    "population standard deviation). No labelled examples are provided. Use "
    "the supplied measurements. Do not assume a class distribution or derive "
    "labels from IDs, order or other observations."
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    path = Path(path)
    need(path.is_file() and not path.is_symlink(), f"Missing/symlinked source: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    digest(path)
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rank(text):
    return hashlib.sha256(f"{DOMAIN}|{SEED}|{text}".encode()).hexdigest()


def dictionary():
    return [
        {"name": f"{channel} {kind} {stat}", "unit": unit}
        for channel, (kind, unit) in zip(CHANNELS, KINDS)
        for stat in ["mean", "SD"]
    ]


def prior_exclusions(prior, profile, blocks):
    """Read only the old protocol/answer-key provenance, never old predictions."""
    prior = Path(prior)
    protocol = read_json(prior / "protocol.json")
    need(digest(prior / "protocol.json") == (prior / "protocol.sha256").read_text().strip(),
         "Prior protocol hash mismatch")
    need(digest(prior / "answers/records.json") == protocol["answers_sha256"],
         "Prior answer/provenance hash mismatch")
    panels = read_json(prior / "answers/records.json")["panels"]
    need(set(panels) == {"focal", "a", "b"}, "Unexpected prior panel inventory")
    rows = [row for panel in panels.values() for row in panel]
    need(len(rows) == 54, "Expected all 54 old records, including unattempted ones")
    by_id = {block["block_id"]: block for block in blocks}
    excluded = set()
    for row in rows:
        index = row["cycle_1based"] - 1
        block = by_id[row["block_id"]]
        need(0 <= index < len(profile), "Old cycle outside current profile")
        need(profile[index].tolist() == row["profile"], "Prior/current profile mismatch")
        need(block["start_cycle_1based"] <= index + 1 <= block["end_cycle_1based_inclusive"],
             "Old cycle outside declared joint-condition block")
        excluded.add(row["block_id"])
    need(len(excluded) == 54, "Expected 54 distinct prior joint-condition blocks")
    return excluded


def select_panels(blocks, profile, excluded_blocks):
    """Hash-select six unused shared nuisance settings, without measurement values."""
    profile = np.asarray(profile)
    need(profile.ndim == 2 and profile.shape[1] == 5, "Expected five profile columns")
    need(np.isfinite(profile).all() and np.equal(profile, np.round(profile)).all(),
         "Profile must contain finite integer annotations")
    need(len({b["block_id"] for b in blocks}) == len(blocks), "Duplicate block ID")
    excluded_blocks = set(excluded_blocks)
    strata = {}
    for pump in [0, 1, 2]:
        for cooler in [3, 20, 100]:
            strata[pump, cooler] = [
                b for b in blocks
                if b["block_id"] not in excluded_blocks
                and b["condition_values"][0] == cooler
                and b["condition_values"][2] == pump
                and b["n_stable_cycles_flag_0"] >= 5
            ]
    common = set.intersection(*[
        {(b["condition_values"][1], b["condition_values"][3]) for b in group}
        for group in strata.values()
    ])
    need(len(common) >= 6, "Fewer than six unused nuisance settings shared by all strata")
    chosen = sorted(common, key=lambda s: rank(f"nuisance|{s[0]}|{s[1]}"))[:6]
    panels = [{"panel_id": f"hydraulic_p{i+1:02d}", "focal": [], "background": []}
              for i in range(3)]
    selected_blocks = set()
    for setting_index, setting in enumerate(chosen):
        panel = panels[setting_index // 2]
        role = "focal" if setting_index % 2 == 0 else "background"
        for (pump, cooler), candidates in sorted(strata.items()):
            compatible = [b for b in candidates
                          if (b["condition_values"][1], b["condition_values"][3]) == setting]
            block = min(compatible, key=lambda b: rank(f"block|{b['block_id']}"))
            begin, end = block["start_cycle_1based"] - 1, block["end_cycle_1based_inclusive"]
            need(0 <= begin < end <= len(profile), "Invalid joint-block cycle range")
            need(np.all(profile[begin:end, :4] == np.asarray(block["condition_values"])),
                 "Block definition disagrees with cycle profile")
            stable = np.flatnonzero(profile[begin:end, 4] == 0) + begin
            need(len(stable) == block["n_stable_cycles_flag_0"] and len(stable) >= 5,
                 "Stable-cycle count mismatch")
            index = int(stable[len(stable) // 2])
            need(block["block_id"] not in selected_blocks, "Joint block selected twice")
            selected_blocks.add(block["block_id"])
            panel[role].append({
                "id": rank(f"cycle|{index+1}")[:8],
                "cycle_1based": index + 1,
                "block_id": block["block_id"],
                "block_start": begin + 1,
                "block_end": end,
                "profile": profile[index].astype(int).tolist(),
                "true_label": LABELS[pump],
            })
    for panel in panels:
        for role in ["focal", "background"]:
            panel[role].sort(key=lambda row: row["id"])
            need(len(panel[role]) == 9, "Every role must contain nine strata")
    audit = {
        "unused_eligible_blocks_per_stratum": [
            {"pump": p, "cooler": c, "n": len(v)} for (p, c), v in sorted(strata.items())
        ],
        "n_unused_common_nuisance_settings": len(common),
        "chosen_nuisance_settings_in_panel_role_order": [list(s) for s in chosen],
        "excluded_joint_condition_blocks": sorted(excluded_blocks),
        "selected_records": [dict(row, panel_id=p["panel_id"], role=role)
                             for p in panels for role in ["focal", "background"] for row in p[role]],
    }
    return panels, audit


def validate_dataset(dataset):
    """Strict allowlist excludes labels/provenance from record payloads."""
    need(set(dataset) == {"task_text", "allowed_labels", "dictionary", "panels"},
         "Unexpected dataset fields")
    need(dataset["task_text"] == TASK_TEXT, "Task description changed")
    need(dataset["allowed_labels"] == LABELS, "Unexpected label vocabulary")
    need(dataset["dictionary"] == dictionary(), "Physical feature dictionary changed")
    need(len(dataset["panels"]) == 3, "Expected three panels")
    all_ids = []
    for number, panel in enumerate(dataset["panels"], start=1):
        need(set(panel) == {"panel_id", "focal_ids", "background_ids", "records"},
             "Unexpected panel fields")
        need(panel["panel_id"] == f"hydraulic_p{number:02d}", "Unexpected panel ID")
        focal, background = panel["focal_ids"], panel["background_ids"]
        need(len(focal) == len(background) == 9, "Expected nine records per role")
        need(focal == sorted(focal) and background == sorted(background), "Role ID order changed")
        need(len(set(focal + background)) == 18, "Duplicate panel ID")
        records = panel["records"]
        need(len(records) == 18, "Expected eighteen panel records")
        need([r["id"] for r in records] == sorted(focal + background), "Record inventory/order mismatch")
        for record in records:
            need(set(record) == {"id", "values"}, "Private/unknown record field")
            need(re.fullmatch(r"[0-9a-f]{8}", record["id"]) is not None, "Nonanonymous record ID")
            values = record["values"]
            need(len(values) == 28, "Expected 28 physical features")
            need(all(isinstance(v, str) and re.fullmatch(r"-?\d+\.\d{6}", v) for v in values),
                 "Values must be finite six-decimal strings")
            need(all(float(values[i]) >= 0 for i in range(1, 28, 2)), "Negative standard deviation")
        all_ids.extend(focal + background)
    need(len(set(all_ids)) == 54, "Anonymous ID collision across panels")


def build_dataset(blocks, profile, measurements, excluded_blocks):
    measurements = np.asarray(measurements)
    need(measurements.shape == (len(profile), 28), "Measurement/profile dimensions differ")
    need(np.isfinite(measurements).all(), "Nonfinite measurements")
    need(np.all(measurements[:, 1::2] >= 0), "Negative source standard deviation")
    selected, audit = select_panels(blocks, profile, excluded_blocks)
    dataset = {"task_text": TASK_TEXT, "allowed_labels": LABELS.copy(),
               "dictionary": dictionary(), "panels": []}
    truth = {"labels": {}}
    for panel in selected:
        rows = sorted(panel["focal"] + panel["background"], key=lambda r: r["id"])
        dataset["panels"].append({
            "panel_id": panel["panel_id"],
            "focal_ids": [r["id"] for r in panel["focal"]],
            "background_ids": [r["id"] for r in panel["background"]],
            "records": [{"id": r["id"], "values": [f"{float(v):.6f}" for v in
                         measurements[r["cycle_1based"]-1]]} for r in rows],
        })
        truth["labels"].update({r["id"]: r["true_label"] for r in rows})
    validate_dataset(dataset)
    need(len(truth["labels"]) == 54, "Truth/input ID mismatch")
    return dataset, truth, audit


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def prepare(data=DATA, prior=PRIOR, out=OUT, extra_excluded_blocks=()):
    data, prior, out = Path(data), Path(prior), Path(out)
    need(not out.exists(), "Refusing to overwrite an existing preparation")
    meta = read_json(data / "metadata.json")
    blocks = read_json(data / "blocks.json")["joint_condition_blocks"]
    profile = np.asarray(read_json(data / "profile.json")["rows"], dtype=int)
    need(profile.shape == (2205, 5), "Unexpected hydraulic source size")
    need(meta["measurement_npz_contains_labels"] is False, "Measurement NPZ may contain labels")
    need(meta["included_physical_channels"] == CHANNELS, "Physical channel metadata differs")
    with np.load(data / "measurement_summary.npz", allow_pickle=False) as arrays:
        need(list(arrays["channel_names"]) == CHANNELS, "NPZ physical channel order differs")
        values = arrays["X"].copy()
    excluded = prior_exclusions(prior, profile, blocks)
    old_excluded = set(excluded)
    excluded.update(extra_excluded_blocks)
    need(all(type(x) is int for x in excluded), "Excluded block IDs must be integers")
    need(excluded <= {b["block_id"] for b in blocks}, "Unknown excluded block")
    dataset, truth, audit = build_dataset(blocks, profile, values, excluded)
    source_paths = [data / n for n in ["metadata.json", "blocks.json", "profile.json", "measurement_summary.npz"]]
    source_paths += [prior / "protocol.json", prior / "protocol.sha256", prior / "answers/records.json"]
    audit.update({
        "schema": "policy-hydraulic-v1-preparation",
        "prepared_at": datetime.now(timezone.utc).isoformat(),
        "seed": SEED, "hash_domain": DOMAIN,
        "source_sha256": {str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): digest(p)
                          for p in source_paths},
        "preparation_code_sha256": digest(Path(__file__)),
        "source_archive_sha256": meta["source_zip_sha256"],
        "prior_excluded_blocks": sorted(old_excluded),
        "additional_excluded_blocks": sorted(set(extra_excluded_blocks) - old_excluded),
        "new_focal_cycles": 27, "new_background_cycles": 27,
        "new_joint_condition_blocks": 54, "physical_rigs": 1,
        "selection_policy": "Six hash-ordered unused valve/accumulator settings shared across all nine pump/cooler strata; alternating focal/background assignments across three panels; later-middle stable cycle; anonymous-ID order. No predictions or measurement-value selection.",
        "representation_policy": "Same 14 physical channels and mean/population SD as v1; new shared-once dictionary and six-decimal values are fixed before calls. F/R changes only record order.",
        "model_input_boundary": "dataset.json only. truth.json and audit.json must not be exposed to inference. Record roles and panel_id are runner-side scheduling metadata, not diagnostic features.",
        "policy_indexing": "F1/R1 mean scheduled pair index 1, not earliest completion or first valid output. Same compares F1/F2; View compares F1/R2; both retain only F1's label on agreement. F2/R2 are paired contemporaneous checkers. Strategies are scored elsewhere after freezing outputs.",
        "limits": ["One rig; selected stable cycles; repeated acquisition order can retain dependence",
                   "Annotations used for balanced sampling, never disclosed per record to the model",
                   "These are previously unused blocks, not independent new devices",
                   "No inference, scoring, strategy fitting, or freezing is performed by this preparation"],
    })
    out.mkdir(parents=True)
    write_new(out / "dataset.json", dataset)
    write_new(out / "truth.json", truth)
    audit["dataset_sha256"] = digest(out / "dataset.json")
    audit["truth_sha256"] = digest(out / "truth.json")
    write_new(out / "audit.json", audit)
    return {"output": str(out), "dataset_sha256": audit["dataset_sha256"],
            "panels": 3, "focal_cycles": 27, "background_cycles": 27,
            "joint_condition_blocks": 54, "physical_rigs": 1}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--exclude-block", type=int, action="append", default=[])
    args = parser.parse_args()
    print(json.dumps(prepare(out=args.out, extra_excluded_blocks=args.exclude_block), indent=2))
