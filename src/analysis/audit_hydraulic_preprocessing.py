"""Independent raw-to-prompt audit, without reading predictions or scoring.

This audit deliberately does not import preprocessing or experiment generators.
It opens only explicitly named source, selection, metadata and inference files.
The protocol and all frozen study inputs are read-only. The sole output is an
audit JSON outside the frozen study directory.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "results/hydraulic_validation_v1"
DATA = ROOT / "data/uci_hydraulic_processed"
OUTPUT = ROOT / "results/hydraulic_independent_preprocessing_audit.json"
CHANNELS = [f"PS{i}" for i in range(1, 7)] + ["EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1"]
KINDS = [("pressure", "bar", 6000)] * 6 + [("motor power", "W", 6000)] + [("volume flow", "L/min", 600)] * 2 + [("temperature", "deg C", 60)] * 4 + [("vibration", "mm/s", 60)]
DESCRIPTORS = [f"{c} {quantity} {stat} ({unit})" for c, (quantity, unit, _) in zip(CHANNELS, KINDS) for stat in ("mean", "SD")]
FEATURES = [f"{c}_{stat}" for c in CHANNELS for stat in ("mean", "std_population")]
ALLOWED_PROFILE = ({3, 20, 100}, {73, 80, 90, 100}, {0, 1, 2}, {90, 100, 115, 130}, {0, 1})


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def profile_blocks(profile: list[list[int]]) -> list[dict]:
    result = []
    begin = 0
    for end in range(1, len(profile) + 1):
        if end == len(profile) or profile[end][:4] != profile[begin][:4]:
            result.append({"id": len(result) + 1, "start": begin + 1, "end": end,
                           "condition": profile[begin][:4],
                           "stable": [i + 1 for i in range(begin, end) if profile[i][4] == 0]})
            begin = end
    return result


def run_audit() -> dict:
    protocol_file = STUDY / "protocol.json"
    protocol_hash_before = digest(protocol_file)
    protocol = load(protocol_file)
    source = ROOT / "data/uci_hydraulic_raw/source.zip"
    source_hash = digest(source)
    meta = load(DATA / "metadata.json")
    require(source_hash == meta["source_zip_sha256"], "Raw source ZIP hash mismatch")
    selected_path = STUDY / "answers/records.json"
    require(digest(selected_path) == protocol["answers_sha256"], "Selection file hash mismatch")
    panels = load(selected_path)["panels"]
    require(set(panels) == {"focal", "a", "b"}, "Unexpected panels")
    records = [r for panel in panels.values() for r in panel]
    ids = [r["id"] for r in records]
    cycles = [r["cycle_1based"] for r in records]
    block_ids = [r["block_id"] for r in records]
    require(len(records) == len(set(ids)) == len(set(cycles)) == len(set(block_ids)) == 54,
            "Expected 54 distinct IDs, cycles and blocks")
    panel_ids = {name: [r["id"] for r in rows] for name, rows in panels.items()}
    require(load(STUDY / "inference/panels.json") == panel_ids, "Inference panel IDs disagree")
    by_id = {r["id"]: r for r in records}
    raw_summary = {cycle: [] for cycle in cycles}
    raw_selected_line_hashes = {}
    raw_file_hashes = {}
    with ZipFile(source) as archive:
        entries = {Path(name).name: name for name in archive.namelist() if not name.endswith("/")}
        profile_bytes = archive.read(entries["profile.txt"])
        profile = [[int(x) for x in line.split()] for line in profile_bytes.splitlines() if line.strip()]
        require(len(profile) == 2205 and all(len(row) == 5 for row in profile), "Raw profile shape mismatch")
        require(all(value in allowed for row in profile for value, allowed in zip(row, ALLOWED_PROFILE)),
                "Raw profile invalid labels")
        require(profile == load(DATA / "profile.json")["rows"], "Processed profile differs from raw")
        blocks = profile_blocks(profile)
        expected_strata = {(p, c) for p in (0, 1, 2) for c in (3, 20, 100)}
        all_counts = Counter()
        panel_counts = {}
        nuisance_by_panel = {}
        for name, rows in panels.items():
            counts = Counter()
            nuisances = {}
            require(len(rows) == 18, f"Wrong size for {name}")
            for row in rows:
                cycle = row["cycle_1based"]
                actual_profile = profile[cycle - 1]
                require(row["profile"] == actual_profile, f"Profile mismatch for {row['id']}")
                b = blocks[row["block_id"] - 1]
                require(row["block_start"] == b["start"] and row["block_end"] == b["end"],
                        "Stored block boundaries mismatch")
                require(len(b["stable"]) >= 5 and cycle == b["stable"][len(b["stable"]) // 2],
                        "Cycle is not the frozen upper-middle stable selection")
                pump, cooler = actual_profile[2], actual_profile[0]
                counts[(pump, cooler)] += 1
                all_counts[(pump, cooler)] += 1
                nuisances.setdefault((pump, cooler), set()).add((actual_profile[1], actual_profile[3]))
            require(set(counts) == expected_strata and set(counts.values()) == {2}, f"Unbalanced panel {name}")
            common = next(iter(nuisances.values()))
            require(len(common) == 2 and all(pairs == common for pairs in nuisances.values()),
                    f"Nuisance settings differ by stratum in {name}")
            panel_counts[name] = {f"pump={p},cooler={c}": n for (p, c), n in sorted(counts.items())}
            nuisance_by_panel[name] = sorted([list(pair) for pair in common])
        require(set(all_counts.values()) == {6}, "Expected six selections in each of nine strata")
        require(len({tuple(pair) for pairs in nuisance_by_panel.values() for pair in pairs}) == 6,
                "Nuisance settings overlap across panels")
        wanted = set(cycles)
        for channel, (_, _, width) in zip(CHANNELS, KINDS):
            member = entries[f"{channel}.txt"]
            whole_hash = hashlib.sha256()
            selected_hashes = {}
            n_lines = 0
            with archive.open(member) as stream:
                for line_number, line in enumerate(stream, start=1):
                    n_lines = line_number
                    whole_hash.update(line)
                    if line_number not in wanted:
                        continue
                    values = [float(token) for token in line.split()]
                    require(len(values) == width and all(math.isfinite(v) for v in values),
                            f"Raw selected row invalid: {channel}, {line_number}")
                    mean = math.fsum(values) / len(values)
                    std = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / len(values))
                    raw_summary[line_number].extend((mean, std))
                    selected_hashes[str(line_number)] = hashlib.sha256(line).hexdigest()
            require(n_lines == 2205 and len(selected_hashes) == 54, f"Wrong row count: {channel}")
            require(whole_hash.hexdigest() == meta["members"][member]["sha256"], f"Member hash mismatch: {channel}")
            raw_file_hashes[channel] = whole_hash.hexdigest()
            raw_selected_line_hashes[channel] = selected_hashes
    require(len(blocks) == 194, "Unexpected independently computed joint block count")
    data = np.load(DATA / "measurement_summary.npz", allow_pickle=False)
    require(set(data.files) == {"X", "mean", "std_population", "channel_names", "feature_names"}, "Unexpected NPZ keys")
    require(data["X"].shape == (2205, 28), "Wrong summary shape")
    require(data["channel_names"].tolist() == CHANNELS and data["feature_names"].tolist() == FEATURES,
            "Feature/channel schema mismatch")
    max_abs_diff = 0.0
    decimal_matches = 0
    rounding_boundary_differences = []
    max_prompt_quantization_error = 0.0
    canonical_fields = {}
    for row in records:
        cycle = row["cycle_1based"]
        independently_computed = raw_summary[cycle]
        require(len(independently_computed) == 28, "Incomplete raw summary")
        stored = data["X"][cycle - 1].tolist()
        for feature, computed, value in zip(FEATURES, independently_computed, stored):
            max_abs_diff = max(max_abs_diff, abs(computed - value))
            require(math.isclose(computed, value, rel_tol=1e-12, abs_tol=1e-10), "Raw summary differs from stored summary")
            serialized = f"{value:.6f}"
            error = abs(float(serialized) - computed)
            max_prompt_quantization_error = max(max_prompt_quantization_error, error)
            require(error <= 0.0000005 + 1e-10, "Prompt value exceeds six-decimal quantization tolerance")
            if f"{computed:.6f}" == serialized:
                decimal_matches += 1
            else:
                rounding_boundary_differences.append({
                    "cycle_1based": cycle, "id": row["id"], "feature": feature,
                    "math_fsum_summary": computed, "numpy_summary": value,
                    "math_fsum_formatted_six_decimals": f"{computed:.6f}",
                    "frozen_numpy_formatted_six_decimals": serialized,
                    "absolute_summary_difference": abs(computed - value),
                    "absolute_prompt_quantization_error": error,
                })
        require(np.array_equal(data["mean"][cycle - 1], data["X"][cycle - 1, ::2]), "Mean matrix differs from X")
        require(np.array_equal(data["std_population"][cycle - 1], data["X"][cycle - 1, 1::2]), "SD matrix differs from X")
        # Exact serializer validation follows the frozen NumPy values. Their
        # independent numerical provenance was checked above; tiny float-order
        # differences at a decimal half-step are preserved explicitly in audit.
        canonical_fields[row["id"]] = [f"{descriptor}={value:.6f}" for descriptor, value in zip(DESCRIPTORS, stored)]
    batches = load(STUDY / "inference/batches.json")["groups"]
    schedule = load(STUDY / "schedule.json")["blocks"]
    planned = [name for block in schedule for name in block]
    require(len(planned) == len(set(planned)) == len(batches) == 63 and set(planned) == set(batches), "Schedule/batch mismatch")
    require(set(batches) == set(protocol["prompts_sha256"]), "Protocol prompt list mismatch")
    prompt_bytes = {}
    slot_count = 0
    numeric_fields_checked = 0
    single_ids = []
    condition_repeats = Counter()
    for name, batch in batches.items():
        path = STUDY / "inference/prompts" / f"{name}.txt"
        payload = path.read_bytes()
        require(hashlib.sha256(payload).hexdigest() == protocol["prompts_sha256"][name], f"Frozen prompt hash mismatch: {name}")
        prompt_bytes[name] = payload
        text = payload.decode("utf-8")
        header, body = text.split("\n\n", 1)
        parsed = json.loads(body)
        require(set(parsed) == {"valid_labels", "samples"}, "Unexpected prompt JSON keys")
        require(parsed["samples"] == batch["samples"] and parsed["valid_labels"] == batch["valid_labels"], "Prompt and batches disagree")
        require(set(parsed["valid_labels"]) == {"No_leakage", "Weak_leakage", "Severe_leakage"}, "Invalid task vocabulary")
        condition = batch["condition"]
        condition_repeats[condition] += 1
        if condition in ("clean", "clean_repeat", "pair_reverse", "descriptor_rotation"):
            expected_ids = panel_ids["focal"]
        elif condition in ("a_front", "a_front_repeat"):
            expected_ids = panel_ids["focal"] + panel_ids["a"]
        elif condition == "a_back":
            expected_ids = panel_ids["a"] + panel_ids["focal"]
        elif condition == "b_front":
            expected_ids = panel_ids["focal"] + panel_ids["b"]
        elif condition == "b_back":
            expected_ids = panel_ids["b"] + panel_ids["focal"]
        elif condition == "singleton":
            require(len(batch["samples"]) == 1, "Wrong singleton size")
            expected_ids = [batch["samples"][0]["id"]]
            require(expected_ids[0] in panel_ids["focal"], "Non-focal singleton")
            single_ids.extend(expected_ids)
        else:
            raise ValueError(f"Unexpected condition {condition}")
        require([s["id"] for s in parsed["samples"]] == expected_ids, f"Sample membership/order differs: {name}")
        for sample in parsed["samples"]:
            require(set(sample) == {"id", "prompt"}, "Per-cycle metadata/target keys leaked into prompt")
            actual = sample["prompt"].split("; ")
            expected = canonical_fields[sample["id"]]
            if condition == "pair_reverse":
                expected = list(reversed(expected))
            elif condition == "descriptor_rotation":
                vals = [part.rsplit("=", 1)[1] for part in expected]
                expected = [f"{DESCRIPTORS[(i + 2) % 28]}={val}" for i, val in enumerate(vals)]
            require(actual == expected, f"Raw-to-prompt field mismatch: {name}/{sample['id']}")
            require(len(actual) == 28, "Wrong scalar count")
            numeric_fields_checked += len(actual)
        slot_count += len(parsed["samples"])
    require(slot_count == protocol["slots_scheduled"] == 1278, "Slot count differs")
    require(len(single_ids) == 18 and set(single_ids) == set(panel_ids["focal"]), "Singleton coverage differs")
    for repeat in range(1, 6):
        require(prompt_bytes[f"clean_r{repeat}"] == prompt_bytes[f"clean_repeat_r{repeat}"] == prompt_bytes["clean_r1"],
                "Clean duplicate or repeat bytes differ")
        require(prompt_bytes[f"a_front_r{repeat}"] == prompt_bytes[f"a_front_repeat_r{repeat}"] == prompt_bytes["a_front_r1"],
                "A-front duplicate or repeat bytes differ")
    for condition in ("pair_reverse", "descriptor_rotation", "a_back", "b_front", "b_back"):
        require(all(prompt_bytes[f"{condition}_r{repeat}"] == prompt_bytes[f"{condition}_r1"] for repeat in range(1, 6)),
                f"Repeated payload changed for {condition}")
    for relative, frozen_hash in protocol["data_sha256"].items():
        require(digest(ROOT / relative) == frozen_hash, f"Frozen data changed: {relative}")
    require(digest(STUDY / "inference/batches.json") == protocol["input_sha256"], "Batches frozen hash mismatch")
    require(digest(STUDY / "schedule.json") == protocol["schedule_sha256"], "Schedule frozen hash mismatch")
    require(digest(protocol_file) == protocol_hash_before, "Protocol changed during audit")
    return {
        "audit_schema": "hydraulic-independent-preprocessing-v1",
        "status": "PASS_WITH_ROUNDING_BOUNDARY_NOTE" if rounding_boundary_differences else "PASS",
        "performed_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Raw source rows to numerical summary and all 63 frozen input prompts; no model responses opened, no scoring or inference",
        "independence_method": "Separate implementation using Python float parsing, math.fsum and math.sqrt; no imports from frozen preprocessing/generator modules",
        "source_zip_sha256": source_hash,
        "protocol_sha256": protocol_hash_before,
        "audit_code_sha256": digest(Path(__file__)),
        "n_selected_cycles": 54,
        "n_unique_joint_condition_blocks": 54,
        "n_joint_blocks_reconstructed_from_raw_profile": len(blocks),
        "n_physical_channels": 14,
        "n_features_per_cycle": 28,
        "n_independent_summary_values_checked": 54 * 28,
        "max_absolute_difference_independent_vs_numpy_summary": max_abs_diff,
        "comparison_tolerance": {"relative": 1e-12, "absolute": 1e-10},
        "n_independent_six_decimal_strings_exact": decimal_matches,
        "n_float_algorithm_rounding_boundary_differences": len(rounding_boundary_differences),
        "rounding_boundary_differences": rounding_boundary_differences,
        "rounding_note": "Independent compensated summation and frozen NumPy reductions may fall on opposite sides of a six-decimal half-step. All such differences are reported, no input was changed, and every frozen prompt exactly reproduces the declared NumPy summary serialization.",
        "max_independent_summary_to_prompt_quantization_error": max_prompt_quantization_error,
        "prompt_quantization_tolerance": 0.0000005 + 1e-10,
        "all_numpy_summary_to_prompt_strings_exact": True,
        "n_prompt_files_checked": len(prompt_bytes),
        "n_input_classification_slots_checked": slot_count,
        "n_prompt_scalar_occurrences_checked": numeric_fields_checked,
        "within_and_across_round_duplicate_bytes_match": True,
        "focal_record_content_preserved_across_all_background_position_conditions": True,
        "pair_reverse_preserves_descriptor_value_association": True,
        "descriptor_rotation_preserves_numeric_strings_and_positions": True,
        "excluded_CE_CP_SE_and_profile_values_not_serialized_as_features": True,
        "exclusion_evidence": "Every per-cycle prompt has only id/prompt keys and exactly the 28 whitelisted physical-channel values, independently reconstructed within declared numerical/quantization tolerances; no additional record metadata",
        "panel_pump_cooler_counts": panel_counts,
        "panel_nuisance_settings_valve_accumulator": nuisance_by_panel,
        "middle_cycle_rule_observed": "stable[len(stable)//2], i.e. later middle item for an even number of stable cycles; verified against the frozen generator's actual selection",
        "singleton_per_focal": 1,
        "raw_physical_member_hashes": raw_file_hashes,
        "raw_selected_line_hashes": raw_selected_line_hashes,
        "limitations": ["Numerical/input audit, not diagnostic accuracy or mitigation-effect evidence",
                        "54 blocks and cycles remain from one rig; no independence or generalization claim",
                        "Excluded feature provenance is established by direct reconstruction, not by demanding their numbers never coincide with legitimate physical values",
                        "This audit was created after protocol freezing and is not retrospectively part of the frozen experiment specification"],
    }


def main() -> None:
    try:
        result = run_audit()
    except Exception as exc:
        result = {"audit_schema": "hydraulic-independent-preprocessing-v1", "status": "FAIL",
                  "performed_at_utc": datetime.now(timezone.utc).isoformat(),
                  "failure": f"{type(exc).__name__}: {exc}",
                  "no_predictions_read_or_scoring_performed": True}
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        raise
    OUTPUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if not k.startswith("raw_")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
