"""Audit and summarize the official UCI 447 archive locally, without inference.

All 17 measurement files are validated. Only the 14 physical channels enter the
summary; CE, CP and SE are deliberately excluded. Profile labels remain in a
separate JSON artifact and never appear in the measurement NPZ. This script does
not select a train/test split or infer independence from contiguous blocks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZipFile, is_zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROWS = 2205
PHYSICAL_CHANNELS = (
    *[f"PS{i}" for i in range(1, 7)],
    "EPS1", "FS1", "FS2", "TS1", "TS2", "TS3", "TS4", "VS1",
)
EXCLUDED_CHANNELS = ("CE", "CP", "SE")
PROFILE_COLUMNS = (
    "cooler_condition_percent", "valve_condition_percent", "pump_leakage_level",
    "accumulator_pressure_bar", "stable_flag",
)
ALLOWED_PROFILE = ({3, 20, 100}, {73, 80, 90, 100}, {0, 1, 2}, {90, 100, 115, 130}, {0, 1})


def channel_metadata(channel: str) -> dict[str, Any]:
    if channel.startswith("PS"):
        quantity, unit, rate = "Pressure", "bar", 100
    elif channel == "EPS1":
        quantity, unit, rate = "Motor power", "W", 100
    elif channel.startswith("FS"):
        quantity, unit, rate = "Volume flow", "L/min", 10
    elif channel.startswith("TS"):
        quantity, unit, rate = "Temperature", "degC", 1
    elif channel == "VS1":
        quantity, unit, rate = "Vibration", "mm/s", 1
    elif channel == "CE":
        quantity, unit, rate = "Cooling efficiency (virtual)", "%", 1
    elif channel == "CP":
        quantity, unit, rate = "Cooling power (virtual)", "kW", 1
    elif channel == "SE":
        quantity, unit, rate = "Efficiency factor", "%", 1
    else:
        raise ValueError(f"Unknown channel: {channel}")
    return {"channel": channel, "physical_quantity": quantity, "unit": unit,
            "sampling_rate_hz": rate, "samples_per_cycle": 60 * rate,
            "included_in_summary": channel in PHYSICAL_CHANNELS}


def sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def validate_profile(profile: np.ndarray, expected_rows: int = EXPECTED_ROWS) -> np.ndarray:
    if profile.shape != (expected_rows, 5):
        raise ValueError(f"profile: expected {(expected_rows, 5)}, got {profile.shape}")
    if not np.isfinite(profile).all() or not np.equal(profile, np.round(profile)).all():
        raise ValueError("profile: values must be finite integers")
    values = profile.astype(np.int64)
    for column, allowed in enumerate(ALLOWED_PROFILE):
        observed = set(values[:, column].tolist())
        if not observed <= allowed:
            raise ValueError(f"profile: illegal {PROFILE_COLUMNS[column]}: {observed - allowed}")
    return values


def summarize_measurements(values: np.ndarray, channel: str,
                           expected_rows: int = EXPECTED_ROWS) -> tuple[np.ndarray, np.ndarray]:
    width = channel_metadata(channel)["samples_per_cycle"]
    if values.shape != (expected_rows, width):
        raise ValueError(f"{channel}: expected {(expected_rows, width)}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError(f"{channel}: non-finite measurement")
    mean = np.mean(values, axis=1, dtype=np.float64)
    std = np.std(values, axis=1, dtype=np.float64, ddof=0)
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError(f"{channel}: summary overflow")
    return mean, std


def contiguous_blocks(profile: np.ndarray, columns: tuple[int, ...]) -> list[dict[str, Any]]:
    """Maximal consecutive equal tuples; flags never implicitly define a block."""
    if profile.ndim != 2 or profile.shape[1] != 5 or not len(profile):
        raise ValueError("Expected a nonempty five-column profile")
    if not columns or any(c not in range(4) for c in columns):
        raise ValueError("Block columns must be selected from the four component labels")
    selected = profile[:, columns]
    boundaries = np.flatnonzero(np.any(selected[1:] != selected[:-1], axis=1)) + 1
    edges = [0, *boundaries.tolist(), len(profile)]
    blocks = []
    for idx, (start, end) in enumerate(zip(edges[:-1], edges[1:]), start=1):
        flags = profile[start:end, 4]
        blocks.append({
            "block_id": idx,
            "start_cycle_1based": start + 1,
            "end_cycle_1based_inclusive": end,
            "n_cycles": end - start,
            "condition_values": profile[start, list(columns)].tolist(),
            "n_stable_cycles_flag_0": int(np.count_nonzero(flags == 0)),
            "n_potentially_unstable_cycles_flag_1": int(np.count_nonzero(flags == 1)),
        })
    return blocks


def profile_audit(profile: np.ndarray) -> dict[str, Any]:
    full_blocks = contiguous_blocks(profile, (0, 1, 2, 3))
    cooler_segments = contiguous_blocks(profile, (0,))
    classes = {}
    for cooler in sorted(ALLOWED_PROFILE[0]):
        mask = profile[:, 0] == cooler
        eligible = [b for b in full_blocks if b["condition_values"][0] == cooler]
        classes[str(cooler)] = {
            "n_cycles": int(mask.sum()),
            "n_stable_cycles_flag_0": int(np.count_nonzero(mask & (profile[:, 4] == 0))),
            "n_potentially_unstable_cycles_flag_1": int(np.count_nonzero(mask & (profile[:, 4] == 1))),
            "n_contiguous_joint_condition_blocks": len(eligible),
            "n_joint_blocks_with_any_stable_cycle": sum(b["n_stable_cycles_flag_0"] > 0 for b in eligible),
            "n_contiguous_cooler_segments": sum(b["condition_values"][0] == cooler for b in cooler_segments),
            "n_distinct_joint_conditions": len({tuple(r) for r in profile[mask, :4].tolist()}),
        }
    joint_occurrences = Counter(tuple(b["condition_values"]) for b in full_blocks)
    return {
        "cycle_numbering": "1-based row order; no wall-clock timestamps or device IDs provided",
        "joint_block_definition": "Maximal contiguous identical profile columns 1-4; stable flag ignored for boundaries",
        "cooler_segment_definition": "Maximal contiguous identical profile column 1",
        "dependency_warning": "Blocks and cycles from the same rig are not independent devices or independent runs.",
        "n_cycles": len(profile),
        "n_contiguous_joint_condition_blocks": len(full_blocks),
        "n_contiguous_cooler_segments": len(cooler_segments),
        "n_distinct_joint_conditions": len(joint_occurrences),
        "n_joint_conditions_occurring_in_multiple_blocks": sum(n > 1 for n in joint_occurrences.values()),
        "cooler_class_summary": classes,
        "joint_condition_blocks": full_blocks,
        "cooler_segments": cooler_segments,
    }


def prepare(source: Path, output: Path, overwrite: bool = False) -> dict[str, Any]:
    if not source.is_file() or not is_zipfile(source):
        raise ValueError(f"A complete ZIP archive is required: {source}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists; use --overwrite only for an unfrozen derived directory: {output}")
    with source.open("rb") as stream:
        source_hash = sha256_stream(stream)
    member_metadata = {}
    means, stds = {}, {}
    with ZipFile(source) as archive:
        members = {}
        for info in archive.infolist():
            if info.is_dir():
                continue
            basename = Path(info.filename).name
            if basename in members:
                raise ValueError(f"Duplicate ZIP basename: {basename}")
            members[basename] = info.filename
            with archive.open(info) as stream:
                digest = sha256_stream(stream)
            member_metadata[info.filename] = {"uncompressed_bytes": info.file_size, "sha256": digest}
        required = {f"{c}.txt" for c in (*PHYSICAL_CHANNELS, *EXCLUDED_CHANNELS)} | {"profile.txt"}
        if not required <= members.keys():
            raise ValueError(f"Missing members: {sorted(required - members.keys())}")
        with archive.open(members["profile.txt"]) as stream:
            profile = validate_profile(np.loadtxt(stream, dtype=np.float64, ndmin=2))
        for channel in (*PHYSICAL_CHANNELS, *EXCLUDED_CHANNELS):
            with archive.open(members[f"{channel}.txt"]) as stream:
                values = np.loadtxt(stream, dtype=np.float64, ndmin=2)
            mean, std = summarize_measurements(values, channel)
            member_metadata[members[f"{channel}.txt"]]["validated_shape"] = list(values.shape)
            del values
            if channel in PHYSICAL_CHANNELS:
                means[channel], stds[channel] = mean, std
    audit = profile_audit(profile)
    mean_matrix = np.column_stack([means[c] for c in PHYSICAL_CHANNELS])
    std_matrix = np.column_stack([stds[c] for c in PHYSICAL_CHANNELS])
    summary = np.stack((mean_matrix, std_matrix), axis=2).reshape(EXPECTED_ROWS, -1)
    features = [f"{c}_{stat}" for c in PHYSICAL_CHANNELS for stat in ("mean", "std_population")]
    metadata = {
        "schema_version": 1,
        "source_filename": source.name,
        "source_zip_sha256": source_hash,
        "source_zip_bytes": source.stat().st_size,
        "source_dataset_doi": "10.24432/C5CW21",
        "source_license": "CC BY 4.0 (UCI metadata; attribution required)",
        "n_cycles": EXPECTED_ROWS,
        "cycle_duration_seconds": 60,
        "channels": [channel_metadata(c) for c in (*PHYSICAL_CHANNELS, *EXCLUDED_CHANNELS)],
        "included_physical_channels": list(PHYSICAL_CHANNELS),
        "excluded_channels": list(EXCLUDED_CHANNELS),
        "exclusion_reason": "Pre-specified physical-sensor-only design; virtual/derived CE, CP, SE excluded without asserting proven target leakage.",
        "summary_feature_order": features,
        "summary_definition": "Per-channel full 60-second cycle arithmetic mean and population SD (ddof=0), native sampling; no resampling, fitting, scaling or feature selection.",
        "measurement_npz_contains_labels": False,
        "profile_columns": list(PROFILE_COLUMNS),
        "n_independent_devices_established": 1,
        "independence_note": "The public source describes one test rig; no public per-cycle device or acquisition-session identifier supports a larger independent-device count.",
        "split_status": "No split or sampling performed by this preprocessing script.",
        "members": member_metadata,
        "validation": {"all_17_sensor_files_validated": True, "shape_checks_passed": True,
                       "finite_checks_passed": True, "profile_labels_valid": True},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".hydraulic-prepared-", dir=output.parent))
    try:
        np.savez_compressed(staging / "measurement_summary.npz", X=summary,
                            mean=mean_matrix, std_population=std_matrix,
                            channel_names=np.asarray(PHYSICAL_CHANNELS),
                            feature_names=np.asarray(features))
        artifacts = {
            "profile.json": {"columns": list(PROFILE_COLUMNS), "rows": profile.tolist()},
            "blocks.json": audit,
            "metadata.json": metadata,
        }
        for name, obj in artifacts.items():
            (staging / name).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if output.exists():
            for name in ("measurement_summary.npz", *artifacts):
                (staging / name).replace(output / name)
            staging.rmdir()
        else:
            staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return {"source_zip_sha256": source_hash, "output": str(output),
            "n_cycles": EXPECTED_ROWS, "n_physical_channels": len(PHYSICAL_CHANNELS),
            "n_features": len(features), "cooler_class_summary": audit["cooler_class_summary"],
            "n_contiguous_joint_condition_blocks": audit["n_contiguous_joint_condition_blocks"],
            "n_contiguous_cooler_segments": audit["n_contiguous_cooler_segments"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, default=ROOT / "data/uci_hydraulic_raw/source.zip")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/uci_hydraulic_processed")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    result = prepare(args.source_zip, args.output_dir, args.overwrite)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
