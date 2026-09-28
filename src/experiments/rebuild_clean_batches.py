"""
Rebuild clean batch files:
  1. Fix TEP preprocessing (strip faultNumber/simulationRun/sample columns)
  2. Anonymize C-MAPSS sample IDs (remove the class-correlated 0-9/50-59/100-109 pattern)
  3. Increase sample sizes: CWRU 100, C-MAPSS 60, TEP 42
  4. Save to results/batches_clean_v2/ (leaves old files untouched)
"""
import json
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.preprocessing.tep_preprocess import process_tep, MEASURED_VARS, MANIPULATED_VARS
from src.prompts.prompt_templates import get_prompt

ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "results" / "batches_clean_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42


# ── helpers ──────────────────────────────────────────────────────────────────

def stratified_sample(df: pd.DataFrame, label_col: str, n_per_class: int, rng) -> pd.DataFrame:
    frames = []
    for cls in sorted(df[label_col].unique()):
        sub = df[df[label_col] == cls]
        k = min(len(sub), n_per_class)
        frames.append(sub.sample(k, random_state=rng.integers(0, 2**31)))
    return pd.concat(frames, ignore_index=True)


def make_anon_ids(n: int, rng) -> list[str]:
    """Return n short opaque hex IDs in shuffled order."""
    # Use two int64 halves to build a 128-bit UUID
    hi = rng.integers(0, 2**62, size=n, dtype=np.int64)
    lo = rng.integers(0, 2**62, size=n, dtype=np.int64)
    ids = [f"{int(h):016x}{int(l):016x}"[:8] for h, l in zip(hi, lo)]
    perm = rng.permutation(n)
    return [ids[i] for i in perm]


def save_batch(records: list[dict], key_records: list[dict], name: str) -> None:
    (OUT_DIR / f"{name}.json").write_text(json.dumps(records, indent=2))
    (OUT_DIR / f"{name}_key.json").write_text(json.dumps(key_records, indent=2))
    print(f"  ✓ {name}: {len(records)} samples")


# ── TEP ──────────────────────────────────────────────────────────────────────

def build_tep_batches(n_per_class: int = 6) -> None:
    print("\n[TEP] Re-preprocessing with fixed column selection …")
    df = process_tep(faults_to_use=[0, 1, 2, 4, 6, 7, 13], samples_per_fault=200)
    if df.empty:
        print("  ERROR: TEP data empty, skipping")
        return

    # Verify fix: var_0 must NOT be the faultNumber/metadata cue
    var0_normal = df[df.label == "Normal"]["var_0"].unique()
    var0_fault1 = df[df.label == "Fault_1"]["var_0"].unique()
    print(f"  var_0 Normal: {var0_normal[:3]}  Fault_1: {var0_fault1[:3]}")
    if 0.0 in var0_normal and (np.isnan(var0_fault1).any() or 0.0 not in var0_fault1):
        print("  WARNING: faultNumber cue still present! Check preprocessing.")
    else:
        print("  ✓ No faultNumber cue in var_0")

    rng = np.random.default_rng(SEED)
    sampled = stratified_sample(df, "label", n_per_class, rng)
    anon_ids = make_anon_ids(len(sampled), rng)

    classes = sorted(df["label"].unique().tolist())
    for strategy in ["zero_shot", "few_shot", "cot", "domain"]:
        records, keys = [], []
        for anon_id, (_, row) in zip(anon_ids, sampled.iterrows()):
            prompt = get_prompt("tep", strategy, row["text_representation"], classes)
            records.append({"id": anon_id, "prompt": prompt})
            keys.append({"id": anon_id, "true_label": row["label"]})
        save_batch(records, keys, f"tep_{strategy}_v2_{n_per_class * len(classes)}")


# ── C-MAPSS ──────────────────────────────────────────────────────────────────

def build_cmapss_batches(n_per_class: int = 20) -> None:
    print("\n[C-MAPSS] Rebuilding with anonymized IDs …")
    df = pd.read_csv(DATA_DIR / "cmapss" / "cmapss_processed.csv")
    df["true_label"] = df["health_stage"]
    classes = ["Healthy", "Degrading", "Critical"]

    rng = np.random.default_rng(SEED)
    sampled = stratified_sample(df, "true_label", n_per_class, rng)
    anon_ids = make_anon_ids(len(sampled), rng)

    # Verify IDs are no longer correlated
    numeric_ids = set(range(200))
    old_ids = set(sampled.index.tolist())
    overlapping = old_ids & numeric_ids
    print(f"  Old numeric IDs sample: {sorted(list(overlapping))[:6]} → replaced with opaque hex")

    for strategy in ["zero_shot", "few_shot", "cot", "domain"]:
        records, keys = [], []
        for anon_id, (_, row) in zip(anon_ids, sampled.iterrows()):
            prompt = get_prompt("cmapss", strategy, row["text_representation"], classes)
            records.append({"id": anon_id, "prompt": prompt})
            keys.append({"id": anon_id, "true_label": row["true_label"]})
        save_batch(records, keys, f"cmapss_{strategy}_v2_{n_per_class * len(classes)}")


# ── CWRU ─────────────────────────────────────────────────────────────────────

def build_cwru_batches(n_per_class: int = 10) -> None:
    print("\n[CWRU] Rebuilding with larger sample …")
    df = pd.read_csv(DATA_DIR / "cwru" / "cwru_processed.csv")
    df["true_label"] = df["label"]
    classes = ["Normal", "IR007", "IR014", "IR021", "B007", "B014", "B021",
               "OR007", "OR014", "OR021"]

    rng = np.random.default_rng(SEED)
    sampled = stratified_sample(df, "true_label", n_per_class, rng)
    anon_ids = make_anon_ids(len(sampled), rng)

    for strategy in ["zero_shot", "few_shot", "cot", "domain"]:
        records, keys = [], []
        for anon_id, (_, row) in zip(anon_ids, sampled.iterrows()):
            prompt = get_prompt("cwru", strategy, row["text_representation"], classes)
            records.append({"id": anon_id, "prompt": prompt})
            keys.append({"id": anon_id, "true_label": row["true_label"]})
        save_batch(records, keys, f"cwru_{strategy}_v2_{n_per_class * len(classes)}")


if __name__ == "__main__":
    build_tep_batches(n_per_class=6)       # 42 samples (6×7)
    build_cmapss_batches(n_per_class=20)   # 60 samples (20×3)
    build_cwru_batches(n_per_class=10)     # 100 samples (10×10)
    print(f"\nAll v2 batches written to {OUT_DIR}")
