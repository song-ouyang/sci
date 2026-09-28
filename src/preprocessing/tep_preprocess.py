"""
Tennessee Eastman Process (TEP) Dataset Preprocessing
Dataset: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1

Chemical process fault detection with 52 process variables and 21 fault types.
Converts process readings to structured text for LLM input.
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    import pyreadr
    HAS_PYREADR = True
except ImportError:
    HAS_PYREADR = False

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "tep"

# TEP variable names (41 measured + 11 manipulated = 52 total)
MEASURED_VARS = [
    "A feed (stream 1) kscmh",
    "D feed (stream 2) kg/hr",
    "E feed (stream 3) kg/hr",
    "A and C feed (stream 4) kscmh",
    "Recycle flow (stream 8) kscmh",
    "Reactor feed rate (stream 6) kscmh",
    "Reactor pressure kPa gauge",
    "Reactor level %",
    "Reactor temperature deg C",
    "Purge rate (stream 9) kscmh",
    "Product separator temperature deg C",
    "Product separator level %",
    "Product separator pressure kPa gauge",
    "Product separator underflow (stream 10) m3/hr",
    "Stripper level %",
    "Stripper pressure kPa gauge",
    "Stripper underflow (stream 11) m3/hr",
    "Stripper temperature deg C",
    "Stripper steam flow kg/hr",
    "Compressor work kW",
    "Reactor cooling water outlet temperature deg C",
    "Separator cooling water outlet temperature deg C",
    "Component A (stream 6) mol%",
    "Component B (stream 6) mol%",
    "Component C (stream 6) mol%",
    "Component D (stream 6) mol%",
    "Component E (stream 6) mol%",
    "Component F (stream 6) mol%",
    "Component A (stream 9) mol%",
    "Component B (stream 9) mol%",
    "Component C (stream 9) mol%",
    "Component D (stream 9) mol%",
    "Component E (stream 9) mol%",
    "Component F (stream 9) mol%",
    "Component G (stream 9) mol%",
    "Component H (stream 9) mol%",
    "Component D (stream 11) mol%",
    "Component E (stream 11) mol%",
    "Component F (stream 11) mol%",
    "Component G (stream 11) mol%",
    "Component H (stream 11) mol%",
]

MANIPULATED_VARS = [
    "D feed flow (valve, stream 2)",
    "E feed flow (valve, stream 3)",
    "A feed flow (valve, stream 1)",
    "A and C feed flow (valve, stream 4)",
    "Compressor recycle valve",
    "Purge valve (stream 9)",
    "Separator pot liquid flow (stream 10)",
    "Stripper liquid product flow (stream 11)",
    "Stripper steam valve",
    "Reactor cooling water flow",
    "Condenser cooling water flow",
]

# Fault descriptions
FAULT_DESCRIPTIONS = {
    0: "Normal operation (no fault)",
    1: "IDV(1): A/C feed ratio, B composition constant (stream 4) - Step",
    2: "IDV(2): B composition, A/C ratio constant (stream 4) - Step",
    3: "IDV(3): D feed temperature (stream 2) - Step",
    4: "IDV(4): Reactor cooling water inlet temperature - Step",
    5: "IDV(5): Condenser cooling water inlet temperature - Step",
    6: "IDV(6): A feed loss (stream 1) - Step",
    7: "IDV(7): C header pressure loss (reduced availability, stream 4) - Step",
    8: "IDV(8): A, B, C feed composition (stream 4) - Random variation",
    9: "IDV(9): D feed temperature (stream 2) - Random variation",
    10: "IDV(10): C feed temperature (stream 4) - Random variation",
    11: "IDV(11): Reactor cooling water inlet temperature - Random variation",
    12: "IDV(12): Condenser cooling water inlet temperature - Random variation",
    13: "IDV(13): Reaction kinetics - Slow drift",
    14: "IDV(14): Reactor cooling water valve - Sticking",
    15: "IDV(15): Condenser cooling water valve - Sticking",
    16: "IDV(16): Unknown",
    17: "IDV(17): Unknown",
    18: "IDV(18): Unknown",
    19: "IDV(19): Unknown",
    20: "IDV(20): Unknown",
    21: "IDV(21): Valve fixed at steady state position",
}


def load_tep_simulation(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load TEP simulation data.

    Supports both .dat format and .RData format (Harvard Dataverse version).
    """
    raw_dir = data_dir / "raw"

    # Try RData format first (Harvard Dataverse download)
    rdata_normal = raw_dir / "TEP_FaultFree_Training.RData"
    if rdata_normal.exists():
        if not HAS_PYREADR:
            print("RData files found. Installing pyreadr...")
            import subprocess
            subprocess.check_call(["pip", "install", "pyreadr", "-q"])
            import pyreadr as pyreadr_mod
            globals()["pyreadr"] = pyreadr_mod

        # Metadata columns present in both RData files that must be excluded from
        # sensor features. faultNumber is 0 in FaultFree and 1-20 in Faulty —
        # leaving it in creates a trivial normal-vs-fault cue; simulationRun and
        # sample are administrative indices, not physical measurements.
        _META_COLS = {"faultNumber", "simulationRun", "sample", "fault_id"}

        print("  Loading RData files...")
        # Load fault-free training data
        result = pyreadr.read_r(str(rdata_normal))
        df_name = list(result.keys())[0]
        normal_df = result[df_name].copy()
        # Drop metadata before adding fault_id so it never leaks into var_cols
        for c in list(normal_df.columns):
            if c.lower() in {m.lower() for m in _META_COLS}:
                normal_df.drop(columns=c, inplace=True)
        normal_df["fault_id"] = 0

        # Load faulty training data
        rdata_faulty = raw_dir / "TEP_Faulty_Training.RData"
        if rdata_faulty.exists():
            result = pyreadr.read_r(str(rdata_faulty))
            df_name = list(result.keys())[0]
            faulty_df = result[df_name].copy()

            # Extract fault_id first, then drop all metadata
            if "faultNumber" in faulty_df.columns:
                fault_ids = faulty_df["faultNumber"].values
            else:
                fault_ids = np.zeros(len(faulty_df), dtype=int)
            for c in list(faulty_df.columns):
                if c.lower() in {m.lower() for m in _META_COLS}:
                    faulty_df.drop(columns=c, inplace=True)
            faulty_df["fault_id"] = fault_ids

            all_df = pd.concat([normal_df, faulty_df], ignore_index=True)
        else:
            all_df = normal_df

        return all_df, normal_df

    # Fallback: try .dat format
    train_normal = raw_dir / "d00.dat"
    if not train_normal.exists():
        train_normal = raw_dir / "d00_te.dat"

    if not train_normal.exists():
        print(f"No TEP data found in {raw_dir}/")
        print("Download from: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1")
        return pd.DataFrame(), pd.DataFrame()

    normal_df = pd.read_csv(train_normal, sep=r"\s+", header=None)
    normal_df["fault_id"] = 0

    fault_dfs = [normal_df]
    for fault_id in range(1, 22):
        fault_file = raw_dir / f"d{fault_id:02d}.dat"
        if not fault_file.exists():
            fault_file = raw_dir / f"d{fault_id:02d}_te.dat"
        if fault_file.exists():
            fdf = pd.read_csv(fault_file, sep=r"\s+", header=None)
            fdf["fault_id"] = fault_id
            fault_dfs.append(fdf)

    all_df = pd.concat(fault_dfs, ignore_index=True)
    return all_df, normal_df


def process_readings_to_text(row: pd.Series, fault_id: int,
                             var_names: list[str]) -> str:
    """Convert a single timestep's process readings to text."""
    lines = [
        f"Equipment: Tennessee Eastman chemical process plant",
        f"Process: Continuous stirred-tank reactor with separator and stripper",
        f"",
        f"Measured process variables:",
    ]

    n_measured = len(MEASURED_VARS)
    for i in range(min(n_measured, len(var_names))):
        val = row.iloc[i] if i < len(row) - 1 else 0
        lines.append(f"  - {MEASURED_VARS[i]}: {val:.4f}")

    lines.append(f"")
    lines.append(f"Manipulated variables:")
    for i in range(len(MANIPULATED_VARS)):
        idx = n_measured + i
        val = row.iloc[idx] if idx < len(row) - 1 else 0
        lines.append(f"  - {MANIPULATED_VARS[i]}: {val:.4f}")

    return "\n".join(lines)


def compute_process_features(readings: np.ndarray, var_indices: list[int]) -> dict:
    """Compute statistical features from process variable readings."""
    features = {}
    for idx in var_indices:
        if idx >= readings.shape[1]:
            continue
        vals = readings[:, idx]
        features[f"var{idx}_mean"] = float(np.mean(vals))
        features[f"var{idx}_std"] = float(np.std(vals))
        features[f"var{idx}_range"] = float(np.max(vals) - np.min(vals))
    return features


def process_tep(faults_to_use: list[int] | None = None,
                samples_per_fault: int = 50) -> pd.DataFrame:
    """Process TEP dataset.

    Args:
        faults_to_use: Which fault IDs to include. None = use a representative subset.
        samples_per_fault: Number of samples per fault type.
    """
    if faults_to_use is None:
        # Use a representative subset: normal + 6 diverse fault types
        faults_to_use = [0, 1, 2, 4, 6, 7, 13]

    all_df, _ = load_tep_simulation(DATA_DIR)
    if all_df.empty:
        return pd.DataFrame()

    var_cols = [c for c in all_df.columns if c != "fault_id"]
    records = []

    for fault_id in faults_to_use:
        fault_df = all_df[all_df["fault_id"] == fault_id]
        if fault_df.empty:
            print(f"No data for fault {fault_id}")
            continue

        # Sample evenly
        indices = np.linspace(0, len(fault_df) - 1, samples_per_fault, dtype=int)
        sampled = fault_df.iloc[indices]

        for _, row in sampled.iterrows():
            text = process_readings_to_text(row, fault_id, var_cols)
            label = "Normal" if fault_id == 0 else f"Fault_{fault_id}"
            label_coarse = "Normal" if fault_id == 0 else "Fault"

            record = {
                "source": "TEP",
                "fault_id": fault_id,
                "label": label,
                "label_coarse": label_coarse,
                "description": FAULT_DESCRIPTIONS.get(fault_id, "Unknown"),
                "text_representation": text,
            }
            # Add raw variable values as features
            for i, col in enumerate(var_cols[:52]):
                record[f"var_{i}"] = float(row[col])

            records.append(record)

    result_df = pd.DataFrame(records)
    output_path = DATA_DIR / "tep_processed.csv"
    result_df.to_csv(output_path, index=False)
    print(f"TEP: {len(result_df)} samples saved to {output_path}")
    return result_df


if __name__ == "__main__":
    df = process_tep()
    if not df.empty:
        print(f"\nFault distribution:\n{df['label'].value_counts()}")
