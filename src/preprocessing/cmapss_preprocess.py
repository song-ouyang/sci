"""
NASA C-MAPSS Turbofan Engine Dataset Preprocessing
Dataset: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6

Processes multi-sensor time series for Remaining Useful Life (RUL) prediction.
Converts sensor readings to structured text for LLM input.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "cmapss"

SENSOR_NAMES = [
    "(Fan inlet temperature) T2",
    "(LPC outlet temperature) T24",
    "(HPC outlet temperature) T30",
    "(LPT outlet temperature) T50",
    "(Fan inlet pressure) P2",
    "(Bypass-duct pressure) P15",
    "(HPC outlet pressure) P30",
    "(Physical fan speed) Nf",
    "(Physical core speed) Nc",
    "(Engine pressure ratio) epr",
    "(HPC outlet static pressure) Ps30",
    "(Ratio of fuel flow to Ps30) phi",
    "(Corrected fan speed) NRf",
    "(Corrected core speed) NRc",
    "(Bypass ratio) BPR",
    "(Burner fuel-air ratio) farB",
    "(Bleed enthalpy) htBleed",
    "(Demanded fan speed) Nf_dmd",
    "(Demanded corrected fan speed) PCNfR_dmd",
    "(HPT coolant bleed) W31",
    "(LPT coolant bleed) W32",
]

COLUMN_NAMES = (
    ["unit_id", "cycle"] +
    ["op_setting_1", "op_setting_2", "op_setting_3"] +
    [f"sensor_{i+1}" for i in range(21)]
)

# RUL clipping: cap max RUL at 125 cycles (standard practice)
MAX_RUL = 125

# Health stages for classification
RUL_BINS = {
    "Healthy": (80, float("inf")),       # RUL > 80
    "Degrading": (30, 80),               # 30 < RUL <= 80
    "Critical": (0, 30),                 # RUL <= 30
}


def load_cmapss_file(filepath: Path) -> pd.DataFrame:
    """Load a C-MAPSS text file into DataFrame."""
    df = pd.read_csv(filepath, sep=r"\s+", header=None, names=COLUMN_NAMES)
    return df


def add_rul_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add RUL and health stage labels."""
    max_cycles = df.groupby("unit_id")["cycle"].max()
    df = df.merge(max_cycles.rename("max_cycle"), on="unit_id")
    df["rul"] = df["max_cycle"] - df["cycle"]
    df["rul_clipped"] = df["rul"].clip(upper=MAX_RUL)
    df.drop(columns=["max_cycle"], inplace=True)

    def get_health_stage(rul):
        for stage, (low, high) in RUL_BINS.items():
            if low < rul <= high:
                return stage
        return "Critical"

    df["health_stage"] = df["rul"].apply(get_health_stage)
    return df


def compute_window_features(window: pd.DataFrame, sensor_cols: list[str]) -> dict:
    """Compute statistical features over a time window of sensor readings."""
    features = {}
    for col in sensor_cols:
        vals = window[col].values
        features[f"{col}_mean"] = float(np.mean(vals))
        features[f"{col}_std"] = float(np.std(vals))
        features[f"{col}_trend"] = float(np.polyfit(range(len(vals)), vals, 1)[0]) if len(vals) > 1 else 0.0
        features[f"{col}_last"] = float(vals[-1])
    return features


def sensor_readings_to_text(row: pd.Series, sensor_cols: list[str]) -> str:
    """Convert a single cycle's sensor readings to text."""
    lines = [
        f"Equipment: Turbofan jet engine (C-MAPSS simulation)",
        f"Engine unit: #{int(row['unit_id'])}",
        f"Operating cycle: {int(row['cycle'])}",
        f"Operating conditions: setting1={row['op_setting_1']:.4f}, "
        f"setting2={row['op_setting_2']:.4f}, setting3={row['op_setting_3']:.4f}",
        f"",
        f"Sensor readings:",
    ]

    for i, col in enumerate(sensor_cols):
        lines.append(f"  - {SENSOR_NAMES[i]}: {row[col]:.4f}")

    return "\n".join(lines)


def window_features_to_text(features: dict, unit_id: int, cycle: int,
                            window_size: int) -> str:
    """Convert windowed features to text."""
    sensor_cols_unique = sorted(set(k.rsplit("_", 1)[0] for k in features))

    lines = [
        f"Equipment: Turbofan jet engine (C-MAPSS simulation)",
        f"Engine unit: #{unit_id}",
        f"Current cycle: {cycle}",
        f"Analysis window: last {window_size} cycles",
        f"",
        f"Sensor trend analysis:",
    ]

    for i, base_col in enumerate(sensor_cols_unique):
        idx = int(base_col.replace("sensor_", "")) - 1
        name = SENSOR_NAMES[idx] if idx < len(SENSOR_NAMES) else base_col
        mean_val = features.get(f"{base_col}_mean", 0)
        std_val = features.get(f"{base_col}_std", 0)
        trend_val = features.get(f"{base_col}_trend", 0)
        last_val = features.get(f"{base_col}_last", 0)

        trend_dir = "increasing" if trend_val > 0.001 else "decreasing" if trend_val < -0.001 else "stable"
        lines.append(
            f"  - {name}: current={last_val:.4f}, "
            f"mean={mean_val:.4f}, std={std_val:.4f}, trend={trend_dir} ({trend_val:+.6f}/cycle)"
        )

    return "\n".join(lines)


def process_cmapss(subset: str = "FD001", window_size: int = 30,
                   sample_per_unit: int = 5) -> pd.DataFrame:
    """Process C-MAPSS dataset and produce text representations.

    Args:
        subset: Which C-MAPSS subset (FD001-FD004)
        window_size: Number of cycles for windowed features
        sample_per_unit: Max samples per engine unit (to limit total size)
    """
    train_path = DATA_DIR / "raw" / f"train_{subset}.txt"
    if not train_path.exists():
        print(f"Please download C-MAPSS data to {DATA_DIR / 'raw'}/")
        print("Download from: https://data.nasa.gov/Aerospace/CMAPSS-Jet-Engine-Simulated-Data/ff5v-kuh6")
        print("Or: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
        return pd.DataFrame()

    df = load_cmapss_file(train_path)
    df = add_rul_labels(df)

    sensor_cols = [f"sensor_{i+1}" for i in range(21)]
    # Drop near-constant sensors (standard practice for FD001)
    constant_sensors = []
    for col in sensor_cols:
        if df[col].std() < 0.001:
            constant_sensors.append(col)
    useful_sensors = [c for c in sensor_cols if c not in constant_sensors]

    records = []
    for unit_id in df["unit_id"].unique():
        unit_df = df[df["unit_id"] == unit_id].sort_values("cycle")
        max_cycle = unit_df["cycle"].max()

        # Sample cycles: early, mid, late life
        sample_cycles = np.linspace(
            window_size, max_cycle, sample_per_unit, dtype=int
        )

        for cycle in sample_cycles:
            row = unit_df[unit_df["cycle"] == cycle]
            if row.empty:
                continue
            row = row.iloc[0]

            # Window features
            window = unit_df[
                (unit_df["cycle"] >= cycle - window_size) &
                (unit_df["cycle"] <= cycle)
            ]
            if len(window) < 5:
                continue

            features = compute_window_features(window, useful_sensors)
            text = window_features_to_text(features, int(unit_id), int(cycle), window_size)

            records.append({
                "source": "C-MAPSS",
                "subset": subset,
                "unit_id": int(unit_id),
                "cycle": int(cycle),
                "rul": int(row["rul"]),
                "rul_clipped": int(row["rul_clipped"]),
                "health_stage": row["health_stage"],
                "label": row["health_stage"],
                "text_representation": text,
                **features,
            })

    result_df = pd.DataFrame(records)
    output_path = DATA_DIR / "cmapss_processed.csv"
    result_df.to_csv(output_path, index=False)
    print(f"C-MAPSS ({subset}): {len(result_df)} samples saved to {output_path}")
    return result_df


if __name__ == "__main__":
    df = process_cmapss()
    if not df.empty:
        print(f"\nHealth stage distribution:\n{df['health_stage'].value_counts()}")
