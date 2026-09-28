"""
CWRU Bearing Dataset Preprocessing
Dataset: https://engineering.case.edu/bearingdatacenter/download-data-file

Downloads .mat files, extracts vibration signals, computes features,
and converts to structured text for LLM input.
"""

import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.fft import fft, fftfreq
from scipy.stats import kurtosis, skew
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "cwru"

# CWRU file mapping: (filename, label, description)
# Drive end bearing data, 12kHz sampling rate, loads 0-3 HP
CWRU_FILES = {
    # Normal baseline
    "97.mat":  ("Normal", "Normal baseline, 0 HP load"),
    "98.mat":  ("Normal", "Normal baseline, 1 HP load"),
    "99.mat":  ("Normal", "Normal baseline, 2 HP load"),
    "100.mat": ("Normal", "Normal baseline, 3 HP load"),
    # Inner race fault (0.007 inch)
    "105.mat": ("IR007", "Inner race fault 0.007 inch, 0 HP"),
    "106.mat": ("IR007", "Inner race fault 0.007 inch, 1 HP"),
    "107.mat": ("IR007", "Inner race fault 0.007 inch, 2 HP"),
    "108.mat": ("IR007", "Inner race fault 0.007 inch, 3 HP"),
    # Ball fault (0.007 inch)
    "118.mat": ("B007", "Ball fault 0.007 inch, 0 HP"),
    "119.mat": ("B007", "Ball fault 0.007 inch, 1 HP"),
    "120.mat": ("B007", "Ball fault 0.007 inch, 2 HP"),
    "121.mat": ("B007", "Ball fault 0.007 inch, 3 HP"),
    # Outer race fault (0.007 inch, centered @6:00)
    "130.mat": ("OR007", "Outer race fault 0.007 inch, 0 HP"),
    "131.mat": ("OR007", "Outer race fault 0.007 inch, 1 HP"),
    "132.mat": ("OR007", "Outer race fault 0.007 inch, 2 HP"),
    "133.mat": ("OR007", "Outer race fault 0.007 inch, 3 HP"),
    # Inner race fault (0.014 inch)
    "169.mat": ("IR014", "Inner race fault 0.014 inch, 0 HP"),
    "170.mat": ("IR014", "Inner race fault 0.014 inch, 1 HP"),
    "171.mat": ("IR014", "Inner race fault 0.014 inch, 2 HP"),
    "172.mat": ("IR014", "Inner race fault 0.014 inch, 3 HP"),
    # Ball fault (0.014 inch)
    "185.mat": ("B014", "Ball fault 0.014 inch, 0 HP"),
    "186.mat": ("B014", "Ball fault 0.014 inch, 1 HP"),
    "187.mat": ("B014", "Ball fault 0.014 inch, 2 HP"),
    "188.mat": ("B014", "Ball fault 0.014 inch, 3 HP"),
    # Outer race fault (0.014 inch)
    "197.mat": ("OR014", "Outer race fault 0.014 inch, 0 HP"),
    "198.mat": ("OR014", "Outer race fault 0.014 inch, 1 HP"),
    "199.mat": ("OR014", "Outer race fault 0.014 inch, 2 HP"),
    "200.mat": ("OR014", "Outer race fault 0.014 inch, 3 HP"),
    # Inner race fault (0.021 inch)
    "209.mat": ("IR021", "Inner race fault 0.021 inch, 0 HP"),
    "210.mat": ("IR021", "Inner race fault 0.021 inch, 1 HP"),
    "211.mat": ("IR021", "Inner race fault 0.021 inch, 2 HP"),
    "212.mat": ("IR021", "Inner race fault 0.021 inch, 3 HP"),
    # Ball fault (0.021 inch)
    "222.mat": ("B021", "Ball fault 0.021 inch, 0 HP"),
    "223.mat": ("B021", "Ball fault 0.021 inch, 1 HP"),
    "224.mat": ("B021", "Ball fault 0.021 inch, 2 HP"),
    "225.mat": ("B021", "Ball fault 0.021 inch, 3 HP"),
    # Outer race fault (0.021 inch)
    "234.mat": ("OR021", "Outer race fault 0.021 inch, 0 HP"),
    "235.mat": ("OR021", "Outer race fault 0.021 inch, 1 HP"),
    "236.mat": ("OR021", "Outer race fault 0.021 inch, 2 HP"),
    "237.mat": ("OR021", "Outer race fault 0.021 inch, 3 HP"),
}

SAMPLE_RATE = 12000  # 12 kHz
SEGMENT_LENGTH = 2048  # samples per segment


def download_cwru_file(filename: str) -> Path:
    """Download a single .mat file from CWRU."""
    filepath = DATA_DIR / "raw" / filename
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if filepath.exists():
        return filepath

    file_num = filename.replace(".mat", "")
    url = f"https://engineering.case.edu/sites/default/files/{file_num}.mat"

    print(f"Downloading {filename}...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    filepath.write_bytes(resp.content)
    return filepath


def extract_vibration_signal(filepath: Path) -> np.ndarray:
    """Extract drive-end accelerometer signal from .mat file."""
    mat = loadmat(str(filepath))
    # CWRU keys follow pattern: X{num}_DE_time
    for key in mat:
        if "DE_time" in key:
            return mat[key].flatten()
    raise ValueError(f"No DE_time key found in {filepath}")


def segment_signal(signal: np.ndarray, segment_len: int = SEGMENT_LENGTH,
                   overlap: float = 0.5) -> list[np.ndarray]:
    """Split signal into overlapping segments."""
    step = int(segment_len * (1 - overlap))
    segments = []
    for start in range(0, len(signal) - segment_len + 1, step):
        segments.append(signal[start:start + segment_len])
    return segments


def compute_time_features(segment: np.ndarray) -> dict:
    """Compute time-domain statistical features."""
    return {
        "mean": float(np.mean(segment)),
        "std": float(np.std(segment)),
        "rms": float(np.sqrt(np.mean(segment ** 2))),
        "peak": float(np.max(np.abs(segment))),
        "peak_to_peak": float(np.max(segment) - np.min(segment)),
        "crest_factor": float(np.max(np.abs(segment)) / np.sqrt(np.mean(segment ** 2))),
        "kurtosis": float(kurtosis(segment)),
        "skewness": float(skew(segment)),
        "shape_factor": float(np.sqrt(np.mean(segment ** 2)) / np.mean(np.abs(segment))),
    }


def compute_freq_features(segment: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
    """Compute frequency-domain features via FFT."""
    n = len(segment)
    yf = np.abs(fft(segment))[:n // 2]
    xf = fftfreq(n, 1 / sr)[:n // 2]

    # Normalize
    yf = yf / n

    # Spectral features
    total_power = np.sum(yf ** 2)
    freq_center = np.sum(xf * yf ** 2) / total_power if total_power > 0 else 0
    freq_std = np.sqrt(np.sum((xf - freq_center) ** 2 * yf ** 2) / total_power) if total_power > 0 else 0

    # Top 5 dominant frequencies
    top_indices = np.argsort(yf)[-5:][::-1]
    top_freqs = xf[top_indices].tolist()
    top_amps = yf[top_indices].tolist()

    return {
        "spectral_centroid": float(freq_center),
        "spectral_spread": float(freq_std),
        "total_spectral_power": float(total_power),
        "dominant_freq_1": float(top_freqs[0]),
        "dominant_amp_1": float(top_amps[0]),
        "dominant_freq_2": float(top_freqs[1]),
        "dominant_amp_2": float(top_amps[1]),
        "dominant_freq_3": float(top_freqs[2]),
        "dominant_amp_3": float(top_amps[2]),
    }


def features_to_text(time_feats: dict, freq_feats: dict, description: str) -> str:
    """Convert features to structured text for LLM input.
    NOTE: description is NOT included to avoid label leakage.
    """
    load_info = description.split(",")[-1].strip() if "," in description else ""
    header = f"Equipment: Rolling element bearing (6205-2RS JEM SKF deep groove ball bearing)\n"
    header += f"Sensor: Drive-end accelerometer, sampling rate {SAMPLE_RATE} Hz\n"
    if load_info:
        header += f"Operating load: {load_info}\n"
    text = (
        header +
        f"\n"
        f"Time-domain features:\n"
        f"  - Mean amplitude: {time_feats['mean']:.6f} g\n"
        f"  - Standard deviation: {time_feats['std']:.6f} g\n"
        f"  - RMS value: {time_feats['rms']:.6f} g\n"
        f"  - Peak amplitude: {time_feats['peak']:.6f} g\n"
        f"  - Peak-to-peak: {time_feats['peak_to_peak']:.6f} g\n"
        f"  - Crest factor: {time_feats['crest_factor']:.4f}\n"
        f"  - Kurtosis: {time_feats['kurtosis']:.4f}\n"
        f"  - Skewness: {time_feats['skewness']:.4f}\n"
        f"  - Shape factor: {time_feats['shape_factor']:.4f}\n"
        f"\n"
        f"Frequency-domain features:\n"
        f"  - Spectral centroid: {freq_feats['spectral_centroid']:.2f} Hz\n"
        f"  - Spectral spread: {freq_feats['spectral_spread']:.2f} Hz\n"
        f"  - Total spectral power: {freq_feats['total_spectral_power']:.6f}\n"
        f"  - Dominant frequency 1: {freq_feats['dominant_freq_1']:.2f} Hz "
        f"(amplitude: {freq_feats['dominant_amp_1']:.6f})\n"
        f"  - Dominant frequency 2: {freq_feats['dominant_freq_2']:.2f} Hz "
        f"(amplitude: {freq_feats['dominant_amp_2']:.6f})\n"
        f"  - Dominant frequency 3: {freq_feats['dominant_freq_3']:.2f} Hz "
        f"(amplitude: {freq_feats['dominant_amp_3']:.6f})\n"
    )
    return text


def process_cwru(max_segments_per_file: int = 20) -> pd.DataFrame:
    """Process all CWRU files and return a DataFrame of samples."""
    records = []

    for filename, (label, description) in CWRU_FILES.items():
        try:
            filepath = download_cwru_file(filename)
            signal = extract_vibration_signal(filepath)
            segments = segment_signal(signal)

            # Limit segments per file
            for seg in segments[:max_segments_per_file]:
                time_feats = compute_time_features(seg)
                freq_feats = compute_freq_features(seg)
                text = features_to_text(time_feats, freq_feats, description)

                record = {
                    "source": "CWRU",
                    "filename": filename,
                    "label": label,
                    "label_coarse": "Normal" if label == "Normal" else "Fault",
                    "description": description,
                    "text_representation": text,
                    **{f"time_{k}": v for k, v in time_feats.items()},
                    **{f"freq_{k}": v for k, v in freq_feats.items()},
                }
                records.append(record)

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    df = pd.DataFrame(records)
    output_path = DATA_DIR / "cwru_processed.csv"
    df.to_csv(output_path, index=False)
    print(f"CWRU: {len(df)} samples saved to {output_path}")
    return df


if __name__ == "__main__":
    df = process_cwru()
    print(f"\nLabel distribution:\n{df['label'].value_counts()}")
