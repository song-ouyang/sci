"""
Prepare experiment batches for manual LLM evaluation.
Samples data, generates prompts, saves to JSON files for batch processing.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.prompts.prompt_templates import get_prompt

DATA_DIR = Path(__file__).parent.parent.parent / "data"
BATCH_DIR = Path(__file__).parent.parent.parent / "results" / "batches"


def prepare_batch(dataset: str, strategy: str, n_samples: int = 50, seed: int = 42):
    """Prepare a batch of prompts for evaluation."""
    paths = {
        "cwru": DATA_DIR / "cwru" / "cwru_processed.csv",
        "cmapss": DATA_DIR / "cmapss" / "cmapss_processed.csv",
        "tep": DATA_DIR / "tep" / "tep_processed.csv",
    }

    df = pd.read_csv(paths[dataset])

    # Get classes and labels (CWRU uses fine-grained 10-class labels)
    if dataset == "cwru":
        df["true_label"] = df["label"]
        classes = ["Normal", "IR007", "IR014", "IR021", "B007", "B014", "B021", "OR007", "OR014", "OR021"]
    elif dataset == "cmapss":
        df["true_label"] = df["health_stage"]
        classes = ["Healthy", "Degrading", "Critical"]
    elif dataset == "tep":
        df["true_label"] = df["label"]
        classes = sorted(df["label"].unique().tolist())

    # Stratified sample
    np.random.seed(seed)
    per_class = max(1, n_samples // len(classes))
    frames = []
    for cls in classes:
        subset = df[df["true_label"] == cls]
        if len(subset) > 0:
            frames.append(subset.sample(min(len(subset), per_class), random_state=seed))
    sampled = pd.concat(frames, ignore_index=True)

    # Generate prompts and save
    batch = []
    for idx, row in sampled.iterrows():
        prompt = get_prompt(dataset, strategy, row["text_representation"], classes)
        batch.append({
            "id": int(idx),
            "dataset": dataset,
            "strategy": strategy,
            "true_label": row["true_label"],
            "prompt": prompt,
        })

    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BATCH_DIR / f"{dataset}_{strategy}.json"
    with open(out_path, "w") as f:
        json.dump(batch, f, indent=2)

    print(f"Saved {len(batch)} prompts to {out_path}")
    return batch


if __name__ == "__main__":
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["cwru", "cmapss", "tep"]
    for ds in datasets:
        n = 200 if ds == "cwru" else 150 if ds == "cmapss" else 70
        for strategy in ["zero_shot", "few_shot", "cot", "domain"]:
            prepare_batch(ds, strategy, n_samples=n)
