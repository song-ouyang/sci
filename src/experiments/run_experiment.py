"""
Main experiment runner.

Runs all combinations of: datasets × LLM models × prompt strategies.
Saves results to CSV and computes evaluation metrics.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.prompts.prompt_templates import get_prompt, PROMPT_REGISTRY
from src.experiments.llm_client import MODELS, query_llm, parse_diagnosis, get_available_models

RESULTS_DIR = Path(__file__).parent.parent.parent / "results"
DATA_DIR = Path(__file__).parent.parent.parent / "data"

DATASETS = ["cwru", "cmapss", "tep"]
STRATEGIES = ["zero_shot", "few_shot", "cot", "domain"]


def load_dataset(name: str) -> pd.DataFrame:
    """Load a preprocessed dataset."""
    paths = {
        "cwru": DATA_DIR / "cwru" / "cwru_processed.csv",
        "cmapss": DATA_DIR / "cmapss" / "cmapss_processed.csv",
        "tep": DATA_DIR / "tep" / "tep_processed.csv",
    }
    path = paths[name]
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}. Run preprocessing first.")
    return pd.read_csv(path)


def get_label_column(dataset_name: str) -> str:
    """Get the appropriate label column for each dataset."""
    if dataset_name == "cwru":
        return "label_coarse"  # Normal / Fault -> mapped to 4-class
    elif dataset_name == "cmapss":
        return "health_stage"
    elif dataset_name == "tep":
        return "label"
    return "label"


def get_classes(dataset_name: str, df: pd.DataFrame) -> list[str]:
    """Get the valid class labels for a dataset."""
    if dataset_name == "cwru":
        return ["Normal", "Inner Race Fault", "Ball Fault", "Outer Race Fault"]
    elif dataset_name == "cmapss":
        return ["Healthy", "Degrading", "Critical"]
    elif dataset_name == "tep":
        return sorted(df["label"].unique().tolist())
    return []


def map_cwru_label(label: str) -> str:
    """Map fine-grained CWRU labels to coarse 4-class labels."""
    if label == "Normal":
        return "Normal"
    elif label.startswith("IR"):
        return "Inner Race Fault"
    elif label.startswith("B"):
        return "Ball Fault"
    elif label.startswith("OR"):
        return "Outer Race Fault"
    return label


def run_single_experiment(
    dataset_name: str,
    model_key: str,
    strategy: str,
    max_samples: int = 100,
) -> dict:
    """Run one experiment: dataset × model × strategy."""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name} | Model: {model_key} | Strategy: {strategy}")
    print(f"{'='*60}")

    df = load_dataset(dataset_name)
    classes = get_classes(dataset_name, df)
    label_col = "label"

    # Map CWRU to coarse labels
    if dataset_name == "cwru":
        df["label_mapped"] = df["label"].apply(map_cwru_label)
        label_col = "label_mapped"

    elif dataset_name == "cmapss":
        label_col = "health_stage"

    # Sample if needed
    if len(df) > max_samples:
        df_sampled = df.groupby(label_col, group_keys=False).apply(
            lambda x: x.sample(min(len(x), max_samples // len(classes)), random_state=42)
        )
    else:
        df_sampled = df

    model_config = MODELS[model_key]
    y_true = []
    y_pred = []
    raw_responses = []
    latencies = []

    for idx, row in tqdm(df_sampled.iterrows(), total=len(df_sampled), desc=f"{model_key}/{strategy}"):
        prompt = get_prompt(dataset_name, strategy, row["text_representation"], classes)
        true_label = row[label_col]

        start = time.time()
        response = query_llm(prompt, model_config)
        latency = time.time() - start

        predicted = parse_diagnosis(response, classes)

        y_true.append(true_label)
        y_pred.append(predicted)
        raw_responses.append(response)
        latencies.append(latency)

    # Compute metrics
    valid_mask = [p != "UNPARSEABLE" for p in y_pred]
    y_true_valid = [t for t, v in zip(y_true, valid_mask) if v]
    y_pred_valid = [p for p, v in zip(y_pred, valid_mask) if v]

    parseable_rate = sum(valid_mask) / len(valid_mask) if valid_mask else 0

    if y_true_valid:
        accuracy = accuracy_score(y_true_valid, y_pred_valid)
        f1_macro = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
        f1_weighted = f1_score(y_true_valid, y_pred_valid, average="weighted", zero_division=0)
        precision = precision_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
        recall = recall_score(y_true_valid, y_pred_valid, average="macro", zero_division=0)
    else:
        accuracy = f1_macro = f1_weighted = precision = recall = 0.0

    result = {
        "dataset": dataset_name,
        "model": model_config.display_name,
        "model_key": model_key,
        "strategy": strategy,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision,
        "recall_macro": recall,
        "parseable_rate": parseable_rate,
        "avg_latency_s": np.mean(latencies),
        "total_samples": len(df_sampled),
        "valid_samples": sum(valid_mask),
        "timestamp": datetime.now().isoformat(),
    }

    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    print(f"  Parseable rate: {parseable_rate:.2%}")
    print(f"  Avg latency: {np.mean(latencies):.2f}s")

    # Save detailed predictions
    detail_df = pd.DataFrame({
        "true_label": y_true,
        "predicted": y_pred,
        "response": raw_responses,
        "latency_s": latencies,
    })
    detail_path = RESULTS_DIR / "details" / f"{dataset_name}_{model_key}_{strategy}.csv"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    detail_df.to_csv(detail_path, index=False)

    return result


def run_all_experiments(
    datasets: list[str] | None = None,
    model_keys: list[str] | None = None,
    strategies: list[str] | None = None,
    max_samples: int = 100,
):
    """Run all experiment combinations."""
    datasets = datasets or DATASETS
    strategies = strategies or STRATEGIES

    if model_keys is None:
        model_keys = get_available_models()
        if not model_keys:
            print("No models available. Set API keys:")
            print("  export OPENAI_API_KEY=sk-...")
            print("  export DASHSCOPE_API_KEY=sk-...")
            print("  export DEEPSEEK_API_KEY=sk-...")
            print("Or install Ollama for free local models.")
            return

    print(f"Running experiments:")
    print(f"  Datasets: {datasets}")
    print(f"  Models: {model_keys}")
    print(f"  Strategies: {strategies}")
    print(f"  Max samples per dataset: {max_samples}")
    print(f"  Total experiments: {len(datasets) * len(model_keys) * len(strategies)}")

    all_results = []

    for dataset in datasets:
        for model_key in model_keys:
            for strategy in strategies:
                try:
                    result = run_single_experiment(
                        dataset, model_key, strategy, max_samples
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"FAILED: {dataset}/{model_key}/{strategy}: {e}")
                    all_results.append({
                        "dataset": dataset,
                        "model_key": model_key,
                        "strategy": strategy,
                        "error": str(e),
                    })

    # Save summary
    results_df = pd.DataFrame(all_results)
    summary_path = RESULTS_DIR / "tables" / "experiment_summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(summary_path, index=False)
    print(f"\nResults saved to {summary_path}")

    # Print summary table
    if "accuracy" in results_df.columns:
        pivot = results_df.pivot_table(
            values="accuracy",
            index=["dataset", "model"],
            columns="strategy",
            aggfunc="first",
        )
        print("\n=== Accuracy Summary ===")
        print(pivot.to_string(float_format="%.4f"))

    return results_df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run LLM fault diagnosis experiments")
    parser.add_argument("--datasets", nargs="+", default=None, help="Datasets to use")
    parser.add_argument("--models", nargs="+", default=None, help="Model keys to use")
    parser.add_argument("--strategies", nargs="+", default=None, help="Prompt strategies")
    parser.add_argument("--max-samples", type=int, default=100, help="Max samples per dataset")
    args = parser.parse_args()

    run_all_experiments(
        datasets=args.datasets,
        model_keys=args.models,
        strategies=args.strategies,
        max_samples=args.max_samples,
    )
