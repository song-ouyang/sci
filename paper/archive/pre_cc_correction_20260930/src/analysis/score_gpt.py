"""Validate frozen GPT predictions and compare their scores with Claude baselines.

Run from the project root with:
    .venv/bin/python3 src/analysis/score_gpt.py

Uses the strict multi-model scorer without making inference calls or rewriting
predictions, manifests, or score files. Legacy *_gpt_s.json files are not used.
"""

import sys
from statistics import mean

if __package__:
    from . import score_gpt_models as scorer
else:
    import score_gpt_models as scorer


STRATEGIES = ["zero_shot", "few_shot", "cot", "domain"]
GPT_MODELS = {
    "gpt-5.5": "GPT-5.5",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "gpt-6-astra": "GPT-6 Astra",
}
CLAUDE_BASELINES = {
    "CMAPSS": {
        "Opus 4.6":  [86.7, 83.3, 83.3, 86.7],
        "Sonnet 5":  [86.7, 86.7, 83.3, 83.3],
        "Haiku 4.5": [86.7, 86.7, 80.0, 80.0],
    },
    "TEP": {
        "Opus 4.6":  [19.0, 33.3, 28.6, 33.3],
        "Sonnet 5":  [28.6, 28.6, 28.6, 28.6],
        "Haiku 4.5": [19.0, 19.0, 19.0, 19.0],
    },
}
# Preserve the previously reported means instead of averaging rounded cells.
CLAUDE_MEANS = {
    "CMAPSS": {"Opus 4.6": 85.0, "Sonnet 5": 85.0, "Haiku 4.5": 83.3},
    "TEP": {"Opus 4.6": 28.6, "Sonnet 5": 28.6, "Haiku 4.5": 19.0},
}


def score_frozen_models():
    """Validate every model and frozen manifest before opening any answer keys."""
    batches, input_digest = scorer.load_batches(scorer.ROOT / "gpt_experiment_batches.json")
    expected_groups = {f"{dataset}_{strategy}" for dataset in ("cmapss", "tep") for strategy in STRATEGIES}
    scorer.require(set(batches) == expected_groups, "Expected exactly the eight experiment groups")
    results_root = scorer.ROOT / "results" / "gpt_models"
    predictions = {}
    for model in GPT_MODELS:
        predictions[model], snapshot = scorer.validate_model(model, results_root, batches, input_digest)
        scorer.check_manifest(model, results_root, snapshot)
    keys, _ = scorer.load_keys(batches, scorer.ROOT / "results" / "batches_clean")
    return {
        model: {
            group: scorer.metrics(
                keys[group], [row["predicted"] for row in rows], batches[group]["valid_labels"]
            )
            for group, rows in groups.items()
        }
        for model, groups in predictions.items()
    }


def print_row(model, values, precision=2, reported_mean=None):
    average = mean(values) if reported_mean is None else reported_mean
    print(f"  {model:<16} " + " ".join(f"{value:8.{precision}f}" for value in values) + f" {average:8.{precision}f}")


def main():
    try:
        scores = score_frozen_models()
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("=" * 76)
    print("GPT Experiment Results vs Claude Baselines")
    print("Validated 3 frozen models, 24 groups, 612 predictions; no samples excluded.")
    print("These are repeated evaluations of 51 unique samples, not 612 independent samples.")
    print("GPT protocol: grouped Codex HTTPS sessions, reasoning effort = medium.")
    print("Claude accuracy values below are the previously reported baselines.")
    print("Both datasets expose class-correlated IDs: archived prompt benchmarks,")
    print("not clean sensor-only validation; this is a shortcut risk, not proof of its use.")
    for dataset, label, classes, random_baseline in [
        ("cmapss", "C-MAPSS", 3, 33.3),
        ("tep", "TEP", 7, 14.3),
    ]:
        print(f"\n{label} ({classes}-class) | Random baseline: {random_baseline}%")
        if dataset == "tep":
            print("Legacy prompt benchmark: metadata contaminated sensor fields; not clean diagnosis validation.")
        print("Accuracy (%)")
        print(f"  {'Model':<16} {'ZS':>8} {'FS':>8} {'CoT':>8} {'Domain':>8} {'Mean':>8}")
        for model, values in CLAUDE_BASELINES[dataset.upper()].items():
            print_row(model, values, precision=1, reported_mean=CLAUDE_MEANS[dataset.upper()][model])
        for model, display_name in GPT_MODELS.items():
            print_row(display_name, [scores[model][f"{dataset}_{strategy}"]["accuracy_pct"] for strategy in STRATEGIES])

        print("\nGPT macro F1 (%) over all valid labels")
        print(f"  {'Model':<16} {'ZS':>8} {'FS':>8} {'CoT':>8} {'Domain':>8} {'Mean':>8}")
        for model, display_name in GPT_MODELS.items():
            print_row(display_name, [scores[model][f"{dataset}_{strategy}"]["macro_f1_pct"] for strategy in STRATEGIES])

    print("\nMeans average the four prompt strategies; stored experiment files are unchanged.")
    print(f"Machine-readable scores: {scorer.ROOT / 'results' / 'gpt_models' / 'scores.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
