# Revision-v3 diagnostic figure data

Generate from the project root with:

```sh
.venv/bin/python3 src/analysis/plot_revision_v3_diagnostics.py
```

Only Python, NumPy, and Matplotlib are required. The `--scores`, `--datasets`, `--figure-dir`, and `--csv-dir` options allow reuse. The script verifies the dataset hash, sample counts, class supports, metric identities, and paired contrast arithmetic before plotting. It never runs inference or edits source scores, prompts, or protocols. `diagnostics_provenance.json` records source/artifact hashes and statistical scope.

CSV rates and CI limits are percentages; effect differences are percentage points (pp). Blank numeric cells represent unavailable results, not zero. Primary means LLM repeat 1; baseline repeat is blank because it was fitted once. All class supports describe the frozen test set, including the unscored primary batch.

## Suggested captions and interpretation limits

**primary_accuracy:** Clean-input accuracy for the prespecified primary session and the matched RF/SVM baselines, evaluated on 60 C-MAPSS engines and 70 TEP run IDs. Bars show two-sided descriptive 95% Wilson intervals. The invalid GPT-5.5 C-MAPSS batch remains N/A; no later repeat replaces it. LLMs use zero-shot grouped prompts whereas baselines use labeled training data. These are distinct information regimes.

**mapping_sensitivity:** Paired TEP accuracy difference (clean minus misnamed descriptors, pp) and prediction flip rate (%), using the same 70 run IDs in each pair. The filled R1 points have 95% paired group-bootstrap effect intervals (10,000 draws); open R2/R3 points show session variation without pooling sessions as additional test observations. The intervention cyclically shifts all 52 complete descriptors while preserving numeric values and their order. Astra has zero observed effect and zero flips in all three pairs; its degenerate [0, 0] empirical effect interval does not establish equivalence or general robustness. A per-pair Wilson interval for zero flips is retained in the CSV.

**class_recall:** Primary-session class recall (%) for clean C-MAPSS inputs (20 engines per class) and both TEP descriptor conditions (10 run IDs per class). Each heatmap cell uses its true-class denominator. Gray N/A cells denote the invalid primary batch. Per-class Wilson intervals are retained in the CSV; the heatmap is descriptive and does not support class-wise superiority claims.

All intervals condition on the selected devices/runs and observed grouped outputs. They do not capture shared-batch dependence, model/session variability, selection bias, or benchmark contamination. They are unadjusted descriptive intervals, not confirmatory tests. Zero observed errors on the selected late-stage TEP subset do not establish industrial safety or early detection performance.
