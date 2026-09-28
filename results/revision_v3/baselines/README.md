# Matched-feature supervised and trivial baselines

Run `.venv/bin/python3 -B src/experiments/build_revision_v3.py --baselines-only`
from the project root after the data freeze. This reuses and validates the
immutable `../data/datasets.json`; it does not run an LLM.

- `predictions.json`: RF, SVM and training-majority predictions with the same
  test IDs and order as the frozen data.
- `scores.json`: fixed settings and seed, software versions, complete-set
  accuracy/macro F1, confusion matrices, training priors and analytical expected
  accuracy for prior-random/uniform-random prediction.
- `manifest.json`: SHA-256 for the exact dataset and both result files.

Both supervised methods use exactly the stored rounded features used to render
LLM prompts. StandardScaler is fitted only on the training rows. No parameter is
chosen by inspecting test scores. These comparisons use learned training labels;
they are not zero-shot baselines. See `../data/README.md` for split provenance and
limitations. Do not reuse v1/v2 cross-validation scores as this held-out baseline.
