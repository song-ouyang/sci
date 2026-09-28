# Post-v3 mechanism and context extension

This is a separately frozen exploratory follow-up designed after inspecting the original v3 results. It reuses 70 TEP runs and selects two services for their contrasting original behavior. It does **not** add independent test sources, validate industrial deployment, or constitute a new pre-registered confirmatory study.

## Fixed design

- Models: `gpt-5.6-sol`, `gpt-6-astra`; requested medium reasoning.
- Three fresh repetitions of each request configuration.
- Six 70-record conditions: contemporaneous clean, original cyclic mapping fault, correct pair reordering, anonymous positional fields, fixed random descriptor derangement, and clean sample reordering.
- Seven probe records selected by a salted hash of their anonymous IDs, without inspecting labels or predictions. Each is evaluated in the full clean batch, a seven-record batch, and an independent single-record session.
- 84 requests; 2,604 requested classifications; **70 unique sources, zero new sources**.
- One CLI invocation per scheduled job. No manual repair, replacement or semantic retry. Internal stream reconnects remain logged. A failed request has no scored partial output.
- No mid-run scoring. All scheduled outputs must be terminal and frozen before the scorer opens the answer key.

`protocol.json` defines the exact rules and hashes; `session_schedule.json` fixes each model's request order. All inference inputs are in `inference/`, which contains no ground-truth field.

## Commands

Run from the project root with the project Python environment:

```bash
.venv/bin/python src/experiments/mechanism_extension.py validate
.venv/bin/python src/experiments/mechanism_extension.py status
```

Only the `run` command makes new service requests. It preserves terminal outcomes and resumes only never-attempted jobs. A transport failure stops that model's schedule. An interrupted job is not silently repeated, and a process lock prevents duplicate launchers.

```bash
.venv/bin/python src/experiments/mechanism_extension.py run
```

After all jobs have terminal outcomes, freeze exactly once, then score:

```bash
.venv/bin/python src/experiments/mechanism_extension.py freeze
.venv/bin/python src/experiments/mechanism_extension.py score
```

Subsequent offline reproduction runs **score only**, never `prepare`, `run` or `freeze`. The audit verifies protocol/code/prompt hashes, complete event records, unique session IDs and raw-response equality before opening labels.

## Interpretation

| Contrast | What it probes | What it does not establish |
|---|---|---|
| Clean vs correct pair reordering | Sensitivity to representation order with physical pairings preserved | A model's internal reasoning mechanism |
| Clean vs anonymous fields | Joint removal of identity, name and unit with numeric positions preserved | Separate causal contributions of names vs units |
| Clean vs random mapping | A noncyclic alternative to the original regular rotation | Robustness across all mapping permutations |
| Clean vs sample reordering | Presentation order/context sensitivity on the same observations | Pure order effects separated from service stochasticity |
| Full vs seven-record vs single-record | Context-associated label changes for the same seven cases | A population estimate of batch-size effects |

The random descriptor derangement and the field-order permutation are each fixed once, with seeds specified in code and protocol. The seven context probes need not contain every class. Predictions are paired **by ID**, particularly after sample reordering; order alignment would be incorrect.

`analysis/scores.json` reports every request and every prespecified paired contrast; `analysis/summary.md` is a compact full-batch summary. Conditional intervals do not account for shared-context dependence, unknown server model revisions, or sample selection. Repeats do not increase the number of independent source units. No publication/DOI/license is implied by this local package.

## Case-level exports

After scoring, run `python src/analysis/export_mechanism_cases.py`. It writes
`analysis/cases/predictions.csv` (including unavailable requested rows) and
`paired_transitions.csv`, and independently recomputes every accuracy and paired
flip/difference aggregate. The exporter never triggers model inference.

For the complete manuscript rebuild, use
`python src/analysis/rebuild_mechanism_review.py` from the project root.
