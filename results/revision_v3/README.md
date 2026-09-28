# Revision v3: independent-unit evaluation and paired sensor-mapping audit

This revision is a new exploratory experiment. It does not overwrite or merge the
historical `results/gpt_models`, `results/gpt_models_v2`, or Claude archives.
The complete protocol is `protocol.json`, frozen before the first model request.
The whole-batch N/A reporting rule is documented separately in the timestamped
pre-score `failure_reporting_amendment.json`; it was not in the original protocol.

For a portable offline quickstart, model-adapter rules, and case-level export
examples, start with [REUSE_GUIDE.md](REUSE_GUIDE.md).

## What is evaluated

- C-MAPSS FD001: 60 official test engines, one final observed window per engine,
  20 per RUL category. Training uses 500 past-only windows from 100 other engines.
- TEP: 70 different held-out simulation-run identifiers, one observation per run,
  ten per condition. Training uses 2,100 observations from 100 other run IDs.
  These are new splits within the publisher's Training files, with observations
  long after fault injection. This is not an early-warning experiment.
- Models: the exact requested service aliases `gpt-5.5`, `gpt-5.6-sol`,
  `gpt-6-astra`; `medium` reasoning; tools disabled; fresh grouped sessions.
- Conditions: C-MAPSS clean, TEP clean, TEP descriptor-mapping fault. Each is
  repeated three times, for 27 sessions and 1,800 requested predictions on 130
  distinct evaluation units. These are not 1,800 independent test cases.
- Mapping intervention: cyclic +1 rotation of all 52 full TEP descriptors,
  including sensor identities and units; numeric strings, sample IDs, task text,
  and within-batch order stay fixed. This is a severe synthetic schema fault,
  not a reconstruction of the older input defect or a meaning-preserving rewrite.

## Files

| Path | Role |
|---|---|
| `data/datasets.json` | Rounded arrays, labels, source groups and provenance; never read by inference |
| `data/manifest.json` | Raw source hashes, fixed split and extraction parameters |
| `data/validation.json` | Schema, group, past-only window, and representation checks |
| `data/independent_verification.json` | Separate reconstruction of raw TEP rows and baseline checks |
| `inference/batches.json` | Only sample IDs, prompts, permitted labels and condition metadata |
| `inference/prompts/*.txt` | Exact request payloads |
| `protocol.json` | Prespecified metrics, repeats, failure handling, uncertainty and limitations |
| `predictions/<model>/` | Exact outputs, full responses, requests, event logs and metadata |
| `predictions/frozen_manifest.json` | SHA-256 freeze of all scheduled terminal outcomes, including validation failure, before scoring |
| `failure_reporting_amendment.json` | Timestamped pre-score amendment specifying N/A reporting for incomplete output |
| `baselines/` | Matched RF/SVM/majority outputs, settings and provenance |
| `analysis/` | Verified scores, repeated-session summaries and paired contrasts |
| `analysis/figure_data/` | Values behind all four quantitative figures, with source hashes |
| `analysis/reuse/` | Sample-level clean correctness, paired TEP cases and export provenance |
| `prediction_contract.schema.json` | Dataset-specific successful-response JSON shapes and exact label vocabularies; the scorer also validates ID coverage and order |

## Reproduce the analysis of frozen outputs

Run from the project root, `/Users/guoba/work/sci-paper-llm-pdm`:

```bash
.venv/bin/python3 src/analysis/freeze_revision_v3.py
.venv/bin/python3 src/analysis/score_revision_v3.py
.venv/bin/python3 src/analysis/export_revision_v3_cases.py
.venv/bin/python3 src/analysis/render_revision_v3_paper.py
```

Then compile in the `paper` directory:

```bash
tectonic main_inline.tex
tectonic main.tex
```

Both manuscript sources are generated identically from the reviewed template
and verified numerical outputs. Neither relies on the old broken table inputs.
`main_inline.tex` remains the canonical compilation entry.
The renderer generates five figures: the evaluation workflow, primary accuracy
with intervals, paired mapping sensitivity, class-level recall, and repeated
session accuracy. Class-level visualizations and CSV exports were added during
manuscript development as descriptive views of existing frozen results. They
do not change the primary session, frozen predictions, or scoring rules.

## Preparation and inference

The scripts below define the preparation and inference steps. Existing frozen
files are validated and reused; they are not replaced by a rerun.

The initial worker stops when a batch fails semantic validation. The recorded
GPT-5.5 C-MAPSS repeat-1 response returned 59 objects instead of 60. Its full raw
response is retained and its classification metrics are N/A. It is not repaired,
rescored partially, or replaced by a later repeat. The pre-score reporting
amendment documents this distinction; it does not rewrite the original protocol.
`continue_revision_v3.py --model gpt-5.5` executed only the remaining never-attempted
scheduled job. Do not restart a failed group through the initial runner: semantic
retries are prohibited and the cumulative transport-attempt ceiling is two.

```bash
.venv/bin/python3 src/experiments/build_revision_v3.py --data-only
.venv/bin/python3 src/experiments/build_revision_v3.py --baselines-only
.venv/bin/python3 src/experiments/prepare_revision_v3.py
.venv/bin/python3 src/experiments/run_revision_v3.py
```

Raw files must be obtained from the sources recorded in `data/README.md` and
match the archived hashes for an exact data reconstruction. Fresh model calls
require a locally authenticated Codex CLI with the three requested models.
Preserve this frozen version when creating a separate future experiment.

## Interpretation

Repeat 1 was specified as the primary descriptive comparison. Repeats 2 and 3
measure changes across fresh sessions receiving the same grouped input; they
are reported separately, not pooled to inflate sample size. Accuracy Wilson
intervals use an independent-source-unit approximation. Group bootstrap
intervals are conditional on the fixed batch responses and do not resolve
shared-context dependence, provider changes, or population-selection effects.
No significance-screened subset of the planned conditions is reported.

The supervised models fit labeled data; the LLMs receive semantic descriptors
and task definitions but no in-context labeled examples. This is a practical
comparison of interfaces, not an equal-training comparison. Balanced subsets,
late TEP snapshots and a single simulated engine condition limit generalization.

The CLI records requested aliases but does not expose fixed server snapshots or
complete decoding settings. Token usage and latency are recorded; monetary cost
is not inferred from subscription-authenticated requests. No public release,
repository, archive DOI, license or journal acceptance is claimed by this local
package.
