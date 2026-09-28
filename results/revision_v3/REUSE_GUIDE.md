# Reusing the frozen revision-v3 study

The stored model responses, predictions, rounded features and labels are enough
to recompute the reported statistics, export individual cases, and regenerate
the manuscript figures. This workflow makes **no new model calls** and requires
no model account or paid inference. It reanalyzes a fixed, small exploratory
study; it does not reproduce the provider's future behavior.

Run commands from the project root, or from the `revision_v3_reproduction/`
directory after extracting the reproduction archive. Paths in this guide are
relative to that root. The original raw NASA/TEP downloads are needed only for
reconstructing the extracted dataset, not for analyzing the frozen predictions.

## Counts and the unit of analysis

| Quantity | Meaning |
|---|---|
| 130 source units | 60 C-MAPSS engines and 70 TEP simulation-run IDs, with one test observation per source unit |
| 27 planned sessions | Three requested GPT models × three task/condition combinations × three repetitions |
| 1,800 scheduled predictions | Repeated classifications of the same source units across models, conditions and sessions |
| 26 valid sessions / 1,740 validated predictions | Complete batches with exact IDs and labels |
| One failed session | GPT-5.5 C-MAPSS repeat 1 returned 59 of 60 objects; its whole batch is N/A and none of its partial predictions is scored |
| 27 CLI attempts + one recovered stream reconnect | The reconnect is retained in the diagnostic record; it is not another independent study session |

These counts must not be interchanged. CSV row counts are not independent sample
sizes. C-MAPSS has 60 independent source units, and TEP has 70, irrespective of
how many methods or repeated sessions appear in a table.

**Repeat 1 remains the primary comparison, including its N/A outcome.** Repeats
2 and 3 are sensitivity analyses of fresh sessions with identical within-condition
inputs. They cannot replace a failed primary session. The N/A reporting rule is
documented in the timestamped pre-score `failure_reporting_amendment.json`;
it was not an originally prespecified missing-data rule. The separate
`transport_audit_note.json` records the narrowly defined recovered reconnect.

## Offline reproduction in one command block

Prerequisites are a Python environment containing the versions in
`requirements-revision-v3.txt`, plus Tectonic and its already cached TeX resources.
The reference Python environment is `.venv/bin/python3`; change that path if your
environment has another name. Package installation and the first TeX-resource
download are setup steps, not offline analysis. An offline machine can instead
use an existing environment, a local wheel collection, and a prepared TeX cache.
The command below passes `--only-cached` so compilation will fail rather than
silently fetch a missing resource.

```bash
.venv/bin/python3 - <<'PY'
from pathlib import Path
import subprocess

root = Path.cwd()
python = root / ".venv/bin/python3"
for script in [
    "src/analysis/score_revision_v3.py",
    "src/analysis/export_revision_v3_cases.py",
    "src/analysis/render_revision_v3_paper.py",
]:
    subprocess.run([str(python), script], cwd=root, check=True)
subprocess.run(
    ["tectonic", "--only-cached", "main_inline.tex"],
    cwd=root / "paper", check=True,
)
PY
```

The scorer first verifies all frozen terminal outcomes, requests, raw responses,
logs, protocol hashes and amendments. Only then does it open evaluation labels.
The exporter rechecks that audit and its agreement with the scores. The renderer
generates all five figures, their plotting data, and synchronized `main.tex` and
`main_inline.tex`. `paper/main_inline.pdf` is the canonical compiled manuscript.
The second identical source can also be compiled with
`tectonic --only-cached main.tex` from `paper/`.

The commands update derived statistics, reuse CSVs, figures and manuscript
outputs. They do not change `data/`, `inference/`, `predictions/`, or the original
protocol. Derived manifests contain timestamps, so identical numerical results
need not imply byte-identical derived JSON or PDF files across machines.

To perform an audit without reading evaluation labels or writing results:

```bash
.venv/bin/python3 src/analysis/score_revision_v3.py --validate-only
```

A consumer of this release should use the supplied
`predictions/frozen_manifest.json`. Do not create a new freeze to make altered
predictions pass. The original inference and continuation scripts are provenance
artifacts; they are not part of the offline reproduction command.

## Reusable case files

Run the exporter alone when the verified scores already exist:

```bash
.venv/bin/python3 src/analysis/export_revision_v3_cases.py
```

It writes three files under `results/revision_v3/analysis/reuse/`:

| File | Rows and purpose |
|---|---|
| `cases.csv` | 630 TEP pairs: 3 models × 3 repetitions × 70 source units; compare clean and descriptor-mapping-fault predictions for the same observation |
| `clean_correctness.csv` | 1,560 rows: 1,170 planned clean LLM classifications plus 390 single-fit baseline classifications; includes 60 explicitly unavailable LLM rows |
| `manifest.json` | CSV hashes/counts, source-file hashes, exporter/scorer hashes, null policy, and the frozen outcome provenance |

All paths in the `sources_relative_to_revision_root` manifest map are relative
to `results/revision_v3/`. The `predictions_file` and `metadata_file` columns use
the same convention. Baselines are fitted once: their `repeat` and `session_id`
cells are blank, rather than duplicating the fit three times.

Important columns in `cases.csv` are:

| Column | Definition |
|---|---|
| `model`, `repeat`, `id` | Unique row key within this TEP-only file |
| `source_group` | Original held-out simulation-run grouping used for resampling |
| `true_label` | Evaluation label; this file is an analysis artifact, not a label-free inference prompt |
| `clean_predicted`, `misnamed_predicted` | Exact labels from the two frozen results |
| `clean_correct`, `misnamed_correct` | 1 if correct, 0 if incorrect; blank if the batch is unavailable |
| `prediction_flipped` | 1 if the two labels differ, even when both are wrong |
| `correctness_change_clean_minus_misnamed` | +1 for correct→incorrect, −1 for incorrect→correct, 0 if correctness is unchanged; blank if unavailable |
| `transition_clean_to_misnamed` | `both_correct`, `correct_to_incorrect`, `incorrect_to_correct`, `both_incorrect`, or `unavailable` |
| `pair_available` and condition-specific `status` | Explicit availability and terminal outcome, so an absent result is not counted as an incorrect prediction |
| Condition-specific session IDs and prompt hashes | Join the case back to its exact grouped-session evidence |

The name `misnamed` is retained as the frozen internal condition identifier.
The intervention actually rotates **all 52 complete sensor descriptors, including
their stated units**, by one position. Numeric strings, sample IDs, sensor-value
row order, task instructions and sample order are unchanged:
`(descriptor_1, value_1) → (descriptor_2, value_1)`, with wraparound at 52.
It is a descriptor-mapping stress test, not a harmless change of wording.

In `clean_correctness.csv`, join on **`dataset` + `id`**, not on `id` alone.
The unique row key additionally contains `model` and `repeat`. `available=0`,
`status=invalid_response`, a failure reason, and blank `predicted`/`correct`
cells represent N/A. All 60 planned rows of the failed batch are retained; the
59 partial labels in its raw response are deliberately absent from the exports.
Do not replace these blanks with zero or a majority label.

## Example analysis that retains unavailable primary results

Read IDs as strings to preserve leading zeros. Pandas' nullable integer type
preserves missing correctness values and the baseline's missing repeat index.

```python
from pathlib import Path
import pandas as pd

base = Path("results/revision_v3/analysis/reuse")
clean = pd.read_csv(
    base / "clean_correctness.csv",
    dtype={"id": "string", "repeat": "Int64", "correct": "Int64"},
)
primary = clean[clean["primary"] == 1]
print(primary.groupby(["dataset", "model"]).agg(
    planned=("id", "size"),
    scored=("correct", "count"),
    accuracy=("correct", "mean"),
))

pairs = pd.read_csv(base / "cases.csv", dtype={"id": "string"})
print(pairs[pairs["primary"] == 1].groupby("model").agg(
    planned_pairs=("id", "size"),
    available_pairs=("pair_available", "sum"),
    flip_rate=("prediction_flipped", "mean"),
    clean_minus_mapping_accuracy=("correctness_change_clean_minus_misnamed", "mean"),
))
```

The failed C-MAPSS primary remains a 60-planned, 0-scored, N/A-accuracy row.
The mapping differences above are fractions; multiply by 100 for percentage
points. Flip rate and accuracy difference are different quantities.

For confidence intervals, reuse `score_revision_v3.py`: it uses 10,000 paired
source-group resamples with seed 20260926, fixed class vocabularies, and the same
draws across paired methods. Accuracy summaries use Wilson intervals. These are
descriptive intervals conditional on the selected source units and observed
grouped responses; they do not capture shared-batch dependence or future model
variability. Pooling the three repetitions as three times as many independent
observations would change the analysis incorrectly.

## Prediction contract and extending the study

[`prediction_contract.schema.json`](prediction_contract.schema.json) is a
Draft 2020-12 JSON Schema for complete successful prediction arrays. It enforces
exact object fields, eight-character lowercase hexadecimal string IDs, batch
sizes, and case-sensitive C-MAPSS or TEP label enums. For TEP, `Fault_1` and
`Fault_13` are different exact labels. `N/A` and `UNPARSEABLE` are not valid
prediction labels.

The schema covers the two frozen vocabularies. It cannot compare an output to a
separate `valid_labels` field, prove IDs match the particular input in its exact
order, or enforce unique ID values when other fields differ. Nor can it verify
raw responses, source hashes or inference logs. The Python audit performs those
cross-file checks. A syntactically schema-valid array is not sufficient evidence
of a valid experiment. For another dataset, generate its enum from that dataset's
frozen `valid_labels`; do not broaden labels to the union of unrelated tasks.

Useful extensions can begin with these existing cases: class-specific error
audits, paired transition analyses, sensitivity to the declared source-unit
resampling rule, or an independently implemented exact-output validator. Clearly
label any new metric or changed analysis as a secondary analysis. A new prompt,
model, sample order or perturbation requires a separate experimental version,
rather than changes to this release's frozen files.

When describing reuse, identify revision v3, the frozen-manifest SHA-256, the
primary-repeat rule, and any departures from the analysis. Dataset provenance
and upstream references are in `data/README.md` and the manuscript bibliography.
This guide does not assign an author list, public repository, archive DOI, or
reuse license; it is a local reproduction artifact. Use verified publication
metadata and explicitly assigned release terms if a public release is made.
