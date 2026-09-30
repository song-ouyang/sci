# Claude extension: integrity audit blocks primary-table integration

**Status on 2026-09-27: 0 of the 18 submitted sessions is verified as eligible
for the controlled primary comparison.** All 18 stored prediction arrays have
the expected row counts, ordered IDs and permitted labels. That structural
check does not establish their inference provenance, complete input delivery,
model identity or isolation from other results.

This finding does not mean that every label is incorrect or that every session
copied another result. The available evidence cannot support the claimed
controlled cross-provider experiment.

## Confirmed record problems

- No pre-score prediction `frozen_manifest.json` or complete provider-native
  request/response/event and attempt logs are present in this collection.
- The save helper constructs request files from standard prompts after receiving
  response text. Matching saved prompt hashes therefore do not prove what was
  actually delivered to the model. `answer_keys_read=False` is assigned by that
  helper rather than measured from access records.
- The recorded transport is a Claude Code agent with tools available. Existing
  metadata and the archived writeup describe Bash/Read paging, truncation and
  continuations. These are departures from the tool-disabled, full-payload
  grouped protocol used by the GPT study.
- Sonnet `tep_misnamed_r3` is explicitly flagged `CONTAMINATED`. The historical
  scorer nevertheless reported it as complete with accuracy 1.0 and F1 1.0.
  Equality of prediction files alone is not proof of copying; the stored flag
  is sufficient to exclude this record from an independent-inference comparison.
- Sonnet `tep_clean_r2` and `tep_misnamed_r1` have prediction hashes that disagree
  with metadata. The former also has `request_sha256="pending"`.
- Sonnet `tep_clean_r3` has no request file. Its response text is a post-score
  explanation containing true labels, not a preserved original model response.
- Model identities were inferred from model self-reports. The schedule also
  names Opus 4.6 while the protocol/output directories claim Opus 5.5. These
  records cannot establish a verified service snapshot or consistent request plan.

## Current artifacts

Original prediction, response, request and metadata files remain unchanged.
The entire pre-correction collection and old scorer are preserved at:

`results/claude_extension_audit_archive/2026-09-27/`

That archive's snapshot was made **after** scoring for preservation. It is not
evidence that the original predictions were frozen before scoring.

[The integrity audit](audit/report.md) and its JSON counterpart separate structural
validity from primary-comparison eligibility. The current `analysis/scores.json`,
CSV and summary explicitly report unavailable metrics. They do not turn missing
provenance into zero accuracy. Previous numerical tables remain in the archive
as unverified historical computations.

## Recheck without loading answer labels

From the project root:

```sh
.venv/bin/python3 src/analysis/score_revision_v3_claude.py --validate-only
```

The expected exit code is **2** because primary-table eligibility is blocked.
The command does not open dataset labels or change inference artifacts. Without
`--validate-only`, it refreshes the audit and unavailable-result summaries; it
still does not load labels or compute classification metrics.

## Next step

Follow `paper/CLAUDE_EXTENSION_REMEDIATION_2026-09-27.md`. Retrieve genuine original
logs where available; do not reconstruct them from prompts, scores or another
session. A corrected evaluation belongs in a new extension version with a
predeclared schedule, actual tool isolation, full-message delivery and native
transport evidence. Do not append a new manifest or adjust old hashes to make
this collection appear valid.

The main manuscript and frozen GPT study are unchanged by this audit. Claims
that these Claude outputs establish numerical-only reasoning, a cross-provider
capability boundary or universal mapping robustness are not supported.
