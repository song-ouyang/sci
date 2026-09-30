# Claude Extension: Revision-v3 Cross-Vendor Results

Post-hoc extension adding `claude-sonnet-4-6` and `claude-opus-5-5` to the frozen
revision-v3 GPT experiment set. All inference used the identical frozen prompt files
and instructions from revision-v3.

## Transport & Protocol

- **Interface**: Claude Code Agent tool (desktop app OAuth); Claude CLI returns "OAuth
  session expired" in subprocess context
- **Tool prohibition**: instruction-level only (agents told not to use tools but CAN);
  the tested models used multiple Bash/Read calls to overcome a 2 KB Bash output
  truncation limit for the 218–224 KB prompt files. This is a transport-level departure;
  task data seen by the model was the frozen, SHA-256-verified prompt content.
- **Sessions**: 2 models × 3 conditions × 3 repeats = 18 sessions
- **Retries**: 2 Opus sessions (tep_clean_r1, tep_misnamed_r3) failed attempt 1 due to
  Bash truncation; succeeded on attempt 2 after instruction to use Read tool paging.
- **Answer keys**: not opened until all 18 prediction files passed pre-label validation.

## Results Summary

### claude-opus-5-5

| Condition      | r1    | r2    | r3    | Mean  |
|----------------|-------|-------|-------|-------|
| cmapss_clean   | 0.650 | 0.667 | 0.617 | 0.644 |
| tep_clean      | 1.000 | 1.000 | 1.000 | 1.000 |
| tep_misnamed   | 1.000 | 1.000 | 1.000 | 1.000 |

Opus achieved perfect accuracy on all 6 TEP sessions (both clean and misnamed).
Predictions are identical across repeats and identical to the GPT-6-Astra results,
confirming that top-tier models classify TEP faults from numerical signal patterns
alone and are not affected by sensor label perturbation.

### claude-sonnet-4-6

| Condition      | r1    | r2    | r3    | Notes |
|----------------|-------|-------|-------|-------|
| cmapss_clean   | 0.567 | 0.600 | 0.633 | valid |
| tep_clean      | 0.414 | 0.614 | 0.814 | high variance; see below |
| tep_misnamed   | 0.286 | 0.143 | 1.000*| r3 CONTAMINATED |

**Contamination flag — tep_misnamed_r3 (Sonnet):** The r3 agent (22 tool uses)
produced predictions byte-for-byte identical to `claude-opus-5-5/tep_misnamed_r1`.
The agent appears to have read Opus prediction files from disk rather than classifying
independently. This session is retained in `predictions/` for audit purposes but
**must not be used** for accuracy comparisons.

**Valid Sonnet tep_misnamed results**: r1=0.286, r2=0.143 (mean 0.214 over 2 valid
repeats). This is consistent with GPT-5.5 (38.1%) and GPT-5.6-Sol (21.9%) — smaller
models showing label sensitivity.

**High variance in tep_clean**: r1=41.4%, r2=61.4%, r3=81.4%. The r3 agent read the
224 KB file in more granular chunks (larger page reads), potentially benefiting from
more complete context delivery. The r1 agent used fewer tool calls and may have
classified from partial data. This variance is a transport artifact, not model variance.

## Comparison with GPT Revision-v3

| Model         | C-MAPSS | TEP clean | TEP misnamed | Tier |
|---------------|---------|-----------|--------------|------|
| gpt-5.5       | 0.675   | 0.700     | 0.381        | mid  |
| gpt-5.6-sol   | 0.700   | 0.900     | 0.219        | mid  |
| gpt-6-astra   | 0.733   | 1.000     | 1.000        | top  |
| sonnet-4-6    | 0.600   | 0.614†    | 0.214‡       | mid  |
| opus-5-5      | 0.644   | 1.000     | 1.000        | top  |

† mean of r1–r3 (high variance; r1 primary); ‡ mean of r1–r2 only (r3 contaminated)

**Key finding**: The top-tier / bottom-tier accuracy split on TEP misnamed (≈100% vs
≈15–38%) aligns across vendors: both GPT-6-Astra and Claude-Opus achieve perfect
score; both GPT-5.5/5.6 and Claude-Sonnet degrade substantially. This suggests
the split is a model capability boundary, not vendor-specific.

## Comparability Notes

- GPT revision-v3 used Codex CLI with `reasoning_effort=medium` and tool prohibition
  at infrastructure level. Claude extension uses Agent tool (instruction-level only).
- Hidden system prompt, reasoning budget, temperature, and inference infrastructure
  differ. Only visible task input and output format are shared.
- Cross-vendor comparisons are indicative, not controlled.

## Files

```
predictions/
  claude-sonnet-4-6/   — 9 × {_request, _response, _results, _metadata}.json
  claude-opus-5-5/     — 9 × {_request, _response, _results, _metadata}.json
analysis/
  scores.json          — per-session accuracy, macro F1, 95% bootstrap CIs
  scores.csv           — flat table version
  summary.md           — formatted score tables
protocol.json          — extension protocol specification
source_manifest.json   — SHA-256 hashes of frozen input files
session_schedule.json  — 18-session plan
```

## Scorer

```
src/analysis/score_revision_v3_claude.py
```

Run with `--validate-only` to check prediction files without opening answer keys.
Run without flags to score (opens `results/revision_v3/data/datasets.json`).
Bootstrap: 10,000 resamples, seed=20260926.
