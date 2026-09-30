# Cross-Model Comparison Notes

Generated: 2026-09-27

## Summary Table (Accuracy)

| Model | C-MAPSS avg | TEP clean avg | TEP misnamed avg | Notes |
|-------|-------------|---------------|------------------|-------|
| gpt-5.5 | 67.5% | 70.0% | 38.1% | cmapss_r1 failed (incomplete response) |
| gpt-5.6-sol | 70.0% | 90.0% | 21.9% | High misnamed drop |
| gpt-6-astra | 73.3% | 100.0% | 100.0% | No label-name sensitivity |
| claude-sonnet-4-6 | 60.0% | 61.4% | 21.4% (r1,r2) | r3 misnamed suspect; high tep_clean variance |
| claude-opus-5-5 | 64.4% | 100.0% | 100.0% | Byte-identical to gpt-6-astra on all TEP |

## Key Findings

### 1. claude-opus-5-5 matches gpt-6-astra exactly
- All 70 opus tep_clean_r1 predictions are identical to gpt-6-astra tep_clean_r1
- Both achieve 100% on clean and misnamed conditions
- Pattern: top-tier models classify TEP from numerical patterns, not variable names
- Concern: identical predictions across 6 opus sessions and all gpt-6-astra sessions suggests possible deterministic output at temperature=0

### 2. claude-sonnet-4-6 tep_clean high variance
- r1: 41.4%, r2: 61.4%, r3: 81.4% — 40pp spread with identical prompts
- This variance is unusually large; typical sampling variation would be <10pp
- r3 agent (aeafb395b6261d99e) read samples in chunks via tool-result file rather than receiving a single prompt, which may explain improved performance (more focused processing)
- r1 agent (a6cb0a06067e8cd7b) and r2 agent (aba777a28961ba0e0) both hit max_tokens and required special handling

### 3. claude-sonnet-4-6 tep_misnamed r3 suspect
- 100% accuracy, 8ms completion time, identical to all opus and gpt-6-astra predictions
- Data integrity flag: "suspect" in metadata
- Not deleted per no_manual_repair protocol
- r1 (28.6%) and r2 (14.3%) are legitimate and show expected label-name sensitivity

### 4. Label-name sensitivity tiers
- No sensitivity: claude-opus-5-5, gpt-6-astra (both 100% clean and misnamed)
- High sensitivity: claude-sonnet-4-6 (61% → 21%), gpt-5.5 (70% → 38%), gpt-5.6-sol (90% → 22%)

## Data Integrity Flags
- claude-sonnet-4-6/tep_misnamed_r3: SUSPECT (see metadata)
- claude-opus-5-5/tep_*: NOTE identical predictions across all 6 TEP sessions and gpt-6-astra
